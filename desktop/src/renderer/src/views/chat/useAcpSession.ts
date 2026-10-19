import { useCallback, useEffect, useRef, useState } from 'react'
import { useRpc } from '@/app/providers'
import { harnessDisplayName, isAcpSessionKey, type AcpLiveRow } from '@/views/chat/transcript/acp'

/** How often a running turn's status is re-read (the transcript's run cards use the same). */
export const ACP_STATUS_POLL_MS = 1500
/** How long a Stop / End result note stays up. */
const NOTE_MS = 4000

export type AcpBarState = 'working' | 'idle' | 'closed'

export interface AcpSessionView {
  /** `acp.status` answered for this key: it is a harness session. */
  confirmed: boolean
  info: AcpLiveRow | null
  agentId: string
  name: string
  state: AcpBarState
  /** Wall-clock ms the running turn started (for the elapsed clock), when known. */
  startedAt: number | null
  stopState: 'idle' | 'pending' | 'sent'
  endState: 'idle' | 'pending'
  /** The last Stop / End result worth saying; clears itself. */
  note: { kind: 'nothing' } | { kind: 'error'; message: string } | null
  stop: () => void
  end: () => void
}

type Note = AcpSessionView['note']

function errorCode(error: unknown): string {
  const code = (error as { code?: unknown } | null)?.code
  return typeof code === 'string' ? code : ''
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

/**
 * The state of the harness behind the open chat, for the session bar and
 * the composer. A key of the form `agent:<agent>:acp:<id>` is only a hint:
 * the bar exists once `acp.status` answers for it. An older gateway without
 * `acp.status` falls back to the live row of `acp.sessions`, when there is
 * one.
 *
 * Polling discipline (the same as the transcript's run cards): `acp.status`
 * is re-read every 1.5 s only while a turn runs and the window is visible;
 * once more when the turn ends (idle, closed, the last turn's outcome); never
 * after unmount or a switch to another session.
 */
export function useAcpSession(sessionKey: string, turnRunning: boolean): AcpSessionView {
  const rpc = useRpc()
  const hint = isAcpSessionKey(sessionKey)
  const [info, setInfo] = useState<AcpLiveRow | null>(null)
  const [confirmed, setConfirmed] = useState(false)
  const [startedAt, setStartedAt] = useState<number | null>(null)
  const [stopState, setStopState] = useState<AcpSessionView['stopState']>('idle')
  const [endState, setEndState] = useState<AcpSessionView['endState']>('idle')
  const [note, setNote] = useState<Note>(null)

  // A switch to another chat starts over (adjusted during render, not in an
  // effect, so the old session's bar never paints over the new one).
  const [viewKey, setViewKey] = useState(sessionKey)
  if (viewKey !== sessionKey) {
    setViewKey(sessionKey)
    setInfo(null)
    setConfirmed(false)
    setStartedAt(null)
    setStopState('idle')
    setEndState('idle')
    setNote(null)
  }

  // The key an answer belongs to; a late answer for another chat is dropped.
  const liveKeyRef = useRef(sessionKey)
  useEffect(() => {
    liveKeyRef.current = sessionKey
  }, [sessionKey])

  const apply = useCallback((key: string, row: AcpLiveRow) => {
    if (liveKeyRef.current !== key) return
    // An answer that names no session is no confirmation.
    if (typeof row.session_key !== 'string' && typeof row.agent_id !== 'string') return
    setInfo(row)
    setConfirmed(true)
    const elapsed = typeof row.turn_elapsed_s === 'number' ? row.turn_elapsed_s : null
    if (row.active_turn === true && elapsed !== null) {
      setStartedAt(Date.now() - elapsed * 1000)
    } else if (row.active_turn !== true) {
      setStartedAt(null)
    }
  }, [])

  const refresh = useCallback(async (): Promise<void> => {
    const key = sessionKey
    try {
      const row = (await rpc.call('acp.status', { session_key: key })) as AcpLiveRow | null
      if (row && typeof row === 'object') apply(key, row)
    } catch (error: unknown) {
      if (liveKeyRef.current !== key) return
      if (errorCode(error) !== 'METHOD_NOT_FOUND') {
        // ACP_SESSION_INVALID (not a harness after all), ACP_UNAVAILABLE, or
        // the connection: no bar rather than a wrong one.
        if (errorCode(error) === 'ACP_SESSION_INVALID') setConfirmed(false)
        return
      }
      try {
        const live = (await rpc.call('acp.sessions', {})) as { sessions?: AcpLiveRow[] } | null
        const row = live?.sessions?.find((r) => r?.session_key === key)
        if (row) apply(key, row)
      } catch {
        /* no manager: nothing to show */
      }
    }
  }, [apply, rpc, sessionKey])

  const working = turnRunning || info?.active_turn === true

  // Read on entry and on every turn edge; poll while the turn runs.
  useEffect(() => {
    if (!hint) return
    let disposed = false
    const read = () => {
      if (!disposed) void refresh()
    }
    read()
    if (!working) {
      return () => {
        disposed = true
      }
    }
    let timer: ReturnType<typeof setInterval> | null = null
    const arm = () => {
      if (timer !== null || document.hidden) return
      timer = setInterval(read, ACP_STATUS_POLL_MS)
    }
    const disarm = () => {
      if (timer !== null) clearInterval(timer)
      timer = null
    }
    const onVisibility = () => {
      if (document.hidden) {
        disarm()
        return
      }
      read()
      arm()
    }
    arm()
    document.addEventListener('visibilitychange', onVisibility)
    return () => {
      disposed = true
      disarm()
      document.removeEventListener('visibilitychange', onVisibility)
    }
  }, [hint, working, refresh])

  // A Stop that went through waits for the turn to end; then it is done.
  const [wasWorking, setWasWorking] = useState(working)
  if (wasWorking !== working) {
    setWasWorking(working)
    if (!working && stopState === 'sent') setStopState('idle')
  }

  // Result notes clear themselves.
  useEffect(() => {
    if (!note) return
    const timer = setTimeout(() => setNote(null), NOTE_MS)
    return () => clearTimeout(timer)
  }, [note])

  const stop = useCallback(() => {
    if (stopState !== 'idle') return
    const key = sessionKey
    setStopState('pending')
    setNote(null)
    rpc
      .call('acp.cancel', { session_key: key })
      .then((result) => {
        if (liveKeyRef.current !== key) return
        const cancelled = (result as { cancelled?: unknown } | null)?.cancelled !== false
        setStopState(cancelled ? 'sent' : 'idle')
        if (!cancelled) setNote({ kind: 'nothing' })
        void refresh()
      })
      .catch((error: unknown) => {
        if (liveKeyRef.current !== key) return
        setStopState('idle')
        setNote({ kind: 'error', message: errorMessage(error) })
      })
  }, [refresh, rpc, sessionKey, stopState])

  const end = useCallback(() => {
    if (endState !== 'idle') return
    const key = sessionKey
    setEndState('pending')
    setNote(null)
    rpc
      .call('acp.close', { session_key: key })
      .then(() => {
        if (liveKeyRef.current !== key) return
        setEndState('idle')
        setInfo((prev) => (prev ? { ...prev, closed: true, active_turn: false } : prev))
        void refresh()
      })
      .catch((error: unknown) => {
        if (liveKeyRef.current !== key) return
        setEndState('idle')
        setNote({ kind: 'error', message: errorMessage(error) })
      })
  }, [endState, refresh, rpc, sessionKey])

  const agentId = typeof info?.agent_id === 'string' ? info.agent_id : ''
  const closed = info?.closed === true
  return {
    confirmed: hint && confirmed,
    info,
    agentId,
    name: harnessDisplayName(agentId, info?.agent_name),
    state: closed ? 'closed' : working ? 'working' : 'idle',
    startedAt,
    stopState,
    endState,
    note,
    stop,
    end,
  }
}
