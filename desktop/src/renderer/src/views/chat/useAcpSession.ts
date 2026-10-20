import { useCallback, useEffect, useRef, useState } from 'react'
import { useRpc } from '@/app/providers'
import { harnessDisplayName, isAcpSessionKey, type AcpLiveRow } from '@/views/chat/transcript/acp'

/** How often a running turn's status is re-read (the transcript's run cards use the same). */
export const ACP_STATUS_POLL_MS = 1500
/** How long a Stop / End result note stays up. */
const NOTE_MS = 4000
/** Re-reads after a failure that is not an answer (no socket yet, a dropped frame). */
const STATUS_RETRIES = 4
/** Error codes that are an answer: retrying will not change them. */
const FINAL_CODES = new Set(['ACP_SESSION_INVALID', 'ACP_UNAVAILABLE', 'METHOD_NOT_FOUND'])

export type AcpBarState = 'working' | 'idle' | 'closed'

export interface AcpSessionView {
  /** `acp.status` answered for this key: it is a harness session. */
  confirmed: boolean
  info: AcpLiveRow | null
  agentId: string
  name: string
  state: AcpBarState
  /**
   * Working, but the harness has not been seen running this turn yet (the
   * chat sent it; the harness is starting or queued).
   */
  starting: boolean
  /** The harness's current step, this turn's only (never the last turn's). */
  activity: string | null
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

function activityOf(row: AcpLiveRow | null): string | null {
  return typeof row?.activity === 'string' && row.activity.trim() ? row.activity.trim() : null
}

/**
 * The state of the harness behind the open chat, for the session bar and
 * the composer. A key of the form `agent:<agent>:acp:<id>` is only a hint:
 * the bar exists once `acp.status` answers for it. An older gateway without
 * `acp.status` falls back to the live row of `acp.sessions`, when there is
 * one.
 *
 * Working and Stop follow the harness (`active_turn`), not the chat's run
 * state: the chat keeps running for a few seconds after the harness is done
 * (persisting, announcing), and the bar must not show a live step, a ticking
 * clock and a Stop with nothing left to stop. The chat's run only starts the
 * reads and marks the turn as starting until the harness is seen.
 *
 * Polling discipline (the same as the transcript's run cards): `acp.status`
 * is re-read every 1.5 s only while the chat or the harness runs and the
 * window is visible; once more when that ends; again whenever the connection
 * comes (back) up; never after unmount or a switch to another session. A read
 * that fails for want of a connection is retried a bounded number of times.
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
  // This chat run has seen the harness's turn active.
  const [runSeenActive, setRunSeenActive] = useState(false)
  // The step shown before this run began: hidden until a fresh one arrives.
  const [stale, setStale] = useState<{ activity: string | null } | null>(null)

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
    setRunSeenActive(false)
    setStale(null)
  }

  // A new chat run: the harness has not been seen yet, and whatever step the
  // bar knew belongs to the previous turn.
  const [prevRunning, setPrevRunning] = useState(turnRunning)
  if (prevRunning !== turnRunning) {
    setPrevRunning(turnRunning)
    if (turnRunning) {
      setRunSeenActive(false)
      setStale({ activity: activityOf(info) })
    }
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
    const active = row.active_turn === true
    if (active) setRunSeenActive(true)
    setStale((prev) => (prev && activityOf(row) !== prev.activity ? null : prev))
    const elapsed = typeof row.turn_elapsed_s === 'number' ? row.turn_elapsed_s : null
    if (active && elapsed !== null) setStartedAt(Date.now() - elapsed * 1000)
    else if (!active) setStartedAt(null)
  }, [])

  // Bounded re-reads after a failure that is not an answer.
  const attemptsRef = useRef(0)
  const retryRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const clearRetry = useCallback(() => {
    if (retryRef.current !== null) clearTimeout(retryRef.current)
    retryRef.current = null
  }, [])

  // The retry timer calls the newest `refresh` through this ref.
  const refreshRef = useRef<() => Promise<void>>(async () => {})
  const refresh = useCallback(async (): Promise<void> => {
    const key = sessionKey
    try {
      // On an app reload the bar mounts before the socket is up. Connected,
      // the read goes out in this very tick: after a turn it must not queue
      // behind the composer's `chat.suggestNext` (a model call).
      if (rpc.state !== 'connected') {
        await rpc.waitForConnection()
        if (liveKeyRef.current !== key) return
      }
      const row = (await rpc.call('acp.status', { session_key: key })) as AcpLiveRow | null
      attemptsRef.current = 0
      if (row && typeof row === 'object') apply(key, row)
    } catch (error: unknown) {
      if (liveKeyRef.current !== key) return
      const code = errorCode(error)
      if (code === 'METHOD_NOT_FOUND') {
        try {
          const live = (await rpc.call('acp.sessions', {})) as { sessions?: AcpLiveRow[] } | null
          const row = live?.sessions?.find((r) => r?.session_key === key)
          if (row) apply(key, row)
        } catch {
          /* no manager: nothing to show */
        }
        return
      }
      if (code === 'ACP_SESSION_INVALID') setConfirmed(false)
      if (FINAL_CODES.has(code) || attemptsRef.current >= STATUS_RETRIES) return
      // Not an answer: the socket dropped or was not up yet. Try again soon.
      const delay = 1000 * 2 ** attemptsRef.current
      attemptsRef.current += 1
      clearRetry()
      retryRef.current = setTimeout(() => {
        retryRef.current = null
        if (liveKeyRef.current === key) void refreshRef.current()
      }, delay)
    }
  }, [apply, clearRetry, rpc, sessionKey])
  useEffect(() => {
    refreshRef.current = refresh
  }, [refresh])

  // No retry outlives its session or the view.
  useEffect(() => {
    attemptsRef.current = 0
    return clearRetry
  }, [sessionKey, clearRetry])

  // The connection came (back) up: read again, whatever happened before.
  useEffect(() => {
    if (!hint) return
    return rpc.on('_state', (state: unknown) => {
      if (state !== 'connected') return
      attemptsRef.current = 0
      void refresh()
    })
  }, [hint, refresh, rpc])

  const closed = info?.closed === true
  const active = !closed && info?.active_turn === true
  const starting = !closed && !active && turnRunning && !runSeenActive
  const working = active || starting
  const polling = turnRunning || active

  // The chat's run just ended while the last answer still said the harness
  // was active (so the polling below goes on and does not read at once):
  // read now, before anything else the turn's end sets off. Otherwise the
  // poll's own edge read below covers it.
  const prevRunningRef = useRef(turnRunning)
  useEffect(() => {
    const was = prevRunningRef.current
    prevRunningRef.current = turnRunning
    if (hint && was && !turnRunning && active) void refresh()
  }, [hint, turnRunning, active, refresh])

  // Read on entry (mount, a session switch) and on every edge, whether or not
  // the window is visible: a session opened while hidden must not keep the
  // last answer it happened to have. Only the repeating poll — every 1.5 s
  // while the chat or the harness runs — pauses while hidden; coming back,
  // it reads at once and then resumes.
  useEffect(() => {
    if (!hint) return
    let disposed = false
    const read = () => {
      if (!disposed) void refresh()
    }
    let timer: ReturnType<typeof setInterval> | null = null
    const arm = () => {
      if (!polling || timer !== null || document.hidden) return
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
    read()
    arm()
    document.addEventListener('visibilitychange', onVisibility)
    return () => {
      disposed = true
      disarm()
      document.removeEventListener('visibilitychange', onVisibility)
    }
  }, [hint, polling, refresh])

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
  const current = activityOf(info)
  return {
    confirmed: hint && confirmed,
    info,
    agentId,
    name: harnessDisplayName(agentId, info?.agent_name),
    state: closed ? 'closed' : working ? 'working' : 'idle',
    starting,
    activity: active && !(stale && current === stale.activity) ? current : null,
    startedAt: active ? startedAt : null,
    stopState,
    endState,
    note,
    stop,
    end,
  }
}
