import { useCallback, useEffect, useRef, useState } from 'react'
import type { WsRpcClient } from '@/lib/ws-rpc'

export interface NextSuggestion {
  /** The suggested next prompt, or null when there is nothing to offer. */
  suggestion: string | null
  /** Hide the current suggestion (the user sent something). */
  clear: () => void
}

/**
 * The composer's suggested next prompt (`chat.suggestNext`).
 *
 * Asked for on each busy → idle edge, i.e. when a turn ends; the gateway
 * answers null for a turn that errored or was aborted, and caches per turn,
 * so the web console and the desktop app on one session cost one model call.
 * An answer is dropped if the session changed or another turn started while
 * it was on its way. Errors (a gateway without the method) mean no suggestion.
 */
export function useNextSuggestion(
  rpc: WsRpcClient,
  sessionKey: string,
  busy: boolean,
): NextSuggestion {
  const [suggestion, setSuggestion] = useState<string | null>(null)
  // A new turn or another session retires the suggestion. Adjusted during
  // render (React's "state from changed props" pattern) rather than in an
  // effect, so the stale text never paints for a frame.
  const [seen, setSeen] = useState({ sessionKey, busy })
  if (seen.sessionKey !== sessionKey || seen.busy !== busy) {
    setSeen({ sessionKey, busy })
    if (busy || seen.sessionKey !== sessionKey) setSuggestion(null)
  }

  // Bumped on every edge that makes an in-flight answer stale.
  const generationRef = useRef(0)
  const wasBusyRef = useRef(busy)

  useEffect(() => {
    const wasBusy = wasBusyRef.current
    wasBusyRef.current = busy
    const generation = ++generationRef.current
    if (busy || !wasBusy) return
    // Asked a beat after the turn ends, not in the same tick: the reads that
    // say what state the chat is in (history, `sessions.resolve`, a coding
    // harness's `acp.status`) go out first. A suggestion is a model call and
    // can take seconds; nothing should wait behind it, on a gateway that
    // answers one request at a time per connection least of all.
    const timer = setTimeout(() => {
      if (generation !== generationRef.current) return
      rpc
        .call<{ text?: string | null }>('chat.suggestNext', { sessionKey })
        .then((res) => {
          if (generation !== generationRef.current) return
          const text = typeof res?.text === 'string' ? res.text.trim() : ''
          setSuggestion(text || null)
        })
        .catch(() => {
          /* no suggestion — the composer keeps its normal placeholder */
        })
    }, 0)
    return () => clearTimeout(timer)
  }, [busy, rpc, sessionKey])

  const clear = useCallback(() => {
    generationRef.current += 1
    setSuggestion(null)
  }, [])

  return { suggestion, clear }
}
