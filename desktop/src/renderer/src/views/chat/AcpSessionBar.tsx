import { useEffect, useRef, useState } from 'react'
import { formatElapsedClock } from '@/views/chat/transcript/acp'
import { t } from '~/i18n'
import { ProviderLogo } from '~/views/settings/ProviderLogo'
import type { AcpSessionView } from './useAcpSession'

/** Harness id → the provider whose brand mark it carries (ProviderLogo). */
const MARK_FOR: Record<string, string> = {
  claude: 'anthropic',
  codex: 'openai',
  gemini: 'gemini',
}

function basename(path: string): string {
  const parts = path.split(/[\\/]+/).filter(Boolean)
  return parts.length ? (parts[parts.length - 1] as string) : path
}

function str(value: unknown): string {
  return typeof value === 'string' ? value.trim() : ''
}

/** The status line's second half: what the harness is doing, or what comes next. */
function detailFor(view: AcpSessionView): { text: string; tone: '' | 'error' } {
  const { info, name, state } = view
  if (state === 'closed') return { text: t('acp.bar.closedHint'), tone: '' }
  if (state === 'working') {
    const activity = str(info?.activity)
    if (activity) return { text: activity, tone: '' }
    if (info?.active_turn === true)
      return { text: t('acp.bar.busy').replace('{name}', name), tone: '' }
    return { text: t('acp.bar.starting').replace('{name}', name), tone: '' }
  }
  const lastTurn = info?.last_turn
  const error = lastTurn && typeof lastTurn === 'object' ? str(lastTurn.error) : ''
  if (error) return { text: t('acp.bar.lastFailed').replace('{message}', error), tone: 'error' }
  return { text: t('acp.bar.idleHint').replace('{name}', name), tone: '' }
}

/**
 * The bar over a coding harness's own chat (`agent:<agent>:acp:<id>`): which
 * harness, where it works, how it runs, and what it is doing now — Working
 * with its live step, Idle (the next message resumes it) or Closed. Stop
 * cancels the running turn (`acp.cancel`); End session closes the harness for
 * good (`acp.close`) after an inline confirm, never a native dialog.
 */
export function AcpSessionBar({ view }: { view: AcpSessionView }) {
  const [confirming, setConfirming] = useState(false)
  const [clock, setClock] = useState(() => Date.now())
  const keepRef = useRef<HTMLButtonElement>(null)
  const { info, name, state, startedAt } = view
  const working = state === 'working'
  const closed = state === 'closed'

  // The elapsed clock ticks only while a turn runs.
  useEffect(() => {
    if (!working || startedAt === null) return
    const timer = setInterval(() => setClock(Date.now()), 1000)
    return () => clearInterval(timer)
  }, [working, startedAt])

  // Confirm: the safe answer takes focus; Escape backs out.
  useEffect(() => {
    if (!confirming) return
    keepRef.current?.focus({ preventScroll: true })
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return
      e.preventDefault()
      e.stopPropagation()
      setConfirming(false)
    }
    document.addEventListener('keydown', onKey, true)
    return () => document.removeEventListener('keydown', onKey, true)
  }, [confirming])

  const agentId = view.agentId || 'unknown'
  const cwd = str(info?.cwd)
  const mode = str(info?.mode)
  const harnessMode = str(info?.harness_mode) || str(info?.current_mode_id)
  const detail = detailFor(view)
  const label = closed ? t('acp.bar.closed') : working ? t('acp.bar.working') : t('acp.bar.idle')
  const elapsed =
    working && startedAt !== null ? formatElapsedClock((clock - startedAt) / 1000) : null
  const noteText =
    view.note?.kind === 'nothing'
      ? t('acp.bar.stopNothing')
      : view.note?.kind === 'error'
        ? t('acp.bar.failed').replace('{message}', view.note.message)
        : ''

  return (
    <section
      className="acp-bar"
      data-state={state}
      data-acp-agent={agentId}
      data-enter="true"
      aria-label={t('acp.bar.label')}
      data-testid="acp-bar"
    >
      <span className="acp-bar__badge" data-acp-agent={agentId}>
        <ProviderLogo id={MARK_FOR[agentId] ?? agentId} label={name || agentId} size={28} />
      </span>
      <div className="acp-bar__id">
        <div className="acp-bar__name">{name}</div>
        {cwd ? (
          <div className="acp-bar__cwd" title={cwd}>
            {basename(cwd)}
          </div>
        ) : null}
      </div>
      {mode || harnessMode ? (
        <div className="acp-chips acp-bar__chips">
          {mode ? (
            <span className="acp-chip" data-kind="mode" title={t('acp.bar.modeTitle')}>
              {mode}
            </span>
          ) : null}
          {harnessMode ? (
            <span
              className="acp-chip"
              data-kind="harness-mode"
              title={t('acp.bar.harnessModeTitle')}
            >
              {harnessMode}
            </span>
          ) : null}
        </div>
      ) : null}

      {confirming && !closed ? (
        <div className="acp-bar__confirm" role="group" aria-label={t('acp.bar.end')}>
          <span className="acp-bar__confirm-text">
            {t('acp.bar.confirm').replace('{name}', name)}
          </span>
          <button
            ref={keepRef}
            type="button"
            className="acp-btn"
            onClick={() => setConfirming(false)}
          >
            {t('acp.bar.confirmKeep')}
          </button>
          <button
            type="button"
            className="acp-btn acp-btn--solid-danger"
            disabled={view.endState !== 'idle'}
            data-pending={view.endState === 'pending' || undefined}
            onClick={() => {
              view.end()
              setConfirming(false)
            }}
          >
            {view.endState === 'pending' ? t('acp.bar.ending') : t('acp.bar.confirmEnd')}
          </button>
        </div>
      ) : (
        <>
          <div className="acp-bar__status" role="status" aria-live="polite">
            <span className="acp-bar__dot" aria-hidden="true" />
            <span className="acp-bar__state">{label}</span>
            <span
              className="acp-bar__detail"
              data-tone={detail.tone || undefined}
              title={detail.text}
            >
              <span key={detail.text}>{detail.text}</span>
            </span>
            {elapsed ? (
              <span className="acp-bar__elapsed" title={t('acp.bar.elapsedTitle')}>
                {elapsed}
              </span>
            ) : null}
          </div>
          {noteText ? (
            <span
              className="acp-bar__note"
              data-tone={view.note?.kind === 'error' ? 'error' : undefined}
              role="note"
            >
              {noteText}
            </span>
          ) : null}
          {!closed ? (
            <div className="acp-bar__actions">
              {working ? (
                <button
                  type="button"
                  className="acp-btn acp-btn--danger"
                  disabled={view.stopState !== 'idle'}
                  data-pending={view.stopState === 'pending' || undefined}
                  title={t('acp.bar.stopTitle').replace('{name}', name)}
                  onClick={view.stop}
                >
                  {view.stopState === 'idle' ? t('acp.bar.stop') : t('acp.bar.stopping')}
                </button>
              ) : null}
              <button
                type="button"
                className="acp-btn acp-btn--quiet"
                disabled={view.endState !== 'idle'}
                title={t('acp.bar.endTitle').replace('{name}', name)}
                onClick={() => setConfirming(true)}
              >
                {t('acp.bar.end')}
              </button>
            </div>
          ) : null}
        </>
      )}
    </section>
  )
}
