import { useEffect, useRef } from 'react'
import { useLocation, useNavigate } from 'react-router'
import { useRpc } from '@/app/providers'
import { useApprovals } from '@/services/approval-monitor'
import { useConnection } from '@/stores/connection'
import { sessionRunStatus } from '@/views/sessions/logic'
import type { GatewayState } from '@shared/gateway'
import type { NotifyTarget } from '@shared/notify'
import { sessionPath } from '~/components/sidebar/SessionRow'
import { t } from '~/i18n'
import { desktopApi, isDesktop } from '~/lib/desktop-api'
import { orderToastFinished } from '~/lib/order-toasts'
import { useGateway } from '~/stores/gateway'
import { useNotifyCenter, unseenCount } from '~/stores/notify-center'
import { useSessionMarks } from '~/stores/session-marks'
import { useSessions } from '~/stores/sessions'
import { useSettings } from '~/stores/settings'
import { useUi } from '~/stores/ui'
import { orderKindWord, orderLine as deskOrderLine } from '~/views/trading/desk/desk-logic'
import type { OrderKind, Token } from '~/views/trading/types'
import { bindActivation, notify } from './notifications/dispatch'
import {
  diffSessionRuns,
  taskEpochMs,
  excerpt,
  formatDuration,
  isMuted,
  runKind,
  triggerFiredEvent,
  type RunTrack,
  type TriggerFiredPayload,
} from './notifications/logic'

/**
 * Everything that turns app events into notifications, bound once from the
 * shell. Each hook watches one source and hands the dispatcher an event;
 * the dispatcher reads settings at fire time so a toggle applies to the
 * next event without remounting anything.
 */
export function useNotificationSignals(): void {
  useSessionRunSignals()
  useApprovalSignal()
  useJobSignals()
  useTradingSignals()
  useGatewaySignal()
  useBadgeSync()
  useActivation()
  useMuteExpiry()
  useSeenOnOpen()
}

/**
 * Replies, in every session. The gateway broadcasts `sessions.changed` on
 * each task transition (the sessions store subscribes), the list refetches,
 * and a row that stops being live is a reply that settled. Tracking resets
 * when the connection drops so a reconnect never reads as "everything
 * finished at once".
 */
function useSessionRunSignals(): void {
  const { rows } = useSessions()
  const connected = useConnection((s) => s.state === 'connected')
  const tracked = useRef<Map<string, RunTrack>>(new Map())
  const seeded = useRef(false)

  useEffect(() => {
    if (!connected) {
      tracked.current = new Map()
      seeded.current = false
    }
  }, [connected])

  useEffect(() => {
    if (!connected) return
    const now = Date.now()
    const runRows = rows.map((r) => {
      const raw = r.raw as Record<string, unknown>
      const last = (raw.last_task ?? raw.lastTask) as Record<string, unknown> | null | undefined
      return {
        key: r.key,
        title: r.title,
        live: r.live,
        status: sessionRunStatus(r.raw),
        taskStartedAt: taskEpochMs(last?.started_at ?? last?.startedAt),
        taskFinishedAt: taskEpochMs(last?.finished_at ?? last?.finishedAt),
      }
    })
    const { next, finished } = diffSessionRuns(tracked.current, runRows, now)
    tracked.current = next
    if (!seeded.current) {
      seeded.current = true
      return
    }
    const preview = useSettings.getState().settings.notifications.preview
    const onScreen = currentSessionKey(window.location.hash || window.location.pathname)
    for (const run of finished) {
      // A reply that settled while the user was elsewhere (another session,
      // another app) leaves the row bold until it is opened.
      if (run.key !== onScreen || !document.hasFocus()) {
        useSessionMarks.getState().setUnread(run.key, true)
      }
      const kind = runKind(run.status)
      const failedWhy = excerpt(String(rowTerminalMessage(rows, run.key) ?? ''))
      void notify({
        kind,
        title: kind === 'reply' ? t('notify.reply.title') : t('notify.replyFailed.title'),
        subtitle: preview ? run.title : undefined,
        body:
          kind === 'reply'
            ? preview
              ? `${t('notify.reply.took')} ${formatDuration(run.durationMs)}`
              : undefined
            : preview && failedWhy
              ? failedWhy
              : t('notify.replyFailed.body'),
        target: { type: 'session', key: run.key },
        durationMs: run.durationMs,
      })
    }
  }, [rows, connected])
}

function rowTerminalMessage(rows: ReturnType<typeof useSessions>['rows'], key: string): unknown {
  const raw = rows.find((r) => r.key === key)?.raw as Record<string, unknown> | undefined
  const last = (raw?.last_task ?? raw?.lastTask) as Record<string, unknown> | undefined
  return last?.terminal_message ?? last?.terminalMessage
}

/** Approvals arrive through the console's poller; a growing count is news. */
function useApprovalSignal(): void {
  const count = useApprovals((s) => s.pending.length)
  const pending = useApprovals((s) => s.pending)
  const prev = useRef(count)
  useEffect(() => {
    const grew = count > prev.current
    prev.current = count
    if (!grew) return
    const preview = useSettings.getState().settings.notifications.preview
    const head = pending[0]
    const what = head ? excerpt(head.command || head.toolName || head.actionKind || '', 120) : ''
    void notify({
      kind: 'approval',
      title: t('notify.approval.title'),
      subtitle: preview && what ? what : undefined,
      body: t('notify.approval.body'),
      target: { type: 'approvals' },
    })
  }, [count, pending])
}

/**
 * Scheduled job runs. `cron.subscribe` with no id is the wildcard topic;
 * the Jobs panel subscribes too while open, which is harmless.
 */
function useJobSignals(): void {
  const rpc = useRpc()
  const connected = useConnection((s) => s.state === 'connected')
  useEffect(() => {
    if (!connected) return
    let cancelled = false
    rpc
      .waitForConnection()
      .then(() => (cancelled ? undefined : rpc.call('cron.subscribe', {})))
      .catch(() => undefined)
    const off = rpc.on('cron.run.finished', (payload) => {
      const p = (payload ?? {}) as {
        jobId?: string
        jobName?: string
        success?: boolean
        summary?: string | null
      }
      const ok = p.success !== false
      const preview = useSettings.getState().settings.notifications.preview
      const name = String(p.jobName || p.jobId || t('notify.job.unnamed'))
      void notify({
        kind: ok ? 'job' : 'jobFailed',
        title: ok ? t('notify.job.title') : t('notify.jobFailed.title'),
        subtitle: name,
        body: preview ? excerpt(p.summary) || undefined : undefined,
        target: { type: 'jobs', jobId: p.jobId },
      })
    })
    return () => {
      cancelled = true
      off()
    }
  }, [rpc, connected])
}

/**
 * The desk. An agent swap above the threshold asks for a decision; a swap
 * that settles (confirmed, failed, lapsed, rejected) is news too, and so is
 * a price trigger that fires. All are gateway broadcasts, so nothing is
 * polled.
 */
function useTradingSignals(): void {
  const rpc = useRpc()
  const connected = useConnection((s) => s.state === 'connected')
  useEffect(() => {
    if (!connected) return
    // The title names the kind — a send is not a "Swap", and a rejected
    // send read "Swap rejected · 0.00001 ETH → ETH" before.
    const offRequested = rpc.on('trading.approval.requested', (payload) => {
      const order = (payload as { order?: TradingOrderLike } | undefined)?.order
      const preview = useSettings.getState().settings.notifications.preview
      void notify({
        kind: 'approval',
        title: order
          ? `${t(`trading.card.kind.${orderKindWord(order)}`)} ${t('trading.notify.needsApproval')}`
          : t('notify.trade.approval.title'),
        subtitle: preview && order ? orderLine(order) : undefined,
        body: t('notify.trade.approval.body'),
        target: { type: 'trading', orderId: order?.orderId },
      })
    })
    const offFinished = rpc.on('trading.order.finished', (payload) => {
      const order = (payload as { order?: TradingOrderLike } | undefined)?.order
      if (!order) return
      // The "Approved. Sending…" toast has said all it can; the banner below says how it ended.
      orderToastFinished(order.orderId)
      const preview = useSettings.getState().settings.notifications.preview
      const ok = order.status === 'confirmed'
      const outcome =
        order.status === 'confirmed'
          ? t('trading.notify.confirmed')
          : order.status === 'expired'
            ? t('trading.notify.expired')
            : order.status === 'rejected'
              ? t('trading.notify.rejected')
              : t('trading.notify.failed')
      void notify({
        kind: ok ? 'trade' : 'tradeFailed',
        title: `${t(`trading.card.kind.${orderKindWord(order)}`)} ${outcome}`,
        subtitle: preview ? orderLine(order) : undefined,
        body: !ok && preview && order.reason ? excerpt(order.reason) || undefined : undefined,
        target: { type: 'trading', orderId: order.orderId },
      })
    })
    // A price trigger fired (docs/triggers.md): once per fire, when the
    // alert is sent, the order placed, or the fire skipped or failed.
    const offFired = rpc.on('trading.trigger.fired', (payload) => {
      const preview = useSettings.getState().settings.notifications.preview
      const ev = triggerFiredEvent(payload as TriggerFiredPayload | undefined, preview)
      if (ev) void notify(ev)
    })
    return () => {
      offRequested()
      offFinished()
      offFired()
    }
  }, [rpc, connected])
}

interface TradingOrderLike {
  orderId?: string
  status?: string
  reason?: string | null
  amountIn?: string
  tokenIn?: { symbol?: string; address?: string }
  tokenOut?: { symbol?: string }
  kind?: OrderKind
  recipient?: string | null
  recipientLabel?: string | null
  batchId?: string | null
}

/** The desk's own one-liner, over the loose shape a broadcast carries. */
function orderLine(order: TradingOrderLike): string {
  return deskOrderLine({
    kind: order.kind,
    amountIn: order.amountIn ?? '',
    tokenIn: {
      symbol: order.tokenIn?.symbol ?? '',
      address: order.tokenIn?.address ?? '',
    } as Token,
    tokenOut: { symbol: order.tokenOut?.symbol ?? '' } as Token,
    recipient: order.recipient ?? null,
    recipientLabel: order.recipientLabel ?? null,
    batchId: order.batchId ?? null,
  })
}

/** The gateway went from running to error on its own (a crash, a port grab). */
function useGatewaySignal(): void {
  const status = useGateway((s) => s.status)
  const prev = useRef<GatewayState>(status.state)
  useEffect(() => {
    const was = prev.current
    prev.current = status.state
    if (status.state !== 'error' || was !== 'running') return
    void notify({
      kind: 'gateway',
      title: t('notify.gateway.title'),
      body: excerpt(status.error) || t('notify.gateway.body'),
      target: { type: 'settings' },
    })
  }, [status])
}

/** Dock badge = unseen notifications + approvals waiting, when enabled. */
function useBadgeSync(): void {
  const badge = useSettings((s) => s.settings.notifications.badge)
  const enabled = useSettings((s) => s.settings.notifications.enabled)
  const unseen = useNotifyCenter((s) => unseenCount(s.items))
  const approvals = useApprovals((s) => s.pending.length)
  useEffect(() => {
    if (!isDesktop()) return
    void desktopApi().notify.badge(badge && enabled ? unseen + approvals : 0)
  }, [badge, enabled, unseen, approvals])
}

/** A click on a notification (native or banner) or a menu bar row lands somewhere in the app. */
function useActivation(): void {
  const navigate = useNavigate()
  const openJobs = useUi((s) => s.openJobs)
  const openSettings = useUi((s) => s.openSettings)
  useEffect(() => {
    const go = (target: NotifyTarget) => {
      switch (target.type) {
        case 'session':
          useNotifyCenter.getState().markSessionSeen(target.key)
          void navigate(sessionPath(target.key))
          break
        case 'jobs':
          openJobs()
          break
        case 'settings':
          openSettings('gateway')
          break
        case 'trading':
          useUi.getState().closeSettings()
          useUi.getState().closeJobs()
          useUi.getState().closeSkills()
          void navigate(
            target.orderId ? `/trading?order=${encodeURIComponent(target.orderId)}` : '/trading',
          )
          break
        case 'newChat':
          // The menu bar's "New chat": a fresh chat even from the desk, unlike
          // ⌘N there, with the composer asking for focus as ⌘N does (#3524).
          useUi.getState().closeSettings()
          useUi.getState().closeJobs()
          useUi.getState().closeSkills()
          void navigate('/sessions')
          useUi.getState().requestComposerFocus()
          break
        case 'approvals':
        case 'none':
          // The approval prompt is a modal that shows itself while one is pending.
          break
      }
    }
    const unbind = bindActivation(go)
    const off = desktopApi().notify.onActivated(go)
    // The menu bar item's rows land here too: one router for every click.
    const offTray = desktopApi().tray.onNavigate(go)
    return () => {
      unbind()
      off()
      offTray()
    }
  }, [navigate, openJobs, openSettings])
}

/** When Do not disturb runs out, clear it so the bell stops showing muted. */
function useMuteExpiry(): void {
  const muteUntil = useSettings((s) => s.settings.notifications.muteUntil)
  const update = useSettings((s) => s.update)
  useEffect(() => {
    if (muteUntil === null) return
    const wait = muteUntil - Date.now()
    if (wait <= 0) {
      void update({ notifications: { muteUntil: null } })
      return
    }
    const id = setTimeout(() => {
      if (!isMuted({ muteUntil }, Date.now())) void update({ notifications: { muteUntil: null } })
    }, wait + 50)
    return () => clearTimeout(id)
  }, [muteUntil, update])
}

/** The session key a route path points at, or null off the chat route. */
export function currentSessionKey(path: string): string | null {
  const m = /\/sessions\/([^/?#]+)/.exec(path)
  if (!m?.[1]) return null
  try {
    return decodeURIComponent(m[1])
  } catch {
    return null
  }
}

/** Opening a session reads every notification that pointed at it, and the row. */
function useSeenOnOpen(): void {
  const { pathname } = useLocation()
  useEffect(() => {
    const key = currentSessionKey(pathname)
    if (!key) return
    const read = () => {
      useNotifyCenter.getState().markSessionSeen(key)
      useSessionMarks.getState().setUnread(key, false)
    }
    read()
    // A reply that landed while the window was in the background is read
    // the moment the user comes back to it.
    window.addEventListener('focus', read)
    return () => window.removeEventListener('focus', read)
  }, [pathname])
}
