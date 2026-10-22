import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import { DEFAULT_NOTIFICATION_SETTINGS, type NotificationSettings } from '@shared/settings'
import type { Trigger, TriggerFire, TriggerPayload } from '~/views/trading/types'
import {
  decideDelivery,
  diffSessionRuns,
  eventWanted,
  excerpt,
  formatDuration,
  muteUntilFor,
  runKind,
  sessionKeyFromHash,
  taskEpochMs,
  triggerFiredEvent,
  type DeliveryContext,
  type NotifyEvent,
  type RunTrack,
} from './logic'

const NOW = 1_800_000_000_000
const reply: NotifyEvent = {
  kind: 'reply',
  title: 'Reply ready',
  target: { type: 'session', key: 'agent:main:webchat:a' },
  durationMs: 12_000,
}
const background: DeliveryContext = { now: NOW, focused: false, currentSessionKey: null }
const front: DeliveryContext = { now: NOW, focused: true, currentSessionKey: 'other' }

function settings(patch: Partial<NotificationSettings> = {}): NotificationSettings {
  return { ...DEFAULT_NOTIFICATION_SETTINGS, ...patch }
}

describe('eventWanted', () => {
  it('follows the per-event switches', () => {
    expect(eventWanted(settings({ replyDone: false }), reply)).toBe(false)
    expect(eventWanted(settings(), { ...reply, kind: 'replyFailed' })).toBe(true)
    expect(eventWanted(settings({ replyFailed: false }), { ...reply, kind: 'replyFailed' })).toBe(
      false,
    )
    expect(eventWanted(settings({ approvals: false }), { ...reply, kind: 'approval' })).toBe(false)
    expect(eventWanted(settings({ gateway: false }), { ...reply, kind: 'gateway' })).toBe(false)
  })

  it('applies the reply-length threshold', () => {
    const s = settings({ replyMinSeconds: 30 })
    expect(eventWanted(s, { ...reply, durationMs: 12_000 })).toBe(false)
    expect(eventWanted(s, { ...reply, durationMs: 30_000 })).toBe(true)
    expect(eventWanted(s, { ...reply, durationMs: undefined })).toBe(false)
  })

  it('jobs: off / failures / all', () => {
    const job: NotifyEvent = { ...reply, kind: 'job' }
    const failed: NotifyEvent = { ...reply, kind: 'jobFailed' }
    expect(eventWanted(settings({ jobs: 'off' }), job)).toBe(false)
    expect(eventWanted(settings({ jobs: 'off' }), failed)).toBe(false)
    expect(eventWanted(settings({ jobs: 'failures' }), job)).toBe(false)
    expect(eventWanted(settings({ jobs: 'failures' }), failed)).toBe(true)
    expect(eventWanted(settings({ jobs: 'all' }), job)).toBe(true)
  })
})

describe('decideDelivery', () => {
  it('posts a system notification in the background, with sound and bounce', () => {
    expect(decideDelivery(settings(), reply, background)).toEqual({
      system: true,
      banner: false,
      sound: true,
      bounce: true,
      record: true,
      seen: false,
    })
  })

  it('does nothing when the master switch is off', () => {
    expect(decideDelivery(settings({ enabled: false }), reply, background).record).toBe(false)
  })

  it('only records while muted', () => {
    const d = decideDelivery(settings({ muteUntil: NOW + 1 }), reply, background)
    expect(d).toMatchObject({ system: false, banner: false, sound: false, record: true })
    // An expired mute no longer applies.
    expect(decideDelivery(settings({ muteUntil: NOW - 1 }), reply, background).system).toBe(true)
  })

  it('in front: banner, system or skip per whenActive', () => {
    expect(decideDelivery(settings({ whenActive: 'banner' }), reply, front)).toMatchObject({
      system: false,
      banner: true,
      sound: true,
    })
    expect(decideDelivery(settings({ whenActive: 'system' }), reply, front)).toMatchObject({
      system: true,
      banner: false,
    })
    expect(decideDelivery(settings({ whenActive: 'skip' }), reply, front)).toMatchObject({
      system: false,
      banner: false,
      sound: false,
      record: true,
    })
  })

  it('never bounces the Dock while the window is in front', () => {
    expect(decideDelivery(settings({ whenActive: 'system' }), reply, front).bounce).toBe(false)
  })

  it('only chimes for the session on screen', () => {
    const watching: DeliveryContext = { ...front, currentSessionKey: 'agent:main:webchat:a' }
    expect(decideDelivery(settings(), reply, watching)).toEqual({
      system: false,
      banner: false,
      sound: true,
      bounce: false,
      record: true,
      seen: true,
    })
    expect(decideDelivery(settings({ sound: false }), reply, watching).sound).toBe(false)
  })

  it('a test notification ignores the switches and the mute', () => {
    const test: NotifyEvent = { kind: 'test', title: 'AgentOS', target: { type: 'none' } }
    const s = settings({ enabled: false, muteUntil: NOW + 1, whenActive: 'skip' })
    expect(decideDelivery(s, test, front).system).toBe(true)
  })
})

describe('diffSessionRuns', () => {
  const rows = (live: boolean, status = live ? 'running' : 'succeeded') => [
    { key: 'a', title: 'Alpha', live, status },
  ]

  it('seeds on the first snapshot without reporting anything', () => {
    const { next, finished } = diffSessionRuns(new Map(), rows(true), NOW)
    expect(finished).toEqual([])
    expect(next.get('a')).toEqual({ startedAt: NOW, title: 'Alpha' })
  })

  it('reports a run when a live row settles, with its duration', () => {
    const tracked = new Map<string, RunTrack>([['a', { startedAt: NOW - 5_000, title: 'Alpha' }]])
    const { next, finished } = diffSessionRuns(tracked, rows(false), NOW)
    expect(finished).toEqual([{ key: 'a', title: 'Alpha', status: 'succeeded', durationMs: 5_000 }])
    expect(next.size).toBe(0)
  })

  it('keeps the original start across snapshots', () => {
    const tracked = new Map<string, RunTrack>([['a', { startedAt: NOW - 5_000, title: 'Alpha' }]])
    const { next } = diffSessionRuns(tracked, rows(true), NOW)
    expect(next.get('a')?.startedAt).toBe(NOW - 5_000)
  })

  // Round 2: a harness's reply toast said "Took 1m 56s" while its card said
  // "2m 08s". The list sees a row go live and settle when it happens to
  // refetch; the gateway timed the task itself, and that is what the card's
  // clock is close to.
  it("takes the duration from the gateway's task times when the row carries them", () => {
    const tracked = new Map<string, RunTrack>([['a', { startedAt: NOW - 116_000, title: 'Alpha' }]])
    const row = {
      key: 'a',
      title: 'Alpha',
      live: false,
      status: 'succeeded',
      taskStartedAt: NOW - 130_000,
      taskFinishedAt: NOW - 2_000,
    }
    expect(diffSessionRuns(tracked, [row], NOW).finished[0]?.durationMs).toBe(128_000)
    // A task not finished yet (or no times at all): what the list saw.
    const open = { ...row, taskFinishedAt: null }
    expect(diffSessionRuns(tracked, [open], NOW).finished[0]?.durationMs).toBe(116_000)
  })

  it('reads gateway timestamps in seconds or milliseconds', () => {
    expect(taskEpochMs(1_800_000_000)).toBe(1_800_000_000_000)
    expect(taskEpochMs(1_800_000_000_123)).toBe(1_800_000_000_123)
    expect(taskEpochMs('1800000000123')).toBe(1_800_000_000_123)
    expect(taskEpochMs(null)).toBeNull()
    expect(taskEpochMs('soon')).toBeNull()
  })

  it('stays quiet for runs the user stopped', () => {
    const tracked = new Map<string, RunTrack>([['a', { startedAt: NOW, title: 'Alpha' }]])
    expect(diffSessionRuns(tracked, rows(false, 'cancelled'), NOW).finished).toEqual([])
  })

  it('drops rows that vanished while live', () => {
    const tracked = new Map<string, RunTrack>([['gone', { startedAt: NOW, title: 'x' }]])
    const { next, finished } = diffSessionRuns(tracked, [], NOW)
    expect(finished).toEqual([])
    expect(next.size).toBe(0)
  })

  it('maps terminal status to an event kind', () => {
    expect(runKind('succeeded')).toBe('reply')
    expect(runKind('failed')).toBe('replyFailed')
    expect(runKind('timeout')).toBe('replyFailed')
  })
})

describe('muteUntilFor', () => {
  it('offsets from now, and tomorrow means 9:00 next day', () => {
    expect(muteUntilFor('off', NOW)).toBeNull()
    expect(muteUntilFor('30m', NOW)).toBe(NOW + 30 * 60_000)
    expect(muteUntilFor('1h', NOW)).toBe(NOW + 60 * 60_000)
    expect(muteUntilFor('3h', NOW)).toBe(NOW + 3 * 60 * 60_000)
    const tomorrow = new Date(muteUntilFor('tomorrow', NOW)!)
    const today = new Date(NOW)
    expect(tomorrow.getHours()).toBe(9)
    expect(tomorrow.getMinutes()).toBe(0)
    expect(tomorrow.getTime()).toBeGreaterThan(today.getTime())
    expect(tomorrow.getTime() - today.getTime()).toBeLessThanOrEqual(33 * 3_600_000)
  })
})

describe('formatting', () => {
  it('formatDuration', () => {
    expect(formatDuration(900)).toBe('1s')
    expect(formatDuration(42_000)).toBe('42s')
    expect(formatDuration(72_000)).toBe('1m 12s')
    expect(formatDuration(3_780_000)).toBe('1h 03m')
  })

  it('sessionKeyFromHash', () => {
    expect(sessionKeyFromHash('#/sessions/agent%3Amain%3Awebchat%3Aabc')).toBe(
      'agent:main:webchat:abc',
    )
    expect(sessionKeyFromHash('#/sessions')).toBeNull()
    expect(sessionKeyFromHash('#/projects/p1')).toBeNull()
  })

  it('excerpt collapses whitespace and trims with an ellipsis', () => {
    expect(excerpt('  a\n\n b  ')).toBe('a b')
    expect(excerpt('x'.repeat(200), 20)).toBe(`${'x'.repeat(19)}…`)
    expect(excerpt(null)).toBe('')
  })
})

describe('trading.trigger.fired', () => {
  const TRIGGER = (
    JSON.parse(
      readFileSync('src/renderer/src/views/trading/desk/__fixtures__/trigger/trigger.json', 'utf8'),
    ) as TriggerPayload
  ).trigger
  const fire = (extra: Partial<TriggerFire> = {}): TriggerFire => ({
    n: 2,
    at: '2026-10-04T06:01:00Z',
    manual: false,
    status: 'pending',
    reasonCode: null,
    reason: null,
    priceUsd: 3790,
    orderId: 'ord_7',
    txHash: null,
    explorerUrl: null,
    ...extra,
  })
  const alert: Trigger = {
    ...TRIGGER,
    id: 'trg_al',
    name: 'Alert ETH under $3,800',
    kind: 'alert',
    action: {
      ...TRIGGER.action,
      kind: 'alert',
      amountPct: null,
      estimatedUsd: null,
      label: 'notify',
    },
  }

  it('says an alert as the news itself, a trade notification with no order', () => {
    const ev = triggerFiredEvent(
      { triggerId: alert.id, trigger: alert, fire: fire({ status: 'alerted', orderId: null }) },
      true,
    )
    expect(ev).toEqual({
      kind: 'trade',
      title: 'ETH under $3,800',
      subtitle: 'Alert · $3,790 now',
      target: { type: 'trading' },
    })
  })

  it('says a sell as what it is doing, pointing at the order it placed', () => {
    const ev = triggerFiredEvent({ triggerId: TRIGGER.id, trigger: TRIGGER, fire: fire() }, true)
    expect(ev).toEqual({
      kind: 'trade',
      title: 'Stop-loss ETH fired',
      subtitle: 'selling 50 % of ETH at $3,790',
      target: { type: 'trading', orderId: 'ord_7' },
    })
    // A parked fire is still news: the approval notification follows it.
    expect(
      triggerFiredEvent({ trigger: TRIGGER, fire: fire({ status: 'parked' }) }, true)?.kind,
    ).toBe('trade')
    const buy: Trigger = {
      ...TRIGGER,
      name: 'Buy ETH under $3,500',
      kind: 'buy',
      action: { ...TRIGGER.action, kind: 'buy', amountPct: null, amountUsd: 50 },
    }
    expect(
      triggerFiredEvent({ trigger: buy, fire: fire({ priceUsd: 3488.2 }) }, true),
    ).toMatchObject({
      title: 'Buy ETH under $3,500 fired',
      subtitle: 'buying $50 of ETH at $3,488',
    })
  })

  it('says a skip or a failure as a failed trade, with the reason in the body', () => {
    const skipped = triggerFiredEvent(
      {
        trigger: TRIGGER,
        fire: fire({
          status: 'skipped',
          orderId: null,
          reasonCode: 'insufficient_balance',
          reason: 'paused: nothing to sell',
        }),
      },
      true,
    )
    expect(skipped).toMatchObject({
      kind: 'tradeFailed',
      title: 'Stop-loss ETH skipped',
      body: 'paused: nothing to sell',
      target: { type: 'trading' },
    })
    const failed = triggerFiredEvent(
      {
        trigger: TRIGGER,
        fire: fire({ status: 'failed', orderId: null, reasonCode: 'trading.quote_failed' }),
      },
      true,
    )
    expect(failed).toMatchObject({
      kind: 'tradeFailed',
      title: 'Stop-loss ETH could not fire',
      body: 'trading.quote_failed',
    })
  })

  it('keeps subtitles and reasons behind the preview switch, and drops a thin payload', () => {
    const ev = triggerFiredEvent({ trigger: TRIGGER, fire: fire() }, false)
    expect(ev?.title).toBe('Stop-loss ETH fired')
    expect(ev?.subtitle).toBeUndefined()
    const failed = triggerFiredEvent(
      { trigger: TRIGGER, fire: fire({ status: 'failed', reason: 'no route' }) },
      false,
    )
    expect(failed?.body).toBeUndefined()
    expect(triggerFiredEvent({ triggerId: 'trg_1' }, true)).toBeNull()
    expect(triggerFiredEvent(undefined, true)).toBeNull()
  })

  it('follows the trade switches like every other trade notification', () => {
    const ok = triggerFiredEvent({ trigger: TRIGGER, fire: fire() }, true)!
    const bad = triggerFiredEvent({ trigger: TRIGGER, fire: fire({ status: 'failed' }) }, true)!
    expect(eventWanted(settings({ trades: 'all' }), ok)).toBe(true)
    expect(eventWanted(settings({ trades: 'failures' }), ok)).toBe(false)
    expect(eventWanted(settings({ trades: 'failures' }), bad)).toBe(true)
    expect(eventWanted(settings({ trades: 'off' }), bad)).toBe(false)
  })
})

describe('trading.trigger.fired · a leg of a bracket', () => {
  const BRACKET = (
    JSON.parse(
      readFileSync('src/renderer/src/views/trading/desk/__fixtures__/bracket/bracket.json', 'utf8'),
    ) as { bracket: { takeProfit: Trigger; stopLoss: Trigger } }
  ).bracket
  const fire = (extra: Partial<TriggerFire> = {}): TriggerFire => ({
    n: 1,
    at: '2026-10-04T06:01:00Z',
    manual: false,
    status: 'pending',
    reasonCode: null,
    reason: null,
    priceUsd: 4561,
    orderId: 'ord_8',
    txHash: null,
    explorerUrl: null,
    ...extra,
  })

  it('titles a sell leg with the bracket and the leg word', () => {
    expect(triggerFiredEvent({ trigger: BRACKET.takeProfit, fire: fire() }, true)).toEqual({
      kind: 'trade',
      title: 'Protect ETH · take-profit fired',
      subtitle: 'selling 100 % of ETH at $4,561',
      target: { type: 'trading', orderId: 'ord_8' },
    })
    expect(
      triggerFiredEvent({ trigger: BRACKET.stopLoss, fire: fire({ priceUsd: 3419 }) }, false)
        ?.title,
    ).toBe('Protect ETH · stop-loss fired')
  })

  it('titles a range alert leg with the bracket and the line it crossed', () => {
    const alertLeg: Trigger = {
      ...BRACKET.takeProfit,
      kind: 'alert',
      name: 'Alert ETH over $4,560',
      action: { ...BRACKET.takeProfit.action, kind: 'alert', amountPct: null, label: 'notify' },
      bracket: { id: 'brk_1a2b3c4d', name: 'Watch ETH', leg: 'tp' },
    }
    expect(
      triggerFiredEvent(
        { trigger: alertLeg, fire: fire({ status: 'alerted', orderId: null }) },
        true,
      ),
    ).toEqual({
      kind: 'trade',
      title: 'Watch ETH · over $4,560',
      subtitle: 'Alert · $4,561 now',
      target: { type: 'trading' },
    })
    // A range alert's legs are its ceiling and floor, never a take-profit.
    expect(
      triggerFiredEvent(
        {
          trigger: alertLeg,
          fire: fire({ status: 'failed', orderId: null, reason: 'price feed down' }),
        },
        true,
      )?.title,
    ).toBe('Watch ETH · ceiling could not fire')
  })

  it('names the leg on a skip or a failure too', () => {
    expect(
      triggerFiredEvent(
        {
          trigger: BRACKET.stopLoss,
          fire: fire({ status: 'skipped', orderId: null, reason: 'paused: nothing to sell' }),
        },
        true,
      ),
    ).toMatchObject({ kind: 'tradeFailed', title: 'Protect ETH · stop-loss skipped' })
  })
})
