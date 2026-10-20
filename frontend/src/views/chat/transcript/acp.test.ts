import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import {
  buildCompletionCard,
  completionOutcome,
  completionUsage,
  createAcpMounter,
  formatAcpCost,
  formatAcpDuration,
  formatAcpTokens,
  formatElapsedClock,
  harnessDisplayName,
  harnessMonogram,
  isAcpSessionKey,
  parseAcpSpawn,
  parseAcpStats,
  type AcpSpawnSpec,
} from './acp'
import { createToolRenderer, toolDisplayName, toolIconName } from './tools'

const markdown = {
  render: (text: string) => `<p>${text.replace(/</g, '&lt;')}</p>`,
  bindCopy: vi.fn(),
  bindHighlight: vi.fn(),
}

const CHILD = 'agent:main:acp:cfb95897'

/** The live event / persisted system-message JSON of contract B. */
function fullPayload(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    type: 'subagent_completion',
    parent_session_key: 'agent:main:webchat:abc',
    child_session_key: CHILD,
    task_id: 't1',
    status: 'succeeded',
    terminal_reason: 'completed',
    result: { text: 'The workspace holds **three** projects.', truncated: false },
    stats: 'runtime 50s • tokens 211207 (in 10 / out 4044) • cost $0.3995',
    runtime: 'acp',
    acp_agent: 'claude',
    acp_agent_name: 'Claude Code',
    cwd: '/Users/x/.agentos/workspace',
    mode: 'oneshot',
    harness_mode: 'plan',
    usage: {
      duration_ms: 48290,
      input_tokens: 10,
      output_tokens: 4044,
      total_tokens: 211207,
      cost_usd: 0.3995,
      tool_calls: 6,
    },
    ...overrides,
  }
}

const SPEC: AcpSpawnSpec = {
  childKey: CHILD,
  agentId: 'claude',
  agentName: 'Claude Code',
  cwd: '/Users/x/.agentos/workspace',
  mode: 'oneshot',
  harnessMode: 'plan',
  task: 'Look through the workspace and summarize it',
}

/** What the card says outside its raw-payload "Details" panel. */
function shownText(card: Element): string {
  const clone = card.cloneNode(true) as Element
  clone.querySelectorAll('.acp-details').forEach((node) => node.remove())
  return clone.textContent || ''
}

function chipTexts(root: Element): string[] {
  return Array.from(root.querySelectorAll('.acp-chips .acp-chip')).map((c) => c.textContent || '')
}

describe('ACP pure helpers', () => {
  it('recognises a harness session key by its acp segment', () => {
    expect(isAcpSessionKey('agent:main:acp:cfb95897')).toBe(true)
    expect(isAcpSessionKey('agent:main:webchat:abc')).toBe(false)
    expect(isAcpSessionKey('agent:main:acp:x:extra')).toBe(false)
    expect(isAcpSessionKey(undefined)).toBe(false)
  })

  it('names harnesses: the payload first, then the registry, then the id', () => {
    expect(harnessDisplayName('claude', 'Claude Code (beta)')).toBe('Claude Code (beta)')
    expect(harnessDisplayName('codex')).toBe('Codex')
    expect(harnessDisplayName('gemini')).toBe('Gemini CLI')
    expect(harnessDisplayName('mystery')).toBe('mystery')
    expect(harnessMonogram('Claude Code')).toBe('CC')
    expect(harnessMonogram('Codex')).toBe('CO')
  })

  it('parses the legacy stats line, every part optional', () => {
    expect(parseAcpStats('runtime 50s • tokens 211207 (in 10 / out 4044) • cost $0.3995')).toEqual({
      durationMs: 50_000,
      inputTokens: 10,
      outputTokens: 4044,
      totalTokens: 211207,
      costUsd: 0.3995,
      toolCalls: null,
    })
    expect(parseAcpStats('runtime 5m12s • tokens 900 (in 400 / out 500)').durationMs).toBe(312_000)
    expect(parseAcpStats('runtime 1h02m03s').durationMs).toBe(3_723_000)
    expect(parseAcpStats('').totalTokens).toBeNull()
    expect(parseAcpStats('garbage')).toMatchObject({ durationMs: null, costUsd: null })
  })

  it('prefers the structured usage and fills its gaps from the stats line', () => {
    const usage = completionUsage({
      usage: { duration_ms: 48290, total_tokens: null, cost_usd: null, tool_calls: 6 },
      stats: 'runtime 50s • tokens 211207 (in 10 / out 4044) • cost $0.3995',
    })
    expect(usage).toMatchObject({
      durationMs: 48290,
      totalTokens: 211207,
      costUsd: 0.3995,
      toolCalls: 6,
    })
  })

  it('maps task statuses to an outcome', () => {
    expect(completionOutcome('succeeded')).toBe('done')
    expect(completionOutcome('failed')).toBe('failed')
    expect(completionOutcome('timeout')).toBe('failed')
    expect(completionOutcome('cancelled')).toBe('cancelled')
    expect(completionOutcome('abandoned')).toBe('cancelled')
    expect(completionOutcome('failed', 'cancelled')).toBe('cancelled')
  })

  it('formats figures for chips and clocks', () => {
    expect(formatAcpDuration(48290)).toBe('48s')
    expect(formatAcpDuration(312_000)).toBe('5m 12s')
    expect(formatAcpDuration(3_723_000)).toBe('1h 02m')
    expect(formatAcpDuration(null)).toBe('')
    expect(formatAcpTokens(211207)).toBe('211k')
    expect(formatAcpTokens(4044)).toBe('4k')
    expect(formatAcpTokens(950)).toBe('950')
    expect(formatAcpCost(0.3995)).toBe('$0.40')
    expect(formatAcpCost(0.004)).toBe('<$0.01')
    expect(formatElapsedClock(7)).toBe('0:07')
    expect(formatElapsedClock(725)).toBe('12:05')
    expect(formatElapsedClock(3723)).toBe('1:02:03')
  })

  it('reads a harness spawn from its arguments and its result', () => {
    const spec = parseAcpSpawn(
      { runtime: 'acp', acp_agent: 'claude', harness_mode: 'plan', mode: 'oneshot', task: 'Go' },
      JSON.stringify({ session_key: CHILD, runtime: 'acp', acp_agent: 'claude', cwd: '/w' }),
    )
    expect(spec).toEqual({
      childKey: CHILD,
      agentId: 'claude',
      agentName: 'Claude Code',
      cwd: '/w',
      mode: 'oneshot',
      harnessMode: 'plan',
      task: 'Go',
    })
    // A plain subagent spawn, or one without a child key, stays a tool row.
    expect(parseAcpSpawn({ task: 'x' }, '{"session_key":"agent:main:subagent:1"}')).toBeNull()
    expect(parseAcpSpawn({ runtime: 'acp' }, 'spawn failed')).toBeNull()
  })
})

describe('completion card', () => {
  it('draws a full ACP payload: badge, title, pill, chips, reply, actions', () => {
    const openSession = vi.fn()
    const card = buildCompletionCard(fullPayload(), { markdown, openSession })
    document.body.appendChild(card)

    expect(card).toHaveAttribute('data-outcome', 'done')
    expect(card).toHaveAttribute('data-runtime', 'acp')
    expect(card.querySelector('.acp-badge')).toHaveAttribute('data-acp-agent', 'claude')
    expect(card.querySelector('.acp-badge')).toHaveTextContent('CC')
    expect(card.querySelector('.acp-done__title')).toHaveTextContent('Claude Code finished')
    expect(card.querySelector('.acp-done__sub')).toHaveTextContent('workspace · oneshot · plan')
    expect(card.querySelector('.acp-done__sub')).toHaveAttribute(
      'title',
      '/Users/x/.agentos/workspace',
    )
    expect(card.querySelector('.acp-pill')).toHaveTextContent('Done')
    expect(chipTexts(card)).toEqual(['48s', '211k tokens', '$0.40', '6 tool calls'])
    expect(card.querySelector('[data-kind="tokens"]')).toHaveAttribute(
      'title',
      '211,207 tokens (in 10 / out 4,044)',
    )
    // The reply goes through the markdown renderer.
    expect(card.querySelector('.acp-done__result-text p')).toHaveTextContent(
      'The workspace holds **three** projects.',
    )
    expect(markdown.bindCopy).toHaveBeenCalled()
    // Short: no clamp, no "Show more".
    expect(card.querySelector('.acp-done__result')).toHaveAttribute('data-clamp', 'false')
    expect(card.querySelector('[data-action="toggle-result"]')).toBeNull()

    card.querySelector<HTMLButtonElement>('[data-action="open-session"]')!.click()
    expect(openSession).toHaveBeenCalledWith(CHILD)

    // Details still shows the raw payload, closed by default and inert.
    const toggle = card.querySelector<HTMLButtonElement>('[data-action="toggle-details"]')!
    const region = card.querySelector<HTMLElement>('.acp-details')!
    expect(toggle).toHaveAttribute('aria-expanded', 'false')
    expect(region).toHaveAttribute('data-open', 'false')
    expect(region.firstElementChild).toHaveAttribute('inert')
    toggle.click()
    expect(toggle).toHaveAttribute('aria-expanded', 'true')
    expect(region).toHaveAttribute('data-open', 'true')
    expect(region.firstElementChild).not.toHaveAttribute('inert')
    expect(JSON.parse(region.textContent || '{}')).toMatchObject({ child_session_key: CHILD })
  })

  it('clamps a long reply behind "Show more"', () => {
    const long = Array.from({ length: 12 }, (_, i) => `Line ${i + 1}`).join('\n')
    const card = buildCompletionCard(fullPayload({ result: { text: long } }), { markdown })
    const region = card.querySelector<HTMLElement>('.acp-done__result')!
    const toggle = card.querySelector<HTMLButtonElement>('[data-action="toggle-result"]')!
    expect(region).toHaveAttribute('data-clamp', 'true')
    expect(region).toHaveAttribute('data-expanded', 'false')
    expect(toggle).toHaveTextContent('Show more')
    expect(toggle).toHaveAttribute('aria-controls', region.id)
    toggle.click()
    expect(region).toHaveAttribute('data-expanded', 'true')
    expect(toggle).toHaveTextContent('Show less')
    expect(toggle).toHaveAttribute('aria-expanded', 'true')
  })

  it('draws a legacy payload from its stats line alone', () => {
    const legacy = {
      type: 'subagent_completion',
      child_session_key: CHILD,
      status: 'succeeded',
      result: { text: 'Done.', truncated: true },
      stats: 'runtime 50s • tokens 211207 (in 10 / out 4044) • cost $0.3995',
    }
    const card = buildCompletionCard(legacy, { markdown })
    // No runtime field, but the key says harness; no name to give it.
    expect(card).toHaveAttribute('data-runtime', 'acp')
    expect(card.querySelector('.acp-done__title')).toHaveTextContent('Coding agent finished')
    expect(chipTexts(card)).toEqual(['50s', '211k tokens', '$0.40'])
    expect(card.querySelector('.acp-done__note')).toHaveTextContent('cut short')
    // No navigator handed over: no "Open session".
    expect(card.querySelector('[data-action="open-session"]')).toBeNull()

    // The mounter knows its spawn's harness and fills the name in.
    const named = buildCompletionCard(legacy, {
      markdown,
      knownAgent: () => ({ agentId: 'claude', agentName: 'Claude Code' }),
    })
    expect(named.querySelector('.acp-done__title')).toHaveTextContent('Claude Code finished')
    expect(named.querySelector('.acp-badge')).toHaveAttribute('data-acp-agent', 'claude')
  })

  it('names a plain subagent by its agent id', () => {
    const card = buildCompletionCard(
      {
        type: 'subagent_completion',
        child_session_key: 'agent:research:subagent:1',
        status: 'succeeded',
        runtime: 'subagent',
        result: { text: 'ok' },
      },
      { markdown },
    )
    expect(card).toHaveAttribute('data-runtime', 'subagent')
    expect(card.querySelector('.acp-done__title')).toHaveTextContent('Subagent research finished')
  })

  it('shows the failure message instead of the reply on a failed run', () => {
    const card = buildCompletionCard(
      fullPayload({
        status: 'failed',
        terminal_reason: 'error',
        terminal_message: 'Claude Code stopped: the harness exited (code 1).',
        error_message: 'exit 1',
        result: { text: 'partial reply' },
        usage: { duration_ms: 3000, total_tokens: null, cost_usd: null, tool_calls: null },
        stats: '',
      }),
      { markdown },
    )
    expect(card).toHaveAttribute('data-outcome', 'failed')
    expect(card.querySelector('.acp-done__title')).toHaveTextContent('Claude Code failed')
    expect(card.querySelector('.acp-pill')).toHaveTextContent('Failed')
    // The harness's own error, never the parent-facing terminal_message.
    expect(card.querySelector('.acp-done__error')).toHaveTextContent('exit 1')
    expect(shownText(card)).not.toContain('the harness exited')
    expect(card.querySelector('.acp-done__result')).toBeNull()
    // Unknown figures are omitted, not drawn as zero.
    expect(chipTexts(card)).toEqual(['3s'])

    const timedOut = buildCompletionCard(
      fullPayload({ status: 'timeout', error_message: 'Timed out after 30m' }),
      { markdown },
    )
    expect(timedOut.querySelector('.acp-pill')).toHaveTextContent('Timed out')
    expect(timedOut.querySelector('.acp-done__error')).toHaveTextContent(
      'Claude Code timed out after 6 tool calls (0:48).',
    )

    const cancelled = buildCompletionCard(fullPayload({ status: 'cancelled' }), { markdown })
    expect(cancelled.querySelector('.acp-done__title')).toHaveTextContent(
      'Claude Code was cancelled',
    )
  })

  // Round 4: terminal_message is now written for the parent model ("…Do not
  // restart it; tell the user what was done so far…"): the user never sees it.
  it("shows the gateway's user_message for a stop, never the parent-facing text", () => {
    const parentText =
      'The user stopped this Claude Code run after 21 tool calls (1:00). Do not restart it; ' +
      'tell the user what was done so far and ask how to continue.'
    const stopped = fullPayload({
      status: 'cancelled',
      terminal_reason: 'cancelled',
      stopped_by: 'user',
      terminal_message: parentText,
      user_message: 'You stopped this Claude Code run after 21 tool calls (1:00).',
      result: { text: 'Read 12 of 20 files so far.' },
    })
    const card = buildCompletionCard(stopped, { markdown })
    expect(card.querySelector('.acp-done__notice')).toHaveTextContent(
      'You stopped this Claude Code run after 21 tool calls (1:00).',
    )
    expect(card.querySelector('.acp-done__error')).toBeNull()
    expect(shownText(card)).not.toContain('Do not restart it')
    // What it did before the stop stays readable under the notice.
    expect(card.querySelector('.acp-done__result-text')).toHaveTextContent('Read 12 of 20 files')
  })

  it('derives the user line from who stopped it and the numbers when user_message is missing', () => {
    const base = {
      status: 'cancelled',
      terminal_message: 'The user stopped this run. Do not restart it; tell the user …',
      usage: { duration_ms: 60_000, tool_calls: 21 },
      stats: '',
      result: { text: '' },
    }
    const byYou = buildCompletionCard(fullPayload({ ...base, stopped_by: 'user' }), { markdown })
    expect(byYou.querySelector('.acp-done__notice')).toHaveTextContent(
      'You stopped this Claude Code run after 21 tool calls (1:00).',
    )
    const byAgent = buildCompletionCard(fullPayload({ ...base, stopped_by: 'agent' }), {
      markdown,
    })
    expect(byAgent.querySelector('.acp-done__notice')).toHaveTextContent(
      'Stopped by the agent after 21 tool calls (1:00).',
    )
    const early = buildCompletionCard(
      fullPayload({ ...base, stopped_by: 'user', usage: { duration_ms: 4_000, tool_calls: 0 } }),
      { markdown },
    )
    expect(early.querySelector('.acp-done__notice')).toHaveTextContent(
      'You stopped this Claude Code run before any tool call (0:04).',
    )
    const bare = buildCompletionCard(fullPayload({ ...base, stopped_by: 'user', usage: null }), {
      markdown,
    })
    expect(bare.querySelector('.acp-done__notice')).toHaveTextContent(
      'You stopped this Claude Code run.',
    )
    for (const card of [byYou, byAgent, early, bare]) {
      expect(shownText(card)).not.toContain('Do not restart it')
    }
  })
})

describe('harness run card + poller', () => {
  let rows: Array<Record<string, unknown>>
  let statusAnswer: Record<string, unknown>
  let call: ReturnType<typeof vi.fn>

  beforeEach(() => {
    vi.useFakeTimers()
    document.body.innerHTML = ''
    rows = []
    statusAnswer = {}
    call = vi.fn(async (method: string) => {
      if (method === 'acp.sessions') return { sessions: rows, count: rows.length }
      if (method === 'acp.status') return statusAnswer
      if (method === 'acp.cancel') return { session_key: CHILD, cancelled: true }
      return {}
    })
  })

  afterEach(() => {
    vi.useRealTimers()
    Reflect.deleteProperty(document, 'hidden')
  })

  function mounter(openSession: ((key: string) => void) | null = null) {
    return createAcpMounter({ call, markdown, openSession: () => openSession, pollMs: 1500 })
  }

  const callsOf = (method: string) => call.mock.calls.filter((c) => c[0] === method).length

  it('runs live, follows the harness, and settles when the completion arrives', async () => {
    const openSession = vi.fn()
    const m = mounter(openSession)
    const card = m.harnessCard(SPEC, { live: true, toolId: 'tool-1' })
    document.body.appendChild(card)

    expect(card).toHaveAttribute('data-state', 'running')
    expect(card).toHaveAttribute('data-enter', 'live')
    expect(card).toHaveAttribute('data-tool-id', 'tool-1')
    expect(card.querySelector('.acp-run__name')).toHaveTextContent('Claude Code')
    expect(card.querySelector('.acp-run__cwd')).toHaveTextContent('workspace')
    expect(chipTexts(card)).toEqual(['plan', 'oneshot'])
    expect(card.querySelector('.acp-run__activity')).toHaveTextContent('Starting Claude Code…')
    expect(card.querySelector('[data-action="stop"]')).not.toHaveAttribute('hidden')
    expect(m.isPolling()).toBe(true)

    rows = [
      {
        session_key: CHILD,
        active_turn: true,
        activity: 'Read src/app.py',
        activity_kind: 'read',
        turn_tool_calls: 3,
        turn_elapsed_s: 12,
      },
    ]
    await vi.advanceTimersByTimeAsync(1500)
    expect(callsOf('acp.sessions')).toBe(1)
    const activity = card.querySelector('.acp-run__activity')!
    // The new step slides in over the old one, which leaves after its exit.
    expect(activity.querySelector('[data-entering]')).toHaveTextContent('Read src/app.py')
    expect(activity.querySelector('[data-leaving]')).toHaveTextContent('Starting Claude Code…')
    await vi.advanceTimersByTimeAsync(400)
    expect(activity.querySelectorAll('.acp-run__activity-text')).toHaveLength(1)
    expect(card.querySelector('.acp-run__tools')).toHaveTextContent('3 tool calls')
    expect(card.querySelector('.acp-run__elapsed')!.textContent).toMatch(/^0:1[2-4]$/)
    expect(card.querySelector('.acp-run__status')).toHaveAttribute('data-activity-kind', 'read')

    card.querySelector<HTMLButtonElement>('[data-action="open-session"]')!.click()
    expect(openSession).toHaveBeenCalledWith(CHILD)

    // The completion row settles the card, with the settle animation.
    m.completionCard(fullPayload(), '{}')
    expect(card).toHaveAttribute('data-state', 'done')
    expect(card).toHaveAttribute('data-settled', 'true')
    expect(card.querySelector('.acp-run__activity')).toHaveTextContent('Finished in 48s')
    expect(card.querySelector('.acp-run__tools')).toHaveTextContent('6 tool calls')
    expect(card.querySelector('.acp-run__cost')).toHaveTextContent('$0.40')
    expect(card.querySelector('[data-action="stop"]')).toHaveAttribute('hidden')
    expect(card.querySelector('.acp-run__elapsed')).toHaveAttribute('hidden')
    expect(m.isPolling()).toBe(false)

    const before = callsOf('acp.sessions')
    await vi.advanceTimersByTimeAsync(6000)
    expect(callsOf('acp.sessions')).toBe(before)
  })

  it('keeps its state through a transcript rebuild, without replaying the enter', async () => {
    const m = mounter()
    const first = m.harnessCard(SPEC, { live: true, toolId: 't' })
    document.body.appendChild(first)
    rows = [{ session_key: CHILD, active_turn: true, activity: 'Run pytest -q' }]
    await vi.advanceTimersByTimeAsync(1500)

    // The history resync rebuilds the bubble: the old card leaves the DOM.
    first.remove()
    const rebuilt = m.harnessCard(SPEC, { live: false, toolId: 't' })
    document.body.appendChild(rebuilt)
    expect(rebuilt).toHaveAttribute('data-state', 'running')
    expect(rebuilt).not.toHaveAttribute('data-enter')
    expect(rebuilt.querySelector('.acp-run__activity')).toHaveTextContent('Run pytest -q')
    expect(callsOf('acp.status')).toBe(0)
  })

  it('on history, settles from a later completion with no status call and no poller', async () => {
    const m = mounter()
    const card = m.harnessCard(SPEC, { live: false, toolId: 't' })
    document.body.appendChild(card)
    // Never a spinner it cannot vouch for.
    expect(card).toHaveAttribute('data-state', 'checking')
    expect(card.querySelector('.acp-run__activity')).toHaveTextContent('Checking status…')
    // ... the completion row later in the same history pass:
    m.completionCard(fullPayload(), '{}')
    expect(card).toHaveAttribute('data-state', 'done')
    expect(card).not.toHaveAttribute('data-settled')
    await vi.advanceTimersByTimeAsync(5000)
    expect(callsOf('acp.status')).toBe(0)
    expect(callsOf('acp.sessions')).toBe(0)
    expect(m.isPolling()).toBe(false)

    // A card built after the completion was seen settles at once.
    const again = m.harnessCard(SPEC, { live: false, toolId: 't' })
    expect(again).toHaveAttribute('data-state', 'done')
  })

  it('on history without a completion, asks acp.status once and settles from last_turn', async () => {
    statusAnswer = {
      session_key: CHILD,
      active_turn: false,
      idle: true,
      last_turn: { duration_ms: 9000, tool_calls: 2, stop_reason: 'end_turn' },
    }
    const m = mounter()
    const card = m.harnessCard(SPEC, { live: false })
    document.body.appendChild(card)
    await vi.advanceTimersByTimeAsync(0)
    expect(callsOf('acp.status')).toBe(1)
    expect(call).toHaveBeenCalledWith('acp.status', { session_key: CHILD })
    expect(card).toHaveAttribute('data-state', 'done')
    expect(card.querySelector('.acp-run__activity')).toHaveTextContent('Finished in 9s')
    expect(m.isPolling()).toBe(false)
  })

  it('reads a stopped last turn as cancelled, never as failed', async () => {
    statusAnswer = {
      session_key: CHILD,
      last_turn: {
        stopped_by: 'user',
        error: 'Harness turn was cancelled by an operator or the parent agent; do not retry it',
        tool_calls: 3,
        duration_ms: 20_000,
      },
    }
    const m = mounter()
    const card = m.harnessCard(SPEC, { live: false })
    document.body.appendChild(card)
    await vi.advanceTimersByTimeAsync(0)
    expect(card).toHaveAttribute('data-state', 'cancelled')
    expect(card.querySelector('.acp-run__activity')).toHaveTextContent('Cancelled')
  })

  it('drops a running step whole on a narrow card and keeps it in the row tooltip', async () => {
    const m = mounter()
    const card = m.harnessCard(SPEC, { live: true })
    document.body.appendChild(card)
    rows = [{ session_key: CHILD, active_turn: true, activity: 'Read src/app.py' }]
    await vi.advanceTimersByTimeAsync(1500)
    expect(card.querySelector('.acp-run__status')).toHaveAttribute('title', 'Read src/app.py')
  })

  it('on history with the child still running, starts polling from acp.status', async () => {
    statusAnswer = {
      session_key: CHILD,
      active_turn: true,
      turn_elapsed_s: 30,
      activity: 'Edit a.py',
    }
    const m = mounter()
    const card = m.harnessCard(SPEC, { live: false })
    document.body.appendChild(card)
    await vi.advanceTimersByTimeAsync(0)
    expect(card).toHaveAttribute('data-state', 'running')
    expect(card.querySelector('.acp-run__elapsed')).toHaveTextContent('0:30')
    expect(m.isPolling()).toBe(true)
  })

  it('shows a quiet unknown state when acp.status is not there (older gateway)', async () => {
    call.mockImplementation(async (method: string) => {
      if (method === 'acp.status')
        throw Object.assign(new Error('nope'), { code: 'METHOD_NOT_FOUND' })
      return {}
    })
    const m = mounter()
    const card = m.harnessCard(SPEC, { live: false })
    document.body.appendChild(card)
    await vi.advanceTimersByTimeAsync(0)
    expect(card).toHaveAttribute('data-state', 'unknown')
    expect(card.querySelector('.acp-run__activity')).toHaveTextContent('No live status')
    expect(m.isPolling()).toBe(false)
  })

  it('stops polling when its card leaves the DOM, on unmount, and while hidden', async () => {
    const m = mounter()
    const card = m.harnessCard(SPEC, { live: true })
    document.body.appendChild(card)
    rows = [{ session_key: CHILD, active_turn: true }]
    await vi.advanceTimersByTimeAsync(1500)
    expect(callsOf('acp.sessions')).toBe(1)

    // Hidden window: no calls; back in front: an immediate read.
    Object.defineProperty(document, 'hidden', { configurable: true, get: () => true })
    document.dispatchEvent(new Event('visibilitychange'))
    await vi.advanceTimersByTimeAsync(6000)
    expect(callsOf('acp.sessions')).toBe(1)
    Object.defineProperty(document, 'hidden', { configurable: true, get: () => false })
    document.dispatchEvent(new Event('visibilitychange'))
    await vi.advanceTimersByTimeAsync(0)
    expect(callsOf('acp.sessions')).toBe(2)

    // The card is gone (a session switch): the next tick finds nothing to poll.
    card.remove()
    await vi.advanceTimersByTimeAsync(1500)
    await vi.advanceTimersByTimeAsync(6000)
    expect(m.isPolling()).toBe(false)
    const after = callsOf('acp.sessions')

    // A new running card re-arms; unmount stops it for good.
    const next = m.harnessCard({ ...SPEC, childKey: 'agent:main:acp:2' }, { live: true })
    document.body.appendChild(next)
    expect(m.isPolling()).toBe(true)
    m.destroyAll()
    expect(m.isPolling()).toBe(false)
    await vi.advanceTimersByTimeAsync(6000)
    expect(callsOf('acp.sessions')).toBe(after)
  })

  it('asks acp.status itself when the turn ended without a completion', async () => {
    const m = mounter()
    const card = m.harnessCard(SPEC, { live: true })
    document.body.appendChild(card)
    rows = [{ session_key: CHILD, active_turn: true }]
    await vi.advanceTimersByTimeAsync(1500)
    rows = []
    statusAnswer = { session_key: CHILD, last_turn: { error: 'harness crashed', error_code: 'X' } }
    await vi.advanceTimersByTimeAsync(1500 * 2)
    expect(callsOf('acp.status')).toBe(1)
    expect(card).toHaveAttribute('data-state', 'failed')
    expect(card).toHaveAttribute('data-settled', 'true')
    expect(m.isPolling()).toBe(false)
  })

  it('Stop calls acp.cancel once, disabled while in flight, and reports the answer', async () => {
    let resolveCancel: (value: unknown) => void = () => {}
    call.mockImplementation((method: string) => {
      if (method === 'acp.cancel') return new Promise((resolve) => (resolveCancel = resolve))
      if (method === 'acp.sessions') return Promise.resolve({ sessions: rows })
      return Promise.resolve(statusAnswer)
    })
    const m = mounter()
    const card = m.harnessCard(SPEC, { live: true })
    document.body.appendChild(card)
    const stop = card.querySelector<HTMLButtonElement>('[data-action="stop"]')!
    stop.click()
    stop.click()
    expect(call.mock.calls.filter((c) => c[0] === 'acp.cancel')).toEqual([
      ['acp.cancel', { session_key: CHILD }],
    ])
    expect(stop).toBeDisabled()
    expect(stop).toHaveAttribute('data-pending', 'true')
    resolveCancel({ session_key: CHILD, cancelled: true })
    await vi.advanceTimersByTimeAsync(0)
    expect(stop).toBeDisabled()
    expect(stop).toHaveTextContent('Stopping…')
    m.completionCard(fullPayload({ status: 'cancelled' }), '{}')
    expect(card).toHaveAttribute('data-state', 'cancelled')
    expect(stop).toHaveAttribute('hidden')
  })

  it('Stop says so when nothing was running, and shows an error inline', async () => {
    // Queued behind the task runtime: no turn yet, the session row still runs.
    statusAnswer = { session_key: CHILD, active_turn: false, session_status: 'running' }
    const m = mounter()
    const card = m.harnessCard(SPEC, { live: true })
    document.body.appendChild(card)
    const stop = card.querySelector<HTMLButtonElement>('[data-action="stop"]')!
    const note = card.querySelector<HTMLElement>('.acp-run__note')!

    call.mockImplementationOnce(async () => ({ session_key: CHILD, cancelled: false }))
    stop.click()
    await vi.advanceTimersByTimeAsync(0)
    expect(note).toHaveTextContent('Nothing was running.')
    expect(note).not.toHaveAttribute('hidden')
    expect(stop).not.toBeDisabled()

    call.mockImplementationOnce(async () => {
      throw new Error('ACP_UNAVAILABLE')
    })
    stop.click()
    await vi.advanceTimersByTimeAsync(0)
    expect(note).toHaveTextContent('Could not stop it: ACP_UNAVAILABLE')
    expect(note).toHaveAttribute('data-tone', 'error')
    expect(stop).not.toBeDisabled()
  })

  // Round 2 (live test): the harness is done but the chat still persists and
  // announces; the card must not keep a ticking clock and a live Stop.
  it('wraps up as soon as the harness turn ends: clock stops, Stop goes, status asked', async () => {
    const m = mounter()
    const card = m.harnessCard(SPEC, { live: true })
    document.body.appendChild(card)
    rows = [{ session_key: CHILD, active_turn: true, activity: 'Edit a.py', turn_elapsed_s: 40 }]
    await vi.advanceTimersByTimeAsync(1500)
    expect(card.querySelector('.acp-run__elapsed')).toHaveTextContent('0:40')

    // The harness finished; its record is not written yet.
    rows = []
    statusAnswer = { session_key: CHILD, active_turn: false }
    await vi.advanceTimersByTimeAsync(1500)
    expect(callsOf('acp.status')).toBe(1)
    expect(card).toHaveAttribute('data-state', 'running')
    expect(card.querySelector('[data-action="stop"]')).toHaveAttribute('hidden')
    await vi.advanceTimersByTimeAsync(400)
    expect(card.querySelector('.acp-run__activity')).toHaveTextContent('Wrapping up…')
    const frozen = card.querySelector('.acp-run__elapsed')!.textContent
    await vi.advanceTimersByTimeAsync(5000)
    expect(card.querySelector('.acp-run__elapsed')!.textContent).toBe(frozen)
    // Still wrapping, never back to the stale step.
    expect(card.querySelector('.acp-run__activity')).toHaveTextContent('Wrapping up…')

    m.completionCard(fullPayload(), '{}')
    expect(card).toHaveAttribute('data-state', 'done')
  })

  it('keeps the live numbers when a cancelled turn reports no usage', async () => {
    const m = mounter()
    const card = m.harnessCard(SPEC, { live: true })
    document.body.appendChild(card)
    rows = [{ session_key: CHILD, active_turn: true, turn_tool_calls: 4, turn_elapsed_s: 100 }]
    await vi.advanceTimersByTimeAsync(1500)
    const cancelled = fullPayload({
      status: 'cancelled',
      terminal_reason: 'cancelled',
      terminal_message: 'Stopped by the operator.',
      usage: null,
      stats: '',
      result: { text: '' },
    })
    const completion = m.completionCard(cancelled, JSON.stringify(cancelled))
    expect(card).toHaveAttribute('data-state', 'cancelled')
    expect(card.querySelector('.acp-run__tools')).toHaveTextContent('4 tool calls')
    expect(card.querySelector('.acp-run__activity')).toHaveTextContent('Cancelled')
    // The completion card shows them too, and the stop as a notice, not an error.
    expect(chipTexts(completion)).toEqual(['1m 40s', '4 tool calls'])
    expect(completion.querySelector('.acp-done__notice')).toHaveTextContent(
      'Claude Code was stopped after 4 tool calls (1:40).',
    )
    expect(shownText(completion)).not.toContain('Stopped by the operator.')
    expect(completion.querySelector('.acp-done__error')).toBeNull()
  })

  it('retries a status read that failed for want of a connection, a bounded number of times', async () => {
    let failures = 2
    call.mockImplementation(async (method: string) => {
      if (method === 'acp.status') {
        if (failures > 0) {
          failures -= 1
          throw new Error('Not connected')
        }
        return { session_key: CHILD, last_turn: { duration_ms: 3000, stop_reason: 'end_turn' } }
      }
      return {}
    })
    const m = mounter()
    const card = m.harnessCard(SPEC, { live: false })
    document.body.appendChild(card)
    await vi.advanceTimersByTimeAsync(0)
    expect(card).toHaveAttribute('data-state', 'checking')
    await vi.advanceTimersByTimeAsync(1000)
    expect(card).toHaveAttribute('data-state', 'checking')
    await vi.advanceTimersByTimeAsync(2000)
    expect(callsOf('acp.status')).toBe(3)
    expect(card).toHaveAttribute('data-state', 'done')
  })

  it('gives up after its retries, and asks again when the connection comes back', async () => {
    let connected = false
    call.mockImplementation(async (method: string) => {
      if (method !== 'acp.status') return {}
      if (!connected) throw new Error('Not connected')
      return { session_key: CHILD, last_turn: { duration_ms: 2000 } }
    })
    const m = mounter()
    const card = m.harnessCard(SPEC, { live: false })
    document.body.appendChild(card)
    await vi.advanceTimersByTimeAsync(0)
    await vi.advanceTimersByTimeAsync(1000 + 2000 + 4000 + 8000)
    expect(callsOf('acp.status')).toBe(5)
    expect(card).toHaveAttribute('data-state', 'unknown')
    await vi.advanceTimersByTimeAsync(30_000)
    expect(callsOf('acp.status')).toBe(5)

    connected = true
    m.reconnected()
    await vi.advanceTimersByTimeAsync(0)
    expect(callsOf('acp.status')).toBe(6)
    expect(card).toHaveAttribute('data-state', 'done')
  })
})

describe('tool renderer: a harness spawn renders as its run card', () => {
  function harness() {
    const bubble = document.createElement('div')
    const body = document.createElement('div')
    body.className = 'msg-body'
    bubble.appendChild(body)
    document.body.appendChild(bubble)
    const call = vi.fn(async () => ({}))
    const m = createAcpMounter({ call, markdown })
    const renderer = createToolRenderer({
      ensureStreamBubble: () => bubble,
      markVisibleStreamEvent: () => {},
      flushPendingTextSegment: () => {},
      newTextSegment: () => document.createElement('div'),
      scrollToBottom: () => {},
      getAutoScroll: () => false,
      pushSegment: () => {},
      getSearchProvider: () => '',
      setSearchProvider: () => {},
      getSessionKey: () => 'agent:main:webchat:abc',
      addMessage: () => null,
      pushMessage: () => {},
      harnessCard: (spec, options) => m.harnessCard(spec, options),
    })
    return { body, renderer, m }
  }

  const spawnResult = JSON.stringify({
    session_key: CHILD,
    runtime: 'acp',
    acp_agent: 'claude',
    cwd: '/Users/x/.agentos/workspace',
    mode: 'oneshot',
    status: 'queued',
  })

  it('replaces the live tool row on the tool result', () => {
    const { body, renderer, m } = harness()
    renderer.appendToolCall({ name: 'sessions_spawn', tool_use_id: 'sp-1' })
    expect(body.querySelector('.chat-tools-collapse')).not.toBeNull()
    renderer.appendToolResult({
      tool_name: 'sessions_spawn',
      tool_use_id: 'sp-1',
      result: spawnResult,
      arguments: { runtime: 'acp', acp_agent: 'claude', harness_mode: 'plan', task: 'Go' },
    })
    expect(body.querySelector('.chat-tools-collapse')).toBeNull()
    const card = body.querySelector('.acp-run')!
    expect(card).toHaveAttribute('data-tool-id', 'sp-1')
    expect(card).toHaveAttribute('data-state', 'running')
    // A replayed result is not drawn twice.
    renderer.appendToolResult({
      tool_name: 'sessions_spawn',
      tool_use_id: 'sp-1',
      result: spawnResult,
    })
    expect(body.querySelectorAll('.acp-run')).toHaveLength(1)
    expect(body.querySelector('.chat-tool-result')).toBeNull()
    m.destroyAll()
  })

  it('rebuilds the card from persisted segments, and leaves a plain spawn alone', () => {
    const { body, renderer, m } = harness()
    const bubble = body.parentElement as HTMLElement
    renderer.reconstructToolCalls(
      bubble,
      [
        {
          type: 'tool_use',
          name: 'sessions_spawn',
          tool_use_id: 'sp-1',
          input: { runtime: 'acp', acp_agent: 'claude', task: 'Go' },
        },
        { type: 'tool_result', tool_use_id: 'sp-1', content: spawnResult },
        {
          type: 'tool_use',
          name: 'sessions_spawn',
          tool_use_id: 'sp-2',
          input: { task: 'research' },
        },
        {
          type: 'tool_result',
          tool_use_id: 'sp-2',
          content: '{"session_key":"agent:main:subagent:1","status":"queued"}',
        },
      ],
      { renderText: (text, into) => (into.textContent = text), stripText: (text) => text },
    )
    expect(body.querySelector('.acp-run')).toHaveAttribute('data-state', 'checking')
    expect(body.querySelector('[data-tool-id="sp-2"]')).toHaveClass('chat-tools-collapse')
    m.destroyAll()
  })

  it('keeps the generic row for a failed spawn', () => {
    const { body, renderer, m } = harness()
    renderer.appendToolCall({ name: 'sessions_spawn', tool_use_id: 'sp-3' })
    renderer.appendToolResult({
      tool_name: 'sessions_spawn',
      tool_use_id: 'sp-3',
      result: 'cwd refused',
      is_error: true,
      arguments: { runtime: 'acp' },
    })
    expect(body.querySelector('.acp-run')).toBeNull()
    expect(body.querySelector('.chat-tools-collapse--error')).not.toBeNull()
    m.destroyAll()
  })
})

describe('tool renderer: a harness tool reads as its title', () => {
  function harness() {
    const bubble = document.createElement('div')
    const body = document.createElement('div')
    body.className = 'msg-body'
    bubble.appendChild(body)
    const renderer = createToolRenderer({
      ensureStreamBubble: () => bubble,
      markVisibleStreamEvent: () => {},
      flushPendingTextSegment: () => {},
      newTextSegment: () => document.createElement('div'),
      scrollToBottom: () => {},
      getAutoScroll: () => false,
      pushSegment: () => {},
      getSearchProvider: () => '',
      setSearchProvider: () => {},
      getSessionKey: () => CHILD,
      addMessage: () => null,
      pushMessage: () => {},
    })
    return { bubble, body, renderer }
  }
  const nameOf = (body: Element, id: string) =>
    body.querySelector(`[data-tool-id="${id}"] .chat-tools-name`)

  it('prefers the title over the raw acp:<kind> name, and picks the kind icon', () => {
    expect(toolDisplayName('acp:read', { title: 'Read src/app.py', path: 'src/app.py' })).toBe(
      'Read src/app.py',
    )
    expect(toolDisplayName('acp:execute', '{"title":"Run pytest -q"}')).toBe('Run pytest -q')
    expect(toolDisplayName('acp:execute', {}, 'Run make')).toBe('Run make')
    expect(toolDisplayName('acp:other', {})).toBe('acp:other')
    // Only harness tools: an ordinary tool's `title` argument is not its name.
    expect(toolDisplayName('write_file', { title: 'x' })).toBe('write_file')
    expect(toolIconName('acp:read')).toBe('file-text')
    expect(toolIconName('acp:execute')).toBe('terminal')
    expect(toolIconName('acp:mystery')).toBe('zap')
  })

  it('relabels a live row when the title arrives with the result', () => {
    const { body, renderer } = harness()
    renderer.appendToolCall({ tool_name: 'acp:execute', tool_use_id: 't1' })
    expect(nameOf(body, 't1')).toHaveTextContent('acp:execute')
    renderer.appendToolResult({
      tool_name: 'acp:execute',
      tool_use_id: 't1',
      result: 'ok',
      arguments: { title: 'Run pytest -q', command: 'pytest -q' },
    })
    expect(nameOf(body, 't1')).toHaveTextContent('Run pytest -q')
    expect(nameOf(body, 't1')).toHaveAttribute('title', 'acp:execute')
    expect(body.querySelector('[data-tool-id="t1"] .chat-tools-icon')).toHaveAttribute(
      'data-icon',
      'terminal',
    )
    // An explicit title on the event wins too.
    renderer.appendToolCall({ tool_name: 'acp:read', tool_use_id: 't2', title: 'Read a.py' })
    expect(nameOf(body, 't2')).toHaveTextContent('Read a.py')
  })

  it('rebuilds titled rows from history, and a result without one keeps the title', () => {
    const { bubble, body, renderer } = harness()
    renderer.reconstructToolCalls(
      bubble,
      [
        {
          type: 'tool_use',
          name: 'acp:read',
          tool_use_id: 'r1',
          input: { title: 'Read src/app.py', path: 'src/app.py' },
        },
        { type: 'tool_result', name: 'acp:read', tool_use_id: 'r1', result: 'contents' },
        { type: 'tool_use', name: 'acp:edit', tool_use_id: 'r2', title: 'Edit b.py', input: {} },
        { type: 'tool_result', name: 'acp:edit', tool_use_id: 'r2', result: 'ok' },
      ],
      { renderText: (text, into) => (into.textContent = text), stripText: (text) => text },
    )
    expect(nameOf(body, 'r1')).toHaveTextContent('Read src/app.py')
    expect(nameOf(body, 'r2')).toHaveTextContent('Edit b.py')
  })
})
