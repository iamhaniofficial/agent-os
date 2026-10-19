// Chat transcript — coding-harness (ACP) cards.
//
// Two cards, both imperative DOM like the rest of the transcript region:
//
//   1. The completion card (`buildCompletionCard`): a `subagent_completion`
//      system row. Replaces the raw-JSON disclosure with a harness badge, a
//      title, a status pill, stat chips, the reply (clamped, "Show more"), an
//      "Open session" action and a small "Details" toggle that still shows the
//      raw payload for debugging.
//   2. The harness run card (`createAcpMounter().harnessCard`): what a
//      `sessions_spawn(runtime="acp")` tool call renders as instead of the
//      generic tool row. Live status (activity, elapsed, tool calls) comes from
//      one shared `acp.sessions` poller per mounted transcript, which runs only
//      while a connected card is running and the window is visible.
//
// Both read the contract in BRIEF-acpfix-contract.md (B and C) and degrade
// when a field is missing: an older gateway, or a payload persisted before
// the fields existed, carries only the `stats` string
// (`runtime 50s • tokens 211207 (in 10 / out 4044) • cost $0.3995`).
//
// The DOM is skinned twice: plainly by the web console (chat.css) and with
// the desktop's own motion and brand marks (desktop views/chat/acp.css). The
// renderer stamps `data-acp-agent`, `data-state` and `data-outcome` for that.

import { t, tPlural } from '@/i18n'
import '@/i18n/en/chat'

import { agentIdFromSessionKey } from '../logic'
import type { MarkdownDep } from './stream'

/* ── Types ──────────────────────────────────────────────────────────────── */

export type AcpOutcome = 'done' | 'failed' | 'cancelled'
export type AcpRunState = 'running' | 'checking' | 'unknown' | AcpOutcome

/** Usage figures of one harness turn; `null` = unknown (omitted from the card). */
export interface AcpUsage {
  durationMs: number | null
  inputTokens: number | null
  outputTokens: number | null
  totalTokens: number | null
  costUsd: number | null
  toolCalls: number | null
}

/** What a `sessions_spawn(runtime="acp")` call tells the transcript. */
export interface AcpSpawnSpec {
  childKey: string
  agentId: string
  agentName: string
  cwd: string
  mode: string
  harnessMode: string
  task: string
}

/** One row of `acp.sessions` / the object `acp.status` returns (contract C). */
export interface AcpLiveRow {
  session_key?: string
  active_turn?: boolean
  agent_id?: string
  agent_name?: string | null
  activity?: string | null
  activity_kind?: string | null
  turn_tool_calls?: number | null
  turn_elapsed_s?: number | null
  cwd?: string | null
  mode?: string | null
  harness_mode?: string | null
  idle?: boolean
  closed?: boolean
  last_turn?: Record<string, unknown> | null
  session_status?: string | null
  [key: string]: unknown
}

export type AcpCall = (method: string, params: Record<string, unknown>) => Promise<unknown>
/** Navigate the chat to a session. Absent: no "Open session" action. */
export type AcpOpenSession = (sessionKey: string) => void

/* ── Pure helpers (unit-tested) ─────────────────────────────────────────── */

/** `agent:<agent>:acp:<id>` — a hint only; `acp.status` is what confirms it. */
export function isAcpSessionKey(key: unknown): boolean {
  return typeof key === 'string' && /^agent:[^:]+:acp:[^:]+$/.test(key.trim())
}

/**
 * Registry display names (src/agentos/acp/registry.py) for the harnesses a
 * payload may name without `acp_agent_name` (an older gateway). Anything else
 * reads as its id.
 */
const HARNESS_NAMES: Record<string, string> = {
  claude: 'Claude Code',
  codex: 'Codex',
  gemini: 'Gemini CLI',
  cursor: 'Cursor',
  copilot: 'GitHub Copilot',
  opencode: 'OpenCode',
  pi: 'Pi',
  droid: 'Factory Droid',
  'grok-build': 'Grok Build',
  kilocode: 'Kilo Code',
  kimi: 'Kimi Code',
  kiro: 'Kiro',
  qwen: 'Qwen Code',
  antigravity: 'Google Antigravity',
}

function text(value: unknown): string {
  return typeof value === 'string' ? value.trim() : ''
}

/** Display name: the payload's own name first, then the registry's, then the id. */
export function harnessDisplayName(agentId: unknown, explicit?: unknown): string {
  const named = text(explicit)
  if (named) return named
  const id = text(agentId).toLowerCase()
  return HARNESS_NAMES[id] || text(agentId)
}

/** Two letters for the badge: initials of two words, else the first two letters. */
export function harnessMonogram(name: string): string {
  const words = name
    .trim()
    .split(/[\s_-]+/)
    .filter(Boolean)
  if (words.length >= 2) return ((words[0] ?? '')[0]! + (words[1] ?? '')[0]!).toUpperCase()
  return ((words[0] ?? '').slice(0, 2) || '?').toUpperCase()
}

function finite(value: unknown): number | null {
  if (value === null || value === undefined || value === '') return null
  const n = typeof value === 'number' ? value : Number(value)
  return Number.isFinite(n) ? n : null
}

const EMPTY_USAGE: AcpUsage = {
  durationMs: null,
  inputTokens: null,
  outputTokens: null,
  totalTokens: null,
  costUsd: null,
  toolCalls: null,
}

/** `5m12s` / `1h02m03s` / `50s` → milliseconds. */
function parseRuntime(raw: string): number | null {
  const match = raw.match(/^(?:(\d+)h)?(?:(\d+)m)?(?:(\d+(?:\.\d+)?)s)?$/)
  if (!match || !raw) return null
  const [, h, m, s] = match
  if (h === undefined && m === undefined && s === undefined) return null
  return Math.round(((Number(h || 0) * 60 + Number(m || 0)) * 60 + Number(s || 0)) * 1000)
}

/**
 * Parse the legacy `stats` line the announce writes
 * (`runtime 50s • tokens 211207 (in 10 / out 4044) • cost $0.3995`). Any
 * part may be missing; an unparseable line yields all-null usage.
 */
export function parseAcpStats(stats: unknown): AcpUsage {
  const line = text(stats)
  if (!line) return { ...EMPTY_USAGE }
  const usage: AcpUsage = { ...EMPTY_USAGE }
  const runtime = line.match(/runtime\s+([0-9hms.]+)/i)
  if (runtime?.[1]) usage.durationMs = parseRuntime(runtime[1])
  const tokens = line.match(/tokens\s+([\d,]+)/i)
  if (tokens?.[1]) usage.totalTokens = finite(tokens[1].replace(/,/g, ''))
  const input = line.match(/\bin\s+([\d,]+)/i)
  if (input?.[1]) usage.inputTokens = finite(input[1].replace(/,/g, ''))
  const output = line.match(/\bout\s+([\d,]+)/i)
  if (output?.[1]) usage.outputTokens = finite(output[1].replace(/,/g, ''))
  const cost = line.match(/cost\s+\$?([\d.]+)/i)
  if (cost?.[1]) usage.costUsd = finite(cost[1])
  const tools = line.match(/(\d+)\s+tool(?:\s+call)?s?\b/i)
  if (tools?.[1]) usage.toolCalls = finite(tools[1])
  return usage
}

/** A `usage` object (contract B) or a `last_turn` record → normalized usage. */
export function usageFromRecord(record: unknown): AcpUsage {
  if (!record || typeof record !== 'object') return { ...EMPTY_USAGE }
  const r = record as Record<string, unknown>
  return {
    durationMs: finite(r.duration_ms),
    inputTokens: finite(r.input_tokens),
    outputTokens: finite(r.output_tokens),
    totalTokens: finite(r.total_tokens),
    costUsd: finite(r.cost_usd),
    toolCalls: finite(r.tool_calls),
  }
}

/** The completion's usage: the structured `usage` first, the `stats` line for the gaps. */
export function completionUsage(payload: Record<string, unknown>): AcpUsage {
  const structured = usageFromRecord(payload.usage)
  const legacy = parseAcpStats(payload.stats)
  const merged = { ...structured }
  ;(Object.keys(merged) as Array<keyof AcpUsage>).forEach((key) => {
    if (merged[key] === null) merged[key] = legacy[key]
  })
  return merged
}

/** `usage` with each unknown (null) figure taken from `fallback`, when it has one. */
export function fillUsage(usage: AcpUsage, fallback: AcpUsage | null | undefined): AcpUsage {
  if (!fallback) return usage
  const filled = { ...usage }
  ;(Object.keys(filled) as Array<keyof AcpUsage>).forEach((key) => {
    if (filled[key] === null) filled[key] = fallback[key]
  })
  return filled
}

/** Task status (`succeeded|failed|timeout|cancelled|abandoned`) → card outcome. */
export function completionOutcome(status: unknown, terminalReason?: unknown): AcpOutcome {
  const s = text(status).toLowerCase()
  const reason = text(terminalReason).toLowerCase()
  if (s === 'succeeded' || s === 'done' || s === 'completed' || s === 'success') return 'done'
  if (s === 'cancelled' || s === 'canceled' || s === 'abandoned' || s === 'killed') {
    return 'cancelled'
  }
  if (reason === 'cancelled' || reason === 'canceled') return 'cancelled'
  return 'failed'
}

/** `48s`, `5m 12s`, `1h 02m`. Zero/unknown → ''. */
export function formatAcpDuration(ms: number | null | undefined): string {
  if (ms === null || ms === undefined || !Number.isFinite(ms) || ms <= 0) return ''
  const total = Math.max(1, Math.round(ms / 1000))
  const h = Math.floor(total / 3600)
  const m = Math.floor((total % 3600) / 60)
  const s = total % 60
  if (h) return `${h}h ${String(m).padStart(2, '0')}m`
  if (m) return `${m}m ${String(s).padStart(2, '0')}s`
  return `${s}s`
}

/** Elapsed clock for a running turn: `0:07`, `12:05`, `1:02:03`. */
export function formatElapsedClock(seconds: number): string {
  const total = Math.max(0, Math.floor(seconds))
  const h = Math.floor(total / 3600)
  const m = Math.floor((total % 3600) / 60)
  const s = String(total % 60).padStart(2, '0')
  return h ? `${h}:${String(m).padStart(2, '0')}:${s}` : `${m}:${s}`
}

/** `211k`, `1.2M`, `950`. */
export function formatAcpTokens(n: number): string {
  if (n >= 1_000_000) return `${+(n / 1_000_000).toFixed(1)}M`
  if (n >= 10_000) return `${Math.round(n / 1_000)}k`
  if (n >= 1_000) return `${+(n / 1_000).toFixed(1)}k`
  return String(Math.round(n))
}

/** `$0.40`, `$12.35`, `<$0.01`. */
export function formatAcpCost(usd: number): string {
  if (usd <= 0) return '$0'
  if (usd < 0.01) return '<$0.01'
  return `$${usd.toFixed(2)}`
}

function trailingSegment(path: string): string {
  const parts = path.split(/[\\/]+/).filter(Boolean)
  return parts.length ? (parts[parts.length - 1] as string) : path
}

function objectFrom(value: unknown): Record<string, unknown> | null {
  if (value && typeof value === 'object' && !Array.isArray(value)) {
    return value as Record<string, unknown>
  }
  if (typeof value !== 'string') return null
  const trimmed = value.trim()
  if (!trimmed.startsWith('{')) return null
  try {
    const parsed = JSON.parse(trimmed) as unknown
    return parsed && typeof parsed === 'object' && !Array.isArray(parsed)
      ? (parsed as Record<string, unknown>)
      : null
  } catch {
    return null
  }
}

/**
 * Read a `sessions_spawn` call: its arguments and its result JSON. A spawn is
 * a harness spawn when either says `runtime: "acp"`; the child key comes from
 * the result (`session_key`). Anything else → null (the generic tool row).
 */
export function parseAcpSpawn(input: unknown, result: unknown): AcpSpawnSpec | null {
  const args = objectFrom(input) ?? {}
  const res = objectFrom(result) ?? {}
  const runtime = text(res.runtime) || text(args.runtime)
  if (runtime.toLowerCase() !== 'acp') return null
  const childKey = text(res.session_key) || text(res.child_session_key)
  if (!childKey) return null
  const agentId = text(res.acp_agent) || text(args.acp_agent) || text(args.agent)
  return {
    childKey,
    agentId,
    agentName: harnessDisplayName(agentId, res.acp_agent_name ?? res.agent_name),
    cwd: text(res.cwd) || text(args.cwd),
    mode: text(res.mode) || text(args.mode),
    harnessMode: text(res.harness_mode) || text(args.harness_mode),
    task: text(args.task) || text(args.message) || text(args.prompt),
  }
}

/* ── Shared DOM pieces ──────────────────────────────────────────────────── */

let idCounter = 0
function nextId(prefix: string): string {
  idCounter += 1
  return `${prefix}-${idCounter}`
}

function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className: string,
  textContent?: string,
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag)
  if (className) node.className = className
  if (textContent !== undefined) node.textContent = textContent
  return node
}

function button(className: string, action: string, label: string): HTMLButtonElement {
  const b = el('button', className, label)
  b.type = 'button'
  b.dataset.action = action
  return b
}

/** The harness badge: a monogram the skins may replace with the brand mark. */
export function buildHarnessBadge(agentId: string, name: string): HTMLSpanElement {
  const badge = el('span', 'acp-badge')
  badge.dataset.acpAgent = agentId || 'unknown'
  badge.setAttribute('aria-hidden', 'true')
  badge.appendChild(el('span', 'acp-badge__mono', harnessMonogram(name || agentId || '?')))
  return badge
}

function chip(kind: string, label: string, title = ''): HTMLSpanElement {
  const c = el('span', 'acp-chip', label)
  c.dataset.kind = kind
  if (title) c.title = title
  return c
}

/** Stat chips (duration, tokens, cost, tool calls); unknown figures are omitted. */
export function buildUsageChips(usage: AcpUsage): HTMLDivElement | null {
  const row = el('div', 'acp-chips')
  const duration = formatAcpDuration(usage.durationMs)
  if (duration) row.appendChild(chip('duration', duration, t('chat.acpChipRuntime')))
  if (usage.totalTokens !== null && usage.totalTokens > 0) {
    const detail =
      usage.inputTokens !== null || usage.outputTokens !== null
        ? t('chat.acpChipTokensDetail', {
            total: usage.totalTokens,
            input: usage.inputTokens ?? 0,
            output: usage.outputTokens ?? 0,
          })
        : t('chat.acpChipTokensTotal', { total: usage.totalTokens })
    row.appendChild(
      chip(
        'tokens',
        t('chat.acpChipTokens', { tokens: formatAcpTokens(usage.totalTokens) }),
        detail,
      ),
    )
  }
  if (usage.costUsd !== null) {
    row.appendChild(
      chip(
        'cost',
        formatAcpCost(usage.costUsd),
        t('chat.acpChipCost', { usd: usage.costUsd.toFixed(4) }),
      ),
    )
  }
  if (usage.toolCalls !== null && usage.toolCalls > 0) {
    row.appendChild(chip('tools', tPlural('chat.acpToolCalls', usage.toolCalls)))
  }
  return row.childElementCount ? row : null
}

/**
 * A `0fr → 1fr` grid disclosure. The clip is inert while closed so its
 * buttons and text stay out of the tab order and the accessibility tree.
 */
function setDisclosure(region: HTMLElement, toggle: HTMLButtonElement, open: boolean): void {
  region.dataset.open = open ? 'true' : 'false'
  toggle.setAttribute('aria-expanded', open ? 'true' : 'false')
  const clip = region.firstElementChild as HTMLElement | null
  if (!clip) return
  if (open) {
    clip.removeAttribute('inert')
    clip.removeAttribute('aria-hidden')
  } else {
    clip.setAttribute('inert', '')
    clip.setAttribute('aria-hidden', 'true')
  }
}

function buildDetails(
  payload: unknown,
  raw: string,
): { toggle: HTMLButtonElement; region: HTMLElement } {
  const id = nextId('acp-details')
  const toggle = button('acp-btn acp-btn--quiet', 'toggle-details', t('chat.acpDetails'))
  toggle.setAttribute('aria-controls', id)
  const region = el('div', 'acp-details')
  region.id = id
  const clip = el('div', 'acp-details__clip')
  const pre = el('pre', 'acp-details__json')
  let body = raw
  if (payload !== null && payload !== undefined) {
    try {
      body = JSON.stringify(payload, null, 2)
    } catch {
      body = raw
    }
  }
  pre.textContent = body
  clip.appendChild(pre)
  region.appendChild(clip)
  setDisclosure(region, toggle, false)
  toggle.addEventListener('click', (event) => {
    event.preventDefault()
    setDisclosure(region, toggle, region.dataset.open !== 'true')
  })
  return { toggle, region }
}

/* ── Completion card ────────────────────────────────────────────────────── */

/** Longer than this (or more lines than CLAMP_LINES) gets the clamp + "Show more". */
const CLAMP_CHARS = 320
const CLAMP_LINES = 4

export interface CompletionCardOptions {
  markdown: MarkdownDep
  openSession?: AcpOpenSession | null
  /** The raw system-row text, shown by "Details" when the payload is not JSON. */
  raw?: string
  /** Fills a name the payload lacks (the mounter knows its spawn's harness). */
  knownAgent?: (childKey: string) => { agentId: string; agentName: string } | null
  /**
   * Figures to show where the payload has none (a cancelled turn reports no
   * usage): the last live values the run card had for this child.
   */
  fallbackUsage?: AcpUsage | null
}

/** The title subject: the harness name for ACP, "Subagent <id>" otherwise. */
export function completionSubject(
  payload: Record<string, unknown>,
  known?: { agentId: string; agentName: string } | null,
): { acp: boolean; agentId: string; name: string } {
  const childKey = text(payload.child_session_key) || text(payload.session_key)
  const runtime = text(payload.runtime).toLowerCase()
  const acp = runtime === 'acp' || (runtime !== 'subagent' && isAcpSessionKey(childKey))
  if (acp) {
    const agentId = text(payload.acp_agent) || known?.agentId || ''
    const name =
      harnessDisplayName(agentId, payload.acp_agent_name) ||
      known?.agentName ||
      t('chat.acpUnknownHarness')
    return { acp, agentId, name }
  }
  const agent = text(payload.agent_id) || (childKey ? agentIdFromSessionKey(childKey) : '')
  return {
    acp,
    agentId: agent,
    name: agent ? t('chat.acpSubagentNamed', { agent }) : t('chat.acpSubagent'),
  }
}

function outcomeTitle(outcome: AcpOutcome, name: string): string {
  if (outcome === 'done') return t('chat.acpTitleFinished', { name })
  if (outcome === 'cancelled') return t('chat.acpTitleCancelled', { name })
  return t('chat.acpTitleFailed', { name })
}

function statusLabel(status: string, outcome: AcpOutcome): string {
  if (outcome === 'done') return t('chat.acpStatusDone')
  if (outcome === 'cancelled') return t('chat.acpStatusCancelled')
  return status === 'timeout' ? t('chat.acpStatusTimedOut') : t('chat.acpStatusFailed')
}

function pill(outcome: AcpRunState, label: string): HTMLSpanElement {
  const p = el('span', 'acp-pill')
  p.dataset.status = outcome
  p.appendChild(el('span', 'acp-pill__icon'))
  p.appendChild(el('span', 'acp-pill__label', label))
  return p
}

function resultText(payload: Record<string, unknown>): { text: string; truncated: boolean } {
  const result = payload.result
  if (typeof result === 'string') return { text: result.trim(), truncated: false }
  if (result && typeof result === 'object') {
    const r = result as { text?: unknown; truncated?: unknown }
    return { text: text(r.text), truncated: r.truncated === true }
  }
  return { text: '', truncated: false }
}

/** Measure once laid out: a reply that fits needs no clamp. jsdom has no layout. */
function unclampIfItFits(region: HTMLElement, toggle: HTMLButtonElement): void {
  if (typeof requestAnimationFrame !== 'function') return
  requestAnimationFrame(() => {
    const clip = region.firstElementChild as HTMLElement | null
    if (!clip || !region.isConnected || clip.scrollHeight === 0) return
    if (clip.scrollHeight <= clip.clientHeight + 1) {
      region.dataset.clamp = 'false'
      toggle.remove()
    }
  })
}

/**
 * Build the completion card for a parsed `subagent_completion` payload. Pure
 * DOM: no polling, no RPC. Live and history rows render the same card.
 */
export function buildCompletionCard(
  payload: Record<string, unknown>,
  options: CompletionCardOptions,
): HTMLElement {
  const childKey = text(payload.child_session_key) || text(payload.session_key)
  const status = text(payload.status).toLowerCase()
  const outcome = completionOutcome(payload.status, payload.terminal_reason)
  const subject = completionSubject(payload, childKey ? options.knownAgent?.(childKey) : null)

  const card = el('section', 'acp-done')
  card.dataset.outcome = outcome
  card.dataset.runtime = subject.acp ? 'acp' : 'subagent'
  card.dataset.acpAgent = subject.agentId || 'unknown'
  if (childKey) card.dataset.childSessionKey = childKey
  const titleId = nextId('acp-done-title')
  card.setAttribute('aria-labelledby', titleId)

  const head = el('div', 'acp-done__head')
  head.appendChild(buildHarnessBadge(subject.agentId, subject.name))
  const titles = el('div', 'acp-done__titles')
  const title = el('div', 'acp-done__title', outcomeTitle(outcome, subject.name))
  title.id = titleId
  titles.appendChild(title)
  const cwd = text(payload.cwd)
  const subParts = [
    cwd ? trailingSegment(cwd) : '',
    text(payload.mode),
    text(payload.harness_mode),
  ].filter(Boolean)
  if (subParts.length) {
    const sub = el('div', 'acp-done__sub', subParts.join(' · '))
    if (cwd) sub.title = cwd
    titles.appendChild(sub)
  }
  head.appendChild(titles)
  head.appendChild(pill(outcome, statusLabel(status, outcome)))
  card.appendChild(head)

  const chips = buildUsageChips(fillUsage(completionUsage(payload), options.fallbackUsage))
  if (chips) card.appendChild(chips)

  const foot = el('div', 'acp-done__foot')
  const { text: reply, truncated } = resultText(payload)
  const failureText = text(payload.terminal_message) || text(payload.error_message)
  if (outcome === 'failed' && failureText) {
    card.appendChild(el('p', 'acp-done__error', failureText))
  } else if (outcome === 'cancelled' && failureText && !reply) {
    // Someone stopped it: a notice, never the error style.
    card.appendChild(el('p', 'acp-done__notice', failureText))
  } else if (reply) {
    const region = el('div', 'acp-done__result')
    const clip = el('div', 'acp-done__result-clip')
    const body = el('div', 'acp-done__result-text msg-text-seg')
    body.innerHTML = options.markdown.render(reply)
    options.markdown.bindCopy(body)
    options.markdown.bindHighlight?.(body)
    clip.appendChild(body)
    region.appendChild(clip)
    card.appendChild(region)
    const long = reply.length > CLAMP_CHARS || reply.split('\n').length > CLAMP_LINES
    region.dataset.clamp = long ? 'true' : 'false'
    if (long) {
      const id = nextId('acp-result')
      region.id = id
      region.dataset.expanded = 'false'
      const toggle = button('acp-btn', 'toggle-result', t('chat.acpShowMore'))
      toggle.setAttribute('aria-controls', id)
      toggle.setAttribute('aria-expanded', 'false')
      toggle.addEventListener('click', (event) => {
        event.preventDefault()
        const expanded = region.dataset.expanded !== 'true'
        region.dataset.expanded = expanded ? 'true' : 'false'
        toggle.setAttribute('aria-expanded', expanded ? 'true' : 'false')
        toggle.textContent = expanded ? t('chat.acpShowLess') : t('chat.acpShowMore')
      })
      foot.appendChild(toggle)
      unclampIfItFits(region, toggle)
    }
    if (truncated) card.appendChild(el('p', 'acp-done__note', t('chat.acpResultTruncated')))
  } else if (outcome === 'done') {
    card.appendChild(el('p', 'acp-done__note', t('chat.acpNoReply')))
  }

  if (childKey && options.openSession) {
    const open = button('acp-btn', 'open-session', t('chat.acpOpenSession'))
    open.title = childKey
    const navigate = options.openSession
    open.addEventListener('click', (event) => {
      event.preventDefault()
      navigate(childKey)
    })
    foot.appendChild(open)
  }
  const details = buildDetails(payload, options.raw ?? '')
  foot.appendChild(details.toggle)
  card.appendChild(foot)
  card.appendChild(details.region)
  return card
}

/* ── Harness run card + the shared poller ───────────────────────────────── */

export interface AcpMounterDeps {
  call: AcpCall
  markdown: MarkdownDep
  /** Read live: the chat may gain or lose a navigator without a remount. */
  openSession?: () => AcpOpenSession | null
  now?: () => number
  pollMs?: number
  diag?: (event: string, detail: Record<string, unknown>) => void
}

interface ChildState {
  spec: AcpSpawnSpec | null
  state: AcpRunState
  /** Wall-clock start of the running turn (ms), for the elapsed clock. */
  startedAt: number | null
  activity: string
  activityKind: string
  toolCalls: number | null
  usage: AcpUsage | null
  /** A poll has seen this child's turn active at least once. */
  seenActive: boolean
  /** Polls in a row that did not see the turn active. */
  quietPolls: number
  /**
   * The harness turn is over (`active_turn` went false) but the outcome is
   * not in yet: the clock stops, Stop goes, the card says it is wrapping up.
   */
  endedAt: number | null
  /** The last `acp.status` failed for a reason a retry can fix (connection). */
  transient: boolean
  statusAttempts: number
  stop: 'idle' | 'pending' | 'sent'
  note: string
  noteTone: '' | 'error'
}

/** Polls without the turn before a running card checks `acp.status` itself. */
const ENDED_GRACE_POLLS = 2
const NEVER_SEEN_GRACE_POLLS = 12
const MAX_BACKOFF_MS = 12_000
/** `acp.status` retries for a failure that is not an answer (no connection yet). */
const STATUS_RETRIES = 4
/** Error codes that are an answer: retrying will not change them. */
const FINAL_STATUS_CODES = new Set(['ACP_SESSION_INVALID', 'ACP_UNAVAILABLE', 'METHOD_NOT_FOUND'])
/** How long a leaving activity line stays in the DOM for its exit animation. */
const ACTIVITY_SWAP_MS = 320

function rowsFrom(result: unknown): AcpLiveRow[] {
  const rows = (result as { sessions?: unknown } | null)?.sessions
  return Array.isArray(rows)
    ? rows.filter((r): r is AcpLiveRow => Boolean(r) && typeof r === 'object')
    : []
}

function activityLabel(row: AcpLiveRow, name: string): string {
  const explicit = text(row.activity)
  if (explicit) return explicit
  if (text(row.activity_kind) === 'text') return t('chat.acpActivityWriting')
  return t('chat.acpActivityWorking', { name })
}

function outcomeFromLastTurn(lastTurn: Record<string, unknown>): AcpOutcome {
  if (text(lastTurn.error) || text(lastTurn.error_code)) {
    const code = text(lastTurn.error_code).toLowerCase()
    return code.includes('cancel') ? 'cancelled' : 'failed'
  }
  const stop = text(lastTurn.stop_reason).toLowerCase()
  if (stop === 'cancelled' || stop === 'canceled') return 'cancelled'
  if (stop === 'error' || stop === 'refusal') return 'failed'
  return 'done'
}

/**
 * The per-transcript owner of every harness card: builds them, remembers each
 * child's last known state across transcript rebuilds (history resyncs
 * rebuild the bubble), and runs the one `acp.sessions` poller.
 *
 * Poller discipline: it runs only while at least one connected card is
 * running and the document is visible; it stops by itself when the last one
 * settles or leaves the DOM, pauses while the window is hidden, backs off on
 * errors, and `destroyAll()` (unmount) stops it for good.
 */
export function createAcpMounter(deps: AcpMounterDeps) {
  const now = deps.now ?? ((): number => Date.now())
  const pollMs = deps.pollMs ?? 1500
  const diag = deps.diag ?? ((): void => {})
  const children = new Map<string, ChildState>()
  const cards = new Map<string, Set<HTMLElement>>()
  /** Cards built in the current task, not inserted yet (see connectedCards). */
  const fresh = new Set<HTMLElement>()
  const completions = new Map<string, Record<string, unknown>>()
  const statusInFlight = new Set<string>()
  const swapTimers = new Set<ReturnType<typeof setTimeout>>()
  const retryTimers = new Set<ReturnType<typeof setTimeout>>()
  let pollTimer: ReturnType<typeof setTimeout> | null = null
  let clockTimer: ReturnType<typeof setInterval> | null = null
  let polling = false
  let failures = 0
  // Bumped by destroyAll (unmount): an RPC answer from before it is dropped.
  // Not a permanent kill — StrictMode unmounts and remounts the same mounter.
  let generation = 0
  let visibilityBound = false

  const hidden = (): boolean => typeof document !== 'undefined' && document.hidden === true

  function childState(key: string): ChildState {
    let state = children.get(key)
    if (!state) {
      state = {
        spec: null,
        state: 'checking',
        startedAt: null,
        activity: '',
        activityKind: '',
        toolCalls: null,
        usage: null,
        seenActive: false,
        quietPolls: 0,
        endedAt: null,
        transient: false,
        statusAttempts: 0,
        stop: 'idle',
        note: '',
        noteTone: '',
      }
      children.set(key, state)
    }
    return state
  }

  /**
   * The cards still on screen for `key`. A card counts from the moment it is
   * built: its caller inserts it right after, in the same task. One that has
   * left the DOM since (a history rebuild, a session switch) is forgotten.
   */
  function connectedCards(key: string): HTMLElement[] {
    const set = cards.get(key)
    if (!set) return []
    for (const card of [...set]) {
      if (!card.isConnected && !fresh.has(card)) set.delete(card)
    }
    if (!set.size) cards.delete(key)
    return [...(cards.get(key) ?? [])]
  }

  function runningKeys(): string[] {
    return [...children.entries()]
      .filter(([key, s]) => s.state === 'running' && connectedCards(key).length > 0)
      .map(([key]) => key)
  }

  /* ── painting ─────────────────────────────────────────────────────── */

  function swapActivity(slot: HTMLElement, label: string, animate: boolean): void {
    const current = slot.querySelector<HTMLElement>('.acp-run__activity-text:not([data-leaving])')
    if (current && current.textContent === label) return
    const next = el('span', 'acp-run__activity-text', label)
    if (current && animate) {
      current.dataset.leaving = 'true'
      next.dataset.entering = 'true'
      slot.appendChild(next)
      const timer = setTimeout(() => {
        swapTimers.delete(timer)
        current.remove()
        delete next.dataset.entering
      }, ACTIVITY_SWAP_MS)
      swapTimers.add(timer)
    } else {
      slot.querySelectorAll('.acp-run__activity-text').forEach((node) => node.remove())
      slot.appendChild(next)
    }
    slot.title = label
  }

  function stateLabel(state: ChildState, name: string): string {
    switch (state.state) {
      case 'running':
        if (state.endedAt !== null) return t('chat.acpRunWrapping')
        return state.activity || t('chat.acpActivityStarting', { name })
      case 'done': {
        const duration = formatAcpDuration(state.usage?.durationMs)
        return duration ? t('chat.acpRunFinishedIn', { duration }) : t('chat.acpRunFinished')
      }
      case 'failed':
        return t('chat.acpRunFailed')
      case 'cancelled':
        return t('chat.acpRunCancelled')
      case 'checking':
        return t('chat.acpRunChecking')
      default:
        return t('chat.acpRunUnknown')
    }
  }

  function paint(card: HTMLElement, key: string, animate: boolean): void {
    const state = childState(key)
    const name = state.spec?.agentName || t('chat.acpUnknownHarness')
    const previous = card.dataset.state
    card.dataset.state = state.state
    if (animate && previous === 'running' && state.state !== 'running') {
      card.dataset.settled = 'true'
    }
    const status = card.querySelector<HTMLElement>('.acp-run__status')
    if (status) {
      status.dataset.state = state.state
      if (state.activityKind) status.dataset.activityKind = state.activityKind
      else delete status.dataset.activityKind
    }
    const slot = card.querySelector<HTMLElement>('.acp-run__activity')
    if (slot) swapActivity(slot, stateLabel(state, name), animate && previous !== undefined)

    const tools = card.querySelector<HTMLElement>('.acp-run__tools')
    const toolCount = state.state === 'running' ? state.toolCalls : (state.usage?.toolCalls ?? null)
    if (tools) {
      tools.textContent = toolCount && toolCount > 0 ? tPlural('chat.acpToolCalls', toolCount) : ''
      tools.hidden = !tools.textContent
    }
    const cost = card.querySelector<HTMLElement>('.acp-run__cost')
    if (cost) {
      const usd = state.state === 'running' ? null : (state.usage?.costUsd ?? null)
      cost.textContent = usd !== null ? formatAcpCost(usd) : ''
      cost.hidden = !cost.textContent
    }
    paintClock(card, state)

    const stop = card.querySelector<HTMLButtonElement>('[data-action="stop"]')
    if (stop) {
      stop.hidden = state.state !== 'running' || state.endedAt !== null
      stop.disabled = state.stop !== 'idle'
      if (state.stop === 'pending') stop.dataset.pending = 'true'
      else delete stop.dataset.pending
      stop.textContent = state.stop === 'sent' ? t('chat.acpStopping') : t('chat.acpStop')
    }
    const note = card.querySelector<HTMLElement>('.acp-run__note')
    if (note) {
      note.textContent = state.note
      note.hidden = !state.note
      if (state.noteTone) note.dataset.tone = state.noteTone
      else delete note.dataset.tone
    }
  }

  function paintClock(card: HTMLElement, state: ChildState): void {
    const clock = card.querySelector<HTMLElement>('.acp-run__elapsed')
    if (!clock) return
    if (state.state === 'running' && state.startedAt !== null) {
      // Wrapping up: the clock stops where the harness did.
      const end = state.endedAt ?? now()
      clock.textContent = formatElapsedClock((end - state.startedAt) / 1000)
      clock.hidden = false
    } else {
      clock.textContent = ''
      clock.hidden = true
    }
  }

  function paintAll(key: string, animate: boolean): void {
    connectedCards(key).forEach((card) => paint(card, key, animate))
  }

  /* ── state transitions ────────────────────────────────────────────── */

  /** What the card saw live: the elapsed time and the tool calls of the turn. */
  function liveUsage(state: ChildState): AcpUsage | null {
    if (state.state !== 'running') return null
    const end = state.endedAt ?? now()
    return {
      ...EMPTY_USAGE,
      durationMs: state.startedAt !== null ? Math.max(0, end - state.startedAt) : null,
      toolCalls: state.toolCalls,
    }
  }

  function settle(key: string, outcome: AcpOutcome, usage: AcpUsage | null, animate = true): void {
    const state = childState(key)
    // A cancelled turn may report no usage: keep the numbers the card had.
    const live = liveUsage(state)
    state.usage = fillUsage(fillUsage(usage ?? { ...EMPTY_USAGE }, live), state.usage)
    state.state = outcome
    state.endedAt = null
    state.stop = 'idle'
    state.note = ''
    state.noteTone = ''
    state.activityKind = ''
    paintAll(key, animate)
    reconcileTimers()
  }

  function markRunning(key: string, row: AcpLiveRow | null): void {
    const state = childState(key)
    const wasRunning = state.state === 'running'
    state.state = 'running'
    state.transient = false
    if (row?.active_turn === true) state.endedAt = null
    if (row) {
      const name = state.spec?.agentName || harnessDisplayName(row.agent_id, row.agent_name)
      state.activity = activityLabel(row, name)
      state.activityKind = text(row.activity_kind)
      const tools = finite(row.turn_tool_calls)
      if (tools !== null) state.toolCalls = tools
      const elapsed = finite(row.turn_elapsed_s)
      if (elapsed !== null) state.startedAt = now() - elapsed * 1000
    }
    if (state.startedAt === null) state.startedAt = now()
    paintAll(key, wasRunning)
    reconcileTimers()
  }

  /** Settle (or revive) one child from `acp.status`; at most one call in flight per child. */
  function checkStatus(key: string): void {
    if (statusInFlight.has(key)) return
    statusInFlight.add(key)
    const gen = generation
    void deps
      .call('acp.status', { session_key: key })
      .then((result) => {
        if (gen !== generation) return
        const state = childState(key)
        state.statusAttempts = 0
        state.transient = false
        if (completions.has(key)) return
        if (state.state !== 'running' && state.state !== 'checking' && state.state !== 'unknown') {
          return
        }
        const row = (result ?? {}) as AcpLiveRow
        if (row.active_turn === true) {
          markRunning(key, row)
          return
        }
        const lastTurn =
          row.last_turn && typeof row.last_turn === 'object'
            ? (row.last_turn as Record<string, unknown>)
            : null
        if (lastTurn) {
          settle(
            key,
            outcomeFromLastTurn(lastTurn),
            usageFromRecord(lastTurn),
            state.state === 'running',
          )
          return
        }
        if (row.closed === true) {
          settle(key, 'cancelled', null, state.state === 'running')
          return
        }
        // A turn that just ended may not have written its record yet: one
        // wrapping up waits for the next check rather than guess.
        if (state.state === 'running' && state.endedAt !== null) return
        // Spawned but not picked up yet (the task runtime queues it): keep
        // waiting while the session row still says it runs.
        if (text(row.session_status) === 'running') {
          markRunning(key, null)
          return
        }
        state.state = 'unknown'
        paintAll(key, false)
        reconcileTimers()
      })
      .catch((error: unknown) => {
        if (gen !== generation) return
        const code = (error as { code?: unknown } | null)?.code
        diag('acp.status.error', {
          key,
          code: typeof code === 'string' ? code : '',
          message: error instanceof Error ? error.message : '',
        })
        const state = childState(key)
        const final = typeof code === 'string' && FINAL_STATUS_CODES.has(code)
        if (!final && state.statusAttempts < STATUS_RETRIES) {
          // Not an answer (the socket was not up yet, a dropped frame): try
          // again shortly, a bounded number of times.
          const delay = 1000 * 2 ** state.statusAttempts
          state.statusAttempts += 1
          const timer = setTimeout(() => {
            retryTimers.delete(timer)
            if (gen === generation) checkStatus(key)
          }, delay)
          retryTimers.add(timer)
          return
        }
        state.statusAttempts = 0
        state.transient = !final
        if (state.state === 'checking') {
          state.state = 'unknown'
          paintAll(key, false)
        }
      })
      .finally(() => statusInFlight.delete(key))
  }

  /**
   * The connection is (back) up: a card whose status never came for want of
   * one asks again.
   */
  function reconnected(): void {
    children.forEach((state, key) => {
      if (state.state === 'unknown' && state.transient && connectedCards(key).length) {
        checkStatus(key)
      }
    })
    reconcileTimers(true)
  }

  function applyRows(rows: AcpLiveRow[]): void {
    const byKey = new Map<string, AcpLiveRow>()
    rows.forEach((row) => {
      const key = text(row.session_key)
      if (key) byKey.set(key, row)
    })
    runningKeys().forEach((key) => {
      const state = childState(key)
      const row = byKey.get(key)
      if (row && row.active_turn === true) {
        state.seenActive = true
        state.quietPolls = 0
        markRunning(key, row)
        return
      }
      if (state.seenActive && state.endedAt === null) {
        // The harness is done; the outcome is on its way. Stop the clock and
        // the Stop button now rather than at the completion, and ask.
        state.endedAt = now()
        state.quietPolls = 0
        paintAll(key, true)
        checkStatus(key)
        return
      }
      state.quietPolls += 1
      // The completion event normally settles the card first. When it does
      // not come (a turn that does not announce, a dropped frame), ask once.
      const grace = state.seenActive ? ENDED_GRACE_POLLS : NEVER_SEEN_GRACE_POLLS
      if (state.quietPolls >= grace) {
        state.quietPolls = 0
        checkStatus(key)
      }
    })
  }

  /* ── timers ───────────────────────────────────────────────────────── */

  function onVisibility(): void {
    if (hidden()) {
      stopTimers()
      return
    }
    reconcileTimers(true)
  }

  function bindVisibility(): void {
    if (visibilityBound || typeof document === 'undefined') return
    document.addEventListener('visibilitychange', onVisibility)
    visibilityBound = true
  }

  function unbindVisibility(): void {
    if (!visibilityBound) return
    document.removeEventListener('visibilitychange', onVisibility)
    visibilityBound = false
  }

  function stopTimers(): void {
    if (pollTimer !== null) clearTimeout(pollTimer)
    pollTimer = null
    if (clockTimer !== null) clearInterval(clockTimer)
    clockTimer = null
    polling = false
  }

  function schedulePoll(delay: number): void {
    if (pollTimer !== null) clearTimeout(pollTimer)
    pollTimer = setTimeout(() => {
      pollTimer = null
      void poll()
    }, delay)
  }

  async function poll(): Promise<void> {
    if (hidden() || runningKeys().length === 0) {
      reconcileTimers()
      return
    }
    const gen = generation
    try {
      const result = await deps.call('acp.sessions', {})
      if (gen !== generation) return
      failures = 0
      applyRows(rowsFrom(result))
    } catch (error: unknown) {
      if (gen !== generation) return
      failures += 1
      diag('acp.sessions.error', {
        failures,
        message: error instanceof Error ? error.message : '',
      })
    }
    if (!polling) return
    if (runningKeys().length === 0 || hidden()) {
      reconcileTimers()
      return
    }
    schedulePoll(Math.min(MAX_BACKOFF_MS, pollMs * 2 ** Math.min(failures, 3)))
  }

  /** Start the poller + clock when a visible card runs; stop both when none does. */
  function reconcileTimers(immediate = false): void {
    const running = runningKeys().length > 0
    if (!running) {
      stopTimers()
      unbindVisibility()
      return
    }
    bindVisibility()
    if (hidden()) {
      stopTimers()
      return
    }
    if (clockTimer === null) {
      clockTimer = setInterval(() => {
        runningKeys().forEach((key) => {
          const state = childState(key)
          connectedCards(key).forEach((card) => paintClock(card, state))
        })
        if (runningKeys().length === 0) reconcileTimers()
      }, 1000)
    }
    if (!polling) {
      polling = true
      schedulePoll(immediate ? 0 : pollMs)
    }
  }

  /* ── cards ────────────────────────────────────────────────────────── */

  function stopRun(key: string): void {
    const state = childState(key)
    if (state.state !== 'running' || state.stop !== 'idle') return
    state.stop = 'pending'
    state.note = ''
    state.noteTone = ''
    paintAll(key, true)
    const gen = generation
    void deps
      .call('acp.cancel', { session_key: key })
      .then((result) => {
        if (gen !== generation) return
        const s = childState(key)
        const cancelled = (result as { cancelled?: unknown } | null)?.cancelled !== false
        if (s.state !== 'running') return
        if (cancelled) {
          s.stop = 'sent'
          s.note = ''
        } else {
          s.stop = 'idle'
          s.note = t('chat.acpStopNothing')
          s.noteTone = ''
          checkStatus(key)
        }
        paintAll(key, true)
      })
      .catch((error: unknown) => {
        if (gen !== generation) return
        const s = childState(key)
        s.stop = 'idle'
        s.note = t('chat.acpStopFailed', {
          message: error instanceof Error ? error.message : String(error),
        })
        s.noteTone = 'error'
        paintAll(key, true)
      })
  }

  function buildHarnessShell(spec: AcpSpawnSpec, toolId: string): HTMLElement {
    const card = el('section', 'acp-run')
    card.dataset.acpAgent = spec.agentId || 'unknown'
    card.dataset.childSessionKey = spec.childKey
    if (toolId) card.dataset.toolId = toolId
    card.setAttribute('data-tool-name', 'sessions_spawn')
    const titleId = nextId('acp-run-title')
    card.setAttribute('aria-labelledby', titleId)

    const head = el('div', 'acp-run__head')
    head.appendChild(buildHarnessBadge(spec.agentId, spec.agentName))
    const titles = el('div', 'acp-run__titles')
    const name = el('div', 'acp-run__name', spec.agentName || t('chat.acpUnknownHarness'))
    name.id = titleId
    titles.appendChild(name)
    if (spec.cwd) {
      const cwd = el('div', 'acp-run__cwd', trailingSegment(spec.cwd))
      cwd.title = spec.cwd
      titles.appendChild(cwd)
    }
    head.appendChild(titles)
    const modes = el('div', 'acp-chips acp-run__modes')
    if (spec.mode) modes.appendChild(chip('mode', spec.mode, t('chat.acpChipMode')))
    if (spec.harnessMode) {
      modes.appendChild(chip('harness-mode', spec.harnessMode, t('chat.acpChipHarnessMode')))
    }
    if (modes.childElementCount) head.appendChild(modes)
    card.appendChild(head)

    if (spec.task) {
      const line = spec.task.replace(/\s+/g, ' ').trim()
      const task = el('p', 'acp-run__task', line)
      task.title = spec.task.length > 600 ? `${spec.task.slice(0, 600)}…` : spec.task
      card.appendChild(task)
    }

    const status = el('div', 'acp-run__status')
    status.setAttribute('role', 'status')
    status.setAttribute('aria-live', 'polite')
    const indicator = el('span', 'acp-run__indicator')
    indicator.setAttribute('aria-hidden', 'true')
    status.appendChild(indicator)
    status.appendChild(el('span', 'acp-run__activity'))
    const meta = el('span', 'acp-run__meta')
    const tools = el('span', 'acp-run__tools')
    tools.hidden = true
    const cost = el('span', 'acp-run__cost')
    cost.hidden = true
    const elapsed = el('span', 'acp-run__elapsed')
    elapsed.hidden = true
    elapsed.setAttribute('aria-hidden', 'true')
    meta.append(tools, cost, elapsed)
    status.appendChild(meta)
    card.appendChild(status)

    const note = el('p', 'acp-run__note')
    note.hidden = true
    card.appendChild(note)

    const actions = el('div', 'acp-run__actions')
    const navigate = deps.openSession?.() ?? null
    if (navigate) {
      const open = button('acp-btn', 'open-session', t('chat.acpOpenSession'))
      open.title = spec.childKey
      open.addEventListener('click', (event) => {
        event.preventDefault()
        // Read again: the navigator is live, the card may be older.
        ;(deps.openSession?.() ?? navigate)(spec.childKey)
      })
      actions.appendChild(open)
    }
    const stop = button('acp-btn acp-btn--danger', 'stop', t('chat.acpStop'))
    stop.hidden = true
    stop.addEventListener('click', (event) => {
      event.preventDefault()
      stopRun(spec.childKey)
    })
    actions.appendChild(stop)
    card.appendChild(actions)
    return card
  }

  /**
   * The harness card for a spawn. `live`: the spawn's tool result just
   * arrived on the stream, so the child is genuinely starting and the card
   * enters animated. Otherwise (history) the card shows what is known about
   * the child: its remembered state after a resync, its completion if one is
   * rendered later in the same pass, or one `acp.status` answer — never a
   * spinner it cannot vouch for.
   */
  function harnessCard(
    spec: AcpSpawnSpec,
    options: { live: boolean; toolId?: string },
  ): HTMLElement {
    const key = spec.childKey
    const known = children.has(key)
    const state = childState(key)
    state.spec = { ...spec, agentName: spec.agentName || state.spec?.agentName || '' }
    const card = buildHarnessShell(state.spec, options.toolId ?? '')
    const set = cards.get(key) ?? new Set<HTMLElement>()
    set.add(card)
    cards.set(key, set)
    fresh.add(card)
    void Promise.resolve().then(() => fresh.delete(card))

    const completion = completions.get(key)
    if (completion && (state.state === 'checking' || state.state === 'running')) {
      state.state = completionOutcome(completion.status, completion.terminal_reason)
      state.usage = completionUsage(completion)
    } else if (options.live && !known) {
      state.state = 'running'
      state.startedAt = now()
    }
    if (options.live && !known) card.dataset.enter = 'live'
    paint(card, key, false)

    if (state.state === 'checking') {
      // History renders synchronously; a completion later in this pass lands
      // before the timeout fires and settles the card without a status call.
      const gen = generation
      setTimeout(() => {
        if (gen === generation && childState(key).state === 'checking') checkStatus(key)
      }, 0)
    }
    reconcileTimers()
    return card
  }

  /** A completion row for `child_session_key`: settle its harness cards. */
  function noteCompletion(payload: Record<string, unknown>): void {
    const key = text(payload.child_session_key) || text(payload.session_key)
    if (!key) return
    completions.set(key, payload)
    const state = children.get(key)
    if (!state) return
    // The completion is the authority, also over an outcome read from
    // `acp.status` a moment earlier; only a live card animates the settle.
    settle(
      key,
      completionOutcome(payload.status, payload.terminal_reason),
      completionUsage(payload),
      state.state === 'running',
    )
  }

  /** The completion card, registered so the spawn's harness card settles too. */
  function completionCard(payload: Record<string, unknown>, raw: string): HTMLElement {
    noteCompletion(payload)
    // Offered when a navigator exists now; the click reads the live one.
    const navigator = deps.openSession?.() ?? null
    const childKey = text(payload.child_session_key) || text(payload.session_key)
    return buildCompletionCard(payload, {
      markdown: deps.markdown,
      openSession: navigator ? (key) => (deps.openSession?.() ?? navigator)(key) : null,
      fallbackUsage: childKey ? (children.get(childKey)?.usage ?? null) : null,
      raw,
      knownAgent: (childKey) => {
        const spec = children.get(childKey)?.spec
        return spec ? { agentId: spec.agentId, agentName: spec.agentName } : null
      },
    })
  }

  function destroyAll(): void {
    generation += 1
    stopTimers()
    unbindVisibility()
    swapTimers.forEach((timer) => clearTimeout(timer))
    swapTimers.clear()
    retryTimers.forEach((timer) => clearTimeout(timer))
    retryTimers.clear()
    cards.clear()
    fresh.clear()
  }

  return {
    harnessCard,
    completionCard,
    noteCompletion,
    reconnected,
    destroyAll,
    /** Test/diagnostic surface: is the poller armed right now. */
    isPolling: (): boolean => polling,
  }
}

export type AcpMounter = ReturnType<typeof createAcpMounter>
