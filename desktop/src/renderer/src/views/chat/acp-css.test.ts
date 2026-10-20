import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

const css = readFileSync('src/renderer/src/views/chat/acp.css', 'utf8')
const chatView = readFileSync('src/renderer/src/views/chat/ChatView.tsx', 'utf8')
const bar = readFileSync('src/renderer/src/views/chat/AcpSessionBar.tsx', 'utf8')
const bare = css.replace(/\/\*[\s\S]*?\*\//g, '')

/** `name → body` for every @keyframes block. */
function keyframes(): Map<string, string> {
  const out = new Map<string, string>()
  const re = /@keyframes ([\w-]+) \{([\s\S]*?)\n\}/g
  for (const m of bare.matchAll(re)) out.set(m[1]!, m[2]!)
  return out
}

/** The first top-level rule for exactly this selector. */
function rule(selector: string): string {
  const escaped = selector.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  return bare.match(new RegExp(`(?:^|\\n)${escaped} \\{[\\s\\S]*?\\n\\}`))?.[0] ?? ''
}

/** `[below px, body]` of every `@container acp-bar (width < N px)` block. */
function steps(): Array<readonly [number, string]> {
  return [...bare.matchAll(/@container acp-bar \(width < (\d+)px\) \{([\s\S]*?)\n\}/g)].map(
    (m) => [Number(m[1]), m[2]!] as const,
  )
}

// The harness cards and the session bar are the desktop's own skin of the
// console's transcript DOM: the motion has to stay smooth (compositor-only
// properties), spring-shaped, and off for anyone who asked for less motion.
describe('desktop ACP skin', () => {
  it('is loaded by the chat, and never borrows the console stylesheet', () => {
    expect(chatView).toMatch(/import '\.\/acp\.css'/)
    for (const source of [chatView, bar]) {
      expect(source).not.toMatch(/frontend\/src\/views\/chat\/[\w-]+\.css/)
      expect(source).not.toMatch(/@\/views\/chat\/[\w-]+\.css/)
    }
  })

  it('animates only transform and opacity in every keyframe', () => {
    const frames = keyframes()
    expect([...frames.keys()].sort()).toEqual(
      [
        'acp-breathe',
        'acp-card-in',
        'acp-glow',
        'acp-line-in',
        'acp-line-out',
        'acp-pop',
        'acp-ring',
        'acp-shimmer',
      ].sort(),
    )
    for (const [name, body] of frames) {
      const props = [...body.matchAll(/^\s*([a-z-]+):/gm)].map((m) => m[1])
      expect(props.length, name).toBeGreaterThan(0)
      for (const prop of props) expect(['opacity', 'transform'], `${name}: ${prop}`).toContain(prop)
    }
  })

  it('enters cards on a spring: fade, a 6 px rise and 0.98 → 1', () => {
    expect(bare).toMatch(/--acp-spring: linear\(/)
    expect(bare).toMatch(/--acp-pop: linear\(/)
    expect(keyframes().get('acp-card-in')).toMatch(
      /opacity: 0;\s*transform: translateY\(6px\) scale\(0\.98\);/,
    )
    expect(rule(".acp-run[data-enter='live']")).toMatch(
      /animation: acp-card-in \d+ms var\(--acp-spring\) both;/,
    )
    // A completion row takes the spring too, under chat.css's bulk-insert gate.
    expect(
      rule(".chat-thread[data-history-ready='true'] > .msg.subagent:not([data-enter='none'])"),
    ).toMatch(/animation: acp-card-in/)
  })

  it('opens and closes panels on grid rows, 0fr → 1fr', () => {
    expect(rule('.acp-details')).toMatch(/grid-template-rows: 0fr;/)
    expect(rule('.acp-details')).toMatch(/transition: grid-template-rows/)
    expect(rule(".acp-details[data-open='true']")).toMatch(/grid-template-rows: 1fr;/)
    expect(rule(".acp-done__result[data-clamp='true']")).toMatch(/grid-template-rows: 0fr;/)
    expect(rule(".acp-done__result[data-clamp='true'][data-expanded='true']")).toMatch(
      /grid-template-rows: 1fr;/,
    )
    // The clamp height is the clip's floor, so the closed row shows ~4 lines.
    expect(rule(".acp-done__result[data-clamp='true'] > .acp-done__result-clip")).toMatch(
      /min-height: 6\.\dem;/,
    )
  })

  it('moves only while running: the shimmer, the dot, the settle pop', () => {
    expect(rule(".acp-run[data-state='running'] .acp-run__status::before")).toMatch(
      /animation: acp-shimmer/,
    )
    expect(rule(".acp-run[data-state='running'] .acp-run__indicator::after")).toMatch(
      /animation: acp-ring/,
    )
    expect(bare).toMatch(
      /\.acp-run\[data-settled='true'\] \.acp-run__indicator::before,\s*\.acp-run\[data-settled='true'\] \.acp-run__indicator::after \{\s*animation: acp-pop 640ms var\(--acp-pop\) both;/,
    )
    expect(rule(".acp-bar[data-state='working']::before")).toMatch(/animation: acp-shimmer/)
  })

  it('tints each harness: Claude orange, Codex neutral, Gemini blue, others the signal', () => {
    expect(rule('.acp-run,\n.acp-done,\n.acp-bar,\n.acp-badge')).toMatch(
      /--acp-tint: var\(--primary\);/,
    )
    expect(rule("[data-acp-agent='claude']")).toMatch(/--acp-tint: #d97757;/)
    expect(rule("[data-acp-agent='codex']")).toMatch(/--acp-tint: var\(--foreground\);/)
    expect(rule("[data-acp-agent='gemini']")).toMatch(/--acp-tint: #3186ff;/)
  })

  it('stands everything still under prefers-reduced-motion', () => {
    const block = bare.match(/@media \(prefers-reduced-motion: reduce\) \{([\s\S]*?)\n\}/)?.[1]
    expect(block).toBeTruthy()
    for (const selector of ['.acp-run', '.acp-done', '.acp-bar', '.chat-thread > .msg.subagent']) {
      expect(block).toContain(selector)
    }
    expect(block).toMatch(/animation: none !important;/)
    expect(block).toMatch(/transition: none !important;/)
  })
})

// Round 2 (live test, 900 px and 640 px windows with the sidebar open): the
// bar cut the harness name ("Claude C…"), clipped a chip mid-word ("pla")
// and the idle hint ("…resumes Clau…"). These guard the order things give way.
describe('desktop ACP skin at narrow widths', () => {
  it('sizes the bar by its own container and steps down by width, in order', () => {
    expect(rule('.acp-bar-host')).toMatch(/container: acp-bar \/ inline-size;/)
    expect(bar).toMatch(/className="acp-bar-host"/)
    const step = (selector: RegExp) =>
      Math.max(
        ...steps()
          .filter(([, body]) => selector.test(body))
          .map(([width]) => width),
      )
    // Chips first (sooner while a turn runs), then the step and the clock,
    // then End session while working, then the state word.
    const chipsWorking = step(/\.acp-bar\[data-state='working'\] \.acp-bar__chips/)
    const chipsAll = step(/^\s*\.acp-bar__chips \{/m)
    const detail = step(/\.acp-bar__detail,/)
    const end = step(/\.acp-bar__end/)
    const state = step(/\.acp-bar__state/)
    expect(chipsWorking).toBeGreaterThan(chipsAll)
    expect(chipsAll).toBeGreaterThan(detail)
    expect(detail).toBeGreaterThan(end)
    expect(end).toBeGreaterThan(state)
    // The name is never hidden or cut.
    for (const [, body] of steps()) expect(body).not.toMatch(/acp-bar__name/)
    expect(rule('.acp-bar__name')).toMatch(/white-space: nowrap;/)
    expect(rule('.acp-bar__name')).not.toMatch(/overflow: hidden|text-overflow/)
    expect(rule('.acp-bar__id')).toMatch(/flex: none;/)
  })

  // A width budget of the bar, from the stylesheet's own numbers plus the
  // measured text widths (13 px semibold name, 12.5 px state word, 11.5 px
  // buttons). What may never shrink must fit at every width the bar can
  // have: from a 640 px window with the sidebar open (~355 px) up.
  it('fits everything that cannot shrink, at every width from 340 px up', () => {
    const rem = 16
    const px = (re: RegExp, body: string) => Number(body.match(re)?.[1])
    // The bar's gap and side padding, and the tighter pair a narrow bar uses.
    const spacing = (width: number) => {
      let gap = px(/\n {2}gap: (\d+)px;/, rule('.acp-bar'))
      let padding = 20
      for (const [below, body] of steps()) {
        const m = body.match(/\.acp-bar \{\s*gap: (\d+)px;\s*padding-inline: (\d+)px;/)
        if (m && width < below) {
          gap = Number(m[1])
          padding = 2 * Number(m[2])
        }
      }
      return { gap, padding }
    }
    const badge = 28
    const idFloor = px(/min-width: ([\d.]+)rem;/, rule('.acp-bar__id')) * rem
    const names = { Codex: 42, 'Claude Code': 85, 'GitHub Copilot': 100 }
    const words = { working: 55, idle: 25, closed: 45 }
    const stop = 50
    const endButton = 95
    const clock = 32
    const statusLead = 12 + 1 + 8 + 8 // padding, separator, dot, gap
    const activityFloor = (width: number) => {
      let floor = px(/min-width: ([\d.]+)rem;/, rule(".acp-bar__detail[data-kind='activity']"))
      for (const [below, body] of steps()) {
        const m = body.match(
          /\.acp-bar__detail\[data-kind='activity'\] \{\s*min-width: ([\d.]+)rem;/,
        )
        if (m && width < below) floor = Math.min(floor, Number(m[1]))
      }
      return floor * rem
    }
    const hidden = (width: number, selector: RegExp) =>
      steps().some(([below, body]) => width < below && selector.test(body))

    for (const [name, nameWidth] of Object.entries(names)) {
      for (const state of ['working', 'idle', 'closed'] as const) {
        for (let width = 340; width <= 900; width += 5) {
          const { gap, padding } = spacing(width)
          const stateWord = hidden(width, /\.acp-bar__state/) ? 0 : 8 + words[state]
          const detailShown = !hidden(width, /\.acp-bar__detail,/)
          let status = statusLead + stateWord
          if (state === 'working' && detailShown) status += 8 + activityFloor(width) + 8 + clock
          let actions = 0
          if (state === 'working') {
            actions =
              stop +
              (hidden(width, /\.acp-bar\[data-state='working'\] \.acp-bar__end/)
                ? 0
                : 6 + endButton)
          } else if (state === 'idle') {
            actions = endButton
          }
          const floor =
            padding +
            badge +
            gap +
            Math.max(idFloor, nameWidth) +
            gap +
            status +
            (actions ? gap + actions : 0)
          expect(floor, `${name}, ${state}, ${width}px`).toBeLessThanOrEqual(width)
        }
      }
    }
    // The 900 px window of the live test (a ~615 px bar): idle, the hint
    // ("Your next message resumes Claude Code", ~235 px) and the plan chip fit
    // beside each other, so the hint is shown whole, not dropped.
    const { gap, padding } = spacing(615)
    const idle615 =
      padding +
      badge +
      gap +
      85 +
      gap +
      41 +
      gap +
      statusLead +
      8 +
      words.idle +
      235 +
      gap +
      endButton
    expect(idle615).toBeLessThanOrEqual(615)
  })

  it('drops chips whole instead of clipping them mid-word', () => {
    const chips = rule('.acp-run__modes,\n.acp-bar__chips')
    expect(chips).toMatch(/flex-wrap: wrap;/)
    expect(chips).toMatch(/height: 20px;/)
    expect(chips).toMatch(/overflow: hidden;/)
    // They give way only after everything else (a near-zero shrink factor)...
    expect(chips).toMatch(/flex: 0 0\.01 auto;/)
    // ...and the zero-width spacer in front lets even the first chip leave
    // whole: a flex line always keeps its first item, however wide (the
    // live "read-only" with its right edge cut).
    const spacer = rule('.acp-run__modes::before,\n.acp-bar__chips::before')
    expect(spacer).toMatch(/content: '';/)
    expect(spacer).toMatch(/height: 20px;/)
    expect(spacer).toMatch(/margin-right: -5px;/)
    expect(rule('.acp-chips')).toMatch(/gap: 5px;/)
    expect(bare).toMatch(/\n\.acp-chip \{\s*flex-shrink: 0;\s*\}/)
    // The more telling chip comes first, so the mode chip is the one to go.
    expect(bar.indexOf('data-kind="harness-mode"')).toBeLessThan(bar.indexOf('data-kind="mode"'))
  })

  it('lets the folder line ellipsize without widening anything, and the hint leave whole', () => {
    expect(rule('.acp-bar__cwd')).toMatch(/contain: inline-size;[\s\S]*text-overflow: ellipsis;/)
    expect(rule('.acp-run__cwd,\n.acp-done__sub')).toMatch(/contain: inline-size;/)
    expect(rule('.acp-bar__detail')).toMatch(
      /flex-wrap: wrap;[\s\S]*height: 18px;[\s\S]*overflow: hidden;/,
    )
    expect(rule(".acp-bar__detail[data-kind='hint'] > span")).toMatch(/flex-shrink: 0;/)
    // A lone hint wraps away too (the live "…resumes Claude", cut silently).
    expect(rule(".acp-bar__detail[data-kind='hint']::before")).toMatch(/height: 18px;/)
    // The step adds no width of its own; a live step keeps a readable floor.
    expect(rule('.acp-bar__detail')).toMatch(/contain: inline-size;/)
    expect(rule(".acp-bar__detail[data-kind='activity']")).toMatch(/min-width: \d+rem;/)
    // A narrow completion card puts its pill on the next line, whole; the
    // title wraps beside the badge instead of leaving it alone on a line.
    expect(bare).toMatch(/\n\.acp-done__head \{\s*flex-wrap: wrap;/)
    expect(rule('.acp-done__head > .acp-pill')).toMatch(/flex-shrink: 0;/)
    expect(rule('.acp-done__head > .acp-done__titles')).toMatch(/flex: 1 1 \d+rem;/)
    expect(bare).toMatch(/\n\.acp-done__title \{\s*white-space: normal;/)
    // A running step keeps a readable floor or leaves whole ("R 12 tool
    // calls 1:07" at 640 px): the status row is its own size container.
    expect(rule('.acp-run__status')).toMatch(/container: acp-run-status \/ inline-size;/)
    expect(rule(".acp-run[data-state='running'] .acp-run__activity")).toMatch(/min-width: \d+rem;/)
    const hideStep = bare.match(
      /@container acp-run-status \(width < (\d+)px\) \{\s*\.acp-run\[data-state='running'\] \.acp-run__activity \{\s*display: none;/,
    )
    expect(hideStep).not.toBeNull()
    // The floor fits beside the dot (16), two gaps (9) and the widest figures
    // ("120 tool calls" + the clock, ~124) at the step's threshold.
    const floor =
      Number(rule(".acp-run[data-state='running'] .acp-run__activity").match(/([\d.]+)rem/)?.[1]) *
      16
    expect(16 + 9 + floor + 9 + 124).toBeLessThanOrEqual(Number(hideStep?.[1]))
    // A settled run card keeps its outcome whole ("Cancell…" at 640 px):
    // the figures wrap onto a second line instead.
    expect(rule(".acp-run:not([data-state='running']) .acp-run__status")).toMatch(
      /flex-wrap: wrap;[\s\S]*height: auto;/,
    )
    expect(rule(".acp-run:not([data-state='running']) .acp-run__activity")).toMatch(
      /flex: 1 0 auto;/,
    )
  })
})
