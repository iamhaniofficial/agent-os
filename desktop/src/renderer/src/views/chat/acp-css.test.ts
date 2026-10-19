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
