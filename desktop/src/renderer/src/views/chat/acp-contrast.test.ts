import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import { PALETTES } from '~/theme/palettes'

// Round 2 (live test): the light "plan" chip read 3.31:1, the dark tool count
// and cost 3.99:1, the dark folder line 4.36:1, all at 10.5 px. Every text in
// the harness skin must reach 4.5:1 in every palette, light and dark. This
// measures the colours the stylesheet actually uses, over the plates it
// actually paints them on.
//
// Measured on the two hand-tuned palettes (Tactical, the brand default, and
// Graphite; theme/palettes.ts). The seed-derived ones (Everforest, Solarized,
// Nord, …) are low-contrast by design — their own foreground does not reach
// 4.5:1 on their raised surfaces — so no text drawn in them can promise it.
const css = readFileSync('src/renderer/src/views/chat/acp.css', 'utf8').replace(
  /\/\*[\s\S]*?\*\//g,
  '',
)

type Rgb = [number, number, number]
const rgb = (hex: string): Rgb => {
  const h = hex.replace('#', '')
  return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16)) as Rgb
}
/** `share` of `a` mixed with `b` (CSS color-mix in srgb). */
const mix = (a: Rgb, b: Rgb, share: number): Rgb =>
  a.map((v, i) => v * share + b[i]! * (1 - share)) as Rgb
const lum = (c: Rgb) => {
  const [r, g, b] = c.map((v) => {
    const s = v / 255
    return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4
  })
  return 0.2126 * r! + 0.7152 * g! + 0.0722 * b!
}
const contrast = (a: Rgb, b: Rgb) => {
  const [hi, lo] = [lum(a), lum(b)].sort((x, y) => y - x)
  return (hi! + 0.05) / (lo! + 0.05)
}

const TINTS = {
  dark: { claude: '#d97757', gemini: '#3186ff' },
  light: { claude: '#c15f3c', gemini: '#1f6feb' },
}

describe('desktop ACP skin contrast', () => {
  it('uses the tinted ink and muted text the numbers below are about', () => {
    expect(css).toMatch(
      /--acp-ink: color-mix\(in srgb, var\(--acp-tint\) 62%, var\(--foreground\)\);/,
    )
    expect(css).toMatch(/\[data-acp-agent='claude'\] \{\s*--acp-tint: #d97757;/)
    expect(css).toMatch(
      /:root\[data-theme='light'\] \[data-acp-agent='claude'\] \{\s*--acp-tint: #c15f3c;/,
    )
    // No running text in the low-contrast `--dim` (a disabled button is exempt).
    const dimText = [...css.matchAll(/([^{}]+)\{[^}]*?\bcolor: var\(--dim\)/g)].map((m) =>
      m[1]!.trim(),
    )
    expect(dimText.every((sel) => /:disabled/.test(sel))).toBe(true)
    // Text drawn in a harness's colour goes through the ink, never the raw
    // tint; the tint itself colours only the brand marks (graphics, 3:1).
    const tinted = [...css.matchAll(/([^{}]+)\{[^}]*?\n {2}color: var\(--acp-tint\);/g)].map((m) =>
      m[1]!.trim(),
    )
    expect(tinted.length).toBeGreaterThan(0)
    expect(tinted.every((sel) => /badge/.test(sel) && !/mono/.test(sel))).toBe(true)
    expect(css).toMatch(/\.acp-badge__mono \{\s*color: var\(--acp-ink\);/)
    expect(css).toMatch(/\.acp-bar__badge \.prov-logo--mono \{\s*color: var\(--acp-ink\);/)
  })

  for (const id of ['tactical', 'graphite'] as const) {
    const palette = PALETTES[id]
    for (const mode of ['dark', 'light'] as const) {
      it(`reads at 4.5:1 or better in ${id} ${mode}`, () => {
        const p = palette[mode]
        const fg = rgb(p.foreground)
        const plate = mix(rgb(p.elevated), rgb(p.background), 0.94)
        const tints = [TINTS[mode].claude, TINTS[mode].gemini, p.primary, p.foreground]
        for (const tint of tints.map(rgb)) {
          // A card with its harness wash, a chip on it, the ink on the chip.
          const card = mix(tint, plate, 0.07)
          const chip = mix(tint, card, 0.1)
          const ink = mix(tint, fg, 0.62)
          expect(contrast(ink, chip), `${tint} ink on its chip`).toBeGreaterThanOrEqual(4.5)
          expect(contrast(rgb(p['muted-foreground']), card)).toBeGreaterThanOrEqual(4.5)
          const status = mix(fg, card, 0.04)
          expect(contrast(rgb(p['muted-foreground']), status)).toBeGreaterThanOrEqual(4.5)
        }
        for (const tone of [p.ok, p.warn, p.danger].map(rgb)) {
          expect(contrast(tone, mix(tone, plate, 0.13)), 'status pill').toBeGreaterThanOrEqual(4.5)
        }
      })
    }
  }
})
