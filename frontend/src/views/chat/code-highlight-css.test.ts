import { readFileSync } from 'node:fs'
import { createRequire } from 'node:module'
import { describe, expect, it } from 'vitest'

const css = readFileSync('src/views/chat/chat.css', 'utf8')
const githubDark = readFileSync(
  createRequire(import.meta.url).resolve('highlight.js/styles/github-dark.css'),
  'utf8',
)

// Every token selector github-dark colors, e.g. `.hljs-meta .hljs-string`.
function coloredTokenSelectors(source: string): string[] {
  const selectors: string[] = []
  for (const [, list = '', body = ''] of source.matchAll(/([^{}]+)\{([^}]*)\}/g)) {
    if (!/(^|[\s;])color:/.test(body)) continue
    for (const selector of list.replace(/\/\*[\s\S]*?\*\//g, '').split(',')) {
      const trimmed = selector.trim()
      if (trimmed.startsWith('.hljs-')) selectors.push(trimmed)
    }
  }
  return selectors
}

describe('code block highlight CSS contract', () => {
  it('overrides every github-dark token color on the light theme', () => {
    const selectors = coloredTokenSelectors(githubDark)
    expect(selectors.length).toBeGreaterThan(20)
    for (const selector of selectors) {
      expect(css).toContain(`:root[data-theme='light'] .msg .code-block ${selector}`)
    }
  })

  it('gives the light theme a near-white panel instead of the dark-mode gray', () => {
    expect(css).toMatch(
      /:root\[data-theme='light'\] \.msg \.code-block pre \{\s*background: color-mix\(in srgb, var\(--background\) \d+%, #fff\);/,
    )
  })
})
