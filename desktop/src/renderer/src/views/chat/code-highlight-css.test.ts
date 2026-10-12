import { readFileSync } from 'node:fs'
import { createRequire } from 'node:module'
import { describe, expect, it } from 'vitest'

const css = readFileSync('src/renderer/src/views/chat/chat.css', 'utf8')
// The shared renderer (frontend/src/views/chat/markdown.ts) imports this sheet.
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

describe('desktop code block highlight CSS contract', () => {
  it('overrides every github-dark token color on the light theme', () => {
    const selectors = coloredTokenSelectors(githubDark)
    expect(selectors.length).toBeGreaterThan(20)
    for (const selector of selectors) {
      expect(css).toContain(`:root[data-theme='light'] .msg .code-block ${selector}`)
    }
  })

  it('draws plain code in the theme foreground, not github-dark gray', () => {
    expect(css).toMatch(
      /\.msg \.code-block code,\n\.msg \.code-block \.hljs \{[^}]*color: var\(--foreground\);/,
    )
  })
})
