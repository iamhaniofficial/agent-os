import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { Composer } from './Composer'

// The suggested next prompt (`chat.suggestNext`): ghost text in the empty
// composer that Tab takes and Enter sends — same contract as the console.

const textbox = () => screen.getByRole('textbox') as HTMLTextAreaElement
const tab = (init: Partial<KeyboardEventInit> = {}) =>
  fireEvent.keyDown(textbox(), { key: 'Tab', ...init })

describe('Composer · suggested next prompt', () => {
  it('shows the suggestion as the placeholder while the input is empty', () => {
    render(<Composer onSend={vi.fn()} busy={false} suggestion="Now add tests" />)
    expect(textbox()).toHaveAttribute('placeholder', 'Now add tests')
    expect(textbox()).toHaveAttribute('aria-keyshortcuts', 'Tab')
    expect(screen.getByTestId('composer-suggestion-key')).toHaveTextContent('Tab')

    fireEvent.change(textbox(), { target: { value: 'my own words' } })
    expect(screen.queryByTestId('composer-suggestion-key')).toBeNull()
  })

  it('outranks the desk’s rotating placeholder, which returns without one', () => {
    const { rerender } = render(
      <Composer onSend={vi.fn()} busy={false} placeholder="Buy 1 NVDA" suggestion="Ship it" />,
    )
    expect(textbox()).toHaveAttribute('placeholder', 'Ship it')

    rerender(<Composer onSend={vi.fn()} busy={false} placeholder="Buy 1 NVDA" suggestion={null} />)
    expect(textbox()).toHaveAttribute('placeholder', 'Buy 1 NVDA')
    expect(textbox()).not.toHaveAttribute('aria-keyshortcuts')
  })

  it('Tab fills the suggestion, then Enter sends it', () => {
    const onSend = vi.fn()
    render(<Composer onSend={onSend} busy={false} suggestion="Now add tests" />)

    expect(tab()).toBe(false)
    expect(textbox().value).toBe('Now add tests')

    fireEvent.keyDown(textbox(), { key: 'Enter' })
    expect(onSend).toHaveBeenCalledWith('Now add tests')
  })

  it('leaves Tab to the browser over a draft, without a suggestion, or with modifiers', () => {
    const { rerender } = render(<Composer onSend={vi.fn()} busy={false} suggestion="Ship it" />)
    expect(tab({ shiftKey: true })).toBe(true)
    expect(tab({ metaKey: true })).toBe(true)

    fireEvent.change(textbox(), { target: { value: 'draft' } })
    expect(tab()).toBe(true)
    expect(textbox().value).toBe('draft')

    rerender(<Composer onSend={vi.fn()} busy={false} suggestion={null} />)
    fireEvent.change(textbox(), { target: { value: '' } })
    expect(tab()).toBe(true)
    expect(textbox().value).toBe('')
  })
})
