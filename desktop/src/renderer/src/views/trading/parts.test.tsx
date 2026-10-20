import { fireEvent, render, screen } from '@testing-library/react'
import { useState } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { t } from '~/i18n'
import { AssetCell, PanelBoundary } from './parts'
import { USDC } from './test-utils'

let fail = true
function Flaky() {
  if (fail) throw new Error('boom: unknown kind')
  return <p>panel body</p>
}

function Host() {
  const [n, setN] = useState(0)
  return (
    <div>
      <button type="button" onClick={() => setN(n + 1)}>
        sibling {n}
      </button>
      <PanelBoundary name="history">
        <Flaky />
      </PanelBoundary>
    </div>
  )
}

describe('PanelBoundary', () => {
  beforeEach(() => {
    fail = true
    // React logs the caught error; keep the run quiet.
    vi.spyOn(console, 'error').mockImplementation(() => {})
  })
  afterEach(() => vi.restoreAllMocks())

  it('contains a throwing child: says so, names the error, keeps the rest alive', () => {
    render(<Host />)
    const alert = screen.getByTestId('panel-error')
    expect(alert).toHaveTextContent(t('trading.panel.failed'))
    expect(alert).toHaveTextContent('boom: unknown kind')
    // The rest of the window still works.
    fireEvent.click(screen.getByText('sibling 0'))
    expect(screen.getByText('sibling 1')).toBeInTheDocument()
  })

  it('retries the panel', () => {
    render(<Host />)
    fail = false
    fireEvent.click(screen.getByTestId('panel-error-retry'))
    expect(screen.queryByTestId('panel-error')).toBeNull()
    expect(screen.getByText('panel body')).toBeInTheDocument()
  })

  it('has its copy in the catalogue', () => {
    expect(t('trading.panel.failed')).toBe('This panel failed to render')
  })
})

describe('AssetCell · chain', () => {
  it('shows the network as its mark alone, named on hover and for readers', () => {
    render(<AssetCell token={USDC} showChain />)
    const mark = document.querySelector('.trd-asset__chainmark')
    expect(mark?.getAttribute('title')).toBe('Base')
    expect(screen.getByRole('img', { name: 'Base' })).toBeTruthy()
    // The ticker line carries no network name to wrap on.
    expect(document.querySelector('.trd-asset__symbol')?.textContent).toBe('USDC')
  })
})
