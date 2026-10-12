import { act, renderHook, waitFor } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { useNextSuggestion } from './useNextSuggestion'
import type { WsRpcClient } from '@/lib/ws-rpc'

type Pending = {
  method: string
  params: unknown
  resolve: (v: unknown) => void
  reject: (e: unknown) => void
}

/** An rpc whose calls stay pending until the test answers them. */
function fakeRpc() {
  const pending: Pending[] = []
  const rpc = {
    call: vi.fn(
      (method: string, params: unknown) =>
        new Promise((resolve, reject) => pending.push({ method, params, resolve, reject })),
    ),
  }
  return { rpc: rpc as unknown as WsRpcClient, pending }
}

function setup(sessionKey = 'agent:main:webchat:a', busy = false) {
  const { rpc, pending } = fakeRpc()
  const hook = renderHook(
    ({ key, isBusy }: { key: string; isBusy: boolean }) => useNextSuggestion(rpc, key, isBusy),
    { initialProps: { key: sessionKey, isBusy: busy } },
  )
  return { ...hook, pending }
}

describe('useNextSuggestion', () => {
  it('asks once a turn ends and exposes the answer', async () => {
    const { result, rerender, pending } = setup('agent:main:webchat:a', true)
    expect(pending).toHaveLength(0)

    rerender({ key: 'agent:main:webchat:a', isBusy: false })
    expect(pending).toHaveLength(1)
    expect(pending[0]!.method).toBe('chat.suggestNext')
    expect(pending[0]!.params).toEqual({ sessionKey: 'agent:main:webchat:a' })

    await act(async () => pending[0]!.resolve({ text: 'Now add tests' }))
    expect(result.current.suggestion).toBe('Now add tests')
  })

  it('does not ask on mount, only on a busy → idle edge', () => {
    const { pending } = setup('agent:main:webchat:a', false)
    expect(pending).toHaveLength(0)
  })

  it('clears when the next turn starts', async () => {
    const { result, rerender, pending } = setup('agent:main:webchat:a', true)
    rerender({ key: 'agent:main:webchat:a', isBusy: false })
    await act(async () => pending[0]!.resolve({ text: 'Ship it' }))
    expect(result.current.suggestion).toBe('Ship it')

    rerender({ key: 'agent:main:webchat:a', isBusy: true })
    expect(result.current.suggestion).toBeNull()
  })

  it('drops an answer that lands after the session switched', async () => {
    const { result, rerender, pending } = setup('agent:main:webchat:a', true)
    rerender({ key: 'agent:main:webchat:a', isBusy: false })
    rerender({ key: 'agent:main:webchat:b', isBusy: false })

    await act(async () => pending[0]!.resolve({ text: 'stale' }))
    expect(result.current.suggestion).toBeNull()
  })

  it('drops an answer that lands after the next turn started', async () => {
    const { result, rerender, pending } = setup('agent:main:webchat:a', true)
    rerender({ key: 'agent:main:webchat:a', isBusy: false })
    rerender({ key: 'agent:main:webchat:a', isBusy: true })

    await act(async () => pending[0]!.resolve({ text: 'stale' }))
    expect(result.current.suggestion).toBeNull()
  })

  it('swallows an rpc error (an older gateway without the method)', async () => {
    const { result, rerender, pending } = setup('agent:main:webchat:a', true)
    rerender({ key: 'agent:main:webchat:a', isBusy: false })

    await act(async () => pending[0]!.reject(new Error('Unknown method')))
    await waitFor(() => expect(result.current.suggestion).toBeNull())
  })

  it('treats a null or blank text as no suggestion', async () => {
    const { result, rerender, pending } = setup('agent:main:webchat:a', true)
    rerender({ key: 'agent:main:webchat:a', isBusy: false })
    await act(async () => pending[0]!.resolve({ text: '   ' }))
    expect(result.current.suggestion).toBeNull()
  })

  it('clear() hides the current suggestion', async () => {
    const { result, rerender, pending } = setup('agent:main:webchat:a', true)
    rerender({ key: 'agent:main:webchat:a', isBusy: false })
    await act(async () => pending[0]!.resolve({ text: 'Ship it' }))

    act(() => result.current.clear())
    expect(result.current.suggestion).toBeNull()
  })
})
