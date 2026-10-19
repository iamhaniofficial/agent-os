import { act, renderHook, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import type { ReactNode } from 'react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { toast } from 'sonner'
import { useSlashCommands } from './useSlashCommands'

// `/restart [force]` asks the gateway to restart itself in place over
// `gateway.restart`. The socket then drops and reconnects on its own; the
// "back" toast must fire on that reconnect, not on the connection that was
// already up when the command was sent.

vi.mock('sonner', () => ({
  toast: { success: vi.fn(), warning: vi.fn(), error: vi.fn(), info: vi.fn() },
}))

const KEY = 'agent:main:webchat:restart-test'
const CATALOG = [
  {
    name: '/restart',
    usage: '/restart [force]',
    description: 'Restart the gateway',
    aliases: [],
    execution: { action: 'gateway.restart' },
  },
]

type StateHandler = (state: unknown) => void
let restartResponse: unknown = { accepted: true, restartId: 'r1', mode: 'safe' }
let failRestart = false
let stateHandlers: StateHandler[] = []

function makeRpc() {
  return {
    waitForConnection: vi.fn().mockResolvedValue(undefined),
    call: vi.fn(async (method: string) => {
      if (method === 'commands.list_for_surface') return { surface: 'web_chat', commands: CATALOG }
      if (method === 'gateway.restart') {
        if (failRestart) throw new Error('Gateway restart from chat is disabled')
        return restartResponse
      }
      return {}
    }),
    on: vi.fn((event: string, handler: StateHandler): (() => void) => {
      if (event === '_state') stateHandlers.push(handler)
      return () => {
        stateHandlers = stateHandlers.filter((h) => h !== handler)
      }
    }),
  }
}
let mockRpc = makeRpc()
vi.mock('@/app/providers', () => ({ useRpc: () => mockRpc }))

function setup() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const wrapper = ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  )
  return renderHook(() => useSlashCommands({ sessionKey: KEY }), { wrapper })
}

async function run(result: ReturnType<typeof setup>['result'], text: string) {
  await waitFor(() => expect(result.current.commands.length).toBe(1))
  await act(async () => {
    expect(await result.current.execute(text)).toBe(true)
  })
}

beforeEach(() => {
  mockRpc = makeRpc()
  stateHandlers = []
  restartResponse = { accepted: true, restartId: 'r1', mode: 'safe' }
  failRestart = false
  vi.mocked(toast.info).mockClear()
  vi.mocked(toast.success).mockClear()
  vi.mocked(toast.error).mockClear()
})

describe('/restart', () => {
  it('calls gateway.restart in safe mode and toasts once the socket is back', async () => {
    const { result } = setup()
    await run(result, '/restart')

    await waitFor(() =>
      expect(mockRpc.call).toHaveBeenCalledWith('gateway.restart', {
        key: KEY,
        mode: 'safe',
        reason: 'slash_command',
      }),
    )
    await waitFor(() =>
      expect(toast.info).toHaveBeenCalledWith(
        'Restarting the gateway… draining active turns first',
      ),
    )
    await waitFor(() => expect(stateHandlers).toHaveLength(1))
    const handler = stateHandlers[0]!

    // The connection that was up when we sent the command is not "back".
    act(() => handler('connected'))
    expect(toast.success).not.toHaveBeenCalled()

    act(() => {
      handler('disconnected')
      handler('connecting')
      handler('connected')
    })
    expect(toast.success).toHaveBeenCalledTimes(1)
    expect(toast.success).toHaveBeenCalledWith('Gateway is back')
    // One toast only: the watcher unsubscribed itself.
    expect(stateHandlers).toHaveLength(0)
    act(() => {
      handler('disconnected')
      handler('connected')
    })
    expect(toast.success).toHaveBeenCalledTimes(1)
  })

  it('/restart force passes mode: force', async () => {
    restartResponse = { accepted: true, restartId: 'r2', mode: 'force' }
    const { result } = setup()
    await run(result, '/restart force')

    await waitFor(() =>
      expect(mockRpc.call).toHaveBeenCalledWith('gateway.restart', {
        key: KEY,
        mode: 'force',
        reason: 'slash_command',
      }),
    )
    await waitFor(() =>
      expect(toast.info).toHaveBeenCalledWith('Restarting the gateway now (force)'),
    )
  })

  it('reports a restart that is already in progress without watching again', async () => {
    restartResponse = { accepted: false, alreadyPending: true, restartId: 'first' }
    const { result } = setup()
    await run(result, '/restart')

    await waitFor(() =>
      expect(toast.info).toHaveBeenCalledWith('A gateway restart is already in progress'),
    )
    expect(stateHandlers).toHaveLength(0)
  })

  it('surfaces a refusal as an error toast', async () => {
    failRestart = true
    const { result } = setup()
    await run(result, '/restart')

    await waitFor(() =>
      expect(toast.error).toHaveBeenCalledWith(
        'Restart failed: Gateway restart from chat is disabled',
      ),
    )
    expect(toast.info).not.toHaveBeenCalled()
    expect(stateHandlers).toHaveLength(0)
  })
})
