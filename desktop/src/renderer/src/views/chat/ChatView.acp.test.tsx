import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, fireEvent, render, renderHook, screen, waitFor, within } from '@testing-library/react'
import { createRef } from 'react'
import { MemoryRouter, Route, Routes } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { useConnection } from '@/stores/connection'
import { useGateway } from '~/stores/gateway'
import { ChatView } from './ChatView'
import { ACP_STATUS_POLL_MS, useAcpSession } from './useAcpSession'

// A coding harness's own chat (defects 1 and 5): the session bar over the
// transcript, its Stop / End session, and a composer that says the session
// was ended instead of swallowing the message.
const KEY = 'agent:main:acp:cfb95897'
let status: Record<string, unknown> | Error = {}
type Handler = (...args: unknown[]) => void
const handlers = new Map<string, Set<Handler>>()
function emit(event: string, ...args: unknown[]) {
  handlers.get(event)?.forEach((h) => h(...args))
}
const rpc = {
  waitForConnection: vi.fn(async () => {}),
  call: vi.fn(async (method: string): Promise<unknown> => {
    if (method === 'acp.status') {
      if (status instanceof Error) throw status
      return status
    }
    if (method === 'acp.cancel') return { session_key: KEY, cancelled: true }
    if (method === 'acp.close') return { session_key: KEY, closed: true }
    if (method === 'commands.list_for_surface') return { commands: [] }
    return {}
  }),
  on: vi.fn((event: string, handler: Handler) => {
    if (!handlers.has(event)) handlers.set(event, new Set())
    handlers.get(event)!.add(handler)
    return () => handlers.get(event)?.delete(handler)
  }),
}
vi.mock('@/app/providers', () => ({ useRpc: () => rpc }))
// A stand-in route picker with a real button, to see the closed composer
// switch it off along with its own controls.
vi.mock('@/views/chat/RoutePicker', async () => {
  const { createElement } = await import('react')
  return {
    RoutePicker: () =>
      createElement('button', { type: 'button', className: 'composer-route' }, 'Route'),
  }
})

// The transcript's own run state, steered per test.
const transcript = { busy: false, status: 'idle' }
vi.mock('@/views/chat/useTranscript', () => ({
  useTranscript: () => ({
    containerRef: createRef(),
    routerFxDockRef: createRef(),
    send: vi.fn(),
    abort: vi.fn(),
    busy: transcript.busy,
    routerFxEnabled: false,
    setRouterFxEnabled: vi.fn(),
    history: [],
    runState: { status: transcript.status, label: '' },
    pinnedToTail: true,
    scrollToTail: vi.fn(),
    isCompactInFlightForCurrentSession: () => false,
    setStreamIdlePausedForApproval: vi.fn(),
    setPendingDelegates: vi.fn(),
  }),
}))

function mount(key = KEY) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <MemoryRouter initialEntries={[`/sessions/${encodeURIComponent(key)}`]}>
      <QueryClientProvider client={client}>
        <Routes>
          <Route path="/sessions/:key?" element={<ChatView />} />
        </Routes>
      </QueryClientProvider>
    </MemoryRouter>,
  )
}

const base = {
  session_key: KEY,
  agent_id: 'claude',
  agent_name: 'Claude Code',
  cwd: '/Users/x/Code/my-app',
  mode: 'oneshot',
  harness_mode: 'plan',
  live: false,
  active_turn: false,
  idle: true,
  closed: false,
}

const acpCalls = (method: string) => rpc.call.mock.calls.filter((c) => c[0] === method)

beforeEach(() => {
  rpc.call.mockClear()
  status = { ...base }
  transcript.busy = false
  transcript.status = 'idle'
  useConnection.getState().setState('connected')
  useGateway.setState({ status: { state: 'running', pid: 1, url: 'http://x', error: null } })
})

afterEach(() => {
  vi.useRealTimers()
  Reflect.deleteProperty(document, 'hidden')
})

describe('ACP session bar', () => {
  it('shows Working with the live step, the clock, and Stop while a turn runs', async () => {
    transcript.busy = true
    transcript.status = 'running'
    status = {
      ...base,
      active_turn: true,
      idle: false,
      activity: 'Read src/app.py',
      activity_kind: 'read',
      turn_tool_calls: 3,
      turn_elapsed_s: 12,
    }
    mount()
    const bar = await screen.findByTestId('acp-bar')
    expect(rpc.call).toHaveBeenCalledWith('acp.status', { session_key: KEY })
    expect(bar).toHaveAttribute('data-state', 'working')
    expect(bar).toHaveAttribute('data-acp-agent', 'claude')
    expect(within(bar).getByText('Claude Code')).toBeInTheDocument()
    expect(within(bar).getByText('my-app')).toHaveAttribute('title', '/Users/x/Code/my-app')
    expect(within(bar).getByText('oneshot')).toBeInTheDocument()
    expect(within(bar).getByText('plan')).toBeInTheDocument()
    const live = within(bar).getByRole('status')
    expect(live).toHaveTextContent('Working')
    expect(live).toHaveTextContent('Read src/app.py')
    expect(live).toHaveTextContent(/0:1[2-3]/)

    const stop = within(bar).getByRole('button', { name: 'Stop' })
    await act(async () => {
      fireEvent.click(stop)
    })
    expect(acpCalls('acp.cancel')).toEqual([['acp.cancel', { session_key: KEY }]])
    expect(within(bar).getByRole('button', { name: 'Stopping…' })).toBeDisabled()
    // A turn in flight: the composer stays open for the next message.
    expect(screen.getByRole('textbox')).not.toBeDisabled()
  })

  it('shows Idle with what the next message does, and ends the session after an inline confirm', async () => {
    mount()
    const bar = await screen.findByTestId('acp-bar')
    expect(bar).toHaveAttribute('data-state', 'idle')
    expect(within(bar).getByRole('status')).toHaveTextContent(
      'IdleYour next message resumes Claude Code',
    )
    expect(within(bar).queryByRole('button', { name: 'Stop' })).toBeNull()

    // End session asks first, inline, with the safe answer focused.
    fireEvent.click(within(bar).getByRole('button', { name: 'End session' }))
    const confirm = within(bar).getByRole('group', { name: 'End session' })
    expect(confirm).toHaveTextContent('End this Claude Code session? It will take no new messages.')
    expect(within(confirm).getByRole('button', { name: 'Keep it' })).toHaveFocus()
    expect(acpCalls('acp.close')).toEqual([])
    fireEvent.click(within(confirm).getByRole('button', { name: 'Keep it' }))
    expect(within(bar).queryByRole('group', { name: 'End session' })).toBeNull()

    fireEvent.click(within(bar).getByRole('button', { name: 'End session' }))
    status = { ...base, closed: true, idle: false }
    await act(async () => {
      fireEvent.click(
        within(within(bar).getByRole('group', { name: 'End session' })).getByRole('button', {
          name: 'End session',
        }),
      )
    })
    expect(acpCalls('acp.close')).toEqual([['acp.close', { session_key: KEY }]])
    await waitFor(() => expect(bar).toHaveAttribute('data-state', 'closed'))
  })

  it('names the last failed turn while idle', async () => {
    status = { ...base, last_turn: { error: 'ACP session was closed', duration_ms: 10 } }
    mount()
    const bar = await screen.findByTestId('acp-bar')
    const detail = bar.querySelector('.acp-bar__detail')
    expect(detail).toHaveTextContent('Last turn failed: ACP session was closed')
    expect(detail).toHaveAttribute('data-tone', 'error')
  })

  it('shows Closed and switches the composer off with the reason', async () => {
    status = { ...base, closed: true, idle: false }
    mount()
    const bar = await screen.findByTestId('acp-bar')
    expect(bar).toHaveAttribute('data-state', 'closed')
    expect(within(bar).getByRole('status')).toHaveTextContent('ClosedThis session has ended')
    expect(within(bar).queryByRole('button')).toBeNull()
    const composer = screen.getByRole('textbox')
    expect(composer).toBeDisabled()
    expect(composer).toHaveAttribute(
      'placeholder',
      'This Claude Code session has ended and takes no new messages.',
    )
    expect(composer.closest('form')).toHaveAttribute('data-disabled', 'true')
    expect(screen.getByRole('button', { name: 'Send' })).toBeDisabled()
    // The whole composer is off, not just Send: attach, run modes, the route.
    expect(screen.getByRole('button', { name: 'Attach files' })).toBeDisabled()
    expect(screen.getByRole('button', { name: 'Run modes' })).toBeDisabled()
    expect(screen.getByRole('button', { name: 'Route' })).toBeDisabled()
    // ...and says why, in a line of its own.
    expect(screen.getByRole('note')).toHaveTextContent(
      'This Claude Code session has ended and takes no new messages.',
    )
  })

  it('keeps the bar after an app reload, when the first read finds no connection', async () => {
    status = Object.assign(new Error('Not connected'))
    mount()
    await waitFor(() => expect(acpCalls('acp.status')).toHaveLength(1))
    expect(screen.queryByTestId('acp-bar')).toBeNull()
    // The socket comes up: the bar reads again and appears, composer off.
    status = { ...base, closed: true, idle: false }
    await act(async () => {
      emit('_state', 'connected')
    })
    const bar = await screen.findByTestId('acp-bar')
    expect(bar).toHaveAttribute('data-state', 'closed')
    expect(screen.getByRole('textbox')).toBeDisabled()
  })

  it('draws no bar for an ordinary chat, and never asks', async () => {
    mount('agent:main:webchat:plain')
    await waitFor(() => expect(screen.getByRole('textbox')).toBeInTheDocument())
    expect(screen.queryByTestId('acp-bar')).toBeNull()
    expect(acpCalls('acp.status')).toEqual([])
  })

  it('draws no bar when the gateway says the key is not a harness session', async () => {
    status = Object.assign(new Error('not an ACP session'), { code: 'ACP_SESSION_INVALID' })
    mount()
    await waitFor(() => expect(acpCalls('acp.status')).toHaveLength(1))
    expect(screen.queryByTestId('acp-bar')).toBeNull()
    expect(screen.getByRole('textbox')).not.toBeDisabled()
  })
})

describe('useAcpSession polling', () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })

  it('reads once at rest, polls only while a turn runs and the window is visible', async () => {
    const { rerender, unmount } = renderHook(({ running }) => useAcpSession(KEY, running), {
      initialProps: { running: false },
    })
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0)
    })
    expect(acpCalls('acp.status')).toHaveLength(1)
    await act(async () => {
      await vi.advanceTimersByTimeAsync(ACP_STATUS_POLL_MS * 4)
    })
    expect(acpCalls('acp.status')).toHaveLength(1)

    // A turn starts: read at once, then every 1.5 s.
    rerender({ running: true })
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0)
    })
    expect(acpCalls('acp.status')).toHaveLength(2)
    await act(async () => {
      await vi.advanceTimersByTimeAsync(ACP_STATUS_POLL_MS * 2)
    })
    expect(acpCalls('acp.status')).toHaveLength(4)

    // Hidden: nothing. Back in front: a read straight away, then the beat.
    Object.defineProperty(document, 'hidden', { configurable: true, get: () => true })
    act(() => {
      document.dispatchEvent(new Event('visibilitychange'))
    })
    await act(async () => {
      await vi.advanceTimersByTimeAsync(ACP_STATUS_POLL_MS * 4)
    })
    expect(acpCalls('acp.status')).toHaveLength(4)
    Object.defineProperty(document, 'hidden', { configurable: true, get: () => false })
    act(() => {
      document.dispatchEvent(new Event('visibilitychange'))
    })
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0)
    })
    expect(acpCalls('acp.status')).toHaveLength(5)

    // The turn ends: one more read for the final state, then quiet.
    rerender({ running: false })
    await act(async () => {
      await vi.advanceTimersByTimeAsync(ACP_STATUS_POLL_MS * 4)
    })
    expect(acpCalls('acp.status')).toHaveLength(6)

    // Unmount while running: no more reads.
    rerender({ running: true })
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0)
    })
    const before = acpCalls('acp.status').length
    unmount()
    await vi.advanceTimersByTimeAsync(ACP_STATUS_POLL_MS * 4)
    expect(acpCalls('acp.status')).toHaveLength(before)
  })
})

describe('useAcpSession state', () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })

  const settle = () =>
    act(async () => {
      await vi.advanceTimersByTimeAsync(0)
    })
  const tick = (ms = ACP_STATUS_POLL_MS) =>
    act(async () => {
      await vi.advanceTimersByTimeAsync(ms)
    })

  it('retries a read that failed for want of a connection, a bounded number of times', async () => {
    status = new Error('Not connected')
    renderHook(() => useAcpSession(KEY, false))
    await settle()
    expect(acpCalls('acp.status')).toHaveLength(1)
    await tick(1000)
    await tick(2000)
    await tick(4000)
    await tick(8000)
    expect(acpCalls('acp.status')).toHaveLength(5)
    await tick(60_000)
    expect(acpCalls('acp.status')).toHaveLength(5)
  })

  it('does not retry an answer (not a harness session)', async () => {
    status = Object.assign(new Error('nope'), { code: 'ACP_SESSION_INVALID' })
    renderHook(() => useAcpSession(KEY, false))
    await settle()
    await tick(20_000)
    expect(acpCalls('acp.status')).toHaveLength(1)
  })

  it('follows the harness, not the chat: Working ends when active_turn does', async () => {
    status = { ...base }
    const { result, rerender } = renderHook(({ running }) => useAcpSession(KEY, running), {
      initialProps: { running: false },
    })
    await settle()
    expect(result.current.state).toBe('idle')

    // The chat sent a turn; the harness is not up yet: starting, no step.
    rerender({ running: true })
    await settle()
    expect(result.current.state).toBe('working')
    expect(result.current.starting).toBe(true)
    expect(result.current.activity).toBeNull()
    expect(result.current.startedAt).toBeNull()

    status = { ...base, active_turn: true, idle: false, activity: 'Edit a.py', turn_elapsed_s: 5 }
    await tick()
    expect(result.current.state).toBe('working')
    expect(result.current.starting).toBe(false)
    expect(result.current.activity).toBe('Edit a.py')
    expect(result.current.startedAt).not.toBeNull()

    // The harness is done; the chat still persists and announces for a while.
    status = { ...base, active_turn: false, activity: 'Edit a.py' }
    await tick()
    expect(result.current.state).toBe('idle')
    expect(result.current.activity).toBeNull()
    expect(result.current.startedAt).toBeNull()
  })

  it("never shows the previous turn's step when a new turn starts", async () => {
    status = { ...base, activity: 'Read old.py' }
    const { result, rerender } = renderHook(({ running }) => useAcpSession(KEY, running), {
      initialProps: { running: false },
    })
    await settle()
    rerender({ running: true })
    // The harness is running again but still reports the last turn's step.
    status = { ...base, active_turn: true, activity: 'Read old.py' }
    await settle()
    await tick()
    expect(result.current.state).toBe('working')
    expect(result.current.activity).toBeNull()
    // A fresh step replaces it.
    status = { ...base, active_turn: true, activity: 'Run pytest -q' }
    await tick()
    expect(result.current.activity).toBe('Run pytest -q')
  })
})
