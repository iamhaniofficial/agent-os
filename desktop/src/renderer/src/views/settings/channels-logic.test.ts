import { describe, expect, it } from 'vitest'
import type { ChannelSpec } from '@/views/setup/logic'
import type { MergedChannel } from '@/views/channels/logic'
import {
  buildChannelEntry,
  canRestartChannel,
  configuredEntry,
  draftDirty,
  fieldIsVisible,
  hasGuidedSetup,
  isRevisionConflict,
  seedDraft,
  splitFields,
  statusTone,
} from './channels-logic'

const TELEGRAM: ChannelSpec = {
  type: 'telegram',
  label: 'Telegram',
  fields: [
    { name: 'name', label: 'Channel name', required: true },
    { name: 'agent_id', label: 'Agent id', default: 'main' },
    { name: 'enabled', label: 'Enabled', type: 'bool', default: true },
    { name: 'token', label: 'Bot token', type: 'password', secret: true, required: true },
    {
      name: 'transport_name',
      label: 'Transport',
      type: 'select',
      choices: ['polling', 'webhook'],
      default: 'polling',
    },
    { name: 'webhook_url', label: 'Webhook URL', showWhen: { transport_name: 'webhook' } },
    { name: 'debounce_window_s', label: 'Debounce', type: 'float', default: 2.5, advanced: true },
  ],
}

const STORED = {
  type: 'telegram',
  name: 'tg-main',
  agent_id: 'main',
  enabled: true,
  token: '[redacted]',
  transport_name: 'polling',
  debounce_window_s: 2.5,
}

function channel(patch: Partial<MergedChannel>): MergedChannel {
  return { access: null, ...patch }
}

describe('seedDraft', () => {
  it('lays the stored entry over the catalog defaults, never a secret', () => {
    const draft = seedDraft(TELEGRAM.fields!, { ...STORED, agent_id: 'ops', enabled: false })
    expect(draft.name).toEqual({ value: 'tg-main', checked: false })
    expect(draft.agent_id!.value).toBe('ops')
    expect(draft.enabled).toEqual({ value: '', checked: false })
    expect(draft.token!.value).toBe('')
    expect(draft.transport_name!.value).toBe('polling')
    expect(draft.debounce_window_s!.value).toBe('2.5')
  })

  it('falls back to the catalog defaults for a new channel', () => {
    const draft = seedDraft(TELEGRAM.fields!)
    expect(draft.name!.value).toBe('')
    expect(draft.agent_id!.value).toBe('main')
    expect(draft.enabled!.checked).toBe(true)
  })
})

describe('fieldIsVisible / splitFields', () => {
  it('hides a field whose showWhen does not match, and folds advanced ones', () => {
    const draft = seedDraft(TELEGRAM.fields!)
    expect(fieldIsVisible(TELEGRAM.fields![5]!, draft)).toBe(false)
    const { primary, advanced } = splitFields(TELEGRAM, draft)
    expect(primary.map((f) => f.name)).toEqual([
      'name',
      'agent_id',
      'enabled',
      'token',
      'transport_name',
    ])
    expect(advanced.map((f) => f.name)).toEqual(['debounce_window_s'])
    draft.transport_name = { value: 'webhook', checked: false }
    expect(splitFields(TELEGRAM, draft).primary.map((f) => f.name)).toContain('webhook_url')
  })

  it('matches boolean conditions against the switch', () => {
    const field = { name: 'x', showWhen: { enabled: true } }
    expect(fieldIsVisible(field, { enabled: { value: '', checked: true } })).toBe(true)
    expect(fieldIsVisible(field, { enabled: { value: '', checked: false } })).toBe(false)
  })
})

describe('draftDirty', () => {
  it('is false until a value or a switch changes', () => {
    const saved = seedDraft(TELEGRAM.fields!, STORED)
    expect(draftDirty(saved, { ...saved })).toBe(false)
    expect(draftDirty(saved, { ...saved, name: { value: 'tg-2', checked: false } })).toBe(true)
    expect(draftDirty(saved, { ...saved, enabled: { value: '', checked: false } })).toBe(true)
  })
})

describe('buildChannelEntry', () => {
  it('reports the first missing required field by label', () => {
    const draft = seedDraft(TELEGRAM.fields!)
    expect(buildChannelEntry({ spec: TELEGRAM, draft, channels: [] })).toEqual({
      ok: false,
      error: 'missing',
      field: 'Channel name',
    })
    draft.name = { value: 'tg', checked: false }
    expect(buildChannelEntry({ spec: TELEGRAM, draft, channels: [] })).toEqual({
      ok: false,
      error: 'missing',
      field: 'Bot token',
    })
  })

  it('builds a new entry from the visible fields only, switches as booleans', () => {
    const draft = seedDraft(TELEGRAM.fields!)
    draft.name = { value: ' tg ', checked: false }
    draft.token = { value: '123:abc', checked: false }
    const result = buildChannelEntry({ spec: TELEGRAM, draft, channels: [] })
    expect(result).toEqual({
      ok: true,
      entry: {
        type: 'telegram',
        name: 'tg',
        agent_id: 'main',
        enabled: true,
        token: '123:abc',
        transport_name: 'polling',
        debounce_window_s: '2.5',
      },
    })
  })

  it('keeps the stored secret on an edit when the field is left blank', () => {
    const draft = seedDraft(TELEGRAM.fields!, STORED)
    draft.agent_id = { value: 'ops', checked: false }
    const result = buildChannelEntry({
      spec: TELEGRAM,
      draft,
      editingName: 'tg-main',
      initialEntry: STORED,
      channels: [channel({ name: 'tg-main', type: 'telegram' })],
    })
    expect(result.ok).toBe(true)
    if (!result.ok) return
    expect(result.entry).not.toHaveProperty('token')
    expect(result.entry.agent_id).toBe('ops')
    expect(result.entry.name).toBe('tg-main')
  })

  it('refuses a name another channel already uses', () => {
    const draft = seedDraft(TELEGRAM.fields!)
    draft.name = { value: 'disc', checked: false }
    draft.token = { value: 't', checked: false }
    expect(
      buildChannelEntry({
        spec: TELEGRAM,
        draft,
        channels: [channel({ name: 'disc', type: 'discord' })],
      }),
    ).toEqual({ ok: false, error: 'duplicate', name: 'disc' })
  })
})

describe('list helpers', () => {
  it('tones statuses', () => {
    expect(statusTone('running')).toBe('ok')
    expect(statusTone('connected')).toBe('ok')
    expect(statusTone('dead')).toBe('danger')
    expect(statusTone('restarting')).toBe('warn')
    expect(statusTone('stopped')).toBeUndefined()
    expect(statusTone('disabled')).toBeUndefined()
  })

  it('offers a restart only for an adapter the gateway has loaded', () => {
    expect(canRestartChannel(channel({ status: 'running' }))).toBe(true)
    expect(canRestartChannel(channel({ status: 'dead' }))).toBe(true)
    expect(canRestartChannel(channel({ status: 'stopped' }))).toBe(false)
    expect(canRestartChannel(channel({ status: 'running', enabled: false }))).toBe(false)
    expect(canRestartChannel(channel({ status: 'disabled' }))).toBe(false)
  })

  it('knows which channels the guided form can edit', () => {
    expect(hasGuidedSetup(channel({ type: 'telegram' }), [TELEGRAM])).toBe(true)
    expect(hasGuidedSetup(channel({ type: 'irc' }), [TELEGRAM])).toBe(false)
  })

  it('finds the configured entry by name', () => {
    expect(configuredEntry([STORED], 'tg-main')).toBe(STORED)
    expect(configuredEntry([STORED], 'other')).toBeUndefined()
    expect(configuredEntry(undefined, 'tg-main')).toBeUndefined()
  })

  it('recognises the gateway revision conflict', () => {
    expect(isRevisionConflict(new Error('config revision mismatch'))).toBe(true)
    expect(isRevisionConflict(new Error('boom'))).toBe(false)
  })
})
