import type { ChannelSpec, FieldSpec, ScopedField } from '@/views/setup/logic'
import { readScopedFields, validateScopedRequiredFields } from '@/views/setup/logic'
import type { MergedChannel } from '@/views/channels/logic'
import type { Tone } from './parts'

/* ── The setup form's draft ─────────────────────────────────────────────── */

/** One field as typed: text for inputs and selects, a flag for switches. */
export interface DraftValue {
  value: string
  checked: boolean
}
export type ChannelDraft = Record<string, DraftValue>

export const EMPTY_VALUE: DraftValue = { value: '', checked: false }

export function isSecretField(field: FieldSpec): boolean {
  return Boolean(field.secret || field.type === 'password')
}

/**
 * Seed the form from the catalog defaults laid under the configured entry
 * when one is being edited. A secret is never seeded: the public config only
 * carries a redaction marker, and a blank secret on save means "keep".
 */
export function seedDraft(fields: FieldSpec[], initial?: Record<string, unknown>): ChannelDraft {
  return Object.fromEntries(
    fields.map((field) => {
      const configured = isSecretField(field) ? undefined : initial?.[field.name]
      const seed = configured ?? field.default
      return [
        field.name,
        {
          value: field.type === 'bool' ? '' : String(seed ?? ''),
          checked: field.type === 'bool' ? Boolean(seed) : false,
        },
      ]
    }),
  )
}

/** `showWhen` gates a field on another field's current value. */
export function fieldIsVisible(field: FieldSpec, draft: ChannelDraft): boolean {
  const conditions = field.showWhen
  if (!conditions || Object.keys(conditions).length === 0) return true
  return Object.entries(conditions).every(([name, expected]) => {
    const actual = draft[name]
    return typeof expected === 'boolean'
      ? actual?.checked === expected
      : String(actual?.value ?? '') === String(expected ?? '')
  })
}

/** The visible fields, split into the ones shown first and the folded rest. */
export function splitFields(
  spec: ChannelSpec | undefined,
  draft: ChannelDraft,
): { primary: FieldSpec[]; advanced: FieldSpec[] } {
  const visible = (spec?.fields ?? []).filter((field) => fieldIsVisible(field, draft))
  return {
    primary: visible.filter((field) => !field.advanced),
    advanced: visible.filter((field) => Boolean(field.advanced)),
  }
}

export function draftDirty(saved: ChannelDraft, draft: ChannelDraft): boolean {
  const names = new Set([...Object.keys(saved), ...Object.keys(draft)])
  for (const name of names) {
    const a = saved[name] ?? EMPTY_VALUE
    const b = draft[name] ?? EMPTY_VALUE
    if (a.value !== b.value || a.checked !== b.checked) return true
  }
  return false
}

export type BuildResult =
  | { ok: true; entry: Record<string, unknown> }
  | { ok: false; error: 'missing'; field: string }
  | { ok: false; error: 'duplicate'; name: string }

/**
 * The `onboarding.channel.upsert` entry from the form. Only the visible
 * fields are read (a hidden webhook URL is not sent for a polling bot); a
 * blank secret is omitted and, on an edit, the stored one is kept because
 * the entry is laid over the configured one minus its secrets. Names must
 * be unique across the runtime list, which already includes every
 * configured entry.
 */
export function buildChannelEntry(args: {
  spec: ChannelSpec
  draft: ChannelDraft
  editingName?: string
  initialEntry?: Record<string, unknown>
  channels: readonly MergedChannel[]
}): BuildResult {
  const { spec, draft, editingName, initialEntry, channels } = args
  const visible = (spec.fields ?? []).filter((field) => fieldIsVisible(field, draft))
  const fields: ScopedField[] = visible.map((field) => ({
    name: field.name,
    value: draft[field.name]?.value ?? '',
    checked: draft[field.name]?.checked ?? false,
    type: field.type === 'bool' ? 'checkbox' : isSecretField(field) ? 'password' : 'text',
    secret: isSecretField(field),
    required: Boolean(field.required),
    hidden: false,
    label: field.label,
  }))
  const editing = Boolean(editingName)
  const canKeepSecret = Boolean(
    editing && initialEntry && String(initialEntry.type || '') === spec.type,
  )
  const missing = validateScopedRequiredFields(fields, canKeepSecret)
  if (missing) return { ok: false, error: 'missing', field: missing }

  const secretNames = new Set(
    (spec.fields ?? []).filter((field) => isSecretField(field)).map((field) => field.name),
  )
  const preserved =
    editing && initialEntry
      ? Object.fromEntries(Object.entries(initialEntry).filter(([name]) => !secretNames.has(name)))
      : {}
  const entry: Record<string, unknown> = {
    ...preserved,
    type: spec.type,
    ...readScopedFields(fields, 'channel'),
  }
  const name = String(entry.name || '').trim()
  entry.name = name
  const duplicate = channels.some(
    (channel) => String(channel.name || '') === name && channel.name !== editingName,
  )
  if (duplicate) return { ok: false, error: 'duplicate', name }
  return { ok: true, entry }
}

/* ── The list ───────────────────────────────────────────────────────────── */

/** The status pill's tone: running is good, dead is bad, the rest are quiet. */
export function statusTone(status: string): Tone | undefined {
  if (status === 'running' || status === 'connected') return 'ok'
  if (status === 'dead') return 'danger'
  if (status === 'restarting' || status === 'exhausted') return 'warn'
  return undefined
}

/**
 * `channels.restart` only knows adapters the running gateway has loaded: a
 * disabled entry, or one added since the last restart, has nothing to restart.
 */
export function canRestartChannel(channel: MergedChannel): boolean {
  if (channel.enabled === false) return false
  const status = String(channel.status || '')
  return status !== 'disabled' && status !== 'stopped'
}

/** Whether the guided form can edit this channel: its type is in the catalog. */
export function hasGuidedSetup(channel: MergedChannel, specs: readonly ChannelSpec[]): boolean {
  const type = String(channel.type || '')
  return specs.some((spec) => spec.type === type)
}

/** The configured entry behind a runtime row, from the config snapshot. */
export function configuredEntry(
  entries: ReadonlyArray<Record<string, unknown>> | undefined,
  name: string | undefined,
): Record<string, unknown> | undefined {
  if (!name) return undefined
  return entries?.find((entry) => String(entry.name || '') === name)
}

export function errorText(err: unknown): string {
  return err instanceof Error ? err.message : String(err)
}

export function isRevisionConflict(err: unknown): boolean {
  return /revision mismatch/i.test(errorText(err))
}
