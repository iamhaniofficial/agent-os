import { Gamepad2, Hash, Mail, Plug, Send, Users, type LucideIcon } from 'lucide-react'
import { cn } from '~/lib/utils'

/**
 * A plate per messaging adapter, the Providers section's posture: the app
 * bundles no brand artwork for channels, so each gets a glyph that says
 * what kind of place it is rather than whose. Unknown types get a plug.
 */
const GLYPHS: Record<string, LucideIcon> = {
  telegram: Send,
  discord: Gamepad2,
  slack: Hash,
  msteams: Users,
  microsoftteams: Users,
  email: Mail,
}

function normalise(type: string): string {
  return type
    .trim()
    .toLowerCase()
    .replace(/[\s_-]+/g, '')
}

export function adapterKey(type: string): string {
  const key = normalise(type)
  return key in GLYPHS ? key : 'generic'
}

export function AdapterMark({
  type,
  size = 28,
  className,
}: {
  type: string
  size?: number
  className?: string
}) {
  const Icon = GLYPHS[normalise(type)] ?? Plug
  return (
    <span
      className={cn('prov-logo', className)}
      style={{ width: size, height: size }}
      data-adapter={adapterKey(type)}
      aria-hidden
    >
      <Icon className="prov-logo__glyph" strokeWidth={1.75} />
    </span>
  )
}
