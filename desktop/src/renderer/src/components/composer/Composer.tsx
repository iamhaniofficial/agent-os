import { ArrowUp, Paperclip, SlidersHorizontal, Square, X } from 'lucide-react'
import { AnimatePresence, motion, useReducedMotion } from 'motion/react'
import {
  useCallback,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
  type ReactNode,
  type Ref,
} from 'react'
import { toast } from 'sonner'
import type { ComposerHandle } from '@/views/chat/Composer'
import { MAX_PENDING, sendButtonState } from '@/views/chat/logic'
import { t } from '~/i18n'
import { quick } from '~/lib/motion'

export type { ComposerHandle }

/**
 * Same contract as the web console's Composer (frontend/src/views/chat/
 * Composer.tsx) so the shared chat hooks drive it unchanged; the markup and
 * styling are the desktop's own. Keyboard behaviour is ported verbatim:
 * Enter sends, Shift+Enter newline, Escape aborts → recovers queue → clears,
 * Alt+↑/↓ pop/enqueue pending, ↑/↓ walk sent history, Tab takes the
 * suggested next prompt into an empty composer.
 */
export interface ComposerProps {
  onSend: (text: string) => void
  onValueChange?: (value: string) => void
  onSlashKeyDown?: (e: React.KeyboardEvent<HTMLTextAreaElement>) => boolean
  composerRef?: Ref<ComposerHandle>
  slashMenu?: ReactNode
  slashListboxId?: string
  slashActiveDescendant?: string
  onAbort?: () => void
  busy: boolean
  pendingCompaction?: boolean
  history?: string[]
  hasPendingAttachments?: boolean
  hasPendingWork?: boolean
  onAttachFiles?: (files: File[] | FileList) => void
  tray?: ReactNode
  routerFxDock?: ReactNode
  routePicker?: ReactNode
  /**
   * Run modes body (the console's chat Toolbar: execution mode, Pilot Router,
   * plan mode, session usage). Mounted in a popover behind a sliders button at
   * the left of the capsule, and only while that popover is open, so every
   * open reads fresh state. Absent: no button.
   */
  toolbar?: ReactNode
  pendingCount?: number
  onRecoverPending?: () => boolean
  onPopPendingTail?: () => void
  onEnqueueCurrent?: () => void
  autoFocus?: boolean
  /**
   * Enter sends and Shift+Enter breaks the line (default). Off: Enter breaks
   * the line and ⌘Enter sends. Settings > General.
   */
  enterToSend?: boolean
  /** A row that sits between the tray and the capsule (the desk's seats). */
  seats?: ReactNode
  /** Overrides the default placeholder (the desk rotates real orders). */
  placeholder?: string
  /**
   * The suggested next prompt (`chat.suggestNext`): the placeholder while the
   * composer is empty, ahead of `placeholder`; Tab takes it, Enter sends it.
   */
  suggestion?: string | null
  onFocusChange?: (focused: boolean) => void
}

const MIN_TEXTAREA_HEIGHT = 26
const MAX_TEXTAREA_HEIGHT = 160

export function Composer({
  onSend,
  onValueChange,
  onSlashKeyDown,
  slashMenu,
  slashListboxId,
  slashActiveDescendant,
  composerRef,
  onAbort,
  busy,
  pendingCompaction = false,
  history = [],
  hasPendingAttachments = false,
  hasPendingWork = false,
  onAttachFiles,
  tray,
  routerFxDock,
  routePicker,
  toolbar,
  pendingCount = 0,
  onRecoverPending,
  onPopPendingTail,
  onEnqueueCurrent,
  autoFocus = true,
  enterToSend = true,
  seats,
  placeholder,
  suggestion = null,
  onFocusChange,
}: ComposerProps) {
  const [value, setValue] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const historyIdxRef = useRef<number | null>(null)
  const historyDraftRef = useRef('')
  const [runModesOpen, setRunModesOpen] = useState(false)
  const runModesWrapRef = useRef<HTMLDivElement>(null)
  const runModesTriggerRef = useRef<HTMLButtonElement>(null)
  const runModesCloseRef = useRef<HTMLButtonElement>(null)
  const reduceMotion = useReducedMotion()

  const closeRunModes = useCallback(() => {
    setRunModesOpen(false)
    runModesTriggerRef.current?.focus({ preventScroll: true })
  }, [])

  // Outside click and Escape close the popover. The body's bypass confirm is
  // a ModalShell portalled to <body>, outside the wrap: a press inside it must
  // not close the popover (that would unmount the dialog before its buttons
  // receive the click), and its own Escape belongs to the dialog.
  useEffect(() => {
    if (!runModesOpen) return
    runModesCloseRef.current?.focus({ preventScroll: true })
    const inside = (target: EventTarget | null): boolean =>
      target instanceof Node &&
      (Boolean(runModesWrapRef.current?.contains(target)) ||
        (target instanceof Element && target.closest('[role="alertdialog"]') !== null))
    const onDocMouseDown = (e: MouseEvent) => {
      if (!inside(e.target)) setRunModesOpen(false)
    }
    const onDocKeyDown = (e: KeyboardEvent) => {
      if (e.key !== 'Escape' || document.querySelector('[role="alertdialog"]')) return
      // The open popover owns Escape before the textarea's abort/clear chain.
      e.preventDefault()
      e.stopPropagation()
      closeRunModes()
    }
    document.addEventListener('mousedown', onDocMouseDown)
    document.addEventListener('keydown', onDocKeyDown, true)
    return () => {
      document.removeEventListener('mousedown', onDocMouseDown)
      document.removeEventListener('keydown', onDocKeyDown, true)
    }
  }, [runModesOpen, closeRunModes])

  const autoResize = useCallback(() => {
    const ta = textareaRef.current
    if (!ta) return
    if (!ta.value) {
      ta.style.height = ''
      return
    }
    ta.style.height = 'auto'
    ta.style.height =
      Math.max(MIN_TEXTAREA_HEIGHT, Math.min(ta.scrollHeight, MAX_TEXTAREA_HEIGHT)) + 'px'
  }, [])

  const setProgrammatic = useCallback(
    (text: string) => {
      setValue(text)
      const ta = textareaRef.current
      if (ta) {
        ta.value = text
        try {
          ta.setSelectionRange(text.length, text.length)
        } catch {
          /* detached */
        }
      }
      onValueChange?.(text)
      autoResize()
    },
    [autoResize, onValueChange],
  )

  useEffect(() => {
    if (autoFocus) textareaRef.current?.focus({ preventScroll: true })
  }, [autoFocus])

  useImperativeHandle(
    composerRef,
    (): ComposerHandle => ({
      clear: () => {
        setProgrammatic('')
        historyIdxRef.current = null
        historyDraftRef.current = ''
      },
      focus: () => textareaRef.current?.focus({ preventScroll: true }),
      setValue: (text: string) => {
        setProgrammatic(text)
        historyIdxRef.current = null
        historyDraftRef.current = ''
        textareaRef.current?.focus({ preventScroll: true })
      },
      getValue: () => textareaRef.current?.value ?? '',
    }),
    [setProgrammatic],
  )

  const cycleHistory = useCallback(
    (dir: number): boolean => {
      if (history.length === 0) return false
      if (dir < 0) {
        if (historyIdxRef.current === null) {
          historyDraftRef.current = textareaRef.current?.value ?? value ?? ''
          historyIdxRef.current = history.length - 1
        } else {
          historyIdxRef.current = Math.max(0, historyIdxRef.current - 1)
        }
        setProgrammatic(history[historyIdxRef.current] ?? '')
        return true
      }
      if (historyIdxRef.current === null) return false
      const next = historyIdxRef.current + 1
      if (next >= history.length) {
        historyIdxRef.current = null
        setProgrammatic(historyDraftRef.current)
        historyDraftRef.current = ''
      } else {
        historyIdxRef.current = next
        setProgrammatic(history[next] ?? '')
      }
      return true
    },
    [history, setProgrammatic, value],
  )

  const doSend = useCallback(() => {
    const text = value.trim()
    if (hasPendingWork) {
      toast.warning('Wait for file attachment processing to finish')
      return
    }
    if (!text && !hasPendingAttachments) return
    onSend(text)
    setProgrammatic('')
    historyIdxRef.current = null
    historyDraftRef.current = ''
  }, [value, hasPendingAttachments, hasPendingWork, onSend, setProgrammatic])

  const onKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.nativeEvent.isComposing || e.keyCode === 229) return
      if (onSlashKeyDown?.(e)) return
      // Bare Tab over an empty composer takes the suggestion; otherwise Tab
      // keeps moving focus.
      if (
        e.key === 'Tab' &&
        suggestion &&
        !textareaRef.current?.value &&
        !e.shiftKey &&
        !e.altKey &&
        !e.ctrlKey &&
        !e.metaKey
      ) {
        e.preventDefault()
        setProgrammatic(suggestion)
        historyIdxRef.current = null
        historyDraftRef.current = ''
        return
      }
      if (e.key === 'Escape') {
        if (busy) {
          e.preventDefault()
          onAbort?.()
          return
        }
        if (pendingCount > 0) {
          e.preventDefault()
          onRecoverPending?.()
          return
        }
        if (textareaRef.current?.value) {
          e.preventDefault()
          setProgrammatic('')
          historyIdxRef.current = null
          historyDraftRef.current = ''
        }
        return
      }
      if (e.key === 'ArrowUp' && e.altKey && pendingCount > 0) {
        e.preventDefault()
        onPopPendingTail?.()
        return
      }
      if (
        e.key === 'ArrowDown' &&
        e.altKey &&
        textareaRef.current?.value &&
        pendingCount < MAX_PENDING
      ) {
        e.preventDefault()
        onEnqueueCurrent?.()
        return
      }
      if (
        e.key === 'ArrowUp' &&
        !e.altKey &&
        !e.shiftKey &&
        (!textareaRef.current?.value || historyIdxRef.current !== null)
      ) {
        if (cycleHistory(-1)) {
          e.preventDefault()
          return
        }
      }
      if (e.key === 'ArrowDown' && !e.altKey && !e.shiftKey && historyIdxRef.current !== null) {
        if (cycleHistory(1)) {
          e.preventDefault()
          return
        }
      }
      if (e.key === 'Enter') {
        const sends = enterToSend ? !e.shiftKey && !e.metaKey : e.metaKey
        if (sends) {
          e.preventDefault()
          doSend()
        }
      }
    },
    [
      enterToSend,
      suggestion,
      busy,
      onAbort,
      cycleHistory,
      doSend,
      setProgrammatic,
      onSlashKeyDown,
      pendingCount,
      onRecoverPending,
      onPopPendingTail,
      onEnqueueCurrent,
    ],
  )

  const onChange = useCallback(
    (e: React.ChangeEvent<HTMLTextAreaElement>) => {
      setValue(e.target.value)
      historyIdxRef.current = null
      historyDraftRef.current = ''
      onValueChange?.(e.target.value)
      autoResize()
    },
    [autoResize, onValueChange],
  )

  const showSuggestion = Boolean(suggestion) && !value

  const { disabled: sendDisabled, label: sendLabel } = sendButtonState(
    value,
    busy,
    pendingCompaction,
    hasPendingAttachments,
  )

  return (
    <div className="composer-shell">
      {routerFxDock}
      {tray}
      {seats}
      {slashMenu}
      <form
        className="composer"
        data-busy={busy}
        onSubmit={(e) => {
          e.preventDefault()
          doSend()
        }}
      >
        {toolbar ? (
          <div className="composer-run-modes" ref={runModesWrapRef}>
            <button
              ref={runModesTriggerRef}
              type="button"
              className="composer-chip"
              aria-haspopup="dialog"
              aria-expanded={runModesOpen}
              aria-controls={runModesOpen ? 'composer-run-modes-popover' : undefined}
              aria-label={t('composer.runModes')}
              title={t('composer.runModesTitle')}
              data-open={runModesOpen || undefined}
              onClick={() => setRunModesOpen((v) => !v)}
            >
              <SlidersHorizontal className="size-4" strokeWidth={1.75} aria-hidden />
            </button>
            <AnimatePresence initial={false}>
              {runModesOpen ? (
                <motion.div
                  id="composer-run-modes-popover"
                  className="composer-run-modes__popover"
                  role="dialog"
                  aria-labelledby="composer-run-modes-title"
                  initial={reduceMotion ? false : { opacity: 0, scale: 0.98, y: 4 }}
                  animate={{ opacity: 1, scale: 1, y: 0 }}
                  exit={reduceMotion ? { opacity: 0 } : { opacity: 0, scale: 0.98, y: 2 }}
                  transition={reduceMotion ? { duration: 0 } : quick}
                >
                  <header className="composer-run-modes__header">
                    <h2 id="composer-run-modes-title" className="composer-run-modes__title">
                      {t('composer.runModes')}
                    </h2>
                    <button
                      ref={runModesCloseRef}
                      type="button"
                      className="composer-run-modes__close"
                      aria-label={t('composer.runModesClose')}
                      title={t('composer.runModesClose')}
                      onClick={closeRunModes}
                    >
                      <X className="size-3.5" strokeWidth={2} aria-hidden />
                    </button>
                  </header>
                  <div className="composer-run-modes__body">{toolbar}</div>
                </motion.div>
              ) : null}
            </AnimatePresence>
          </div>
        ) : null}
        {onAttachFiles ? (
          <>
            <input
              ref={fileInputRef}
              type="file"
              multiple
              hidden
              accept="image/png,image/jpeg,image/gif,image/webp,application/pdf,text/plain,text/markdown,text/html,text/csv,application/json,.png,.jpg,.jpeg,.gif,.webp,.pdf,.txt,.md,.markdown,.html,.htm,.csv,.json"
              onChange={(e) => {
                if (e.target.files && e.target.files.length > 0) onAttachFiles(e.target.files)
                e.target.value = ''
              }}
              aria-label={t('composer.attach')}
            />
            <button
              type="button"
              className="composer-chip"
              aria-label={t('composer.attach')}
              title={t('composer.attach')}
              onClick={() => fileInputRef.current?.click()}
            >
              <Paperclip className="size-4" strokeWidth={1.75} aria-hidden />
            </button>
          </>
        ) : null}
        <textarea
          ref={textareaRef}
          rows={1}
          value={value}
          onChange={onChange}
          onKeyDown={onKeyDown}
          onFocus={() => onFocusChange?.(true)}
          onBlur={() => onFocusChange?.(false)}
          placeholder={(showSuggestion && suggestion) || (placeholder ?? t('composer.placeholder'))}
          aria-label={t('composer.placeholder')}
          aria-keyshortcuts={showSuggestion ? 'Tab' : undefined}
          aria-autocomplete={slashListboxId ? 'list' : undefined}
          aria-expanded={slashListboxId ? Boolean(slashActiveDescendant) : undefined}
          aria-controls={slashActiveDescendant ? slashListboxId : undefined}
          aria-activedescendant={slashActiveDescendant}
        />
        {showSuggestion ? (
          <kbd
            className="composer-suggestion-key"
            data-testid="composer-suggestion-key"
            title={t('composer.suggestionKeyTitle')}
            aria-hidden="true"
          >
            {t('composer.suggestionKey')}
          </kbd>
        ) : null}
        {routePicker}
        {busy ? (
          <button
            type="button"
            className="composer-send"
            data-abort="true"
            onClick={() => onAbort?.()}
            aria-label={t('composer.stop')}
            title={t('composer.stop')}
          >
            <Square className="size-3.5 fill-current" strokeWidth={2} aria-hidden />
          </button>
        ) : (
          <button
            type="submit"
            className="composer-send"
            disabled={sendDisabled}
            aria-label={t('composer.send')}
            title={sendLabel}
          >
            <ArrowUp className="size-4" strokeWidth={2.5} aria-hidden />
          </button>
        )}
      </form>
    </div>
  )
}
