import { useEffect } from 'react'
import type { Segment } from '../types'

interface ShortcutHandlers {
  segments: Segment[]
  selectedSegIdx: number
  editingSegId: string | null
  onPlayPause: () => void
  onToggleSearch: () => void
  onExport: () => void
  onRateChange: (r: number) => void
  onSegmentNav: (next: number) => void
  onEditStart: (id: string) => void
  onEditCancel: () => void
  onRetranslate: () => void
  onDeselect: () => void
}

const RATE_PRESETS = [0.75, 1, 1.25, 1.5]

export function useKeyboardShortcuts({
  segments,
  selectedSegIdx,
  editingSegId,
  onPlayPause,
  onToggleSearch,
  onExport,
  onRateChange,
  onSegmentNav,
  onEditStart,
  onEditCancel,
  onRetranslate,
  onDeselect,
}: ShortcutHandlers) {
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if ((e.target as HTMLElement).tagName === 'TEXTAREA' ||
          (e.target as HTMLElement).tagName === 'INPUT') return

      // Space → play/pause
      if (e.code === 'Space' && !e.metaKey && !e.ctrlKey) {
        e.preventDefault()
        onPlayPause()
      }

      // ⌘F → search
      if (e.key === 'f' && (e.metaKey || e.ctrlKey)) {
        e.preventDefault()
        onToggleSearch()
      }

      // ⌘E → export
      if (e.key === 'e' && (e.metaKey || e.ctrlKey)) {
        e.preventDefault()
        onExport()
      }

      // ⌘1-4 → speed presets
      if (e.metaKey && !e.shiftKey && ['1', '2', '3', '4'].includes(e.key)) {
        e.preventDefault()
        onRateChange(RATE_PRESETS[parseInt(e.key) - 1])
      }

      // Tab / Shift+Tab → navigate segments
      if (e.key === 'Tab' && segments.length > 0) {
        e.preventDefault()
        const current = selectedSegIdx < 0 ? 0 : selectedSegIdx
        const next = e.shiftKey
          ? Math.max(0, current - 1)
          : Math.min(segments.length - 1, current + 1)
        onSegmentNav(next)
      }

      // Enter → edit selected segment
      if (e.key === 'Enter' && selectedSegIdx >= 0 && selectedSegIdx < segments.length) {
        const seg = segments[selectedSegIdx]
        if (seg && !editingSegId) onEditStart(seg.id)
      }

      // Escape → cancel edit or deselect
      if (e.key === 'Escape') {
        if (editingSegId) onEditCancel()
        else onDeselect()
      }

      // ⌘⇧T → re-translate
      if (e.key === 't' && (e.metaKey || e.ctrlKey) && e.shiftKey) {
        e.preventDefault()
        if (editingSegId) {
          onRetranslate()
        } else if (selectedSegIdx >= 0 && selectedSegIdx < segments.length) {
          onEditStart(segments[selectedSegIdx].id)
        }
      }
    }

    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [segments, selectedSegIdx, editingSegId, onPlayPause, onToggleSearch, onExport, onRateChange, onSegmentNav, onEditStart, onEditCancel, onRetranslate, onDeselect])
}
