import { useEffect, useRef, useState } from 'react'
import { fmt } from '../constants'
import type { Segment } from '../types'
import { Icon } from './Icon'
import { Typewriter } from './Typewriter'

interface SegmentRowProps {
  seg: Segment
  isActive: boolean
  isSelected: boolean
  isEditing: boolean
  onSeek: (t: number) => void
  onEdit: (id: string) => void
  onEditSave: () => void
  onEditCancel: () => void
  onRetranslate: () => void
  editDraft: string
  onEditChange: (val: string) => void
  retranslating: boolean
}

export function SegmentRow({
  seg,
  isActive,
  isSelected,
  isEditing,
  onSeek,
  onEdit,
  onEditSave,
  onEditCancel,
  onRetranslate,
  editDraft,
  onEditChange,
  retranslating,
}: SegmentRowProps) {
  const isKeyboardFocused = isSelected && !isActive
  const textRef = useRef<HTMLTextAreaElement>(null)
  const segRef = useRef<HTMLDivElement>(null)
  const [englishKey, setEnglishKey] = useState(0)
  const prevEnglish = useRef(seg.english)

  // Only type-animate segments arriving live via WebSocket.
  // Cached/loaded segments (ids like "c0", "p0") render instantly.
  const isStreaming = seg.id.startsWith('ws_')

  useEffect(() => {
    if (isActive && segRef.current) {
      segRef.current.scrollIntoView({ block: 'nearest', behavior: 'smooth' })
    }
  }, [isActive])

  useEffect(() => {
    if (seg.english && seg.english !== prevEnglish.current) {
      prevEnglish.current = seg.english
      setEnglishKey(k => k + 1)
    }
  }, [seg.english])

  useEffect(() => {
    if (isEditing && textRef.current) {
      textRef.current.focus()
      textRef.current.selectionStart = textRef.current.value.length
    }
  }, [isEditing])

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === 'Escape') onEditCancel()
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) onEditSave()
  }

  return (
    <div
      ref={segRef}
      className={`segment ${isActive ? 'active' : ''} ${isKeyboardFocused ? 'keyboard-selected' : ''}`}
      style={{ padding: '10px 20px' }}
    >
      {isEditing ? (
        /* ── Edit mode: full-width ── */
        <div style={{ paddingLeft: 84 }}>
          <textarea
            ref={textRef}
            className="font-indic"
            value={editDraft}
            onChange={e => onEditChange(e.target.value)}
            onKeyDown={handleKeyDown}
            rows={3}
            style={{ marginBottom: 10 }}
          />
          <div className="flex gap-2">
            <button className="btn btn-primary" style={{ padding: '5px 12px', fontSize: 12 }} onClick={onEditSave}>
              <Icon name="check" size={12} /> Save
            </button>
            <button
              className="btn btn-secondary"
              style={{ padding: '5px 12px', fontSize: 12 }}
              onClick={onRetranslate}
              disabled={retranslating}
            >
              {retranslating ? <Icon name="spinner" size={12} /> : <Icon name="refresh" size={12} />}
              Re-translate
            </button>
            <button className="btn btn-ghost" style={{ padding: '5px 10px', fontSize: 12 }} onClick={onEditCancel}>
              Cancel
            </button>
          </div>
        </div>
      ) : (
        /* ── Read mode: 3-column grid (time | source | english) ── */
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: '68px 1fr 1fr',
            gap: '0 16px',
            alignItems: 'baseline',
          }}
        >
          {/* Timestamp */}
          <button
            onClick={() => onSeek(seg.start)}
            className="shrink-0 px-1.5 py-0.5 rounded text-xs font-mono cursor-pointer transition-colors text-left"
            style={{
              background: 'transparent',
              color: isActive ? 'var(--accent)' : 'var(--text-muted)',
              border: 'none',
              fontWeight: isActive ? 600 : 400,
              fontSize: 11,
              lineHeight: 1.75,
              whiteSpace: 'nowrap',
            }}
          >
            {fmt(seg.start)}
          </button>

          {/* Source language */}
          <div
            className="font-indic"
            style={{
              fontSize: 15,
              lineHeight: 1.75,
              color: 'var(--text-primary)',
              borderRight: '1px solid var(--border)',
              paddingRight: 16,
            }}
          >
            <Typewriter
              text={seg.marathi}
              skip={!isStreaming}
              speed={4}
              interval={18}
            />
          </div>

          {/* English translation */}
          <div className="flex items-baseline gap-2">
            <div
              className="flex-1"
              style={{
                fontSize: 14,
                lineHeight: 1.75,
                color: 'var(--text-secondary)',
                fontStyle: seg.english ? 'normal' : 'italic',
                opacity: seg.english ? 1 : 0.4,
              }}
            >
              {seg.english ? (
                <Typewriter
                  key={englishKey}
                  text={seg.english}
                  skip={!isStreaming}
                  speed={3}
                  interval={15}
                />
              ) : (
                isStreaming ? (
                  <span className="typewriter-cursor" style={{ opacity: 0.5 }}>|</span>
                ) : '…'
              )}
            </div>

            {/* Edit button — compact, appears on hover */}
            <button
              className="btn btn-ghost edit-reveal shrink-0"
              style={{ padding: '2px 5px', alignSelf: 'flex-start' }}
              onClick={() => onEdit(seg.id)}
              title="Edit (Enter)"
            >
              <Icon name="edit" size={12} />
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
