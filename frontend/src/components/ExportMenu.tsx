import { useState } from 'react'
import { exportTranscript } from '../api'
import type { Segment } from '../types'
import { Icon } from './Icon'

interface ExportMenuProps {
  segments: Segment[]
  fileId: string | null
  onClose: () => void
}

type ContentMode = 'both' | 'source' | 'english'

const EXPORT_FORMATS = [
  { id: 'srt',  label: 'SRT',  desc: 'Subtitle file',    icon: 'file-text' as const },
  { id: 'vtt',  label: 'VTT',  desc: 'Web subtitles',    icon: 'file-text' as const },
  { id: 'txt',  label: 'TXT',  desc: 'Plain text',       icon: 'file-text' as const },
  { id: 'docx', label: 'DOCX', desc: 'Word document',    icon: 'file-text' as const },
  { id: 'pdf',  label: 'PDF',  desc: 'PDF document',     icon: 'file-text' as const },
  { id: 'json', label: 'JSON', desc: 'Structured data',  icon: 'file-text' as const },
]

const CONTENT_MODES: { id: ContentMode; label: string; desc: string }[] = [
  { id: 'both',    label: 'Both',            desc: 'Source + English' },
  { id: 'source',  label: 'Source only',     desc: 'Original language' },
  { id: 'english', label: 'English only',    desc: 'Translation' },
]

export function ExportMenu({ segments, fileId, onClose }: ExportMenuProps) {
  const [status, setStatus] = useState<string | null>(null)
  const [contentMode, setContentMode] = useState<ContentMode>('both')

  async function doExport(format: string) {
    if (!fileId) { setStatus('No transcript to export.'); return }
    setStatus(`Exporting ${format.toUpperCase()}…`)
    try {
      // Filter segments based on content mode
      const filtered = contentMode === 'both'
        ? segments
        : segments.map(s => ({
            ...s,
            marathi: contentMode === 'source' ? s.marathi : '',
            english: contentMode === 'english' ? s.english : '',
          }))
      const blob = await exportTranscript(format, filtered, fileId)
      const suffix = contentMode === 'both' ? '' : `_${contentMode}`
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `transcript${suffix}.${format}`
      a.click()
      URL.revokeObjectURL(url)
      setStatus(null)
      onClose()
    } catch (e) {
      setStatus(`Error: ${(e as Error).message}`)
    }
  }

  return (
    <div className="modal-backdrop" onClick={e => e.target === e.currentTarget && onClose()}>
      <div className="modal" style={{ width: 420 }}>
        <div className="modal-header">
          <div className="flex justify-between items-center pb-3">
            <h2 className="text-xl font-semibold" style={{ color: 'var(--text-primary)' }}>Export Transcript</h2>
            <button className="btn btn-ghost" style={{ padding: '4px 6px' }} onClick={onClose}>
              <Icon name="x" size={16} />
            </button>
          </div>
        </div>

        <div className="modal-body">
          {/* Content mode selector */}
          <p className="text-xs font-semibold mb-2" style={{ color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Content
          </p>
          <div className="flex gap-2 mb-4">
            {CONTENT_MODES.map(m => (
              <button
                key={m.id}
                className="flex-1 flex flex-col items-center gap-0.5 py-2 px-3 rounded-lg cursor-pointer transition-all text-center"
                style={{
                  background: contentMode === m.id ? 'var(--accent-muted)' : 'var(--surface-2)',
                  border: `1.5px solid ${contentMode === m.id ? 'var(--accent)' : 'var(--border)'}`,
                  color: contentMode === m.id ? 'var(--accent)' : 'var(--text-secondary)',
                }}
                onClick={() => setContentMode(m.id)}
              >
                <span className="text-sm font-medium">{m.label}</span>
                <span className="text-xs" style={{ opacity: 0.7 }}>{m.desc}</span>
              </button>
            ))}
          </div>

          {/* Format grid */}
          <p className="text-xs font-semibold mb-2" style={{ color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Format
          </p>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8 }}>
            {EXPORT_FORMATS.map(f => (
              <button
                key={f.id}
                className="flex flex-col items-center gap-1 p-3 rounded-xl cursor-pointer transition-all"
                style={{
                  background: 'var(--surface-2)',
                  border: '1px solid var(--border)',
                }}
                onMouseOver={e => {
                  e.currentTarget.style.borderColor = 'var(--accent)'
                  e.currentTarget.style.transform = 'scale(1.02)'
                }}
                onMouseOut={e => {
                  e.currentTarget.style.borderColor = 'var(--border)'
                  e.currentTarget.style.transform = 'scale(1)'
                }}
                onClick={() => doExport(f.id)}
                disabled={!!status}
              >
                <Icon name={f.icon} size={16} />
                <span className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>{f.label}</span>
                <span className="text-xs" style={{ color: 'var(--text-muted)' }}>{f.desc}</span>
              </button>
            ))}
          </div>
          {status && (
            <p className="mt-3 text-xs text-center" style={{ color: 'var(--text-secondary)' }}>{status}</p>
          )}
        </div>
      </div>
    </div>
  )
}
