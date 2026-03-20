import { useRef } from 'react'
import { Icon } from './Icon'

interface AudioInputAreaProps {
  onFileSelected: (file: File) => void
  isDragging: boolean
  onDragEnter: (e: React.DragEvent) => void
  onDragLeave: (e: React.DragEvent) => void
  onDrop: (e: React.DragEvent) => void
  uploading: boolean
}

export function AudioInputArea({
  onFileSelected,
  isDragging,
  onDragEnter,
  onDragLeave,
  onDrop,
  uploading,
}: AudioInputAreaProps) {
  const fileRef = useRef<HTMLInputElement>(null)

  return (
    <div
      className={`drop-zone ${isDragging ? 'drag-over' : ''}`}
      style={{
        padding: '28px 20px',
        border: '2px dashed var(--border-strong)',
        borderRadius: 12,
        textAlign: 'center',
        cursor: 'pointer',
        background: 'var(--surface-2)',
        transition: 'all 0.15s',
      }}
      onClick={() => !uploading && fileRef.current?.click()}
      onDragEnter={onDragEnter}
      onDragLeave={onDragLeave}
      onDragOver={e => e.preventDefault()}
      onDrop={onDrop}
    >
      <input
        ref={fileRef}
        type="file"
        accept=".mp3,.wav,.m4a,.flac,.ogg,.mp4,.mkv,.webm,.mov,.avi,.wmv"
        style={{ display: 'none' }}
        onChange={e => e.target.files?.[0] && onFileSelected(e.target.files[0])}
      />

      {uploading ? (
        <div className="flex flex-col items-center gap-2" style={{ color: 'var(--text-secondary)' }}>
          <Icon name="spinner" size={24} />
          <span className="text-sm font-medium">Uploading…</span>
        </div>
      ) : (
        <div className="flex flex-col items-center gap-1.5">
          <div className="mb-1" style={{ color: 'var(--text-muted)' }}>
            <Icon name="upload" size={24} />
          </div>
          <span className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>
            Drop audio or video file here
          </span>
          <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
            or click to browse
          </span>
          <span className="text-xs mt-1" style={{ color: 'var(--text-muted)', opacity: 0.7 }}>
            Audio: MP3 · WAV · M4A · FLAC &nbsp;|&nbsp; Video: MP4 · MOV · MKV · AVI
          </span>
        </div>
      )}
    </div>
  )
}
