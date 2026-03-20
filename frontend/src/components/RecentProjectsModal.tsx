import type { Project } from '../types'
import { Icon } from './Icon'

interface RecentProjectsModalProps {
  projects: Project[]
  onLoad: (hash: string) => void
  onDelete: (hash: string) => void
  onClose: () => void
}

function timeAgo(dateStr: string): string {
  const diff = Date.now() - new Date(dateStr).getTime()
  const minutes = Math.floor(diff / 60000)
  if (minutes < 1) return 'just now'
  if (minutes < 60) return `${minutes}m ago`
  const hours = Math.floor(minutes / 60)
  if (hours < 24) return `${hours}h ago`
  const days = Math.floor(hours / 24)
  if (days < 7) return `${days}d ago`
  return new Date(dateStr).toLocaleDateString()
}

export function RecentProjectsModal({ projects, onLoad, onDelete, onClose }: RecentProjectsModalProps) {
  return (
    <div className="modal-backdrop" onClick={e => e.target === e.currentTarget && onClose()}>
      <div className="modal" style={{ width: 480, maxHeight: '80vh', display: 'flex', flexDirection: 'column' }}>
        <div className="modal-header">
          <div className="flex justify-between items-center pb-4">
            <h2 className="text-xl font-semibold" style={{ color: 'var(--text-primary)' }}>Recent Projects</h2>
            <button className="btn btn-ghost" style={{ padding: '4px 6px' }} onClick={onClose}>
              <Icon name="x" size={16} />
            </button>
          </div>
        </div>

        <div className="modal-body overflow-y-auto flex-1" style={{ paddingTop: 0 }}>
          {projects.length === 0 ? (
            <div className="flex flex-col items-center gap-3 py-8" style={{ color: 'var(--text-muted)' }}>
              <Icon name="folder" size={32} />
              <p className="text-sm">No saved transcripts yet.</p>
            </div>
          ) : (
            <div className="flex flex-col gap-2">
              {projects.map(p => (
                <div key={p.file_hash} className="flex items-center gap-3 p-3 rounded-xl transition-colors"
                     style={{ background: 'var(--surface-2)', border: '1px solid var(--border)' }}>
                  {/* File icon */}
                  <div className="shrink-0 w-9 h-9 rounded-lg flex items-center justify-center"
                       style={{ background: 'var(--accent-subtle)', color: 'var(--accent)' }}>
                    <Icon name="mic" size={16} />
                  </div>
                  {/* Info */}
                  <div className="flex-1 min-w-0">
                    <div className="text-sm font-medium truncate" style={{ color: 'var(--text-primary)' }}>
                      {p.filename}
                    </div>
                    <div className="text-xs" style={{ color: 'var(--text-muted)' }}>
                      {p.segment_count} segments · {Math.round(p.duration_seconds)}s · {p.detected_language ?? p.language}
                      {p.created_at && <> · {timeAgo(p.created_at)}</>}
                    </div>
                  </div>
                  {/* Actions */}
                  <button className="btn btn-primary shrink-0" style={{ fontSize: 12, padding: '5px 12px' }}
                          onClick={() => { onLoad(p.file_hash); onClose() }}>
                    Load
                  </button>
                  <button className="btn btn-ghost shrink-0"
                          style={{ padding: '6px 8px' }}
                          onClick={(e) => { e.stopPropagation(); onDelete(p.file_hash) }}
                          onMouseOver={e => (e.currentTarget.style.color = '#DC2626')}
                          onMouseOut={e => (e.currentTarget.style.color = '')}
                          title="Delete project">
                    <Icon name="x" size={14} />
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
