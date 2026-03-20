import { useState } from 'react'
import type { ModelStatus } from '../types'
import { Icon } from './Icon'

interface ModelSetupModalProps {
  modelStatus: ModelStatus
  onClose: () => void
}

interface DownloadProgress {
  type: string
  elapsed_seconds?: number
  message?: string
}

export function ModelSetupModal({ modelStatus, onClose }: ModelSetupModalProps) {
  const [downloading, setDownloading] = useState<string | null>(null)
  const [progress, setProgress] = useState<Record<string, DownloadProgress>>({})

  function downloadModel(modelId: string) {
    setDownloading(modelId)
    const es = new EventSource(`/api/models/download/${encodeURIComponent(modelId)}`)
    es.onmessage = (e) => {
      const msg: DownloadProgress = JSON.parse(e.data)
      setProgress(prev => ({ ...prev, [modelId]: msg }))
      if (msg.type === 'done' || msg.type === 'error') {
        es.close()
        setDownloading(null)
        fetch('/api/models/status').then(r => r.json()).catch(() => null)
      }
    }
    es.onerror = () => {
      es.close()
      setDownloading(null)
      setProgress(prev => ({ ...prev, [modelId]: { type: 'error', message: 'Connection failed' } }))
    }
  }

  const models = modelStatus.models

  return (
    <div className="modal-backdrop" onClick={e => e.target === e.currentTarget && !downloading && onClose()}>
      <div className="modal" style={{ width: 440 }}>
        <div className="modal-header">
          <div className="flex justify-between items-center pb-3">
            <h2 className="text-xl font-semibold" style={{ color: 'var(--text-primary)' }}>Download Models</h2>
            {!downloading && (
              <button className="btn btn-ghost" style={{ padding: '4px 6px' }} onClick={onClose}>
                <Icon name="x" size={16} />
              </button>
            )}
          </div>
          <p className="text-xs pb-4" style={{ color: 'var(--text-muted)' }}>
            VaniLipi runs 100% offline. Models are downloaded once and cached locally (~2.5 GB total).
          </p>
        </div>

        <div className="modal-body">
          <div className="flex flex-col gap-3">
            {models.map(m => {
              const prog = progress[m.id]
              const isDownloading = downloading === m.id
              return (
                <div key={m.id} className="flex items-center gap-3 p-3 rounded-xl"
                     style={{ background: 'var(--surface-2)', border: '1px solid var(--border)' }}>
                  <div className="flex-1 min-w-0">
                    <div className="text-sm font-medium" style={{ color: 'var(--text-primary)' }}>{m.label}</div>
                    <div className="text-xs truncate" style={{ color: 'var(--text-muted)' }}>{m.id}</div>
                    {/* Progress bar */}
                    {isDownloading && prog?.type === 'progress' && (
                      <div className="mt-2">
                        <div className="h-1 rounded-full overflow-hidden" style={{ background: 'var(--surface-3)' }}>
                          <div className="h-full rounded-full animate-pulse" style={{ background: 'var(--accent)', width: '60%' }} />
                        </div>
                        <div className="text-xs mt-1" style={{ color: 'var(--accent)' }}>
                          Downloading… {prog.elapsed_seconds}s
                        </div>
                      </div>
                    )}
                    {prog?.type === 'error' && (
                      <div className="text-xs mt-1" style={{ color: 'var(--error)' }}>{prog.message}</div>
                    )}
                  </div>
                  {m.downloaded
                    ? (
                      <span className="flex items-center gap-1 text-xs font-medium px-2 py-1 rounded-full"
                            style={{ background: 'rgba(22, 163, 74, 0.1)', color: 'var(--success)' }}>
                        <Icon name="check" size={12} /> Ready
                      </span>
                    )
                    : (
                      <button
                        className="btn btn-primary shrink-0"
                        style={{ fontSize: 12, padding: '5px 12px' }}
                        onClick={() => downloadModel(m.id)}
                        disabled={downloading !== null}
                      >
                        {isDownloading ? <Icon name="spinner" size={13} /> : 'Download'}
                      </button>
                    )
                  }
                </div>
              )
            })}
          </div>
        </div>

        {!downloading && (
          <div className="px-6 pb-5 flex justify-end">
            <button className="btn btn-primary" onClick={onClose}>Close</button>
          </div>
        )}
      </div>
    </div>
  )
}
