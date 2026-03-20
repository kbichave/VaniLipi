import { MODEL_OPTIONS } from '../constants'
import type { Theme } from '../types'
import { Icon } from './Icon'

interface SettingsModalProps {
  onClose: () => void
  theme: Theme
  onThemeChange: (t: Theme) => void
  model: string
  onModelChange: (m: string) => void
}

const THEME_OPTIONS: { id: Theme; label: string; icon: 'sun' | 'moon' | 'monitor'; lightBg: string; lightFg: string; darkBg: string; darkFg: string }[] = [
  { id: 'light',  label: 'Light',  icon: 'sun',     lightBg: '#FAFAF9', lightFg: '#1C1917', darkBg: '#FAFAF9', darkFg: '#1C1917' },
  { id: 'dark',   label: 'Dark',   icon: 'moon',    lightBg: '#1C1917', lightFg: '#FAFAF9', darkBg: '#1C1917', darkFg: '#FAFAF9' },
  { id: 'system', label: 'System', icon: 'monitor', lightBg: '#F5F5F4', lightFg: '#57534E', darkBg: '#292524', darkFg: '#A8A29E' },
]

export function SettingsModal({ onClose, theme, onThemeChange, model, onModelChange }: SettingsModalProps) {
  return (
    <div className="modal-backdrop" onClick={e => e.target === e.currentTarget && onClose()}>
      <div className="modal" style={{ width: 400 }}>
        <div className="modal-header">
          <div className="flex justify-between items-center pb-4">
            <h2 className="text-xl font-semibold" style={{ color: 'var(--text-primary)' }}>Settings</h2>
            <button className="btn btn-ghost" style={{ padding: '4px 6px' }} onClick={onClose}>
              <Icon name="x" size={16} />
            </button>
          </div>
        </div>

        <div className="modal-body flex flex-col gap-5">
          {/* Theme */}
          <div>
            <label className="text-xs font-medium uppercase tracking-wider block mb-2"
                   style={{ color: 'var(--text-muted)', letterSpacing: '0.08em' }}>
              Theme
            </label>
            <div className="flex gap-2">
              {THEME_OPTIONS.map(t => (
                <button
                  key={t.id}
                  className="flex-1 flex flex-col items-center gap-2 p-3 rounded-xl cursor-pointer transition-all"
                  style={{
                    background: theme === t.id ? 'var(--accent-subtle)' : 'var(--surface-2)',
                    border: theme === t.id ? '2px solid var(--accent)' : '2px solid var(--border)',
                  }}
                  onClick={() => onThemeChange(t.id)}
                >
                  {/* Mini preview swatch */}
                  <div className="w-8 h-5 rounded" style={{ background: t.lightBg, border: '1px solid var(--border)' }}>
                    <div className="w-4 h-1 rounded-sm mt-1 ml-1" style={{ background: t.lightFg, opacity: 0.6 }} />
                    <div className="w-3 h-1 rounded-sm mt-0.5 ml-1" style={{ background: t.lightFg, opacity: 0.3 }} />
                  </div>
                  <span className="text-xs font-medium" style={{ color: theme === t.id ? 'var(--accent)' : 'var(--text-secondary)' }}>
                    {t.label}
                  </span>
                </button>
              ))}
            </div>
          </div>

          {/* Divider */}
          <div style={{ height: 1, background: 'var(--border)' }} />

          {/* Model quality */}
          <div>
            <label className="text-xs font-medium uppercase tracking-wider block mb-2"
                   style={{ color: 'var(--text-muted)', letterSpacing: '0.08em' }}>
              Model Quality
            </label>
            <div className="flex flex-col gap-2">
              {MODEL_OPTIONS.map(m => (
                <label
                  key={m.id}
                  className="flex items-center gap-3 p-3 rounded-lg cursor-pointer transition-colors"
                  style={{
                    background: model === m.id ? 'var(--accent-subtle)' : 'var(--surface-2)',
                    border: model === m.id ? '1px solid var(--accent)' : '1px solid var(--border)',
                  }}
                >
                  {/* Custom radio */}
                  <span className="shrink-0 w-4 h-4 rounded-full border-2 flex items-center justify-center"
                        style={{ borderColor: model === m.id ? 'var(--accent)' : 'var(--border-strong)' }}>
                    {model === m.id && (
                      <span className="w-2 h-2 rounded-full" style={{ background: 'var(--accent)' }} />
                    )}
                  </span>
                  <input type="radio" name="model" value={m.id} checked={model === m.id}
                         onChange={() => onModelChange(m.id)} className="sr-only" />
                  <span className="text-sm" style={{ color: 'var(--text-primary)' }}>{m.label}</span>
                </label>
              ))}
            </div>
          </div>

          {/* Privacy note */}
          <div className="flex items-start gap-2 p-3 rounded-lg text-xs"
               style={{ background: 'var(--surface-2)', color: 'var(--text-secondary)' }}>
            <Icon name="shield" size={14} />
            <span>All processing happens 100% on your Mac. No audio is ever sent to the internet.</span>
          </div>
        </div>

        <div className="px-6 pb-5 flex justify-end">
          <button className="btn btn-primary" onClick={onClose}>Done</button>
        </div>
      </div>
    </div>
  )
}
