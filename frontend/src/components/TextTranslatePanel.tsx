import { useState } from 'react'
import { retranslateSegment } from '../api'
import { Icon } from './Icon'

interface TextTranslatePanelProps {
  language: string
  onBack: () => void
}

export function TextTranslatePanel({ language, onBack }: TextTranslatePanelProps) {
  const [inputText, setInputText] = useState('')
  const [translation, setTranslation] = useState('')
  const [translating, setTranslating] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function handleTranslate() {
    if (!inputText.trim()) return
    setTranslating(true)
    setError(null)
    try {
      const langCode = language === 'auto' ? 'mr' : language
      const data = await retranslateSegment('text-translate', inputText.trim(), langCode)
      setTranslation(data.english)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setTranslating(false)
    }
  }

  function handleCopy() {
    if (translation) {
      navigator.clipboard.writeText(translation)
    }
  }

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      {/* Toolbar */}
      <div className="flex items-center gap-3 px-5 py-3 shrink-0 border-b"
           style={{ borderColor: 'var(--border)' }}>
        <button className="btn btn-ghost" style={{ padding: '4px 8px' }} onClick={onBack}>
          <Icon name="x" size={14} /> Back
        </button>
        <span className="text-sm font-medium" style={{ color: 'var(--text-primary)' }}>
          <Icon name="globe" size={14} /> Text Translation
        </span>
      </div>

      {/* Two-pane translation */}
      <div className="flex-1 flex gap-0 overflow-hidden">
        {/* Input pane */}
        <div className="flex-1 flex flex-col p-5 border-r" style={{ borderColor: 'var(--border)' }}>
          <label className="text-xs font-medium uppercase tracking-wider mb-2"
                 style={{ color: 'var(--text-muted)', letterSpacing: '0.08em' }}>
            Source Text
          </label>
          <textarea
            className="font-indic flex-1"
            placeholder="Paste or type Indian language text here…"
            value={inputText}
            onChange={e => setInputText(e.target.value)}
            style={{ fontSize: 17, lineHeight: 1.75, resize: 'none' }}
          />
          <div className="flex items-center gap-2 mt-3">
            <button
              className="btn btn-primary"
              onClick={handleTranslate}
              disabled={!inputText.trim() || translating}
            >
              {translating ? <><Icon name="spinner" size={14} /> Translating…</> : <><Icon name="globe" size={14} /> Translate to English</>}
            </button>
            {inputText && (
              <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                {inputText.length} characters
              </span>
            )}
          </div>
        </div>

        {/* Output pane */}
        <div className="flex-1 flex flex-col p-5" style={{ background: 'var(--surface-1)' }}>
          <div className="flex items-center justify-between mb-2">
            <label className="text-xs font-medium uppercase tracking-wider"
                   style={{ color: 'var(--text-muted)', letterSpacing: '0.08em' }}>
              English Translation
            </label>
            {translation && (
              <button className="btn btn-ghost text-xs" style={{ padding: '2px 8px' }} onClick={handleCopy}>
                <Icon name="copy" size={12} /> Copy
              </button>
            )}
          </div>
          <div className="flex-1 overflow-y-auto">
            {error ? (
              <p className="text-sm" style={{ color: 'var(--error)' }}>{error}</p>
            ) : translation ? (
              <p className="text-base" style={{ color: 'var(--text-primary)', lineHeight: 1.7 }}>
                {translation}
              </p>
            ) : (
              <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
                Translation will appear here…
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
