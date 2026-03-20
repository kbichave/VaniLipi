import { useEffect, useRef } from 'react'
import { Icon } from './Icon'

interface SearchBarProps {
  query: string
  onChange: (q: string) => void
  onClose: () => void
}

export function SearchBar({ query, onChange, onClose }: SearchBarProps) {
  const ref = useRef<HTMLInputElement>(null)
  useEffect(() => { ref.current?.focus() }, [])

  return (
    <div className="animate-slideDown flex items-center gap-2 px-5 py-2 border-b"
         style={{ background: 'var(--surface-1)', borderColor: 'var(--border)' }}>
      <Icon name="search" size={14} />
      <input
        ref={ref}
        type="text"
        placeholder="Search transcript… (⌘F)"
        value={query}
        onChange={e => onChange(e.target.value)}
        style={{
          flex: 1,
          border: 'none',
          background: 'transparent',
          outline: 'none',
          fontSize: 14,
          color: 'var(--text-primary)',
          padding: '4px 0',
        }}
        onKeyDown={e => e.key === 'Escape' && onClose()}
      />
      <button
        className="btn btn-ghost"
        style={{ padding: '3px 6px' }}
        onClick={onClose}
        title="Close search"
      >
        <Icon name="x" size={14} />
      </button>
    </div>
  )
}
