import { useEffect, useState } from 'react'
import { Icon } from './Icon'

interface ProcessingStatusProps {
  progress: number
  statusMsg: string
  transcribing: boolean
  segmentCount: number
}

const FUN_MESSAGES = [
  'Warming up the neural pathways…',
  'Unfurling the acoustic waves…',
  'Wiggling through spectrograms…',
  'Decoding syllables with care…',
  'Listening intently…',
  'Sifting through phonemes…',
  'Consulting the attention heads…',
  'Aligning tokens with meaning…',
  'Whispering to the model…',
  'Weaving words from sound…',
  'Crunching mel-frequencies…',
  'Nudging the beam search along…',
  'Polishing the transcript…',
  'Herding attention weights…',
  'Tickling the transformer layers…',
]

const FUN_TRANSLATE_MESSAGES = [
  'Bridging languages…',
  'Translating with finesse…',
  'Mapping meanings across scripts…',
  'IndicTrans is thinking…',
  'Converting Devanagari to English thoughts…',
  'Cross-lingual magic in progress…',
  'Teaching English what was said…',
  'Shuffling between vocabularies…',
]

function useRotatingMessage(messages: string[], enabled: boolean, intervalMs: number = 3000): string {
  const [index, setIndex] = useState(() => Math.floor(Math.random() * messages.length))

  useEffect(() => {
    if (!enabled) return
    const timer = setInterval(() => {
      setIndex(i => (i + 1) % messages.length)
    }, intervalMs)
    return () => clearInterval(timer)
  }, [enabled, messages.length, intervalMs])

  return messages[index]
}

/**
 * A polished processing widget with circular progress ring,
 * step indicators, fun rotating messages, and real-time stats.
 */
export function ProcessingStatus({ progress, statusMsg, transcribing, segmentCount }: ProcessingStatusProps) {
  const isUploading = statusMsg.toLowerCase().includes('upload')
  const isActive = transcribing || isUploading || progress > 0
  if (!isActive) return null

  const isTranslating = statusMsg.toLowerCase().includes('translat')
  const funMessage = useRotatingMessage(
    isTranslating ? FUN_TRANSLATE_MESSAGES : FUN_MESSAGES,
    transcribing,
  )

  // Circular progress ring dimensions
  const size = 48
  const stroke = 3
  const radius = (size - stroke) / 2
  const circumference = 2 * Math.PI * radius
  const offset = circumference - (progress / 100) * circumference

  // Determine current step
  const step = statusMsg.toLowerCase().includes('translat') ? 2
    : statusMsg.toLowerCase().includes('transcri') ? 1
    : 0

  return (
    <div className="px-4 pb-3">
      {/* Main progress card */}
      <div className="p-3 rounded-xl" style={{ background: 'var(--surface-2)', border: '1px solid var(--border)' }}>

        {/* Top row: ring + status */}
        <div className="flex items-center gap-3 mb-3">
          {/* Circular progress ring */}
          <div className="relative shrink-0" style={{ width: size, height: size }}>
            <svg width={size} height={size} className="rotate-[-90deg]">
              {/* Background circle */}
              <circle
                cx={size / 2} cy={size / 2} r={radius}
                fill="none"
                stroke="var(--border-strong)"
                strokeWidth={stroke}
              />
              {/* Progress arc */}
              <circle
                cx={size / 2} cy={size / 2} r={radius}
                fill="none"
                stroke="var(--accent)"
                strokeWidth={stroke}
                strokeLinecap="round"
                strokeDasharray={circumference}
                strokeDashoffset={offset}
                style={{ transition: 'stroke-dashoffset 0.4s ease' }}
              />
            </svg>
            {/* Center text */}
            <span className="absolute inset-0 flex items-center justify-center text-xs font-semibold"
                  style={{ color: 'var(--accent)' }}>
              {Math.round(progress)}%
            </span>
          </div>

          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium truncate" style={{ color: 'var(--text-primary)' }}>
              {statusMsg || 'Processing…'}
            </p>
            {transcribing && (
              <p className="text-xs mt-0.5 italic" style={{ color: 'var(--accent)', transition: 'opacity 0.3s', opacity: 0.85 }}>
                {funMessage}
              </p>
            )}
            {segmentCount > 0 && (
              <p className="text-xs mt-0.5" style={{ color: 'var(--text-muted)' }}>
                {segmentCount} segment{segmentCount !== 1 ? 's' : ''} found
              </p>
            )}
          </div>
        </div>

        {/* Step indicators */}
        <div className="flex gap-1">
          {['Loading model', 'Transcribing', 'Translating'].map((label, i) => {
            const isActive = step === i && transcribing
            const isDone = step > i
            return (
              <div key={label} className="flex-1 flex flex-col items-center gap-1">
                <div className="w-full h-1 rounded-full overflow-hidden"
                     style={{ background: 'var(--border)' }}>
                  <div className="h-full rounded-full"
                       style={{
                         background: isDone ? 'var(--success)' : isActive ? 'var(--accent)' : 'transparent',
                         width: isDone ? '100%' : isActive ? '60%' : '0%',
                         transition: 'width 0.4s ease, background 0.2s',
                         animation: isActive ? 'pulse 1.5s ease-in-out infinite' : 'none',
                       }} />
                </div>
                <span className="text-xs" style={{
                  color: isDone ? 'var(--success)' : isActive ? 'var(--accent)' : 'var(--text-muted)',
                  fontWeight: isActive ? 600 : 400,
                }}>
                  {isDone ? <Icon name="check" size={10} /> : null} {label}
                </span>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
