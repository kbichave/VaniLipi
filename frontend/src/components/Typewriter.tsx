import { useEffect, useRef, useState } from 'react'

interface TypewriterProps {
  text: string
  /** Characters revealed per tick */
  speed?: number
  /** Milliseconds between ticks */
  interval?: number
  /** Skip animation entirely — render full text immediately */
  skip?: boolean
  className?: string
  style?: React.CSSProperties
}

/**
 * Reveals text character-by-character like ChatGPT / Claude streaming.
 *
 * - Devanagari and other multi-byte scripts work correctly because we
 *   use Array.from() which splits on code points, not bytes.
 * - When `text` changes (e.g. translation arrives), the new text types
 *   in from the beginning.
 * - Once fully revealed, the blinking cursor disappears.
 */
export function Typewriter({
  text,
  speed = 3,
  interval = 20,
  skip = false,
  className,
  style,
}: TypewriterProps) {
  const chars = useRef<string[]>([])
  const [shown, setShown] = useState('')
  const [done, setDone] = useState(false)
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  useEffect(() => {
    // Clean up any running timer
    if (timerRef.current) {
      clearInterval(timerRef.current)
      timerRef.current = null
    }

    if (skip || !text) {
      setShown(text)
      setDone(true)
      return
    }

    // Split into code points (handles Devanagari, emoji, CJK correctly)
    chars.current = Array.from(text)
    let pos = 0
    setShown('')
    setDone(false)

    timerRef.current = setInterval(() => {
      pos += speed
      if (pos >= chars.current.length) {
        pos = chars.current.length
        setShown(text)
        setDone(true)
        if (timerRef.current) {
          clearInterval(timerRef.current)
          timerRef.current = null
        }
      } else {
        setShown(chars.current.slice(0, pos).join(''))
      }
    }, interval)

    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current)
        timerRef.current = null
      }
    }
  }, [text, speed, interval, skip])

  return (
    <span className={className} style={style}>
      {shown}
      {!done && <span className="typewriter-cursor">|</span>}
    </span>
  )
}
