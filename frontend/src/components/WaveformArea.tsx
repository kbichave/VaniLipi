import { useEffect, useRef, useState } from 'react'
import WaveSurfer from 'wavesurfer.js'
import { fmt } from '../constants'
import { Icon } from './Icon'

const RATES = [0.5, 0.75, 1, 1.25, 1.5, 2]

interface SeekTarget {
  time: number
  version: number
}

interface WaveformAreaProps {
  audioUrl: string | null
  seekTo: SeekTarget | null
  playbackRate: number
  onRateChange: (r: number) => void
  onTimeUpdate?: (t: number) => void
  onDurationChange?: (d: number) => void
  onPlayStateChange?: (playing: boolean) => void
  onSeekReady?: (seekFn: (t: number) => void) => void
  onPlayPauseReady?: (fn: () => void) => void
}

/** Resolve a CSS variable to a hex color for canvas rendering. */
function resolveColor(varName: string, fallback: string): string {
  if (typeof document === 'undefined') return fallback
  const value = getComputedStyle(document.documentElement).getPropertyValue(varName).trim()
  return value || fallback
}

export function WaveformArea({
  audioUrl,
  seekTo,
  playbackRate,
  onRateChange,
  onTimeUpdate,
  onDurationChange,
  onPlayStateChange,
  onSeekReady,
  onPlayPauseReady,
}: WaveformAreaProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const wsRef = useRef<WaveSurfer | null>(null)
  const [wsReady, setWsReady] = useState(false)
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const lastSeekRef = useRef<number | null>(null)

  // Init wavesurfer once on mount
  useEffect(() => {
    if (!containerRef.current) return

    const ws = WaveSurfer.create({
      container: containerRef.current,
      waveColor: resolveColor('--wave-base', '#D6D3D1'),
      progressColor: resolveColor('--wave-progress', '#D97706'),
      cursorColor: resolveColor('--wave-progress', '#D97706'),
      cursorWidth: 2,
      barWidth: 2,
      barGap: 1,
      barRadius: 2,
      height: 56,
      normalize: true,
      interact: true,
      hideScrollbar: true,
    })

    ws.on('ready', (dur: number) => {
      setWsReady(true)
      setDuration(dur)
      onDurationChange?.(dur)
    })
    ws.on('timeupdate', (t: number) => {
      setCurrentTime(t)
      onTimeUpdate?.(t)
    })
    ws.on('play', () => { setIsPlaying(true); onPlayStateChange?.(true) })
    ws.on('pause', () => { setIsPlaying(false); onPlayStateChange?.(false) })
    ws.on('finish', () => { setIsPlaying(false); onPlayStateChange?.(false) })

    wsRef.current = ws

    onSeekReady?.((t: number) => {
      if (ws.getDuration() > 0) ws.seekTo(t / ws.getDuration())
    })
    onPlayPauseReady?.(() => ws.playPause())

    return () => { ws.destroy(); wsRef.current = null; setWsReady(false) }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Load new audio
  useEffect(() => {
    if (!wsRef.current || !audioUrl) return
    setWsReady(false)
    setCurrentTime(0)
    setDuration(0)
    wsRef.current.load(audioUrl)
  }, [audioUrl])

  // Sync playback rate
  useEffect(() => {
    if (!wsRef.current) return
    wsRef.current.setPlaybackRate(playbackRate, true)
  }, [playbackRate])

  // External seek from segment click
  useEffect(() => {
    if (!seekTo || !wsRef.current || !wsReady) return
    if (seekTo.version === lastSeekRef.current) return
    lastSeekRef.current = seekTo.version
    const dur = wsRef.current.getDuration()
    if (dur > 0) wsRef.current.seekTo(seekTo.time / dur)
  }, [seekTo, wsReady])

  function handlePlayPause() {
    if (!wsRef.current || !wsReady) return
    wsRef.current.playPause()
  }

  return (
    <div style={{ padding: '12px 16px', background: 'var(--surface-2)', borderBottom: '1px solid var(--border)' }}>
      {/* Section label */}
      <div className="text-xs font-medium uppercase tracking-wider mb-3"
           style={{ color: 'var(--text-muted)', letterSpacing: '0.08em' }}>
        Player
      </div>

      {/* Controls row */}
      <div className="flex items-center gap-3 mb-3">
        {/* Play button */}
        <button
          className="shrink-0 flex items-center justify-center rounded-full transition-all cursor-pointer"
          style={{
            width: 36,
            height: 36,
            background: isPlaying ? 'var(--accent)' : 'transparent',
            color: isPlaying ? '#fff' : 'var(--text-secondary)',
            border: isPlaying ? '2px solid var(--accent)' : '2px solid var(--border-strong)',
          }}
          onClick={handlePlayPause}
          disabled={!wsReady}
          title="Play/Pause (Space)"
        >
          <Icon name={isPlaying ? 'pause' : 'play'} size={14} />
        </button>

        {/* Time */}
        <span className="text-sm font-mono" style={{ color: 'var(--text-muted)', minWidth: 90, fontVariantNumeric: 'tabular-nums' }}>
          {fmt(currentTime)} / {fmt(duration)}
        </span>

        <div className="flex-1" />

        {/* Speed segmented control */}
        <div className="speed-control">
          {RATES.map(r => (
            <button
              key={r}
              className={playbackRate === r ? 'active' : ''}
              onClick={() => onRateChange(r)}
              title={`${r}x speed`}
            >
              {r}x
            </button>
          ))}
        </div>
      </div>

      {/* Waveform */}
      <div ref={containerRef} style={{ minHeight: 56, cursor: wsReady ? 'pointer' : 'default' }} />

      {!audioUrl && (
        <p className="text-xs text-center mt-2" style={{ color: 'var(--text-muted)' }}>
          Load an audio file to see the waveform
        </p>
      )}
    </div>
  )
}
