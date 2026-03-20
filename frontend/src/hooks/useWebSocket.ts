import { useRef, useState } from 'react'
import { transcribeREST } from '../api'
import type { Segment } from '../types'

interface WsState {
  transcribing: boolean
  statusMsg: string
  progress: number
  segments: Segment[]
  detectedLang: string | null
}

interface UseWebSocketReturn extends WsState {
  connect: (fileId: string, language: string, model: string) => void
  disconnect: () => void
  setSegments: React.Dispatch<React.SetStateAction<Segment[]>>
  setStatusMsg: React.Dispatch<React.SetStateAction<string>>
  setDetectedLang: React.Dispatch<React.SetStateAction<string | null>>
  setProgress: React.Dispatch<React.SetStateAction<number>>
}

export function useWebSocket(onToast: (msg: string) => void, initialSegments: Segment[] = []): UseWebSocketReturn {
  const wsRef = useRef<WebSocket | null>(null)
  const [transcribing, setTranscribing] = useState(false)
  const [statusMsg, setStatusMsg] = useState('Ready.')
  const [progress, setProgress] = useState(0)
  const [segments, setSegments] = useState<Segment[]>(initialSegments)
  const [detectedLang, setDetectedLang] = useState<string | null>(null)

  function disconnect() {
    wsRef.current?.close()
    wsRef.current = null
  }

  async function fallbackToREST(fileId: string, language: string, model: string) {
    setTranscribing(true)
    setProgress(10)
    setStatusMsg('Transcribing (REST)…')
    try {
      const data = await transcribeREST(fileId, language, model)
      if (data.detected_language) setDetectedLang(data.detected_language)
      setSegments((data.segments ?? []).map((s, i) => ({
        id: `s${i}`,
        start: s.start ?? 0,
        end: s.end ?? 0,
        marathi: s.text ?? '',
        english: s.english ?? '',
      })))
      setProgress(100)
      setStatusMsg(`Done. ${(data.segments ?? []).length} segments.`)
      onToast('Transcription complete!')
    } catch (e) {
      setStatusMsg(`Transcription failed: ${(e as Error).message}`)
      onToast(`Error: ${(e as Error).message}`)
    } finally {
      setTranscribing(false)
      setTimeout(() => setProgress(0), 1500)
    }
  }

  function connect(fileId: string, language: string, model: string) {
    disconnect()
    setTranscribing(true)
    setProgress(5)
    setStatusMsg('Connecting…')
    setSegments([])
    setDetectedLang(null)

    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const ws = new WebSocket(`${proto}//${window.location.host}/api/stream/${fileId}`)
    wsRef.current = ws

    let segCount = 0
    // Track whether the pipeline finished cleanly so onclose knows to stay quiet.
    let completedNormally = false

    ws.onopen = () => {
      ws.send(JSON.stringify({ action: 'transcribe', language, model }))
    }

    ws.onmessage = (evt) => {
      let msg: { type: string; [key: string]: unknown }
      try { msg = JSON.parse(evt.data) } catch { return }

      switch (msg.type) {
        case 'status':
          setStatusMsg(msg.message as string)
          break
        case 'language_detected':
          setDetectedLang((msg.name ?? msg.code) as string)
          break
        case 'segment':
          segCount++
          setSegments(prev => {
            const id = `ws_${msg.id}`
            if (prev.find(s => s.id === id)) return prev
            return [...prev, {
              id,
              start: (msg.start as number) ?? 0,
              end: (msg.end as number) ?? 0,
              marathi: (msg.text as string) ?? '',
              english: '',
            }]
          })
          setProgress(p => Math.min(60, p + 4))
          break
        case 'translation':
          setSegments(prev => prev.map(s =>
            s.id === `ws_${msg.id}` ? { ...s, english: (msg.english as string) ?? '' } : s
          ))
          setProgress(p => Math.min(95, p + 2))
          break
        case 'warning':
          // Persist translation/pipeline warnings in status bar (don't just toast)
          setStatusMsg(msg.message as string)
          onToast(msg.message as string)
          break
        case 'complete':
          completedNormally = true
          setProgress(100)
          setStatusMsg(`Done. ${(msg.total_segments as number | undefined) ?? segCount} segments.`)
          onToast('Transcription complete!')
          setTranscribing(false)
          setTimeout(() => setProgress(0), 1500)
          ws.close()
          break
        case 'error':
          completedNormally = true  // handled — don't double-report in onclose
          setStatusMsg(`Error: ${msg.message as string}`)
          onToast(`Error: ${msg.message as string}`)
          setTranscribing(false)
          setTimeout(() => setProgress(0), 1500)
          ws.close()
          break
      }
    }

    ws.onerror = () => {
      if (!completedNormally) {
        setStatusMsg('WebSocket error. Retrying with REST fallback…')
        setTranscribing(false)
        fallbackToREST(fileId, language, model)
      }
    }

    ws.onclose = (evt) => {
      // Only report an unexpected close if the pipeline never signalled completion.
      // Normal close codes: 1000 (normal), 1005 (no code), 1001 (going away from server).
      if (!completedNormally && evt.code !== 1000 && evt.code !== 1005 && evt.code !== 1001) {
        setStatusMsg('Connection closed unexpectedly. Retrying with REST…')
        setTranscribing(false)
        setTimeout(() => setProgress(0), 1500)
        fallbackToREST(fileId, language, model)
      }
    }
  }

  return { transcribing, statusMsg, progress, segments, detectedLang, connect, disconnect, setSegments, setStatusMsg, setDetectedLang, setProgress }
}
