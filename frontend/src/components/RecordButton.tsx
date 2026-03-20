import { useRef, useState } from 'react'
import { fmt } from '../constants'
import { Icon } from './Icon'

interface RecordButtonProps {
  onRecordingComplete: (file: File) => void
}

export function RecordButton({ onRecordingComplete }: RecordButtonProps) {
  const [isRecording, setIsRecording] = useState(false)
  const [recordingTime, setRecordingTime] = useState(0)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  async function startRecording() {
    if (!navigator.mediaDevices?.getUserMedia) {
      alert('Microphone access not supported in this browser.')
      return
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const mr = new MediaRecorder(stream, { mimeType: 'audio/webm' })
      chunksRef.current = []

      mr.ondataavailable = e => { if (e.data.size > 0) chunksRef.current.push(e.data) }
      mr.onstop = () => {
        stream.getTracks().forEach(t => t.stop())
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' })
        const file = new File([blob], `recording_${Date.now()}.webm`, { type: 'audio/webm' })
        onRecordingComplete(file)
        setRecordingTime(0)
      }

      mr.start(250)
      mediaRecorderRef.current = mr
      setIsRecording(true)

      let t = 0
      timerRef.current = setInterval(() => { t++; setRecordingTime(t) }, 1000)
    } catch (e) {
      alert(`Microphone error: ${(e as Error).message}`)
    }
  }

  function stopRecording() {
    if (timerRef.current) clearInterval(timerRef.current)
    mediaRecorderRef.current?.stop()
    setIsRecording(false)
  }

  return (
    <button
      className="btn flex-1 justify-center"
      style={{
        background: isRecording ? '#EF4444' : 'transparent',
        color: isRecording ? '#fff' : 'var(--text-secondary)',
        borderColor: isRecording ? '#EF4444' : 'var(--border)',
        animation: isRecording ? 'pulse 1.5s infinite' : 'none',
      }}
      onClick={isRecording ? stopRecording : startRecording}
      title="Record microphone (⌘⇧R)"
    >
      <Icon name="mic" size={14} />
      {isRecording ? (
        <span className="font-mono text-xs">{`Stop ${fmt(recordingTime)}`}</span>
      ) : (
        'Record'
      )}
    </button>
  )
}
