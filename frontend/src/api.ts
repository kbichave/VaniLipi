import type {
  ModelStatus,
  Project,
  ProjectDetail,
  RetranslateResponse,
  Segment,
  TranscribeResponse,
  UploadResponse,
} from './types'

export async function uploadAudio(
  file: File,
  onProgress?: (percent: number) => void,
): Promise<UploadResponse> {
  return new Promise((resolve, reject) => {
    const fd = new FormData()
    fd.append('file', file)
    const xhr = new XMLHttpRequest()
    xhr.open('POST', '/api/upload')

    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && onProgress) {
        onProgress(Math.round((e.loaded / e.total) * 100))
      }
    }

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(JSON.parse(xhr.responseText))
      } else {
        try {
          const err = JSON.parse(xhr.responseText)
          reject(new Error(err.detail || xhr.statusText))
        } catch {
          reject(new Error(xhr.statusText))
        }
      }
    }

    xhr.onerror = () => reject(new Error('Upload failed — connection error'))
    xhr.send(fd)
  })
}

export async function transcribeREST(
  fileId: string,
  language: string,
  model: string,
): Promise<TranscribeResponse> {
  const fd = new FormData()
  fd.append('file_id', fileId)
  fd.append('language', language === 'auto' ? '' : language)
  fd.append('model', model)
  const resp = await fetch('/api/transcribe', { method: 'POST', body: fd })
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: resp.statusText }))
    throw new Error(err.detail || resp.statusText)
  }
  return resp.json()
}

export async function retranslateSegment(
  segmentId: string,
  text: string,
  languageCode: string,
): Promise<RetranslateResponse> {
  const resp = await fetch('/api/retranslate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ segment_id: segmentId, text, language_code: languageCode }),
  })
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: resp.statusText }))
    throw new Error(err.detail || resp.statusText)
  }
  return resp.json()
}

export async function exportTranscript(
  format: string,
  segments: Segment[],
  fileId: string,
): Promise<Blob> {
  const resp = await fetch(`/api/export/${format}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ file_id: fileId, segments }),
  })
  if (!resp.ok) throw new Error(await resp.text())
  return resp.blob()
}

export async function fetchProjects(): Promise<Project[]> {
  const resp = await fetch('/api/projects')
  if (!resp.ok) throw new Error(resp.statusText)
  return resp.json()
}

export async function fetchProject(hash: string): Promise<ProjectDetail> {
  const resp = await fetch(`/api/projects/${hash}`)
  if (!resp.ok) throw new Error(resp.statusText)
  return resp.json()
}

export async function deleteProject(hash: string): Promise<void> {
  await fetch(`/api/projects/${hash}`, { method: 'DELETE' })
}

export async function fetchModelsStatus(): Promise<ModelStatus> {
  const resp = await fetch('/api/models/status')
  if (!resp.ok) throw new Error(resp.statusText)
  return resp.json()
}
