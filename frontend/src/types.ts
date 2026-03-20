export interface Segment {
  id: string
  start: number
  end: number
  marathi: string
  english: string
}

export interface Language {
  code: string
  name: string
  script: string | null
  tier: 'recommended' | 'supported'
}

export interface ModelOption {
  id: string
  label: string
}

export interface ModelInfo {
  id: string
  label: string
  downloaded: boolean
  required: boolean
}

export interface ModelStatus {
  ready: boolean
  models: ModelInfo[]
}

export interface Project {
  file_hash: string
  filename: string
  segment_count: number
  duration_seconds: number
  language: string
  detected_language?: string
  created_at?: string
  updated_at: string
}

export interface ProjectDetail extends Project {
  segments: Segment[]
}

export interface UploadResponse {
  file_id: string
  filename: string
  file_hash: string
  cached: boolean
}

export interface TranscribeResponse {
  segments: Array<{
    start: number
    end: number
    text: string
    english: string
  }>
  detected_language?: string
}

export interface RetranslateResponse {
  segment_id: unknown
  english: string
}

export type Theme = 'light' | 'dark' | 'system'
