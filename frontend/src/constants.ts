import type { Language, ModelOption } from './types'

export const LANGUAGES: Language[] = [
  { code: 'auto', name: 'Auto-detect',  script: null,          tier: 'recommended' },
  { code: 'mr',   name: 'Marathi',      script: 'Devanagari',  tier: 'recommended' },
  { code: 'hi',   name: 'Hindi',        script: 'Devanagari',  tier: 'recommended' },
  { code: 'bn',   name: 'Bengali',      script: 'Bengali',     tier: 'recommended' },
  { code: 'ta',   name: 'Tamil',        script: 'Tamil',       tier: 'recommended' },
  { code: 'te',   name: 'Telugu',       script: 'Telugu',      tier: 'recommended' },
  { code: 'gu',   name: 'Gujarati',     script: 'Gujarati',    tier: 'supported' },
  { code: 'kn',   name: 'Kannada',      script: 'Kannada',     tier: 'supported' },
  { code: 'ml',   name: 'Malayalam',    script: 'Malayalam',   tier: 'supported' },
  { code: 'pa',   name: 'Punjabi',      script: 'Gurmukhi',    tier: 'supported' },
  { code: 'ur',   name: 'Urdu',         script: 'Arabic',      tier: 'supported' },
  { code: 'ne',   name: 'Nepali',       script: 'Devanagari',  tier: 'supported' },
  { code: 'as',   name: 'Assamese',     script: 'Bengali',     tier: 'supported' },
  { code: 'sa',   name: 'Sanskrit',     script: 'Devanagari',  tier: 'supported' },
]

export const MODEL_OPTIONS: ModelOption[] = [
  { id: 'turbo', label: 'Best (large-v3-turbo)' },
  { id: 'small', label: 'Fast (small)' },
]

export function fmt(secs: number): string {
  const m = Math.floor(secs / 60).toString().padStart(2, '0')
  const s = Math.floor(secs % 60).toString().padStart(2, '0')
  return `${m}:${s}`
}
