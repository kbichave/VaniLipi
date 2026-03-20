import { useCallback, useRef, useState, useEffect } from 'react'
import { deleteProject, fetchProject, fetchProjects, fetchModelsStatus, retranslateSegment, uploadAudio } from './api'
import { AudioInputArea } from './components/AudioInputArea'
import { ExportMenu } from './components/ExportMenu'
import { Icon } from './components/Icon'
import { LandingPage } from './components/LandingPage'
import { ModelSetupModal } from './components/ModelSetupModal'
import { ProcessingStatus } from './components/ProcessingStatus'
import { TextTranslatePanel } from './components/TextTranslatePanel'
import { WelcomeSplash } from './components/WelcomeSplash'
import { RecentProjectsModal } from './components/RecentProjectsModal'
import { RecordButton } from './components/RecordButton'
import { SearchBar } from './components/SearchBar'
import { SegmentRow } from './components/SegmentRow'
import { SettingsModal } from './components/SettingsModal'
import { WaveformArea } from './components/WaveformArea'
import { LANGUAGES, fmt } from './constants'
import { useKeyboardShortcuts } from './hooks/useKeyboardShortcuts'
import { useTheme } from './hooks/useTheme'
import { useWebSocket } from './hooks/useWebSocket'
import type { ModelStatus, Project, Segment, Theme } from './types'

interface SeekTarget { time: number; version: number }

export function App() {
  const { theme, setTheme, cycleTheme } = useTheme()

  // Settings / modals
  const [showSettings, setShowSettings] = useState(false)
  const [showExport, setShowExport] = useState(false)
  const [model, setModel] = useState('turbo')

  // Language
  const [language, setLanguage] = useState('mr')

  // Audio state
  const [audioUrl, setAudioUrl] = useState<string | null>(null)
  const [playbackRate, setPlaybackRate] = useState(1)
  const waveSeekFnRef = useRef<((t: number) => void) | null>(null)
  const wavePlayPauseFnRef = useRef<(() => void) | null>(null)
  const [seekTo, setSeekTo] = useState<SeekTarget | null>(null)

  // Model status
  const [modelsReady, setModelsReady] = useState(true)
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null)
  const [showModelSetup, setShowModelSetup] = useState(false)

  // Recent projects
  const [recentProjects, setRecentProjects] = useState<Project[]>([])
  const [showRecentProjects, setShowRecentProjects] = useState(false)

  // File/upload
  const [fileId, setFileId] = useState<string | null>(null)
  const [uploading, setUploading] = useState(false)
  const [isDragging, setIsDragging] = useState(false)
  const [fileName, setFileName] = useState<string | null>(null)

  // Toast
  const [toast, setToast] = useState<string | null>(null)
  const toastTimer = useRef<ReturnType<typeof setTimeout> | null>(null)

  // Copy feedback
  const [copied, setCopied] = useState(false)

  // App mode: splash -> landing -> workspace | translate
  const [showSplash, setShowSplash] = useState(true)
  const [appMode, setAppMode] = useState<'landing' | 'workspace' | 'translate'>('landing')

  function showToast(msg: string) {
    if (toastTimer.current) clearTimeout(toastTimer.current)
    setToast(msg)
    toastTimer.current = setTimeout(() => setToast(null), 2500)
  }

  // WebSocket / transcription (no demo segments — landing page is the default)
  const ws = useWebSocket(showToast, [])

  // Editing
  const [activeSegId, setActiveSegId] = useState<string | null>(null)
  const [editingSegId, setEditingSegId] = useState<string | null>(null)
  const [editDraft, setEditDraft] = useState('')
  const [retranslating, setRetranslating] = useState(false)
  const [selectedSegIdx, setSelectedSegIdx] = useState(-1)

  // Search
  const [showSearch, setShowSearch] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')

  // ── Boot: check model status + load projects ────────────────────────────
  useEffect(() => {
    fetchModelsStatus()
      .then(data => {
        setModelStatus(data)
        setModelsReady(data.ready !== false)
        if (data.ready === false) setShowModelSetup(true)
      })
      .catch(() => {})
  }, [])

  useEffect(() => {
    fetchProjects().then(setRecentProjects).catch(() => {})
  }, [])

  // ── Wavesurfer callbacks ─────────────────────────────────────────────────
  function handleWsTimeUpdate(t: number) {
    const active = ws.segments.find(s => t >= s.start && t < s.end)
    setActiveSegId(active?.id ?? null)
  }

  function handlePlayPause() { wavePlayPauseFnRef.current?.() }

  function handleSeek(t: number) {
    setSeekTo(prev => ({ time: t, version: (prev?.version ?? 0) + 1 }))
  }

  // ── File upload ──────────────────────────────────────────────────────────
  async function handleFileSelected(file: File) {
    setAppMode('workspace')
    setUploading(true)
    setFileName(file.name)
    ws.setStatusMsg(`Uploading ${file.name}…`)
    ws.setProgress(0)
    try {
      const isVideo = /\.(mp4|mkv|webm|mov|avi|wmv)$/i.test(file.name)
      const data = await uploadAudio(file, (pct) => {
        ws.setProgress(pct)
        ws.setStatusMsg(
          pct < 100
            ? `Uploading ${file.name}… ${pct}%`
            : isVideo
              ? 'Extracting audio from video…'
              : 'Processing…',
        )
      })
      setFileId(data.file_id)
      const url = URL.createObjectURL(file)
      if (audioUrl) URL.revokeObjectURL(audioUrl)
      setAudioUrl(url)

      // If this file was previously transcribed, auto-load the cached result
      if (data.cached && data.file_hash) {
        try {
          const project = await fetchProject(data.file_hash)
          if (project.segments && project.segments.length > 0) {
            ws.setSegments(project.segments.map((s: Segment, i: number) => ({ ...s, id: `c${i}` })))
            ws.setDetectedLang(project.detected_language ?? null)
            ws.setStatusMsg(`Loaded cached transcript (${project.segments.length} segments)`)
            showToast('Loaded previous transcript!')
            setUploading(false)
            return
          }
        } catch {
          // Cache load failed — fall through to normal flow
        }
      }

      ws.setStatusMsg('Uploaded. Ready to transcribe.')
      ws.setSegments([])
      showToast(`Uploaded: ${file.name}`)
    } catch (e) {
      const msg = (e as Error).message
      ws.setStatusMsg(`Upload failed: ${msg}`)
      showToast(`Upload failed: ${msg}`)
    } finally {
      setUploading(false)
    }
  }

  // ── Drag and drop ────────────────────────────────────────────────────────
  function handleDragEnter(e: React.DragEvent) { e.preventDefault(); setIsDragging(true) }
  function handleDragLeave(e: React.DragEvent) { e.preventDefault(); setIsDragging(false) }
  function handleDrop(e: React.DragEvent) {
    e.preventDefault(); setIsDragging(false)
    const file = e.dataTransfer.files[0]
    if (file) handleFileSelected(file)
  }

  // ── Editing ──────────────────────────────────────────────────────────────
  const handleEdit = useCallback((id: string) => {
    const seg = ws.segments.find(s => s.id === id)
    if (!seg) return
    setEditingSegId(id)
    setEditDraft(seg.marathi)
  }, [ws.segments])

  function handleEditSave() {
    ws.setSegments(prev => prev.map(s => s.id === editingSegId ? { ...s, marathi: editDraft } : s))
    setEditingSegId(null)
  }

  const handleEditCancel = useCallback(() => setEditingSegId(null), [])

  async function handleRetranslate() {
    if (!editingSegId) return
    setRetranslating(true)
    try {
      const langCode = language === 'auto' ? 'mr' : language
      const data = await retranslateSegment(editingSegId, editDraft, langCode)
      ws.setSegments(prev => prev.map(s =>
        s.id === editingSegId ? { ...s, marathi: editDraft, english: data.english } : s
      ))
      setEditingSegId(null)
      showToast('Re-translated!')
    } catch (e) {
      showToast(`Re-translate failed: ${(e as Error).message}`)
    } finally {
      setRetranslating(false)
    }
  }

  // ── Copy all ─────────────────────────────────────────────────────────────
  function handleCopyAll() {
    const text = ws.segments.map(s => `${fmt(s.start)}\n${s.marathi}\n${s.english}`).join('\n\n')
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true)
      showToast('Copied to clipboard!')
      setTimeout(() => setCopied(false), 1500)
    })
  }

  // ── Load project ─────────────────────────────────────────────────────────
  async function handleLoadProject(hash: string) {
    const resp = await fetchProject(hash)
    if (resp.segments) {
      setAppMode('workspace')
      ws.setSegments(resp.segments.map((s: Segment, i: number) => ({ ...s, id: `p${i}` })))
      ws.setDetectedLang(resp.detected_language ?? null)
      ws.setStatusMsg(`Loaded: ${resp.filename}`)
      setFileName(resp.filename)
      showToast(`Loaded project: ${resp.filename}`)
    }
  }

  async function handleDeleteProject(hash: string) {
    await deleteProject(hash)
    setRecentProjects(prev => prev.filter(p => p.file_hash !== hash))
  }

  // ── Keyboard shortcuts ────────────────────────────────────────────────────
  useKeyboardShortcuts({
    segments: ws.segments,
    selectedSegIdx,
    editingSegId,
    onPlayPause: handlePlayPause,
    onToggleSearch: () => setShowSearch(s => !s),
    onExport: () => setShowExport(true),
    onRateChange: setPlaybackRate,
    onSegmentNav: setSelectedSegIdx,
    onEditStart: handleEdit,
    onEditCancel: handleEditCancel,
    onRetranslate: handleRetranslate,
    onDeselect: () => setSelectedSegIdx(-1),
  })

  // ── Search filter ─────────────────────────────────────────────────────────
  const filteredSegments = searchQuery
    ? ws.segments.filter(s =>
        s.marathi.toLowerCase().includes(searchQuery.toLowerCase()) ||
        s.english.toLowerCase().includes(searchQuery.toLowerCase())
      )
    : ws.segments

  // ── Render ────────────────────────────────────────────────────────────────

  if (showSplash) {
    return <WelcomeSplash onComplete={() => setShowSplash(false)} />
  }

  // Header component (reused in workspace and translate modes)
  const headerBar = (
    <header className="flex items-center gap-3 px-5 h-14 border-b shrink-0"
            style={{ borderColor: 'var(--border)', background: 'var(--surface-0)' }}>
      {/* Left: Logo + Language */}
      <div className="flex items-center gap-3">
        <button className="flex items-center gap-1.5 cursor-pointer"
                style={{ background: 'none', border: 'none', padding: 0 }}
                onClick={() => setAppMode('landing')}
                title="Home">
          <img src="/logo.png" alt="VaniLipi" style={{ width: 36, height: 'auto' }} />
          <span className="font-indic font-bold" style={{ fontSize: 17, color: 'var(--text-primary)' }}>
            वाणीलिपी
          </span>
        </button>
        <div className="flex items-center gap-1.5">
          <Icon name="globe" size={13} />
          <select value={language} onChange={e => setLanguage(e.target.value)}>
            <optgroup label="Recommended">
              {LANGUAGES.filter(l => l.tier === 'recommended').map(l => (
                <option key={l.code} value={l.code}>{l.name}</option>
              ))}
            </optgroup>
            <optgroup label="Supported">
              {LANGUAGES.filter(l => l.tier === 'supported').map(l => (
                <option key={l.code} value={l.code}>{l.name}</option>
              ))}
            </optgroup>
          </select>
        </div>
      </div>

      {/* Center: Filename */}
      <div className="flex-1 flex justify-center">
        {fileName && (
          <span className="text-xs font-medium px-3 py-1 rounded-full"
                style={{ color: 'var(--text-muted)', background: 'var(--surface-2)' }}>
            {fileName}
          </span>
        )}
      </div>

      {/* Right: Actions */}
      <div className="flex items-center gap-1">
        {recentProjects.length > 0 && (
          <button className="btn btn-ghost" style={{ padding: '6px 8px' }}
                  onClick={() => setShowRecentProjects(true)} title="Recent projects">
            <Icon name="folder" size={16} />
          </button>
        )}
        {!modelsReady && (
          <button className="btn btn-primary" style={{ padding: '5px 12px', fontSize: 12 }}
                  onClick={() => setShowModelSetup(true)}>
            Setup Models
          </button>
        )}
        <button className="btn btn-ghost" style={{ padding: '6px 8px' }}
                onClick={() => setShowSettings(true)} title="Settings">
          <Icon name="settings" size={16} />
        </button>
        <button className="btn btn-ghost" style={{ padding: '6px 8px' }}
                onClick={cycleTheme} title={`Theme: ${theme}`}>
          {theme === 'dark' ? <Icon name="moon" size={16} /> : <Icon name="sun" size={16} />}
        </button>
      </div>
    </header>
  )

  return (
    <>
      {/* ── Landing / Workspace / Translate ───────────────────────── */}
      {appMode === 'landing' ? (
        <>
          {headerBar}
          <LandingPage
            onGetStarted={() => setAppMode('workspace')}
            onTranslateText={() => setAppMode('translate')}
            modelsReady={modelsReady}
            onSetupModels={() => setShowModelSetup(true)}
          />
        </>
      ) : appMode === 'translate' ? (
        <>
          {headerBar}
          <TextTranslatePanel
            language={language}
            onBack={() => setAppMode('landing')}
          />
        </>
      ) : (
      <div className="flex h-screen overflow-hidden">

        {/* ── LEFT: Full-height sidebar ──────────────────────────── */}
        <aside className="sidebar" style={{ height: '100vh' }}>
          {/* Sidebar header with logo */}
          <div className="flex items-center gap-2 px-4 py-3 border-b" style={{ borderColor: 'var(--border)' }}>
            <button className="flex items-center gap-1.5 cursor-pointer"
                    style={{ background: 'none', border: 'none', padding: 0 }}
                    onClick={() => setAppMode('landing')}
                    title="Home">
              <img src="/logo.png" alt="VaniLipi" style={{ width: 30, height: 'auto' }} />
              <span className="font-indic font-bold" style={{ fontSize: 15, color: 'var(--text-primary)' }}>
                वाणीलिपी
              </span>
            </button>
            <div className="flex-1" />
            <select value={language} onChange={e => setLanguage(e.target.value)} style={{ fontSize: 12, padding: '4px 24px 4px 8px' }}>
              <optgroup label="Recommended">
                {LANGUAGES.filter(l => l.tier === 'recommended').map(l => (
                  <option key={l.code} value={l.code}>{l.name}</option>
                ))}
              </optgroup>
              <optgroup label="Supported">
                {LANGUAGES.filter(l => l.tier === 'supported').map(l => (
                  <option key={l.code} value={l.code}>{l.name}</option>
                ))}
              </optgroup>
            </select>
          </div>

          <div className="px-4 pt-4 pb-3">
            <AudioInputArea
              onFileSelected={handleFileSelected}
              isDragging={isDragging}
              onDragEnter={handleDragEnter}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              uploading={uploading}
            />
          </div>

          <div className="flex gap-2 px-4 pb-3">
            <button
              className="btn btn-primary flex-[2] justify-center"
              onClick={() => fileId && ws.connect(fileId, language, model)}
              disabled={!fileId || ws.transcribing || uploading}
              title="Start transcription"
            >
              {ws.transcribing
                ? <><Icon name="spinner" size={14} /> Transcribing…</>
                : 'Transcribe'}
            </button>
            <RecordButton onRecordingComplete={handleFileSelected} />
          </div>

          {/* Processing status widget */}
          <ProcessingStatus
            progress={ws.progress}
            statusMsg={ws.statusMsg}
            transcribing={ws.transcribing}
            segmentCount={ws.segments.length}
          />
          {!ws.transcribing && ws.progress === 0 && ws.statusMsg && ws.statusMsg !== 'Ready.' && (
            <div className="px-4 pb-3 text-xs" style={{ color: 'var(--text-secondary)' }}>
              {ws.statusMsg}
            </div>
          )}

          {/* Waveform */}
          <WaveformArea
            audioUrl={audioUrl}
            seekTo={seekTo}
            playbackRate={playbackRate}
            onRateChange={setPlaybackRate}
            onTimeUpdate={handleWsTimeUpdate}
            onDurationChange={() => {}}
            onPlayStateChange={() => {}}
            onSeekReady={fn => { waveSeekFnRef.current = fn }}
            onPlayPauseReady={fn => { wavePlayPauseFnRef.current = fn }}
          />

          <div className="flex-1" />

          {/* Sidebar footer */}
          <div className="flex items-center gap-2 px-4 py-2 text-xs border-t"
               style={{ borderColor: 'var(--border)', color: 'var(--text-muted)' }}>
            <span className="w-1.5 h-1.5 rounded-full shrink-0" style={{ background: 'var(--success)' }} />
            100% offline · Apple Silicon
          </div>
        </aside>

        {/* ── RIGHT: Header + Transcript ─────────────────────────── */}
        <div className="main-panel">
          {/* Top bar with filename + actions */}
          <div className="flex items-center gap-2 px-5 shrink-0 border-b"
               style={{ borderColor: 'var(--border)', minHeight: 48 }}>
            {fileName && (
              <span className="text-xs font-medium px-3 py-1 rounded-full"
                    style={{ color: 'var(--text-muted)', background: 'var(--surface-2)' }}>
                {fileName}
              </span>
            )}
            <div className="flex-1" />
            {recentProjects.length > 0 && (
              <button className="btn btn-ghost" style={{ padding: '6px 8px' }}
                      onClick={() => setShowRecentProjects(true)} title="Recent projects">
                <Icon name="folder" size={16} />
              </button>
            )}
            <button className="btn btn-ghost" style={{ padding: '6px 8px' }}
                    onClick={() => setShowSettings(true)} title="Settings">
              <Icon name="settings" size={16} />
            </button>
            <button className="btn btn-ghost" style={{ padding: '6px 8px' }}
                    onClick={cycleTheme} title={`Theme: ${theme}`}>
              {theme === 'dark' ? <Icon name="moon" size={16} /> : <Icon name="sun" size={16} />}
            </button>
          </div>

          {/* Transcript toolbar */}
          <div className="flex items-center gap-2 px-5 shrink-0 border-b"
               style={{ borderColor: 'var(--border)', minHeight: 44 }}>
            {ws.detectedLang && (
              <span className="text-xs font-medium px-2.5 py-1 rounded-full"
                    style={{ background: 'var(--accent-muted)', color: 'var(--accent)', border: '1px solid var(--accent)' }}>
                {ws.detectedLang}
              </span>
            )}
            {ws.segments.length > 0 && (
              <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                {filteredSegments.length !== ws.segments.length
                  ? `${filteredSegments.length} of ${ws.segments.length} segments`
                  : `${ws.segments.length} segment${ws.segments.length !== 1 ? 's' : ''}`}
              </span>
            )}
            {ws.segments.length > 0 && !fileId && (
              <span className="text-xs px-2 py-0.5 rounded"
                    style={{ background: 'var(--surface-2)', color: 'var(--text-muted)', border: '1px solid var(--border)' }}>
                demo
              </span>
            )}
            <div className="flex-1" />
            <button className="btn btn-ghost" style={{ padding: '5px 8px' }}
                    onClick={() => setShowSearch(s => !s)} title="Search (⌘F)">
              <Icon name="search" size={14} />
            </button>
            <button className="btn btn-secondary text-xs"
                    onClick={handleCopyAll}
                    disabled={ws.segments.length === 0 || !fileId}
                    title="Copy all text">
              <Icon name={copied ? 'check' : 'copy'} size={13} />
              {copied ? 'Copied!' : 'Copy'}
            </button>
            <button className="btn btn-secondary text-xs"
                    onClick={() => setShowExport(true)}
                    disabled={ws.segments.length === 0 || !fileId}
                    title="Export (⌘E)">
              <Icon name="download" size={13} /> Export
            </button>
          </div>

          {/* Search */}
          {showSearch && (
            <SearchBar
              query={searchQuery}
              onChange={setSearchQuery}
              onClose={() => { setShowSearch(false); setSearchQuery('') }}
            />
          )}

          {/* Transcript body */}
          <div className="flex-1 overflow-y-auto">
            {filteredSegments.length === 0 && !ws.transcribing ? (
              <div className="empty-state">
                {searchQuery ? (
                  <p className="text-base" style={{ color: 'var(--text-secondary)' }}>
                    No results for "{searchQuery}"
                  </p>
                ) : fileId ? (
                  <>
                    <div className="flex items-center justify-center mb-4"
                         style={{ width: 56, height: 56, borderRadius: 16, background: 'var(--accent-subtle)', border: '1px solid var(--accent-muted)' }}>
                      <Icon name="play" size={22} />
                    </div>
                    <p className="text-base font-medium mb-1.5" style={{ color: 'var(--text-primary)' }}>
                      Audio file ready
                    </p>
                    <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>
                      Click <strong>Transcribe</strong> to begin.
                    </p>
                  </>
                ) : (
                  <>
                    <div className="flex items-center justify-center mb-4"
                         style={{ width: 56, height: 56, borderRadius: 16, background: 'var(--surface-2)', border: '1px solid var(--border)' }}>
                      <Icon name="upload" size={22} />
                    </div>
                    <p className="text-base font-medium mb-1.5" style={{ color: 'var(--text-primary)' }}>
                      No transcript yet
                    </p>
                    <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>
                      Drop an audio file or use the upload area.
                    </p>
                    <p className="text-xs mt-2" style={{ color: 'var(--text-muted)' }}>
                      Supports MP3 · WAV · M4A · FLAC · OGG · MP4 · MKV
                    </p>
                  </>
                )}
              </div>
            ) : (
              <div className="segment-list">
                {/* Column header */}
                <div className="segment-list-header">
                  <span>Time</span>
                  <span>Source</span>
                  <span>English</span>
                </div>
                {filteredSegments.map((seg, idx) => (
                  <SegmentRow
                    key={seg.id}
                    seg={seg}
                    isActive={seg.id === activeSegId}
                    isSelected={idx === selectedSegIdx}
                    isEditing={seg.id === editingSegId}
                    onSeek={handleSeek}
                    onEdit={handleEdit}
                    onEditSave={handleEditSave}
                    onEditCancel={handleEditCancel}
                    onRetranslate={handleRetranslate}
                    editDraft={editDraft}
                    onEditChange={setEditDraft}
                    retranslating={retranslating}
                  />
                ))}
              </div>
            )}
            {ws.transcribing && (
              <div className="flex items-center justify-center gap-3 py-8 text-base"
                   style={{ color: 'var(--text-secondary)' }}>
                <Icon name="spinner" size={18} /> Transcribing audio…
              </div>
            )}
          </div>
        </div>
      </div>
      )}

      {/* ── Modals ──────────────────────────────────────────────── */}
      {showSettings && (
        <SettingsModal
          onClose={() => setShowSettings(false)}
          theme={theme as Theme}
          onThemeChange={setTheme}
          model={model}
          onModelChange={setModel}
        />
      )}
      {showExport && (
        <ExportMenu segments={ws.segments} fileId={fileId} onClose={() => setShowExport(false)} />
      )}
      {showModelSetup && modelStatus && (
        <ModelSetupModal modelStatus={modelStatus} onClose={() => setShowModelSetup(false)} />
      )}
      {showRecentProjects && (
        <RecentProjectsModal
          projects={recentProjects}
          onLoad={handleLoadProject}
          onDelete={handleDeleteProject}
          onClose={() => setShowRecentProjects(false)}
        />
      )}

      {/* ── Toast ───────────────────────────────────────────────── */}
      {toast && <div className="toast">{toast}</div>}
    </>
  )
}
