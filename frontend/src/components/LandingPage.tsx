import { Icon, type IconName } from './Icon'

interface LandingPageProps {
  onGetStarted: () => void
  onTranslateText: () => void
  modelsReady: boolean
  onSetupModels: () => void
}

const FEATURES: { icon: IconName; title: string; desc: string }[] = [
  { icon: 'shield',    title: '100% Offline',  desc: 'Everything runs on your Mac. No data ever leaves your device.' },
  { icon: 'globe',     title: '14 Languages',  desc: 'Marathi, Hindi, Bengali, Tamil, Telugu, and 9 more Indian languages.' },
  { icon: 'file-text', title: 'Export Ready',   desc: 'SRT, VTT, TXT, DOCX, PDF, and JSON formats.' },
]

export function LandingPage({ onGetStarted, onTranslateText, modelsReady, onSetupModels }: LandingPageProps) {
  return (
    <div className="flex-1 overflow-y-auto" style={{ background: 'var(--surface-0)' }}>
      <div style={{
        minHeight: '100%',
        display: 'grid',
        placeItems: 'center',
        position: 'relative',
      }}>

        {/* Warm ambient background glow */}
        <div className="absolute inset-0 pointer-events-none" style={{ overflow: 'hidden' }}>
          <div className="absolute" style={{
            top: '50%', left: '50%', transform: 'translate(-50%, -50%)',
            width: 800, height: 600, borderRadius: '50%',
            background: 'radial-gradient(circle, var(--accent-muted) 0%, transparent 60%)',
            opacity: 0.7,
          }} />
        </div>

        <div className="relative flex flex-col items-center px-8 py-10" style={{ maxWidth: 740, width: '100%' }}>

          {/* Hero */}
          <img src="/logo.png" alt="VaniLipi" style={{ width: 140, height: 'auto', marginBottom: 14 }} />
          <h1 className="font-indic font-bold mb-0" style={{ fontSize: 32, color: 'var(--text-primary)' }}>
            वाणीलिपी
          </h1>
          <span className="font-bold tracking-tight mb-1"
                style={{ fontSize: 14, color: 'var(--text-muted)', letterSpacing: '0.15em', textTransform: 'uppercase' }}>
            VaniLipi
          </span>
          <div className="mb-3 mt-2" style={{ width: 32, height: 3, borderRadius: 2, background: 'var(--accent)' }} />
          <p className="text-base text-center mb-8" style={{ color: 'var(--text-secondary)', maxWidth: 440, lineHeight: 1.8 }}>
            Your voice, your language, your words — <br/>
            <span style={{ color: 'var(--accent)', fontWeight: 600 }}>transcribed and translated</span> in seconds.
          </p>

          {/* CTA buttons */}
          {modelsReady ? (
            <div className="flex flex-col items-center gap-3 mb-12">
              <div className="flex gap-3">
                <button className="btn btn-primary" style={{ padding: '12px 28px', fontSize: 15 }} onClick={onGetStarted}>
                  <Icon name="mic" size={18} /> Transcribe Audio
                </button>
                <button className="btn btn-secondary" style={{ padding: '12px 28px', fontSize: 15, borderColor: 'var(--accent)', color: 'var(--accent)' }} onClick={onTranslateText}>
                  <Icon name="globe" size={18} /> Translate Text
                </button>
              </div>
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                Upload audio to transcribe + translate, or paste text for translation only
              </p>
            </div>
          ) : (
            <div className="flex flex-col items-center gap-2 mb-12">
              <button className="btn btn-primary" style={{ padding: '12px 28px', fontSize: 15 }} onClick={onSetupModels}>
                Download Models to Get Started
              </button>
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                One-time download · models are cached locally
              </p>
            </div>
          )}

          {/* Features row */}
          <div className="w-full flex gap-3">
            {FEATURES.map(f => (
              <div key={f.title} className="flex-1 flex items-center gap-3 p-4 rounded-xl"
                   style={{ background: 'var(--surface-0)', border: '1px solid var(--border)', boxShadow: 'var(--shadow-md)' }}>
                <div className="shrink-0 w-9 h-9 rounded-lg flex items-center justify-center"
                     style={{ background: 'var(--accent-subtle)', color: 'var(--accent)' }}>
                  <Icon name={f.icon} size={16} />
                </div>
                <div>
                  <p className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>{f.title}</p>
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{f.desc}</p>
                </div>
              </div>
            ))}
          </div>

          {/* PERSONALIZATION: Remove before publishing to GitHub */}
          <p className="mt-8 text-xs" style={{ color: 'var(--accent)', opacity: 0.5 }}>
            crafted with ♥ for Aishu — because every voice deserves a script
          </p>

        </div>
      </div>
    </div>
  )
}
