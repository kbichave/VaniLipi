import { useEffect, useState } from 'react'

interface WelcomeSplashProps {
  onComplete: () => void
}

export function WelcomeSplash({ onComplete }: WelcomeSplashProps) {
  const [phase, setPhase] = useState<'logo-enter' | 'logo' | 'logo-exit' | 'welcome-enter' | 'welcome' | 'welcome-exit'>('logo-enter')

  useEffect(() => {
    const timers = [
      setTimeout(() => setPhase('logo'), 400),
      setTimeout(() => setPhase('logo-exit'), 4200),
      setTimeout(() => setPhase('welcome-enter'), 4800),
      setTimeout(() => setPhase('welcome'), 5400),
      setTimeout(() => setPhase('welcome-exit'), 7800),
      setTimeout(() => onComplete(), 8500),
    ]
    return () => timers.forEach(clearTimeout)
  }, [onComplete])

  const showLogo = phase.startsWith('logo')
  const showWelcome = phase.startsWith('welcome')
  const entering = phase.endsWith('-enter')
  const exiting = phase.endsWith('-exit')

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center"
         style={{ background: 'var(--surface-0)' }}>

      {/* Warm ambient glow — amber radial in center */}
      <div className="absolute" style={{
        width: 600, height: 600, borderRadius: '50%',
        background: 'radial-gradient(circle, var(--accent-muted) 0%, transparent 70%)',
        opacity: 0.6,
        pointerEvents: 'none',
      }} />

      {/* Decorative corner accents */}
      <div className="absolute top-0 left-0" style={{
        width: 200, height: 200,
        background: 'radial-gradient(circle at top left, var(--accent-subtle) 0%, transparent 70%)',
        pointerEvents: 'none',
      }} />
      <div className="absolute bottom-0 right-0" style={{
        width: 200, height: 200,
        background: 'radial-gradient(circle at bottom right, var(--accent-subtle) 0%, transparent 70%)',
        pointerEvents: 'none',
      }} />

      {/* Phase 1: Logo + Name */}
      {showLogo && (
        <div className="flex flex-col items-center relative"
             style={{
               opacity: entering ? 0 : exiting ? 0 : 1,
               transform: entering ? 'scale(0.9) translateY(10px)' : 'scale(1) translateY(0)',
               transition: 'opacity 0.5s ease, transform 0.5s ease',
             }}>
          <img src="/logo.png" alt="" style={{
            width: 180, height: 'auto', marginBottom: 20,
          }} />
          <span className="font-indic font-bold"
                style={{ fontSize: 36, color: 'var(--text-primary)' }}>
            वाणीलिपी
          </span>
          <span className="font-bold tracking-tight mt-1"
                style={{ fontSize: 18, color: 'var(--text-muted)', letterSpacing: '0.15em', textTransform: 'uppercase' }}>
            VaniLipi
          </span>
          <p className="mt-3" style={{ fontSize: 14, color: 'var(--text-muted)' }}>
            Voice to Script
          </p>
          {/* Amber accent bar */}
          <div className="mt-4" style={{ width: 40, height: 3, borderRadius: 2, background: 'var(--accent)' }} />
        </div>
      )}

      {/* Phase 2: Welcome message */}
      {showWelcome && (
        <div className="flex flex-col items-center text-center px-8 relative"
             style={{
               maxWidth: 480,
               opacity: entering ? 0 : exiting ? 0 : 1,
               transform: entering ? 'translateY(16px)' : 'translateY(0)',
               transition: 'opacity 0.6s ease, transform 0.6s ease',
             }}>
          <div className="mb-5" style={{ width: 40, height: 3, borderRadius: 2, background: 'var(--accent)' }} />
          <p className="font-indic font-semibold mb-2"
             style={{ fontSize: 28, color: 'var(--text-primary)' }}>
            वाणीलिपीमध्ये स्वागत आहे
          </p>
          <p className="font-semibold mb-4"
             style={{ fontSize: 20, color: 'var(--text-secondary)' }}>
            Welcome to VaniLipi
          </p>
          <p style={{ fontSize: 15, color: 'var(--text-muted)', lineHeight: 1.7 }}>
            Transcribe and translate Indian language audio — entirely on your Mac.
          </p>
          {/* PERSONALIZATION: Remove before publishing to GitHub */}
          <p className="mt-5" style={{ fontSize: 13, color: 'var(--accent)', opacity: 0.6 }}>
            for Aishu ♥
          </p>
        </div>
      )}
    </div>
  )
}
