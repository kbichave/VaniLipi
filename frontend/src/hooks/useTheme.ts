import { useEffect, useState } from 'react'
import type { Theme } from '../types'

function resolveTheme(theme: Theme): 'light' | 'dark' {
  if (theme === 'system') {
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
  }
  return theme
}

function readStoredTheme(): Theme {
  try {
    return (localStorage.getItem('vl_theme') as Theme | null) ?? 'light'
  } catch {
    return 'light'
  }
}

export function useTheme() {
  const [theme, setThemeState] = useState<Theme>(readStoredTheme)

  useEffect(() => {
    const actual = resolveTheme(theme)
    document.documentElement.setAttribute('data-theme', actual)
    try { localStorage.setItem('vl_theme', theme) } catch { /* ignore */ }
  }, [theme])

  function setTheme(t: Theme) {
    setThemeState(t)
  }

  function cycleTheme() {
    setThemeState(t => {
      if (t === 'light') return 'dark'
      if (t === 'dark') return 'system'
      return 'light'
    })
  }

  return { theme, setTheme, cycleTheme }
}
