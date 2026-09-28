import { useEffect, useState } from 'react'
import { api } from './api'
import { ErrorState, Loading } from './components/ui'
import ColdStart from './pages/ColdStart'
import Forecast from './pages/Forecast'
import Overview from './pages/Overview'
import Segments from './pages/Segments'
import { useApi } from './useApi'

const TABS = ['Overview', 'Forecast', 'Segments', 'Cold start'] as const
type Tab = (typeof TABS)[number]

function useTheme() {
  const [theme, setTheme] = useState<'light' | 'dark'>(
    () =>
      (localStorage.getItem('theme') as 'light' | 'dark' | null) ??
      (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'),
  )
  useEffect(() => {
    document.documentElement.dataset.theme = theme
    localStorage.setItem('theme', theme)
  }, [theme])
  return [theme, () => setTheme(theme === 'dark' ? 'light' : 'dark')] as const
}

export default function App() {
  const [tab, setTab] = useState<Tab>(() => (TABS.find((t) => `#${slug(t)}` === location.hash) ?? 'Overview'))
  const [theme, toggleTheme] = useTheme()
  const overview = useApi(api.overview)

  useEffect(() => {
    history.replaceState(null, '', `#${slug(tab)}`)
  }, [tab])

  return (
    <div className="min-h-screen">
      <header className="border-b border-line bg-surface">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-4 px-6 py-4">
          <div>
            <h1 className="text-lg font-semibold text-ink">SalesML</h1>
            <p className="text-sm text-ink-2">E-commerce forecasting & customer segmentation · Olist, Brazil 2016-2018</p>
          </div>
          <div className="flex items-center gap-4 text-xs text-muted">
            {overview.data && <span>models trained {overview.data.metadata.trained_at.replace('T', ' ').slice(0, 16)} UTC</span>}
            <button
              type="button"
              onClick={toggleTheme}
              className="rounded-lg border border-line px-2.5 py-1 text-ink-2 hover:bg-grid"
              aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
            >
              {theme === 'dark' ? 'Light' : 'Dark'} mode
            </button>
          </div>
        </div>
        <nav className="mx-auto flex max-w-7xl gap-1 overflow-x-auto px-6" aria-label="Sections">
          {TABS.map((t) => (
            <button
              key={t}
              type="button"
              onClick={() => setTab(t)}
              aria-current={t === tab ? 'page' : undefined}
              className={`border-b-2 px-3 py-2 text-sm font-medium whitespace-nowrap transition-colors ${
                t === tab ? 'border-s1 text-ink' : 'border-transparent text-ink-2 hover:text-ink'
              }`}
            >
              {t}
            </button>
          ))}
        </nav>
      </header>

      <main className="mx-auto max-w-7xl px-6 py-6">
        {overview.error && !overview.data ? (
          <ErrorState message={overview.error} />
        ) : !overview.data ? (
          <Loading />
        ) : (
          <>
            {tab === 'Overview' && <Overview data={overview.data} />}
            {tab === 'Forecast' && <Forecast />}
            {tab === 'Segments' && <Segments />}
            {tab === 'Cold start' && <ColdStart dataAov={overview.data.kpis.average_order_value} />}
          </>
        )}
      </main>
    </div>
  )
}

const slug = (t: string) => t.toLowerCase().replace(/\s+/g, '-')
