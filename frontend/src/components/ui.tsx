import { useState, type ReactNode } from 'react'

export function Card({
  title,
  subtitle,
  actions,
  children,
  className = '',
}: {
  title?: string
  subtitle?: ReactNode
  actions?: ReactNode
  children: ReactNode
  className?: string
}) {
  return (
    <section className={`rounded-xl border border-line bg-surface p-5 ${className}`}>
      {(title || actions) && (
        <header className="mb-4 flex flex-wrap items-start justify-between gap-3">
          <div>
            {title && <h2 className="text-base font-semibold text-ink">{title}</h2>}
            {subtitle && <p className="mt-1 max-w-3xl text-sm text-ink-2">{subtitle}</p>}
          </div>
          {actions}
        </header>
      )}
      {children}
    </section>
  )
}

export function StatTile({ label, value, sub }: { label: string; value: ReactNode; sub?: ReactNode }) {
  return (
    <div className="rounded-xl border border-line bg-surface px-4 py-3">
      <div className="text-xs font-medium tracking-wide text-muted uppercase">{label}</div>
      <div className="mt-1 text-2xl font-semibold text-ink">{value}</div>
      {sub && <div className="mt-1 text-xs text-ink-2">{sub}</div>}
    </div>
  )
}

type Status = 'good' | 'warning' | 'critical'
const STATUS_ICON: Record<Status, string> = { good: '✓', warning: '!', critical: '✕' }
const STATUS_BG: Record<Status, string> = { good: 'bg-good', warning: 'bg-warning', critical: 'bg-critical' }

/** Status is never color-only: always icon + label. */
export function StatusBadge({ status, children }: { status: Status; children: ReactNode }) {
  return (
    <span className="inline-flex items-center gap-1.5 text-sm text-ink-2">
      <span
        aria-hidden
        className={`inline-flex size-4 items-center justify-center rounded-full text-[10px] font-bold text-white ${STATUS_BG[status]}`}
      >
        {STATUS_ICON[status]}
      </span>
      {children}
    </span>
  )
}

/** "Learn" callouts explain the ML reasoning behind each view - the point of this project. */
export function Insight({ title, children }: { title: string; children: ReactNode }) {
  return (
    <aside className="rounded-xl border border-line bg-surface p-4 text-sm text-ink-2">
      <div className="mb-1 flex items-center gap-2 font-semibold text-ink">
        <span aria-hidden className="inline-flex size-5 items-center justify-center rounded-full border border-line text-xs">
          i
        </span>
        {title}
      </div>
      <div className="space-y-2 leading-relaxed">{children}</div>
    </aside>
  )
}

export interface Column<T> {
  key: keyof T & string
  label: string
  align?: 'left' | 'right'
  format?: (value: T[keyof T], row: T) => ReactNode
}

export function DataTable<T>({ columns, rows, scroll = false }: { columns: Column<T>[]; rows: T[]; scroll?: boolean }) {
  return (
    <div className={`overflow-auto rounded-lg border border-line ${scroll ? 'max-h-96' : ''}`}>
      <table className="w-full text-sm">
        <thead className="sticky top-0 bg-surface text-left text-xs text-muted uppercase">
          <tr>
            {columns.map((c) => (
              <th key={c.key} className={`border-b border-line px-3 py-2 font-medium ${c.align === 'right' ? 'text-right' : ''}`}>
                {c.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="tabular">
          {rows.map((row, i) => (
            <tr key={i} className="border-b border-line last:border-0">
              {columns.map((c) => (
                <td key={c.key} className={`px-3 py-2 text-ink-2 ${c.align === 'right' ? 'text-right' : ''}`}>
                  {c.format ? c.format(row[c.key], row) : String(row[c.key])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export interface LegendItem {
  label: string
  color: string
  kind?: 'line' | 'dash' | 'box' | 'band'
}

export function Legend({ items }: { items: LegendItem[] }) {
  return (
    <ul className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-ink-2">
      {items.map((item) => (
        <li key={item.label} className="flex items-center gap-1.5">
          <LegendSwatch {...item} />
          {item.label}
        </li>
      ))}
    </ul>
  )
}

function LegendSwatch({ color, kind = 'box' }: LegendItem) {
  if (kind === 'line' || kind === 'dash')
    return (
      <svg width="16" height="8" aria-hidden>
        <line x1="0" y1="4" x2="16" y2="4" stroke={color} strokeWidth="2" strokeDasharray={kind === 'dash' ? '4 3' : undefined} />
      </svg>
    )
  return (
    <span
      aria-hidden
      className="inline-block size-3 rounded-sm"
      style={{ background: color, opacity: kind === 'band' ? 0.25 : 1 }}
    />
  )
}

/** Chart with a table-view twin: every value stays readable without hover or color. */
export function ChartCard<T>({
  title,
  subtitle,
  legend,
  controls,
  columns,
  rows,
  children,
  dimmed = false,
}: {
  title: string
  subtitle?: ReactNode
  legend?: LegendItem[]
  controls?: ReactNode
  columns: Column<T>[]
  rows: T[]
  children: ReactNode
  dimmed?: boolean
}) {
  const [view, setView] = useState<'chart' | 'table'>('chart')
  return (
    <Card
      title={title}
      subtitle={subtitle}
      actions={
        <div className="flex items-center gap-3">
          {controls}
          <Segmented
            value={view}
            onChange={setView}
            options={[
              { value: 'chart', label: 'Chart' },
              { value: 'table', label: 'Table' },
            ]}
          />
        </div>
      }
    >
      {view === 'chart' ? (
        <div className={`transition-opacity ${dimmed ? 'opacity-50' : ''}`}>
          {legend && legend.length > 1 && (
            <div className="mb-3">
              <Legend items={legend} />
            </div>
          )}
          {children}
        </div>
      ) : (
        <DataTable columns={columns} rows={rows} scroll />
      )}
    </Card>
  )
}

export function Segmented<V extends string | number>({
  value,
  onChange,
  options,
  label,
}: {
  value: V
  onChange: (v: V) => void
  options: { value: V; label: string }[]
  label?: string
}) {
  return (
    <div role="radiogroup" aria-label={label} className="inline-flex rounded-lg border border-line p-0.5 text-xs">
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          role="radio"
          aria-checked={o.value === value}
          onClick={() => onChange(o.value)}
          className={`rounded-md px-2.5 py-1 font-medium transition-colors ${
            o.value === value ? 'bg-ink text-page' : 'text-ink-2 hover:bg-grid'
          }`}
        >
          {o.label}
        </button>
      ))}
    </div>
  )
}

export function ErrorState({ message }: { message: string }) {
  return (
    <Card title="Can't reach the model API">
      <p className="text-sm text-ink-2">{message}</p>
      <pre className="mt-3 rounded-lg bg-page p-3 text-xs text-ink-2">
        {`cd backend\nuv run fetch-data   # once\nuv run train        # builds artifacts/\nuv run serve        # starts the API on :5000`}
      </pre>
    </Card>
  )
}

export function Loading() {
  return <div className="py-24 text-center text-sm text-muted">Loading…</div>
}
