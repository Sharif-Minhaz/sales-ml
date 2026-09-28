// Shared tooltip so every chart's hover layer looks the same.

export interface TooltipRow {
  label: string
  value: string
  color?: string
}

/**
 * Recharts passes { active, payload, label }; `rows` turns the hovered datum into labelled lines.
 * Values use ink colours - the small swatch carries series identity, never the text colour.
 */
export function ChartTooltip<T>({
  active,
  payload,
  title,
  rows,
}: {
  active?: boolean
  payload?: readonly { payload?: unknown }[]
  title: (datum: T) => string
  rows: (datum: T) => TooltipRow[]
}) {
  if (!active || !payload?.length || !payload[0].payload) return null
  const datum = payload[0].payload as T
  return (
    <div className="rounded-lg border border-line bg-surface px-3 py-2 text-xs shadow-lg">
      <div className="mb-1 font-semibold text-ink">{title(datum)}</div>
      <ul className="space-y-0.5 tabular">
        {rows(datum).map((r) => (
          <li key={r.label} className="flex items-center justify-between gap-4 text-ink-2">
            <span className="flex items-center gap-1.5">
              {r.color && <span className="inline-block size-2 rounded-full" style={{ background: r.color }} />}
              {r.label}
            </span>
            <span className="font-medium text-ink">{r.value}</span>
          </li>
        ))}
      </ul>
    </div>
  )
}
