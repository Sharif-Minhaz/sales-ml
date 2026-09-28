import { useEffect, useState } from 'react'
import { Area, CartesianGrid, ComposedChart, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { api, type ColdStart as ColdStartData, type ColdStartInput } from '../api'
import { ChartTooltip } from '../components/chart'
import { axisProps, gridProps } from '../components/chartStyle'
import { Card, ChartCard, DataTable, Insight, StatTile } from '../components/ui'
import { count, money, moneyShort } from '../format'

type Month = ColdStartData['months'][number] & { band: [number, number] }

const READINESS = [
  { history: 'None (pre-launch)', method: 'Funnel math with scenarios (this page)', why: 'No sales to learn from; borrow benchmarks.' },
  { history: '2-8 weeks', method: 'Daily/weekly run-rate + funnel', why: 'Real conversion & AOV replace the assumptions.' },
  { history: '3-12 months', method: 'Naive / moving average / ETS on weekly data', why: 'Enough points for trend, too few for seasonality.' },
  { history: '2+ years', method: 'Seasonal models, ML with lag & calendar features', why: 'Each season seen at least twice.' },
]

export default function ColdStart({ dataAov }: { dataAov: number }) {
  const [input, setInput] = useState<ColdStartInput>({
    monthly_visitors: 20000,
    conversion_rate_pct: 1.5,
    monthly_traffic_growth_pct: 8,
    months: 12,
  })
  const [result, setResult] = useState<ColdStartData | null>(null)
  const [error, setError] = useState<string | null>(null)

  // Debounced live recompute as the user types.
  useEffect(() => {
    const timer = setTimeout(() => {
      api
        .coldStart(input)
        .then((r) => {
          setResult(r)
          setError(null)
        })
        .catch((e: Error) => setError(e.message))
    }, 250)
    return () => clearTimeout(timer)
  }, [input])

  const set = (key: keyof ColdStartInput) => (e: React.ChangeEvent<HTMLInputElement>) =>
    setInput({ ...input, [key]: e.target.value === '' ? undefined : Number(e.target.value) })

  const months: Month[] = result?.months.map((m) => ({ ...m, band: [m.conservative, m.optimistic] })) ?? []

  return (
    <div className="space-y-6">
      <div className="grid gap-6 lg:grid-cols-3">
        <Card title="Your store's assumptions" subtitle="Revenue = visitors × conversion rate × average order value.">
          <div className="space-y-3">
            <Field label="Monthly visitors" value={input.monthly_visitors} onChange={set('monthly_visitors')} step={1000} />
            <Field label="Conversion rate (%)" value={input.conversion_rate_pct} onChange={set('conversion_rate_pct')} step={0.1} hint="e-commerce typically 1-3%" />
            <Field
              label="Average order value (R$)"
              value={input.average_order_value}
              onChange={set('average_order_value')}
              step={5}
              placeholder={dataAov.toFixed(0)}
              hint={`blank = learned from Olist data (${money(dataAov)})`}
            />
            <Field label="Monthly traffic growth (%)" value={input.monthly_traffic_growth_pct} onChange={set('monthly_traffic_growth_pct')} step={1} />
            <Field label="Months" value={input.months} onChange={set('months')} step={1} />
          </div>
          {error && <p className="mt-3 text-sm text-critical">{error}</p>}
        </Card>

        <div className="space-y-6 lg:col-span-2">
          {result && (
            <div className="grid grid-cols-3 gap-3">
              <StatTile label="Conservative" value={moneyShort(result.totals.conservative)} sub="conversion × 0.5" />
              <StatTile label="Base" value={moneyShort(result.totals.base)} sub={`first ${input.months} months`} />
              <StatTile label="Optimistic" value={moneyShort(result.totals.optimistic)} sub="conversion × 1.5" />
            </div>
          )}
          <ChartCard<Month>
            title="Projected monthly revenue"
            subtitle="The line is the base case; the band spans conservative to optimistic conversion."
            dimmed={!!error}
            legend={[
              { label: 'Base case', color: 'var(--series-1)', kind: 'line' },
              { label: 'Conservative - optimistic', color: 'var(--series-1)', kind: 'band' },
            ]}
            columns={[
              { key: 'month', label: 'Month' },
              { key: 'visitors', label: 'Visitors', align: 'right', format: (v) => count(Number(v)) },
              { key: 'conservative', label: 'Conservative', align: 'right', format: (v) => money(Number(v)) },
              { key: 'base', label: 'Base', align: 'right', format: (v) => money(Number(v)) },
              { key: 'optimistic', label: 'Optimistic', align: 'right', format: (v) => money(Number(v)) },
            ]}
            rows={months}
          >
            <ResponsiveContainer width="100%" height={300}>
              <ComposedChart data={months} margin={{ top: 8, right: 8, left: 8, bottom: 0 }} accessibilityLayer>
                <CartesianGrid {...gridProps} />
                <XAxis dataKey="month" tickFormatter={(m) => `M${m}`} {...axisProps} />
                <YAxis tickFormatter={moneyShort} width={72} {...axisProps} axisLine={false} />
                <Tooltip
                  cursor={{ stroke: 'var(--axis)' }}
                  content={(p) => (
                    <ChartTooltip<Month>
                      {...p}
                      title={(d) => `Month ${d.month} · ${count(d.visitors)} visitors`}
                      rows={(d) => [
                        { label: 'Optimistic', value: money(d.optimistic) },
                        { label: 'Base', value: money(d.base), color: 'var(--series-1)' },
                        { label: 'Conservative', value: money(d.conservative) },
                      ]}
                    />
                  )}
                />
                <Area dataKey="band" stroke="none" fill="var(--series-1)" fillOpacity={0.18} isAnimationActive={false} />
                <Line dataKey="base" stroke="var(--series-1)" strokeWidth={2} dot={{ r: 3, strokeWidth: 0, fill: 'var(--series-1)' }} isAnimationActive={false} />
              </ComposedChart>
            </ResponsiveContainer>
          </ChartCard>
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        <Card className="lg:col-span-2" title="Which method fits how much history you have">
          <DataTable
            columns={[
              { key: 'history', label: 'Sales history' },
              { key: 'method', label: 'Forecasting approach' },
              { key: 'why', label: 'Why' },
            ]}
            rows={READINESS}
          />
        </Card>
        <Insight title="Why a new store can't use the ML model">
          <p>
            Every model on the Forecast tab learns from past weeks of <em>this</em> store. With zero history there is
            nothing to learn - so we switch from learning to <strong>assumptions</strong>, and make the uncertainty
            explicit with scenarios.
          </p>
          <p>
            The one thing we can transfer is a benchmark: the average order value learned from a comparable business
            (Olist). Replace it with your own numbers as soon as real orders arrive.
          </p>
        </Insight>
      </div>
    </div>
  )
}

function Field({
  label,
  value,
  onChange,
  step,
  hint,
  placeholder,
}: {
  label: string
  value: number | undefined
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void
  step: number
  hint?: string
  placeholder?: string
}) {
  return (
    <label className="block text-sm">
      <span className="text-ink-2">{label}</span>
      <input
        type="number"
        step={step}
        value={value ?? ''}
        placeholder={placeholder}
        onChange={onChange}
        className="mt-1 w-full rounded-lg border border-line bg-page px-3 py-1.5 text-ink tabular focus:ring-2 focus:ring-s1 focus:outline-none"
      />
      {hint && <span className="mt-0.5 block text-xs text-muted">{hint}</span>}
    </label>
  )
}
