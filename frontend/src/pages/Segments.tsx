import { useState, type FormEvent } from 'react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  LabelList,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { api, type Segment, type SegmentPrediction, type Segments as SegmentsData } from '../api'
import { ChartTooltip } from '../components/chart'
import { axisProps, gridProps } from '../components/chartStyle'
import { Card, ChartCard, ErrorState, Insight, Loading, StatTile } from '../components/ui'
import { count, money, pct } from '../format'
import { useApi } from '../useApi'

type ScatterPoint = SegmentsData['scatter'][number]

export default function Segments() {
  const { data, error } = useApi(api.segments)
  if (error && !data) return <ErrorState message={error} />
  if (!data) return <Loading />

  const best = data.silhouette.find((s) => s.k === data.k)
  const top = data.segments[0] // sorted by revenue on the server
  const loyal = data.segments.find((s) => s.name.startsWith('Loyal'))

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <StatTile label="Customers segmented" value={count(data.n_customers)} />
        <StatTile label="Segments (k)" value={data.k} sub="chosen by silhouette, see below" />
        <StatTile label="Silhouette" value={best ? best.score.toFixed(3) : '-'} sub="-1 worst · 0 overlap · 1 perfect" />
        <StatTile label="Repeat buyers" value={loyal ? pct(loyal.customer_share) : '-'} sub={loyal ? `yet ${pct(loyal.revenue_share)} of revenue` : undefined} />
      </div>

      <Card>
        <p className="text-lg text-ink">
          <strong>{pct(top.customer_share, 0)}</strong> of customers - <em>{top.name.toLowerCase()}</em> - generated{' '}
          <strong>{pct(top.revenue_share, 0)}</strong> of revenue.
        </p>
        <p className="mt-1 text-sm text-ink-2">{top.action}</p>
      </Card>

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        {data.segments.map((s) => (
          <SegmentCard key={s.cluster} segment={s} />
        ))}
      </div>

      <ChartCard<Segment>
        title="Share of customers vs share of revenue"
        subtitle="Where a segment's revenue bar is longer than its customer bar, each customer is worth more than average."
        legend={[
          { label: 'Customers', color: 'var(--series-1)' },
          { label: 'Revenue', color: 'var(--series-2)' },
        ]}
        columns={[
          { key: 'name', label: 'Segment' },
          { key: 'customer_share', label: 'Customers', align: 'right', format: (v) => pct(Number(v)) },
          { key: 'revenue_share', label: 'Revenue', align: 'right', format: (v) => pct(Number(v)) },
        ]}
        rows={data.segments}
      >
        <ResponsiveContainer width="100%" height={data.segments.length * 70 + 30}>
          <BarChart data={data.segments} layout="vertical" barGap={2} barCategoryGap={14} margin={{ top: 0, right: 48, left: 8, bottom: 0 }} accessibilityLayer>
            <CartesianGrid stroke="var(--grid)" horizontal={false} />
            <XAxis type="number" tickFormatter={(v) => `${v}%`} {...axisProps} />
            <YAxis type="category" dataKey="name" width={230} {...axisProps} axisLine={false} />
            <Tooltip
              cursor={{ fill: 'var(--grid)', opacity: 0.5 }}
              content={(p) => (
                <ChartTooltip<Segment>
                  {...p}
                  title={(d) => d.name}
                  rows={(d) => [
                    { label: 'Customers', value: `${pct(d.customer_share)} (${count(d.customers)})`, color: 'var(--series-1)' },
                    { label: 'Revenue', value: `${pct(d.revenue_share)} (${money(d.revenue)})`, color: 'var(--series-2)' },
                  ]}
                />
              )}
            />
            <Bar dataKey="customer_share" fill="var(--series-1)" radius={[0, 4, 4, 0]} isAnimationActive={false}>
              <LabelList dataKey="customer_share" position="right" formatter={(v) => pct(Number(v), 0)} fill="var(--ink-2)" fontSize={11} />
            </Bar>
            <Bar dataKey="revenue_share" fill="var(--series-2)" radius={[0, 4, 4, 0]} isAnimationActive={false}>
              <LabelList dataKey="revenue_share" position="right" formatter={(v) => pct(Number(v), 0)} fill="var(--ink-2)" fontSize={11} />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </ChartCard>

      <SegmentMaps data={data} />

      <div className="grid gap-6 lg:grid-cols-3">
        <ChartCard<{ k: number; score: number }>
          title="Choosing k"
          subtitle="Silhouette score per number of clusters (computed on a 15k-customer sample)."
          columns={[
            { key: 'k', label: 'k' },
            { key: 'score', label: 'Silhouette', align: 'right', format: (v) => Number(v).toFixed(3) },
          ]}
          rows={data.silhouette}
        >
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={data.silhouette} margin={{ top: 16, right: 16, left: 0, bottom: 0 }} accessibilityLayer>
              <CartesianGrid {...gridProps} />
              <XAxis dataKey="k" {...axisProps} />
              <YAxis domain={['auto', 'auto']} tickFormatter={(v) => Number(v).toFixed(2)} width={48} {...axisProps} axisLine={false} />
              <Tooltip
                cursor={{ stroke: 'var(--axis)' }}
                content={(p) => (
                  <ChartTooltip<{ k: number; score: number }> {...p} title={(d) => `k = ${d.k}`} rows={(d) => [{ label: 'Silhouette', value: d.score.toFixed(3) }]} />
                )}
              />
              <ReferenceLine x={data.k} stroke="var(--axis)" label={{ value: 'chosen', position: 'top', fill: 'var(--muted)', fontSize: 11 }} />
              <Line dataKey="score" stroke="var(--series-1)" strokeWidth={2} dot={{ r: 4, strokeWidth: 0, fill: 'var(--series-1)' }} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </ChartCard>
        <Insight title="What changed from v1">
          <p>
            <strong>log(1+x) before scaling.</strong> Spend is heavily skewed; without it, a handful of big spenders
            decide where every centroid lands.
          </p>
          <p>
            <strong>k from data, not hardcoded.</strong> Scores within 0.02 of the best are a statistical tie; among
            ties the more granular k wins, because each segment must map to a different action.
          </p>
          <p>
            <strong>Names from medians.</strong> Means are dragged up by outliers - with means, every cluster looked
            "high value".
          </p>
        </Insight>
        <PredictForm />
      </div>
    </div>
  )
}

function SegmentCard({ segment: s }: { segment: Segment }) {
  return (
    <Card className="flex flex-col">
      <h3 className="font-semibold text-ink">{s.name}</h3>
      <dl className="mt-3 grid grid-cols-2 gap-y-2 text-sm tabular">
        <dt className="text-muted">Customers</dt>
        <dd className="text-right text-ink">{count(s.customers)}</dd>
        <dt className="text-muted">Median recency</dt>
        <dd className="text-right text-ink">{Math.round(s.recency_median)} days</dd>
        <dt className="text-muted">Avg orders</dt>
        <dd className="text-right text-ink">{s.frequency.toFixed(2)}</dd>
        <dt className="text-muted">Median spend</dt>
        <dd className="text-right text-ink">{money(s.monetary_median)}</dd>
      </dl>
      <p className="mt-3 border-t border-line pt-3 text-sm text-ink-2">{s.action}</p>
    </Card>
  )
}

/** Small multiples: one panel per segment, highlighted over the grey whole - avoids 4+ colours in one scatter. */
function SegmentMaps({ data }: { data: SegmentsData }) {
  const points = data.scatter.slice(0, 1200)
  return (
    <ChartCard<ScatterPoint>
      title="Where each segment lives"
      subtitle="Each dot is a sampled customer: days since last purchase vs total spend (log scale). One panel per segment."
      columns={[
        { key: 'segment', label: 'Segment' },
        { key: 'recency_days', label: 'Recency (days)', align: 'right' },
        { key: 'frequency', label: 'Orders', align: 'right' },
        { key: 'monetary', label: 'Spend', align: 'right', format: (v) => money(Number(v)) },
      ]}
      rows={points.slice(0, 200)}
    >
      <div className="grid gap-4 md:grid-cols-2">
        {data.segments.map((s) => {
          const inside = points.filter((p) => p.cluster === s.cluster)
          const outside = points.filter((p) => p.cluster !== s.cluster)
          return (
            <div key={s.cluster}>
              <div className="mb-1 text-sm font-medium text-ink">{s.name}</div>
              <ResponsiveContainer width="100%" height={220}>
                <ScatterChart margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
                  <CartesianGrid stroke="var(--grid)" />
                  <XAxis type="number" dataKey="recency_days" name="Recency" unit="d" domain={[0, 'auto']} {...axisProps} />
                  <YAxis
                    type="number"
                    dataKey="monetary"
                    name="Spend"
                    scale="log"
                    domain={[10, 10000]}
                    ticks={[10, 100, 1000, 10000]}
                    allowDataOverflow
                    tickFormatter={(v) => `R$${v >= 1000 ? `${v / 1000}k` : v}`}
                    width={56}
                    {...axisProps}
                    axisLine={false}
                  />
                  <Tooltip
                    cursor={false}
                    content={(p) => (
                      <ChartTooltip<ScatterPoint>
                        {...p}
                        title={(d) => d.segment}
                        rows={(d) => [
                          { label: 'Recency', value: `${d.recency_days} days` },
                          { label: 'Orders', value: String(d.frequency) },
                          { label: 'Spend', value: money(d.monetary) },
                        ]}
                      />
                    )}
                  />
                  <Scatter data={outside} fill="var(--neutral)" shape={dot} isAnimationActive={false} />
                  <Scatter data={inside} fill="var(--series-1)" shape={dot} isAnimationActive={false} />
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          )
        })}
      </div>
    </ChartCard>
  )
}

function dot(props: unknown) {
  const { cx, cy, fill } = props as { cx: number; cy: number; fill: string }
  return <circle cx={cx} cy={cy} r={2.5} fill={fill} fillOpacity={0.7} />
}

function PredictForm() {
  const [input, setInput] = useState({ recency_days: 30, frequency: 1, monetary: 250 })
  const [result, setResult] = useState<SegmentPrediction | null>(null)
  const [error, setError] = useState<string | null>(null)

  const submit = async (e: FormEvent) => {
    e.preventDefault()
    try {
      setResult(await api.predictSegment(input))
      setError(null)
    } catch (err) {
      setError((err as Error).message)
    }
  }

  const field = (key: keyof typeof input, label: string, step = 1) => (
    <label className="block text-sm">
      <span className="text-ink-2">{label}</span>
      <input
        type="number"
        step={step}
        value={input[key]}
        onChange={(e) => setInput({ ...input, [key]: Number(e.target.value) })}
        className="mt-1 w-full rounded-lg border border-line bg-page px-3 py-1.5 text-ink tabular focus:ring-2 focus:ring-s1 focus:outline-none"
      />
    </label>
  )

  const affinity = result ? Object.entries(result.affinity).sort((a, b) => b[1] - a[1]) : []

  return (
    <Card title="Classify a customer" subtitle="Live inference with the persisted joblib pipeline (log → scale → KMeans).">
      <form onSubmit={submit} className="space-y-3">
        <div className="grid grid-cols-3 gap-2">
          {field('recency_days', 'Days since')}
          {field('frequency', 'Orders')}
          {field('monetary', 'Spend R$', 10)}
        </div>
        <button type="submit" className="w-full rounded-lg bg-ink px-3 py-2 text-sm font-medium text-page hover:opacity-90">
          Predict segment
        </button>
      </form>
      {error && <p className="mt-3 text-sm text-critical">{error}</p>}
      {result && (
        <div className="mt-4 space-y-3">
          <div>
            <div className="font-semibold text-ink">{result.segment}</div>
            <p className="text-sm text-ink-2">{result.action}</p>
          </div>
          <ul className="space-y-1.5 text-xs" aria-label="Closeness to each segment">
            {affinity.map(([name, w]) => (
              <li key={name}>
                <div className="flex justify-between text-ink-2">
                  <span>{name}</span>
                  <span className="tabular">{pct(w * 100, 0)}</span>
                </div>
                <div className="mt-0.5 h-1.5 rounded-full bg-grid">
                  <div
                    className="h-1.5 rounded-full"
                    style={{ width: `${w * 100}%`, background: name === result.segment ? 'var(--series-1)' : 'var(--neutral)' }}
                  />
                </div>
              </li>
            ))}
          </ul>
        </div>
      )}
    </Card>
  )
}
