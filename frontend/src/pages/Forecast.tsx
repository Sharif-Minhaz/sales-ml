import { useState } from 'react'
import {
  Area,
  Bar,
  BarChart,
  CartesianGrid,
  ComposedChart,
  Line,
  LineChart,
  Rectangle,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { api, type LeaderboardRow } from '../api'
import { ChartTooltip } from '../components/chart'
import { axisProps, gridProps } from '../components/chartStyle'
import { ChartCard, ErrorState, Insight, Loading, Segmented, StatTile, StatusBadge } from '../components/ui'
import { date, money, moneyShort, monthLabel, pct } from '../format'
import { useApi } from '../useApi'

interface Point {
  week_start: string
  actual?: number
  forecast?: number
  band?: [number, number]
}

const SERIES = ['var(--series-1)', 'var(--series-2)', 'var(--series-3)']

interface BarShapeProps {
  x: number
  y: number
  width: number
  height: number
  payload: LeaderboardRow
}

/** Bar + bootstrap-CI whisker. The bar starts at value 0, so pixels-per-unit = width / mae. */
function BarWithInterval({ x, y, width, height, payload, selected }: BarShapeProps & { selected: string }) {
  const scale = width / payload.mae
  const lo = x + payload.mae_ci_low * scale
  const hi = x + payload.mae_ci_high * scale
  const cy = y + height / 2
  return (
    <g>
      <Rectangle x={x} y={y} width={width} height={height} radius={[0, 4, 4, 0]} fill={payload.key === selected ? 'var(--series-1)' : 'var(--neutral)'} />
      <g stroke="var(--ink-2)" strokeWidth={1.5}>
        <line x1={lo} x2={hi} y1={cy} y2={cy} />
        <line x1={lo} x2={lo} y1={cy - 5} y2={cy + 5} />
        <line x1={hi} x2={hi} y1={cy - 5} y2={cy + 5} />
      </g>
    </g>
  )
}

export default function Forecast() {
  const [horizon, setHorizon] = useState(12)
  const { data, error, loading } = useApi(api.forecast, horizon)

  if (error && !data) return <ErrorState message={error} />
  if (!data) return <Loading />

  const last = data.history[data.history.length - 1]
  // Join the dashed forecast line to the last actual so there is no visual gap.
  const points: Point[] = [
    ...data.history.map((h) => ({ week_start: h.week_start, actual: h.revenue })),
    ...data.forecast.map((f) => ({ week_start: f.week_start, forecast: f.forecast, band: [f.lo, f.hi] as [number, number] })),
  ]
  points[data.history.length - 1] = { ...points[data.history.length - 1], forecast: last.revenue, band: [last.revenue, last.revenue] }

  const naive = data.leaderboard.find((r) => r.key === 'naive')
  const leaderboard = data.leaderboard
  const ciMax = Math.ceil(Math.max(...leaderboard.map((r) => r.mae_ci_high)) / 10000) * 10000

  // Error-by-horizon: the chosen model, the naive baseline, and the next best model - max 3 validated slots.
  const compareKeys = [...new Set([data.model.key, 'naive', ...data.leaderboard.map((r) => r.key)])].slice(0, 3)
  const labelOf = (key: string) => data.leaderboard.find((r) => r.key === key)?.label ?? key

  const coverageOk = data.coverage >= data.level - 0.05
  const change = (data.summary.next_4_weeks / data.summary.last_4_weeks - 1) * 100

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <StatTile label="Selected model" value={<span className="text-lg">{data.model.label}</span>} sub="lowest walk-forward MAE" />
        <StatTile
          label="Next 4 weeks"
          value={moneyShort(data.summary.next_4_weeks)}
          sub={`${moneyShort(data.summary.next_4_weeks_lo)} - ${moneyShort(data.summary.next_4_weeks_hi)} (${pct(data.level * 100, 0)} band)`}
        />
        <StatTile label="Last 4 weeks (actual)" value={moneyShort(data.summary.last_4_weeks)} sub={`forecast change ${change >= 0 ? '+' : ''}${pct(change)}`} />
        <StatTile
          label="Interval coverage"
          value={pct(data.coverage * 100, 0)}
          sub={
            <StatusBadge status={coverageOk ? 'good' : 'warning'}>
              {coverageOk ? 'calibrated' : `below the ${pct(data.level * 100, 0)} target - bands are optimistic`}
            </StatusBadge>
          }
        />
      </div>

      <ChartCard<Point>
        title="Weekly revenue forecast"
        subtitle={`${data.model.label}, refit on all ${data.history.length} complete weeks. The shaded band is an empirical ${pct(data.level * 100, 0)} interval built from backtest errors at each horizon step.`}
        dimmed={loading}
        controls={
          <Segmented
            label="Forecast horizon"
            value={horizon}
            onChange={setHorizon}
            options={[4, 8, 12].map((h) => ({ value: h, label: `${h} wk` }))}
          />
        }
        legend={[
          { label: 'Actual', color: 'var(--series-1)', kind: 'line' },
          { label: 'Forecast', color: 'var(--series-2)', kind: 'dash' },
          { label: `${pct(data.level * 100, 0)} interval`, color: 'var(--series-2)', kind: 'band' },
        ]}
        columns={[
          { key: 'week_start', label: 'Week of', format: (v) => date(String(v)) },
          { key: 'actual', label: 'Actual', align: 'right', format: (v) => (v == null ? '' : money(Number(v))) },
          { key: 'forecast', label: 'Forecast', align: 'right', format: (v, r) => (v == null || r.actual != null ? '' : money(Number(v))) },
          { key: 'band', label: 'Interval', align: 'right', format: (v, r) => (!v || r.actual != null ? '' : `${moneyShort((v as [number, number])[0])} - ${moneyShort((v as [number, number])[1])}`) },
        ]}
        rows={points}
      >
        <ResponsiveContainer width="100%" height={340}>
          <ComposedChart data={points} margin={{ top: 16, right: 8, left: 8, bottom: 0 }} accessibilityLayer>
            <CartesianGrid {...gridProps} />
            <XAxis dataKey="week_start" tickFormatter={monthLabel} minTickGap={32} {...axisProps} />
            <YAxis tickFormatter={moneyShort} width={72} {...axisProps} axisLine={false} />
            <Tooltip
              cursor={{ stroke: 'var(--axis)' }}
              content={(p) => (
                <ChartTooltip<Point>
                  {...p}
                  title={(d) => `Week of ${date(d.week_start)}`}
                  rows={(d) =>
                    d.actual != null
                      ? [{ label: 'Actual', value: money(d.actual), color: 'var(--series-1)' }]
                      : [
                          { label: 'Forecast', value: money(d.forecast ?? 0), color: 'var(--series-2)' },
                          { label: 'Low', value: money(d.band?.[0] ?? 0) },
                          { label: 'High', value: money(d.band?.[1] ?? 0) },
                        ]
                  }
                />
              )}
            />
            <ReferenceLine
              x={last.week_start}
              stroke="var(--axis)"
              label={{ value: 'forecast →', position: 'insideTopRight', fill: 'var(--muted)', fontSize: 11 }}
            />
            <Area dataKey="band" stroke="none" fill="var(--series-2)" fillOpacity={0.18} isAnimationActive={false} />
            <Line dataKey="actual" stroke="var(--series-1)" strokeWidth={2} dot={false} activeDot={{ r: 4 }} isAnimationActive={false} />
            <Line dataKey="forecast" stroke="var(--series-2)" strokeWidth={2} strokeDasharray="5 4" dot={false} activeDot={{ r: 4 }} isAnimationActive={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </ChartCard>

      <div className="grid gap-6 lg:grid-cols-3">
        <div className="lg:col-span-2">
          <ChartCard<LeaderboardRow>
            title="Model leaderboard - walk-forward backtest"
            subtitle="Mean absolute error when each model forecasts 12 unseen weeks, repeated from 21 different start points. Whiskers: 90% bootstrap CI. Lower is better."
            columns={[
              { key: 'label', label: 'Model' },
              { key: 'mae', label: 'MAE', align: 'right', format: (v) => moneyShort(Number(v)) },
              { key: 'mae_ci_low', label: '90% CI', align: 'right', format: (_, r) => `${moneyShort(r.mae_ci_low)} - ${moneyShort(r.mae_ci_high)}` },
              { key: 'mape', label: 'MAPE', align: 'right', format: (v) => pct(Number(v)) },
              { key: 'bias', label: 'Bias', align: 'right', format: (v) => moneyShort(Number(v)) },
              { key: 'skill_vs_naive', label: 'vs naive', align: 'right', format: (v) => `${Number(v) >= 0 ? '+' : ''}${pct(Number(v))}` },
            ]}
            rows={data.leaderboard}
          >
            <ResponsiveContainer width="100%" height={leaderboard.length * 36 + 40}>
              <BarChart data={leaderboard} layout="vertical" barCategoryGap={8} margin={{ top: 16, right: 24, left: 8, bottom: 0 }} accessibilityLayer>
                <CartesianGrid stroke="var(--grid)" horizontal={false} />
                <XAxis type="number" domain={[0, ciMax]} tickFormatter={moneyShort} {...axisProps} />
                <YAxis type="category" dataKey="label" width={250} {...axisProps} axisLine={false} />
                <Tooltip
                  cursor={{ fill: 'var(--grid)', opacity: 0.5 }}
                  content={(p) => (
                    <ChartTooltip<LeaderboardRow>
                      {...p}
                      title={(d) => d.label}
                      rows={(d) => [
                        { label: 'MAE', value: money(d.mae) },
                        { label: '90% CI', value: `${moneyShort(d.mae_ci_low)} - ${moneyShort(d.mae_ci_high)}` },
                        { label: 'MAPE', value: pct(d.mape) },
                        { label: 'vs naive', value: `${d.skill_vs_naive >= 0 ? '+' : ''}${pct(d.skill_vs_naive)}` },
                      ]}
                    />
                  )}
                />
                {naive && (
                  <ReferenceLine
                    x={naive.mae}
                    stroke="var(--muted)"
                    label={{ value: 'naive baseline', position: 'top', fill: 'var(--muted)', fontSize: 11 }}
                  />
                )}
                <Bar
                  dataKey="mae"
                  isAnimationActive={false}
                  shape={(p: unknown) => <BarWithInterval {...(p as BarShapeProps)} selected={data.model.key} />}
                />
              </BarChart>
            </ResponsiveContainer>
          </ChartCard>
        </div>
        <div className="space-y-4">
          <Insight title="Honest evaluation, honest result">
            <p>
              v1 reported R² = 0.97 for its Random Forest - but scored on the <em>same months it trained on</em>. That
              measures memorisation, not forecasting.
            </p>
            <p>
              Here every model only sees the past and is scored on the future it never saw. The top models' CIs
              overlap heavily: <strong>none is clearly better than a 4-week average</strong>. Reporting that is the
              point - the fixes to data and evaluation mattered more than model choice.
            </p>
          </Insight>
          <Insight title="Why the ML growth models lose">
            <p>
              2017 grew fast; 2018 went flat. Models that learn growth rates extrapolate 2017's momentum into 2018 and
              over-forecast (large negative bias). Level-based models happen to suit a plateau.
            </p>
            <p>This is a regime change - the kind of risk a forecast owner has to flag, not hide.</p>
          </Insight>
        </div>
      </div>

      <ChartCard<{ step: number } & Record<string, number>>
        title="Error grows with the horizon"
        subtitle="Mean absolute error by how many weeks ahead the forecast is. This is why the interval band widens."
        legend={compareKeys.map((k, i) => ({ label: labelOf(k), color: SERIES[i], kind: 'line' as const }))}
        columns={[
          { key: 'step', label: 'Weeks ahead' },
          ...compareKeys.map((k) => ({ key: k, label: labelOf(k), align: 'right' as const, format: (v: unknown) => money(Number(v)) })),
        ]}
        rows={data.step_mae}
      >
        <ResponsiveContainer width="100%" height={260}>
          <LineChart data={data.step_mae} margin={{ top: 8, right: 8, left: 8, bottom: 0 }} accessibilityLayer>
            <CartesianGrid {...gridProps} />
            <XAxis dataKey="step" tickFormatter={(s) => `${s}w`} {...axisProps} />
            <YAxis tickFormatter={moneyShort} width={72} {...axisProps} axisLine={false} />
            <Tooltip
              cursor={{ stroke: 'var(--axis)' }}
              content={(p) => (
                <ChartTooltip<Record<string, number>>
                  {...p}
                  title={(d) => `${d.step} week${d.step > 1 ? 's' : ''} ahead`}
                  rows={(d) => compareKeys.map((k, i) => ({ label: labelOf(k), value: money(d[k]), color: SERIES[i] }))}
                />
              )}
            />
            {compareKeys.map((k, i) => (
              <Line key={k} dataKey={k} stroke={SERIES[i]} strokeWidth={2} dot={{ r: 3, strokeWidth: 0, fill: SERIES[i] }} isAnimationActive={false} />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </ChartCard>
    </div>
  )
}
