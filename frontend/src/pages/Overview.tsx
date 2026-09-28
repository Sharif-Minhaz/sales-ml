import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { CleaningStep, Overview as OverviewData } from '../api'
import { ChartTooltip } from '../components/chart'
import { axisProps, gridProps } from '../components/chartStyle'
import { Card, ChartCard, DataTable, Insight, StatTile } from '../components/ui'
import { count, date, money, moneyShort, monthLabel, pct } from '../format'

type Week = OverviewData['weekly'][number]

export default function Overview({ data }: { data: OverviewData }) {
  const { kpis, window, cleaning, weekly } = data

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 gap-3 md:grid-cols-5">
        <StatTile label="Revenue" value={moneyShort(kpis.revenue)} sub="price + freight, real sales only" />
        <StatTile label="Orders" value={count(kpis.orders)} />
        <StatTile label="Customers" value={count(kpis.customers)} sub="unique people, not order accounts" />
        <StatTile label="Avg order value" value={money(kpis.average_order_value)} sub={`median ${money(kpis.median_order_value)}`} />
        <StatTile label="Repeat customers" value={pct(kpis.repeat_customer_rate)} sub="bought more than once" />
      </div>

      <ChartCard<Week>
        title="Weekly revenue"
        subtitle={
          <>
            Grey weeks are incomplete data at the edges of the export and are <strong>excluded from training</strong>.
            The complete window ({date(window.start)} → {date(window.end)}, {window.weeks_used} weeks) is detected
            automatically from the data, not hardcoded.
          </>
        }
        legend={[
          { label: 'Used for training', color: 'var(--series-1)' },
          { label: 'Excluded (incomplete)', color: 'var(--neutral)' },
        ]}
        columns={[
          { key: 'week_start', label: 'Week of', format: (v) => date(String(v)) },
          { key: 'revenue', label: 'Revenue', align: 'right', format: (v) => money(Number(v)) },
          { key: 'used', label: 'Used', format: (v) => (v ? 'yes' : 'excluded') },
        ]}
        rows={weekly}
      >
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={weekly} barCategoryGap={2} margin={{ top: 8, right: 8, left: 8, bottom: 0 }} accessibilityLayer>
            <CartesianGrid {...gridProps} />
            <XAxis dataKey="week_start" tickFormatter={monthLabel} minTickGap={32} {...axisProps} />
            <YAxis tickFormatter={moneyShort} width={72} {...axisProps} axisLine={false} />
            <Tooltip
              cursor={{ fill: 'var(--grid)', opacity: 0.5 }}
              content={(p) => (
                <ChartTooltip<Week>
                  {...p}
                  title={(d) => `Week of ${date(d.week_start)}`}
                  rows={(d) => [
                    { label: 'Revenue', value: money(d.revenue), color: d.used ? 'var(--series-1)' : 'var(--neutral)' },
                    { label: 'Status', value: d.used ? 'used for training' : 'excluded - incomplete' },
                  ]}
                />
              )}
            />
            <Bar dataKey="revenue" radius={[3, 3, 0, 0]} isAnimationActive={false}>
              {weekly.map((w) => (
                <Cell key={w.week_start} fill={w.used ? 'var(--series-1)' : 'var(--neutral)'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </ChartCard>

      <div className="grid gap-6 lg:grid-cols-3">
        <Card
          className="lg:col-span-2"
          title="Data-quality audit"
          subtitle="Every cleaning step logs what it removed and why - no silent dropna()."
        >
          <DataTable<CleaningStep>
            columns={[
              { key: 'step', label: 'Step' },
              { key: 'rows_after', label: 'Rows after', align: 'right', format: (v) => count(Number(v)) },
              { key: 'removed', label: 'Removed', align: 'right', format: (v) => (Number(v) ? count(Number(v)) : '-') },
              { key: 'note', label: 'Why' },
            ]}
            rows={cleaning}
          />
        </Card>
        <Insight title="Why cut the edges?">
          <p>
            The export starts with a sparse pilot (and an empty Nov-Dec 2016) and ends while orders were still being
            extracted - daily revenue decays to almost zero in the last two weeks.
          </p>
          <p>
            A model trained on that tail "learns" that sales collapse. That is exactly why the original Random Forest
            forecast R$ ~180k/month when real months were ~R$ 800k.
          </p>
          <p>
            Rule used: start after the last near-empty week; end at the last week still ≥ 60% of the median of the
            8 weeks before it.
          </p>
        </Insight>
      </div>
    </div>
  )
}
