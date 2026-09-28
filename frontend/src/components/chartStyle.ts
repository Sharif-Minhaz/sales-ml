// Shared Recharts styling so every chart has the same recessive axes and hairline grid.

export const axisProps = {
  stroke: 'var(--axis)',
  tick: { fill: 'var(--muted)', fontSize: 12 },
  tickLine: false,
} as const

export const gridProps = { stroke: 'var(--grid)', vertical: false } as const
