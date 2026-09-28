// Typed client for the Flask API. Shapes mirror backend/src/salesml/api.py.

export interface CleaningStep {
  step: string
  rows_before: number
  rows_after: number
  removed: number
  note: string
}

export interface Overview {
  kpis: {
    revenue: number
    orders: number
    customers: number
    average_order_value: number
    median_order_value: number
    repeat_customer_rate: number
    first_order: string
    last_order: string
  }
  cleaning: CleaningStep[]
  window: { start: string; end: string; weeks_used: number; weeks_excluded: number }
  metadata: {
    trained_at: string
    data_sha256: Record<string, string>
    python: string
    pandas: string
    scikit_learn: string
    timings_s: Record<string, number>
  }
  weekly: { week_start: string; revenue: number; used: boolean }[]
}

export interface LeaderboardRow {
  key: string
  label: string
  note: string
  mae: number
  rmse: number
  mape: number
  bias: number
  skill_vs_naive: number
  mae_ci_low: number
  mae_ci_high: number
}

export interface Forecast {
  model: { key: string; label: string }
  level: number
  coverage: number
  history: { week_start: string; revenue: number }[]
  forecast: { week_start: string; forecast: number; lo: number; hi: number }[]
  leaderboard: LeaderboardRow[]
  step_mae: ({ step: number } & Record<string, number>)[]
  summary: { last_4_weeks: number; next_4_weeks: number; next_4_weeks_lo: number; next_4_weeks_hi: number }
}

export interface Segment {
  cluster: number
  name: string
  action: string
  customers: number
  customer_share: number
  revenue_share: number
  recency_days: number
  recency_median: number
  frequency: number
  monetary: number
  monetary_median: number
  revenue: number
}

export interface Segments {
  k: number
  n_customers: number
  silhouette: { k: number; score: number }[]
  segments: Segment[]
  scatter: { recency_days: number; frequency: number; monetary: number; cluster: number; segment: string }[]
}

export interface SegmentPrediction {
  cluster: number
  segment: string
  action: string
  affinity: Record<string, number>
}

export interface ColdStartInput {
  monthly_visitors: number
  conversion_rate_pct: number
  average_order_value?: number
  monthly_traffic_growth_pct: number
  months: number
}

export interface ColdStart {
  months: { month: number; visitors: number; conservative: number; base: number; optimistic: number }[]
  totals: { conservative: number; base: number; optimistic: number }
  scenario_multipliers: Record<string, number>
  average_order_value_used: number
}

export interface Health {
  status: string
  artifacts_loaded: boolean
  trained_at: string | null
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...init?.headers },
  })
  const body = await response.json().catch(() => ({}))
  if (!response.ok) throw new Error(body.error ?? `${response.status} ${response.statusText}`)
  return body as T
}

export const api = {
  health: () => request<Health>('/api/health'),
  overview: () => request<Overview>('/api/overview'),
  forecast: (horizon: number) => request<Forecast>(`/api/forecast?horizon=${horizon}`),
  segments: () => request<Segments>('/api/segments'),
  predictSegment: (input: { recency_days: number; frequency: number; monetary: number }) =>
    request<SegmentPrediction>('/api/segments/predict', { method: 'POST', body: JSON.stringify(input) }),
  coldStart: (input: ColdStartInput) =>
    request<ColdStart>('/api/coldstart', { method: 'POST', body: JSON.stringify(input) }),
}
