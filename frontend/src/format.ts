const brl = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'BRL', maximumFractionDigits: 0 })
const brlCompact = new Intl.NumberFormat('en-US', {
  style: 'currency',
  currency: 'BRL',
  notation: 'compact',
  maximumFractionDigits: 1,
})
const int = new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 })
const compact = new Intl.NumberFormat('en-US', { notation: 'compact', maximumFractionDigits: 1 })

export const money = (v: number) => brl.format(v)
export const moneyShort = (v: number) => brlCompact.format(v)
export const count = (v: number) => int.format(v)
export const countShort = (v: number) => compact.format(v)
export const pct = (v: number, digits = 1) => `${v.toFixed(digits)}%`

const dateFmt = new Intl.DateTimeFormat('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
const monthFmt = new Intl.DateTimeFormat('en-US', { month: 'short', year: '2-digit' })

// API dates are ISO "YYYY-MM-DD"; parse as UTC so no timezone shifts a day.
const parse = (iso: string) => new Date(`${iso.slice(0, 10)}T00:00:00Z`)
export const date = (iso: string) => dateFmt.format(parse(iso))
export const monthLabel = (iso: string) => monthFmt.format(parse(iso))
