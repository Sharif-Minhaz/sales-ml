import { useEffect, useState } from 'react'

interface Result<T, K> {
  data: T | null
  error: string | null
  key: K | undefined // which request the data/error belongs to
  settled: boolean
}

/**
 * Call `load(key)` on mount and whenever `key` changes (e.g. the forecast horizon).
 * `load` must be stable (a module-level function such as `api.forecast`).
 *
 * Previous data stays visible while a new key loads, so charts dim instead of flashing a skeleton;
 * `loading` is derived rather than stored, and responses from superseded requests are ignored.
 */
export function useApi<T, K = undefined>(load: (key: K) => Promise<T>, key?: K) {
  const [result, setResult] = useState<Result<T, K>>({ data: null, error: null, key: undefined, settled: false })

  useEffect(() => {
    let current = true
    load(key as K)
      .then((data) => current && setResult({ data, error: null, key, settled: true }))
      .catch((e: Error) => current && setResult((r) => ({ ...r, error: e.message, key, settled: true })))
    return () => {
      current = false
    }
  }, [load, key])

  return { data: result.data, error: result.error, loading: !result.settled || result.key !== key }
}
