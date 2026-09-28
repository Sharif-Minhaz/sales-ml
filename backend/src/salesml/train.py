"""End-to-end training pipeline: raw CSVs -> cleaned data -> models -> versioned artifacts.

    uv run train

Artifacts (loaded by the API, never retrained on request):
    artifacts/forecast.joblib   ForecastBundle (fitted winner + backtest results + interval quantiles)
    artifacts/segments.joblib   SegmentBundle (fitted sklearn Pipeline + cluster names/profile)
    artifacts/report.json       data-quality audit, KPIs and run metadata (data hashes, library versions)
"""

import hashlib
import json
import platform
import time
from datetime import datetime, timezone

import joblib
import pandas as pd
import sklearn

from salesml import config, data
from salesml.forecasting import train_forecaster
from salesml.segmentation import train_segmentation


def sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def kpis(df: pd.DataFrame) -> dict:
    orders = df.groupby("order_id")["revenue"].sum()
    per_customer = df.groupby("customer_unique_id")["order_id"].nunique()
    return {
        "revenue": float(df["revenue"].sum()),
        "orders": int(orders.size),
        "customers": int(per_customer.size),
        "average_order_value": float(orders.mean()),
        "median_order_value": float(orders.median()),
        "repeat_customer_rate": float((per_customer > 1).mean() * 100),
        "first_order": df["order_purchase_timestamp"].min().isoformat(),
        "last_order": df["order_purchase_timestamp"].max().isoformat(),
    }


def run() -> dict:
    timings = {}
    t0 = time.perf_counter()

    data.fetch()
    raw = data.load_raw()
    df, cleaning = data.clean(**raw)
    weekly = data.weekly_revenue(df)
    start, end = data.find_complete_window(weekly)
    timings["data_s"] = time.perf_counter() - t0

    t = time.perf_counter()
    forecast = train_forecaster(weekly, start, end)
    timings["forecast_s"] = time.perf_counter() - t

    t = time.perf_counter()
    segments = train_segmentation(df)
    timings["segments_s"] = time.perf_counter() - t

    config.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(forecast, config.ARTIFACT_DIR / "forecast.joblib", compress=3)
    joblib.dump(segments, config.ARTIFACT_DIR / "segments.joblib", compress=3)

    report = {
        "kpis": kpis(df),
        "cleaning": cleaning,
        "window": {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "weeks_used": len(forecast.history),
            "weeks_excluded": len(forecast.excluded),
        },
        "metadata": {
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "data_sha256": {name: sha256(config.DATA_DIR / f)[:16] for name, f in config.RAW_FILES.items()},
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "timings_s": {k: round(v, 2) for k, v in timings.items()},
        },
    }
    (config.ARTIFACT_DIR / "report.json").write_text(json.dumps(report, indent=2))

    print(f"Complete window: {start.date()} -> {end.date()} ({len(forecast.history)} weeks)")
    print(forecast.leaderboard[["label", "mae", "mae_ci_low", "mae_ci_high", "skill_vs_naive"]].round(1).to_string(index=False))
    print(f"Selected: {forecast.model_label} | interval coverage on held-out folds: {forecast.coverage:.0%} "
          f"(target {forecast.level:.0%})")
    print(f"Segments: k={segments.k} -> {sorted(set(segments.names.values()))}")
    print(f"Artifacts written to {config.ARTIFACT_DIR}")
    return report


def main() -> None:
    run()
