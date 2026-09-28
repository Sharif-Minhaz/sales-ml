"""Data acquisition, cleaning and aggregation.

Every cleaning step records what it removed and why, so the API can show a full data-quality audit
instead of a silent `dropna()`.
"""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from salesml import config


def fetch(data_dir: Path = config.DATA_DIR, force: bool = False) -> Path:
    """Download the Olist dataset from Kaggle (works anonymously for this public dataset)."""
    data_dir.mkdir(parents=True, exist_ok=True)
    if not force and all((data_dir / f).exists() for f in config.RAW_FILES.values()):
        print(f"Dataset already present in {data_dir}")
        return data_dir

    import kagglehub

    source = Path(kagglehub.dataset_download(config.KAGGLE_DATASET))
    for filename in config.RAW_FILES.values():
        shutil.copy2(source / filename, data_dir / filename)
    print(f"Copied {len(config.RAW_FILES)} files to {data_dir}")
    return data_dir


def load_raw(data_dir: Path = config.DATA_DIR) -> dict[str, pd.DataFrame]:
    missing = [f for f in config.RAW_FILES.values() if not (data_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing {missing} in {data_dir}. Run `uv run fetch-data` first.")
    return {name: pd.read_csv(data_dir / filename) for name, filename in config.RAW_FILES.items()}


def clean(orders: pd.DataFrame, items: pd.DataFrame, customers: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Return one row per order item, plus an audit log of every step."""
    report: list[dict] = []

    def log(step: str, before: int, after: int, note: str) -> None:
        report.append({"step": step, "rows_before": before, "rows_after": after, "removed": before - after, "note": note})

    df = orders.merge(items, on="order_id", how="inner")
    orders_without_items = orders.loc[~orders["order_id"].isin(items["order_id"]), "order_status"]
    mostly = f" (mostly {orders_without_items.mode().iat[0]})" if len(orders_without_items) else ""
    log(
        "Join orders + items",
        len(items),
        len(df),
        f"{len(orders_without_items):,} orders have no items{mostly} and carry no revenue",
    )

    before = len(df)
    excluded = df["order_status"].isin(config.EXCLUDED_STATUSES)
    lost_revenue = (df.loc[excluded, "price"] + df.loc[excluded, "freight_value"]).sum()
    df = df.loc[~excluded]
    log(
        "Drop canceled / unavailable orders",
        before,
        len(df),
        f"R$ {lost_revenue:,.0f} of orders that never became sales (v1 counted these as revenue)",
    )

    before = len(df)
    df = df.assign(order_purchase_timestamp=pd.to_datetime(df["order_purchase_timestamp"], format="ISO8601", errors="coerce"))
    df = df.dropna(subset=["order_purchase_timestamp", "price", "freight_value"])
    log("Drop unparseable dates / missing prices", before, len(df), "critical fields must be present")

    before = len(df)
    df = df.drop_duplicates(subset=["order_id", "order_item_id"])
    log("Drop duplicate order items", before, len(df), "key = (order_id, order_item_id)")

    before = len(df)
    df = df.loc[(df["price"] > 0) & (df["freight_value"] >= 0)]
    log("Drop invalid amounts", before, len(df), "price must be > 0, freight >= 0")

    before = len(df)
    df = df.merge(customers[["customer_id", "customer_unique_id", "customer_state"]], on="customer_id", how="inner")
    log("Attach customers", before, len(df), "customer_unique_id identifies a person across orders")

    df = df.assign(revenue=df["price"] + df["freight_value"])
    threshold = df["price"].quantile(config.PRICE_FLAG_QUANTILE)
    flagged = int((df["price"] > threshold).sum())
    log(
        "Flag (not drop) extreme prices",
        len(df),
        len(df),
        f"{flagged} items above R$ {threshold:,.0f} kept - they are real revenue (v1 deleted the top 1% = ~12% of revenue)",
    )

    columns = ["order_id", "order_item_id", "customer_unique_id", "customer_state", "order_status",
               "order_purchase_timestamp", "price", "freight_value", "revenue"]
    return df[columns].reset_index(drop=True), report


def weekly_revenue(df: pd.DataFrame) -> pd.Series:
    """Sum revenue into Monday-starting weeks. Empty weeks become 0 (they are real gaps, not missing rows)."""
    weekly = (
        df.set_index("order_purchase_timestamp")["revenue"]
        .resample(config.WEEK_FREQ, label="left", closed="left")
        .sum()
    )
    weekly.index.name = "week_start"
    weekly.name = "revenue"
    return weekly


def find_complete_window(
    weekly: pd.Series,
    start_zero_ratio: float = config.START_ZERO_RATIO,
    end_drop_ratio: float = config.END_DROP_RATIO,
    lookback: int = config.END_LOOKBACK_WEEKS,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Detect the span where data collection was complete.

    Olist's export starts with a sparse pilot period (and an empty Nov 2016) and ends mid-extraction
    (daily revenue decays to ~0 over the last two weeks). Training on those edges teaches a model that
    sales collapse, which is exactly what broke the v1 Random Forest forecast.

    - start: the week after the last near-empty week in the first half of the series
    - end:   walk back from the last week while it is well below the median of the weeks before it
    """
    values = weekly.to_numpy()
    median = np.median(values)

    first_half = values[: len(values) // 2]
    empty = np.flatnonzero(first_half < start_zero_ratio * median)
    start = int(empty.max()) + 1 if empty.size else 0

    end = len(values) - 1
    while end > start + lookback:
        reference = np.median(values[end - lookback : end])
        if values[end] >= end_drop_ratio * reference:
            break
        end -= 1
    return weekly.index[start], weekly.index[end]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Download the Olist dataset from Kaggle")
    parser.add_argument("--force", action="store_true", help="re-download even if files exist")
    fetch(force=parser.parse_args().force)
