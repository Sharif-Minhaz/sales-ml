"""Central configuration. Every tunable of the pipeline lives here so experiments are one-line changes."""

import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("SALESML_DATA_DIR", BACKEND_DIR / "data" / "raw"))
ARTIFACT_DIR = Path(os.environ.get("SALESML_ARTIFACT_DIR", BACKEND_DIR / "artifacts"))

KAGGLE_DATASET = "olistbr/brazilian-ecommerce"
RAW_FILES = {
    "orders": "olist_orders_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "customers": "olist_customers_dataset.csv",
}

# Orders in these states never turned into revenue, so they must not be counted as sales.
EXCLUDED_STATUSES = ("canceled", "unavailable")

# Items above this price quantile are *flagged* (reported) but kept: expensive items are real revenue.
PRICE_FLAG_QUANTILE = 0.999

# Weekly buckets starting on Monday. Weekly gives ~85 usable points vs ~20 monthly ones.
WEEK_FREQ = "W-MON"

# Complete-window detection (see data.find_complete_window).
START_ZERO_RATIO = 0.05  # a week below 5% of the median week counts as "no data yet"
END_DROP_RATIO = 0.6  # a trailing week below 60% of the recent median is treated as partially extracted
END_LOOKBACK_WEEKS = 8

FORECAST_HORIZON = 12  # weeks
BACKTEST_MIN_TRAIN = 52  # first backtest origin: one full year of history
BACKTEST_STEP = 1  # move the origin one week at a time -> many folds for stable error estimates
INTERVAL_LEVEL = 0.8  # 80% prediction interval

SEGMENT_K_RANGE = range(3, 8)
# Silhouette scores this close are a tie; among tied k we keep the most granular (more actionable segments).
SILHOUETTE_TOLERANCE = 0.02
# A cluster's median spend must differ from the customer median by this factor to be called high/low value.
SPEND_RATIO = 1.5
SILHOUETTE_SAMPLE = 15_000
SCATTER_SAMPLE = 2_500

RANDOM_STATE = 42
