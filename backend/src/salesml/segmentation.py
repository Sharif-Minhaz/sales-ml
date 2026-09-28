"""RFM customer segmentation with a persisted scikit-learn pipeline.

Fixes over v1:
- log1p before scaling: Monetary is heavily right-skewed, so without it a few big spenders dominate KMeans
- k chosen by silhouette score instead of hardcoded 4
- clusters get business names + recommended actions derived from their centroids
- the whole transform->cluster chain is ONE Pipeline, so inference applies identical preprocessing
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from salesml import config

FEATURES = ["recency_days", "frequency", "monetary"]

PLAYBOOK = {
    "Loyal repeat buyers": "Reward them: VIP perks, early access, referral program. Protect this group first.",
    "Recent high-value one-time buyers": "Cross-sell premium categories and invite to loyalty while the first purchase is fresh.",
    "Recent one-time buyers": "Nurture a second purchase: post-delivery follow-up, review request, next-order coupon.",
    "Recent low-value one-time buyers": "Grow basket size: free-shipping threshold, bundles, 'frequently bought together'.",
    "Lapsed high-value one-time buyers": "Win-back campaign with personalised offers - they spent well, then drifted away.",
    "Lapsed one-time buyers": "Low-cost reactivation (email, retargeting) with a time-limited incentive.",
    "Lapsed low-value one-time buyers": "Cheapest channel only (email). Don't spend paid-ads budget here.",
}


def rfm_table(df: pd.DataFrame) -> pd.DataFrame:
    snapshot = df["order_purchase_timestamp"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("customer_unique_id").agg(
        last_purchase=("order_purchase_timestamp", "max"),
        frequency=("order_id", "nunique"),
        monetary=("revenue", "sum"),
    )
    rfm["recency_days"] = (snapshot - rfm.pop("last_purchase")).dt.days
    return rfm[FEATURES]


def build_pipeline(k: int) -> Pipeline:
    return Pipeline(
        [
            ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
            ("kmeans", KMeans(n_clusters=k, n_init=10, random_state=config.RANDOM_STATE)),
        ]
    )


def choose_k(rfm: pd.DataFrame, k_range=config.SEGMENT_K_RANGE) -> tuple[int, dict[int, float]]:
    """Pick k by silhouette score (how much closer points are to their own cluster than to the next one).

    Scores within SILHOUETTE_TOLERANCE of the best are treated as a tie, and the most granular tied k wins:
    segments exist to drive different actions, so an extra segment at no real cost in separation is worth it.
    Scored on a random sample: silhouette is O(n²) and 93k customers would take minutes.
    """
    sample = rfm.sample(min(config.SILHOUETTE_SAMPLE, len(rfm)), random_state=config.RANDOM_STATE)
    scores = {}
    for k in k_range:
        pipe = build_pipeline(k).fit(rfm)
        transformed = pipe[:-1].transform(sample)
        scores[k] = float(silhouette_score(transformed, pipe.predict(sample)))
    best = max(scores.values())
    return max(k for k, s in scores.items() if s >= best - config.SILHOUETTE_TOLERANCE), scores


def name_clusters(profile: pd.DataFrame, rfm: pd.DataFrame) -> dict[int, str]:
    """Translate clusters into business language by comparing their *medians* to the whole customer base.
    Medians, not means: spend is so skewed that every cluster's mean sits above the overall median."""
    recency_median = rfm["recency_days"].median()
    monetary_median = rfm["monetary"].median()
    names: dict[int, str] = {}
    for cluster, row in profile.iterrows():
        if row["frequency"] >= 1.5:
            name = "Loyal repeat buyers"
        else:
            timing = "Recent" if row["recency_median"] < recency_median else "Lapsed"
            ratio = row["monetary_median"] / monetary_median
            value = "high-value " if ratio >= config.SPEND_RATIO else "low-value " if ratio <= 1 / config.SPEND_RATIO else ""
            name = f"{timing} {value}one-time buyers"
        names[cluster] = name

    # Disambiguate if two clusters landed on the same label (possible for larger k).
    counts = pd.Series(names).value_counts()
    for name in counts[counts > 1].index:
        dupes = profile.loc[[c for c, n in names.items() if n == name]].sort_values("monetary", ascending=False)
        for tier, cluster in enumerate(dupes.index, start=1):
            names[cluster] = f"{name} · tier {tier}"
    return names


def base_name(name: str) -> str:
    return name.split(" · ")[0]


@dataclass
class SegmentBundle:
    pipeline: Pipeline
    k: int
    silhouette: dict[int, float]
    names: dict[int, str]
    profile: pd.DataFrame
    scatter: pd.DataFrame
    n_customers: int

    def predict(self, recency_days: float, frequency: float, monetary: float) -> dict:
        X = pd.DataFrame([[recency_days, frequency, monetary]], columns=FEATURES)
        cluster = int(self.pipeline.predict(X)[0])
        # distance to every centroid in the scaled space -> a soft "how sure" signal
        distances = self.pipeline["kmeans"].transform(self.pipeline[:-1].transform(X))[0]
        weights = np.exp(-distances) / np.exp(-distances).sum()
        name = self.names[cluster]
        return {
            "cluster": cluster,
            "segment": name,
            "action": PLAYBOOK[base_name(name)],
            "affinity": {self.names[c]: float(w) for c, w in enumerate(weights)},
        }


def train_segmentation(df: pd.DataFrame) -> SegmentBundle:
    rfm = rfm_table(df)
    k, silhouette = choose_k(rfm)
    pipeline = build_pipeline(k).fit(rfm)
    labels = pipeline.predict(rfm)

    profile = rfm.groupby(labels).agg(
        customers=("frequency", "size"),
        recency_days=("recency_days", "mean"),
        recency_median=("recency_days", "median"),
        frequency=("frequency", "mean"),
        monetary=("monetary", "mean"),
        monetary_median=("monetary", "median"),
        revenue=("monetary", "sum"),
    )
    profile["customer_share"] = profile["customers"] / len(rfm) * 100
    profile["revenue_share"] = profile["revenue"] / rfm["monetary"].sum() * 100
    names = name_clusters(profile, rfm)

    scatter = rfm.assign(cluster=labels).sample(
        min(config.SCATTER_SAMPLE, len(rfm)), random_state=config.RANDOM_STATE
    )
    return SegmentBundle(
        pipeline=pipeline,
        k=k,
        silhouette=silhouette,
        names=names,
        profile=profile,
        scatter=scatter.reset_index(drop=True),
        n_customers=len(rfm),
    )
