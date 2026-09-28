import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingRegressor

from salesml import forecasting as F


def weekly(values) -> pd.Series:
    return pd.Series(np.asarray(values, dtype=float), index=pd.date_range("2017-01-02", periods=len(values), freq="W-MON"))


class SpyForecaster:
    """Records what it was trained on so we can prove the backtest never leaks the future."""

    seen: list[pd.Timestamp] = []

    def fit(self, y):
        SpyForecaster.seen.append(y.index[-1])
        self.last = float(y.iloc[-1])
        return self

    def predict(self, horizon):
        return np.full(horizon, self.last)


def test_backtest_trains_only_on_the_past():
    y = weekly(np.arange(1, 41))
    SpyForecaster.seen = []
    bt = F.backtest(SpyForecaster, y, horizon=4, min_train=20, step=1)

    assert len(SpyForecaster.seen) == 40 - 4 - 20 + 1
    for origin, trained_until in zip(sorted(bt["origin"].unique()), SpyForecaster.seen):
        assert trained_until < origin  # last training week is strictly before the first forecast week
    assert (bt["week"] >= bt["origin"]).all()


def test_growth_model_extrapolates_where_level_forest_cannot():
    y = weekly(1000 * 1.03 ** np.arange(60))  # steady 3% weekly growth

    forest = F.CalendarForestForecaster().fit(y).predict(8)
    growth = F.GrowthLagForecaster(GradientBoostingRegressor(random_state=0)).fit(y).predict(8)

    assert forest.max() <= y.max()  # trees can't predict above anything they've seen (the v1 bug)
    assert growth[-1] > y.iloc[-1] * 1.15  # compounding growth keeps climbing
    np.testing.assert_allclose(growth, y.iloc[-1] * 1.03 ** np.arange(1, 9), rtol=0.05)


@pytest.mark.parametrize("candidate", F.CANDIDATES, ids=lambda c: c.key)
def test_every_candidate_returns_a_finite_forecast(candidate):
    rng = np.random.default_rng(0)
    y = weekly(5000 + 50 * np.arange(60) + rng.normal(0, 200, 60))
    pred = candidate.factory().fit(y).predict(12)
    assert pred.shape == (12,)
    assert np.isfinite(pred).all()


def test_interval_quantiles_widen_with_horizon():
    rng = np.random.default_rng(1)
    rows = [
        {"origin": o, "step": h, "actual": 100 * np.exp(rng.normal(0, 0.05 * h)), "pred": 100.0}
        for o in range(40)
        for h in range(1, 7)
    ]
    q = F.interval_quantiles(pd.DataFrame(rows), level=0.8)
    assert (q["lo"] < 0).all() and (q["hi"] > 0).all()
    assert q["lo"].is_monotonic_decreasing and q["hi"].is_monotonic_increasing


def test_forecast_bundle_end_to_end():
    rng = np.random.default_rng(2)
    values = np.r_[[0, 0], 5000 + 40 * np.arange(70) + rng.normal(0, 300, 70), [300]]
    series = weekly(values)
    start, end = series.index[2], series.index[-2]

    bundle = F.train_forecaster(series, start, end)
    out = bundle.forecast(6)

    assert len(bundle.excluded) == 3
    assert set(bundle.leaderboard["key"]) == {c.key for c in F.CANDIDATES}
    assert (out["lo"] <= out["forecast"]).all() and (out["forecast"] <= out["hi"]).all()
    assert out["week_start"].iloc[0] == end + pd.Timedelta(weeks=1)
