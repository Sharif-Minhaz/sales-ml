"""Weekly revenue forecasting: candidate models, walk-forward backtesting and empirical prediction intervals.

All forecasters share one tiny interface so the backtest can treat them identically:

    model.fit(y)         # y: pd.Series of weekly revenue, oldest first
    model.predict(h)     # -> np.ndarray with the next h weeks
"""

import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from salesml import config


class NaiveForecaster:
    """Next weeks = last observed week. The baseline every model must beat to be worth anything."""

    def fit(self, y: pd.Series) -> "NaiveForecaster":
        self.last_ = float(y.iloc[-1])
        return self

    def predict(self, horizon: int) -> np.ndarray:
        return np.full(horizon, self.last_)


class MovingAverageForecaster:
    """Next weeks = mean of the last `window` weeks. Smooths week-to-week noise that naive copies."""

    def __init__(self, window: int = 4):
        self.window = window

    def fit(self, y: pd.Series) -> "MovingAverageForecaster":
        self.level_ = float(y.iloc[-self.window :].mean())
        return self

    def predict(self, horizon: int) -> np.ndarray:
        return np.full(horizon, self.level_)


class CombinationForecaster:
    """Equal-weight average of several forecasters. Averaging models with different error patterns
    cancels part of their errors - the most robust finding of the M-series forecasting competitions."""

    def __init__(self, factories: list[Callable[[], object]]):
        self.factories = factories

    def fit(self, y: pd.Series) -> "CombinationForecaster":
        self.models_ = [f().fit(y) for f in self.factories]
        return self

    def predict(self, horizon: int) -> np.ndarray:
        return np.mean([m.predict(horizon) for m in self.models_], axis=0)


class DriftForecaster:
    """Naive + the average historical week-over-week change (a straight line through first and last point)."""

    def fit(self, y: pd.Series) -> "DriftForecaster":
        self.last_ = float(y.iloc[-1])
        self.slope_ = (float(y.iloc[-1]) - float(y.iloc[0])) / max(len(y) - 1, 1)
        return self

    def predict(self, horizon: int) -> np.ndarray:
        return self.last_ + self.slope_ * np.arange(1, horizon + 1)


class LinearTrendForecaster:
    """v1 approach #1: least-squares line on the time index."""

    def fit(self, y: pd.Series) -> "LinearTrendForecaster":
        self.n_ = len(y)
        self.model_ = LinearRegression().fit(np.arange(self.n_).reshape(-1, 1), y.to_numpy())
        return self

    def predict(self, horizon: int) -> np.ndarray:
        return self.model_.predict(np.arange(self.n_, self.n_ + horizon).reshape(-1, 1))


class CalendarForestForecaster:
    """v1 approach #2: Random Forest on raw revenue with (time index, month, year) features.

    Kept to demonstrate *why* it fails: trees can only output averages of training targets, so they
    cannot extrapolate a growing series, and 'year' takes values at forecast time it never saw.
    """

    def fit(self, y: pd.Series) -> "CalendarForestForecaster":
        self.index_ = y.index
        self.model_ = RandomForestRegressor(n_estimators=200, random_state=config.RANDOM_STATE)
        self.model_.fit(self._features(y.index, 0), y.to_numpy())
        return self

    def _features(self, index: pd.DatetimeIndex, offset: int) -> np.ndarray:
        t = np.arange(offset, offset + len(index))
        return np.column_stack([t, index.month, index.year])

    def predict(self, horizon: int) -> np.ndarray:
        future = future_index(self.index_, horizon)
        return self.model_.predict(self._features(future, len(self.index_)))


class HoltForecaster:
    """Holt's linear exponential smoothing with a damped trend (classical statistical baseline)."""

    def fit(self, y: pd.Series) -> "HoltForecaster":
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # statsmodels is chatty about optimizer convergence on short series
            self.result_ = ExponentialSmoothing(
                y.to_numpy(dtype=float), trend="add", damped_trend=True, initialization_method="estimated"
            ).fit()
        return self

    def predict(self, horizon: int) -> np.ndarray:
        return np.asarray(self.result_.forecast(horizon))


class GrowthLagForecaster:
    """ML forecaster that models *growth rates* instead of levels.

    Target: g_t = log(y_t) - log(y_{t-1}) (week-over-week log growth).
    Features: the previous `lags` growth rates + their mean.

    Because it predicts growth and compounds it onto the last observed level, any regressor (even a
    tree ensemble) can produce values outside the training range - fixing the v1 extrapolation bug.
    Multi-step forecasts are recursive: each predicted growth rate is fed back as a lag feature.
    """

    def __init__(self, estimator, lags: int = 4):
        self.estimator = estimator
        self.lags = lags

    def _row(self, recent: np.ndarray) -> np.ndarray:
        window = recent[-self.lags :][::-1]  # most recent lag first
        return np.append(window, window.mean())

    def fit(self, y: pd.Series) -> "GrowthLagForecaster":
        log_y = np.log(np.clip(y.to_numpy(dtype=float), 1.0, None))
        growth = np.diff(log_y)
        X = np.array([self._row(growth[i - self.lags : i]) for i in range(self.lags, len(growth))])
        self.model_ = clone(self.estimator).fit(X, growth[self.lags :])
        self.growth_ = growth
        self.last_log_ = log_y[-1]
        return self

    def predict(self, horizon: int) -> np.ndarray:
        history = list(self.growth_)
        steps = []
        for _ in range(horizon):
            g = float(self.model_.predict(self._row(np.array(history)).reshape(1, -1))[0])
            history.append(g)
            steps.append(g)
        return np.exp(self.last_log_ + np.cumsum(steps))


@dataclass(frozen=True)
class Candidate:
    key: str
    label: str
    factory: Callable[[], object]
    note: str


CANDIDATES: list[Candidate] = [
    Candidate("naive", "Naive (last week)", NaiveForecaster, "Baseline: repeat the last week."),
    Candidate("ma4", "Moving average (4 weeks)", MovingAverageForecaster, "Baseline: mean of the last 4 weeks."),
    Candidate("drift", "Drift", DriftForecaster, "Baseline: last week + average weekly change."),
    Candidate("v1_linear", "v1 · Linear trend", LinearTrendForecaster, "Original project model #1."),
    Candidate("v1_forest", "v1 · Random forest (levels)", CalendarForestForecaster,
              "Original project model #2 - cannot extrapolate."),
    Candidate("holt", "Holt damped trend (ETS)", HoltForecaster, "Classical exponential smoothing."),
    Candidate("ridge_growth", "Ridge on growth lags",
              lambda: GrowthLagForecaster(make_pipeline(StandardScaler(), Ridge(alpha=10.0))),
              "Linear model on lagged week-over-week log growth."),
    Candidate("gbm_growth", "GBM on growth lags",
              lambda: GrowthLagForecaster(GradientBoostingRegressor(
                  n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8,
                  random_state=config.RANDOM_STATE)),
              "Tree ensemble on lagged log growth - extrapolates via compounding."),
    Candidate("combo", "Combo: MA4 + Holt + Ridge",
              lambda: CombinationForecaster([
                  MovingAverageForecaster,
                  HoltForecaster,
                  lambda: GrowthLagForecaster(make_pipeline(StandardScaler(), Ridge(alpha=10.0))),
              ]),
              "Equal-weight average of a level model, a smoothing model and an ML growth model."),
]


def future_index(index: pd.DatetimeIndex, horizon: int) -> pd.DatetimeIndex:
    return pd.date_range(index[-1] + pd.Timedelta(weeks=1), periods=horizon, freq=config.WEEK_FREQ)


def backtest(
    factory: Callable[[], object],
    y: pd.Series,
    horizon: int = config.FORECAST_HORIZON,
    min_train: int = config.BACKTEST_MIN_TRAIN,
    step: int = config.BACKTEST_STEP,
) -> pd.DataFrame:
    """Walk-forward (expanding window) evaluation.

    For each origin the model is trained ONLY on weeks before the origin and asked to forecast the
    next `horizon` weeks - exactly how it would be used in production. This is the honest replacement
    for v1's in-sample R², which scored models on the data they were trained on.
    """
    rows = []
    for origin in range(min_train, len(y) - horizon + 1, step):
        train, test = y.iloc[:origin], y.iloc[origin : origin + horizon]
        pred = factory().fit(train).predict(horizon)
        for h, (week, actual, p) in enumerate(zip(test.index, test.to_numpy(), pred), start=1):
            rows.append({"origin": y.index[origin], "step": h, "week": week, "actual": actual, "pred": float(p)})
    return pd.DataFrame(rows)


def score(bt: pd.DataFrame) -> dict[str, float]:
    err = bt["actual"] - bt["pred"]
    return {
        "mae": float(err.abs().mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "mape": float((err.abs() / bt["actual"]).mean() * 100),
        "bias": float(err.mean()),  # > 0 means the model under-forecasts on average
    }


def mae_confidence_interval(bt: pd.DataFrame, n_boot: int = 1000, level: float = 0.9) -> tuple[float, float]:
    """Bootstrap CI for MAE by resampling whole backtest folds (origins), not single points - points within
    a fold are correlated, so resampling them individually would make the interval falsely narrow."""
    fold_mae = (bt["actual"] - bt["pred"]).abs().groupby(bt["origin"]).mean().to_numpy()
    rng = np.random.default_rng(config.RANDOM_STATE)
    boots = rng.choice(fold_mae, size=(n_boot, len(fold_mae)), replace=True).mean(axis=1)
    return float(np.quantile(boots, (1 - level) / 2)), float(np.quantile(boots, 1 - (1 - level) / 2))


def interval_quantiles(bt: pd.DataFrame, level: float = config.INTERVAL_LEVEL) -> pd.DataFrame:
    """Empirical (split-conformal style) prediction intervals from backtest residuals.

    For each horizon step we take the distribution of log(actual / pred) across all folds and keep its
    lower/upper quantiles. Working in log space makes the band scale with the forecast level and lets it
    be asymmetric. Bands are forced to widen with the horizon (uncertainty can't shrink further out).
    """
    alpha = (1 - level) / 2
    log_ratio = np.log(bt["actual"] / bt["pred"].clip(lower=1.0))
    q = log_ratio.groupby(bt["step"]).quantile([alpha, 1 - alpha]).unstack()
    q.columns = ["lo", "hi"]
    q["lo"] = q["lo"].cummin()
    q["hi"] = q["hi"].cummax()
    return q


def interval_coverage(bt: pd.DataFrame, level: float = config.INTERVAL_LEVEL) -> float:
    """Honesty check for the intervals: calibrate on the earlier half of the folds, measure how often the
    later half's actuals fall inside the band. Close to `level` means the band is trustworthy."""
    origins = np.sort(bt["origin"].unique())
    cut = origins[len(origins) // 2]
    calib, holdout = bt[bt["origin"] < cut], bt[bt["origin"] >= cut]
    q = interval_quantiles(calib, level)
    lo = holdout["pred"] * np.exp(holdout["step"].map(q["lo"]))
    hi = holdout["pred"] * np.exp(holdout["step"].map(q["hi"]))
    return float(((holdout["actual"] >= lo) & (holdout["actual"] <= hi)).mean())


def evaluate_candidates(y: pd.Series, candidates: list[Candidate] = CANDIDATES) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Backtest every candidate; return a leaderboard sorted by MAE plus the raw backtest frames."""
    backtests = {c.key: backtest(c.factory, y) for c in candidates}
    naive_mae = score(backtests["naive"])["mae"] if "naive" in backtests else None
    rows = []
    for c in candidates:
        s = score(backtests[c.key])
        s["skill_vs_naive"] = (1 - s["mae"] / naive_mae) * 100 if naive_mae else None
        s["mae_ci_low"], s["mae_ci_high"] = mae_confidence_interval(backtests[c.key])
        rows.append({"key": c.key, "label": c.label, "note": c.note, **s})
    leaderboard = pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)
    return leaderboard, backtests


@dataclass
class ForecastBundle:
    """Everything the API needs at inference time, persisted with joblib as one object."""

    model_key: str
    model_label: str
    model: object
    history: pd.Series
    excluded: pd.Series
    quantiles: pd.DataFrame
    leaderboard: pd.DataFrame
    step_mae: pd.DataFrame
    coverage: float
    level: float

    def forecast(self, horizon: int) -> pd.DataFrame:
        horizon = min(horizon, len(self.quantiles))
        pred = self.model.predict(horizon)
        q = self.quantiles.iloc[:horizon]
        return pd.DataFrame(
            {
                "week_start": future_index(self.history.index, horizon),
                "forecast": pred,
                "lo": pred * np.exp(q["lo"].to_numpy()),
                "hi": pred * np.exp(q["hi"].to_numpy()),
            }
        )


def train_forecaster(weekly: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> ForecastBundle:
    history = weekly.loc[start:end]
    excluded = weekly.drop(history.index)
    leaderboard, backtests = evaluate_candidates(history)

    best = leaderboard.iloc[0]
    candidate = next(c for c in CANDIDATES if c.key == best["key"])
    bt = backtests[candidate.key]
    step_mae = pd.DataFrame(
        {key: (b["actual"] - b["pred"]).abs().groupby(b["step"]).mean() for key, b in backtests.items()}
    )

    return ForecastBundle(
        model_key=candidate.key,
        model_label=candidate.label,
        model=candidate.factory().fit(history),  # refit the winner on ALL complete weeks
        history=history,
        excluded=excluded,
        quantiles=interval_quantiles(bt),
        leaderboard=leaderboard,
        step_mae=step_mae,
        coverage=interval_coverage(bt),
        level=config.INTERVAL_LEVEL,
    )
