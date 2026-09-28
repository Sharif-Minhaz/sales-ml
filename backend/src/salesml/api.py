"""Flask API serving the persisted artifacts. The API never trains - it only loads what `uv run train` wrote.

    uv run serve            # http://127.0.0.1:5000
"""

import json
import math
import os
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

from salesml import coldstart, config
from salesml.segmentation import PLAYBOOK, base_name


class ArtifactsMissing(RuntimeError):
    pass


class Artifacts:
    """Loads the joblib bundles once and keeps them in memory for every request."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.forecast = self.segments = self.report = None

    def load(self) -> "Artifacts":
        paths = [self.directory / n for n in ("forecast.joblib", "segments.joblib", "report.json")]
        if not all(p.exists() for p in paths):
            raise ArtifactsMissing(f"No trained artifacts in {self.directory}. Run `uv run train` first.")
        self.forecast = joblib.load(paths[0])
        self.segments = joblib.load(paths[1])
        self.report = json.loads(paths[2].read_text())
        return self

    @property
    def ready(self) -> bool:
        return self.report is not None


def to_json(value):
    """Make pandas/numpy output JSON-safe: timestamps -> ISO dates, NaN -> null, numpy scalars -> Python."""
    if isinstance(value, dict):
        return {str(k): to_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if hasattr(value, "item"):  # numpy scalar
        value = value.item()
    if isinstance(value, float):
        return None if math.isnan(value) else round(value, 4)
    return value


def records(df: pd.DataFrame) -> list[dict]:
    return to_json(df.to_dict(orient="records"))


def number(payload: dict, key: str, default=None, minimum: float | None = None) -> float:
    raw = payload.get(key, default)
    if raw is None:
        raise ValueError(f"'{key}' is required")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"'{key}' must be a number") from None
    if minimum is not None and value < minimum:
        raise ValueError(f"'{key}' must be >= {minimum}")
    return value


def create_app(artifact_dir: Path = config.ARTIFACT_DIR) -> Flask:
    app = Flask(__name__)
    CORS(app, resources={r"/api/*": {"origins": ["http://localhost:5173", "http://127.0.0.1:5173"]}})
    artifacts = Artifacts(artifact_dir)
    try:
        artifacts.load()
    except ArtifactsMissing as exc:
        app.logger.warning(str(exc))

    @app.errorhandler(ArtifactsMissing)
    def missing(exc):
        return jsonify(error=str(exc)), 503

    @app.errorhandler(ValueError)
    def bad_request(exc):
        return jsonify(error=str(exc)), 400

    def require() -> Artifacts:
        if not artifacts.ready:
            artifacts.load()  # retry: artifacts may have been trained after the server started
        return artifacts

    @app.get("/api/health")
    def health():
        return jsonify(
            status="ok",
            artifacts_loaded=artifacts.ready,
            trained_at=artifacts.report["metadata"]["trained_at"] if artifacts.ready else None,
        )

    @app.post("/api/reload")
    def reload():
        """Hot-swap models after a retrain without restarting the server."""
        artifacts.load()
        return jsonify(status="reloaded", trained_at=artifacts.report["metadata"]["trained_at"])

    @app.get("/api/overview")
    def overview():
        a = require()
        f = a.forecast
        weekly = pd.concat(
            [
                f.history.to_frame("revenue").assign(used=True),
                f.excluded.to_frame("revenue").assign(used=False),
            ]
        ).sort_index()
        return jsonify(to_json({**a.report, "weekly": records(weekly.reset_index())}))

    @app.get("/api/forecast")
    def forecast():
        f = require().forecast
        horizon = int(number(request.args, "horizon", config.FORECAST_HORIZON, minimum=1))
        future = f.forecast(horizon)
        step_mae = f.step_mae.reset_index().rename(columns={"index": "step"})
        return jsonify(
            to_json(
                {
                    "model": {"key": f.model_key, "label": f.model_label},
                    "level": f.level,
                    "coverage": f.coverage,
                    "history": records(f.history.reset_index()),
                    "forecast": records(future),
                    "leaderboard": records(f.leaderboard),
                    "step_mae": records(step_mae),
                    "summary": {
                        "last_4_weeks": float(f.history.iloc[-4:].sum()),
                        "next_4_weeks": float(future["forecast"].iloc[:4].sum()),
                        "next_4_weeks_lo": float(future["lo"].iloc[:4].sum()),
                        "next_4_weeks_hi": float(future["hi"].iloc[:4].sum()),
                    },
                }
            )
        )

    @app.get("/api/segments")
    def segments():
        s = require().segments
        profile = s.profile.reset_index(names="cluster")
        profile["name"] = profile["cluster"].map(s.names)
        profile["action"] = profile["name"].map(lambda n: PLAYBOOK[base_name(n)])
        scatter = s.scatter.assign(segment=s.scatter["cluster"].map(s.names))
        return jsonify(
            to_json(
                {
                    "k": s.k,
                    "n_customers": s.n_customers,
                    "silhouette": [{"k": k, "score": v} for k, v in s.silhouette.items()],
                    "segments": records(profile.sort_values("revenue", ascending=False)),
                    "scatter": records(scatter),
                }
            )
        )

    @app.post("/api/segments/predict")
    def predict_segment():
        s = require().segments
        body = request.get_json(silent=True) or {}
        result = s.predict(
            recency_days=number(body, "recency_days", minimum=0),
            frequency=number(body, "frequency", minimum=1),
            monetary=number(body, "monetary", minimum=0.01),
        )
        return jsonify(to_json(result))

    @app.post("/api/coldstart")
    def cold_start():
        body = request.get_json(silent=True) or {}
        # No AOV supplied -> borrow the one learned from the Olist data (needs trained artifacts).
        if body.get("average_order_value") is None:
            aov = require().report["kpis"]["average_order_value"]
        else:
            aov = number(body, "average_order_value", minimum=0.01)
        result = coldstart.estimate(
            monthly_visitors=number(body, "monthly_visitors", minimum=1),
            conversion_rate_pct=number(body, "conversion_rate_pct", minimum=0.01),
            average_order_value=aov,
            monthly_traffic_growth_pct=number(body, "monthly_traffic_growth_pct", 0.0, minimum=-50),
            months=int(number(body, "months", 12)),
        )
        return jsonify(to_json({**result, "average_order_value_used": aov}))

    return app


def main() -> None:
    port = int(os.environ.get("PORT", 5000))
    create_app().run(host="127.0.0.1", port=port, debug=os.environ.get("FLASK_DEBUG") == "1")
