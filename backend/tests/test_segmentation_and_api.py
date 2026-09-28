import joblib
import numpy as np
import pandas as pd
import pytest

from salesml import segmentation as S
from salesml.api import create_app
from salesml.coldstart import estimate


@pytest.fixture(scope="module")
def item_rows():
    """Synthetic customers: recent/lapsed x cheap/expensive one-timers plus a repeat-buyer group."""
    rng = np.random.default_rng(0)
    rows, order = [], 0
    end = pd.Timestamp("2018-08-01")
    for group, (n, days, price, orders) in {
        "recent_cheap": (300, 20, 40, 1),
        "recent_rich": (300, 20, 600, 1),
        "lapsed_cheap": (300, 400, 40, 1),
        "lapsed_rich": (300, 400, 600, 1),
        "repeat": (100, 150, 150, 3),
    }.items():
        for c in range(n):
            for _ in range(orders):
                order += 1
                rows.append(
                    {
                        "customer_unique_id": f"{group}-{c}",
                        "order_id": f"o{order}",
                        "order_purchase_timestamp": end - pd.Timedelta(days=int(days * rng.uniform(0.7, 1.3))),
                        "revenue": price * rng.uniform(0.7, 1.3),
                    }
                )
    return pd.DataFrame(rows)


def test_segmentation_names_and_joblib_roundtrip(item_rows, tmp_path):
    bundle = S.train_segmentation(item_rows)
    names = set(bundle.names.values())
    assert "Loyal repeat buyers" in names
    assert any(n.startswith("Lapsed") for n in names) and any(n.startswith("Recent") for n in names)

    path = tmp_path / "segments.joblib"
    joblib.dump(bundle, path)
    restored = joblib.load(path)
    rfm = S.rfm_table(item_rows)
    np.testing.assert_array_equal(bundle.pipeline.predict(rfm), restored.pipeline.predict(rfm))

    result = restored.predict(recency_days=10, frequency=1, monetary=700)
    assert result["segment"].startswith("Recent")
    assert pytest.approx(sum(result["affinity"].values())) == 1


def test_coldstart_scenarios_are_ordered_and_grow():
    out = estimate(monthly_visitors=10_000, conversion_rate_pct=2, average_order_value=150,
                   monthly_traffic_growth_pct=5, months=6)
    first, last = out["months"][0], out["months"][-1]
    assert first["base"] == pytest.approx(10_000 * 0.02 * 150)
    assert first["conservative"] < first["base"] < first["optimistic"]
    assert last["base"] > first["base"]
    with pytest.raises(ValueError):
        estimate(0, 2, 150)


def test_api_reports_missing_artifacts_as_503(tmp_path):
    client = create_app(tmp_path).test_client()
    assert client.get("/api/health").get_json()["artifacts_loaded"] is False
    response = client.get("/api/forecast")
    assert response.status_code == 503
    assert "uv run train" in response.get_json()["error"]


def test_api_coldstart_validates_and_works_without_artifacts(tmp_path):
    client = create_app(tmp_path).test_client()
    ok = {"monthly_visitors": 1000, "conversion_rate_pct": 2, "average_order_value": 100, "months": 3}

    response = client.post("/api/coldstart", json=ok)
    assert response.status_code == 200
    assert len(response.get_json()["months"]) == 3

    bad = client.post("/api/coldstart", json={**ok, "monthly_visitors": -5})
    assert bad.status_code == 400
    assert "monthly_visitors" in bad.get_json()["error"]

    assert client.post("/api/coldstart", json={**ok, "months": 99}).status_code == 400
