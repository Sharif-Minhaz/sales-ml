import numpy as np
import pandas as pd

from salesml import data


def make_raw():
    orders = pd.DataFrame(
        {
            "order_id": ["o1", "o2", "o3", "o4"],
            "customer_id": ["c1", "c2", "c3", "c1"],
            "order_status": ["delivered", "canceled", "shipped", "delivered"],
            "order_purchase_timestamp": ["2017-01-02 10:00", "2017-01-03", "2017-01-04", "not a date"],
        }
    )
    items = pd.DataFrame(
        {
            "order_id": ["o1", "o1", "o1", "o2", "o3", "o4"],
            "order_item_id": [1, 2, 2, 1, 1, 1],  # (o1, 2) is duplicated
            "price": [100.0, 50.0, 50.0, 999.0, 20.0, 10.0],
            "freight_value": [10.0, 5.0, 5.0, 1.0, 2.0, 1.0],
        }
    )
    customers = pd.DataFrame(
        {"customer_id": ["c1", "c2", "c3"], "customer_unique_id": ["u1", "u2", "u3"], "customer_state": ["SP"] * 3}
    )
    return orders, items, customers


def test_clean_removes_non_sales_and_bad_rows():
    df, report = data.clean(*make_raw())

    assert "canceled" not in set(df["order_status"])  # canceled order is not revenue
    assert not df.duplicated(["order_id", "order_item_id"]).any()
    assert df["order_purchase_timestamp"].notna().all()  # o4 had an unparseable date
    assert df["revenue"].tolist() == [110.0, 55.0, 22.0]  # price + freight
    assert sum(step["removed"] for step in report) == 3
    assert all({"step", "rows_before", "rows_after", "removed", "note"} <= set(s) for s in report)


def test_clean_keeps_expensive_items():
    orders, items, customers = make_raw()
    items.loc[4, "price"] = 50_000.0  # an extreme but legitimate sale
    df, _ = data.clean(orders, items, customers)
    assert 50_000.0 in df["price"].tolist()


def test_complete_window_trims_empty_start_and_decaying_tail():
    idx = pd.date_range("2017-01-02", periods=30, freq="W-MON")
    values = np.r_[[0, 0, 5], np.full(24, 1000.0), [400, 50, 1]]
    start, end = data.find_complete_window(pd.Series(values, index=idx))
    assert start == idx[3]
    assert end == idx[26]


def test_complete_window_keeps_clean_series_untouched():
    idx = pd.date_range("2017-01-02", periods=20, freq="W-MON")
    series = pd.Series(np.linspace(800, 1200, 20), index=idx)
    assert data.find_complete_window(series) == (idx[0], idx[-1])
