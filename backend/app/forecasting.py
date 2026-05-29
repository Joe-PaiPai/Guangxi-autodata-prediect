from __future__ import annotations

import math
import sqlite3
from datetime import date
from statistics import mean

import pandas as pd

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover - optional runtime dependency
    XGBRegressor = None

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - optional runtime dependency
    torch = None
    nn = None


FEATURE_COLUMNS = [
    "hour",
    "weekday",
    "is_weekend",
    "load_forecast",
    "renewable_forecast",
    "hydro_forecast",
    "intertie_plan",
    "reserve_positive",
    "reserve_negative",
    "supply_total",
    "demand_total",
    "supply_demand_ratio",
    "renewable_load_ratio",
    "prev_day_price",
    "recent_7d_avg",
    "same_hour_median",
]

LSTM_SEQUENCE_DAYS = 14


REAL_TIME_METHODS = {
    "spread_follow": {
        "name": "价差跟随法",
        "description": "以前日及相似日的实时-日前价差为主，适合常规报价。",
    },
    "similar_direct": {
        "name": "相似日直接法",
        "description": "直接参考相似供需小时的实时价格，适合供需结构重复性较强的日期。",
    },
    "tightness_adjusted": {
        "name": "松紧度修正法",
        "description": "用供需松紧度修正日前预测价，适合缺少稳定实时样本时使用。",
    },
    "conservative_range": {
        "name": "保守区间法",
        "description": "扩大不确定性区间，更适合风险厌恶型报价。",
    },
}


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _clean_number(value: float | None, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, float) and math.isnan(value):
        return default
    return float(value)


def _curve_rows(conn: sqlite3.Connection, market_date: str) -> dict[int, dict]:
    rows = conn.execute(
        """
        SELECT hour, data_type, value_mw_avg
        FROM power_curve_hourly
        WHERE market_date = ?
            AND region = '广西'
        """,
        (market_date,),
    ).fetchall()
    by_hour: dict[int, dict] = {hour: {"hour": hour} for hour in range(24)}
    for row in rows:
        by_hour[row["hour"]][row["data_type"]] = row["value_mw_avg"]
    for item in by_hour.values():
        renewable = _clean_number(item.get("renewable_forecast"))
        hydro = _clean_number(item.get("hydro_forecast"))
        load = _clean_number(item.get("load_forecast"))
        intertie = _clean_number(item.get("intertie_plan"))
        supply_total = renewable + hydro
        demand_total = load + intertie
        item["supply_total"] = supply_total
        item["demand_total"] = demand_total
        item["supply_demand_ratio"] = supply_total / demand_total if demand_total else 0.0
        item["renewable_load_ratio"] = renewable / load if load else 0.0
    return by_hour


def _historical_prices(
    conn: sqlite3.Connection,
    market_date: str,
    market_type: str,
    hour: int,
    limit: int | None = None,
) -> list[float]:
    limit_sql = "" if limit is None else "LIMIT ?"
    params: list[object] = [market_type, hour, market_date]
    if limit is not None:
        params.append(limit)
    rows = conn.execute(
        f"""
        SELECT price_yuan_mwh
        FROM spot_price_hourly
        WHERE market_type = ?
            AND hour = ?
            AND market_date < ?
        ORDER BY market_date DESC
        {limit_sql}
        """,
        params,
    ).fetchall()
    return [row["price_yuan_mwh"] for row in rows if row["price_yuan_mwh"] is not None]


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def _feature_row(conn: sqlite3.Connection, market_date: str, hour: int) -> dict:
    curves = _curve_rows(conn, market_date).get(hour, {"hour": hour})
    dt = _parse_date(market_date)
    recent_7d = _historical_prices(conn, market_date, "day_ahead", hour, 7)
    all_same_hour = _historical_prices(conn, market_date, "day_ahead", hour)
    prev_day = recent_7d[0] if recent_7d else None
    row = {
        "hour": hour,
        "weekday": dt.weekday(),
        "is_weekend": 1 if dt.weekday() >= 5 else 0,
        "prev_day_price": _clean_number(prev_day, _clean_number(_median(all_same_hour), 0)),
        "recent_7d_avg": mean(recent_7d) if recent_7d else _clean_number(_median(all_same_hour), 0),
        "same_hour_median": _clean_number(_median(all_same_hour), 0),
    }
    for key in FEATURE_COLUMNS:
        if key not in row:
            row[key] = _clean_number(curves.get(key))
    return row


def _training_frame(conn: sqlite3.Connection, market_date: str, market_type: str) -> pd.DataFrame:
    dates = [
        row["market_date"]
        for row in conn.execute(
            """
            SELECT market_date
            FROM spot_price_hourly
            WHERE market_type = ?
                AND market_date < ?
            GROUP BY market_date
            HAVING COUNT(*) >= 20
            ORDER BY market_date
            """,
            (market_type, market_date),
        ).fetchall()
    ]
    rows = []
    for item_date in dates:
        price_map = {
            row["hour"]: row["price_yuan_mwh"]
            for row in conn.execute(
                """
                SELECT hour, price_yuan_mwh
                FROM spot_price_hourly
                WHERE market_type = ?
                    AND market_date = ?
                """,
                (market_type, item_date),
            ).fetchall()
        }
        for hour, price in price_map.items():
            features = _feature_row(conn, item_date, hour)
            features["target"] = price
            rows.append(features)
    return pd.DataFrame(rows)


def _xgboost_predict(conn: sqlite3.Connection, market_date: str) -> tuple[list[float | None], str]:
    if XGBRegressor is None:
        return [None] * 24, "xgboost_missing"
    frame = _training_frame(conn, market_date, "day_ahead")
    if len(frame) < 240:
        return [None] * 24, "not_enough_samples"
    x_train = frame[FEATURE_COLUMNS]
    y_train = frame["target"]
    model = XGBRegressor(
        n_estimators=180,
        max_depth=3,
        learning_rate=0.055,
        subsample=0.88,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=2,
    )
    model.fit(x_train, y_train)
    x_target = pd.DataFrame([_feature_row(conn, market_date, hour) for hour in range(24)])[FEATURE_COLUMNS]
    return [float(value) for value in model.predict(x_target)], "ok"


if nn is not None:

    class DayAheadLSTM(nn.Module):
        def __init__(self, input_size: int = 4, hidden_size: int = 32) -> None:
            super().__init__()
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
            self.head = nn.Sequential(
                nn.Linear(hidden_size, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
            )

        def forward(self, values: torch.Tensor) -> torch.Tensor:
            output, _ = self.lstm(values)
            return self.head(output[:, -1, :]).squeeze(-1)

else:
    DayAheadLSTM = None


def _sequence_fallback_predict(conn: sqlite3.Connection, market_date: str) -> tuple[list[float | None], str]:
    predictions: list[float | None] = []
    for hour in range(24):
        values = list(reversed(_historical_prices(conn, market_date, "day_ahead", hour, 21)))
        if len(values) < 5:
            predictions.append(None)
            continue
        recent = values[-7:]
        older = values[-14:-7] or values[:-7]
        momentum = mean(recent) - mean(older) if older else 0.0
        predictions.append(float(mean(recent) + 0.35 * momentum))
    return predictions, "sequence_fallback"


def _day_ahead_price_grid(conn: sqlite3.Connection, market_date: str) -> tuple[list[str], dict[tuple[str, int], float]]:
    dates = [
        row["market_date"]
        for row in conn.execute(
            """
            SELECT market_date
            FROM spot_price_hourly
            WHERE market_type = 'day_ahead'
                AND market_date < ?
            GROUP BY market_date
            HAVING COUNT(*) = 24
            ORDER BY market_date
            """,
            (market_date,),
        ).fetchall()
    ]
    grid = {
        (row["market_date"], row["hour"]): row["price_yuan_mwh"]
        for row in conn.execute(
            """
            SELECT market_date, hour, price_yuan_mwh
            FROM spot_price_hourly
            WHERE market_type = 'day_ahead'
                AND market_date < ?
            """,
            (market_date,),
        ).fetchall()
    }
    return dates, grid


def _build_lstm_training_data(
    conn: sqlite3.Connection,
    market_date: str,
    sequence_days: int = LSTM_SEQUENCE_DAYS,
) -> tuple[list[list[list[float]]], list[float], float, float]:
    dates, price_grid = _day_ahead_price_grid(conn, market_date)
    prices = list(price_grid.values())
    if len(dates) <= sequence_days or len(prices) < 360:
        return [], [], 0.0, 1.0

    price_mean = mean(prices)
    variance = mean([(value - price_mean) ** 2 for value in prices])
    price_std = max(variance**0.5, 1.0)
    x_rows: list[list[list[float]]] = []
    y_rows: list[float] = []

    for date_index in range(sequence_days, len(dates)):
        target_date = dates[date_index]
        target_weekday = _parse_date(target_date).weekday() / 6
        for hour in range(24):
            sequence = []
            complete = True
            for lookback in range(date_index - sequence_days, date_index):
                price = price_grid.get((dates[lookback], hour))
                if price is None:
                    complete = False
                    break
                hour_angle = 2 * math.pi * hour / 24
                sequence.append(
                    [
                        (price - price_mean) / price_std,
                        math.sin(hour_angle),
                        math.cos(hour_angle),
                        target_weekday,
                    ]
                )
            target = price_grid.get((target_date, hour))
            if complete and target is not None:
                x_rows.append(sequence)
                y_rows.append((target - price_mean) / price_std)
    return x_rows, y_rows, price_mean, price_std


def _target_lstm_sequences(
    conn: sqlite3.Connection,
    market_date: str,
    price_mean: float,
    price_std: float,
    sequence_days: int = LSTM_SEQUENCE_DAYS,
) -> list[list[list[float]] | None]:
    dates, price_grid = _day_ahead_price_grid(conn, market_date)
    target_weekday = _parse_date(market_date).weekday() / 6
    if len(dates) < sequence_days:
        return [None] * 24
    recent_dates = dates[-sequence_days:]
    sequences: list[list[list[float]] | None] = []
    for hour in range(24):
        sequence = []
        for item_date in recent_dates:
            price = price_grid.get((item_date, hour))
            if price is None:
                sequence = []
                break
            hour_angle = 2 * math.pi * hour / 24
            sequence.append(
                [
                    (price - price_mean) / price_std,
                    math.sin(hour_angle),
                    math.cos(hour_angle),
                    target_weekday,
                ]
            )
        sequences.append(sequence or None)
    return sequences


def _lstm_sequence_predict(conn: sqlite3.Connection, market_date: str) -> tuple[list[float | None], str]:
    if torch is None or DayAheadLSTM is None:
        return _sequence_fallback_predict(conn, market_date)

    x_rows, y_rows, price_mean, price_std = _build_lstm_training_data(conn, market_date)
    if len(x_rows) < 240:
        return _sequence_fallback_predict(conn, market_date)

    torch.manual_seed(42)
    torch.set_num_threads(2)
    x_train = torch.tensor(x_rows, dtype=torch.float32)
    y_train = torch.tensor(y_rows, dtype=torch.float32)
    model = DayAheadLSTM(input_size=4, hidden_size=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.012, weight_decay=0.0005)
    loss_fn = nn.SmoothL1Loss()

    model.train()
    for _ in range(90):
        optimizer.zero_grad()
        loss = loss_fn(model(x_train), y_train)
        loss.backward()
        optimizer.step()

    sequences = _target_lstm_sequences(conn, market_date, price_mean, price_std)
    predictions: list[float | None] = []
    model.eval()
    with torch.no_grad():
        for sequence in sequences:
            if sequence is None:
                predictions.append(None)
                continue
            value = model(torch.tensor([sequence], dtype=torch.float32)).item()
            predictions.append(value * price_std + price_mean)
    return predictions, "pytorch_lstm"


def _similar_baseline(conn: sqlite3.Connection, market_date: str, hour: int) -> float | None:
    values = _historical_prices(conn, market_date, "day_ahead", hour, 14)
    return mean(values) if values else None


def _clip_price(value: float | None) -> float | None:
    if value is None:
        return None
    return round(max(0.0, min(1500.0, value)), 3)


def predict_day_ahead_prices(conn: sqlite3.Connection, market_date: str) -> dict:
    actual_count = conn.execute(
        """
        SELECT COUNT(*) AS hours
        FROM spot_price_hourly
        WHERE market_date = ?
            AND market_type = 'day_ahead'
        """,
        (market_date,),
    ).fetchone()["hours"]
    if actual_count >= 24:
        rows = conn.execute(
            """
            SELECT hour, price_yuan_mwh
            FROM spot_price_hourly
            WHERE market_date = ?
                AND market_type = 'day_ahead'
            ORDER BY hour
            """,
            (market_date,),
        ).fetchall()
        return {
            "market_date": market_date,
            "source": "actual",
            "model": "actual_day_ahead",
            "components": {"actual": "ok"},
            "rows": [
                {
                    "hour": row["hour"],
                    "predicted_price": row["price_yuan_mwh"],
                    "xgboost_price": None,
                    "lstm_price": None,
                    "baseline_price": None,
                    "confidence": 1.0,
                }
                for row in rows
            ],
        }

    xgb_values, xgb_status = _xgboost_predict(conn, market_date)
    lstm_values, lstm_status = _lstm_sequence_predict(conn, market_date)
    rows = []
    for hour in range(24):
        baseline = _similar_baseline(conn, market_date, hour)
        weighted = []
        if xgb_values[hour] is not None:
            weighted.append((xgb_values[hour], 0.55))
        if lstm_values[hour] is not None:
            weighted.append((lstm_values[hour], 0.30))
        if baseline is not None:
            weighted.append((baseline, 0.15 if weighted else 1.0))
        if not weighted:
            predicted = None
        else:
            total_weight = sum(weight for _, weight in weighted)
            predicted = sum(value * weight for value, weight in weighted) / total_weight
        component_values = [value for value, _ in weighted]
        disagreement = max(component_values) - min(component_values) if len(component_values) > 1 else 0.0
        confidence = max(0.45, min(0.92, 0.9 - disagreement / 500)) if predicted is not None else 0.0
        rows.append(
            {
                "hour": hour,
                "predicted_price": _clip_price(predicted),
                "xgboost_price": _clip_price(xgb_values[hour]),
                "lstm_price": _clip_price(lstm_values[hour]),
                "baseline_price": _clip_price(baseline),
                "confidence": round(confidence, 3),
            }
        )
    return {
        "market_date": market_date,
        "source": "forecast",
        "model": "xgboost_lstm_ensemble",
        "components": {
            "xgboost": xgb_status,
            "lstm": lstm_status,
            "baseline": "similar_same_hour",
        },
        "rows": rows,
    }


def predict_real_time_price(
    day_ahead_price: float | None,
    similar_reference: dict | None,
    tightness_score: float | None,
    spread_history: list[float],
    method: str = "spread_follow",
) -> dict:
    method = method if method in REAL_TIME_METHODS else "spread_follow"
    spread_avg = mean(spread_history) if spread_history else None
    similar_rt = similar_reference.get("real_time_avg") if similar_reference else None
    similar_spread = similar_reference.get("spread_avg") if similar_reference else None
    tight_adjust = ((tightness_score or 50) - 50) * 1.8

    if method == "similar_direct" and similar_rt is not None:
        center = similar_rt
    elif method == "tightness_adjusted":
        base_spread = similar_spread if similar_spread is not None else (spread_avg or 0)
        center = (day_ahead_price or 0) + base_spread * 0.35 + tight_adjust
    elif method == "conservative_range":
        base_spread = similar_spread if similar_spread is not None else (spread_avg or 0)
        center = (day_ahead_price or similar_rt or 0) + base_spread * 0.55 + tight_adjust * 0.7
    else:
        base_spread = similar_spread if similar_spread is not None else (spread_avg or 0)
        center = (day_ahead_price or similar_rt or 0) + base_spread * 0.65 + tight_adjust * 0.45

    width = 45 + abs((tightness_score or 50) - 50) * 0.9
    if method == "conservative_range":
        width += 35
    return {
        "method": method,
        "method_name": REAL_TIME_METHODS[method]["name"],
        "predicted_price": _clip_price(center),
        "lower": _clip_price(center - width),
        "upper": _clip_price(center + width),
    }
