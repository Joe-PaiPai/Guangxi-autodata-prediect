from __future__ import annotations

import sqlite3
from statistics import mean

from app.forecasting import predict_day_ahead_prices, predict_real_time_price


def rows_to_dicts(rows: list[sqlite3.Row]) -> list[dict]:
    return [dict(row) for row in rows]


def available_dates(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        """
        WITH dates AS (
            SELECT market_date FROM spot_price_hourly
            UNION
            SELECT market_date FROM power_curve_hourly
        )
        SELECT
            d.market_date,
            SUM(CASE WHEN p.market_type = 'day_ahead' THEN 1 ELSE 0 END) AS day_ahead_hours,
            SUM(CASE WHEN p.market_type = 'real_time' THEN 1 ELSE 0 END) AS real_time_hours,
            (
                SELECT COUNT(DISTINCT data_type)
                FROM power_curve_hourly c
                WHERE c.market_date = d.market_date
            ) AS curve_types
        FROM dates d
        LEFT JOIN spot_price_hourly p
            ON p.market_date = d.market_date
        GROUP BY d.market_date
        ORDER BY d.market_date DESC
        """
    ).fetchall()
    return rows_to_dicts(rows)


def import_status(conn: sqlite3.Connection) -> dict:
    latest_run = conn.execute(
        """
        SELECT
            started_at,
            raw_root,
            files_seen,
            price_rows,
            curve_15min_rows,
            curve_hourly_rows,
            notes
        FROM import_runs
        ORDER BY started_at DESC, id DESC
        LIMIT 1
        """
    ).fetchone()
    price_dates = conn.execute(
        """
        SELECT
            MAX(CASE WHEN market_type = 'day_ahead' THEN market_date END) AS latest_day_ahead_date,
            MAX(CASE WHEN market_type = 'real_time' THEN market_date END) AS latest_real_time_date,
            COUNT(DISTINCT market_date) AS price_date_count
        FROM spot_price_hourly
        """
    ).fetchone()
    curve_dates = conn.execute(
        """
        SELECT
            MAX(market_date) AS latest_curve_date,
            COUNT(DISTINCT market_date) AS curve_date_count
        FROM power_curve_hourly
        """
    ).fetchone()
    return {
        "latest_import": dict(latest_run) if latest_run else None,
        "latest_day_ahead_date": price_dates["latest_day_ahead_date"] if price_dates else None,
        "latest_real_time_date": price_dates["latest_real_time_date"] if price_dates else None,
        "latest_curve_date": curve_dates["latest_curve_date"] if curve_dates else None,
        "price_date_count": price_dates["price_date_count"] if price_dates else 0,
        "curve_date_count": curve_dates["curve_date_count"] if curve_dates else 0,
    }


def daily_prices(
    conn: sqlite3.Connection,
    market_date: str,
    real_time_method: str = "spread_follow",
    include_forecast: bool = True,
) -> list[dict]:
    rows = conn.execute(
        """
        SELECT
            da.hour,
            da.time_label,
            da.price_yuan_mwh AS day_ahead_price,
            da.volume_mwh AS day_ahead_volume_mwh,
            rt.price_yuan_mwh AS real_time_price,
            CASE
                WHEN rt.price_yuan_mwh IS NULL THEN NULL
                ELSE rt.price_yuan_mwh - da.price_yuan_mwh
            END AS spread_real_minus_day_ahead
        FROM spot_price_hourly da
        LEFT JOIN spot_price_hourly rt
            ON rt.market_date = da.market_date
            AND rt.hour = da.hour
            AND rt.market_type = 'real_time'
        WHERE da.market_date = ?
            AND da.market_type = 'day_ahead'
        ORDER BY da.hour
        """,
        (market_date,),
    ).fetchall()
    actual_rows = rows_to_dicts(rows)
    if actual_rows or not include_forecast:
        for row in actual_rows:
            row["day_ahead_source"] = "actual"
            row["real_time_source"] = "actual" if row["real_time_price"] is not None else None
            row["day_ahead_confidence"] = 1.0
        return actual_rows

    forecast = predict_day_ahead_prices(conn, market_date)
    result = []
    for item in forecast["rows"]:
        hour = item["hour"]
        day_price = item["predicted_price"]
        result.append(
            {
                "hour": hour,
                "time_label": f"{hour:02d}:00",
                "day_ahead_price": day_price,
                "day_ahead_volume_mwh": None,
                "real_time_price": None,
                "spread_real_minus_day_ahead": None,
                "day_ahead_source": "forecast",
                "real_time_source": None,
                "day_ahead_confidence": item["confidence"],
                "day_ahead_model": forecast["model"],
                "day_ahead_components": forecast["components"],
                "xgboost_price": item["xgboost_price"],
                "lstm_price": item["lstm_price"],
                "baseline_price": item["baseline_price"],
            }
        )
    return result


def hourly_curves(conn: sqlite3.Connection, market_date: str) -> list[dict]:
    rows = conn.execute(
        """
        SELECT
            hour,
            data_type,
            region,
            ROUND(value_mw_avg, 3) AS value_mw_avg,
            source_15min_count
        FROM power_curve_hourly
        WHERE market_date = ?
            AND region IN ('广西', '全区域', '南网')
        ORDER BY data_type, region, hour
        """,
        (market_date,),
    ).fetchall()
    return rows_to_dicts(rows)


def history_trend(conn: sqlite3.Connection, market_date: str, days: int = 14) -> list[dict]:
    rows = conn.execute(
        """
        SELECT
            market_date,
            ROUND(AVG(CASE WHEN market_type = 'day_ahead' THEN price_yuan_mwh END), 3) AS day_ahead_avg,
            ROUND(MAX(CASE WHEN market_type = 'day_ahead' THEN price_yuan_mwh END), 3) AS day_ahead_max,
            ROUND(MIN(CASE WHEN market_type = 'day_ahead' THEN price_yuan_mwh END), 3) AS day_ahead_min,
            ROUND(AVG(CASE WHEN market_type = 'real_time' THEN price_yuan_mwh END), 3) AS real_time_avg,
            ROUND(
                AVG(CASE WHEN market_type = 'real_time' THEN price_yuan_mwh END)
                - AVG(CASE WHEN market_type = 'day_ahead' THEN price_yuan_mwh END),
                3
            ) AS avg_spread
        FROM spot_price_hourly
        WHERE market_date <= ?
        GROUP BY market_date
        ORDER BY market_date DESC
        LIMIT ?
        """,
        (market_date, days),
    ).fetchall()
    return list(reversed(rows_to_dicts(rows)))


def data_quality(conn: sqlite3.Connection, market_date: str) -> dict:
    price_rows = rows_to_dicts(
        conn.execute(
            """
            SELECT market_type, COUNT(*) AS hours
            FROM spot_price_hourly
            WHERE market_date = ?
            GROUP BY market_type
            """,
            (market_date,),
        ).fetchall()
    )
    price_counts = {row["market_type"]: row["hours"] for row in price_rows}

    expected_curve_types = [
        "load_forecast",
        "renewable_forecast",
        "hydro_forecast",
        "intertie_plan",
        "non_market_generation_forecast",
        "reserve_positive",
        "reserve_negative",
    ]
    curve_rows = rows_to_dicts(
        conn.execute(
            """
            SELECT
                data_type,
                COUNT(*) AS hours,
                SUM(CASE WHEN source_15min_count = 4 THEN 1 ELSE 0 END) AS complete_hours,
                MIN(source_15min_count) AS min_points,
                MAX(source_15min_count) AS max_points
            FROM power_curve_hourly
            WHERE market_date = ?
                AND region = '广西'
                AND data_type IN (
                    'load_forecast',
                    'renewable_forecast',
                    'hydro_forecast',
                    'intertie_plan',
                    'non_market_generation_forecast',
                    'reserve_positive',
                    'reserve_negative'
                )
            GROUP BY data_type
            """,
            (market_date,),
        ).fetchall()
    )
    curve_map = {row["data_type"]: row for row in curve_rows}

    checks = [
        {
            "name": "日前价格",
            "status": "完整" if price_counts.get("day_ahead", 0) == 24 else "缺失",
            "detail": f"{price_counts.get('day_ahead', 0)}/24 小时",
        },
        {
            "name": "实时价格",
            "status": "完整" if price_counts.get("real_time", 0) == 24 else "缺失",
            "detail": f"{price_counts.get('real_time', 0)}/24 小时",
        },
    ]

    for data_type in expected_curve_types:
        row = curve_map.get(data_type)
        hours = row["hours"] if row else 0
        complete = row["complete_hours"] if row else 0
        checks.append(
            {
                "name": data_type,
                "status": "完整" if hours == 24 and complete == 24 else "不完整",
                "detail": f"{hours}/24 小时，完整小时 {complete}/24",
            }
        )

    score = round(sum(1 for item in checks if item["status"] == "完整") / len(checks) * 100, 1)
    return {
        "market_date": market_date,
        "score": score,
        "checks": checks,
    }


def price_summary(conn: sqlite3.Connection, market_date: str) -> dict:
    prices = daily_prices(conn, market_date)
    day_ahead = [row["day_ahead_price"] for row in prices if row["day_ahead_price"] is not None]
    real_time = [row["real_time_price"] for row in prices if row["real_time_price"] is not None]
    spreads = [row["spread_real_minus_day_ahead"] for row in prices if row["spread_real_minus_day_ahead"] is not None]

    def metrics(values: list[float]) -> dict:
        if not values:
            return {"avg": None, "max": None, "min": None, "range": None}
        return {
            "avg": round(mean(values), 3),
            "max": round(max(values), 3),
            "min": round(min(values), 3),
            "range": round(max(values) - min(values), 3),
        }

    high_hours = sorted(prices, key=lambda row: row["day_ahead_price"] or -1, reverse=True)[:3]
    low_hours = sorted(prices, key=lambda row: row["day_ahead_price"] or 10**9)[:3]
    spread_up = sorted(
        [row for row in prices if row["spread_real_minus_day_ahead"] is not None],
        key=lambda row: row["spread_real_minus_day_ahead"],
        reverse=True,
    )[:3]
    spread_down = sorted(
        [row for row in prices if row["spread_real_minus_day_ahead"] is not None],
        key=lambda row: row["spread_real_minus_day_ahead"],
    )[:3]

    return {
        "market_date": market_date,
        "day_ahead": metrics(day_ahead),
        "real_time": metrics(real_time),
        "spread_real_minus_day_ahead": metrics(spreads),
        "high_day_ahead_hours": [
            {"hour": row["hour"], "price": row["day_ahead_price"]} for row in high_hours
        ],
        "low_day_ahead_hours": [
            {"hour": row["hour"], "price": row["day_ahead_price"]} for row in low_hours
        ],
        "largest_real_time_premium_hours": [
            {"hour": row["hour"], "spread": row["spread_real_minus_day_ahead"]} for row in spread_up
        ],
        "largest_real_time_discount_hours": [
            {"hour": row["hour"], "spread": row["spread_real_minus_day_ahead"]} for row in spread_down
        ],
    }


def percentile_rank(value: float | None, history: list[float]) -> float | None:
    if value is None or not history:
        return None
    below_or_equal = sum(1 for item in history if item <= value)
    return round(below_or_equal / len(history) * 100, 1)


def recent_hour_values(
    conn: sqlite3.Connection,
    market_date: str,
    market_type: str,
    hour: int,
    limit: int = 7,
) -> list[float]:
    rows = conn.execute(
        """
        SELECT price_yuan_mwh
        FROM spot_price_hourly
        WHERE market_type = ?
            AND hour = ?
            AND market_date < ?
        ORDER BY market_date DESC
        LIMIT ?
        """,
        (market_type, hour, market_date, limit),
    ).fetchall()
    return [row["price_yuan_mwh"] for row in rows]


def all_prior_hour_values(
    conn: sqlite3.Connection,
    market_date: str,
    market_type: str,
    hour: int,
) -> list[float]:
    rows = conn.execute(
        """
        SELECT price_yuan_mwh
        FROM spot_price_hourly
        WHERE market_type = ?
            AND hour = ?
            AND market_date < ?
        ORDER BY market_date
        """,
        (market_type, hour, market_date),
    ).fetchall()
    return [row["price_yuan_mwh"] for row in rows]


def previous_day_price(
    conn: sqlite3.Connection,
    market_date: str,
    market_type: str,
    hour: int,
) -> float | None:
    row = conn.execute(
        """
        SELECT price_yuan_mwh
        FROM spot_price_hourly
        WHERE market_type = ?
            AND hour = ?
            AND market_date < ?
        ORDER BY market_date DESC
        LIMIT 1
        """,
        (market_type, hour, market_date),
    ).fetchone()
    return row["price_yuan_mwh"] if row else None


def median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def tightness_index(
    supply_total: float,
    demand_total: float,
    reserve: float | None,
    renewable: float | None,
    load: float | None,
) -> tuple[float, str, list[str]]:
    reasons = []
    score = 50.0

    if demand_total > 0:
        supply_ratio = supply_total / demand_total
        # Higher hydro+renewable coverage generally reduces price pressure.
        score += (0.42 - supply_ratio) * 100
        if supply_ratio >= 0.55:
            reasons.append("供给覆盖率偏高")
        elif supply_ratio <= 0.28:
            reasons.append("供给覆盖率偏低")

    if reserve is not None:
        if reserve < 1200:
            score += 18
            reasons.append("正备用偏低")
        elif reserve > 2600:
            score -= 8
            reasons.append("正备用较充足")

    if renewable is not None and load:
        renewable_ratio = renewable / max(load, 1)
        if renewable_ratio > 0.35:
            score -= 12
            reasons.append("新能源占负荷比例较高")
        elif renewable_ratio < 0.15:
            score += 10
            reasons.append("新能源占负荷比例较低")

    score = round(clamp(score, 0, 100), 1)
    if score >= 65:
        label = "偏紧"
    elif score <= 38:
        label = "偏松"
    else:
        label = "均衡"
    return score, label, reasons


def quote_range(
    day_price: float | None,
    recent_avg: float | None,
    price_percentile: float | None,
    tightness_score: float,
    spread_history: list[float],
    similar_reference: dict | None = None,
) -> dict:
    anchors = [value for value in (day_price, recent_avg) if value is not None]
    if similar_reference:
        if similar_reference.get("day_ahead_avg") is not None:
            anchors.append(similar_reference["day_ahead_avg"])
        if similar_reference.get("real_time_avg") is not None:
            anchors.append(similar_reference["real_time_avg"])
    center = mean(anchors) if anchors else 0.0

    if price_percentile is not None:
        if price_percentile >= 85:
            center += 20
        elif price_percentile <= 20:
            center -= 20

    center += (tightness_score - 50) * 1.2
    spread_med = median(spread_history)
    if spread_med is not None:
        center += clamp(spread_med * 0.25, -25, 25)
    if similar_reference and similar_reference.get("spread_avg") is not None:
        center += clamp(similar_reference["spread_avg"] * 0.25, -25, 25)

    width = 35 + abs(tightness_score - 50) * 0.6
    if price_percentile is not None and (price_percentile >= 85 or price_percentile <= 15):
        width += 15

    center = round(clamp(center, 0, 1500), 1)
    lower = round(clamp(center - width, 0, 1500), 1)
    upper = round(clamp(center + width, 0, 1500), 1)
    return {
        "lower": lower,
        "center": center,
        "upper": upper,
    }


def historical_spreads(conn: sqlite3.Connection, market_date: str, hour: int, limit: int = 14) -> list[float]:
    rows = conn.execute(
        """
        SELECT rt.price_yuan_mwh - da.price_yuan_mwh AS spread
        FROM spot_price_hourly da
        JOIN spot_price_hourly rt
            ON rt.market_date = da.market_date
            AND rt.hour = da.hour
            AND rt.market_type = 'real_time'
        WHERE da.market_type = 'day_ahead'
            AND da.hour = ?
            AND da.market_date < ?
        ORDER BY da.market_date DESC
        LIMIT ?
        """,
        (hour, market_date, limit),
    ).fetchall()
    return [row["spread"] for row in rows if row["spread"] is not None]


def curve_value_map(conn: sqlite3.Connection, market_date: str) -> dict[int, dict]:
    return {row["hour"]: row for row in supply_series(conn, market_date)}


def similar_hours(
    conn: sqlite3.Connection,
    market_date: str,
    hour: int,
    target_supply: dict | None,
    limit: int = 8,
) -> dict:
    if not target_supply:
        return {"samples": [], "day_ahead_avg": None, "real_time_avg": None, "spread_avg": None}

    target_values = {
        "demand_total": target_supply.get("demand_total"),
        "supply_total": target_supply.get("supply_total"),
        "reserve_positive": target_supply.get("reserve_positive"),
        "renewable_forecast": target_supply.get("renewable_forecast"),
    }
    candidate_dates = [
        row["market_date"]
        for row in conn.execute(
            """
            SELECT DISTINCT market_date
            FROM spot_price_hourly
            WHERE market_date < ?
            ORDER BY market_date DESC
            LIMIT 45
            """,
            (market_date,),
        ).fetchall()
    ]

    candidates = []
    for candidate_date in candidate_dates:
        candidate_supply = curve_value_map(conn, candidate_date).get(hour)
        if not candidate_supply:
            continue
        distance = 0.0
        used = 0
        for key, scale in (
            ("demand_total", 4000),
            ("supply_total", 4000),
            ("reserve_positive", 1000),
            ("renewable_forecast", 3000),
        ):
            target_value = target_values.get(key)
            candidate_value = candidate_supply.get(key)
            if target_value is None or candidate_value is None:
                continue
            distance += abs(target_value - candidate_value) / scale
            used += 1
        if used < 2:
            continue

        price = conn.execute(
            """
            SELECT
                da.price_yuan_mwh AS day_ahead_price,
                rt.price_yuan_mwh AS real_time_price
            FROM spot_price_hourly da
            LEFT JOIN spot_price_hourly rt
                ON rt.market_date = da.market_date
                AND rt.hour = da.hour
                AND rt.market_type = 'real_time'
            WHERE da.market_type = 'day_ahead'
                AND da.market_date = ?
                AND da.hour = ?
            """,
            (candidate_date, hour),
        ).fetchone()
        if not price:
            continue
        candidates.append(
            {
                "market_date": candidate_date,
                "hour": hour,
                "distance": round(distance, 4),
                "day_ahead_price": price["day_ahead_price"],
                "real_time_price": price["real_time_price"],
                "spread": (
                    round(price["real_time_price"] - price["day_ahead_price"], 3)
                    if price["real_time_price"] is not None
                    else None
                ),
            }
        )

    samples = sorted(candidates, key=lambda item: item["distance"])[:limit]

    def avg(key: str) -> float | None:
        values = [item[key] for item in samples if item.get(key) is not None]
        return round(mean(values), 3) if values else None

    return {
        "samples": samples,
        "day_ahead_avg": avg("day_ahead_price"),
        "real_time_avg": avg("real_time_price"),
        "spread_avg": avg("spread"),
    }


def supply_series(conn: sqlite3.Connection, market_date: str) -> list[dict]:
    rows = conn.execute(
        """
        SELECT hour, data_type, value_mw_avg
        FROM power_curve_hourly
        WHERE market_date = ?
            AND region = '广西'
            AND data_type IN (
                'load_forecast',
                'renewable_forecast',
                'hydro_forecast',
                'intertie_plan',
                'non_market_generation_forecast',
                'reserve_positive',
                'reserve_negative'
            )
        ORDER BY hour, data_type
        """,
        (market_date,),
    ).fetchall()

    by_hour: dict[int, dict] = {hour: {"hour": hour} for hour in range(24)}
    for row in rows:
        by_hour[row["hour"]][row["data_type"]] = round(row["value_mw_avg"], 3)
    for item in by_hour.values():
        load = item.get("load_forecast")
        renewable = item.get("renewable_forecast")
        hydro = item.get("hydro_forecast")
        intertie = item.get("intertie_plan")
        supply_total = (renewable or 0) + (hydro or 0)
        demand_total = (load or 0) + (intertie or 0)
        item["supply_total"] = round(supply_total, 3)
        item["demand_total"] = round(demand_total, 3)
        if demand_total:
            item["supply_demand_ratio"] = round(supply_total / demand_total, 4)
        if load and renewable is not None:
            item["renewable_load_ratio"] = round(renewable / max(load, 1), 4)
    return list(by_hour.values())


def strategy_report(conn: sqlite3.Connection, market_date: str, real_time_method: str = "spread_follow") -> dict:
    summary = price_summary(conn, market_date)
    prices = daily_prices(conn, market_date, real_time_method=real_time_method)
    supply = supply_series(conn, market_date)
    curve_map: dict[tuple[str, int], float] = {}
    for row in supply:
        hour = row["hour"]
        for key in (
            "load_forecast",
            "renewable_forecast",
            "hydro_forecast",
            "intertie_plan",
            "non_market_generation_forecast",
            "reserve_positive",
            "reserve_negative",
        ):
            if key in row:
                curve_map[(key, hour)] = row[key]

    hourly_advice = []
    for row in prices:
        hour = row["hour"]
        day_price = row["day_ahead_price"]
        spread = row["spread_real_minus_day_ahead"]
        load = curve_map.get(("load_forecast", hour))
        renewable = curve_map.get(("renewable_forecast", hour))
        hydro = curve_map.get(("hydro_forecast", hour))
        intertie = curve_map.get(("intertie_plan", hour))
        reserve = curve_map.get(("reserve_positive", hour))
        non_market = curve_map.get(("non_market_generation_forecast", hour))

        recent_day_ahead = recent_hour_values(conn, market_date, "day_ahead", hour)
        prior_day_ahead = all_prior_hour_values(conn, market_date, "day_ahead", hour)
        recent_avg = round(mean(recent_day_ahead), 3) if recent_day_ahead else None
        price_percentile = percentile_rank(day_price, prior_day_ahead)
        previous_price = previous_day_price(conn, market_date, "day_ahead", hour)
        previous_change = (
            round(day_price - previous_price, 3)
            if day_price is not None and previous_price is not None
            else None
        )

        reasons = []
        stance = "观望"
        risk_level = "中"

        if day_price is not None:
            if price_percentile is not None and price_percentile >= 85:
                stance = "报价偏高"
                risk_level = "高"
                reasons.append(f"日前价格处于同小时历史高分位（{price_percentile}%）")
            elif price_percentile is not None and price_percentile <= 20:
                stance = "报价偏低"
                risk_level = "中"
                reasons.append(f"日前价格处于同小时历史低分位（{price_percentile}%）")
            elif day_price >= 300:
                stance = "报价偏高"
                risk_level = "高"
                reasons.append("日前价格绝对值处于高位")
            elif day_price <= 80:
                stance = "报价偏低"
                reasons.append("日前价格绝对值处于低位")
            else:
                reasons.append("日前价格处于历史中间区间")

        if recent_avg is not None and day_price is not None:
            delta_recent = day_price - recent_avg
            if delta_recent > 50:
                reasons.append(f"高于近7日同小时均价 {delta_recent:.1f} 元/MWh")
                risk_level = "高"
            elif delta_recent < -50:
                reasons.append(f"低于近7日同小时均价 {abs(delta_recent):.1f} 元/MWh")

        if previous_change is not None and abs(previous_change) >= 60:
            direction = "上升" if previous_change > 0 else "下降"
            reasons.append(f"较前一日同小时{direction} {abs(previous_change):.1f} 元/MWh")

        if spread is not None:
            if spread > 50:
                reasons.append("实时价格明显高于日前价格")
                risk_level = "高"
            elif spread < -50:
                reasons.append("实时价格明显低于日前价格")
            else:
                reasons.append("实时与日前价差不大")

        supply_total = (renewable or 0) + (hydro or 0)
        demand_total = (load or 0) + (intertie or 0)
        tight_score, tight_label, tight_reasons = tightness_index(
            supply_total=supply_total,
            demand_total=demand_total,
            reserve=reserve,
            renewable=renewable,
            load=load,
        )
        spread_history = historical_spreads(conn, market_date, hour)
        target_supply = next((item for item in supply if item["hour"] == hour), None)
        similar_reference = similar_hours(conn, market_date, hour, target_supply)
        real_time_forecast = predict_real_time_price(
            day_ahead_price=day_price,
            similar_reference=similar_reference,
            tightness_score=tight_score,
            spread_history=spread_history,
            method=real_time_method,
        )
        real_time_price = row["real_time_price"]
        real_time_source = row.get("real_time_source")
        if real_time_price is None:
            real_time_price = real_time_forecast["predicted_price"]
            real_time_source = "forecast"
            spread = (
                round(real_time_price - day_price, 3)
                if real_time_price is not None and day_price is not None
                else None
            )
        quote = quote_range(
            day_price=day_price,
            recent_avg=recent_avg,
            price_percentile=price_percentile,
            tightness_score=tight_score,
            spread_history=spread_history,
            similar_reference=similar_reference,
        )

        if load is not None and renewable is not None:
            renewable_ratio = renewable / max(load, 1)
            if renewable_ratio > 0.35:
                reasons.append("新能源出力占统调负荷比例较高，低价压力较大")
                if stance == "报价偏高":
                    stance = "高价谨慎"
            elif renewable_ratio < 0.15:
                reasons.append("新能源支撑偏弱，价格上行风险增加")
                if stance == "报价偏低":
                    stance = "低价谨慎"

        if demand_total > 0:
            supply_ratio = supply_total / demand_total
            if supply_ratio > 0.55:
                reasons.append("水电+新能源相对统调负荷+省间联络线占比较高")
                if stance == "报价偏高":
                    stance = "高价谨慎"
            elif supply_ratio < 0.25:
                reasons.append("水电+新能源对需求侧覆盖偏弱")
                risk_level = "高"
        reasons.extend(tight_reasons)

        if load is not None and non_market is not None:
            controllable_gap = load - non_market
            if controllable_gap > 20000:
                reasons.append("负荷扣除非市场化机组后缺口偏大")
                risk_level = "高"
            elif controllable_gap < 8000:
                reasons.append("非市场化机组覆盖度较高，市场出清压力偏弱")

        if reserve is not None and reserve < 1000:
            reasons.append("正备用偏低，系统偏紧")
            risk_level = "高"
        if similar_reference["samples"]:
            reasons.append(
                f"相似历史样本均价：日前 {similar_reference['day_ahead_avg']}，实时 {similar_reference['real_time_avg']}"
            )

        if row.get("day_ahead_source") == "forecast":
            reasons.append(
                f"日前价缺失，已用 XGBoost+LSTM 架构混合集成预测；置信度 {row.get('day_ahead_confidence')}"
            )
        if real_time_source == "forecast":
            reasons.append(
                f"实时价采用{real_time_forecast['method_name']}预测，区间 {real_time_forecast['lower']}-{real_time_forecast['upper']}"
            )

        hourly_advice.append(
            {
                "hour": hour,
                "stance": stance,
                "risk_level": risk_level,
                "day_ahead_price": day_price,
                "real_time_price": real_time_price,
                "spread_real_minus_day_ahead": spread,
                "day_ahead_source": row.get("day_ahead_source", "actual"),
                "real_time_source": real_time_source,
                "day_ahead_confidence": row.get("day_ahead_confidence", 1.0),
                "xgboost_price": row.get("xgboost_price"),
                "lstm_price": row.get("lstm_price"),
                "baseline_price": row.get("baseline_price"),
                "real_time_forecast": real_time_forecast,
                "day_ahead_percentile": price_percentile,
                "recent_7d_same_hour_avg": recent_avg,
                "previous_day_change": previous_change,
                "load_forecast_mw": load,
                "renewable_forecast_mw": renewable,
                "hydro_forecast_mw": hydro,
                "intertie_plan_mw": intertie,
                "supply_total_mw": round(supply_total, 3),
                "demand_total_mw": round(demand_total, 3),
                "tightness_score": tight_score,
                "tightness_label": tight_label,
                "quote_lower": quote["lower"],
                "quote_center": quote["center"],
                "quote_upper": quote["upper"],
                "similar_reference": similar_reference,
                "non_market_generation_mw": non_market,
                "positive_reserve_mw": reserve,
                "reasons": reasons,
            }
        )

    headline = "广西现货价格呈现日内分化"
    if summary["day_ahead"]["range"] is not None and summary["day_ahead"]["range"] > 250:
        headline = "广西现货日前价格峰谷差较大，需分时段报价"
    if summary["real_time"]["avg"] is not None and summary["day_ahead"]["avg"] is not None:
        if summary["real_time"]["avg"] > summary["day_ahead"]["avg"] + 30:
            headline = "实时均价高于日前，需关注偏差暴露风险"
        elif summary["real_time"]["avg"] < summary["day_ahead"]["avg"] - 30:
            headline = "实时均价低于日前，午间及低价时段需谨慎"

    high_risk_hours = [item["hour"] for item in hourly_advice if item["risk_level"] == "高"]
    low_price_hours = [
        item["hour"]
        for item in hourly_advice
        if item["day_ahead_percentile"] is not None and item["day_ahead_percentile"] <= 20
    ]
    high_price_hours = [
        item["hour"]
        for item in hourly_advice
        if item["day_ahead_percentile"] is not None and item["day_ahead_percentile"] >= 85
    ]

    narrative = [
        f"日前均价 {summary['day_ahead']['avg']} 元/MWh，峰谷差 {summary['day_ahead']['range']} 元/MWh。",
    ]
    if summary["real_time"]["avg"] is not None:
        narrative.append(
            f"实时均价 {summary['real_time']['avg']} 元/MWh，平均价差 {summary['spread_real_minus_day_ahead']['avg']} 元/MWh。"
        )
    if summary["real_time"]["avg"] is None:
        narrative.append(f"该日暂无实时出清价，实时价格使用 {real_time_method} 方案生成预测区间。")
    if prices and prices[0].get("day_ahead_source") == "forecast":
        narrative.append("该日暂无日前出清价，日前价格已用 XGBoost + LSTM 混合集成模型预测。")
    if high_price_hours:
        narrative.append("高分位时段集中在 " + "、".join(f"{h}:00" for h in high_price_hours[:6]) + "。")
    if low_price_hours:
        narrative.append("低分位时段集中在 " + "、".join(f"{h}:00" for h in low_price_hours[:6]) + "。")

    return {
        "market_date": market_date,
        "mode": "target_day" if summary["real_time"]["avg"] is None else "review",
        "real_time_method": real_time_method,
        "day_ahead_source": prices[0].get("day_ahead_source", "actual") if prices else None,
        "headline": headline,
        "summary": summary,
        "narrative": narrative,
        "high_risk_hours": high_risk_hours,
        "supply_series": supply,
        "hourly_advice": hourly_advice,
    }
