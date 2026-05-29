from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from app.db import DB_PATH, get_connection, init_db, reset_db


DATE_DASH_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
DATE_COMPACT_RE = re.compile(r"(\d{4})(\d{2})(\d{2})")
TIME_RE = re.compile(r"^(?:[01]?\d|2[0-3]):(?:00|15|30|45)$")
HOUR_RE = re.compile(r"^(?:[01]?\d|2[0-3]):00$")

RAW_EXCEL_EXTENSIONS = {".xls", ".xlsx", ".xlsm"}
REGIONS_TO_KEEP = {"广西", "全区域", "南网"}


@dataclass
class ImportStats:
    files_seen: int = 0
    price_rows: int = 0
    curve_15min_rows: int = 0
    curve_hourly_rows: int = 0
    skipped_files: int = 0
    errors: int = 0


def normalize_date_from_name(name: str) -> str | None:
    match = DATE_DASH_RE.search(name)
    if match:
        return "-".join(match.groups())
    match = DATE_COMPACT_RE.search(name)
    if match:
        y, m, d = match.groups()
        return f"{y}-{m}-{d}"
    return None


def normalize_time(value: object) -> str | None:
    if pd.isna(value):
        return None
    if hasattr(value, "strftime"):
        return value.strftime("%H:%M")
    text = str(value).strip()
    if not text:
        return None
    if text.startswith("1900-01-01 "):
        text = text[-8:-3]
    if len(text) >= 5:
        text = text[:5]
    if re.match(r"^(?:[01]?\d|2[0-3]):[0-5]\d$", text):
        hour, minute = text.split(":")
        return f"{int(hour):02d}:{int(minute):02d}"
    return None


def quarter_index(time_label: str) -> int:
    hour, minute = [int(part) for part in time_label.split(":")]
    return hour * 4 + minute // 15


def hour_from_time(time_label: str) -> int:
    return int(time_label.split(":")[0])


def safe_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, str):
        value = value.strip().replace(",", "")
        if not value:
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def raw_excel_files(raw_root: Path) -> Iterable[Path]:
    for path in raw_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in RAW_EXCEL_EXTENSIONS:
            continue
        rel_parts = path.relative_to(raw_root).parts
        if rel_parts and rel_parts[0].startswith("0"):
            continue
        yield path


def read_sheet(path: Path) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name=0, header=None)


def find_header_row(df: pd.DataFrame, markers: tuple[str, ...]) -> int | None:
    for idx in range(min(len(df), 12)):
        row_text = " ".join(str(x).strip() for x in df.iloc[idx].dropna().tolist())
        if all(marker in row_text for marker in markers):
            return idx
    return None


def import_price_file(conn, path: Path, raw_root: Path) -> int:
    market_date = normalize_date_from_name(path.name)
    if not market_date:
        return 0

    name = path.name
    if "日前交易交易结果-用电侧" in name:
        market_type = "day_ahead"
        time_marker = "时刻"
    elif "实时交易结果查询用电侧" in name:
        market_type = "real_time"
        time_marker = "时刻"
    else:
        return 0

    df = read_sheet(path)
    header_idx = find_header_row(df, (time_marker, "用户侧均价"))
    if header_idx is None:
        return 0

    rows = 0
    source_file = str(path.relative_to(raw_root))
    for _, row in df.iloc[header_idx + 1 :].iterrows():
        time_label = normalize_time(row.iloc[0])
        if not time_label or not HOUR_RE.match(time_label):
            continue
        price = safe_float(row.iloc[2] if market_type == "day_ahead" and len(row) > 2 else row.iloc[1])
        if price is None:
            continue
        volume = safe_float(row.iloc[1]) if market_type == "day_ahead" and len(row) > 1 else None
        conn.execute(
            """
            INSERT OR IGNORE INTO spot_price_hourly
            (market_date, market_type, hour, time_label, price_yuan_mwh, volume_mwh, source_file)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (market_date, market_type, hour_from_time(time_label), time_label, price, volume, source_file),
        )
        rows += 1
    return rows


def data_type_from_name(path: Path) -> str | None:
    name = path.name
    if "信息披露-跨省跨区中长期计划" in name:
        return "intertie_plan"
    if "水电周预测出力" in name:
        return "hydro_forecast"
    if "统调负荷" in name:
        return "load_forecast" if "实际运行" not in name else "load_actual"
    if "发电总出力预测" in name:
        return "generation_forecast"
    if "发电总出力" in name and "预测" not in name:
        return "generation_actual"
    if "新能源总出力" in name:
        return "renewable_forecast" if "实际运行" not in name else "renewable_actual"
    if "非市场化机组总出力" in name or "非市场机组总出力" in name:
        return "non_market_generation_forecast" if "实际" not in str(path) else "non_market_generation_actual"
    if "备用信息" in name:
        return "reserve"
    return None


def insert_curve_15min(conn, market_date: str, data_type: str, region: str, time_label: str, value: float, source_file: str) -> None:
    conn.execute(
        """
        INSERT OR IGNORE INTO power_curve_15min
        (market_date, data_type, region, time_point, quarter_index, value_mw, source_file)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (market_date, data_type, region, time_label, quarter_index(time_label), value, source_file),
    )


def expand_hour_to_quarters(hour: int) -> list[str]:
    return [f"{hour:02d}:{minute:02d}" for minute in (0, 15, 30, 45)]


def import_hydro_forecast(conn, path: Path, raw_root: Path, market_date: str) -> int:
    df = pd.read_excel(path, sheet_name=0)
    if "所属区域" not in df.columns:
        return 0
    value_col = next((col for col in df.columns if "平均出力" in str(col)), None)
    if value_col is None:
        return 0
    row = df[df["所属区域"].astype(str).str.strip() == "广西"]
    if row.empty:
        return 0
    value = safe_float(row.iloc[0][value_col])
    if value is None:
        return 0
    source_file = str(path.relative_to(raw_root))
    inserted = 0
    for hour in range(24):
        for time_label in expand_hour_to_quarters(hour):
            insert_curve_15min(conn, market_date, "hydro_forecast", "广西", time_label, value, source_file)
            inserted += 1
    return inserted


def import_intertie_plan(conn, path: Path, raw_root: Path, market_date: str) -> int:
    df = pd.read_excel(path, sheet_name=0)
    if "断面名称" not in df.columns:
        return 0
    row = df[df["断面名称"].astype(str).str.strip() == "广西受西电"]
    if row.empty:
        return 0
    source_file = str(path.relative_to(raw_root))
    inserted = 0
    for hour in range(24):
        hour_label = f"{hour:02d}:00受端出力"
        if hour_label not in df.columns:
            continue
        value = safe_float(row.iloc[0][hour_label])
        if value is None:
            continue
        for time_label in expand_hour_to_quarters(hour):
            insert_curve_15min(conn, market_date, "intertie_plan", "广西", time_label, value, source_file)
            inserted += 1
    return inserted


def current_date_column(columns: list[object], market_date: str, region: str) -> int | None:
    compact = market_date.replace("-", "")
    candidates = []
    for idx, col in enumerate(columns):
        text = str(col).strip()
        if compact in text and region in text:
            candidates.append(idx)
    return candidates[0] if candidates else None


def import_vertical_curve(conn, path: Path, raw_root: Path, data_type: str, market_date: str) -> int:
    df = pd.read_excel(path, sheet_name=0)
    columns = list(df.columns)
    source_file = str(path.relative_to(raw_root))
    inserted = 0

    if "时刻" not in columns:
        return 0

    # Format A: columns include 广西 / 全区域 directly.
    direct_regions = [region for region in REGIONS_TO_KEEP if region in columns]
    for region in direct_regions:
        for _, row in df.iterrows():
            time_label = normalize_time(row.get("时刻"))
            value = safe_float(row.get(region))
            if not time_label or not TIME_RE.match(time_label) or value is None:
                continue
            conn.execute(
                """
                INSERT OR IGNORE INTO power_curve_15min
                (market_date, data_type, region, time_point, quarter_index, value_mw, source_file)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (market_date, data_type, region, time_label, quarter_index(time_label), value, source_file),
            )
            inserted += 1

    if inserted:
        return inserted

    # Format B: rows include 所属区域, value columns include YYYYMMDD预测.
    if "所属区域" in columns:
        for _, row in df.iterrows():
            region = str(row.get("所属区域", "")).strip()
            if region not in REGIONS_TO_KEEP:
                continue
            time_label = normalize_time(row.get("时刻"))
            if not time_label or not TIME_RE.match(time_label):
                continue
            value_col = None
            for idx, col in enumerate(columns):
                if market_date.replace("-", "") in str(col):
                    value_col = col
                    break
            if value_col is None:
                continue
            value = safe_float(row.get(value_col))
            if value is None:
                continue
            conn.execute(
                """
                INSERT OR IGNORE INTO power_curve_15min
                (market_date, data_type, region, time_point, quarter_index, value_mw, source_file)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (market_date, data_type, region, time_label, quarter_index(time_label), value, source_file),
            )
            inserted += 1

    # Format C: actual curves have columns like 20260301广西.
    for region in REGIONS_TO_KEEP:
        value_idx = current_date_column(columns, market_date, region)
        if value_idx is None:
            continue
        value_col = columns[value_idx]
        for _, row in df.iterrows():
            time_label = normalize_time(row.get("时刻"))
            value = safe_float(row.get(value_col))
            if not time_label or not TIME_RE.match(time_label) or value is None:
                continue
            conn.execute(
                """
                INSERT OR IGNORE INTO power_curve_15min
                (market_date, data_type, region, time_point, quarter_index, value_mw, source_file)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (market_date, data_type, region, time_label, quarter_index(time_label), value, source_file),
            )
            inserted += 1

    return inserted


def import_wide_reserve(conn, path: Path, raw_root: Path, market_date: str) -> int:
    df = pd.read_excel(path, sheet_name=0)
    columns = list(df.columns)
    time_columns = [col for col in columns if TIME_RE.match(str(col).strip()[:5])]
    if "所属区域" not in columns or "类型" not in columns or not time_columns:
        return 0

    source_file = str(path.relative_to(raw_root))
    inserted = 0
    for _, row in df.iterrows():
        region = str(row.get("所属区域", "")).strip()
        if region != "广西":
            continue
        reserve_type = str(row.get("类型", "")).strip()
        if reserve_type not in {"正备用", "负备用"}:
            continue
        data_type = "reserve_positive" if reserve_type == "正备用" else "reserve_negative"
        for col in time_columns:
            time_label = normalize_time(col)
            value = safe_float(row.get(col))
            if not time_label or value is None:
                continue
            conn.execute(
                """
                INSERT OR IGNORE INTO power_curve_15min
                (market_date, data_type, region, time_point, quarter_index, value_mw, source_file)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (market_date, data_type, region, time_label, quarter_index(time_label), value, source_file),
            )
            inserted += 1
    return inserted


def import_curve_file(conn, path: Path, raw_root: Path) -> int:
    market_date = normalize_date_from_name(path.name)
    data_type = data_type_from_name(path)
    if not market_date or not data_type:
        return 0

    if data_type == "hydro_forecast":
        return import_hydro_forecast(conn, path, raw_root, market_date)
    if data_type == "intertie_plan":
        return import_intertie_plan(conn, path, raw_root, market_date)
    if data_type == "reserve":
        return import_wide_reserve(conn, path, raw_root, market_date)
    return import_vertical_curve(conn, path, raw_root, data_type, market_date)


def rebuild_hourly_curves(conn) -> int:
    conn.execute("DELETE FROM power_curve_hourly")
    conn.execute(
        """
        INSERT INTO power_curve_hourly
        (market_date, data_type, region, hour, value_mw_avg, source_15min_count, source_file)
        SELECT
            market_date,
            data_type,
            region,
            CAST(SUBSTR(time_point, 1, 2) AS INTEGER) AS hour,
            AVG(value_mw) AS value_mw_avg,
            COUNT(*) AS source_15min_count,
            source_file
        FROM power_curve_15min
        GROUP BY market_date, data_type, region, hour, source_file
        """
    )
    return conn.execute("SELECT COUNT(*) FROM power_curve_hourly").fetchone()[0]


def import_raw_data(raw_root: Path, db_path: Path = DB_PATH, reset: bool = False) -> ImportStats:
    stats = ImportStats()
    conn = get_connection(db_path)
    init_db(conn)
    if reset:
        reset_db(conn)

    files = list(raw_excel_files(raw_root))
    stats.files_seen = len(files)
    for path in files:
        try:
            price_rows = import_price_file(conn, path, raw_root)
            curve_rows = 0 if price_rows else import_curve_file(conn, path, raw_root)
            stats.price_rows += price_rows
            stats.curve_15min_rows += curve_rows
            if not price_rows and not curve_rows:
                stats.skipped_files += 1
        except Exception as exc:  # noqa: BLE001 - importer should keep scanning.
            stats.errors += 1
            print(f"[WARN] failed: {path} :: {exc}")

    stats.curve_hourly_rows = rebuild_hourly_curves(conn)
    conn.execute(
        """
        INSERT INTO import_runs
        (raw_root, files_seen, price_rows, curve_15min_rows, curve_hourly_rows, notes)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            str(raw_root),
            stats.files_seen,
            stats.price_rows,
            stats.curve_15min_rows,
            stats.curve_hourly_rows,
            f"skipped={stats.skipped_files}; errors={stats.errors}",
        ),
    )
    conn.commit()
    conn.close()
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Import Guangxi spot-market raw Excel files.")
    parser.add_argument("--raw-root", type=Path, default=Path.cwd() / "现货交易电网信息")
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    stats = import_raw_data(args.raw_root, args.db, reset=args.reset)
    print("Import finished")
    print(f"files_seen={stats.files_seen}")
    print(f"price_rows={stats.price_rows}")
    print(f"curve_15min_rows={stats.curve_15min_rows}")
    print(f"curve_hourly_rows={stats.curve_hourly_rows}")
    print(f"skipped_files={stats.skipped_files}")
    print(f"errors={stats.errors}")
    print(f"db={args.db}")


if __name__ == "__main__":
    main()
