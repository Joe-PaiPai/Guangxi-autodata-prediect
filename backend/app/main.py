from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.analytics import (
    available_dates,
    daily_prices,
    data_quality,
    history_trend,
    hourly_curves,
    price_summary,
    strategy_report,
)
from app.db import get_connection, init_db
from app.forecasting import REAL_TIME_METHODS, evaluate_day_ahead_model, predict_day_ahead_prices
from app.importer import import_raw_data
from app.report_export import export_strategy_docx


app = FastAPI(title="广西现货交易辅助决策 API", version="0.1.0")
ROOT_DIR = Path(__file__).resolve().parents[2]
FRONTEND_DIR = ROOT_DIR / "frontend"
RAW_DATA_DIR = ROOT_DIR / "现货交易电网信息"

if FRONTEND_DIR.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR), name="assets")


@app.on_event("startup")
def startup() -> None:
    with get_connection() as conn:
        init_db(conn)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/dates")
def get_dates() -> list[dict]:
    with get_connection() as conn:
        return available_dates(conn)


@app.post("/api/import")
def run_import() -> dict:
    if not RAW_DATA_DIR.exists():
        raise HTTPException(status_code=404, detail="未找到原始数据目录")
    stats = import_raw_data(RAW_DATA_DIR, reset=True)
    return {
        "message": "导入完成",
        "files_seen": stats.files_seen,
        "price_rows": stats.price_rows,
        "curve_15min_rows": stats.curve_15min_rows,
        "curve_hourly_rows": stats.curve_hourly_rows,
        "skipped_files": stats.skipped_files,
        "errors": stats.errors,
    }


@app.get("/api/prices/{market_date}")
def get_prices(market_date: str, real_time_method: str = "spread_follow") -> list[dict]:
    with get_connection() as conn:
        rows = daily_prices(conn, market_date, real_time_method=real_time_method)
    if not rows:
        raise HTTPException(status_code=404, detail="该日期没有日前价格数据")
    return rows


@app.get("/api/curves/{market_date}")
def get_curves(market_date: str) -> list[dict]:
    with get_connection() as conn:
        return hourly_curves(conn, market_date)


@app.get("/api/summary/{market_date}")
def get_summary(market_date: str) -> dict:
    with get_connection() as conn:
        summary = price_summary(conn, market_date)
    if summary["day_ahead"]["avg"] is None:
        raise HTTPException(status_code=404, detail="该日期没有价格数据")
    return summary


@app.get("/api/history/{market_date}")
def get_history(market_date: str, days: int = 14) -> list[dict]:
    with get_connection() as conn:
        return history_trend(conn, market_date, days=days)


@app.get("/api/quality/{market_date}")
def get_quality(market_date: str) -> dict:
    with get_connection() as conn:
        return data_quality(conn, market_date)


@app.get("/api/report/{market_date}")
def get_report(market_date: str, real_time_method: str = "spread_follow") -> dict:
    with get_connection() as conn:
        report = strategy_report(conn, market_date, real_time_method=real_time_method)
    if report["summary"]["day_ahead"]["avg"] is None:
        raise HTTPException(status_code=404, detail="该日期没有价格数据")
    return report


@app.get("/api/forecast/day-ahead/{market_date}")
def get_day_ahead_forecast(market_date: str) -> dict:
    with get_connection() as conn:
        return predict_day_ahead_prices(conn, market_date)


@app.get("/api/forecast/real-time-methods")
def get_real_time_methods() -> dict:
    return REAL_TIME_METHODS


@app.get("/api/evaluation/day-ahead")
def get_day_ahead_evaluation(end_date: str | None = None, days: int = 5) -> dict:
    with get_connection() as conn:
        return evaluate_day_ahead_model(conn, end_date=end_date, days=days)


@app.get("/api/export/report/{market_date}")
def export_report(market_date: str, real_time_method: str = "spread_follow") -> FileResponse:
    with get_connection() as conn:
        output_path = export_strategy_docx(conn, market_date, real_time_method=real_time_method)
    return FileResponse(
        output_path,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename=output_path.name,
    )


@app.get("/")
def index() -> FileResponse:
    index_file = FRONTEND_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="前端页面尚未创建")
    return FileResponse(index_file)
