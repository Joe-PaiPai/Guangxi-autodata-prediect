from __future__ import annotations

import shutil
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.analytics import (
    available_dates,
    daily_prices,
    data_quality,
    data_quality_diagnostics,
    history_trend,
    hourly_curves,
    import_status,
    price_summary,
    strategy_report,
)
from app.db import get_connection, init_db
from app.forecasting import (
    REAL_TIME_METHODS,
    evaluate_day_ahead_model,
    model_status,
    predict_day_ahead_prices,
    train_day_ahead_models,
)
from app.importer import import_raw_data
from app.report_export import export_strategy_docx


app = FastAPI(title="广西现货交易辅助决策 API", version="0.1.0")
ROOT_DIR = Path(__file__).resolve().parents[2]
FRONTEND_DIR = ROOT_DIR / "frontend"
UPLOAD_DIR = ROOT_DIR / "data" / "uploads"
RAW_EXCEL_EXTENSIONS = {".xls", ".xlsx", ".xlsm"}
RAW_DATA_DIR = ROOT_DIR / "现货交易电网信息"

if FRONTEND_DIR.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR), name="assets")


def _import_response(stats) -> dict:
    return {
        "message": "导入完成",
        "files_seen": stats.files_seen,
        "price_rows": stats.price_rows,
        "curve_15min_rows": stats.curve_15min_rows,
        "curve_hourly_rows": stats.curve_hourly_rows,
        "skipped_files": stats.skipped_files,
        "errors": stats.errors,
    }


@app.post("/api/import/upload")
async def upload_and_import(file: UploadFile = File(...)) -> dict:
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="请上传 ZIP 压缩包")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    upload_path = UPLOAD_DIR / "latest_raw_data.zip"
    with open(upload_path, "wb") as dst:
        while chunk := await file.read(1024 * 1024):
            dst.write(chunk)

    upload_stats = _replace_raw_data_from_zip(upload_path)
    stats = import_raw_data(RAW_DATA_DIR, reset=True)
    return {**_import_response(stats), **upload_stats}


def _safe_zip_parts(raw_name: str) -> tuple[str, ...] | None:
    name = raw_name.replace("\\", "/").strip("/")
    if not name or name.endswith("/"):
        return None
    parts = PurePosixPath(name).parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        return None
    if parts[0] != RAW_DATA_DIR.name:
        return None
    if len(parts) > 1 and parts[1].startswith("0"):
        return None
    if Path(parts[-1]).suffix.lower() not in RAW_EXCEL_EXTENSIONS:
        return None
    return parts[1:]


def _replace_raw_data_from_zip(zip_path: Path) -> dict:
    temp_dir = UPLOAD_DIR / "raw_extract_tmp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    extracted = 0
    skipped = 0
    try:
        with ZipFile(zip_path) as archive:
            for info in archive.infolist():
                parts = _safe_zip_parts(info.filename)
                if not parts:
                    skipped += 1
                    continue

                target = temp_dir.joinpath(*parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                extracted += 1
    except BadZipFile as exc:
        raise HTTPException(status_code=400, detail="上传文件不是有效 ZIP 压缩包") from exc

    if extracted == 0:
        raise HTTPException(status_code=400, detail="ZIP 中没有可导入的原始 Excel 文件")

    if RAW_DATA_DIR.exists():
        shutil.rmtree(RAW_DATA_DIR)
    temp_dir.rename(RAW_DATA_DIR)
    return {"uploaded_excel_files": extracted, "skipped_zip_entries": skipped}


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


@app.get("/api/import/status")
def get_import_status() -> dict:
    with get_connection() as conn:
        return import_status(conn)


@app.post("/api/import")
def run_import() -> dict:
    if not RAW_DATA_DIR.exists():
        raise HTTPException(status_code=404, detail="未找到原始数据目录")
    stats = import_raw_data(RAW_DATA_DIR, reset=True)
    return _import_response(stats)


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


@app.get("/api/quality/diagnostics")
def get_quality_diagnostics(limit: int = 20) -> dict:
    with get_connection() as conn:
        return data_quality_diagnostics(conn, limit=max(1, min(limit, 90)))


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


@app.get("/api/models/status")
def get_model_status() -> dict:
    return model_status()


@app.post("/api/models/train")
def train_models() -> dict:
    with get_connection() as conn:
        try:
            return train_day_ahead_models(conn)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


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
