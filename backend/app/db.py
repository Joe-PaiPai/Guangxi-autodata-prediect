from __future__ import annotations

import sqlite3
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
DB_PATH = DATA_DIR / "guangxi_spot.db"


def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    """Open a SQLite connection with row dictionaries enabled."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path or DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Create the first-version analytical tables."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS spot_price_hourly (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            market_date TEXT NOT NULL,
            market_type TEXT NOT NULL CHECK (market_type IN ('day_ahead', 'real_time')),
            hour INTEGER NOT NULL CHECK (hour BETWEEN 0 AND 23),
            time_label TEXT NOT NULL,
            price_yuan_mwh REAL NOT NULL,
            volume_mwh REAL,
            source_file TEXT NOT NULL,
            imported_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (market_date, market_type, hour, source_file)
        );

        CREATE INDEX IF NOT EXISTS idx_spot_price_date_type
            ON spot_price_hourly (market_date, market_type, hour);

        CREATE TABLE IF NOT EXISTS power_curve_15min (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            market_date TEXT NOT NULL,
            data_type TEXT NOT NULL,
            region TEXT NOT NULL,
            time_point TEXT NOT NULL,
            quarter_index INTEGER NOT NULL CHECK (quarter_index BETWEEN 0 AND 95),
            value_mw REAL NOT NULL,
            source_file TEXT NOT NULL,
            imported_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (market_date, data_type, region, time_point, source_file)
        );

        CREATE INDEX IF NOT EXISTS idx_power_curve_15min_lookup
            ON power_curve_15min (market_date, data_type, region, quarter_index);

        CREATE TABLE IF NOT EXISTS power_curve_hourly (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            market_date TEXT NOT NULL,
            data_type TEXT NOT NULL,
            region TEXT NOT NULL,
            hour INTEGER NOT NULL CHECK (hour BETWEEN 0 AND 23),
            value_mw_avg REAL NOT NULL,
            source_15min_count INTEGER NOT NULL,
            source_file TEXT NOT NULL,
            imported_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (market_date, data_type, region, hour, source_file)
        );

        CREATE INDEX IF NOT EXISTS idx_power_curve_hourly_lookup
            ON power_curve_hourly (market_date, data_type, region, hour);

        CREATE TABLE IF NOT EXISTS import_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            raw_root TEXT NOT NULL,
            files_seen INTEGER NOT NULL,
            price_rows INTEGER NOT NULL,
            curve_15min_rows INTEGER NOT NULL,
            curve_hourly_rows INTEGER NOT NULL,
            notes TEXT
        );
        """
    )
    conn.commit()


def reset_db(conn: sqlite3.Connection) -> None:
    """Delete imported analytical data while keeping table definitions."""
    conn.executescript(
        """
        DELETE FROM spot_price_hourly;
        DELETE FROM power_curve_15min;
        DELETE FROM power_curve_hourly;
        DELETE FROM import_runs;
        """
    )
    conn.commit()
