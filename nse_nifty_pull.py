#!/usr/bin/env python3
"""Pull NIFTY 50 prices, constituents, and NSE corporate filings."""

from __future__ import annotations

import argparse
import html
from html.parser import HTMLParser
import io
import json
import math
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable
from urllib.parse import quote, urlparse
import zipfile

import numpy as np
import pandas as pd
import requests
from scipy.optimize import minimize


NSE_HOME = "https://www.nseindia.com"
NSE_ARCHIVES = "https://archives.nseindia.com"
NSE_SEARCHIVES = "https://nsearchives.nseindia.com"
BSE_HOME = "https://www.bseindia.com"
BSE_API = "https://api.bseindia.com"
MONEYCONTROL_HOME = "https://www.moneycontrol.com"
ECONOMIC_TIMES_HOME = "https://economictimes.indiatimes.com"
SCREENER_HOME = "https://www.screener.in"
NIFTY_50_INDEX = "NIFTY 50"
FRED_GRAPH_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"

ECONOMIC_SERIES = [
    {
        "key": "cpi_yoy_pct",
        "fred_id": "INDCPIALLMINMEI",
        "label": "India CPI YoY",
        "source": "FRED/OECD Main Economic Indicators",
        "transform": "yoy_pct",
        "component": "inflation_score",
        "weight": 0.20,
        "target": 4.0,
        "higher_is_better": False,
        "max_age_days": 548,
    },
    {
        "key": "iip_yoy_pct",
        "fred_id": "INDPROINDMISMEI",
        "label": "India Industrial Production YoY",
        "source": "FRED/OECD Main Economic Indicators",
        "transform": "yoy_pct",
        "component": "industrial_score",
        "weight": 0.30,
        "higher_is_better": True,
        "max_age_days": 548,
    },
    {
        "key": "exports_yoy_pct",
        "fred_id": "XTEXVA01INM667S",
        "label": "India Exports YoY",
        "source": "FRED/OECD Main Economic Indicators",
        "transform": "yoy_pct",
        "component": "exports_score",
        "weight": 0.20,
        "higher_is_better": True,
        "max_age_days": 548,
    },
    {
        "key": "short_rate_pct",
        "fred_id": "IRSTCI01INM156N",
        "label": "India Short-Term Interest Rate",
        "source": "FRED/OECD Main Economic Indicators",
        "transform": "level",
        "component": "rate_support_score",
        "weight": 0.10,
        "higher_is_better": False,
        "max_age_days": 548,
    },
    {
        "key": "gdp_yoy_pct",
        "fred_id": "MKTGDPINA646NWDB",
        "label": "India GDP YoY",
        "source": "FRED/World Bank",
        "transform": "yoy_pct",
        "component": "gdp_score",
        "weight": 0.20,
        "higher_is_better": True,
        "max_age_days": 1095,
    },
]


@dataclass(frozen=True)
class PullResult:
    constituents_path: Path | None
    prices_path: Path | None
    constituents: pd.DataFrame
    prices: pd.DataFrame
    constituents_paths: tuple[Path, ...] = ()
    prices_paths: tuple[Path, ...] = ()


@dataclass(frozen=True)
class FundamentalsResult:
    financial_results_path: Path
    shareholding_path: Path
    financial_announcements_path: Path
    financial_results: pd.DataFrame
    shareholding: pd.DataFrame
    financial_announcements: pd.DataFrame


@dataclass(frozen=True)
class FundamentalScoreResult:
    scores_path: Path
    scores: pd.DataFrame


@dataclass(frozen=True)
class FundamentalScoreHistoryResult:
    history_path: Path
    score_paths: tuple[Path, ...]
    scores: pd.DataFrame


@dataclass(frozen=True)
class TechnicalScoreHistoryResult:
    history_path: Path
    score_paths: tuple[Path, ...]
    scores: pd.DataFrame


@dataclass(frozen=True)
class IndexHistoryResult:
    history_path: Path | None
    index_paths: tuple[Path, ...]
    index_history: pd.DataFrame


@dataclass(frozen=True)
class EconomyScoreResult:
    variables_path: Path
    history_path: Path
    scores: pd.DataFrame


@dataclass(frozen=True)
class BacktestResult:
    backtest_paths: tuple[Path, ...]
    holdings_paths: tuple[Path, ...]
    symbol_metrics_paths: tuple[Path, ...]
    selected_metrics_path: Path


class NSEArchiveClient:
    """Small NSE client for archive CSVs and corporate filing JSON endpoints."""

    def __init__(self, timeout: int = 20, pause_seconds: float = 0.35) -> None:
        self.timeout = timeout
        self.pause_seconds = pause_seconds
        self.session = requests.Session()
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            ),
            "Accept": "text/csv,application/zip,*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": f"{NSE_HOME}/",
        }
        self._handshake_done = False

    def _handshake(self) -> None:
        if self._handshake_done:
            return
        response = self.session.get(NSE_HOME, headers=self.headers, timeout=self.timeout)
        response.raise_for_status()
        self._handshake_done = True

    def get_bytes(self, url: str, allow_missing: bool = False) -> bytes | None:
        last_error: Exception | None = None

        for attempt in range(3):
            try:
                response = self.session.get(url, headers=self.headers, timeout=self.timeout)
                if allow_missing and response.status_code == 404:
                    return None
                response.raise_for_status()
                return response.content
            except requests.RequestException as exc:
                last_error = exc
                time.sleep(self.pause_seconds * (attempt + 1))

        raise RuntimeError(f"NSE request failed for {url}: {last_error}") from last_error

    def get_json(self, path: str, params: dict[str, str]) -> list[dict] | dict:
        self._handshake()
        url = f"{NSE_HOME}{path}"
        headers = {**self.headers, "Accept": "application/json,text/plain,*/*"}
        last_error: Exception | None = None

        for attempt in range(3):
            try:
                response = self.session.get(url, params=params, headers=headers, timeout=self.timeout)
                if response.status_code in {401, 403}:
                    self._handshake_done = False
                    self._handshake()
                    response = self.session.get(url, params=params, headers=headers, timeout=self.timeout)
                response.raise_for_status()
                return response.json()
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                time.sleep(self.pause_seconds * (attempt + 1))

        raise RuntimeError(f"NSE JSON request failed for {url}: {last_error}") from last_error


class BSEArchiveClient:
    """Small BSE client for corporate filing metadata endpoints."""

    def __init__(self, timeout: int = 25, pause_seconds: float = 0.45) -> None:
        self.timeout = timeout
        self.pause_seconds = pause_seconds
        self.session = requests.Session()
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            ),
            "Accept": "application/json,text/plain,*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Origin": BSE_HOME,
            "Referer": f"{BSE_HOME}/",
        }

    def get_json(self, path: str, params: dict[str, str]) -> list[dict] | dict:
        url = path if path.startswith("http") else f"{BSE_API}{path}"
        last_error: Exception | None = None

        for attempt in range(3):
            try:
                response = self.session.get(url, params=params, headers=self.headers, timeout=self.timeout)
                response.raise_for_status()
                return response.json()
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                time.sleep(self.pause_seconds * (attempt + 1))

        raise RuntimeError(f"BSE JSON request failed for {url}: {last_error}") from last_error


class SimpleTableParser(HTMLParser):
    """Dependency-free HTML table extractor for provider fallback pages."""

    def __init__(self) -> None:
        super().__init__()
        self.tables: list[list[list[str]]] = []
        self._table_depth = 0
        self._current_table: list[list[str]] = []
        self._current_row: list[str] = []
        self._cell_parts: list[str] = []
        self._in_cell = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table":
            self._table_depth += 1
            if self._table_depth == 1:
                self._current_table = []
        elif self._table_depth and tag == "tr":
            self._current_row = []
        elif self._table_depth and tag in {"td", "th"}:
            self._in_cell = True
            self._cell_parts = []

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._cell_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self._table_depth and tag in {"td", "th"} and self._in_cell:
            text = html.unescape(" ".join(" ".join(self._cell_parts).split()))
            self._current_row.append(text)
            self._in_cell = False
            self._cell_parts = []
        elif self._table_depth and tag == "tr":
            if any(cell for cell in self._current_row):
                self._current_table.append(self._current_row)
            self._current_row = []
        elif tag == "table" and self._table_depth:
            if self._table_depth == 1 and self._current_table:
                self.tables.append(self._current_table)
            self._table_depth -= 1


class WebFundamentalsClient:
    """HTTP client for Moneycontrol and Economic Times fundamental pages."""

    def __init__(self, timeout: int = 25, pause_seconds: float = 0.5) -> None:
        self.timeout = timeout
        self.pause_seconds = pause_seconds
        self.session = requests.Session()
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            ),
            "Accept": "application/json,text/html,*/*",
            "Accept-Language": "en-US,en;q=0.9",
        }

    def get_text(self, url: str, referer: str | None = None) -> str:
        headers = dict(self.headers)
        if referer:
            headers["Referer"] = referer
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = self.session.get(url, headers=headers, timeout=self.timeout)
                response.raise_for_status()
                return response.text
            except requests.RequestException as exc:
                last_error = exc
                time.sleep(self.pause_seconds * (attempt + 1))
        raise RuntimeError(f"Web fundamentals request failed for {url}: {last_error}") from last_error

    def get_json(self, url: str, referer: str | None = None) -> list[dict] | dict:
        text = self.get_text(url, referer=referer)
        try:
            return json.loads(text)
        except ValueError as exc:
            raise RuntimeError(f"JSON response was not returned for {url}") from exc


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def nse_display_date(value: date) -> str:
    return value.strftime("%d-%m-%Y")


def parse_date_series(values: pd.Series) -> pd.Series:
    raw = pd.Series(values)
    text = raw.astype("string").str.strip()
    parsed = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]")

    # ISO dates are emitted by the cached daily CSVs. Parse them explicitly so
    # 2026-06-12 remains June 12 instead of being read as December 6.
    iso_mask = text.str.match(r"^\d{4}-\d{1,2}-\d{1,2}$", na=False)
    if iso_mask.any():
        parsed.loc[iso_mask] = pd.to_datetime(text.loc[iso_mask], errors="coerce", format="%Y-%m-%d")

    compact_iso_mask = text.str.match(r"^\d{8}$", na=False) & parsed.isna()
    if compact_iso_mask.any():
        parsed.loc[compact_iso_mask] = pd.to_datetime(
            text.loc[compact_iso_mask],
            errors="coerce",
            format="%Y%m%d",
        )

    remaining = parsed.isna() & text.notna() & text.ne("")
    if remaining.any():
        try:
            parsed.loc[remaining] = pd.to_datetime(
                text.loc[remaining],
                errors="coerce",
                dayfirst=True,
                format="mixed",
            )
        except TypeError:
            parsed.loc[remaining] = pd.to_datetime(text.loc[remaining], errors="coerce", dayfirst=True)
    return parsed


def parse_result_period_end(text: str) -> pd.Timestamp:
    if not isinstance(text, str) or not text.strip():
        return pd.NaT
    month_pattern = (
        r"(?:January|February|March|April|May|June|July|August|September|October|November|December|"
        r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
    )
    match = pd.Series([text]).str.extract(
        rf"ended\s+({month_pattern})\s+(\d{{1,2}}),?\s+(\d{{4}})",
        flags=2,
        expand=True,
    )
    if match.empty or match.iloc[0].isna().all():
        return pd.NaT
    month = match.iloc[0, 0]
    day = match.iloc[0, 1]
    year = match.iloc[0, 2]
    return pd.to_datetime(f"{month} {day} {year}", errors="coerce")


def parse_filing_period_end(text: str) -> pd.Timestamp:
    parsed = parse_result_period_end(text)
    if pd.notna(parsed):
        return parsed
    if not isinstance(text, str) or not text.strip():
        return pd.NaT

    month_pattern = (
        r"January|February|March|April|May|June|July|August|September|October|November|December|"
        r"Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
    )
    patterns = [
        rf"(\d{{1,2}})(?:st|nd|rd|th)?[\s\-/.]+({month_pattern})[\s,\-/.]+(\d{{4}})",
        rf"({month_pattern})[\s,\-/.]+(\d{{1,2}})(?:st|nd|rd|th)?[\s,\-/.]+(\d{{4}})",
        r"(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})",
        r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        groups = match.groups()
        if re.match(month_pattern, groups[0], flags=re.IGNORECASE):
            candidate = f"{groups[0]} {groups[1]} {groups[2]}"
            return pd.to_datetime(candidate, errors="coerce")
        if len(groups[0]) == 4:
            candidate = f"{groups[0]}-{groups[1]}-{groups[2]}"
            return pd.to_datetime(candidate, errors="coerce")
        candidate = f"{groups[0]} {groups[1]} {groups[2]}"
        return pd.to_datetime(candidate, errors="coerce", dayfirst=True)
    return pd.NaT


def first_present(row: pd.Series, names: Iterable[str], default: str = "") -> str:
    for name in names:
        if name in row.index and pd.notna(row[name]):
            value = str(row[name]).strip()
            if value and value.lower() != "nan":
                return value
    return default


def normalize_bse_payload(payload: list[dict] | dict) -> pd.DataFrame:
    if isinstance(payload, list):
        return pd.DataFrame(payload)
    if not isinstance(payload, dict):
        return pd.DataFrame()
    for key in ("Table", "Table1", "data", "Data", "List", "Result"):
        value = payload.get(key)
        if isinstance(value, list):
            return pd.DataFrame(value)
    rows = [value for value in payload.values() if isinstance(value, list)]
    return pd.DataFrame(rows[0]) if rows else pd.DataFrame()


def bse_attachment_url(value: str) -> str:
    value = str(value or "").strip()
    if not value:
        return ""
    if value.lower().startswith("http"):
        return value
    return f"{BSE_HOME}/xml-data/corpfiling/AttachLive/{value}"


def bse_xbrl_url(value: str) -> str:
    value = str(value or "").strip()
    if not value or value.lower() == "nan":
        return ""
    if not re.search(r"xbrl|xml", value, flags=re.IGNORECASE):
        return ""
    if value.lower().startswith("http"):
        return value
    return f"{BSE_HOME}/xml-data/corpfiling/AttachLive/{value}"


def iter_dates(start: date, end: date) -> Iterable[date]:
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def get_nifty_constituents(client: NSEArchiveClient, index_name: str = NIFTY_50_INDEX) -> pd.DataFrame:
    if index_name != NIFTY_50_INDEX:
        raise ValueError("Only NIFTY 50 archive constituent pulls are currently supported.")

    url = f"{NSE_ARCHIVES}/content/indices/ind_nifty50list.csv"
    content = client.get_bytes(url)
    if content is None:
        raise RuntimeError("NIFTY 50 constituent CSV was not returned by NSE archives.")

    df = pd.read_csv(io.BytesIO(content))
    if df.empty:
        raise RuntimeError(f"No constituent rows returned for {index_name}.")

    df.columns = [col.strip() for col in df.columns]
    if "Symbol" not in df.columns:
        raise RuntimeError(f"NSE constituent CSV did not include a Symbol column: {list(df.columns)}")

    rename_map = {
        "Company Name": "company_name",
        "Industry": "industry",
        "Symbol": "symbol",
        "Series": "series",
        "ISIN Code": "isin",
    }
    df = df.rename(columns=rename_map)
    present_columns = [col for col in ["symbol", "company_name", "industry", "series", "isin"] if col in df.columns]
    return df[present_columns].sort_values("symbol").reset_index(drop=True)


def get_cash_bhavcopy(client: NSEArchiveClient, trade_date: date, cache_dir: Path) -> pd.DataFrame:
    date_stamp = trade_date.strftime("%Y%m%d")
    cache_path = cache_dir / f"bhavcopy_cm_{date_stamp}.csv"
    if cache_path.exists():
        return pd.read_csv(cache_path)

    url = f"{NSE_SEARCHIVES}/content/cm/BhavCopy_NSE_CM_0_0_0_{date_stamp}_F_0000.csv.zip"
    content = client.get_bytes(url, allow_missing=True)
    if content is None:
        return pd.DataFrame()

    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        csv_name = zf.namelist()[0]
        with zf.open(csv_name) as handle:
            df = pd.read_csv(handle)

    df.columns = [col.strip() for col in df.columns]
    df.to_csv(cache_path, index=False)
    return df


def normalize_bhavcopy(df: pd.DataFrame, symbols: Iterable[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    wanted = {symbol.upper() for symbol in symbols}
    if "TckrSymb" not in df.columns:
        raise RuntimeError(f"Bhavcopy did not include TckrSymb: {list(df.columns)}")

    filtered = df[df["TckrSymb"].astype(str).str.upper().isin(wanted)].copy()
    if "SctySrs" in filtered.columns:
        filtered = filtered[filtered["SctySrs"].astype(str).str.upper().eq("EQ")]
    if filtered.empty:
        return pd.DataFrame()

    rename_map = {
        "TradDt": "trade_date",
        "BizDt": "business_date",
        "TckrSymb": "symbol",
        "SctySrs": "series",
        "ISIN": "isin",
        "FinInstrmNm": "instrument_name",
        "OpnPric": "open",
        "HghPric": "high",
        "LwPric": "low",
        "ClsPric": "close",
        "LastPric": "last",
        "PrvsClsgPric": "previous_close",
        "SttlmPric": "settlement_price",
        "TtlTradgVol": "volume",
        "TtlTrfVal": "turnover",
        "TtlNbOfTxsExctd": "trades",
    }
    filtered = filtered.rename(columns=rename_map)

    ordered_columns = [
        "symbol",
        "trade_date",
        "business_date",
        "series",
        "isin",
        "instrument_name",
        "open",
        "high",
        "low",
        "close",
        "last",
        "previous_close",
        "settlement_price",
        "volume",
        "turnover",
        "trades",
    ]
    present_columns = [col for col in ordered_columns if col in filtered.columns]
    filtered = filtered[present_columns].copy()

    for date_col in ["trade_date", "business_date"]:
        if date_col in filtered.columns:
            filtered[date_col] = pd.to_datetime(filtered[date_col], errors="coerce")

    text_columns = {"symbol", "trade_date", "business_date", "series", "isin", "instrument_name"}
    for col in filtered.columns:
        if col not in text_columns:
            filtered[col] = pd.to_numeric(filtered[col], errors="coerce")

    return filtered.sort_values(["symbol", "trade_date"]).reset_index(drop=True)


def selected_nifty_symbols(
    constituents: pd.DataFrame,
    symbols: Iterable[str] | None = None,
    limit: int | None = None,
) -> list[str]:
    if symbols:
        wanted = {symbol.upper() for symbol in symbols}
        pull_symbols = [symbol for symbol in constituents["symbol"].astype(str) if symbol.upper() in wanted]
    else:
        pull_symbols = constituents["symbol"].astype(str).tolist()

    if limit is not None:
        pull_symbols = pull_symbols[:limit]

    if not pull_symbols:
        raise RuntimeError("No symbols selected for NSE pull.")
    return pull_symbols


def load_latest_cached_constituents(output_dir: Path) -> pd.DataFrame | None:
    path = latest_csv_file(output_dir / "constituents_by_day", "nifty50_constituents_*.csv")
    if path is None:
        path = latest_csv_file(output_dir, "nifty50_constituents_*.csv")
    return pd.read_csv(path) if path is not None else None


def write_daily_outputs(
    prices: pd.DataFrame,
    constituents: pd.DataFrame,
    trade_date: date,
    output_dir: Path,
) -> tuple[Path, Path]:
    prices_dir = output_dir / "prices_by_day"
    constituents_dir = output_dir / "constituents_by_day"
    prices_dir.mkdir(parents=True, exist_ok=True)
    constituents_dir.mkdir(parents=True, exist_ok=True)

    date_stamp = trade_date.strftime("%Y%m%d")
    prices_path = prices_dir / f"nifty50_prices_{date_stamp}.csv"
    constituents_path = constituents_dir / f"nifty50_constituents_{date_stamp}.csv"
    prices.to_csv(prices_path, index=False)
    constituents.to_csv(constituents_path, index=False)
    return constituents_path, prices_path


def get_nifty_index_close(client: NSEArchiveClient, trade_date: date, cache_dir: Path) -> pd.DataFrame:
    date_stamp = trade_date.strftime("%Y%m%d")
    cache_path = cache_dir / f"ind_close_all_{date_stamp}.csv"
    if cache_path.exists():
        raw = pd.read_csv(cache_path)
    else:
        url = f"{NSE_ARCHIVES}/content/indices/ind_close_all_{trade_date.strftime('%d%m%Y')}.csv"
        content = client.get_bytes(url, allow_missing=True)
        if content is None:
            return pd.DataFrame()
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(content)
        raw = pd.read_csv(io.BytesIO(content))

    if raw.empty:
        return pd.DataFrame()
    raw.columns = [str(col).strip() for col in raw.columns]
    name_col = next((col for col in raw.columns if col.lower().replace(" ", "") == "indexname"), None)
    if name_col is None:
        return pd.DataFrame()
    nifty = raw[raw[name_col].astype(str).str.upper().str.strip().eq(NIFTY_50_INDEX)].copy()
    if nifty.empty:
        return pd.DataFrame()

    def first_col(*names: str) -> pd.Series:
        normalized = {col.lower().replace(" ", "").replace(".", "").replace("_", ""): col for col in nifty.columns}
        for name in names:
            key = name.lower().replace(" ", "").replace(".", "").replace("_", "")
            if key in normalized:
                return nifty[normalized[key]]
        return pd.Series(np.nan, index=nifty.index)

    out = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(first_col("Index Date"), errors="coerce", dayfirst=True).dt.date.astype(str),
            "index_name": NIFTY_50_INDEX,
            "open": pd.to_numeric(first_col("Open Index Value"), errors="coerce"),
            "high": pd.to_numeric(first_col("High Index Value"), errors="coerce"),
            "low": pd.to_numeric(first_col("Low Index Value"), errors="coerce"),
            "close": pd.to_numeric(first_col("Closing Index Value"), errors="coerce"),
            "points_change": pd.to_numeric(first_col("Points Change"), errors="coerce"),
            "change_pct": pd.to_numeric(first_col("Change(%)", "Change %"), errors="coerce"),
            "volume": pd.to_numeric(first_col("Volume"), errors="coerce"),
            "turnover_rs_cr": pd.to_numeric(first_col("Turnover (Rs. Cr.)", "Turnover Rs Cr"), errors="coerce"),
            "pe": pd.to_numeric(first_col("P/E", "PE"), errors="coerce"),
            "pb": pd.to_numeric(first_col("P/B", "PB"), errors="coerce"),
            "div_yield": pd.to_numeric(first_col("Div Yield", "Dividend Yield"), errors="coerce"),
            "source": "nse_index_close_archive",
        }
    )
    out.loc[out["trade_date"].eq("NaT") | out["trade_date"].eq("nan"), "trade_date"] = trade_date.isoformat()
    return out.dropna(subset=["close"]).head(1)


def pull_nifty_index_history(
    start: date,
    end: date,
    output_dir: Path = Path("data_cache/nse_equity"),
    daily_files: bool = True,
    latest_available_only: bool = False,
    fallback_days: int = 10,
) -> IndexHistoryResult:
    if end < start:
        raise ValueError("end date must be on or after start date")

    output_dir.mkdir(parents=True, exist_ok=True)
    index_cache_dir = output_dir / "index_cache"
    index_by_day_dir = output_dir / "index_by_day"
    index_history_dir = output_dir / "index_history"
    index_cache_dir.mkdir(parents=True, exist_ok=True)
    index_by_day_dir.mkdir(parents=True, exist_ok=True)
    index_history_dir.mkdir(parents=True, exist_ok=True)

    client = NSEArchiveClient()
    candidate_dates = (
        (end - timedelta(days=offset) for offset in range(fallback_days + 1))
        if latest_available_only
        else iter_dates(start, end)
    )
    frames = []
    index_paths: list[Path] = []
    for trade_date in candidate_dates:
        if trade_date < start:
            break
        try:
            row = get_nifty_index_close(client, trade_date, index_cache_dir)
        except RuntimeError as exc:
            print(f"NIFTY index archive fetch failed for {trade_date}: {exc}")
            row = pd.DataFrame()
        if not row.empty:
            frames.append(row)
            if daily_files:
                path = index_by_day_dir / f"nifty50_index_{trade_date.strftime('%Y%m%d')}.csv"
                row.to_csv(path, index=False)
                index_paths.append(path)
            if latest_available_only:
                break
        time.sleep(client.pause_seconds)

    history = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    if history.empty:
        return IndexHistoryResult(history_path=None, index_paths=tuple(index_paths), index_history=history)
    history["trade_date"] = pd.to_datetime(history["trade_date"], errors="coerce")
    history = history.dropna(subset=["trade_date"]).sort_values("trade_date").drop_duplicates("trade_date", keep="last")
    history["trade_date"] = history["trade_date"].dt.strftime("%Y-%m-%d")
    history_path = index_history_dir / f"nifty50_index_history_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.csv"
    history.to_csv(history_path, index=False)
    return IndexHistoryResult(history_path=history_path, index_paths=tuple(index_paths), index_history=history)


def get_financial_results(
    client: NSEArchiveClient,
    symbol: str,
    from_date: date,
    to_date: date,
) -> pd.DataFrame:
    frames = []
    for period in ("Quarterly", "Annual"):
        payload = client.get_json(
            "/api/corporates-financial-results",
            params={
                "index": "equities",
                "symbol": symbol,
                "from_date": nse_display_date(from_date),
                "to_date": nse_display_date(to_date),
                "period": period,
            },
        )
        period_df = pd.DataFrame(payload if isinstance(payload, list) else payload.get("data", []))
        if period_df.empty:
            continue
        period_df["requested_period"] = period
        frames.append(period_df)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True, sort=False)
    if "seqNumber" in df.columns:
        df = df.drop_duplicates(subset=["seqNumber"])
    df["symbol"] = symbol
    ordered = ["symbol"] + [col for col in df.columns if col != "symbol"]
    return df[ordered]


def get_shareholding_pattern(client: NSEArchiveClient, symbol: str) -> pd.DataFrame:
    payload = client.get_json(
        "/api/corporate-share-holdings-master",
        params={"index": "equities", "symbol": symbol},
    )
    df = pd.DataFrame(payload if isinstance(payload, list) else payload.get("data", []))
    if df.empty:
        return pd.DataFrame()
    df["symbol"] = symbol
    ordered = ["symbol"] + [col for col in df.columns if col != "symbol"]
    df = df[ordered]
    return df


def get_financial_result_announcements(client: NSEArchiveClient, symbol: str) -> pd.DataFrame:
    payload = client.get_json(
        "/api/corporate-announcements",
        params={"index": "equities", "symbol": symbol},
    )
    df = pd.DataFrame(payload if isinstance(payload, list) else payload.get("data", []))
    if df.empty:
        return pd.DataFrame()

    df["symbol"] = symbol
    text = (
        df.get("desc", pd.Series("", index=df.index)).fillna("").astype(str)
        + " "
        + df.get("attchmntText", pd.Series("", index=df.index)).fillna("").astype(str)
    )
    lower_text = text.str.lower()
    result_mask = lower_text.str.contains("financial result", na=False)
    exclude_mask = lower_text.str.contains(
        "scheduled|schedule|transcript|audio recording|analyst meet|conference call|con. call",
        na=False,
    )
    df = df[result_mask & ~exclude_mask].copy()
    if df.empty:
        return pd.DataFrame()

    df["announcement_text"] = text.loc[df.index]
    df["result_period_end"] = df["announcement_text"].map(parse_result_period_end)
    annual_text = df["announcement_text"].str.lower()
    df["is_annual_result"] = annual_text.str.contains(
        "quarter and year ended|financial year ended|for the year ended",
        na=False,
    ) & ~annual_text.str.contains("half year ended|nine months ended", na=False)
    df["announcement_dt"] = parse_date_series(df.get("sort_date", df.get("an_dt", pd.Series(dtype=str))))
    df = df.dropna(subset=["result_period_end"])
    if df.empty:
        return pd.DataFrame()

    ordered = [
        "symbol",
        "announcement_dt",
        "result_period_end",
        "is_annual_result",
        "desc",
        "attchmntText",
        "attchmntFile",
        "hasXbrl",
        "seq_id",
        "announcement_text",
    ]
    present = [col for col in ordered if col in df.columns]
    return df[present].sort_values(["symbol", "result_period_end", "announcement_dt"], ascending=[True, False, False])


def download_nse_fundamentals(
    output_dir: Path = Path("data_cache/nse_equity"),
    symbols: Iterable[str] | None = None,
    limit: int | None = None,
    run_date: date | None = None,
    lookback_years: int = 8,
) -> FundamentalsResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    fundamentals_dir = output_dir / "fundamentals"
    financial_dir = fundamentals_dir / "financial_results_by_day"
    announcements_dir = fundamentals_dir / "financial_result_announcements_by_day"
    shareholding_dir = fundamentals_dir / "shareholding_by_day"
    financial_dir.mkdir(parents=True, exist_ok=True)
    announcements_dir.mkdir(parents=True, exist_ok=True)
    shareholding_dir.mkdir(parents=True, exist_ok=True)

    client = NSEArchiveClient()
    constituents = get_nifty_constituents(client)
    pull_symbols = selected_nifty_symbols(constituents, symbols=symbols, limit=limit)

    pulled_on = run_date or date.today()
    from_date = date(pulled_on.year - max(lookback_years, 1), 1, 1)
    pulled_stamp = pulled_on.strftime("%Y%m%d")
    pulled_at = datetime.now().isoformat(timespec="seconds")

    financial_frames = []
    announcement_frames = []
    shareholding_frames = []
    for symbol in pull_symbols:
        financial_df = get_financial_results(client, symbol, from_date=from_date, to_date=pulled_on)
        if not financial_df.empty:
            financial_df.insert(1, "pulled_on", pulled_on.isoformat())
            financial_df.insert(2, "from_date", from_date.isoformat())
            financial_df.insert(3, "to_date", pulled_on.isoformat())
            financial_df.insert(4, "pulled_at", pulled_at)
            financial_frames.append(financial_df)

        announcement_df = get_financial_result_announcements(client, symbol)
        if not announcement_df.empty:
            announcement_df.insert(1, "pulled_on", pulled_on.isoformat())
            announcement_df.insert(2, "pulled_at", pulled_at)
            announcement_frames.append(announcement_df)

        shareholding_df = get_shareholding_pattern(client, symbol)
        if not shareholding_df.empty:
            shareholding_df.insert(1, "pulled_on", pulled_on.isoformat())
            shareholding_df.insert(2, "pulled_at", pulled_at)
            shareholding_frames.append(shareholding_df)

        time.sleep(client.pause_seconds)

    financial_results = (
        pd.concat(financial_frames, ignore_index=True, sort=False)
        if financial_frames
        else pd.DataFrame(columns=["symbol", "pulled_on", "from_date", "to_date", "pulled_at"])
    )
    shareholding = (
        pd.concat(shareholding_frames, ignore_index=True, sort=False)
        if shareholding_frames
        else pd.DataFrame(columns=["symbol", "pulled_on", "pulled_at"])
    )
    financial_announcements = (
        pd.concat(announcement_frames, ignore_index=True, sort=False)
        if announcement_frames
        else pd.DataFrame(columns=["symbol", "pulled_on", "pulled_at"])
    )

    financial_results_path = financial_dir / f"nse_financial_results_{pulled_stamp}.csv"
    financial_announcements_path = announcements_dir / f"nse_financial_result_announcements_{pulled_stamp}.csv"
    shareholding_path = shareholding_dir / f"nse_shareholding_pattern_{pulled_stamp}.csv"
    financial_results.to_csv(financial_results_path, index=False)
    financial_announcements.to_csv(financial_announcements_path, index=False)
    shareholding.to_csv(shareholding_path, index=False)

    return FundamentalsResult(
        financial_results_path=financial_results_path,
        financial_announcements_path=financial_announcements_path,
        shareholding_path=shareholding_path,
        financial_results=financial_results,
        financial_announcements=financial_announcements,
        shareholding=shareholding,
    )


def get_bse_scrip_master(client: BSEArchiveClient) -> pd.DataFrame:
    payload = client.get_json(
        "/BseIndiaAPI/api/ListofScripData/w",
        params={
            "Group": "",
            "Scripcode": "",
            "industry": "",
            "segment": "Equity",
            "status": "Active",
        },
    )
    df = normalize_bse_payload(payload)
    if df.empty:
        return pd.DataFrame()

    normalized = df.copy()
    normalized.columns = [str(col).strip() for col in normalized.columns]
    rename_candidates = {
        "scrip_code": ["ScripCode", "SCRIP_CD", "SecurityCode", "SC_CODE", "CODE"],
        "scrip_id": ["ScripId", "ScripID", "SC_ID", "SecurityId", "SCRIP_ID"],
        "company_name": ["ScripName", "SecurityName", "CompanyName", "SC_NAME", "NAME"],
        "isin": ["ISIN", "ISIN_CODE", "ISINCode"],
    }
    rename_map = {}
    lower_lookup = {col.lower(): col for col in normalized.columns}
    for target, choices in rename_candidates.items():
        for choice in choices:
            source = lower_lookup.get(choice.lower())
            if source is not None:
                rename_map[source] = target
                break
    normalized = normalized.rename(columns=rename_map)
    for col in ["scrip_code", "scrip_id", "company_name", "isin"]:
        if col not in normalized.columns:
            normalized[col] = ""
    normalized["scrip_code"] = normalized["scrip_code"].astype(str).str.extract(r"(\d+)")[0]
    normalized["scrip_id"] = normalized["scrip_id"].astype(str).str.upper().str.strip()
    normalized["isin"] = normalized["isin"].astype(str).str.upper().str.strip()
    normalized = normalized.dropna(subset=["scrip_code"])
    return normalized[["scrip_code", "scrip_id", "company_name", "isin"]].drop_duplicates()


def bse_quote_search(client: BSEArchiveClient, symbol: str) -> pd.DataFrame:
    payload = client.get_json(
        "https://api.bseindia.com/Msource/90D/getQouteSearch.aspx",
        params={"Type": "EQ", "text": symbol},
    )
    return normalize_bse_payload(payload)


def load_symbol_isin_map(output_dir: Path) -> pd.DataFrame:
    latest_prices = latest_csv_file(output_dir / "prices_by_day", "nifty50_prices_*.csv")
    if latest_prices is None:
        latest_prices = latest_csv_file(output_dir, "nifty50_prices_*.csv")
    if latest_prices is not None:
        prices = pd.read_csv(latest_prices)
        cols = [col for col in ["symbol", "isin", "instrument_name"] if col in prices.columns]
        if {"symbol", "isin"}.issubset(cols):
            return prices[cols].dropna(subset=["symbol", "isin"]).drop_duplicates("symbol")

    constituents = load_latest_cached_constituents(output_dir)
    if constituents is None:
        return pd.DataFrame(columns=["symbol", "isin"])
    cols = [col for col in ["symbol", "isin", "company_name"] if col in constituents.columns]
    return constituents[cols].dropna(subset=["symbol"]).drop_duplicates("symbol")


def map_symbols_to_bse_codes(
    client: BSEArchiveClient,
    symbols: Iterable[str],
    output_dir: Path,
    run_date: date,
) -> pd.DataFrame:
    symbols_df = load_symbol_isin_map(output_dir)
    if symbols_df.empty:
        symbols_df = pd.DataFrame({"symbol": list(symbols)})
    symbols_df["symbol"] = symbols_df["symbol"].astype(str).str.upper().str.strip()
    if "isin" not in symbols_df.columns:
        symbols_df["isin"] = ""
    symbols_df["isin"] = symbols_df["isin"].astype(str).str.upper().str.strip()
    symbols_df = symbols_df[symbols_df["symbol"].isin({symbol.upper() for symbol in symbols})].copy()

    cache_dir = output_dir / "fundamentals" / "bse_reference"
    cache_dir.mkdir(parents=True, exist_ok=True)
    master_path = cache_dir / f"bse_scrip_master_{run_date.strftime('%Y%m%d')}.csv"
    try:
        master = get_bse_scrip_master(client)
        if not master.empty:
            master.to_csv(master_path, index=False)
    except Exception as exc:
        print(f"BSE scrip master download failed: {exc}")
        master = pd.read_csv(master_path) if master_path.exists() else pd.DataFrame()

    mapped = symbols_df.copy()
    mapped["bse_scrip_code"] = ""
    mapped["bse_scrip_id"] = ""
    mapped["bse_company_name"] = ""

    if not master.empty and "isin" in master.columns:
        by_isin = master.dropna(subset=["isin"]).drop_duplicates("isin")
        mapped = mapped.merge(
            by_isin[["isin", "scrip_code", "scrip_id", "company_name"]],
            on="isin",
            how="left",
        )
        mapped["bse_scrip_code"] = mapped["scrip_code"].fillna(mapped["bse_scrip_code"]).astype(str)
        mapped["bse_scrip_id"] = mapped["scrip_id"].fillna(mapped["bse_scrip_id"]).astype(str)
        mapped["bse_company_name"] = mapped["company_name"].fillna(mapped["bse_company_name"]).astype(str)
        mapped = mapped.drop(columns=[col for col in ["scrip_code", "scrip_id", "company_name"] if col in mapped])

    missing_mask = mapped["bse_scrip_code"].astype(str).str.strip().isin(["", "nan", "None"])
    if not master.empty and "scrip_id" in master.columns and missing_mask.any():
        by_scrip_id = master.dropna(subset=["scrip_id"]).drop_duplicates("scrip_id")
        symbol_matches = mapped.loc[missing_mask, ["symbol"]].merge(
            by_scrip_id[["scrip_id", "scrip_code", "company_name"]],
            left_on="symbol",
            right_on="scrip_id",
            how="left",
        )
        for idx, match in zip(mapped.loc[missing_mask].index, symbol_matches.itertuples(index=False)):
            if pd.notna(match.scrip_code):
                mapped.loc[idx, "bse_scrip_code"] = str(match.scrip_code)
                mapped.loc[idx, "bse_scrip_id"] = str(match.scrip_id)
                mapped.loc[idx, "bse_company_name"] = str(match.company_name)

    missing_mask = mapped["bse_scrip_code"].astype(str).str.strip().isin(["", "nan", "None"])
    for idx, row in mapped[missing_mask].iterrows():
        symbol = str(row["symbol"]).upper()
        try:
            search_df = bse_quote_search(client, symbol)
        except Exception as exc:
            print(f"BSE quote search failed for {symbol}: {exc}")
            continue
        if search_df.empty:
            continue
        search_df.columns = [str(col).strip() for col in search_df.columns]
        best = search_df.iloc[0]
        code = first_present(best, ["SecurityCode", "ScripCode", "SCRIP_CD", "SC_CODE", "code"])
        scrip_id = first_present(best, ["SecurityId", "ScripId", "SCRIP_ID", "SC_ID", "symbol"], symbol)
        name = first_present(best, ["SecurityName", "ScripName", "CompanyName", "name"])
        if code:
            mapped.loc[idx, "bse_scrip_code"] = code
            mapped.loc[idx, "bse_scrip_id"] = scrip_id
            mapped.loc[idx, "bse_company_name"] = name
        time.sleep(client.pause_seconds)

    mapped["bse_scrip_code"] = mapped["bse_scrip_code"].astype(str).str.extract(r"(\d+)")[0]
    return mapped.dropna(subset=["bse_scrip_code"]).drop_duplicates("symbol")


def get_bse_announcements(
    client: BSEArchiveClient,
    scrip_code: str,
    from_date: date,
    to_date: date,
) -> pd.DataFrame:
    frames = []
    chunk_start = from_date
    while chunk_start <= to_date:
        chunk_end = min(date(chunk_start.year, 12, 31), to_date)
        for page in range(1, 21):
            payload = client.get_json(
                "/BseIndiaAPI/api/AnnGetData/w",
                params={
                    "pageno": str(page),
                    "strCat": "-1",
                    "strPrevDate": chunk_start.strftime("%Y%m%d"),
                    "strScrip": str(scrip_code),
                    "strSearch": "P",
                    "strToDate": chunk_end.strftime("%Y%m%d"),
                    "strType": "C",
                },
            )
            df = normalize_bse_payload(payload)
            if df.empty:
                break
            frames.append(df)
            if len(df) < 50:
                break
            time.sleep(client.pause_seconds)
        chunk_start = chunk_end + timedelta(days=1)
        time.sleep(client.pause_seconds)

    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True, sort=False)
    if "NEWSID" in df.columns:
        df = df.drop_duplicates(subset=["NEWSID"])
    return df.drop_duplicates()


def normalize_bse_filings(
    filings: pd.DataFrame,
    symbol: str,
    scrip_code: str,
    pulled_on: date,
    pulled_at: str,
    from_date: date,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if filings.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df = filings.copy()
    df.columns = [str(col).strip() for col in df.columns]
    text_parts = []
    for col in ["NEWSSUB", "HEADLINE", "MORE", "CATEGORYNAME", "SUBCATNAME", "NewsSub"]:
        if col in df.columns:
            text_parts.append(df[col].fillna("").astype(str))
    text = text_parts[0] if text_parts else pd.Series("", index=df.index)
    for part in text_parts[1:]:
        text = text + " " + part
    lower_text = text.str.lower()

    result_mask = lower_text.str.contains("financial result|audited result|unaudited result|limited review", na=False)
    exclude_mask = lower_text.str.contains(
        "scheduled|schedule|transcript|audio recording|analyst meet|conference call|investor presentation",
        na=False,
    )
    shareholding_mask = lower_text.str.contains("shareholding pattern|regulation 31", na=False)

    announcement_dates = parse_date_series(
        df.get(
            "DissemDT",
            df.get("NEWS_DT", df.get("DT_TM", df.get("News_submission_dt", pd.Series(dtype=str)))),
        )
    )
    attachments = df.apply(
        lambda row: bse_attachment_url(first_present(row, ["ATTACHMENTNAME", "NSURL", "ATTACHMENT", "attachment"])),
        axis=1,
    )
    xbrl_urls = df.apply(
        lambda row: bse_xbrl_url(first_present(row, ["XML_NAME", "XBRL", "XBRLFILE", "xbrl"])),
        axis=1,
    )
    seq_ids = df.apply(lambda row: first_present(row, ["NEWSID", "NEWS_ID", "SLNO", "SCRIP_CD"], scrip_code), axis=1)
    period_ends = text.map(parse_filing_period_end)

    result_rows = df[result_mask & ~exclude_mask].copy()
    if not result_rows.empty:
        result_idx = result_rows.index
        result_text = text.loc[result_idx]
        result_periods = period_ends.loc[result_idx]
        result_annual = result_text.str.lower().str.contains(
            "quarter and year ended|year ended|financial year ended|audited financial results",
            na=False,
        ) & ~result_text.str.lower().str.contains("half year|nine months", na=False)
        financial_announcements = pd.DataFrame(
            {
                "symbol": symbol,
                "pulled_on": pulled_on.isoformat(),
                "pulled_at": pulled_at,
                "announcement_dt": announcement_dates.loc[result_idx].values,
                "result_period_end": result_periods.values,
                "is_annual_result": result_annual.values,
                "desc": result_rows.get("CATEGORYNAME", pd.Series("", index=result_idx)).values,
                "attchmntText": result_text.values,
                "attchmntFile": attachments.loc[result_idx].values,
                "hasXbrl": xbrl_urls.loc[result_idx].astype(str).ne("").values,
                "seq_id": seq_ids.loc[result_idx].values,
                "announcement_text": result_text.values,
                "source": "bse",
                "bse_scrip_code": scrip_code,
            }
        ).dropna(subset=["result_period_end"])
        financial_results = pd.DataFrame(
            {
                "symbol": symbol,
                "pulled_on": pulled_on.isoformat(),
                "from_date": from_date.isoformat(),
                "to_date": pulled_on.isoformat(),
                "pulled_at": pulled_at,
                "broadCastDate": announcement_dates.loc[result_idx].values,
                "filingDate": announcement_dates.loc[result_idx].values,
                "toDate": result_periods.values,
                "period": np.where(result_annual.values, "Annual", "Quarterly"),
                "requested_period": np.where(result_annual.values, "Annual", "Quarterly"),
                "format": "bse_filings",
                "xbrl": xbrl_urls.loc[result_idx].values,
                "seqNumber": seq_ids.loc[result_idx].values,
                "companyName": result_rows.get("SLONGNAME", pd.Series("", index=result_idx)).values,
                "resultDescription": result_text.values,
                "resultDetailedDataLink": attachments.loc[result_idx].values,
                "source": "bse",
                "bse_scrip_code": scrip_code,
            }
        ).dropna(subset=["toDate"])
    else:
        financial_announcements = pd.DataFrame()
        financial_results = pd.DataFrame()

    share_rows = df[shareholding_mask].copy()
    if not share_rows.empty:
        share_idx = share_rows.index
        shareholding = pd.DataFrame(
            {
                "symbol": symbol,
                "pulled_on": pulled_on.isoformat(),
                "pulled_at": pulled_at,
                "date": period_ends.loc[share_idx].values,
                "submissionDate": announcement_dates.loc[share_idx].values,
                "broadcastDate": announcement_dates.loc[share_idx].values,
                "recordId": seq_ids.loc[share_idx].values,
                "xbrl": xbrl_urls.loc[share_idx].values,
                "desc": text.loc[share_idx].values,
                "pr_and_prgrp": pd.NA,
                "public_val": pd.NA,
                "source": "bse",
                "bse_scrip_code": scrip_code,
                "attachment": attachments.loc[share_idx].values,
            }
        ).dropna(subset=["date"])
    else:
        shareholding = pd.DataFrame()

    return financial_results, financial_announcements, shareholding


def download_bse_fundamentals(
    output_dir: Path = Path("data_cache/nse_equity"),
    symbols: Iterable[str] | None = None,
    limit: int | None = None,
    run_date: date | None = None,
    lookback_years: int = 8,
) -> FundamentalsResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    fundamentals_dir = output_dir / "fundamentals"
    financial_dir = fundamentals_dir / "financial_results_by_day"
    announcements_dir = fundamentals_dir / "financial_result_announcements_by_day"
    shareholding_dir = fundamentals_dir / "shareholding_by_day"
    raw_dir = fundamentals_dir / "bse_filings_by_day"
    financial_dir.mkdir(parents=True, exist_ok=True)
    announcements_dir.mkdir(parents=True, exist_ok=True)
    shareholding_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    pulled_on = run_date or date.today()
    from_date = date(pulled_on.year - max(lookback_years, 1), 1, 1)
    pulled_stamp = pulled_on.strftime("%Y%m%d")
    pulled_at = datetime.now().isoformat(timespec="seconds")

    client = BSEArchiveClient()
    if symbols:
        pull_symbols = [symbol.upper() for symbol in symbols]
        if limit is not None:
            pull_symbols = pull_symbols[:limit]
    else:
        constituents = load_latest_cached_constituents(output_dir)
        if constituents is None:
            constituents = get_nifty_constituents(NSEArchiveClient())
        pull_symbols = selected_nifty_symbols(constituents, symbols=symbols, limit=limit)

    mapping = map_symbols_to_bse_codes(client, pull_symbols, output_dir, pulled_on)
    if mapping.empty:
        raise RuntimeError("BSE scrip mapping returned no symbols.")

    mapping_path = fundamentals_dir / "bse_reference" / f"bse_symbol_map_{pulled_stamp}.csv"
    mapping.to_csv(mapping_path, index=False)

    financial_frames = []
    announcement_frames = []
    shareholding_frames = []
    raw_frames = []
    for row in mapping.itertuples(index=False):
        symbol = str(row.symbol).upper()
        scrip_code = str(row.bse_scrip_code)
        filings = get_bse_announcements(client, scrip_code, from_date=from_date, to_date=pulled_on)
        if not filings.empty:
            filings.insert(0, "symbol", symbol)
            filings.insert(1, "bse_scrip_code", scrip_code)
            filings.insert(2, "pulled_on", pulled_on.isoformat())
            filings.insert(3, "pulled_at", pulled_at)
            raw_frames.append(filings)
        financial_df, announcement_df, shareholding_df = normalize_bse_filings(
            filings,
            symbol=symbol,
            scrip_code=scrip_code,
            pulled_on=pulled_on,
            pulled_at=pulled_at,
            from_date=from_date,
        )
        if not financial_df.empty:
            financial_frames.append(financial_df)
        if not announcement_df.empty:
            announcement_frames.append(announcement_df)
        if not shareholding_df.empty:
            shareholding_frames.append(shareholding_df)
        time.sleep(client.pause_seconds)

    financial_results = (
        pd.concat(financial_frames, ignore_index=True, sort=False)
        if financial_frames
        else pd.DataFrame(columns=["symbol", "pulled_on", "from_date", "to_date", "pulled_at"])
    )
    financial_announcements = (
        pd.concat(announcement_frames, ignore_index=True, sort=False)
        if announcement_frames
        else pd.DataFrame(columns=["symbol", "pulled_on", "pulled_at"])
    )
    shareholding = (
        pd.concat(shareholding_frames, ignore_index=True, sort=False)
        if shareholding_frames
        else pd.DataFrame(columns=["symbol", "pulled_on", "pulled_at"])
    )
    raw_filings = pd.concat(raw_frames, ignore_index=True, sort=False) if raw_frames else pd.DataFrame()

    financial_results_path = financial_dir / f"bse_financial_results_{pulled_stamp}.csv"
    financial_announcements_path = announcements_dir / f"bse_financial_result_announcements_{pulled_stamp}.csv"
    shareholding_path = shareholding_dir / f"bse_shareholding_pattern_{pulled_stamp}.csv"
    raw_path = raw_dir / f"bse_filings_{pulled_stamp}.csv"
    financial_results.to_csv(financial_results_path, index=False)
    financial_announcements.to_csv(financial_announcements_path, index=False)
    shareholding.to_csv(shareholding_path, index=False)
    raw_filings.to_csv(raw_path, index=False)

    return FundamentalsResult(
        financial_results_path=financial_results_path,
        financial_announcements_path=financial_announcements_path,
        shareholding_path=shareholding_path,
        financial_results=financial_results,
        financial_announcements=financial_announcements,
        shareholding=shareholding,
    )


def slug_from_url(url: str) -> str:
    parts = [part for part in urlparse(url).path.split("/") if part]
    return parts[-2] if len(parts) >= 2 else ""


def clean_company_name(value: str) -> str:
    value = html.unescape(str(value or ""))
    value = re.sub(r"<[^>]+>", " ", value)
    return " ".join(value.split())


def period_header_to_date(value: str) -> pd.Timestamp:
    text = clean_company_name(value).replace("'", " ")
    text = re.sub(r"\bFY\b|\bQ[1-4]\b|Quarter|Year|Ended|Ending", " ", text, flags=re.IGNORECASE)
    text = " ".join(text.replace("/", " ").replace("-", " ").split())
    if not text:
        return pd.NaT

    parsed = pd.to_datetime(text, errors="coerce", dayfirst=True)
    if pd.notna(parsed):
        if parsed.day == 1 and re.search(r"[A-Za-z]", text):
            return parsed + pd.offsets.MonthEnd(0)
        return parsed

    match = re.search(
        r"(Mar|March|Jun|June|Sep|Sept|September|Dec|December)\s+(\d{2,4})",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        month, year_text = match.groups()
        year = int(year_text)
        if year < 100:
            year += 2000 if year < 70 else 1900
        parsed = pd.to_datetime(f"{month} {year}", errors="coerce")
        return parsed + pd.offsets.MonthEnd(0) if pd.notna(parsed) else pd.NaT
    return pd.NaT


def sanitize_metric_name(value: str) -> str:
    value = clean_company_name(value).lower()
    value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
    return value[:80] or "metric"


def numeric_from_text(value: str) -> float | pd.NA:
    text = clean_company_name(value)
    if not text or text in {"--", "-", "N.A.", "NA"}:
        return pd.NA
    negative = text.startswith("(") and text.endswith(")")
    text = re.sub(r"[%₹,]", "", text).replace("(", "").replace(")", "").strip()
    try:
        number = float(text)
    except ValueError:
        return pd.NA
    return -number if negative else number


def extract_html_tables(text: str) -> list[list[list[str]]]:
    parser = SimpleTableParser()
    parser.feed(text)
    return parser.tables


def table_to_financial_rows(
    table: list[list[str]],
    symbol: str,
    source: str,
    source_url: str,
    pulled_on: date,
    pulled_at: str,
    from_date: date,
    requested_period: str,
) -> list[dict]:
    if len(table) < 2:
        return []

    header_idx = None
    period_dates: list[pd.Timestamp] = []
    for idx, row in enumerate(table[:8]):
        dates = [period_header_to_date(cell) for cell in row[1:]]
        valid_dates = [value for value in dates if pd.notna(value)]
        if len(valid_dates) >= 2:
            header_idx = idx
            period_dates = dates
            break
    if header_idx is None:
        return []

    rows_by_period: list[dict] = []
    for col_idx, period_end in enumerate(period_dates, start=1):
        if pd.isna(period_end):
            continue
        row_data = {
            "symbol": symbol,
            "pulled_on": pulled_on.isoformat(),
            "from_date": from_date.isoformat(),
            "to_date": pulled_on.isoformat(),
            "pulled_at": pulled_at,
            "broadCastDate": period_end.date().isoformat(),
            "filingDate": period_end.date().isoformat(),
            "toDate": period_end.date().isoformat(),
            "period": requested_period,
            "requested_period": requested_period,
            "format": source,
            "xbrl": source_url,
            "seqNumber": f"{source}_{symbol}_{requested_period}_{period_end.strftime('%Y%m%d')}",
            "companyName": "",
            "resultDescription": f"{source} {requested_period.lower()} financial table",
            "resultDetailedDataLink": source_url,
            "source": source,
        }
        for metric_row in table[header_idx + 1 :]:
            if len(metric_row) <= col_idx:
                continue
            metric = sanitize_metric_name(metric_row[0])
            if not metric:
                continue
            row_data[metric] = numeric_from_text(metric_row[col_idx])
        rows_by_period.append(row_data)
    return rows_by_period


def rows_to_announcements(rows: pd.DataFrame, source: str) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(columns=["symbol", "pulled_on", "pulled_at"])
    ann = rows[
        [
            "symbol",
            "pulled_on",
            "pulled_at",
            "filingDate",
            "toDate",
            "requested_period",
            "resultDescription",
            "resultDetailedDataLink",
            "seqNumber",
            "source",
        ]
    ].copy()
    ann = ann.rename(
        columns={
            "filingDate": "announcement_dt",
            "toDate": "result_period_end",
            "resultDescription": "desc",
            "resultDetailedDataLink": "attchmntFile",
            "seqNumber": "seq_id",
        }
    )
    ann["is_annual_result"] = ann["requested_period"].astype(str).str.lower().eq("annual")
    ann["attchmntText"] = source + " financial table"
    ann["hasXbrl"] = ann["attchmntFile"].astype(str).str.startswith("http")
    ann["announcement_text"] = ann["attchmntText"]
    return ann.drop(columns=["requested_period"])


def table_has_metric(table: list[list[str]], metric_pattern: str) -> bool:
    return any(row and re.search(metric_pattern, row[0], flags=re.IGNORECASE) for row in table)


def table_period_count(table: list[list[str]]) -> int:
    if not table:
        return 0
    return sum(pd.notna(period_header_to_date(cell)) for cell in table[0][1:])


def screener_shareholding_rows(
    table: list[list[str]],
    symbol: str,
    source_url: str,
    pulled_on: date,
    pulled_at: str,
) -> list[dict]:
    if not table or not table[0]:
        return []
    period_dates = [period_header_to_date(cell) for cell in table[0][1:]]
    promoter_values: dict[int, float | pd.NA] = {}
    public_values: dict[int, float | pd.NA] = {}
    for row in table[1:]:
        if not row:
            continue
        label = clean_company_name(row[0]).lower()
        if label.startswith("promoter"):
            promoter_values = {
                idx: numeric_from_text(value)
                for idx, value in enumerate(row[1:], start=1)
            }
        elif label.startswith("public"):
            public_values = {
                idx: numeric_from_text(value)
                for idx, value in enumerate(row[1:], start=1)
            }

    rows = []
    for idx, period_end in enumerate(period_dates, start=1):
        if pd.isna(period_end):
            continue
        rows.append(
            {
                "symbol": symbol,
                "pulled_on": pulled_on.isoformat(),
                "pulled_at": pulled_at,
                "date": period_end.date().isoformat(),
                "submissionDate": period_end.date().isoformat(),
                "broadcastDate": period_end.date().isoformat(),
                "recordId": f"screener_{symbol}_shareholding_{period_end.strftime('%Y%m%d')}",
                "xbrl": source_url,
                "desc": "screener shareholding table",
                "pr_and_prgrp": promoter_values.get(idx, pd.NA),
                "public_val": public_values.get(idx, pd.NA),
                "source": "screener",
            }
        )
    return rows


def empty_shareholding_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["symbol", "pulled_on", "pulled_at", "source"])


def moneycontrol_search(client: WebFundamentalsClient, symbol: str) -> dict:
    url = (
        f"{MONEYCONTROL_HOME}/mccode/common/autosuggestion_solr.php"
        f"?classic=true&query={quote(symbol)}&type=1&format=json"
    )
    payload = client.get_json(url, referer=MONEYCONTROL_HOME)
    items = payload if isinstance(payload, list) else []
    if not items:
        raise RuntimeError(f"Moneycontrol did not return a stock match for {symbol}.")
    symbol_upper = symbol.upper()
    ranked = sorted(
        items,
        key=lambda item: (
            symbol_upper not in clean_company_name(str(item.get("pdt_dis_nm", ""))).upper(),
            symbol_upper not in clean_company_name(str(item.get("name", ""))).upper(),
        ),
    )
    item = ranked[0]
    link = str(item.get("link_src", ""))
    sc_id = str(item.get("sc_id") or link.rstrip("/").split("/")[-1]).strip()
    stock_name = clean_company_name(str(item.get("stock_name") or item.get("name") or symbol))
    slug = slug_from_url(link)
    if not sc_id:
        raise RuntimeError(f"Moneycontrol match for {symbol} did not include sc_id.")
    return {"sc_id": sc_id, "stock_name": stock_name, "slug": slug, "link": link}


def moneycontrol_financial_urls(match: dict, requested_period: str) -> list[str]:
    sc_id = match["sc_id"]
    slug = match.get("slug") or ""
    stock_name = quote(match.get("stock_name") or sc_id)
    urls = []
    if slug:
        result_path = "quarterly-results" if requested_period == "Quarterly" else "yearly-results"
        urls.append(f"{MONEYCONTROL_HOME}/financials/{slug}/results/{result_path}/{sc_id}")
    urls.append(f"{MONEYCONTROL_HOME}/stocks/hist_stock_result.php?ex=N&sc_id={sc_id}&mycomp={stock_name}")
    return urls


def download_moneycontrol_fundamentals(
    output_dir: Path = Path("data_cache/nse_equity"),
    symbols: Iterable[str] | None = None,
    limit: int | None = None,
    run_date: date | None = None,
    lookback_years: int = 8,
) -> FundamentalsResult:
    return download_web_fundamentals(
        source="moneycontrol",
        output_dir=output_dir,
        symbols=symbols,
        limit=limit,
        run_date=run_date,
        lookback_years=lookback_years,
    )


def download_screener_fundamentals(
    output_dir: Path = Path("data_cache/nse_equity"),
    symbols: Iterable[str] | None = None,
    limit: int | None = None,
    run_date: date | None = None,
    lookback_years: int = 8,
) -> FundamentalsResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    fundamentals_dir = output_dir / "fundamentals"
    financial_dir = fundamentals_dir / "financial_results_by_day"
    announcements_dir = fundamentals_dir / "financial_result_announcements_by_day"
    shareholding_dir = fundamentals_dir / "shareholding_by_day"
    raw_dir = fundamentals_dir / "screener_raw_by_day"
    financial_dir.mkdir(parents=True, exist_ok=True)
    announcements_dir.mkdir(parents=True, exist_ok=True)
    shareholding_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    pulled_on = run_date or date.today()
    from_date = date(pulled_on.year - max(lookback_years, 1), 1, 1)
    pulled_stamp = pulled_on.strftime("%Y%m%d")
    pulled_at = datetime.now().isoformat(timespec="seconds")
    client = WebFundamentalsClient()
    pull_symbols = selected_symbols_from_cache(output_dir, symbols=symbols, limit=limit)

    financial_rows: list[dict] = []
    shareholding_rows: list[dict] = []
    raw_rows: list[dict] = []
    errors = []
    for symbol in pull_symbols:
        urls = [
            f"{SCREENER_HOME}/company/{quote(symbol)}/consolidated/",
            f"{SCREENER_HOME}/company/{quote(symbol)}/",
        ]
        symbol_financial_rows: list[dict] = []
        symbol_shareholding_rows: list[dict] = []
        for url in urls:
            try:
                page_text = client.get_text(url, referer=SCREENER_HOME)
            except Exception as exc:
                errors.append(f"{symbol} {url}: {exc}")
                continue
            tables = extract_html_tables(page_text)
            raw_rows.append(
                {
                    "symbol": symbol,
                    "source": "screener",
                    "url": url,
                    "pulled_on": pulled_on.isoformat(),
                    "pulled_at": pulled_at,
                    "table_count": len(tables),
                    "html_bytes": len(page_text),
                }
            )
            financial_tables = [table for table in tables if table_has_metric(table, r"^(sales|revenue)")]
            for idx, table in enumerate(financial_tables[:2]):
                requested_period = "Quarterly" if idx == 0 else "Annual"
                symbol_financial_rows.extend(
                    table_to_financial_rows(
                        table,
                        symbol=symbol,
                        source="screener",
                        source_url=url,
                        pulled_on=pulled_on,
                        pulled_at=pulled_at,
                        from_date=from_date,
                        requested_period=requested_period,
                    )
                )
            shareholding_tables = [
                table
                for table in tables
                if table_has_metric(table, r"^promoters?") and table_has_metric(table, r"^public")
            ]
            for table in shareholding_tables[:1]:
                symbol_shareholding_rows.extend(
                    screener_shareholding_rows(
                        table,
                        symbol=symbol,
                        source_url=url,
                        pulled_on=pulled_on,
                        pulled_at=pulled_at,
                    )
                )
            if symbol_financial_rows:
                break
        if not symbol_financial_rows:
            errors.append(f"{symbol}: Screener did not return parseable financial tables.")
        financial_rows.extend(symbol_financial_rows)
        shareholding_rows.extend(symbol_shareholding_rows)
        time.sleep(client.pause_seconds)

    financial_results = pd.DataFrame(financial_rows)
    if not financial_results.empty:
        financial_results = financial_results.drop_duplicates(subset=["symbol", "requested_period", "toDate"])
        financial_results = financial_results.sort_values(["symbol", "requested_period", "toDate"])
    financial_announcements = rows_to_announcements(financial_results, source="screener")
    shareholding = pd.DataFrame(shareholding_rows)
    if shareholding.empty:
        shareholding = empty_shareholding_frame()
    else:
        shareholding = shareholding.drop_duplicates(subset=["symbol", "date"]).sort_values(["symbol", "date"])
    raw = pd.DataFrame(raw_rows + [{"symbol": "", "source": "screener", "error": err} for err in errors])

    if financial_results.empty:
        raise RuntimeError(
            "screener fundamentals did not produce financial result rows. "
            f"First errors: {'; '.join(errors[:3]) if errors else 'no parseable Screener tables found'}"
        )

    financial_results_path = financial_dir / f"screener_financial_results_{pulled_stamp}.csv"
    financial_announcements_path = announcements_dir / f"screener_financial_result_announcements_{pulled_stamp}.csv"
    shareholding_path = shareholding_dir / f"screener_shareholding_pattern_{pulled_stamp}.csv"
    raw_path = raw_dir / f"screener_raw_{pulled_stamp}.csv"
    financial_results.to_csv(financial_results_path, index=False)
    financial_announcements.to_csv(financial_announcements_path, index=False)
    shareholding.to_csv(shareholding_path, index=False)
    raw.to_csv(raw_path, index=False)

    return FundamentalsResult(
        financial_results_path=financial_results_path,
        financial_announcements_path=financial_announcements_path,
        shareholding_path=shareholding_path,
        financial_results=financial_results,
        financial_announcements=financial_announcements,
        shareholding=shareholding,
    )


def economic_times_search(client: WebFundamentalsClient, symbol: str) -> dict:
    search_url = f"{ECONOMIC_TIMES_HOME}/markets/stocks/stock-quotes?ticker={quote(symbol)}"
    text = client.get_text(search_url, referer=ECONOMIC_TIMES_HOME)
    company_match = re.search(r"/([^/]+?)/stocks/companyid-(\d+)\.cms", text)
    if not company_match:
        company_match = re.search(r"companyid-(\d+)\.cms", text)
        if not company_match:
            raise RuntimeError(f"Economic Times did not return a company id for {symbol}.")
        company_id = company_match.group(1)
        slug = symbol.lower()
    else:
        slug, company_id = company_match.groups()
    return {
        "company_id": company_id,
        "slug": slug,
        "link": f"{ECONOMIC_TIMES_HOME}/{slug}/stocks/companyid-{company_id}.cms",
    }


def economic_times_financial_urls(match: dict, requested_period: str) -> list[str]:
    return [match["link"]]


def download_economic_times_fundamentals(
    output_dir: Path = Path("data_cache/nse_equity"),
    symbols: Iterable[str] | None = None,
    limit: int | None = None,
    run_date: date | None = None,
    lookback_years: int = 8,
) -> FundamentalsResult:
    return download_web_fundamentals(
        source="economictimes",
        output_dir=output_dir,
        symbols=symbols,
        limit=limit,
        run_date=run_date,
        lookback_years=lookback_years,
    )


def selected_symbols_from_cache(output_dir: Path, symbols: Iterable[str] | None, limit: int | None) -> list[str]:
    if symbols:
        pull_symbols = [symbol.upper() for symbol in symbols]
        return pull_symbols[:limit] if limit is not None else pull_symbols
    constituents = load_latest_cached_constituents(output_dir)
    if constituents is None:
        constituents = get_nifty_constituents(NSEArchiveClient())
    return selected_nifty_symbols(constituents, symbols=symbols, limit=limit)


def download_web_fundamentals(
    source: str,
    output_dir: Path = Path("data_cache/nse_equity"),
    symbols: Iterable[str] | None = None,
    limit: int | None = None,
    run_date: date | None = None,
    lookback_years: int = 8,
) -> FundamentalsResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    fundamentals_dir = output_dir / "fundamentals"
    financial_dir = fundamentals_dir / "financial_results_by_day"
    announcements_dir = fundamentals_dir / "financial_result_announcements_by_day"
    shareholding_dir = fundamentals_dir / "shareholding_by_day"
    raw_dir = fundamentals_dir / f"{source}_raw_by_day"
    financial_dir.mkdir(parents=True, exist_ok=True)
    announcements_dir.mkdir(parents=True, exist_ok=True)
    shareholding_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    pulled_on = run_date or date.today()
    from_date = date(pulled_on.year - max(lookback_years, 1), 1, 1)
    pulled_stamp = pulled_on.strftime("%Y%m%d")
    pulled_at = datetime.now().isoformat(timespec="seconds")
    client = WebFundamentalsClient()
    pull_symbols = selected_symbols_from_cache(output_dir, symbols=symbols, limit=limit)

    financial_rows: list[dict] = []
    raw_rows: list[dict] = []
    errors = []
    for symbol in pull_symbols:
        try:
            if source == "moneycontrol":
                match = moneycontrol_search(client, symbol)
                url_builder = moneycontrol_financial_urls
            elif source == "economictimes":
                match = economic_times_search(client, symbol)
                url_builder = economic_times_financial_urls
            else:
                raise ValueError(f"Unsupported web fundamentals source: {source}")

            for requested_period in ("Quarterly", "Annual"):
                period_rows: list[dict] = []
                for url in url_builder(match, requested_period):
                    try:
                        page_text = client.get_text(url, referer=match.get("link"))
                    except Exception as exc:
                        errors.append(f"{symbol} {requested_period} {url}: {exc}")
                        continue
                    tables = extract_html_tables(page_text)
                    raw_rows.append(
                        {
                            "symbol": symbol,
                            "source": source,
                            "requested_period": requested_period,
                            "url": url,
                            "pulled_on": pulled_on.isoformat(),
                            "pulled_at": pulled_at,
                            "table_count": len(tables),
                            "html_bytes": len(page_text),
                        }
                    )
                    for table in tables:
                        period_rows.extend(
                            table_to_financial_rows(
                                table,
                                symbol=symbol,
                                source=source,
                                source_url=url,
                                pulled_on=pulled_on,
                                pulled_at=pulled_at,
                                from_date=from_date,
                                requested_period=requested_period,
                            )
                        )
                    if period_rows:
                        break
                financial_rows.extend(period_rows)
            time.sleep(client.pause_seconds)
        except Exception as exc:
            errors.append(f"{symbol}: {exc}")

    financial_results = pd.DataFrame(financial_rows)
    if not financial_results.empty:
        financial_results = financial_results.drop_duplicates(subset=["symbol", "requested_period", "toDate"])
        financial_results = financial_results.sort_values(["symbol", "requested_period", "toDate"])
    financial_announcements = rows_to_announcements(financial_results, source=source)
    shareholding = empty_shareholding_frame()
    raw = pd.DataFrame(raw_rows + [{"symbol": "", "source": source, "error": err} for err in errors])

    if financial_results.empty:
        raise RuntimeError(
            f"{source} fundamentals did not produce financial result rows. "
            f"First errors: {'; '.join(errors[:3]) if errors else 'no parseable provider tables found'}"
        )

    financial_results_path = financial_dir / f"{source}_financial_results_{pulled_stamp}.csv"
    financial_announcements_path = announcements_dir / f"{source}_financial_result_announcements_{pulled_stamp}.csv"
    shareholding_path = shareholding_dir / f"{source}_shareholding_pattern_{pulled_stamp}.csv"
    raw_path = raw_dir / f"{source}_raw_{pulled_stamp}.csv"
    financial_results.to_csv(financial_results_path, index=False)
    financial_announcements.to_csv(financial_announcements_path, index=False)
    shareholding.to_csv(shareholding_path, index=False)
    raw.to_csv(raw_path, index=False)

    return FundamentalsResult(
        financial_results_path=financial_results_path,
        financial_announcements_path=financial_announcements_path,
        shareholding_path=shareholding_path,
        financial_results=financial_results,
        financial_announcements=financial_announcements,
        shareholding=shareholding,
    )


def recency_points(days: float | None, buckets: list[tuple[int, int]], stale_points: int = 0) -> int:
    if days is None or pd.isna(days):
        return stale_points
    for max_days, points in buckets:
        if days <= max_days:
            return points
    return stale_points


def iso_date_or_blank(value: pd.Timestamp) -> str:
    return value.date().isoformat() if pd.notna(value) else ""


def bool_flag(value: bool) -> int:
    return int(bool(value))


def days_between(later: pd.Timestamp, earlier: pd.Timestamp) -> int | pd.NA:
    if pd.isna(later) or pd.isna(earlier):
        return pd.NA
    return int((later - earlier).days)


def bounded_score(value: float | int | pd.NA, low: float, high: float, points: float) -> float:
    if pd.isna(value):
        return 0.0
    value = float(value)
    if high == low:
        return points if value >= high else 0.0
    return max(0.0, min(points, (value - low) / (high - low) * points))


def inverse_bounded_score(value: float | int | pd.NA, low: float, high: float, points: float) -> float:
    if pd.isna(value):
        return 0.0
    value = float(value)
    if high == low:
        return points if value <= low else 0.0
    return max(0.0, min(points, (high - value) / (high - low) * points))


def pct_change_between(current: float | int | pd.NA, previous: float | int | pd.NA) -> float | pd.NA:
    if pd.isna(current) or pd.isna(previous):
        return pd.NA
    previous = float(previous)
    if abs(previous) < 1e-9:
        return pd.NA
    return (float(current) / previous - 1.0) * 100.0


def latest_numeric(frame: pd.DataFrame, column: str) -> float | pd.NA:
    if frame.empty or column not in frame.columns:
        return pd.NA
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return values.iloc[-1] if not values.empty else pd.NA


def prior_numeric(frame: pd.DataFrame, column: str, periods_back: int = 1) -> float | pd.NA:
    if frame.empty or column not in frame.columns:
        return pd.NA
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if len(values) <= periods_back:
        return pd.NA
    return values.iloc[-1 - periods_back]


def revenue_column(frame: pd.DataFrame) -> str | None:
    if "sales" in frame.columns and pd.to_numeric(frame["sales"], errors="coerce").notna().any():
        return "sales"
    if "revenue" in frame.columns and pd.to_numeric(frame["revenue"], errors="coerce").notna().any():
        return "revenue"
    return None


def latest_margin(frame: pd.DataFrame) -> float | pd.NA:
    margin = latest_numeric(frame, "opm")
    if pd.isna(margin):
        margin = latest_numeric(frame, "financing_margin")
    return margin


def compute_financial_quality_scores(quarterly: pd.DataFrame, annual: pd.DataFrame) -> dict:
    metrics: dict[str, float | int | pd.NA] = {}
    q = quarterly.sort_values("period_end_dt").copy() if not quarterly.empty else quarterly.copy()
    a = annual.sort_values("period_end_dt").copy() if not annual.empty else annual.copy()
    q_revenue_col = revenue_column(q)
    a_revenue_col = revenue_column(a)

    q_revenue = latest_numeric(q, q_revenue_col) if q_revenue_col else pd.NA
    q_revenue_yoy = pct_change_between(q_revenue, prior_numeric(q, q_revenue_col, 4)) if q_revenue_col else pd.NA
    q_revenue_qoq = pct_change_between(q_revenue, prior_numeric(q, q_revenue_col, 1)) if q_revenue_col else pd.NA
    q_profit = latest_numeric(q, "net_profit")
    q_profit_yoy = pct_change_between(q_profit, prior_numeric(q, "net_profit", 4))
    q_profit_qoq = pct_change_between(q_profit, prior_numeric(q, "net_profit", 1))
    q_eps = latest_numeric(q, "eps_in_rs")
    q_eps_yoy = pct_change_between(q_eps, prior_numeric(q, "eps_in_rs", 4))

    a_revenue = latest_numeric(a, a_revenue_col) if a_revenue_col else pd.NA
    a_revenue_3y_ago = prior_numeric(a, a_revenue_col, 3) if a_revenue_col else pd.NA
    a_revenue_3y_cagr = (
        ((float(a_revenue) / float(a_revenue_3y_ago)) ** (1 / 3) - 1) * 100
        if pd.notna(a_revenue) and pd.notna(a_revenue_3y_ago) and float(a_revenue_3y_ago) > 0
        else pd.NA
    )
    a_profit = latest_numeric(a, "net_profit")
    a_profit_3y_ago = prior_numeric(a, "net_profit", 3)
    a_profit_3y_cagr = (
        ((float(a_profit) / float(a_profit_3y_ago)) ** (1 / 3) - 1) * 100
        if pd.notna(a_profit) and pd.notna(a_profit_3y_ago) and float(a_profit_3y_ago) > 0 and float(a_profit) > 0
        else pd.NA
    )

    latest_opm = latest_margin(q)
    previous_opm = prior_numeric(q, "opm", 4)
    if pd.isna(previous_opm):
        previous_opm = prior_numeric(q, "financing_margin", 4)
    opm_yoy_change = float(latest_opm) - float(previous_opm) if pd.notna(latest_opm) and pd.notna(previous_opm) else pd.NA
    net_margin = (
        float(q_profit) / float(q_revenue) * 100
        if pd.notna(q_profit) and pd.notna(q_revenue) and abs(float(q_revenue)) > 1e-9
        else pd.NA
    )
    interest = latest_numeric(q, "interest")
    interest_to_revenue = (
        float(interest) / float(q_revenue) * 100
        if pd.notna(interest) and pd.notna(q_revenue) and abs(float(q_revenue)) > 1e-9
        else pd.NA
    )
    expense = latest_numeric(q, "expenses")
    expense_ratio = (
        float(expense) / float(q_revenue) * 100
        if pd.notna(expense) and pd.notna(q_revenue) and abs(float(q_revenue)) > 1e-9
        else pd.NA
    )
    previous_expense = prior_numeric(q, "expenses", 4)
    previous_revenue = prior_numeric(q, q_revenue_col, 4) if q_revenue_col else pd.NA
    previous_expense_ratio = (
        float(previous_expense) / float(previous_revenue) * 100
        if pd.notna(previous_expense) and pd.notna(previous_revenue) and abs(float(previous_revenue)) > 1e-9
        else pd.NA
    )
    expense_ratio_yoy_change = (
        float(expense_ratio) - float(previous_expense_ratio)
        if pd.notna(expense_ratio) and pd.notna(previous_expense_ratio)
        else pd.NA
    )
    gross_npa = latest_numeric(q, "gross_npa")
    net_npa = latest_numeric(q, "net_npa")
    dividend_payout = latest_numeric(a, "dividend_payout")

    recent_profits = pd.to_numeric(q.get("net_profit", pd.Series(dtype=float)), errors="coerce").dropna().tail(4)
    positive_profit_quarters = int((recent_profits > 0).sum()) if not recent_profits.empty else 0
    recent_revenue = pd.to_numeric(q[q_revenue_col], errors="coerce").dropna().tail(4) if q_revenue_col else pd.Series(dtype=float)
    positive_revenue_quarters = int((recent_revenue > 0).sum()) if not recent_revenue.empty else 0

    growth_score = (
        bounded_score(q_revenue_yoy, -10, 20, 7)
        + bounded_score(q_profit_yoy, -20, 30, 7)
        + bounded_score(a_revenue_3y_cagr, 0, 18, 5)
        + bounded_score(a_profit_3y_cagr, 0, 20, 4)
        + bounded_score(q_eps_yoy, -15, 25, 2)
    )
    profitability_score = (
        bounded_score(latest_opm, 5, 25, 7)
        + bounded_score(opm_yoy_change, -4, 4, 4)
        + bounded_score(net_margin, 0, 15, 5)
        + bounded_score(positive_profit_quarters, 1, 4, 4)
    )
    efficiency_score = (
        inverse_bounded_score(interest_to_revenue, 2, 25, 4)
        + inverse_bounded_score(expense_ratio_yoy_change, -5, 8, 4)
        + bounded_score(positive_revenue_quarters, 1, 4, 3)
        + inverse_bounded_score(gross_npa, 1, 8, 2)
        + inverse_bounded_score(net_npa, 0.5, 4, 2)
    )
    shareholder_return_score = bounded_score(dividend_payout, 0, 35, 5)

    metrics.update(
        {
            "growth_score": round(growth_score, 2),
            "profitability_score": round(profitability_score, 2),
            "efficiency_score": round(efficiency_score, 2),
            "shareholder_return_score": round(shareholder_return_score, 2),
            "latest_revenue": q_revenue,
            "latest_net_profit": q_profit,
            "latest_eps": q_eps,
            "latest_margin": latest_opm,
            "latest_net_margin": net_margin,
            "latest_interest_to_revenue": interest_to_revenue,
            "latest_expense_ratio": expense_ratio,
            "latest_gross_npa": gross_npa,
            "latest_net_npa": net_npa,
            "latest_dividend_payout": dividend_payout,
            "quarterly_revenue_yoy_pct": q_revenue_yoy,
            "quarterly_revenue_qoq_pct": q_revenue_qoq,
            "quarterly_profit_yoy_pct": q_profit_yoy,
            "quarterly_profit_qoq_pct": q_profit_qoq,
            "quarterly_eps_yoy_pct": q_eps_yoy,
            "annual_revenue_3y_cagr_pct": a_revenue_3y_cagr,
            "annual_profit_3y_cagr_pct": a_profit_3y_cagr,
            "margin_yoy_change_pct": opm_yoy_change,
            "expense_ratio_yoy_change_pct": expense_ratio_yoy_change,
            "positive_profit_quarters": positive_profit_quarters,
            "positive_revenue_quarters": positive_revenue_quarters,
        }
    )
    return metrics


def latest_previous_score_file(scores_dir: Path, score_date: date) -> Path | None:
    if not scores_dir.exists():
        return None
    candidates = []
    for pattern in ("screener_fundamental_scores_*.csv", "nse_fundamental_scores_*.csv"):
        for path in scores_dir.glob(pattern):
            date_text = path.stem.rsplit("_", 1)[-1]
            try:
                path_date = datetime.strptime(date_text, "%Y%m%d").date()
            except ValueError:
                continue
            if path_date < score_date:
                candidates.append((path_date, 0 if path.name.startswith("nse_") else 1, path))
    candidates = sorted(set(candidates))
    return candidates[-1][2] if candidates else None


def latest_csv_file(directory: Path, pattern: str) -> Path | None:
    if not directory.exists():
        return None
    candidates = sorted(directory.glob(pattern))
    return candidates[-1] if candidates else None


def latest_csv_file_any(directory: Path, patterns: Iterable[str]) -> Path | None:
    if not directory.exists():
        return None
    candidates = []
    for pattern in patterns:
        candidates.extend(directory.glob(pattern))
    candidates = sorted(set(candidates))
    return candidates[-1] if candidates else None


def load_latest_cached_fundamentals(output_dir: Path) -> FundamentalsResult:
    fundamentals_dir = output_dir / "fundamentals"
    financial_path = latest_csv_file_any(
        fundamentals_dir / "financial_results_by_day",
        [
            "screener_financial_results_*.csv",
            "moneycontrol_financial_results_*.csv",
            "economictimes_financial_results_*.csv",
            "nse_financial_results_*.csv",
            "bse_financial_results_*.csv",
        ],
    )
    announcements_path = latest_csv_file_any(
        fundamentals_dir / "financial_result_announcements_by_day",
        [
            "screener_financial_result_announcements_*.csv",
            "moneycontrol_financial_result_announcements_*.csv",
            "economictimes_financial_result_announcements_*.csv",
            "nse_financial_result_announcements_*.csv",
            "bse_financial_result_announcements_*.csv",
        ],
    )
    shareholding_path = latest_csv_file_any(
        fundamentals_dir / "shareholding_by_day",
        [
            "screener_shareholding_pattern_*.csv",
            "moneycontrol_shareholding_pattern_*.csv",
            "economictimes_shareholding_pattern_*.csv",
            "nse_shareholding_pattern_*.csv",
            "bse_shareholding_pattern_*.csv",
        ],
    )

    missing = [
        name
        for name, path in [
            ("financial results", financial_path),
            ("financial result announcements", announcements_path),
            ("shareholding pattern", shareholding_path),
        ]
        if path is None
    ]
    if missing:
        raise RuntimeError(
            "Cached fundamentals missing: "
            + ", ".join(missing)
            + ". Run with --fundamentals before generating score history."
        )

    return FundamentalsResult(
        financial_results_path=financial_path,
        financial_announcements_path=announcements_path,
        shareholding_path=shareholding_path,
        financial_results=pd.read_csv(financial_path),
        financial_announcements=pd.read_csv(announcements_path),
        shareholding=pd.read_csv(shareholding_path),
    )


def score_symbol_fundamentals(
    symbol: str,
    financial_results: pd.DataFrame,
    shareholding: pd.DataFrame,
    financial_announcements: pd.DataFrame,
    score_date: date,
) -> dict:
    fin = financial_results[financial_results.get("symbol", pd.Series(dtype=str)).astype(str).eq(symbol)].copy()
    shp = shareholding[shareholding.get("symbol", pd.Series(dtype=str)).astype(str).eq(symbol)].copy()
    ann = financial_announcements[
        financial_announcements.get("symbol", pd.Series(dtype=str)).astype(str).eq(symbol)
    ].copy()

    score_dt = pd.Timestamp(score_date)
    latest_quarter_end = pd.NaT
    latest_annual_end = pd.NaT
    formal_latest_quarter_end = pd.NaT
    formal_latest_annual_end = pd.NaT
    announcement_latest_quarter_end = pd.NaT
    announcement_latest_annual_end = pd.NaT
    latest_result_announcement_dt = pd.NaT
    result_freshness_source = "financial_results"
    latest_shareholding_date = pd.NaT
    latest_quarter_filing_dt = pd.NaT
    latest_annual_filing_dt = pd.NaT
    latest_shareholding_submission_dt = pd.NaT
    latest_shareholding_record_id = ""
    latest_shareholding_xbrl = ""
    latest_quarter_xbrl = ""
    latest_annual_xbrl = ""
    latest_quarter_has_xbrl = False
    latest_annual_has_xbrl = False
    latest_financial_format_new = False
    quarterly_filing_count = 0
    annual_filing_count = 0
    quarterly_period_count = 0
    annual_period_count = 0

    quarterly_recency_score = 0
    annual_recency_score = 0
    disclosure_score = 0
    filing_consistency_score = 0
    shareholding_recency_score = 0
    promoter_score = 0
    shareholding_trend_score = 0
    latest_promoter_holding = pd.NA
    latest_public_holding = pd.NA
    promoter_holding_qoq_change = pd.NA
    public_holding_qoq_change = pd.NA
    financial_quality = {
        "growth_score": 0,
        "profitability_score": 0,
        "efficiency_score": 0,
        "shareholder_return_score": 0,
    }

    if not fin.empty:
        fin["period_end_dt"] = parse_date_series(fin.get("toDate", pd.Series(dtype=str)))
        fin["filing_dt"] = parse_date_series(fin.get("filingDate", fin.get("broadCastDate", pd.Series(dtype=str))))
        fin["requested_period_norm"] = fin.get("requested_period", fin.get("period", "")).astype(str).str.lower()
        quarterly = fin[fin["requested_period_norm"].eq("quarterly")].copy()
        annual = fin[fin["requested_period_norm"].eq("annual")].copy()
        quarterly_filing_count = len(quarterly)
        annual_filing_count = len(annual)

        if not quarterly.empty:
            formal_latest_quarter_end = quarterly["period_end_dt"].max()
            latest_quarter_end = formal_latest_quarter_end
            latest_q = quarterly[quarterly["period_end_dt"].eq(latest_quarter_end)].sort_values("filing_dt").tail(1)
            if not latest_q.empty:
                latest_quarter_filing_dt = latest_q["filing_dt"].iloc[0]
                latest_quarter_xbrl = str(latest_q.get("xbrl", pd.Series([""])).iloc[0])
            q_days = (score_dt - latest_quarter_end).days if pd.notna(latest_quarter_end) else None
            quarterly_recency_score = recency_points(
                q_days,
                buckets=[(120, 30), (210, 24), (365, 15), (540, 8)],
            )

        if not annual.empty:
            formal_latest_annual_end = annual["period_end_dt"].max()
            latest_annual_end = formal_latest_annual_end
            latest_a = annual[annual["period_end_dt"].eq(latest_annual_end)].sort_values("filing_dt").tail(1)
            if not latest_a.empty:
                latest_annual_filing_dt = latest_a["filing_dt"].iloc[0]
                latest_annual_xbrl = str(latest_a.get("xbrl", pd.Series([""])).iloc[0])
            a_days = (score_dt - latest_annual_end).days if pd.notna(latest_annual_end) else None
            annual_recency_score = recency_points(
                a_days,
                buckets=[(460, 20), (730, 14), (1095, 8)],
            )

        if "xbrl" in fin.columns:
            if pd.notna(latest_quarter_end):
                latest_q = quarterly[quarterly["period_end_dt"].eq(latest_quarter_end)]
                latest_quarter_has_xbrl = latest_q["xbrl"].astype(str).str.startswith("http").any()
            if pd.notna(latest_annual_end):
                latest_a = annual[annual["period_end_dt"].eq(latest_annual_end)]
                latest_annual_has_xbrl = latest_a["xbrl"].astype(str).str.startswith("http").any()
            disclosure_score += 8 if latest_quarter_has_xbrl else 0
            disclosure_score += 5 if latest_annual_has_xbrl else 0

        latest_financial_format_new = "format" in fin.columns and fin["format"].astype(str).str.lower().eq("new").any()
        if latest_financial_format_new:
            disclosure_score += 2
        disclosure_score = min(disclosure_score, 15)

        q_unique = quarterly["period_end_dt"].dropna().dt.strftime("%Y-%m-%d").nunique() if not quarterly.empty else 0
        a_unique = annual["period_end_dt"].dropna().dt.strftime("%Y-%m-%d").nunique() if not annual.empty else 0
        quarterly_period_count = q_unique
        annual_period_count = a_unique
        filing_consistency_score = min(7, q_unique) + min(3, a_unique)
        financial_quality = compute_financial_quality_scores(quarterly, annual)

    if not ann.empty:
        ann["announcement_period_end_dt"] = parse_date_series(ann.get("result_period_end", pd.Series(dtype=str)))
        ann["announcement_dt"] = parse_date_series(ann.get("announcement_dt", pd.Series(dtype=str)))
        latest_ann_period_end = ann["announcement_period_end_dt"].max()
        announcement_latest_quarter_end = latest_ann_period_end
        if pd.notna(latest_ann_period_end) and (
            pd.isna(latest_quarter_end) or latest_ann_period_end > latest_quarter_end
        ):
            latest_quarter_end = latest_ann_period_end
            result_freshness_source = "financial_result_announcements"
        annual_ann = ann[ann.get("is_annual_result", pd.Series(False, index=ann.index)).astype(bool)]
        if not annual_ann.empty:
            latest_annual_ann_period_end = annual_ann["announcement_period_end_dt"].max()
            announcement_latest_annual_end = latest_annual_ann_period_end
            if pd.notna(latest_annual_ann_period_end) and (
                pd.isna(latest_annual_end) or latest_annual_ann_period_end > latest_annual_end
            ):
                latest_annual_end = latest_annual_ann_period_end
                result_freshness_source = "financial_result_announcements"
        latest_result_announcement_dt = ann["announcement_dt"].max()

        q_days = (score_dt - latest_quarter_end).days if pd.notna(latest_quarter_end) else None
        a_days = (score_dt - latest_annual_end).days if pd.notna(latest_annual_end) else None
        quarterly_recency_score = recency_points(
            q_days,
            buckets=[(120, 30), (210, 24), (365, 15), (540, 8)],
        )
        annual_recency_score = recency_points(
            a_days,
            buckets=[(460, 20), (730, 14), (1095, 8)],
        )

    if not shp.empty:
        shp["shareholding_dt"] = parse_date_series(shp.get("date", pd.Series(dtype=str)))
        shp["shareholding_submission_dt"] = parse_date_series(
            shp.get("submissionDate", shp.get("broadcastDate", pd.Series(dtype=str)))
        )
        latest_shareholding_date = shp["shareholding_dt"].max()
        s_days = (score_dt - latest_shareholding_date).days if pd.notna(latest_shareholding_date) else None
        shareholding_recency_score = recency_points(
            s_days,
            buckets=[(120, 15), (210, 10), (365, 6), (540, 3)],
        )

        latest_shp = shp[shp["shareholding_dt"].eq(latest_shareholding_date)].sort_values(
            "shareholding_submission_dt"
        ).tail(1)
        if not latest_shp.empty:
            latest_promoter_holding = pd.to_numeric(latest_shp.get("pr_and_prgrp"), errors="coerce").iloc[0]
            latest_public_holding = pd.to_numeric(latest_shp.get("public_val"), errors="coerce").iloc[0]
            latest_shareholding_submission_dt = latest_shp["shareholding_submission_dt"].iloc[0]
            latest_shareholding_record_id = str(latest_shp.get("recordId", pd.Series([""])).iloc[0])
            latest_shareholding_xbrl = str(latest_shp.get("xbrl", pd.Series([""])).iloc[0])
            if pd.notna(latest_promoter_holding):
                if 45 <= latest_promoter_holding <= 75:
                    promoter_score = 10
                elif 25 <= latest_promoter_holding < 45 or 75 < latest_promoter_holding <= 90:
                    promoter_score = 8
                elif latest_promoter_holding > 0:
                    promoter_score = 5
            shp_sorted = shp.sort_values(["shareholding_dt", "shareholding_submission_dt"])
            promoter_series = pd.to_numeric(shp_sorted.get("pr_and_prgrp"), errors="coerce").dropna()
            public_series = pd.to_numeric(shp_sorted.get("public_val"), errors="coerce").dropna()
            if len(promoter_series) >= 2:
                promoter_holding_qoq_change = promoter_series.iloc[-1] - promoter_series.iloc[-2]
                shareholding_trend_score += bounded_score(promoter_holding_qoq_change, -1.0, 1.0, 4)
            if len(public_series) >= 2:
                public_holding_qoq_change = public_series.iloc[-1] - public_series.iloc[-2]
                shareholding_trend_score += inverse_bounded_score(public_holding_qoq_change, -1.0, 1.0, 2)

    freshness_score = (
        min(4, quarterly_recency_score / 30 * 4)
        + min(3, annual_recency_score / 20 * 3)
        + min(2, shareholding_recency_score / 15 * 2)
        + min(1, filing_consistency_score / 10)
    )
    shareholding_quality_score = min(15, promoter_score * 0.7 + shareholding_trend_score + min(2, shareholding_recency_score / 15 * 2))
    computed_score = (
        float(financial_quality.get("growth_score", 0) or 0)
        + float(financial_quality.get("profitability_score", 0) or 0)
        + float(financial_quality.get("efficiency_score", 0) or 0)
        + float(financial_quality.get("shareholder_return_score", 0) or 0)
        + shareholding_quality_score
        + freshness_score
    )
    computed_score = max(0, min(100, computed_score))

    return {
        "symbol": symbol,
        "computed_score": int(round(computed_score)),
        "growth_score": financial_quality.get("growth_score", 0),
        "profitability_score": financial_quality.get("profitability_score", 0),
        "efficiency_score": financial_quality.get("efficiency_score", 0),
        "shareholder_return_score": financial_quality.get("shareholder_return_score", 0),
        "shareholding_quality_score": round(shareholding_quality_score, 2),
        "freshness_score": round(freshness_score, 2),
        "quarterly_recency_score": quarterly_recency_score,
        "annual_recency_score": annual_recency_score,
        "disclosure_score": disclosure_score,
        "filing_consistency_score": filing_consistency_score,
        "shareholding_recency_score": shareholding_recency_score,
        "promoter_score": promoter_score,
        "shareholding_trend_score": round(shareholding_trend_score, 2),
        **financial_quality,
        "quarterly_recency_days": days_between(score_dt, latest_quarter_end),
        "annual_recency_days": days_between(score_dt, latest_annual_end),
        "shareholding_recency_days": days_between(score_dt, latest_shareholding_date),
        "formal_latest_quarter_end": iso_date_or_blank(formal_latest_quarter_end),
        "formal_latest_annual_end": iso_date_or_blank(formal_latest_annual_end),
        "announcement_latest_quarter_end": iso_date_or_blank(announcement_latest_quarter_end),
        "announcement_latest_annual_end": iso_date_or_blank(announcement_latest_annual_end),
        "latest_quarter_end": iso_date_or_blank(latest_quarter_end),
        "latest_annual_end": iso_date_or_blank(latest_annual_end),
        "latest_quarter_filing_dt": iso_date_or_blank(latest_quarter_filing_dt),
        "latest_annual_filing_dt": iso_date_or_blank(latest_annual_filing_dt),
        "latest_result_announcement_dt": iso_date_or_blank(latest_result_announcement_dt),
        "result_freshness_source": result_freshness_source,
        "latest_quarter_has_xbrl": bool_flag(latest_quarter_has_xbrl),
        "latest_annual_has_xbrl": bool_flag(latest_annual_has_xbrl),
        "latest_quarter_xbrl": latest_quarter_xbrl,
        "latest_annual_xbrl": latest_annual_xbrl,
        "latest_financial_format_new": bool_flag(latest_financial_format_new),
        "quarterly_filing_count": quarterly_filing_count,
        "annual_filing_count": annual_filing_count,
        "quarterly_period_count": quarterly_period_count,
        "annual_period_count": annual_period_count,
        "latest_shareholding_date": iso_date_or_blank(latest_shareholding_date),
        "latest_shareholding_submission_dt": iso_date_or_blank(latest_shareholding_submission_dt),
        "latest_shareholding_record_id": latest_shareholding_record_id,
        "latest_shareholding_xbrl": latest_shareholding_xbrl,
        "latest_promoter_holding": latest_promoter_holding,
        "latest_public_holding": latest_public_holding,
        "promoter_holding_qoq_change": promoter_holding_qoq_change,
        "public_holding_qoq_change": public_holding_qoq_change,
    }


def generate_fundamental_scores(
    financial_results: pd.DataFrame,
    shareholding: pd.DataFrame,
    financial_announcements: pd.DataFrame | None = None,
    output_dir: Path = Path("data_cache/nse_equity"),
    symbols: Iterable[str] | None = None,
    score_date: date | None = None,
) -> FundamentalScoreResult:
    score_on = score_date or date.today()
    fundamentals_dir = output_dir / "fundamentals"
    scores_dir = fundamentals_dir / "fundamental_scores_by_day"
    scores_dir.mkdir(parents=True, exist_ok=True)

    if symbols is None:
        symbol_values = pd.concat(
            [
                financial_results.get("symbol", pd.Series(dtype=str)),
                shareholding.get("symbol", pd.Series(dtype=str)),
            ],
            ignore_index=True,
        )
        score_symbols = sorted(symbol_values.dropna().astype(str).unique().tolist())
    else:
        score_symbols = sorted({symbol.upper() for symbol in symbols})

    if financial_announcements is None:
        financial_announcements = pd.DataFrame()

    rows = [
        score_symbol_fundamentals(symbol, financial_results, shareholding, financial_announcements, score_on)
        for symbol in score_symbols
    ]
    scores = pd.DataFrame(rows)
    if scores.empty:
        scores = pd.DataFrame(columns=["symbol", "computed_score"])

    previous_path = latest_previous_score_file(scores_dir, score_on)
    previous_scores = pd.DataFrame()
    if previous_path is not None:
        previous_scores = pd.read_csv(previous_path)

    if previous_scores.empty or "symbol" not in previous_scores.columns:
        scores["previous_score"] = pd.NA
        scores["fundamental_score"] = scores["computed_score"]
        scores["score_changed"] = True
        scores["score_source"] = "computed"
        scores["previous_score_file"] = ""
    else:
        previous_col = "fundamental_score" if "fundamental_score" in previous_scores.columns else "computed_score"
        previous_lookup = previous_scores.set_index("symbol")[previous_col].to_dict()
        scores["previous_score"] = scores["symbol"].map(previous_lookup)
        scores["score_changed"] = scores["previous_score"].isna() | (
            scores["computed_score"] != scores["previous_score"]
        )
        scores["fundamental_score"] = scores["computed_score"].where(
            scores["score_changed"],
            scores["previous_score"],
        )
        scores["score_source"] = scores["score_changed"].map({True: "computed_changed", False: "carried_forward"})
        scores["previous_score_file"] = str(previous_path)

    scores.insert(0, "score_date", score_on.isoformat())
    ordered_cols = [
        "score_date",
        "symbol",
        "fundamental_score",
        "computed_score",
        "previous_score",
        "score_changed",
        "score_source",
        "growth_score",
        "profitability_score",
        "efficiency_score",
        "shareholder_return_score",
        "shareholding_quality_score",
        "freshness_score",
        "quarterly_recency_score",
        "annual_recency_score",
        "disclosure_score",
        "filing_consistency_score",
        "shareholding_recency_score",
        "promoter_score",
        "shareholding_trend_score",
        "latest_revenue",
        "latest_net_profit",
        "latest_eps",
        "latest_margin",
        "latest_net_margin",
        "latest_interest_to_revenue",
        "latest_expense_ratio",
        "latest_gross_npa",
        "latest_net_npa",
        "latest_dividend_payout",
        "quarterly_revenue_yoy_pct",
        "quarterly_revenue_qoq_pct",
        "quarterly_profit_yoy_pct",
        "quarterly_profit_qoq_pct",
        "quarterly_eps_yoy_pct",
        "annual_revenue_3y_cagr_pct",
        "annual_profit_3y_cagr_pct",
        "margin_yoy_change_pct",
        "expense_ratio_yoy_change_pct",
        "positive_profit_quarters",
        "positive_revenue_quarters",
        "quarterly_recency_days",
        "annual_recency_days",
        "shareholding_recency_days",
        "formal_latest_quarter_end",
        "formal_latest_annual_end",
        "announcement_latest_quarter_end",
        "announcement_latest_annual_end",
        "latest_quarter_end",
        "latest_annual_end",
        "latest_quarter_filing_dt",
        "latest_annual_filing_dt",
        "latest_result_announcement_dt",
        "result_freshness_source",
        "latest_quarter_has_xbrl",
        "latest_annual_has_xbrl",
        "latest_quarter_xbrl",
        "latest_annual_xbrl",
        "latest_financial_format_new",
        "quarterly_filing_count",
        "annual_filing_count",
        "quarterly_period_count",
        "annual_period_count",
        "latest_shareholding_date",
        "latest_shareholding_submission_dt",
        "latest_shareholding_record_id",
        "latest_shareholding_xbrl",
        "latest_promoter_holding",
        "latest_public_holding",
        "promoter_holding_qoq_change",
        "public_holding_qoq_change",
        "previous_score_file",
    ]
    scores = scores[[col for col in ordered_cols if col in scores.columns]]

    scores_path = scores_dir / f"screener_fundamental_scores_{score_on.strftime('%Y%m%d')}.csv"
    scores.to_csv(scores_path, index=False)
    return FundamentalScoreResult(scores_path=scores_path, scores=scores)


def filter_fundamentals_as_of(
    financial_results: pd.DataFrame,
    shareholding: pd.DataFrame,
    financial_announcements: pd.DataFrame,
    as_of: date,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    as_of_ts = pd.Timestamp(as_of) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)

    fin = financial_results.copy()
    if not fin.empty:
        fin_dt = parse_date_series(fin.get("filingDate", fin.get("broadCastDate", pd.Series(dtype=str))))
        fin = fin[fin_dt.le(as_of_ts).fillna(False)].copy()

    shp = shareholding.copy()
    if not shp.empty:
        shp_dt = parse_date_series(shp.get("submissionDate", shp.get("broadcastDate", pd.Series(dtype=str))))
        shp_period_dt = parse_date_series(shp.get("date", pd.Series(dtype=str)))
        shp = shp[shp_dt.le(as_of_ts).fillna(False) & shp_period_dt.le(as_of_ts).fillna(False)].copy()

    ann = financial_announcements.copy()
    if not ann.empty:
        ann_dt = parse_date_series(ann.get("announcement_dt", pd.Series(dtype=str)))
        ann_period_dt = parse_date_series(ann.get("result_period_end", pd.Series(dtype=str)))
        ann = ann[ann_dt.le(as_of_ts).fillna(False) & ann_period_dt.le(as_of_ts).fillna(False)].copy()

    return fin, shp, ann


def score_dates_from_price_files(output_dir: Path, start: date | None = None, end: date | None = None) -> list[date]:
    prices_dir = output_dir / "prices_by_day"
    dates = []
    for path in sorted(prices_dir.glob("nifty50_prices_*.csv")):
        date_part = path.stem.replace("nifty50_prices_", "")
        try:
            score_date = datetime.strptime(date_part, "%Y%m%d").date()
        except ValueError:
            continue
        if start and score_date < start:
            continue
        if end and score_date > end:
            continue
        dates.append(score_date)
    return dates


def generate_fundamental_score_history(
    financial_results: pd.DataFrame,
    shareholding: pd.DataFrame,
    financial_announcements: pd.DataFrame,
    output_dir: Path = Path("data_cache/nse_equity"),
    symbols: Iterable[str] | None = None,
    start: date | None = None,
    end: date | None = None,
) -> FundamentalScoreHistoryResult:
    score_dates = score_dates_from_price_files(output_dir, start=start, end=end)
    if not score_dates:
        if start is None or end is None:
            raise RuntimeError("No price-by-day files found and no explicit score history date range supplied.")
        score_dates = list(iter_dates(start, end))

    history_dir = output_dir / "fundamentals" / "fundamental_scores_history"
    history_dir.mkdir(parents=True, exist_ok=True)
    wanted_date_values = {score_date.isoformat() for score_date in score_dates}
    existing_by_date: dict[str, pd.DataFrame] = {}
    existing_path_by_date: dict[str, Path] = {}
    for path in sorted(history_dir.glob("*fundamental_scores_history_*.csv")):
        existing = pd.read_csv(path)
        if existing.empty or "score_date" not in existing.columns:
            continue
        existing["score_date"] = parse_date_series(existing["score_date"]).dt.strftime("%Y-%m-%d")
        existing = existing[existing["score_date"].isin(wanted_date_values)].copy()
        for score_date, day_scores in existing.groupby("score_date", sort=False):
            existing_by_date[score_date] = day_scores.copy()
            existing_path_by_date[score_date] = path

    daily_scores_dir = output_dir / "fundamentals" / "fundamental_scores_by_day"
    for score_date in score_dates:
        score_date_value = score_date.isoformat()
        if score_date_value in existing_by_date:
            continue
        day_path = daily_scores_dir / f"screener_fundamental_scores_{score_date.strftime('%Y%m%d')}.csv"
        if not day_path.exists():
            legacy_day_path = daily_scores_dir / f"nse_fundamental_scores_{score_date.strftime('%Y%m%d')}.csv"
            day_path = legacy_day_path if legacy_day_path.exists() else day_path
        if not day_path.exists():
            continue
        day_scores = pd.read_csv(day_path)
        if day_scores.empty:
            continue
        if "score_date" not in day_scores.columns:
            day_scores["score_date"] = score_date_value
        day_scores["score_date"] = parse_date_series(day_scores["score_date"]).dt.strftime("%Y-%m-%d")
        day_scores = day_scores[day_scores["score_date"].eq(score_date_value)].copy()
        if day_scores.empty:
            continue
        existing_by_date[score_date_value] = day_scores
        existing_path_by_date[score_date_value] = day_path

    all_scores = []
    score_paths = []
    for score_date in score_dates:
        score_date_value = score_date.isoformat()
        if score_date_value in existing_by_date:
            all_scores.append(existing_by_date[score_date_value])
            if score_date_value in existing_path_by_date:
                score_paths.append(existing_path_by_date[score_date_value])
            continue
        fin_asof, shp_asof, ann_asof = filter_fundamentals_as_of(
            financial_results,
            shareholding,
            financial_announcements,
            as_of=score_date,
        )
        result = generate_fundamental_scores(
            financial_results=fin_asof,
            shareholding=shp_asof,
            financial_announcements=ann_asof,
            output_dir=output_dir,
            symbols=symbols,
            score_date=score_date,
        )
        all_scores.append(result.scores)
        score_paths.append(result.scores_path)

    history = pd.concat(all_scores, ignore_index=True, sort=False) if all_scores else pd.DataFrame()
    if not history.empty and {"score_date", "symbol"}.issubset(history.columns):
        history = (
            history.sort_values(["score_date", "symbol"])
            .drop_duplicates(["score_date", "symbol"], keep="last")
            .reset_index(drop=True)
        )
    history_path = history_dir / (
        f"screener_fundamental_scores_history_{score_dates[0].strftime('%Y%m%d')}_"
        f"{score_dates[-1].strftime('%Y%m%d')}.csv"
    )
    for path in history_dir.glob("*fundamental_scores_history_*.csv"):
        if path != history_path:
            path.unlink()
    history.to_csv(history_path, index=False)
    return FundamentalScoreHistoryResult(
        history_path=history_path,
        score_paths=tuple(score_paths),
        scores=history,
    )


def load_cached_price_history(
    output_dir: Path = Path("data_cache/nse_equity"),
    symbols: Iterable[str] | None = None,
    start: date | None = None,
    end: date | None = None,
) -> pd.DataFrame:
    prices_dir = output_dir / "prices_by_day"
    wanted = {symbol.upper() for symbol in symbols} if symbols else None
    frames = []
    for path in sorted(prices_dir.glob("nifty50_prices_*.csv")):
        date_part = path.stem.replace("nifty50_prices_", "")
        try:
            trade_date = datetime.strptime(date_part, "%Y%m%d").date()
        except ValueError:
            continue
        if start and trade_date < start:
            continue
        if end and trade_date > end:
            continue

        df = pd.read_csv(path)
        if df.empty or "symbol" not in df.columns:
            continue
        df["symbol"] = df["symbol"].astype(str).str.upper()
        if wanted:
            df = df[df["symbol"].isin(wanted)].copy()
        if df.empty:
            continue
        frames.append(df)

    if not frames:
        raise RuntimeError("No cached price-by-day rows found for technical score generation.")

    prices = pd.concat(frames, ignore_index=True, sort=False)
    prices["trade_date"] = parse_date_series(prices.get("trade_date", prices.get("business_date", pd.Series(dtype=str))))
    prices = prices.dropna(subset=["trade_date", "symbol"]).copy()
    prices["trade_date"] = prices["trade_date"].dt.date
    numeric_cols = ["open", "high", "low", "close", "last", "previous_close", "volume", "turnover", "trades"]
    for col in numeric_cols:
        if col in prices.columns:
            prices[col] = pd.to_numeric(prices[col], errors="coerce")
    return prices.sort_values(["symbol", "trade_date"]).reset_index(drop=True)


def cross_sectional_percentile(values: pd.Series, higher_is_better: bool = True) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() == 0:
        return pd.Series(pd.NA, index=values.index, dtype="Float64")
    return numeric.rank(pct=True, ascending=not higher_is_better) * 100


def cleanup_stale_score_files(directory: Path, pattern: str, valid_dates: set[str]) -> int:
    removed = 0
    for path in directory.glob(pattern):
        date_part = path.stem.rsplit("_", 1)[-1]
        if date_part not in valid_dates:
            path.unlink()
            removed += 1
    return removed


def generate_technical_score_history(
    output_dir: Path = Path("data_cache/nse_equity"),
    symbols: Iterable[str] | None = None,
    start: date | None = None,
    end: date | None = None,
) -> TechnicalScoreHistoryResult:
    prices = load_cached_price_history(output_dir=output_dir, symbols=symbols, start=start, end=end)

    frames = []
    for symbol, group in prices.groupby("symbol", sort=True):
        g = group.sort_values("trade_date").copy()
        close = g["close"]
        high = g["high"] if "high" in g.columns else close
        low = g["low"] if "low" in g.columns else close
        volume = g["volume"] if "volume" in g.columns else pd.Series(pd.NA, index=g.index)

        delta = close.diff()
        gains = delta.clip(lower=0)
        losses = -delta.clip(upper=0)
        avg_gain = gains.rolling(14, min_periods=14).mean()
        avg_loss = losses.rolling(14, min_periods=14).mean()
        rs = avg_gain / avg_loss.replace(0, pd.NA)

        g["return_21d_pct"] = close.pct_change(21) * 100
        g["return_63d_pct"] = close.pct_change(63) * 100
        g["return_126d_pct"] = close.pct_change(126) * 100
        g["return_252d_pct"] = close.pct_change(252) * 100
        g["sma_20"] = close.rolling(20, min_periods=10).mean()
        g["sma_50"] = close.rolling(50, min_periods=25).mean()
        g["sma_200"] = close.rolling(200, min_periods=100).mean()
        g["volume_sma_20"] = volume.rolling(20, min_periods=10).mean()
        g["volume_vs_sma20_pct"] = ((volume / g["volume_sma_20"]) - 1) * 100
        g["rolling_high_252"] = high.rolling(252, min_periods=60).max()
        g["rolling_low_252"] = low.rolling(252, min_periods=60).min()
        g["discount_to_252d_high_pct"] = ((g["rolling_high_252"] - close) / g["rolling_high_252"]) * 100
        g["premium_to_252d_low_pct"] = ((close - g["rolling_low_252"]) / g["rolling_low_252"]) * 100
        g["discount_to_sma200_pct"] = ((g["sma_200"] - close) / g["sma_200"]) * 100
        g["sma50_over_sma200_pct"] = ((g["sma_50"] / g["sma_200"]) - 1) * 100
        g["close_vs_sma20_pct"] = ((close / g["sma_20"]) - 1) * 100
        rolling_std_20 = close.rolling(20, min_periods=10).std()
        g["bollinger_upper_20_2"] = g["sma_20"] + 2 * rolling_std_20
        g["bollinger_lower_20_2"] = g["sma_20"] - 2 * rolling_std_20
        band_width = g["bollinger_upper_20_2"] - g["bollinger_lower_20_2"]
        g["bollinger_pct_b"] = (close - g["bollinger_lower_20_2"]) / band_width.replace(0, pd.NA)
        g["rsi_14"] = 100 - (100 / (1 + rs))
        g["symbol"] = symbol
        frames.append(g)

    indicators = pd.concat(frames, ignore_index=True, sort=False)
    indicators["score_date"] = pd.to_datetime(indicators["trade_date"]).dt.date

    ranked_frames = []
    for _, day in indicators.groupby("score_date", sort=True):
        d = day.copy()
        d["value_discount_high_rank"] = cross_sectional_percentile(d["discount_to_252d_high_pct"], True)
        d["value_discount_sma200_rank"] = cross_sectional_percentile(d["discount_to_sma200_pct"], True)
        d["value_rsi_rank"] = cross_sectional_percentile(100 - pd.to_numeric(d["rsi_14"], errors="coerce"), True)
        d["value_bollinger_rank"] = cross_sectional_percentile(1 - pd.to_numeric(d["bollinger_pct_b"], errors="coerce"), True)
        d["momentum_21d_rank"] = cross_sectional_percentile(d["return_21d_pct"], True)
        d["momentum_63d_rank"] = cross_sectional_percentile(d["return_63d_pct"], True)
        d["momentum_126d_rank"] = cross_sectional_percentile(d["return_126d_pct"], True)
        d["momentum_trend_rank"] = cross_sectional_percentile(d["sma50_over_sma200_pct"], True)

        d["value_score_0_50"] = (
            0.40 * d["value_discount_high_rank"].fillna(50)
            + 0.25 * d["value_discount_sma200_rank"].fillna(50)
            + 0.20 * d["value_rsi_rank"].fillna(50)
            + 0.15 * d["value_bollinger_rank"].fillna(50)
        ) / 2
        d["momentum_score_0_50"] = (
            0.35 * d["momentum_21d_rank"].fillna(50)
            + 0.35 * d["momentum_63d_rank"].fillna(50)
            + 0.20 * d["momentum_126d_rank"].fillna(50)
            + 0.10 * d["momentum_trend_rank"].fillna(50)
        ) / 2
        d["technical_score"] = d["value_score_0_50"] + d["momentum_score_0_50"]
        ranked_frames.append(d)

    scores = pd.concat(ranked_frames, ignore_index=True, sort=False)
    scores["score_date"] = pd.to_datetime(scores["score_date"]).dt.date.astype(str)
    scores["technical_score"] = scores["technical_score"].round(2)
    scores["value_score_0_50"] = scores["value_score_0_50"].round(2)
    scores["momentum_score_0_50"] = scores["momentum_score_0_50"].round(2)

    scores["value_discount_high_weight"] = 0.40
    scores["value_discount_sma200_weight"] = 0.25
    scores["value_rsi_weight"] = 0.20
    scores["value_bollinger_weight"] = 0.15
    scores["momentum_21d_weight"] = 0.35
    scores["momentum_63d_weight"] = 0.35
    scores["momentum_126d_weight"] = 0.20
    scores["momentum_trend_weight"] = 0.10

    ordered_cols = [
        "score_date",
        "symbol",
        "technical_score",
        "value_score_0_50",
        "momentum_score_0_50",
        "open",
        "high",
        "low",
        "close",
        "previous_close",
        "volume",
        "turnover",
        "trades",
        "return_21d_pct",
        "return_63d_pct",
        "return_126d_pct",
        "return_252d_pct",
        "sma_20",
        "sma_50",
        "sma_200",
        "sma50_over_sma200_pct",
        "close_vs_sma20_pct",
        "volume_sma_20",
        "volume_vs_sma20_pct",
        "rolling_high_252",
        "rolling_low_252",
        "discount_to_252d_high_pct",
        "premium_to_252d_low_pct",
        "discount_to_sma200_pct",
        "bollinger_upper_20_2",
        "bollinger_lower_20_2",
        "bollinger_pct_b",
        "rsi_14",
        "value_discount_high_rank",
        "value_discount_sma200_rank",
        "value_rsi_rank",
        "value_bollinger_rank",
        "momentum_21d_rank",
        "momentum_63d_rank",
        "momentum_126d_rank",
        "momentum_trend_rank",
        "value_discount_high_weight",
        "value_discount_sma200_weight",
        "value_rsi_weight",
        "value_bollinger_weight",
        "momentum_21d_weight",
        "momentum_63d_weight",
        "momentum_126d_weight",
        "momentum_trend_weight",
        "instrument_name",
        "isin",
    ]
    scores = scores[[col for col in ordered_cols if col in scores.columns]].sort_values(["score_date", "symbol"])

    scores_dir = output_dir / "technicals" / "technical_scores_by_day"
    scores_dir.mkdir(parents=True, exist_ok=True)
    score_paths = []
    date_values = sorted(scores["score_date"].unique().tolist())
    valid_date_parts = {score_date.replace("-", "") for score_date in date_values}
    stale_count = cleanup_stale_score_files(scores_dir, "nse_technical_scores_*.csv", valid_date_parts)
    if stale_count:
        print(f"Removed stale technical score files: {stale_count}")

    for score_date, day in scores.groupby("score_date", sort=True):
        score_path = scores_dir / f"nse_technical_scores_{score_date.replace('-', '')}.csv"
        day.to_csv(score_path, index=False)
        score_paths.append(score_path)

    history_dir = output_dir / "technicals" / "technical_scores_history"
    history_dir.mkdir(parents=True, exist_ok=True)
    history_path = history_dir / (
        f"nse_technical_scores_history_{date_values[0].replace('-', '')}_"
        f"{date_values[-1].replace('-', '')}.csv"
    )
    for path in history_dir.glob("nse_technical_scores_history_*.csv"):
        if path != history_path:
            path.unlink()
    scores.to_csv(history_path, index=False)
    return TechnicalScoreHistoryResult(
        history_path=history_path,
        score_paths=tuple(score_paths),
        scores=scores,
    )


def latest_required_csv(directory: Path, pattern: str, label: str) -> Path:
    path = latest_csv_file(directory, pattern)
    if path is None:
        raise RuntimeError(f"Missing {label}; generate score history before running backtests.")
    return path


def latest_required_csv_any(directory: Path, patterns: Iterable[str], label: str) -> Path:
    if not directory.exists():
        raise RuntimeError(f"Missing {label}; generate score history before running backtests.")
    candidates = []
    for priority, pattern in enumerate(patterns):
        for path in directory.glob(pattern):
            candidates.append((priority, path.name, path))
    if not candidates:
        raise RuntimeError(f"Missing {label}; generate score history before running backtests.")
    return sorted(candidates)[-1][2]


def backtest_metrics(returns: pd.Series, trading_days: int = 252) -> dict[str, float | int]:
    daily = pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if daily.empty:
        return {
            "total_return": np.nan,
            "annualized_return": np.nan,
            "annualized_volatility": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "best_day": np.nan,
            "worst_day": np.nan,
            "positive_day_rate": np.nan,
            "observations": 0,
        }
    cumulative = (1 + daily).cumprod()
    total_return = cumulative.iloc[-1] - 1
    annualized_return = (1 + total_return) ** (trading_days / len(daily)) - 1 if 1 + total_return > 0 else np.nan
    annualized_volatility = daily.std(ddof=1) * np.sqrt(trading_days) if len(daily) > 1 else np.nan
    sharpe = annualized_return / annualized_volatility if annualized_volatility and annualized_volatility > 0 else np.nan
    drawdown = cumulative / cumulative.cummax() - 1
    return {
        "total_return": float(total_return),
        "annualized_return": float(annualized_return),
        "annualized_volatility": float(annualized_volatility),
        "sharpe": float(sharpe),
        "max_drawdown": float(drawdown.min()),
        "best_day": float(daily.max()),
        "worst_day": float(daily.min()),
        "positive_day_rate": float((daily > 0).mean()),
        "observations": int(len(daily)),
    }


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def black_scholes_call_price(
    spot: float,
    strike: float,
    time_to_expiry_years: float,
    volatility: float,
    risk_free_rate: float = 0.065,
    dividend_yield: float = 0.012,
) -> float:
    if spot <= 0 or strike <= 0 or time_to_expiry_years <= 0:
        return 0.0
    sigma = max(float(volatility), 0.01)
    sqrt_t = math.sqrt(time_to_expiry_years)
    d1 = (
        math.log(spot / strike)
        + (risk_free_rate - dividend_yield + 0.5 * sigma * sigma) * time_to_expiry_years
    ) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    return spot * math.exp(-dividend_yield * time_to_expiry_years) * normal_cdf(d1) - strike * math.exp(
        -risk_free_rate * time_to_expiry_years
    ) * normal_cdf(d2)


def generate_nifty_buy_write_backtest(
    output_dir: Path = Path("data_cache/nse_equity"),
    initial_capital: float = 100000.0,
    holding_days: int = 30,
    lookback_years: int = 2,
    risk_free_rate: float = 0.065,
    dividend_yield: float = 0.012,
) -> tuple[Path, Path, pd.DataFrame]:
    history = load_nifty_index_history(output_dir)
    if history.empty:
        raise RuntimeError("No cached NIFTY index history found for buy-write backtest.")
    history = history.sort_values("trade_date").drop_duplicates("trade_date", keep="last").reset_index(drop=True)
    end_date = history["trade_date"].max()
    start_cutoff = end_date - pd.DateOffset(years=lookback_years)
    history = history[history["trade_date"] >= start_cutoff].reset_index(drop=True)
    if len(history) < 25:
        raise RuntimeError("Need at least 25 cached NIFTY index observations for buy-write backtest.")

    history["daily_return"] = history["close"].pct_change()
    history["trailing_vol_30d"] = history["daily_return"].rolling(30, min_periods=10).std(ddof=1) * np.sqrt(252)

    trades = []
    equity = float(initial_capital)
    benchmark_equity = float(initial_capital)
    index_position = 1
    while index_position < len(history) - 1:
        entry = history.iloc[index_position]
        target_exit = entry["trade_date"] + pd.Timedelta(days=holding_days)
        exit_candidates = history.index[(history.index > index_position) & (history["trade_date"] >= target_exit)]
        if len(exit_candidates) == 0:
            break
        exit_position = int(exit_candidates[0])
        exit_row = history.iloc[exit_position]

        spot = float(entry["close"])
        exit_close = float(exit_row["close"])
        strike = round(spot / 50.0) * 50.0
        realized_vol = entry["trailing_vol_30d"]
        volatility = float(realized_vol) if pd.notna(realized_vol) and realized_vol > 0 else 0.18
        actual_holding_days = max(1, int((exit_row["trade_date"] - entry["trade_date"]).days))
        time_to_expiry = actual_holding_days / 365.0
        premium = black_scholes_call_price(
            spot=spot,
            strike=strike,
            time_to_expiry_years=time_to_expiry,
            volatility=volatility,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
        )
        call_payoff = max(exit_close - strike, 0.0)
        index_return = exit_close / spot - 1.0
        buy_write_return = (exit_close - spot + premium - call_payoff) / spot
        equity *= 1.0 + buy_write_return
        benchmark_equity *= 1.0 + index_return

        trades.append(
            {
                "trade_number": len(trades) + 1,
                "entry_date": entry["trade_date"].strftime("%Y-%m-%d"),
                "exit_date": exit_row["trade_date"].strftime("%Y-%m-%d"),
                "holding_calendar_days": actual_holding_days,
                "entry_close": spot,
                "exit_close": exit_close,
                "strike": strike,
                "moneyness": strike / spot - 1.0,
                "trailing_vol_30d": volatility,
                "risk_free_rate": risk_free_rate,
                "dividend_yield": dividend_yield,
                "call_premium": premium,
                "call_premium_pct": premium / spot,
                "call_payoff": call_payoff,
                "call_payoff_pct": call_payoff / spot,
                "index_return": index_return,
                "buy_write_return": buy_write_return,
                "strategy_value": equity,
                "strategy_cumulative_return": equity / initial_capital - 1.0,
                "benchmark_value": benchmark_equity,
                "benchmark_cumulative_return": benchmark_equity / initial_capital - 1.0,
                "option_model": "black_scholes_atm_call_trailing_30d_realized_vol",
                "source": "official_nse_nifty50_index_close_with_modelled_option_premium",
            }
        )
        index_position = exit_position + 1

    if not trades:
        raise RuntimeError("No complete 30-calendar-day NIFTY buy-write trades could be generated.")

    backtest = pd.DataFrame(trades)
    backtest_dir = output_dir / "backtests"
    backtest_dir.mkdir(parents=True, exist_ok=True)
    backtest_path = backtest_dir / "nifty_buy_write_30d_backtest.csv"
    trades_path = backtest_dir / "nifty_buy_write_30d_trades.csv"
    backtest.to_csv(backtest_path, index=False)
    backtest.to_csv(trades_path, index=False)
    return backtest_path, trades_path, backtest


def download_fred_series(series_id: str, timeout: int = 20) -> pd.DataFrame:
    response = requests.get(FRED_GRAPH_CSV, params={"id": series_id}, timeout=timeout)
    response.raise_for_status()
    frame = pd.read_csv(io.StringIO(response.text))
    if "observation_date" not in frame or series_id not in frame:
        raise RuntimeError(f"FRED response for {series_id} did not contain expected columns.")
    frame = frame.rename(columns={"observation_date": "observation_date", series_id: "raw_value"})
    frame["observation_date"] = pd.to_datetime(frame["observation_date"], errors="coerce")
    frame["raw_value"] = pd.to_numeric(frame["raw_value"].replace(".", pd.NA), errors="coerce")
    return frame.dropna(subset=["observation_date"]).sort_values("observation_date").reset_index(drop=True)


def latest_economy_history_path(output_dir: Path = Path("data_cache/nse_equity")) -> Path | None:
    return latest_csv_file(output_dir / "economy" / "economy_score_history", "fred_india_economy_scores_history_*.csv")


def rolling_percentile_score(values: pd.Series, higher_is_better: bool = True, window: int = 120, min_periods: int = 24) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")

    def percentile(window_values: np.ndarray) -> float:
        clean = pd.Series(window_values).dropna()
        if clean.empty:
            return np.nan
        current = clean.iloc[-1]
        rank = (clean <= current).mean() * 100
        return rank if higher_is_better else 100 - rank

    return numeric.rolling(window, min_periods=min_periods).apply(percentile, raw=True)


def economy_state_from_score(score: float | int | None) -> tuple[str, str]:
    if score is None or not np.isfinite(score):
        return "unknown", "Economy Unknown"
    if score >= 65:
        return "expansion", "Expansionary Economy"
    if score <= 40:
        return "slowdown", "Slowdown Economy"
    return "neutral", "Mixed Economy"


def generate_economy_score_history(
    output_dir: Path = Path("data_cache/nse_equity"),
    start: date | None = None,
    end: date | None = None,
) -> EconomyScoreResult:
    economy_dir = output_dir / "economy"
    variables_dir = economy_dir / "economic_variables_history"
    scores_dir = economy_dir / "economy_score_history"
    variables_dir.mkdir(parents=True, exist_ok=True)
    scores_dir.mkdir(parents=True, exist_ok=True)

    variable_frames = []
    component_frames = []
    for spec in ECONOMIC_SERIES:
        series = download_fred_series(str(spec["fred_id"]))
        series["series_id"] = spec["fred_id"]
        series["variable"] = spec["key"]
        series["label"] = spec["label"]
        series["source"] = spec["source"]
        series["transform"] = spec["transform"]
        if spec["transform"] == "yoy_pct":
            periods = 1 if str(spec["fred_id"]) == "MKTGDPINA646NWDB" else 12
            series["value"] = series["raw_value"].pct_change(periods) * 100
        else:
            series["value"] = series["raw_value"]
        score_input = (series["value"] - float(spec["target"])).abs() if "target" in spec else series["value"]
        series[str(spec["component"])] = rolling_percentile_score(score_input, bool(spec["higher_is_better"]))
        series["component_weight"] = float(spec["weight"])
        series["max_age_days"] = int(spec["max_age_days"])
        variable_frames.append(series)
        component_frame = series[["observation_date", str(spec["component"])]].copy()
        component_frame[f"{spec['component']}_observation_date"] = component_frame["observation_date"]
        component_frames.append(
            component_frame.set_index("observation_date")
        )

    variables = pd.concat(variable_frames, ignore_index=True, sort=False)
    variables_stamp = date.today().strftime("%Y%m%d")
    variables_path = variables_dir / f"fred_india_economic_variables_{variables_stamp}.csv"
    variables.to_csv(variables_path, index=False)

    components = pd.concat(component_frames, axis=1).sort_index()
    for spec in ECONOMIC_SERIES:
        col = str(spec["component"])
        components[col] = pd.to_numeric(components[col], errors="coerce").ffill()
        obs_col = f"{col}_observation_date"
        if obs_col in components:
            components[obs_col] = pd.to_datetime(components[obs_col], errors="coerce").ffill()

    index_history = load_nifty_index_history(output_dir)
    if not index_history.empty:
        dates = pd.Series(pd.to_datetime(index_history["trade_date"], errors="coerce")).dropna().sort_values()
    else:
        dates = pd.Series(pd.to_datetime(score_dates_from_price_files(output_dir, start=start, end=end)))
    if len(dates) == 0:
        raise RuntimeError("No cached NIFTY dates found for economy score alignment.")
    if start is not None:
        dates = dates[dates.dt.date >= start]
    if end is not None:
        dates = dates[dates.dt.date <= end]
    if len(dates) == 0:
        raise RuntimeError("No cached NIFTY dates remain after applying the economy score date range.")

    daily = pd.DataFrame({"score_date": dates.dt.normalize().drop_duplicates()})
    daily = pd.merge_asof(
        daily.sort_values("score_date"),
        components.reset_index().rename(columns={"observation_date": "score_date"}).sort_values("score_date"),
        on="score_date",
        direction="backward",
    )

    weighted_cols = []
    total_weight = 0.0
    for spec in ECONOMIC_SERIES:
        col = str(spec["component"])
        weight = float(spec["weight"])
        if col in daily:
            obs_col = f"{col}_observation_date"
            age_col = f"{col}_age_days"
            if obs_col in daily:
                daily[obs_col] = pd.to_datetime(daily[obs_col], errors="coerce")
                daily[age_col] = (pd.to_datetime(daily["score_date"]) - daily[obs_col]).dt.days
                daily.loc[daily[age_col] > int(spec["max_age_days"]), col] = np.nan
            weighted_cols.append(daily[col] * weight)
    valid_weight = pd.Series(0.0, index=daily.index)
    weighted_sum = pd.Series(0.0, index=daily.index)
    for spec in ECONOMIC_SERIES:
        col = str(spec["component"])
        if col in daily:
            weight = float(spec["weight"])
            valid = daily[col].notna()
            valid_weight = valid_weight + valid.astype(float) * weight
            weighted_sum = weighted_sum + daily[col].fillna(0) * weight
    daily["economy_score"] = weighted_sum / valid_weight.replace(0, np.nan)
    state_values = daily["economy_score"].apply(economy_state_from_score)
    daily["economy_state"] = state_values.apply(lambda item: item[0])
    daily["economy_state_label"] = state_values.apply(lambda item: item[1])
    daily["score_source"] = "FRED India macro series"
    daily["score_date"] = daily["score_date"].dt.strftime("%Y-%m-%d")

    latest_values = (
        variables.sort_values("observation_date")
        .dropna(subset=["value"])
        .groupby("variable", as_index=False)
        .tail(1)[["variable", "observation_date", "value"]]
    )
    for _, row in latest_values.iterrows():
        daily[f"{row['variable']}_latest_observation_date"] = row["observation_date"].strftime("%Y-%m-%d")
        daily[f"{row['variable']}_latest_value"] = row["value"]

    component_cols = [str(spec["component"]) for spec in ECONOMIC_SERIES]
    component_age_cols = [f"{spec['component']}_age_days" for spec in ECONOMIC_SERIES]
    component_obs_cols = [f"{spec['component']}_observation_date" for spec in ECONOMIC_SERIES]
    latest_value_cols = [f"{spec['key']}_latest_value" for spec in ECONOMIC_SERIES]
    latest_date_cols = [f"{spec['key']}_latest_observation_date" for spec in ECONOMIC_SERIES]
    ordered_cols = [
        "score_date",
        "economy_score",
        "economy_state",
        "economy_state_label",
        *component_cols,
        *component_age_cols,
        *component_obs_cols,
        *latest_value_cols,
        *latest_date_cols,
        "score_source",
    ]
    daily = daily[[col for col in ordered_cols if col in daily]].sort_values("score_date")
    history_path = scores_dir / (
        f"fred_india_economy_scores_history_{daily['score_date'].iloc[0].replace('-', '')}_"
        f"{daily['score_date'].iloc[-1].replace('-', '')}.csv"
    )
    for path in scores_dir.glob("fred_india_economy_scores_history_*.csv"):
        if path != history_path:
            path.unlink()
    daily.to_csv(history_path, index=False)
    return EconomyScoreResult(variables_path=variables_path, history_path=history_path, scores=daily)


def load_nifty_index_history(output_dir: Path = Path("data_cache/nse_equity")) -> pd.DataFrame:
    files = sorted((output_dir / "index_history").glob("nifty50_index_history_*.csv"))
    files.extend(sorted((output_dir / "index_by_day").glob("nifty50_index_*.csv")))
    frames = [pd.read_csv(path) for path in files if path.exists()]
    history = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    if history.empty or "trade_date" not in history or "close" not in history:
        return pd.DataFrame()
    history = history.copy()
    history["trade_date"] = pd.to_datetime(history["trade_date"], errors="coerce")
    history["close"] = pd.to_numeric(history["close"], errors="coerce")
    history = history.dropna(subset=["trade_date", "close"]).sort_values("trade_date")
    return history.drop_duplicates("trade_date", keep="last").reset_index(drop=True)


def relative_strength_index(close: pd.Series, window: int = 14) -> pd.Series:
    delta = pd.to_numeric(close, errors="coerce").diff()
    gain = delta.clip(lower=0).rolling(window, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window, min_periods=window).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)


def nifty_index_dashboard_payload(output_dir: Path = Path("data_cache/nse_equity")) -> dict | None:
    history = load_nifty_index_history(output_dir)
    if history.empty:
        return None
    history = history.sort_values("trade_date").copy()
    close = history["close"]
    history["sma50"] = close.rolling(50, min_periods=20).mean()
    history["sma200"] = close.rolling(200, min_periods=60).mean()
    history["return63d"] = close.pct_change(63)
    history["rsi14"] = relative_strength_index(close, 14)
    latest = history.dropna(subset=["close"]).iloc[-1]
    close_value = float(latest["close"])
    sma50 = float(latest["sma50"]) if pd.notna(latest["sma50"]) else np.nan
    sma200 = float(latest["sma200"]) if pd.notna(latest["sma200"]) else np.nan
    return63d = float(latest["return63d"]) if pd.notna(latest["return63d"]) else np.nan
    rsi14 = float(latest["rsi14"]) if pd.notna(latest["rsi14"]) else np.nan
    if np.isfinite(sma50) and np.isfinite(sma200) and np.isfinite(return63d) and close_value > sma200 and sma50 > sma200 and return63d > 0.03:
        regime = "bull"
        label = "Bull Market"
    elif np.isfinite(sma50) and np.isfinite(sma200) and np.isfinite(return63d) and close_value < sma200 and sma50 < sma200 and return63d < -0.03:
        regime = "bear"
        label = "Bear Market"
    else:
        regime = "rangebound"
        label = "Range-bound Market"

    return {
        "source": "Official NSE index close archive",
        "regime": {
            "key": regime,
            "label": label,
            "asOf": latest["trade_date"].strftime("%Y-%m-%d"),
            "close": json_clean(close_value),
            "sma50": json_clean(sma50),
            "sma200": json_clean(sma200),
            "return63d": json_clean(return63d),
            "rsi14": json_clean(rsi14),
        },
        "series": {
            "dates": history["trade_date"].dt.strftime("%Y-%m-%d").tolist(),
            "close": [json_clean(value) for value in history["close"]],
            "sma50": [json_clean(value) for value in history["sma50"]],
            "sma200": [json_clean(value) for value in history["sma200"]],
            "return63d": [json_clean(value) for value in history["return63d"]],
            "rsi14": [json_clean(value) for value in history["rsi14"]],
        },
    }


def economy_dashboard_payload(output_dir: Path = Path("data_cache/nse_equity")) -> dict | None:
    path = latest_economy_history_path(output_dir)
    if path is None or not path.exists():
        return None
    history = pd.read_csv(path)
    if history.empty or "score_date" not in history or "economy_score" not in history:
        return None
    history = history.copy()
    history["score_date"] = pd.to_datetime(history["score_date"], errors="coerce")
    history["economy_score"] = pd.to_numeric(history["economy_score"], errors="coerce")
    history = history.dropna(subset=["score_date"]).sort_values("score_date")
    latest = history.dropna(subset=["economy_score"]).iloc[-1] if history["economy_score"].notna().any() else history.iloc[-1]
    state_key, state_label = economy_state_from_score(float(latest["economy_score"]) if pd.notna(latest["economy_score"]) else np.nan)
    variable_payload = []
    for spec in ECONOMIC_SERIES:
        value_col = f"{spec['key']}_latest_value"
        date_col = f"{spec['key']}_latest_observation_date"
        variable_payload.append(
            {
                "key": spec["key"],
                "label": spec["label"],
                "source": spec["source"],
                "value": json_clean(latest.get(value_col)),
                "observationDate": str(latest.get(date_col, "")) if pd.notna(latest.get(date_col, pd.NA)) else "",
                "weight": json_clean(spec["weight"]),
            }
        )
    return {
        "source": "FRED India macro series: CPI, industrial production, exports, short-term interest rate, GDP",
        "state": {
            "key": state_key,
            "label": state_label,
            "asOf": latest["score_date"].strftime("%Y-%m-%d"),
            "score": json_clean(latest.get("economy_score")),
        },
        "variables": variable_payload,
        "series": {
            "dates": history["score_date"].dt.strftime("%Y-%m-%d").tolist(),
            "score": [json_clean(value) for value in history["economy_score"]],
            "state": history.get("economy_state", pd.Series(dtype=str)).astype(str).tolist(),
        },
    }


def add_benchmark_columns(
    backtest: pd.DataFrame,
    prices: pd.DataFrame,
    initial_capital: float,
    output_dir: Path = Path("data_cache/nse_equity"),
) -> pd.DataFrame:
    if backtest.empty:
        return backtest
    index_history = load_nifty_index_history(output_dir)
    if not index_history.empty:
        index_frame = index_history[["trade_date", "close"]].rename(columns={"trade_date": "date", "close": "benchmark_close"})
        index_frame["date"] = pd.to_datetime(index_frame["date"], errors="coerce")
        index_frame = index_frame.dropna(subset=["date"]).sort_values("date")
        returns = index_frame["benchmark_close"].pct_change(fill_method=None).fillna(0.0)
        benchmark = pd.DataFrame(
            {
                "date": index_frame["date"].dt.strftime("%Y-%m-%d"),
                "benchmark_daily_return": returns.to_numpy(),
                "benchmark_cumulative_return": (1 + returns).cumprod().to_numpy() - 1,
                "benchmark_source": "official_nse_nifty50_index",
            }
        )
        benchmark["benchmark_value"] = initial_capital * (1 + benchmark["benchmark_cumulative_return"])
        out = backtest.merge(benchmark, on="date", how="left")
        if out["benchmark_daily_return"].notna().any():
            return out

    pivot = prices.pivot_table(index="score_date", columns="symbol", values="close", aggfunc="last").sort_index()
    benchmark_return = pivot.pct_change(fill_method=None).mean(axis=1, skipna=True).fillna(0.0)
    benchmark = pd.DataFrame(
        {
            "date": benchmark_return.index.strftime("%Y-%m-%d"),
            "benchmark_daily_return": benchmark_return.to_numpy(),
            "benchmark_cumulative_return": (1 + benchmark_return).cumprod().to_numpy() - 1,
            "benchmark_source": "equal_weight_nifty50_proxy",
        }
    )
    benchmark["benchmark_value"] = initial_capital * (1 + benchmark["benchmark_cumulative_return"])
    return backtest.merge(benchmark, on="date", how="left")


def optimize_mvo_weights(history: pd.DataFrame, min_weight: float = 0.02, max_weight: float = 0.20) -> np.ndarray:
    n = history.shape[1]
    if n == 0:
        return np.array([])
    if history.shape[0] < 20:
        return np.repeat(1 / n, n)
    mu = history.mean().to_numpy() * 252
    cov = history.cov().to_numpy() * 252
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0) + np.eye(n) * 1e-6

    def objective(weights: np.ndarray) -> float:
        port_return = float(np.dot(weights, mu))
        port_vol = float(np.sqrt(max(np.dot(weights, np.dot(cov, weights)), 1e-12)))
        return -(port_return / port_vol)

    x0 = np.repeat(1 / n, n)
    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=[(min_weight, max_weight)] * n,
        constraints=[{"type": "eq", "fun": lambda weights: np.sum(weights) - 1}],
        options={"maxiter": 500, "ftol": 1e-10},
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        return x0
    weights = np.clip(result.x, min_weight, max_weight)
    return weights / weights.sum()


def optimize_score_weights(
    history: pd.DataFrame,
    scores: pd.Series,
    min_weight: float = 0.02,
    max_weight: float = 0.25,
    risk_aversion: float = 0.55,
) -> np.ndarray:
    symbols = list(scores.index)
    n = len(symbols)
    if n == 0:
        return np.array([])
    clean_scores = pd.to_numeric(scores, errors="coerce").fillna(scores.median() if scores.notna().any() else 50)
    score_alpha = ((clean_scores - clean_scores.min()) / max(clean_scores.max() - clean_scores.min(), 1e-9)).to_numpy()
    if history is None or history.empty or history.shape[0] < 20:
        base = np.maximum(score_alpha, 0.05)
        return base / base.sum()

    hist = history[symbols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    mu = hist.mean().to_numpy() * 252
    vol = hist.std(ddof=1).replace(0, np.nan).fillna(hist.stack().std() or 0.01).to_numpy() * np.sqrt(252)
    cov = np.nan_to_num(hist.cov().to_numpy() * 252, nan=0.0, posinf=0.0, neginf=0.0) + np.eye(n) * 1e-6
    alpha = 0.55 * score_alpha + 0.30 * ((mu - np.nanmin(mu)) / max(np.nanmax(mu) - np.nanmin(mu), 1e-9)) + 0.15 * (1 - ((vol - np.nanmin(vol)) / max(np.nanmax(vol) - np.nanmin(vol), 1e-9)))

    def objective(weights: np.ndarray) -> float:
        expected = float(np.dot(weights, alpha))
        risk = float(np.sqrt(max(np.dot(weights, np.dot(cov, weights)), 1e-12)))
        concentration = float(np.sum(weights ** 2))
        return -(expected - risk_aversion * risk - 0.05 * concentration)

    x0 = np.maximum(score_alpha, 0.05)
    x0 = x0 / x0.sum()
    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=[(min_weight, max_weight)] * n,
        constraints=[{"type": "eq", "fun": lambda weights: np.sum(weights) - 1}],
        options={"maxiter": 500, "ftol": 1e-10},
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        return x0
    weights = np.clip(result.x, min_weight, max_weight)
    return weights / weights.sum()


def add_market_regime(data: pd.DataFrame) -> pd.DataFrame:
    pivot = data.pivot_table(index="score_date", columns="symbol", values="close", aggfunc="last").sort_index()
    benchmark_return = pivot.pct_change(fill_method=None).mean(axis=1, skipna=True).fillna(0.0)
    benchmark_cumulative_63d = (1 + benchmark_return).rolling(63, min_periods=20).apply(np.prod, raw=True) - 1
    benchmark_vol_21d = benchmark_return.rolling(21, min_periods=10).std() * np.sqrt(252)
    regime = pd.Series("rangebound", index=benchmark_return.index)
    regime[benchmark_cumulative_63d > 0.05] = "bull"
    regime[benchmark_cumulative_63d < -0.05] = "bear"
    regime[(benchmark_cumulative_63d.abs() <= 0.05) | benchmark_cumulative_63d.isna()] = "rangebound"
    out = data.merge(
        pd.DataFrame(
            {
                "score_date": regime.index,
                "market_regime": regime.values,
                "benchmark_63d_return": benchmark_cumulative_63d.values,
                "benchmark_21d_volatility": benchmark_vol_21d.values,
            }
        ),
        on="score_date",
        how="left",
    )
    out = out.sort_values(["symbol", "score_date"])
    out["symbol_volatility_63d"] = out.groupby("symbol")["symbol_return"].transform(lambda s: s.rolling(63, min_periods=20).std() * np.sqrt(252))
    out["symbol_return_63d"] = out.groupby("symbol")["close"].transform(lambda s: s.pct_change(63))
    return out


def build_rebalance_holdings(
    strategy: str,
    blocks: list[dict],
    prices: pd.DataFrame,
    portfolio: pd.DataFrame,
    output_dir: Path,
) -> Path:
    price_lookup = prices.set_index(["score_date", "symbol"])["close"]
    portfolio_lookup = portfolio.assign(date=pd.to_datetime(portfolio["date"])).set_index("date")["portfolio_value"]
    previous_quantities: dict[str, int] = {}
    rows: list[dict] = []
    for block_number, block in enumerate(blocks, start=1):
        start_dt = pd.Timestamp(block["start_date"])
        end_dt = pd.Timestamp(block["end_date"])
        portfolio_value = float(portfolio_lookup.loc[start_dt]) if start_dt in portfolio_lookup.index else np.nan
        current_quantities: dict[str, int] = {}
        for holding in block["holdings"]:
            symbol = holding["symbol"]
            weight = float(holding["weight"])
            entry_close = float(price_lookup.loc[(start_dt, symbol)])
            end_close = float(price_lookup.loc[(end_dt, symbol)])
            target_allocation = portfolio_value * weight
            quantity = int(np.floor(target_allocation / entry_close)) if entry_close > 0 else 0
            previous_quantity = previous_quantities.get(symbol, 0)
            current_quantities[symbol] = quantity
            rows.append(
                {
                    "strategy": strategy,
                    "rebalance_block": block_number,
                    "start_date": start_dt.strftime("%Y-%m-%d"),
                    "end_date": end_dt.strftime("%Y-%m-%d"),
                    "holding_calendar_days": block.get("holding_calendar_days"),
                    "priced_holding_calendar_days": (end_dt.date() - start_dt.date()).days + 1,
                    "symbol": symbol,
                    "score": holding.get("score"),
                    "weight": weight,
                    "portfolio_value_at_rebalance": portfolio_value,
                    "target_allocation_value": target_allocation,
                    "allocation_value": quantity * entry_close,
                    "entry_close": entry_close,
                    "end_close": end_close,
                    "previous_quantity": previous_quantity,
                    "quantity": quantity,
                    "quantity_change": quantity - previous_quantity,
                    "period_return": (end_close / entry_close - 1) if entry_close > 0 else np.nan,
                }
            )
        previous_quantities = current_quantities
    path = output_dir / f"{strategy}_top10_score_weighted_holdings.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def calendar_rebalance_windows(
    dates: Iterable[pd.Timestamp | np.datetime64 | str],
    holding_calendar_days: int,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    ordered = [pd.Timestamp(value).normalize() for value in sorted(dates)]
    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    start_idx = 0
    while start_idx < len(ordered):
        start_dt = ordered[start_idx]
        target_end = start_dt + pd.Timedelta(days=max(holding_calendar_days, 1) - 1)
        end_idx = start_idx
        while end_idx + 1 < len(ordered) and ordered[end_idx + 1] <= target_end:
            end_idx += 1
        windows.append((start_dt, ordered[end_idx]))
        start_idx = end_idx + 1
    return windows


def generate_strategy_backtests(
    output_dir: Path = Path("data_cache/nse_equity"),
    initial_capital: float = 100000.0,
    transaction_cost: float = 0.0001,
    holding_calendar_days: int = 31,
    top_n: int = 10,
) -> BacktestResult:
    backtest_dir = output_dir / "backtests"
    backtest_dir.mkdir(parents=True, exist_ok=True)
    fundamental_path = latest_required_csv_any(
        output_dir / "fundamentals" / "fundamental_scores_history",
        ["nse_fundamental_scores_history_*.csv", "screener_fundamental_scores_history_*.csv"],
        "fundamental score history",
    )
    technical_path = latest_required_csv(
        output_dir / "technicals" / "technical_scores_history",
        "nse_technical_scores_history_*.csv",
        "technical score history",
    )
    fundamental = pd.read_csv(fundamental_path)
    technical = pd.read_csv(technical_path)
    technical_keep = [
        "score_date",
        "symbol",
        "technical_score",
        "value_score_0_50",
        "momentum_score_0_50",
        "return_21d_pct",
        "return_63d_pct",
        "return_126d_pct",
        "sma50_over_sma200_pct",
        "discount_to_252d_high_pct",
        "discount_to_sma200_pct",
        "bollinger_pct_b",
        "rsi_14",
        "close",
    ]
    technical = technical[[col for col in technical_keep if col in technical.columns]]
    fundamental_keep = [
        "score_date",
        "symbol",
        "fundamental_score",
        "growth_score",
        "profitability_score",
        "efficiency_score",
        "shareholding_quality_score",
        "freshness_score",
    ]
    fundamental = fundamental[[col for col in fundamental_keep if col in fundamental.columns]]
    for frame in (fundamental, technical):
        frame["score_date"] = pd.to_datetime(frame["score_date"])
        frame["symbol"] = frame["symbol"].astype(str).str.upper()
    data = technical.merge(fundamental, on=["score_date", "symbol"], how="inner")
    for column in [
        "fundamental_score",
        "technical_score",
        "value_score_0_50",
        "momentum_score_0_50",
        "return_21d_pct",
        "return_63d_pct",
        "return_126d_pct",
        "sma50_over_sma200_pct",
        "discount_to_252d_high_pct",
        "discount_to_sma200_pct",
        "bollinger_pct_b",
        "rsi_14",
        "growth_score",
        "profitability_score",
        "efficiency_score",
        "shareholding_quality_score",
        "freshness_score",
        "close",
    ]:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    data["average_score"] = data[["fundamental_score", "technical_score"]].mean(axis=1)
    data = data.sort_values(["symbol", "score_date"])
    data["symbol_return"] = data.groupby("symbol")["close"].pct_change().fillna(0.0)

    selected_metrics_frames = []
    backtest_paths: list[Path] = []
    holdings_paths: list[Path] = []
    symbol_metrics_paths: list[Path] = []

    for strategy, score_column in [
        ("fundamental", "fundamental_score"),
        ("technical", "technical_score"),
        ("average", "average_score"),
    ]:
        scored = data.copy()
        scored["score_exposure"] = scored[score_column].clip(lower=0, upper=100) / 100
        scored["weighted_return"] = scored["symbol_return"] * scored["score_exposure"]
        symbol_metrics = []
        for symbol, symbol_frame in scored.groupby("symbol"):
            metrics = backtest_metrics(symbol_frame.sort_values("score_date")["weighted_return"])
            metrics.update({"symbol": symbol, "score_type": strategy})
            symbol_metrics.append(metrics)
        symbol_metrics_df = pd.DataFrame(symbol_metrics).sort_values("sharpe", ascending=False)
        metrics_path = backtest_dir / f"{strategy}_score_weighted_symbol_metrics.csv"
        symbol_metrics_df.to_csv(metrics_path, index=False)
        symbol_metrics_paths.append(metrics_path)
        selected_symbols = symbol_metrics_df.head(top_n)["symbol"].tolist()
        selected_metrics_frames.append(symbol_metrics_df.head(top_n))

        selected_data = scored[scored["symbol"].isin(selected_symbols)].copy()
        dates = sorted(selected_data["score_date"].dropna().unique())
        previous_exposure = {symbol: 0.0 for symbol in selected_symbols}
        equity = initial_capital
        rows = []
        for score_date in dates:
            day = selected_data[selected_data["score_date"].eq(score_date)]
            returns = day.set_index("symbol")["symbol_return"].to_dict()
            exposures = day.set_index("symbol")["score_exposure"].to_dict()
            gross_return = 0.0
            turnover = 0.0
            for symbol in selected_symbols:
                exposure = float(exposures.get(symbol, 0.0) or 0.0)
                gross_return += (1 / top_n) * exposure * float(returns.get(symbol, 0.0) or 0.0)
                turnover += (1 / top_n) * abs(exposure - previous_exposure.get(symbol, 0.0))
                previous_exposure[symbol] = exposure
            daily_return = gross_return - transaction_cost * turnover
            equity *= 1 + daily_return
            rows.append(
                {
                    "date": pd.Timestamp(score_date).strftime("%Y-%m-%d"),
                    "daily_return": daily_return,
                    "cumulative_return": equity / initial_capital - 1,
                    "portfolio_value": equity,
                }
            )
        backtest = add_benchmark_columns(pd.DataFrame(rows), data, initial_capital, output_dir=output_dir)
        backtest_path = backtest_dir / f"{strategy}_top10_score_weighted_backtest.csv"
        backtest.to_csv(backtest_path, index=False)
        backtest_paths.append(backtest_path)

        blocks = []
        for start_dt, end_dt in calendar_rebalance_windows(dates, holding_calendar_days):
            snap = selected_data[selected_data["score_date"].eq(start_dt)].copy()
            snap["score"] = snap[score_column].clip(lower=0)
            total_score = snap["score"].sum()
            snap["weight"] = snap["score"] / total_score if total_score else 1 / len(snap)
            blocks.append(
                {
                    "start_date": pd.Timestamp(start_dt).strftime("%Y-%m-%d"),
                    "end_date": pd.Timestamp(end_dt).strftime("%Y-%m-%d"),
                    "holding_calendar_days": holding_calendar_days,
                    "holdings": snap.sort_values("weight", ascending=False)[["symbol", "score", "weight"]].to_dict("records"),
                }
            )
        holdings_paths.append(build_rebalance_holdings(strategy, blocks, data, backtest, backtest_dir))

    selected_metrics_path = backtest_dir / "selected_top10_score_weighted_symbol_metrics.csv"
    pd.concat(selected_metrics_frames, ignore_index=True, sort=False).to_csv(selected_metrics_path, index=False)
    return BacktestResult(
        backtest_paths=tuple(backtest_paths),
        holdings_paths=tuple(holdings_paths),
        symbol_metrics_paths=tuple(symbol_metrics_paths),
        selected_metrics_path=selected_metrics_path,
    )


def json_clean(value):
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return round(float(value), 10) if np.isfinite(value) else None
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if pd.isna(value):
        return None
    return value


def display_label(column: str) -> str:
    replacements = {
        "rsi": "RSI",
        "sma": "SMA",
        "xbrl": "XBRL",
        "isin": "ISIN",
        "ohlcv": "OHLCV",
    }
    label = column.replace("_", " ").title()
    for old, new in replacements.items():
        label = re.sub(rf"\b{old.title()}\b", new, label)
    label = label.replace("Pct", "%").replace("0 50", "0-50")
    return label


def latest_score_histories(output_dir: Path) -> tuple[Path, Path]:
    fundamental_path = latest_required_csv_any(
        output_dir / "fundamentals" / "fundamental_scores_history",
        ["nse_fundamental_scores_history_*.csv", "screener_fundamental_scores_history_*.csv"],
        "fundamental score history",
    )
    technical_path = latest_required_csv(
        output_dir / "technicals" / "technical_scores_history",
        "nse_technical_scores_history_*.csv",
        "technical score history",
    )
    return fundamental_path, technical_path


def build_dashboard_data(output_dir: Path) -> dict:
    fundamental_path, technical_path = latest_score_histories(output_dir)
    fundamental = pd.read_csv(fundamental_path)
    technical = pd.read_csv(technical_path)
    for frame in (fundamental, technical):
        frame["score_date"] = pd.to_datetime(frame["score_date"])
        frame["symbol"] = frame["symbol"].astype(str).str.upper()
    merged = technical.merge(fundamental, on=["score_date", "symbol"], how="outer", suffixes=("", "_fund"))
    merged = merged.sort_values(["symbol", "score_date"])

    fundamental_exclude = {
        "score_date",
        "symbol",
        "fundamental_score",
        "score_source",
        "previous_score_file",
    }
    technical_exclude = {
        "score_date",
        "symbol",
        "instrument_name",
        "isin",
        "technical_score",
        "open",
        "high",
        "low",
        "close",
        "previous_close",
        "volume",
        "turnover",
        "trades",
    }

    fundamental_variables = []
    for column in fundamental.columns:
        if column in fundamental_exclude:
            continue
        numeric = pd.to_numeric(fundamental[column], errors="coerce")
        if numeric.notna().any():
            fundamental_variables.append({"key": column, "label": display_label(column)})
            merged[column] = pd.to_numeric(merged[column], errors="coerce")

    technical_variables = []
    for column in technical.columns:
        if column in technical_exclude:
            continue
        numeric = pd.to_numeric(technical[column], errors="coerce")
        if numeric.notna().any():
            technical_variables.append({"key": column, "label": display_label(column)})
            merged[column] = pd.to_numeric(merged[column], errors="coerce")

    symbols: dict[str, dict] = {}
    for symbol, group in merged.groupby("symbol", sort=True):
        group = group.sort_values("score_date")
        symbols[symbol] = {
            "dates": group["score_date"].dt.strftime("%Y-%m-%d").tolist(),
            "close": [json_clean(value) for value in pd.to_numeric(group.get("close"), errors="coerce")],
            "fundamental_score": [
                json_clean(value) for value in pd.to_numeric(group.get("fundamental_score"), errors="coerce")
            ],
            "technical_score": [
                json_clean(value) for value in pd.to_numeric(group.get("technical_score"), errors="coerce")
            ],
            "fundamentalVars": {
                item["key"]: [json_clean(value) for value in group[item["key"]]]
                for item in fundamental_variables
                if item["key"] in group
            },
            "technicalVars": {
                item["key"]: [json_clean(value) for value in group[item["key"]]]
                for item in technical_variables
                if item["key"] in group
            },
        }

    date_values = merged["score_date"].dropna().sort_values()
    return {
        "generatedFrom": {
            "fundamentalHistory": str(fundamental_path),
            "technicalHistory": str(technical_path),
        },
        "dateRange": [
            date_values.iloc[0].strftime("%Y-%m-%d"),
            date_values.iloc[-1].strftime("%Y-%m-%d"),
        ],
        "fundamentalVariables": fundamental_variables,
        "technicalVariables": technical_variables,
        "niftyIndex": nifty_index_dashboard_payload(output_dir),
        "economy": economy_dashboard_payload(output_dir),
        "symbols": symbols,
    }


def holdings_blocks_from_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    holdings = pd.read_csv(path)
    if holdings.empty:
        return []
    blocks = []
    max_block = int(holdings["rebalance_block"].max())
    for block_number, group in holdings.groupby("rebalance_block", sort=True):
        group = group.sort_values("weight", ascending=False)
        blocks.append(
            {
                "block": int(block_number),
                "startDate": str(group["start_date"].iloc[0]),
                "endDate": str(group["end_date"].iloc[0]),
                "isCurrent": int(block_number) == max_block,
                "portfolioValue": json_clean(group["portfolio_value_at_rebalance"].iloc[0]),
                "holdings": [
                    {
                        "symbol": row["symbol"],
                        "score": json_clean(row.get("score")),
                        "weight": json_clean(row.get("weight")),
                        "previousQuantity": json_clean(row.get("previous_quantity")),
                        "quantity": json_clean(row.get("quantity")),
                        "quantityChange": json_clean(row.get("quantity_change")),
                        "entryClose": json_clean(row.get("entry_close")),
                        "endClose": json_clean(row.get("end_close")),
                        "periodReturn": json_clean(row.get("period_return")),
                    }
                    for _, row in group.iterrows()
                ],
            }
        )
    return blocks


def build_backtest_dashboard_data(output_dir: Path) -> dict:
    backtest_dir = output_dir / "backtests"
    strategy_labels = {
        "fundamental": "Fundamental",
        "technical": "Technical",
        "average": "Average Score",
    }
    selected_path = backtest_dir / "selected_top10_score_weighted_symbol_metrics.csv"
    selected = pd.read_csv(selected_path) if selected_path.exists() else pd.DataFrame()
    payload: dict[str, dict] = {}
    benchmark_series = None
    benchmark_source = None
    for strategy, label in strategy_labels.items():
        backtest_path = backtest_dir / f"{strategy}_top10_score_weighted_backtest.csv"
        if not backtest_path.exists():
            continue
        backtest = pd.read_csv(backtest_path)
        if benchmark_series is None and {"benchmark_daily_return", "benchmark_cumulative_return", "benchmark_value"}.issubset(backtest.columns):
            benchmark_series = {
                "dates": backtest["date"].astype(str).tolist(),
                "dailyReturn": [json_clean(value) for value in backtest["benchmark_daily_return"]],
                "cumulativeReturn": [json_clean(value) for value in backtest["benchmark_cumulative_return"]],
                "portfolioValue": [json_clean(value) for value in backtest["benchmark_value"]],
            }
            if "benchmark_source" in backtest and backtest["benchmark_source"].notna().any():
                benchmark_source = str(backtest["benchmark_source"].dropna().iloc[-1])
        strategy_selected = selected[selected.get("score_type", pd.Series(dtype=str)).eq(strategy)].copy()
        if not strategy_selected.empty and "sharpe" in strategy_selected:
            top_symbols = strategy_selected.sort_values("sharpe", ascending=False)["symbol"].astype(str).tolist()
        else:
            top_symbols = []
        payload[strategy] = {
            "label": label,
            "topSymbols": top_symbols[:10],
            "metrics": {key: json_clean(value) for key, value in backtest_metrics(backtest["daily_return"]).items()},
            "series": {
                "dates": backtest["date"].astype(str).tolist(),
                "dailyReturn": [json_clean(value) for value in backtest["daily_return"]],
                "cumulativeReturn": [json_clean(value) for value in backtest["cumulative_return"]],
                "portfolioValue": [json_clean(value) for value in backtest["portfolio_value"]],
            },
            "holdings": holdings_blocks_from_csv(backtest_dir / f"{strategy}_top10_score_weighted_holdings.csv"),
        }
    if benchmark_series is not None:
        benchmark_label = (
            "Official NIFTY 50 Index"
            if benchmark_source == "official_nse_nifty50_index"
            else "Equal-weight NIFTY 50 proxy"
        )
        payload["benchmark"] = {
            "label": benchmark_label,
            "source": benchmark_source,
            "series": benchmark_series,
            "metrics": {key: json_clean(value) for key, value in backtest_metrics(pd.Series(benchmark_series["dailyReturn"])).items()},
        }
    return payload


def build_buy_write_dashboard_data(output_dir: Path) -> dict:
    path = output_dir / "backtests" / "nifty_buy_write_30d_trades.csv"
    if not path.exists():
        return {}
    trades = pd.read_csv(path)
    if trades.empty or "buy_write_return" not in trades:
        return {}
    period_returns = pd.to_numeric(trades["buy_write_return"], errors="coerce").fillna(0.0)
    benchmark_returns = pd.to_numeric(trades.get("index_return", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    periods_per_year = 365.0 / 30.0
    strategy_metrics = backtest_metrics(period_returns, trading_days=periods_per_year)
    benchmark_metrics = backtest_metrics(benchmark_returns, trading_days=periods_per_year)
    initial_capital = 100000.0
    final_value = trades["strategy_value"].iloc[-1] if "strategy_value" in trades else initial_capital * (1 + period_returns).prod()
    benchmark_final_value = (
        trades["benchmark_value"].iloc[-1]
        if "benchmark_value" in trades
        else initial_capital * (1 + benchmark_returns).prod()
    )
    strategy_metrics["final_value"] = json_clean(final_value)
    benchmark_metrics["final_value"] = json_clean(benchmark_final_value)
    return {
        "label": "NIFTY 30D Buy-Write",
        "source": str(trades.get("source", pd.Series(["modelled"])).dropna().iloc[-1]) if "source" in trades else "modelled",
        "model": str(trades.get("option_model", pd.Series(["modelled"])).dropna().iloc[-1]) if "option_model" in trades else "modelled",
        "dateRange": [str(trades["entry_date"].iloc[0]), str(trades["exit_date"].iloc[-1])],
        "metrics": {key: json_clean(value) for key, value in strategy_metrics.items()},
        "benchmarkMetrics": {key: json_clean(value) for key, value in benchmark_metrics.items()},
        "series": {
            "dates": trades["exit_date"].astype(str).tolist(),
            "strategyReturn": [json_clean(value) for value in trades["buy_write_return"]],
            "benchmarkReturn": [json_clean(value) for value in trades.get("index_return", pd.Series(dtype=float))],
            "cumulativeReturn": [json_clean(value) for value in trades.get("strategy_cumulative_return", pd.Series(dtype=float))],
            "benchmarkCumulativeReturn": [
                json_clean(value) for value in trades.get("benchmark_cumulative_return", pd.Series(dtype=float))
            ],
            "portfolioValue": [json_clean(value) for value in trades.get("strategy_value", pd.Series(dtype=float))],
            "benchmarkValue": [json_clean(value) for value in trades.get("benchmark_value", pd.Series(dtype=float))],
        },
        "trades": [
            {
                "tradeNumber": json_clean(row.get("trade_number")),
                "entryDate": row.get("entry_date"),
                "exitDate": row.get("exit_date"),
                "holdingDays": json_clean(row.get("holding_calendar_days")),
                "entryClose": json_clean(row.get("entry_close")),
                "exitClose": json_clean(row.get("exit_close")),
                "strike": json_clean(row.get("strike")),
                "trailingVol30d": json_clean(row.get("trailing_vol_30d")),
                "premiumPct": json_clean(row.get("call_premium_pct")),
                "payoffPct": json_clean(row.get("call_payoff_pct")),
                "indexReturn": json_clean(row.get("index_return")),
                "strategyReturn": json_clean(row.get("buy_write_return")),
                "cumulativeReturn": json_clean(row.get("strategy_cumulative_return")),
            }
            for _, row in trades.iterrows()
        ],
    }


def generate_dashboard_html(
    output_dir: Path = Path("data_cache/nse_equity"),
    template_path: Path = Path("fundamental_score_dashboard.html"),
    dashboards_dir: Path = Path("dashboards"),
) -> tuple[Path, Path]:
    if not template_path.exists():
        raise RuntimeError(f"Dashboard template not found: {template_path}")
    dashboard_data = build_dashboard_data(output_dir)
    backtest_data = build_backtest_dashboard_data(output_dir)
    buy_write_data = build_buy_write_dashboard_data(output_dir)
    html = template_path.read_text()
    replacements = {
        "dashboard-data": json.dumps(dashboard_data, separators=(",", ":")),
        "backtest-data": json.dumps(backtest_data, separators=(",", ":")),
        "buy-write-data": json.dumps(buy_write_data, separators=(",", ":")),
    }
    for script_id, payload in replacements.items():
        pattern = rf'(<script id="{script_id}" type="application/json">)(.*?)(</script>)'
        html, count = re.subn(pattern, lambda match: f"{match.group(1)}{payload}{match.group(3)}", html, flags=re.S)
        if count != 1:
            raise RuntimeError(f"Could not replace {script_id} JSON in {template_path}")

    end_date = dashboard_data["dateRange"][1].replace("-", "")
    dashboards_dir.mkdir(parents=True, exist_ok=True)
    dated_path = dashboards_dir / f"fundamental_score_dashboard_{end_date}.html"
    dated_path.write_text(html)
    template_path.write_text(html)
    return template_path, dated_path


def pull_nifty_prices(
    start: date,
    end: date,
    output_dir: Path = Path("data_cache/nse_equity"),
    symbols: Iterable[str] | None = None,
    limit: int | None = None,
    daily_files: bool = False,
    latest_available_only: bool = False,
    fallback_days: int = 10,
) -> PullResult:
    if end < start:
        raise ValueError("end date must be on or after start date")

    output_dir.mkdir(parents=True, exist_ok=True)
    bhavcopy_cache_dir = output_dir / "bhavcopy_cache"
    bhavcopy_cache_dir.mkdir(parents=True, exist_ok=True)

    client = NSEArchiveClient()
    constituents = get_nifty_constituents(client)
    pull_symbols = selected_nifty_symbols(constituents, symbols=symbols, limit=limit)

    if latest_available_only:
        candidate_dates = (end - timedelta(days=offset) for offset in range(fallback_days + 1))
    else:
        candidate_dates = iter_dates(start, end)

    history_frames = []
    constituents_paths: list[Path] = []
    prices_paths: list[Path] = []
    for trade_date in candidate_dates:
        if trade_date < start:
            break
        bhavcopy = get_cash_bhavcopy(client, trade_date, bhavcopy_cache_dir)
        normalized = normalize_bhavcopy(bhavcopy, pull_symbols)
        if not normalized.empty:
            history_frames.append(normalized)
            if daily_files:
                constituents_path, prices_path = write_daily_outputs(
                    normalized,
                    constituents,
                    trade_date,
                    output_dir,
                )
                constituents_paths.append(constituents_path)
                prices_paths.append(prices_path)
            if latest_available_only:
                break
        time.sleep(client.pause_seconds)

    prices = pd.concat(history_frames, ignore_index=True) if history_frames else pd.DataFrame()
    if daily_files:
        return PullResult(
            constituents_path=constituents_paths[-1] if constituents_paths else None,
            prices_path=prices_paths[-1] if prices_paths else None,
            constituents=constituents,
            prices=prices,
            constituents_paths=tuple(constituents_paths),
            prices_paths=tuple(prices_paths),
        )

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    start_stamp = start.strftime("%Y%m%d")
    end_stamp = end.strftime("%Y%m%d")

    constituents_path = output_dir / f"nifty50_constituents_{run_stamp}.csv"
    prices_path = output_dir / f"nifty50_prices_{start_stamp}_{end_stamp}_{run_stamp}.csv"
    constituents.to_csv(constituents_path, index=False)
    prices.to_csv(prices_path, index=False)

    return PullResult(
        constituents_path=constituents_path,
        prices_path=prices_path,
        constituents=constituents,
        prices=prices,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pull NIFTY 50 stock data from NSE.")
    parser.add_argument("--from", dest="start", type=parse_date)
    parser.add_argument("--to", dest="end", type=parse_date)
    parser.add_argument("--output-dir", type=Path, default=Path("data_cache/nse_equity"))
    parser.add_argument("--symbols", nargs="+", help="Optional NSE symbols, e.g. RELIANCE TCS INFY")
    parser.add_argument("--limit", type=int, help="Limit the number of NIFTY 50 symbols pulled")
    parser.add_argument(
        "--daily-files",
        action="store_true",
        help="Write one price CSV and one constituent CSV per available trading day",
    )
    parser.add_argument(
        "--last-year",
        action="store_true",
        help="Pull the last 365 calendar days ending on --to, or yesterday by default",
    )
    parser.add_argument(
        "--previous-close",
        action="store_true",
        help="Pull the latest available NSE close on or before yesterday",
    )
    parser.add_argument(
        "--index-history",
        action="store_true",
        help="Also download NIFTY 50 index close history from NSE archives",
    )
    parser.add_argument(
        "--index-history-only",
        action="store_true",
        help="Download only NIFTY 50 index close history from NSE archives",
    )
    parser.add_argument(
        "--economy-history",
        action="store_true",
        help="Download FRED India macro series and generate daily economy score history",
    )
    parser.add_argument(
        "--economy-history-only",
        action="store_true",
        help="Generate economy score history from FRED without pulling NSE data",
    )
    parser.add_argument(
        "--fallback-days",
        type=int,
        default=10,
        help="Lookback window used by --previous-close to skip holidays/weekends",
    )
    parser.add_argument(
        "--fundamentals",
        action="store_true",
        help="Also download fundamental filings from the selected source",
    )
    parser.add_argument(
        "--fundamentals-only",
        action="store_true",
        help="Download only fundamental filings from the selected source",
    )
    parser.add_argument(
        "--fundamentals-years",
        type=int,
        default=8,
        help="Financial Results lookback window in years for quarterly and annual filings",
    )
    parser.add_argument(
        "--fundamentals-source",
        choices=["auto", "screener", "moneycontrol", "economictimes", "cached"],
        default="auto",
        help="Fundamental data source: auto tries Screener, Moneycontrol, Economic Times, then cached files",
    )
    parser.add_argument(
        "--fundamental-scores",
        action="store_true",
        help="Generate daily NIFTY fundamental score CSV from downloaded fundamentals",
    )
    parser.add_argument(
        "--fundamental-score-history",
        action="store_true",
        help="Generate one fundamental score CSV per cached business day plus a combined history CSV",
    )
    parser.add_argument(
        "--scores-only",
        action="store_true",
        help="Download fundamentals and generate only the daily fundamental score CSV",
    )
    parser.add_argument(
        "--score-history-only",
        action="store_true",
        help="Generate historical fundamental scores from cached fundamentals without pulling NSE data",
    )
    parser.add_argument(
        "--technical-score-history",
        action="store_true",
        help="Generate one technical score CSV per cached business day plus a combined history CSV",
    )
    parser.add_argument(
        "--technical-scores",
        action="store_true",
        help="Generate technical score CSVs from all cached price history, suitable for daily runs",
    )
    parser.add_argument(
        "--technical-score-history-only",
        action="store_true",
        help="Generate historical technical scores from cached price files without pulling NSE data",
    )
    parser.add_argument(
        "--backtests",
        action="store_true",
        help="Generate daily strategy backtests and rebalance holdings from cached score histories",
    )
    parser.add_argument(
        "--buy-write-backtest",
        action="store_true",
        help="Generate 30-calendar-day modelled NIFTY index buy-write backtest from cached index history",
    )
    parser.add_argument(
        "--dashboard-html",
        action="store_true",
        help="Generate latest and datestamped standalone HTML dashboard from cached histories and backtests",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    explicit_fundamentals_only = args.fundamentals_only
    default_end = date.today() - timedelta(days=1)
    default_start = default_end - timedelta(days=30)
    if args.previous_close:
        args.end = default_end
        args.start = args.end - timedelta(days=max(args.fallback_days, 0))
        args.daily_files = True
    elif args.last_year:
        args.end = args.end or default_end
        args.start = args.end - timedelta(days=365)

    if args.scores_only:
        args.fundamentals_only = True
        args.fundamental_scores = True
    if args.score_history_only:
        args.fundamentals_only = True
        args.fundamental_score_history = True
    if args.technical_score_history_only:
        args.fundamentals_only = True
        args.technical_score_history = True
    if args.index_history_only:
        args.fundamentals_only = True
        args.index_history = True
    if args.economy_history_only:
        args.fundamentals_only = True
        args.economy_history = True
    if (
        args.backtests
        or args.buy_write_backtest
        or args.dashboard_html
    ) and (
        not args.previous_close
        and not args.last_year
        and not args.daily_files
        and args.start is None
        and args.end is None
        and not args.fundamentals
        and not args.fundamental_scores
        and not args.fundamental_score_history
        and not args.technical_scores
        and not args.technical_score_history
        and not args.scores_only
        and not args.score_history_only
        and not args.technical_score_history_only
        and not args.economy_history
        and not args.economy_history_only
    ):
        args.fundamentals_only = True
    if (
        args.technical_scores
        and not args.previous_close
        and not args.last_year
        and not args.daily_files
        and args.start is None
        and args.end is None
        and not args.fundamentals
        and not args.fundamental_scores
        and not args.fundamental_score_history
        and not args.scores_only
        and not args.score_history_only
        and not args.economy_history
        and not args.economy_history_only
    ):
        args.fundamentals_only = True

    if args.index_history and (args.start is None or args.end is None):
        if args.previous_close:
            args.end = default_end
            args.start = args.end - timedelta(days=max(args.fallback_days, 0))
        elif args.last_year:
            args.end = args.end or default_end
            args.start = args.end - timedelta(days=365)
        elif args.index_history_only:
            args.end = args.end or default_end
            args.start = args.start or (args.end - timedelta(days=365))

    if not args.fundamentals_only:
        args.start = args.start or default_start
        args.end = args.end or default_end
        result = pull_nifty_prices(
            start=args.start,
            end=args.end,
            output_dir=args.output_dir,
            symbols=args.symbols,
            limit=args.limit,
            daily_files=args.daily_files,
            latest_available_only=args.previous_close,
            fallback_days=max(args.fallback_days, 0),
        )
        if args.daily_files:
            print(f"Constituents: {len(result.constituents)} rows")
            print(f"Daily constituent files: {len(result.constituents_paths)}")
            print(f"Daily price files: {len(result.prices_paths)}")
            if result.constituents_path:
                print(f"Latest constituents file: {result.constituents_path}")
            if result.prices_path:
                print(f"Latest prices file: {result.prices_path}")
        else:
            print(f"Constituents: {len(result.constituents)} rows -> {result.constituents_path}")
            print(f"Prices: {len(result.prices)} rows -> {result.prices_path}")

    if args.index_history:
        args.start = args.start or default_start
        args.end = args.end or default_end
        index_result = pull_nifty_index_history(
            start=args.start,
            end=args.end,
            output_dir=args.output_dir,
            daily_files=True,
            latest_available_only=args.previous_close and not args.last_year,
            fallback_days=max(args.fallback_days, 0),
        )
        print(f"NIFTY index files: {len(index_result.index_paths)}")
        print(f"NIFTY index rows: {len(index_result.index_history)} -> {index_result.history_path}")

    if args.economy_history:
        economy = generate_economy_score_history(
            output_dir=args.output_dir,
            start=args.start,
            end=args.end,
        )
        print(f"Economic variables: {len(economy.scores)} aligned rows -> {economy.variables_path}")
        print(f"Economy score history: {len(economy.scores)} rows -> {economy.history_path}")

    fundamentals = None
    needs_fundamentals = (
        args.fundamentals
        or explicit_fundamentals_only
        or args.fundamental_scores
        or args.fundamental_score_history
        or args.scores_only
        or args.score_history_only
    )
    use_cached_fundamentals = (
        args.fundamentals_source == "cached"
        or (args.score_history_only and not args.fundamentals)
    )
    if needs_fundamentals and use_cached_fundamentals:
        fundamentals = load_latest_cached_fundamentals(args.output_dir)
        print(
            f"Cached financial results: {len(fundamentals.financial_results)} rows -> "
            f"{fundamentals.financial_results_path}"
        )
        print(
            f"Cached financial result announcements: {len(fundamentals.financial_announcements)} rows -> "
            f"{fundamentals.financial_announcements_path}"
        )
        print(
            f"Cached shareholding pattern: {len(fundamentals.shareholding)} rows -> "
            f"{fundamentals.shareholding_path}"
        )
    elif needs_fundamentals:
        if args.fundamentals_source == "screener":
            fundamentals = download_screener_fundamentals(
                output_dir=args.output_dir,
                symbols=args.symbols,
                limit=args.limit,
                lookback_years=max(args.fundamentals_years, 1),
            )
            print(
                f"Screener financial results: {len(fundamentals.financial_results)} rows -> "
                f"{fundamentals.financial_results_path}"
            )
            print(
                f"Screener financial result metadata: {len(fundamentals.financial_announcements)} rows -> "
                f"{fundamentals.financial_announcements_path}"
            )
            print(
                f"Screener shareholding pattern: {len(fundamentals.shareholding)} rows -> "
                f"{fundamentals.shareholding_path}"
            )
        elif args.fundamentals_source == "moneycontrol":
            fundamentals = download_moneycontrol_fundamentals(
                output_dir=args.output_dir,
                symbols=args.symbols,
                limit=args.limit,
                lookback_years=max(args.fundamentals_years, 1),
            )
            print(
                f"Moneycontrol financial results: {len(fundamentals.financial_results)} rows -> "
                f"{fundamentals.financial_results_path}"
            )
            print(
                f"Moneycontrol financial result metadata: {len(fundamentals.financial_announcements)} rows -> "
                f"{fundamentals.financial_announcements_path}"
            )
            print(
                f"Moneycontrol shareholding placeholder: {len(fundamentals.shareholding)} rows -> "
                f"{fundamentals.shareholding_path}"
            )
        elif args.fundamentals_source == "economictimes":
            fundamentals = download_economic_times_fundamentals(
                output_dir=args.output_dir,
                symbols=args.symbols,
                limit=args.limit,
                lookback_years=max(args.fundamentals_years, 1),
            )
            print(
                f"Economic Times financial results: {len(fundamentals.financial_results)} rows -> "
                f"{fundamentals.financial_results_path}"
            )
            print(
                f"Economic Times financial result metadata: {len(fundamentals.financial_announcements)} rows -> "
                f"{fundamentals.financial_announcements_path}"
            )
            print(
                f"Economic Times shareholding placeholder: {len(fundamentals.shareholding)} rows -> "
                f"{fundamentals.shareholding_path}"
            )
        else:
            try:
                fundamentals = download_screener_fundamentals(
                    output_dir=args.output_dir,
                    symbols=args.symbols,
                    limit=args.limit,
                    lookback_years=max(args.fundamentals_years, 1),
                )
                print(
                    f"Screener financial results: {len(fundamentals.financial_results)} rows -> "
                    f"{fundamentals.financial_results_path}"
                )
                print(
                    f"Screener financial result metadata: {len(fundamentals.financial_announcements)} rows -> "
                    f"{fundamentals.financial_announcements_path}"
                )
                print(
                    f"Screener shareholding pattern: {len(fundamentals.shareholding)} rows -> "
                    f"{fundamentals.shareholding_path}"
                )
            except Exception as exc:
                print(f"Screener fundamentals download failed: {exc}")
                print("Trying Moneycontrol fundamentals.")
                try:
                    fundamentals = download_moneycontrol_fundamentals(
                        output_dir=args.output_dir,
                        symbols=args.symbols,
                        limit=args.limit,
                        lookback_years=max(args.fundamentals_years, 1),
                    )
                    print(
                        f"Moneycontrol financial results: {len(fundamentals.financial_results)} rows -> "
                        f"{fundamentals.financial_results_path}"
                    )
                    print(
                        f"Moneycontrol financial result metadata: {len(fundamentals.financial_announcements)} rows -> "
                        f"{fundamentals.financial_announcements_path}"
                    )
                    print(
                        f"Moneycontrol shareholding placeholder: {len(fundamentals.shareholding)} rows -> "
                        f"{fundamentals.shareholding_path}"
                    )
                except Exception as mc_exc:
                    print(f"Moneycontrol fundamentals download failed: {mc_exc}")
                    print("Trying Economic Times fundamentals.")
                    try:
                        fundamentals = download_economic_times_fundamentals(
                            output_dir=args.output_dir,
                            symbols=args.symbols,
                            limit=args.limit,
                            lookback_years=max(args.fundamentals_years, 1),
                        )
                        print(
                            f"Economic Times financial results: {len(fundamentals.financial_results)} rows -> "
                            f"{fundamentals.financial_results_path}"
                        )
                        print(
                            f"Economic Times financial result metadata: {len(fundamentals.financial_announcements)} rows -> "
                            f"{fundamentals.financial_announcements_path}"
                        )
                        print(
                            f"Economic Times shareholding placeholder: {len(fundamentals.shareholding)} rows -> "
                            f"{fundamentals.shareholding_path}"
                        )
                    except Exception as et_exc:
                        print(f"Economic Times fundamentals download failed: {et_exc}")
                        print("Falling back to latest cached fundamentals.")
                        fundamentals = load_latest_cached_fundamentals(args.output_dir)
                        print(
                            f"Cached financial results: {len(fundamentals.financial_results)} rows -> "
                            f"{fundamentals.financial_results_path}"
                        )
                        print(
                            f"Cached financial result announcements: {len(fundamentals.financial_announcements)} rows -> "
                            f"{fundamentals.financial_announcements_path}"
                        )
                        print(
                            f"Cached shareholding pattern: {len(fundamentals.shareholding)} rows -> "
                            f"{fundamentals.shareholding_path}"
                        )

    if args.fundamental_scores:
        if fundamentals is None:
            raise RuntimeError("Fundamentals are required before fundamental scores can be generated.")
        constituents = load_latest_cached_constituents(args.output_dir)
        if constituents is None:
            constituents = get_nifty_constituents(NSEArchiveClient())
        score_symbols = selected_nifty_symbols(
            constituents,
            symbols=args.symbols,
            limit=args.limit,
        )
        scores = generate_fundamental_scores(
            financial_results=fundamentals.financial_results,
            shareholding=fundamentals.shareholding,
            financial_announcements=fundamentals.financial_announcements,
            output_dir=args.output_dir,
            symbols=score_symbols,
        )
        changed_count = int(scores.scores["score_changed"].fillna(False).sum()) if "score_changed" in scores.scores else 0
        print(f"Fundamental scores: {len(scores.scores)} rows -> {scores.scores_path}")
        print(f"Fundamental score changes: {changed_count}")

    should_generate_fundamental_history = args.fundamental_score_history or (
        args.backtests and args.fundamental_scores
    )
    if should_generate_fundamental_history:
        if fundamentals is None:
            raise RuntimeError("Fundamentals are required before fundamental score history can be generated.")
        history_start = None if args.backtests else args.start
        history_end = None if args.backtests else args.end
        history = generate_fundamental_score_history(
            financial_results=fundamentals.financial_results,
            shareholding=fundamentals.shareholding,
            financial_announcements=fundamentals.financial_announcements,
            output_dir=args.output_dir,
            symbols=args.symbols,
            start=history_start,
            end=history_end,
        )
        changed_count = (
            int(history.scores["score_changed"].fillna(False).sum())
            if "score_changed" in history.scores
            else 0
        )
        print(f"Historical fundamental score files: {len(history.score_paths)}")
        print(f"Historical fundamental score rows: {len(history.scores)} -> {history.history_path}")
        print(f"Historical fundamental score changes: {changed_count}")

    if args.technical_score_history:
        history = generate_technical_score_history(
            output_dir=args.output_dir,
            symbols=args.symbols,
            start=args.start,
            end=args.end,
        )
        print(f"Historical technical score files: {len(history.score_paths)}")
        print(f"Historical technical score rows: {len(history.scores)} -> {history.history_path}")

    if args.technical_scores:
        history = generate_technical_score_history(
            output_dir=args.output_dir,
            symbols=args.symbols,
        )
        print(f"Technical score files: {len(history.score_paths)}")
        print(f"Technical score rows: {len(history.scores)} -> {history.history_path}")
        if history.score_paths:
            print(f"Latest technical score file: {history.score_paths[-1]}")

    if args.backtests:
        backtests = generate_strategy_backtests(
            output_dir=args.output_dir,
        )
        print(f"Strategy backtest files: {len(backtests.backtest_paths)}")
        for path in backtests.backtest_paths:
            print(f"Backtest: {path}")
        print(f"Strategy holdings files: {len(backtests.holdings_paths)}")
        for path in backtests.holdings_paths:
            print(f"Holdings: {path}")
        print(f"Selected strategy metrics: {backtests.selected_metrics_path}")
        buy_write_path, buy_write_trades_path, buy_write = generate_nifty_buy_write_backtest(
            output_dir=args.output_dir,
        )
        print(f"NIFTY buy-write backtest: {len(buy_write)} trades -> {buy_write_path}")
        print(f"NIFTY buy-write trades: {buy_write_trades_path}")

    if args.buy_write_backtest and not args.backtests:
        buy_write_path, buy_write_trades_path, buy_write = generate_nifty_buy_write_backtest(
            output_dir=args.output_dir,
        )
        print(f"NIFTY buy-write backtest: {len(buy_write)} trades -> {buy_write_path}")
        print(f"NIFTY buy-write trades: {buy_write_trades_path}")

    if args.dashboard_html:
        latest_path, dated_path = generate_dashboard_html(
            output_dir=args.output_dir,
        )
        print(f"Latest dashboard HTML: {latest_path}")
        print(f"Datestamped dashboard HTML: {dated_path}")


if __name__ == "__main__":
    main()
