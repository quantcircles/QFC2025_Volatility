#!/usr/bin/env python3
"""Pull NIFTY 50 prices, constituents, and NSE corporate filings."""

from __future__ import annotations

import argparse
import io
import json
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable
import zipfile

import numpy as np
import pandas as pd
import requests
from scipy.optimize import minimize


NSE_HOME = "https://www.nseindia.com"
NSE_ARCHIVES = "https://archives.nseindia.com"
NSE_SEARCHIVES = "https://nsearchives.nseindia.com"
NIFTY_50_INDEX = "NIFTY 50"


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


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def nse_display_date(value: date) -> str:
    return value.strftime("%d-%m-%Y")


def parse_date_series(values: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(values, errors="coerce", dayfirst=True, format="mixed")
    except TypeError:
        return pd.to_datetime(values, errors="coerce", dayfirst=True)


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


def latest_previous_score_file(scores_dir: Path, score_date: date) -> Path | None:
    if not scores_dir.exists():
        return None
    current_name = f"nse_fundamental_scores_{score_date.strftime('%Y%m%d')}.csv"
    candidates = sorted(
        path for path in scores_dir.glob("nse_fundamental_scores_*.csv")
        if path.name < current_name
    )
    return candidates[-1] if candidates else None


def latest_csv_file(directory: Path, pattern: str) -> Path | None:
    if not directory.exists():
        return None
    candidates = sorted(directory.glob(pattern))
    return candidates[-1] if candidates else None


def load_latest_cached_fundamentals(output_dir: Path) -> FundamentalsResult:
    fundamentals_dir = output_dir / "fundamentals"
    financial_path = latest_csv_file(
        fundamentals_dir / "financial_results_by_day",
        "nse_financial_results_*.csv",
    )
    announcements_path = latest_csv_file(
        fundamentals_dir / "financial_result_announcements_by_day",
        "nse_financial_result_announcements_*.csv",
    )
    shareholding_path = latest_csv_file(
        fundamentals_dir / "shareholding_by_day",
        "nse_shareholding_pattern_*.csv",
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
    latest_promoter_holding = pd.NA
    latest_public_holding = pd.NA

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

    computed_score = (
        quarterly_recency_score
        + annual_recency_score
        + disclosure_score
        + filing_consistency_score
        + shareholding_recency_score
        + promoter_score
    )

    return {
        "symbol": symbol,
        "computed_score": int(round(computed_score)),
        "quarterly_recency_score": quarterly_recency_score,
        "annual_recency_score": annual_recency_score,
        "disclosure_score": disclosure_score,
        "filing_consistency_score": filing_consistency_score,
        "shareholding_recency_score": shareholding_recency_score,
        "promoter_score": promoter_score,
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
        "quarterly_recency_score",
        "annual_recency_score",
        "disclosure_score",
        "filing_consistency_score",
        "shareholding_recency_score",
        "promoter_score",
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
        "previous_score_file",
    ]
    scores = scores[[col for col in ordered_cols if col in scores.columns]]

    scores_path = scores_dir / f"nse_fundamental_scores_{score_on.strftime('%Y%m%d')}.csv"
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

    all_scores = []
    score_paths = []
    for score_date in score_dates:
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
    history_dir = output_dir / "fundamentals" / "fundamental_scores_history"
    history_dir.mkdir(parents=True, exist_ok=True)
    history_path = history_dir / (
        f"nse_fundamental_scores_history_{score_dates[0].strftime('%Y%m%d')}_"
        f"{score_dates[-1].strftime('%Y%m%d')}.csv"
    )
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
    for score_date, day in scores.groupby("score_date", sort=True):
        score_path = scores_dir / f"nse_technical_scores_{score_date.replace('-', '')}.csv"
        day.to_csv(score_path, index=False)
        score_paths.append(score_path)

    history_dir = output_dir / "technicals" / "technical_scores_history"
    history_dir.mkdir(parents=True, exist_ok=True)
    date_values = sorted(scores["score_date"].unique().tolist())
    history_path = history_dir / (
        f"nse_technical_scores_history_{date_values[0].replace('-', '')}_"
        f"{date_values[-1].replace('-', '')}.csv"
    )
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


def add_benchmark_columns(backtest: pd.DataFrame, prices: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
    pivot = prices.pivot_table(index="score_date", columns="symbol", values="close", aggfunc="last").sort_index()
    benchmark_return = pivot.pct_change(fill_method=None).mean(axis=1, skipna=True).fillna(0.0)
    benchmark = pd.DataFrame(
        {
            "date": benchmark_return.index.strftime("%Y-%m-%d"),
            "benchmark_daily_return": benchmark_return.to_numpy(),
            "benchmark_cumulative_return": (1 + benchmark_return).cumprod().to_numpy() - 1,
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


def generate_strategy_backtests(
    output_dir: Path = Path("data_cache/nse_equity"),
    initial_capital: float = 100000.0,
    transaction_cost: float = 0.0001,
    holding_days: int = 30,
    top_n: int = 10,
    mvo_lookback: int = 126,
) -> BacktestResult:
    backtest_dir = output_dir / "backtests"
    backtest_dir.mkdir(parents=True, exist_ok=True)
    fundamental_path = latest_required_csv(
        output_dir / "fundamentals" / "fundamental_scores_history",
        "nse_fundamental_scores_history_*.csv",
        "fundamental score history",
    )
    technical_path = latest_required_csv(
        output_dir / "technicals" / "technical_scores_history",
        "nse_technical_scores_history_*.csv",
        "technical score history",
    )
    fundamental = pd.read_csv(fundamental_path, usecols=["score_date", "symbol", "fundamental_score"])
    technical = pd.read_csv(technical_path, usecols=["score_date", "symbol", "technical_score", "close"])
    for frame in (fundamental, technical):
        frame["score_date"] = pd.to_datetime(frame["score_date"])
        frame["symbol"] = frame["symbol"].astype(str).str.upper()
    data = technical.merge(fundamental, on=["score_date", "symbol"], how="inner")
    for column in ["fundamental_score", "technical_score", "close"]:
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
        backtest = add_benchmark_columns(pd.DataFrame(rows), data, initial_capital)
        backtest_path = backtest_dir / f"{strategy}_top10_score_weighted_backtest.csv"
        backtest.to_csv(backtest_path, index=False)
        backtest_paths.append(backtest_path)

        blocks = []
        for block_number, start_idx in enumerate(range(0, len(dates), holding_days), start=1):
            start_dt = dates[start_idx]
            end_dt = dates[min(start_idx + holding_days - 1, len(dates) - 1)]
            snap = selected_data[selected_data["score_date"].eq(start_dt)].copy()
            snap["score"] = snap[score_column].clip(lower=0)
            total_score = snap["score"].sum()
            snap["weight"] = snap["score"] / total_score if total_score else 1 / len(snap)
            blocks.append(
                {
                    "start_date": pd.Timestamp(start_dt).strftime("%Y-%m-%d"),
                    "end_date": pd.Timestamp(end_dt).strftime("%Y-%m-%d"),
                    "holdings": snap.sort_values("weight", ascending=False)[["symbol", "score", "weight"]].to_dict("records"),
                }
            )
        holdings_paths.append(build_rebalance_holdings(strategy, blocks, data, backtest, backtest_dir))

    prices = data.pivot_table(index="score_date", columns="symbol", values="close", aggfunc="last").sort_index()
    returns = prices.pct_change(fill_method=None).fillna(0.0)
    average_scores = data.pivot_table(index="score_date", columns="symbol", values="average_score", aggfunc="last").sort_index()
    dates = list(prices.index)
    rebalance_dates = dates[::holding_days]
    equity = initial_capital
    previous_weights = {symbol: 0.0 for symbol in prices.columns}
    rows = []
    blocks = []
    last_top_symbols: list[str] = []
    for block_number, start_dt in enumerate(rebalance_dates, start=1):
        start_idx = dates.index(start_dt)
        end_dt = dates[min(start_idx + holding_days - 1, len(dates) - 1)]
        if start_idx < 20:
            top_symbols = average_scores.loc[start_dt].dropna().sort_values(ascending=False).head(top_n).index.tolist()
            history = None
        else:
            history = returns.iloc[max(0, start_idx - mvo_lookback):start_idx]
            sharpe = (history.mean() * 252) / (history.std(ddof=1) * np.sqrt(252)).replace(0, np.nan)
            top_symbols = sharpe.replace([np.inf, -np.inf], np.nan).dropna().sort_values(ascending=False).head(top_n).index.tolist()
            if len(top_symbols) < top_n:
                top_symbols = average_scores.loc[start_dt].dropna().sort_values(ascending=False).head(top_n).index.tolist()
                history = None
        weights = optimize_mvo_weights(history[top_symbols]) if history is not None else np.repeat(1 / len(top_symbols), len(top_symbols))
        current_weights = {symbol: 0.0 for symbol in prices.columns}
        for symbol, weight in zip(top_symbols, weights):
            current_weights[symbol] = float(weight)
        turnover = sum(abs(current_weights[symbol] - previous_weights.get(symbol, 0.0)) for symbol in prices.columns)
        blocks.append(
            {
                "start_date": pd.Timestamp(start_dt).strftime("%Y-%m-%d"),
                "end_date": pd.Timestamp(end_dt).strftime("%Y-%m-%d"),
                "holdings": [
                    {
                        "symbol": symbol,
                        "score": float(average_scores.loc[start_dt, symbol]) if pd.notna(average_scores.loc[start_dt, symbol]) else np.nan,
                        "weight": current_weights[symbol],
                    }
                    for symbol in top_symbols
                ],
            }
        )
        for day_offset, score_date in enumerate(dates[start_idx:dates.index(end_dt) + 1]):
            daily_return = sum(current_weights[symbol] * float(returns.loc[score_date, symbol]) for symbol in top_symbols)
            if day_offset == 0:
                daily_return -= transaction_cost * turnover
            equity *= 1 + daily_return
            rows.append(
                {
                    "date": pd.Timestamp(score_date).strftime("%Y-%m-%d"),
                    "daily_return": daily_return,
                    "cumulative_return": equity / initial_capital - 1,
                    "portfolio_value": equity,
                }
            )
        previous_weights = current_weights
        last_top_symbols = top_symbols
    mvo_backtest = add_benchmark_columns(pd.DataFrame(rows).drop_duplicates("date", keep="last"), data, initial_capital)
    mvo_backtest_path = backtest_dir / "mvo_top10_score_weighted_backtest.csv"
    mvo_backtest.to_csv(mvo_backtest_path, index=False)
    backtest_paths.append(mvo_backtest_path)
    holdings_paths.append(build_rebalance_holdings("mvo", blocks, data, mvo_backtest, backtest_dir))

    mvo_symbol_metrics = []
    for symbol in returns.columns:
        metrics = backtest_metrics(returns[symbol])
        metrics.update({"symbol": symbol, "score_type": "mvo"})
        mvo_symbol_metrics.append(metrics)
    mvo_symbol_metrics_df = pd.DataFrame(mvo_symbol_metrics).sort_values("sharpe", ascending=False)
    mvo_metrics_path = backtest_dir / "mvo_score_weighted_symbol_metrics.csv"
    mvo_symbol_metrics_df.to_csv(mvo_metrics_path, index=False)
    symbol_metrics_paths.append(mvo_metrics_path)
    selected_metrics_frames.append(mvo_symbol_metrics_df[mvo_symbol_metrics_df["symbol"].isin(last_top_symbols)].head(top_n))

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
    fundamental_path = latest_required_csv(
        output_dir / "fundamentals" / "fundamental_scores_history",
        "nse_fundamental_scores_history_*.csv",
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
        "mvo": "Mean Variance",
    }
    selected_path = backtest_dir / "selected_top10_score_weighted_symbol_metrics.csv"
    selected = pd.read_csv(selected_path) if selected_path.exists() else pd.DataFrame()
    payload: dict[str, dict] = {}
    benchmark_series = None
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
        payload["benchmark"] = {
            "label": "Equal-weight NIFTY 50 proxy",
            "series": benchmark_series,
            "metrics": {key: json_clean(value) for key, value in backtest_metrics(pd.Series(benchmark_series["dailyReturn"])).items()},
        }
    return payload


def generate_dashboard_html(
    output_dir: Path = Path("data_cache/nse_equity"),
    template_path: Path = Path("fundamental_score_dashboard.html"),
    dashboards_dir: Path = Path("dashboards"),
) -> tuple[Path, Path]:
    if not template_path.exists():
        raise RuntimeError(f"Dashboard template not found: {template_path}")
    dashboard_data = build_dashboard_data(output_dir)
    backtest_data = build_backtest_dashboard_data(output_dir)
    html = template_path.read_text()
    replacements = {
        "dashboard-data": json.dumps(dashboard_data, separators=(",", ":")),
        "backtest-data": json.dumps(backtest_data, separators=(",", ":")),
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
        "--fallback-days",
        type=int,
        default=10,
        help="Lookback window used by --previous-close to skip holidays/weekends",
    )
    parser.add_argument(
        "--fundamentals",
        action="store_true",
        help="Also download NSE Financial Results and Shareholding Pattern CSVs",
    )
    parser.add_argument(
        "--fundamentals-only",
        action="store_true",
        help="Download only NSE Financial Results and Shareholding Pattern CSVs",
    )
    parser.add_argument(
        "--fundamentals-years",
        type=int,
        default=8,
        help="Financial Results lookback window in years for quarterly and annual filings",
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
        "--dashboard-html",
        action="store_true",
        help="Generate latest and datestamped standalone HTML dashboard from cached histories and backtests",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
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
    if (
        args.backtests
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
    ):
        args.fundamentals_only = True

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

    fundamentals = None
    needs_fundamentals = (
        args.fundamentals
        or args.fundamental_scores
        or args.fundamental_score_history
        or args.scores_only
        or args.score_history_only
    )
    use_cached_fundamentals = args.score_history_only and not args.fundamentals
    if use_cached_fundamentals:
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
        try:
            fundamentals = download_nse_fundamentals(
                output_dir=args.output_dir,
                symbols=args.symbols,
                limit=args.limit,
                lookback_years=max(args.fundamentals_years, 1),
            )
            print(
                f"Financial results: {len(fundamentals.financial_results)} rows -> "
                f"{fundamentals.financial_results_path}"
            )
            print(
                f"Financial result announcements: {len(fundamentals.financial_announcements)} rows -> "
                f"{fundamentals.financial_announcements_path}"
            )
            print(
                f"Shareholding pattern: {len(fundamentals.shareholding)} rows -> "
                f"{fundamentals.shareholding_path}"
            )
        except Exception as exc:
            print(f"Live NSE fundamentals download failed: {exc}")
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

    if args.fundamental_score_history:
        if fundamentals is None:
            raise RuntimeError("Fundamentals are required before fundamental score history can be generated.")
        history = generate_fundamental_score_history(
            financial_results=fundamentals.financial_results,
            shareholding=fundamentals.shareholding,
            financial_announcements=fundamentals.financial_announcements,
            output_dir=args.output_dir,
            symbols=args.symbols,
            start=args.start,
            end=args.end,
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

    if args.dashboard_html:
        latest_path, dated_path = generate_dashboard_html(
            output_dir=args.output_dir,
        )
        print(f"Latest dashboard HTML: {latest_path}")
        print(f"Datestamped dashboard HTML: {dated_path}")


if __name__ == "__main__":
    main()
