#!/usr/bin/env python3
"""Pull NIFTY 50 constituents and cash-market prices from NSE archives."""

from __future__ import annotations

import argparse
import io
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable
import zipfile

import pandas as pd
import requests


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


class NSEArchiveClient:
    """Small NSE archive client for CSV and zipped bhavcopy downloads."""

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


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


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

    if symbols:
        wanted = {symbol.upper() for symbol in symbols}
        pull_symbols = [symbol for symbol in constituents["symbol"].astype(str) if symbol.upper() in wanted]
    else:
        pull_symbols = constituents["symbol"].astype(str).tolist()

    if limit is not None:
        pull_symbols = pull_symbols[:limit]

    if not pull_symbols:
        raise RuntimeError("No symbols selected for NSE history pull.")

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
    default_end = date.today() - timedelta(days=1)
    default_start = default_end - timedelta(days=30)
    parser.add_argument("--from", dest="start", type=parse_date, default=default_start)
    parser.add_argument("--to", dest="end", type=parse_date, default=default_end)
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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.previous_close:
        args.end = date.today() - timedelta(days=1)
        args.start = args.end - timedelta(days=max(args.fallback_days, 0))
        args.daily_files = True
    elif args.last_year:
        args.start = args.end - timedelta(days=365)

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


if __name__ == "__main__":
    main()
