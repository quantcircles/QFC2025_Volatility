import io
import os
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from mcp.server.fastmcp import FastMCP


CACHE_DIR = "data_cache"
REPORTS_DIR = "reports"

mcp = FastMCP("qfc-daily-summary-agent")


def _ensure_dirs() -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)


def _parse_date(value: Optional[str]) -> date:
    if not value:
        return date.today()
    return datetime.strptime(value, "%Y-%m-%d").date()


def _get_yf_symbol(symbol: str) -> str:
    if symbol == "NIFTY":
        return "^NSEI"
    if symbol == "BANKNIFTY":
        return "^NSEBANK"
    return f"{symbol}.NS"


def _extract_close_series(data: pd.DataFrame, ticker: str) -> pd.Series:
    if data is None or data.empty:
        return pd.Series(dtype=float)

    if isinstance(data.columns, pd.MultiIndex):
        adj_key = ("Adj Close", ticker)
        close_key = ("Close", ticker)
        if adj_key in data.columns:
            s = data[adj_key]
        elif close_key in data.columns:
            s = data[close_key]
        else:
            close_like = [c for c in data.columns if c[0] in ("Adj Close", "Close")]
            if not close_like:
                return pd.Series(dtype=float)
            s = data[close_like[0]]
    else:
        if "Adj Close" in data.columns:
            s = data["Adj Close"]
        elif "Close" in data.columns:
            s = data["Close"]
        else:
            return pd.Series(dtype=float)

    return pd.to_numeric(s, errors="coerce").dropna()


def _get_india_vix(target_date: date, d: int = 365) -> pd.Series:
    start_dt = pd.to_datetime(target_date) - timedelta(days=d)
    data = yf.download("^INDIAVIX", start=start_dt, end=target_date + timedelta(days=1), progress=False)
    return _extract_close_series(data, "^INDIAVIX").rename("Implied Vol")


def _get_price_series(ticker_symbol: str, target_date: date, d: int = 365) -> pd.Series:
    start_dt = pd.to_datetime(target_date) - timedelta(days=d)
    data = yf.download(ticker_symbol, start=start_dt, end=target_date + timedelta(days=1), progress=False)
    return _extract_close_series(data, ticker_symbol).rename("Price")


def _get_realized_vol_series(price_series: pd.Series, window: int = 30) -> pd.Series:
    if price_series is None or price_series.empty:
        return pd.Series(dtype=float)
    log_returns = np.log(price_series / price_series.shift(1)).dropna()
    rv = log_returns.rolling(window).std() * np.sqrt(252) * 100
    return rv.dropna().rename("Realized Vol")


def _run_long_only_vrp_strategy(
    vol_df: pd.DataFrame,
    price_series: pd.Series,
    lookback: int = 30,
    entry_z: float = 1.0,
    exit_z: float = 0.25,
    cost_bps: float = 2.0,
) -> pd.DataFrame:
    data = pd.concat([vol_df[["Implied Vol", "Realized Vol"]], price_series.rename("Price")], axis=1).dropna()
    if data.empty:
        return pd.DataFrame()

    data["VRP"] = data["Implied Vol"] - data["Realized Vol"]
    vrp_mean = data["VRP"].rolling(lookback).mean()
    vrp_std = data["VRP"].rolling(lookback).std(ddof=0).replace(0, np.nan)
    data["VRP_Z"] = (data["VRP"] - vrp_mean) / vrp_std

    pos = []
    current = 0
    for z in data["VRP_Z"].fillna(0.0):
        if z < -entry_z:
            current = 1
        elif z > -exit_z:
            current = 0
        pos.append(current)
    data["Position"] = pos
    data["Underlying_Return"] = data["Price"].pct_change().fillna(0.0)
    data["Turnover"] = data["Position"].diff().abs().fillna(abs(data["Position"]))
    data["Cost"] = data["Turnover"] * (cost_bps / 10000.0)
    data["Strategy_Return"] = data["Position"].shift(1).fillna(0.0) * data["Underlying_Return"] - data["Cost"]
    data["Equity"] = (1.0 + data["Strategy_Return"]).cumprod()
    data["CumPnL"] = data["Equity"] - 1.0
    return data.dropna(subset=["VRP_Z"])


def _summarize_performance(returns: pd.Series) -> Dict[str, float]:
    if returns is None or returns.empty:
        return {"annual_return": np.nan, "annual_vol": np.nan, "sharpe": np.nan, "max_drawdown": np.nan}

    ann_ret = ((1 + returns).prod() ** (252 / max(len(returns), 1)) - 1) * 100
    ann_vol = returns.std() * np.sqrt(252) * 100
    sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else np.nan
    eq = (1 + returns).cumprod()
    dd = ((eq / eq.cummax()) - 1).min() * 100
    return {"annual_return": ann_ret, "annual_vol": ann_vol, "sharpe": sharpe, "max_drawdown": dd}


def _optimize_params(vol_df: pd.DataFrame, price_series: pd.Series, cost_bps: float = 2.0) -> Optional[Dict[str, float]]:
    lookback_grid = [20, 30, 40, 60]
    entry_grid = [0.8, 1.0, 1.2, 1.5]
    exit_grid = [0.1, 0.2, 0.3, 0.4]
    best = None

    for lb in lookback_grid:
        for entry in entry_grid:
            for exit_ in exit_grid:
                if exit_ >= entry:
                    continue
                strat = _run_long_only_vrp_strategy(vol_df, price_series, lb, entry, exit_, cost_bps)
                if strat.empty:
                    continue
                stats = _summarize_performance(strat["Strategy_Return"])
                if np.isnan(stats["annual_return"]):
                    continue
                if best is None or stats["annual_return"] > best["annual_return"]:
                    best = {
                        "lookback": lb,
                        "entry_z": entry,
                        "exit_z": exit_,
                        **stats,
                        "trades": int((strat["Turnover"] > 0).sum()),
                        "returns": strat["Strategy_Return"],
                    }
    return best


def _get_nse_symbols_by_oi(target_date: date, max_symbols: int = 40) -> List[str]:
    _ensure_dirs()
    date_str = target_date.strftime("%Y%m%d")
    cache_path = os.path.join(CACHE_DIR, f"fo_{date_str}.csv")

    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path)
    else:
        url = f"https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{date_str}_F_0000.csv.zip"
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://www.nseindia.com/",
        }
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=headers, timeout=5)
        resp = session.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        import zipfile

        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            name = zf.namelist()[0]
            with zf.open(name) as f:
                df = pd.read_csv(f)
        df.columns = [c.strip() for c in df.columns]
        cols = {
            "TckrSymb": "SYMBOL",
            "OptnTp": "TYPE",
            "OpnIntrst": "OI",
        }
        df = df.rename(columns=cols)
        df.to_csv(cache_path, index=False)

    if "SYMBOL" not in df.columns:
        return []
    if "OI" not in df.columns:
        return sorted(df["SYMBOL"].dropna().astype(str).unique().tolist())[:max_symbols]

    df["OI"] = pd.to_numeric(df["OI"], errors="coerce").fillna(0.0)
    oi_by_symbol = df.groupby("SYMBOL", as_index=False)["OI"].sum().sort_values("OI", ascending=False)
    return oi_by_symbol["SYMBOL"].astype(str).head(max_symbols).tolist()


def _generate_daily_summary(trading_date: date, max_symbols: int = 40, top_n: int = 10, cost_bps: float = 2.0) -> str:
    symbols = _get_nse_symbols_by_oi(trading_date, max_symbols=max_symbols)
    if not symbols:
        return f"# Daily Summary ({trading_date})\n\nNo symbols available."

    implied = _get_india_vix(trading_date)
    if implied.empty:
        return f"# Daily Summary ({trading_date})\n\nNo implied-volatility series (INDIAVIX) available."

    rows = []
    return_series = {}
    for symbol in symbols:
        yf_symbol = _get_yf_symbol(symbol)
        price = _get_price_series(yf_symbol, trading_date)
        if price.empty:
            continue
        rv = _get_realized_vol_series(price, window=30)
        vol_df = pd.concat([implied, rv], axis=1).dropna()
        if vol_df.empty:
            continue

        best = _optimize_params(vol_df, price, cost_bps=cost_bps)
        if not best:
            continue

        rows.append(
            {
                "Symbol": symbol,
                "Annual Return (%)": best["annual_return"],
                "Annual Risk (%)": best["annual_vol"],
                "Sharpe": best["sharpe"],
                "Max Drawdown (%)": best["max_drawdown"],
                "Lookback": best["lookback"],
                "Entry Z": best["entry_z"],
                "Exit Z": best["exit_z"],
                "Trades": best["trades"],
            }
        )
        return_series[symbol] = best["returns"]

    if not rows:
        return f"# Daily Summary ({trading_date})\n\nNo valid optimized strategies found."

    rank_df = pd.DataFrame(rows).sort_values("Annual Return (%)", ascending=False).reset_index(drop=True)
    top_df = rank_df.head(top_n).copy()

    # equal-weight portfolio from top-N optimized return series
    selected = [s for s in top_df["Symbol"].tolist() if s in return_series]
    port = pd.concat({s: return_series[s] for s in selected}, axis=1).dropna()
    portfolio_ret = port.mean(axis=1) if not port.empty else pd.Series(dtype=float)
    portfolio_stats = _summarize_performance(portfolio_ret)

    lines = []
    lines.append(f"# Daily Volatility Strategy Summary ({trading_date})")
    lines.append("")
    lines.append(f"- Universe scanned: {len(symbols)} symbols (OI-filtered)")
    lines.append(f"- Valid optimized symbols: {len(rank_df)}")
    lines.append(f"- Portfolio construction: equal-weight top {min(top_n, len(top_df))} by annual return")
    lines.append("")
    lines.append("## Top Symbols")
    lines.append("| Rank | Symbol | Return % | Risk % | Sharpe | Max DD % | Entry Z | Exit Z | Lookback | Trades |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for i, r in top_df.iterrows():
        lines.append(
            f"| {i+1} | {r['Symbol']} | {r['Annual Return (%)']:.2f} | {r['Annual Risk (%)']:.2f} | "
            f"{r['Sharpe']:.2f} | {r['Max Drawdown (%)']:.2f} | {r['Entry Z']:.2f} | "
            f"{r['Exit Z']:.2f} | {int(r['Lookback'])} | {int(r['Trades'])} |"
        )
    lines.append("")
    lines.append("## Top-N Portfolio Stats")
    lines.append(f"- Annual Return (%): {portfolio_stats['annual_return']:.2f}")
    lines.append(f"- Annual Risk (%): {portfolio_stats['annual_vol']:.2f}")
    lines.append(f"- Sharpe: {portfolio_stats['sharpe']:.2f}")
    lines.append(f"- Max Drawdown (%): {portfolio_stats['max_drawdown']:.2f}")
    lines.append("")
    lines.append(f"_Generated at {datetime.now().isoformat(timespec='seconds')}_")

    report = "\n".join(lines)
    _ensure_dirs()
    report_path = os.path.join(REPORTS_DIR, f"daily_summary_{trading_date}.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    with open(os.path.join(REPORTS_DIR, "daily_summary_latest.md"), "w", encoding="utf-8") as f:
        f.write(report)

    return report


@mcp.tool()
def generate_daily_summary(
    trading_date: Optional[str] = None,
    max_symbols: int = 40,
    top_n: int = 10,
    cost_bps: float = 2.0,
) -> str:
    """
    Generate daily strategy summary and save report to /reports.
    trading_date format: YYYY-MM-DD
    """
    dt = _parse_date(trading_date)
    return _generate_daily_summary(dt, max_symbols=max_symbols, top_n=top_n, cost_bps=cost_bps)


@mcp.resource("summary://latest")
def latest_summary_resource() -> str:
    _ensure_dirs()
    latest_path = os.path.join(REPORTS_DIR, "daily_summary_latest.md")
    if not os.path.exists(latest_path):
        return "No summary generated yet. Call tool generate_daily_summary first."
    with open(latest_path, "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    mcp.run()
