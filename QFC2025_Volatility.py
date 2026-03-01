import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import zipfile
import io
import os
from datetime import datetime
from scipy.optimize import brentq
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import plotly.express as px          # For quick scaffolding
import plotly.graph_objects as go    # For complex layering (PDF, SVI)
try:
    import polars as pl
    POLARS_AVAILABLE = True
except Exception:
    pl = None
    POLARS_AVAILABLE = False

import yfinance as yf

from datetime import timedelta
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("vol_hub.log"), # Saves to a file
        logging.StreamHandler()             # Prints to your Spyder/Terminal console
    ]
)
logger = logging.getLogger(__name__)


def integrate_trapezoid(y, x):
    """Compatibility helper for NumPy versions where trapz may be unavailable."""
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)
    return np.trapz(y, x)


def get_yf_symbol(symbol):
    """Map NSE derivative symbols to yfinance tickers."""
    index_map = {
        "NIFTY": "^NSEI",
        "BANKNIFTY": "^NSEBANK",
    }
    if symbol in index_map:
        return index_map[symbol]
    return f"{symbol}.NS"


def extract_close_series(data, ticker):
    """Handle both single-level and MultiIndex yfinance outputs."""
    if data is None or data.empty:
        return pd.Series(dtype=float)

    if isinstance(data.columns, pd.MultiIndex):
        adj_key = ('Adj Close', ticker)
        close_key = ('Close', ticker)
        if adj_key in data.columns:
            series = data[adj_key]
        elif close_key in data.columns:
            series = data[close_key]
        else:
            close_like = [col for col in data.columns if col[0] in ('Adj Close', 'Close')]
            if not close_like:
                return pd.Series(dtype=float)
            series = data[close_like[0]]
    else:
        if 'Adj Close' in data.columns:
            series = data['Adj Close']
        elif 'Close' in data.columns:
            series = data['Close']
        else:
            return pd.Series(dtype=float)

    return pd.to_numeric(series, errors='coerce').dropna()


def load_nse_csv_to_pandas(path_or_buffer):
    """Use Polars for faster CSV parsing when available."""
    if POLARS_AVAILABLE:
        return pl.read_csv(path_or_buffer).to_pandas()
    return pd.read_csv(path_or_buffer)


def to_polars_if_available(df):
    if not POLARS_AVAILABLE or df is None:
        return None
    return pl.from_pandas(df)

# --- PAGE SETUP ---
st.set_page_config(page_title="Nifty Maturity Hub", layout="wide")
CACHE_DIR = "data_cache"
if not os.path.exists(CACHE_DIR) : os.makedirs(CACHE_DIR)
    
# --- INITIALIZE SESSION STATE ---
# This prevents the AttributeError
if 'active_df' not in st.session_state: st.session_state.active_df = None
if 'active_df_pl' not in st.session_state: st.session_state.active_df_pl = None


# --- BLACK-SCHOLES ENGINE ---
def black_scholes(F, K, T, sigma, option_type='CE'):
    d1 = (np.log(F / K) + (0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'CE':
        return (F * norm.cdf(d1) - K * norm.cdf(d2)) * np.exp(-0.05 * T) # Disount factor is negligible for IV shape
    else:
        return (K * norm.cdf(-d2) - F * norm.cdf(-d1)) * np.exp(-0.05 * T)

def find_iv(market_price, F, K, T, option_type):
    if market_price <= 0.5: return 0.0
    
    def objective(sigma):
        return black_scholes(F, K, T, sigma, option_type) - market_price

    try:
        # Solving for sigma between 0.1% and 500%
        return brentq(objective, 0.001, 5.0, xtol=1e-5)
    except:
        return np.nan
    #except (ValueError, RuntimeError):
        # If the price is mathematically impossible (e.g. below intrinsic value)
        #return 0.0
def get_tte(trade_date_str, expiry_date_str):
    t = datetime.strptime(trade_date_str, '%Y-%m-%d')
    e = datetime.strptime(expiry_date_str, '%Y-%m-%d')
    days = (e - t).days
    return max(days, 1) / 365.0  # Time in years




def svi_formula(params, k):
    """Raw SVI Parameterization"""
    a, b, rho, m, sigma = params
    return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))


def svi_objective(params, k, w_market):
    """Objective function: Mean Squared Error"""
    # Constraints: b >= 0, |rho| <= 1, sigma > 0, a + b*sigma*sqrt(1-rho^2) >= 0
    a, b, rho, m, sigma = params
    if b < 0 or abs(rho) > 1 or sigma <= 0:
        return 1e10
    
    w_model = svi_formula(params, k)
    return np.sum((w_model - w_market)**2)

def fit_svi(smile_df, fwd_price, tte):
    # 1. Prepare data for SVI (Log-Moneyness and Total Variance)
    # k = log(K/F), w = IV^2 * T
    smile_df['k'] = np.log(smile_df['STRIKE'] / fwd_price)
    smile_df['w_market'] = (smile_df['IV_pct'] / 100)**2 * tte
    
    k_arr = smile_df['k'].values
    w_arr = smile_df['w_market'].values

    # 2. Initial Guess [a, b, rho, m, sigma]
    # a: vertical shift, b: slope, rho: rotation, m: horizontal shift, sigma: curvature
    # Updated Bounds to prevent 'Degenerate' fits
    bounds = [
    (1e-5, 0.5),      # a > 0
    (1e-3, 0.5),      # b > 0 (forces wings to exist)
    (-0.9, 0.9),      # rho (keeps it from becoming a straight line)
    (-0.5, 0.5),      # m
    (0.01, 0.2)       # sigma (forces a rounded bottom)
        ]
    
    # Better Initial Guess based on your market data
    # a should roughly be the ATM Variance
    # Use the actual lowest IV point to center the model
    min_idx = smile_df['IV_pct'].idxmin()
    initial_m = np.log(smile_df.loc[min_idx, 'STRIKE'] / fwd_price)
    initial_a = (smile_df['IV_pct'].min() / 100)**2 * tte
    
    # [a, b, rho, m, sigma]
    initial_guess = [initial_a, 0.1, -0.5, initial_m, 0.1]
    # initial_a = max(1e-4, w_arr.min())
    # initial_guess = [initial_a, 0.05, -0.5, 0.0, 0.1]
    # initial_guess = [0.01, 0.1, -0.5, 0.0, 0.1]
    
    # 3. Optimization
    res = minimize(svi_objective, initial_guess, args=(k_arr, w_arr), method='SLSQP')
    logger.info("Starting SVI Optimization...")
    if res.success:
        logger.info(f"SVI Fit Successful. Params: {res.x}")
    else:
        logger.warning(f"SVI Fit FAILED: {res.message}")
    return res.x # Returns [a, b, rho, m, sigma]

def get_svi_results(params, fwd_price, tte, strike_range):
    # Create a dense strike grid for the PDF
    strikes = np.linspace(strike_range[0], strike_range[1], 500)
    k_grid = np.log(strikes / fwd_price)
    
    # Calculate Smoothed Variance and IV
    w_svi = svi_formula(params, k_grid)
    logger.info(f"SVI Fit Successful. Params: {w_svi}")
    iv_svi = np.sqrt(w_svi / tte) * 100
    
    # Generate Theoretical Prices for PDF calculation
    # Using small dk for numerical derivative
    def get_price(K):
        k = np.log(K / fwd_price)
        sig = np.sqrt(svi_formula(params, k) / tte)
        # Use your existing black_scholes_price function
        return black_scholes(fwd_price, K, tte, sig, 'CE')

    prices = np.array([get_price(s) for s in strikes])
    
    # PDF = Second derivative of Call Price w.r.t Strike
    dk = strikes[1] - strikes[0]
    pdf = np.gradient(np.gradient(prices, dk), dk)
    
    # Normalize PDF (Area = 1)
    pdf = np.maximum(pdf, 0)
    pdf /= integrate_trapezoid(pdf, strikes)
    area = integrate_trapezoid(pdf, strikes)
    logger.info(f"PDF Integration Area (pre-norm): {area:.6f}")
    if area < 0.9:
        logger.error("Probability Density area is dangerously low. Check SVI fit.")
        
    return strikes, iv_svi, pdf

# --- CORE FUNCTION: Download & Extract ---
def get_nifty_data(target_date):
    date_str = target_date.strftime("%Y%m%d")
    # Official NSE UDiFF URL Pattern
    url = f"https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{date_str}_F_0000.csv.zip"
    local_filename = f"{CACHE_DIR}/fo_{date_str}.csv"

    # Step 1: Check Disk Cache First
    if os.path.exists(local_filename):
        return load_nse_csv_to_pandas(local_filename)

    # Step 2: Download if not in Cache
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://www.nseindia.com/"
    }
    
    try:
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=headers, timeout=5) # Handshake
        response = session.get(url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                csv_name = z.namelist()[0]
                with z.open(csv_name) as f:
                    df = load_nse_csv_to_pandas(f)
                    df.columns = [c.strip() for c in df.columns]
                    # Map new ISO Tags to readable names immediately
                    cols = {
                        'TckrSymb': 'SYMBOL', 'XpryDt': 'EXPIRY', 'StrkPric': 'STRIKE',
                        'OptnTp': 'TYPE', 'ClsPric': 'CLOSE', 'OpnIntrst': 'OI', 'UndrlygPric': 'SPOT'
                    }
                    df = df.rename(columns=cols)
                    df.to_csv(local_filename, index=False)
                    return df
        else:
            return f"NSE Error: Status {response.status_code}. Is it a holiday?"
    except Exception as e:
        return f"Request failed: {str(e)}"

def get_india_vix(target_date,d=365):
    ticker = "^INDIAVIX"
    # Fetch a small window to ensure we get a valid trading day price
    start_dt = pd.to_datetime(target_date) - timedelta(days=d)
    data = yf.download(ticker, start=start_dt, end=target_date + timedelta(days=1), progress=False)
    vix_series = extract_close_series(data, ticker)
    return vix_series.rename('^INDIAVIX').to_frame()


def get_realized_vol(ticker_symbol, target_date,h=30,d=365):
    # 1. Fetch historical data (Lookback ~45 days to get 30 trading days)
    start_dt = pd.to_datetime(target_date) - timedelta(days=d)
    data = yf.download(ticker_symbol, start=start_dt, end=target_date + timedelta(days=1), progress=False)
    
    if data.empty:
        return pd.Series(dtype=float), None
    
    # Use 'Adj Close' for accuracy (dividends/splits), fallback to 'Close'
    prices = extract_close_series(data, ticker_symbol)
    if prices.empty:
        return pd.Series(dtype=float), None
    
    # 2. Calculate Log Returns
    # Formula: ln(Price_t / Price_{t-1})
    log_returns = np.log(prices / prices.shift(1)).dropna()
    
    # 3. Calculate Daily Std Dev and Annualize
    # Annualization Factor = sqrt(252 trading days)
    daily_vol = log_returns.rolling(h).std()
    annualized_rv = daily_vol * np.sqrt(252) * 100  # Convert to %
    
    current_price = float(prices.iloc[-1])
    
    return annualized_rv.rename(ticker_symbol), current_price


def get_price_series(ticker_symbol, target_date, d=365):
    start_dt = pd.to_datetime(target_date) - timedelta(days=d)
    data = yf.download(ticker_symbol, start=start_dt, end=target_date + timedelta(days=1), progress=False)
    return extract_close_series(data, ticker_symbol).rename(ticker_symbol)


def run_vrp_strategy(vol_df, price_series, lookback=30, entry_z=1.0, exit_z=0.25, cost_bps=2.0):
    """Long-only mean-reversion strategy on VRP z-score, traded on underlying returns."""
    if vol_df is None or vol_df.empty or price_series is None or price_series.empty:
        return pd.DataFrame()

    data = pd.concat(
        [
            vol_df.rename(columns={"Implied Vol": "Implied Vol", "Realized Vol": "Realized Vol"}),
            price_series.rename("Price"),
        ],
        axis=1,
    ).dropna()
    if data.empty:
        return pd.DataFrame()

    data["VRP"] = data["Implied Vol"] - data["Realized Vol"]
    rolling_mean = data["VRP"].rolling(lookback).mean()
    rolling_std = data["VRP"].rolling(lookback).std(ddof=0)
    data["VRP_Z"] = (data["VRP"] - rolling_mean) / rolling_std.replace(0, np.nan)

    positions = []
    current_pos = 0
    for z in data["VRP_Z"].fillna(0.0):
        if z < -entry_z:
            current_pos = 1
        elif z > -exit_z:
            current_pos = 0
        positions.append(current_pos)
    data["Position"] = positions

    data["Underlying_Return"] = data["Price"].pct_change().fillna(0.0)
    data["Turnover"] = data["Position"].diff().abs().fillna(abs(data["Position"]))
    data["Cost"] = data["Turnover"] * (cost_bps / 10000.0)
    data["Strategy_Return"] = data["Position"].shift(1).fillna(0.0) * data["Underlying_Return"] - data["Cost"]
    data["Strategy_Equity"] = (1.0 + data["Strategy_Return"]).cumprod()
    data["Strategy_CumPnL"] = data["Strategy_Equity"] - 1.0
    data["BuyHold_Equity"] = (1.0 + data["Underlying_Return"]).cumprod()
    data["BuyHold_CumPnL"] = data["BuyHold_Equity"] - 1.0
    data["Daily_PnL"] = data["Strategy_Return"] * 100.0
    return data.dropna(subset=["VRP_Z"])


def summarize_performance(returns, name):
    if returns is None or returns.empty:
        return {
            "Strategy": name,
            "Annual Return (%)": np.nan,
            "Annual Vol (%)": np.nan,
            "Sharpe": np.nan,
            "Max Drawdown (%)": np.nan,
            "Hit Rate (%)": np.nan,
        }

    n_obs = len(returns)
    ann_ret = ((1.0 + returns).prod() ** (252 / max(n_obs, 1)) - 1.0) * 100
    ann_vol = returns.std() * np.sqrt(252) * 100
    sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else np.nan
    equity = (1.0 + returns).cumprod()
    max_dd = ((equity / equity.cummax()) - 1.0).min() * 100
    hit_rate = (returns > 0).mean() * 100

    return {
        "Strategy": name,
        "Annual Return (%)": ann_ret,
        "Annual Vol (%)": ann_vol,
        "Sharpe": sharpe,
        "Max Drawdown (%)": max_dd,
        "Hit Rate (%)": hit_rate,
    }


def information_ratio(strategy_returns, benchmark_returns):
    if strategy_returns is None or benchmark_returns is None:
        return np.nan
    aligned = pd.concat(
        [strategy_returns.rename("strategy"), benchmark_returns.rename("benchmark")],
        axis=1
    ).dropna()
    if aligned.empty:
        return np.nan
    excess = aligned["strategy"] - aligned["benchmark"]
    te = excess.std()
    if te == 0 or pd.isna(te):
        return np.nan
    return (excess.mean() / te) * np.sqrt(252)


def optimize_vrp_parameters(vol_df, price_series, cost_bps=2.0):
    """Grid-search VRP strategy params for return-focused and drawdown-focused solutions."""
    lookback_grid = [20, 30, 40, 60, 90]
    entry_grid = [0.8, 1.0, 1.2, 1.5, 2.0]
    exit_grid = [0.1, 0.2, 0.3, 0.4, 0.5]
    rows = []

    for lookback in lookback_grid:
        for entry_z in entry_grid:
            for exit_z in exit_grid:
                if exit_z >= entry_z:
                    continue
                test_df = run_vrp_strategy(
                    vol_df,
                    price_series,
                    lookback=lookback,
                    entry_z=entry_z,
                    exit_z=exit_z,
                    cost_bps=cost_bps,
                )
                if test_df.empty:
                    continue

                trades = int((test_df["Turnover"] > 0).sum())
                if trades < 5:
                    continue
                stats = summarize_performance(test_df["Strategy_Return"], "VRP Strategy")
                if pd.isna(stats["Annual Return (%)"]) or pd.isna(stats["Max Drawdown (%)"]):
                    continue

                max_dd_abs = abs(stats["Max Drawdown (%)"])
                calmar = stats["Annual Return (%)"] / max(max_dd_abs, 1e-8)
                rows.append(
                    {
                        "lookback": lookback,
                        "entry_z": entry_z,
                        "exit_z": exit_z,
                        "annual_return": stats["Annual Return (%)"],
                        "annual_vol": stats["Annual Vol (%)"],
                        "sharpe": stats["Sharpe"],
                        "max_drawdown": stats["Max Drawdown (%)"],
                        "calmar_like": calmar,
                        "trades": trades,
                    }
                )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def rank_symbols_by_optimal_strategy(symbols, target_date, cost_bps=2.0, lookback_rv=30):
    """Rank symbols by their best optimizer outcome (annual return)."""
    if not symbols:
        return pd.DataFrame()

    iv_df = get_india_vix(target_date)
    if iv_df.empty or "^INDIAVIX" not in iv_df.columns:
        return pd.DataFrame()
    implied_series = iv_df["^INDIAVIX"].dropna().rename("Implied Vol")

    rows = []
    for sym in sorted(set(symbols)):
        yf_sym = get_yf_symbol(sym)
        price_series = get_price_series(yf_sym, target_date).dropna()
        if price_series.empty:
            continue

        log_returns = np.log(price_series / price_series.shift(1)).dropna()
        rv_series = (log_returns.rolling(lookback_rv).std() * np.sqrt(252) * 100).dropna().rename("Realized Vol")
        vol_df = pd.concat([implied_series, rv_series], axis=1).dropna()
        if vol_df.empty:
            continue

        opt_df = optimize_vrp_parameters(vol_df, price_series, cost_bps=cost_bps)
        if opt_df.empty:
            continue

        best_row = opt_df.sort_values("annual_return", ascending=False).iloc[0]
        rows.append(
            {
                "Symbol": sym,
                "Annual Return (%)": float(best_row["annual_return"]),
                "Annual Risk (%)": float(best_row["annual_vol"]),
                "Max Drawdown (%)": float(best_row["max_drawdown"]),
                "Sharpe": float(best_row["sharpe"]),
                "Lookback": int(best_row["lookback"]),
                "Entry Z": float(best_row["entry_z"]),
                "Exit Z": float(best_row["exit_z"]),
                "Trades": int(best_row["trades"]),
            }
        )

    if not rows:
        return pd.DataFrame()
    rank_df = pd.DataFrame(rows).sort_values("Annual Return (%)", ascending=False).reset_index(drop=True)
    return rank_df


def build_dynamic_top_portfolio(ranked_df, target_date, cost_bps=2.0, top_n=10, lookback_rv=30):
    """Build equal-weight portfolio from top-N ranked symbols using each symbol's optimal params."""
    if ranked_df is None or ranked_df.empty:
        return pd.DataFrame(), {}

    top_df = ranked_df.head(min(top_n, len(ranked_df))).copy()
    iv_df = get_india_vix(target_date)
    if iv_df.empty or "^INDIAVIX" not in iv_df.columns:
        return pd.DataFrame(), {}
    implied_series = iv_df["^INDIAVIX"].dropna().rename("Implied Vol")

    strategy_returns = {}
    benchmark_returns = {}
    trade_counts = {}

    for _, row in top_df.iterrows():
        symbol = row["Symbol"]
        yf_symbol = get_yf_symbol(symbol)
        price_series = get_price_series(yf_symbol, target_date).dropna()
        if price_series.empty:
            continue

        log_returns = np.log(price_series / price_series.shift(1)).dropna()
        rv_series = (log_returns.rolling(lookback_rv).std() * np.sqrt(252) * 100).dropna().rename("Realized Vol")
        vol_df = pd.concat([implied_series, rv_series], axis=1).dropna()
        if vol_df.empty:
            continue

        strategy_df = run_vrp_strategy(
            vol_df,
            price_series,
            lookback=int(row["Lookback"]),
            entry_z=float(row["Entry Z"]),
            exit_z=float(row["Exit Z"]),
            cost_bps=cost_bps,
        )
        if strategy_df.empty:
            continue

        strategy_returns[symbol] = strategy_df["Strategy_Return"]
        benchmark_returns[symbol] = strategy_df["Underlying_Return"]
        trade_counts[symbol] = int((strategy_df["Turnover"] > 0).sum())

    if not strategy_returns:
        return pd.DataFrame(), {}

    strategy_matrix = pd.concat(strategy_returns, axis=1)
    benchmark_matrix = pd.concat(benchmark_returns, axis=1)
    portfolio_strategy = strategy_matrix.mean(axis=1, skipna=True).dropna().rename("Portfolio_Return")
    portfolio_benchmark = benchmark_matrix.mean(axis=1, skipna=True).dropna().rename("Benchmark_Return")

    portfolio_df = pd.concat([portfolio_strategy, portfolio_benchmark], axis=1).dropna()
    if portfolio_df.empty:
        return pd.DataFrame(), {}

    portfolio_df["Daily_PnL"] = portfolio_df["Portfolio_Return"] * 100
    portfolio_df["Portfolio_CumPnL"] = (1.0 + portfolio_df["Portfolio_Return"]).cumprod() - 1.0
    portfolio_df["Benchmark_CumPnL"] = (1.0 + portfolio_df["Benchmark_Return"]).cumprod() - 1.0

    metadata = {
        "n_symbols": len(strategy_returns),
        "symbols": list(strategy_returns.keys()),
        "total_trades": int(sum(trade_counts.values())),
        "avg_trades_per_symbol": float(np.mean(list(trade_counts.values()))) if trade_counts else 0.0,
    }
    return portfolio_df, metadata


def build_rotating_top_portfolio_all_business_dates(
    ranked_df, target_date, cost_bps=2.0, top_n=10, lookback_rv=30, selection_lookback=60
):
    """
    Daily rotating portfolio:
    - Candidate universe: ranked symbols
    - Daily selection: top-N by trailing strategy performance over selection_lookback business days
    """
    if ranked_df is None or ranked_df.empty:
        return pd.DataFrame(), {}

    iv_df = get_india_vix(target_date)
    if iv_df.empty or "^INDIAVIX" not in iv_df.columns:
        return pd.DataFrame(), {}
    implied_series = iv_df["^INDIAVIX"].dropna().rename("Implied Vol")

    strategy_returns = {}
    benchmark_returns = {}
    trade_counts = {}

    for _, row in ranked_df.iterrows():
        symbol = row["Symbol"]
        yf_symbol = get_yf_symbol(symbol)
        price_series = get_price_series(yf_symbol, target_date).dropna()
        if price_series.empty:
            continue

        log_returns = np.log(price_series / price_series.shift(1)).dropna()
        rv_series = (log_returns.rolling(lookback_rv).std() * np.sqrt(252) * 100).dropna().rename("Realized Vol")
        vol_df = pd.concat([implied_series, rv_series], axis=1).dropna()
        if vol_df.empty:
            continue

        strategy_df = run_vrp_strategy(
            vol_df,
            price_series,
            lookback=int(row["Lookback"]),
            entry_z=float(row["Entry Z"]),
            exit_z=float(row["Exit Z"]),
            cost_bps=cost_bps,
        )
        if strategy_df.empty:
            continue

        strategy_returns[symbol] = strategy_df["Strategy_Return"]
        benchmark_returns[symbol] = strategy_df["Underlying_Return"]
        trade_counts[symbol] = int((strategy_df["Turnover"] > 0).sum())

    if not strategy_returns:
        return pd.DataFrame(), {}

    strategy_matrix = pd.concat(strategy_returns, axis=1).sort_index()
    benchmark_matrix = pd.concat(benchmark_returns, axis=1).sort_index()
    common_index = strategy_matrix.index.intersection(benchmark_matrix.index)
    strategy_matrix = strategy_matrix.loc[common_index]
    benchmark_matrix = benchmark_matrix.loc[common_index]
    if strategy_matrix.empty:
        return pd.DataFrame(), {}

    trailing_score = strategy_matrix.rolling(selection_lookback).mean()
    portfolio_returns = []
    benchmark_portfolio_returns = []
    holdings_count = []
    used_dates = []
    ranking_rows = []

    for i in range(1, len(strategy_matrix)):
        dt = strategy_matrix.index[i]
        prev_dt = strategy_matrix.index[i - 1]
        score_row = trailing_score.loc[prev_dt].dropna()
        if score_row.empty:
            continue

        selected_symbols = score_row.sort_values(ascending=False).head(top_n).index.tolist()
        ranked_scores = score_row.sort_values(ascending=False).head(top_n)
        for rank_idx, (sym, score_val) in enumerate(ranked_scores.items(), start=1):
            ranking_rows.append(
                {"Date": dt, "Rank": rank_idx, "Symbol": sym, "Score": float(score_val)}
            )
        todays_strategy = strategy_matrix.loc[dt, selected_symbols].dropna()
        todays_benchmark = benchmark_matrix.loc[dt, selected_symbols].dropna()
        if todays_strategy.empty or todays_benchmark.empty:
            continue

        portfolio_returns.append(todays_strategy.mean())
        benchmark_portfolio_returns.append(todays_benchmark.mean())
        holdings_count.append(len(selected_symbols))
        used_dates.append(dt)

    if not used_dates:
        return pd.DataFrame(), {}

    portfolio_df = pd.DataFrame(
        {
            "Portfolio_Return": portfolio_returns,
            "Benchmark_Return": benchmark_portfolio_returns,
            "Holdings": holdings_count,
        },
        index=pd.Index(used_dates),
    ).sort_index()
    portfolio_df["Daily_PnL"] = portfolio_df["Portfolio_Return"] * 100
    portfolio_df["Portfolio_CumPnL"] = (1.0 + portfolio_df["Portfolio_Return"]).cumprod() - 1.0
    portfolio_df["Benchmark_CumPnL"] = (1.0 + portfolio_df["Benchmark_Return"]).cumprod() - 1.0

    metadata = {
        "n_symbols": len(strategy_returns),
        "symbols": list(strategy_returns.keys()),
        "total_trades": int(sum(trade_counts.values())),
        "avg_trades_per_symbol": float(np.mean(list(trade_counts.values()))) if trade_counts else 0.0,
        "selection_lookback": selection_lookback,
        "avg_holdings": float(np.mean(holdings_count)) if holdings_count else 0.0,
        "daily_ranking_df": pd.DataFrame(ranking_rows),
    }
    return portfolio_df, metadata


def get_cached_bhavcopy_dates():
    """Return cached bhavcopy dates from data_cache as date objects (latest first)."""
    dates = []
    if not os.path.exists(CACHE_DIR):
        return dates
    for name in os.listdir(CACHE_DIR):
        if not (name.startswith("fo_") and name.endswith(".csv")):
            continue
        date_part = name[3:11]
        try:
            dates.append(datetime.strptime(date_part, "%Y%m%d").date())
        except Exception:
            continue
    return sorted(set(dates), reverse=True)

# Example Usage:
# rv, price = get_realized_vol("^NSEI", "2026-01-05")
# print(f"Nifty RV: {rv:.2f}%, Close: {price:.2f}")
# --- SIDEBAR & SESSION STATE ---
if 'active_df' not in st.session_state:
    st.session_state.active_df = None
if 'active_df_pl' not in st.session_state:
    st.session_state.active_df_pl = None

with st.sidebar:
    st.header("📅 Data Controls")
    trading_date = st.date_input("Trading Day", value=datetime(2025, 12, 31))
    default_symbols = ["NIFTY", "BANKNIFTY"]
    if st.session_state.active_df is not None and 'SYMBOL' in st.session_state.active_df.columns:
        available_symbols = sorted(st.session_state.active_df['SYMBOL'].dropna().astype(str).unique().tolist())
        if not available_symbols:
            available_symbols = default_symbols
    else:
        available_symbols = default_symbols
    default_symbol = "NIFTY" if "NIFTY" in available_symbols else available_symbols[0]
    if "selected_symbol_widget" not in st.session_state or st.session_state.selected_symbol_widget not in available_symbols:
        st.session_state.selected_symbol_widget = default_symbol
    if "pending_selected_symbol" in st.session_state:
        pending_symbol = st.session_state.pop("pending_selected_symbol")
        if pending_symbol in available_symbols:
            st.session_state.selected_symbol_widget = pending_symbol
    symbol = st.selectbox("Underlying", available_symbols, key="selected_symbol_widget")
    st.session_state.selected_symbol = symbol
    yf_symbol = get_yf_symbol(symbol)
    # TRIGGER BUTTON
    if st.button("🚀 Get Data", width="stretch"):
        with st.spinner("Fetching from NSE Archives..."):
            result = get_nifty_data(trading_date)
            if isinstance(result, pd.DataFrame):
                st.session_state.active_df = result
                st.session_state.active_df_pl = to_polars_if_available(result)
                st.success("Data Loaded!")
            else:
                st.error(result)

    # # 2. Risk-Free Rate Input
    # # We use format="%.2f" to show two decimal places
    # risk_free_percent = st.number_input(
    #     "Risk-Free Rate (%)", 
    #     min_value=0.0, 
    #     max_value=15.0, 
    #     value=7.0, 
    #     step=0.05,
    #     format="%.2f",
    #     help="The annual yield of a risk-free bond (like a Govt Treasury Bond)."
    # )
    
    # # Convert the percentage to a decimal for the Black-Scholes math
    # risk_free = risk_free_percent / 100
    
    min_volume = st.number_input(
        "Minimum Volume", 
        min_value=0, 
        max_value=10000000, 
        value=10000 ,
        step=1000,
        format="%d",
        help="Minimum volume that the option contracts to have to be considered in the analysis."
    )
    st.divider()

# --- MAIN DISPLAY LOGIC ---
if st.session_state.active_df is not None:
    tab1, tab2, tab3 = st.tabs(["Volatility Smile & PDF", "Term Structure", "Volatility Risk Premium"])
    df = st.session_state.active_df
    df_pl = st.session_state.active_df_pl
    if df_pl is None:
        df_pl = to_polars_if_available(df)
        st.session_state.active_df_pl = df_pl
    
    # Filter for Symbol
    if POLARS_AVAILABLE and df_pl is not None:
        df_symbol = df_pl.filter(pl.col("SYMBOL") == symbol).to_pandas()
    else:
        df_symbol = df[df['SYMBOL'] == symbol].copy()
    if df_symbol.empty:
        st.warning(f"No contracts found for {symbol} on {trading_date}. Try another symbol/date.")
        st.stop()
    
    with tab3:
        rv_series, _ = get_realized_vol(yf_symbol, trading_date)
        rv_series = rv_series.dropna()
        price_series = get_price_series(yf_symbol, trading_date).dropna()
        iv_proxy_df = get_india_vix(trading_date)
        iv_proxy_series = iv_proxy_df["^INDIAVIX"] if "^INDIAVIX" in iv_proxy_df.columns else pd.Series(dtype=float)
        implied_realized_df = pd.DataFrame()
        if symbol == "NIFTY":
            vrp_df = iv_proxy_df.join(rv_series.rename(yf_symbol), how='inner').dropna()
            implied_realized_df = vrp_df.rename(columns={"^INDIAVIX": "Implied Vol", yf_symbol: "Realized Vol"})[
                ["Implied Vol", "Realized Vol"]
            ]
            st.subheader('NIFTY - Volatility Risk Premium (VIX vs 30D Realized Volatility)')
            with st.container(border=True):
                if vrp_df.empty:
                    st.warning("Not enough VIX/price history to compute VRP for this date.")
                else:
                    fig_vol = px.line(
                        vrp_df,
                        x=vrp_df.index,
                        y=["^INDIAVIX", yf_symbol],
                        title="Implied (VIX) vs Realized Volatility",
                        labels={"value": "Volatility (%)", "variable": "Type"},
                        template="plotly_dark"
                    )
                    fig_vol.update_traces(line=dict(width=2))
                    fig_vol.update_layout(xaxis_title=None, hovermode="x unified")
                    st.plotly_chart(fig_vol, use_container_width=True)
        else:
            implied_vs_realized_df = pd.concat(
                [iv_proxy_series.rename("Implied Vol (INDIAVIX)"), rv_series.rename("Realized Vol (30D)")],
                axis=1
            ).dropna()
            if implied_vs_realized_df.empty:
                st.warning("Implied vs realized series unavailable for this symbol/date window.")
            else:
                implied_realized_df = implied_vs_realized_df.rename(
                    columns={"Implied Vol (INDIAVIX)": "Implied Vol", "Realized Vol (30D)": "Realized Vol"}
                )
                fig_iv_rv = px.line(
                    implied_vs_realized_df,
                    x=implied_vs_realized_df.index,
                    y=["Implied Vol (INDIAVIX)", "Realized Vol (30D)"],
                    title=f"{symbol} Realized vs Implied Volatility (Time Series)",
                    labels={"value": "Volatility (%)", "variable": "Series"},
                    template="plotly_dark",
                )
                fig_iv_rv.update_traces(line=dict(width=2))
                fig_iv_rv.update_layout(xaxis_title=None, hovermode="x unified")
                st.plotly_chart(fig_iv_rv, use_container_width=True)
        st.divider()
        st.subheader("Optimal Entry/Exit Suggestions")
        cost_bps = st.number_input(
            "Cost (bps/trade)", min_value=0.0, max_value=50.0, value=2.0, step=0.5, key=f"cost_{symbol}"
        )
        opt_df = optimize_vrp_parameters(implied_realized_df, price_series, cost_bps=cost_bps)
        if opt_df.empty:
            st.info("Optimizer could not find enough valid combinations for this dataset.")
        else:
            best_return = opt_df.sort_values("annual_return", ascending=False).iloc[0]
            best_defensive = opt_df.sort_values(["max_drawdown", "annual_return"], ascending=[False, False]).iloc[0]
            summary_df = pd.DataFrame(
                [
                    {
                        "Profile": "Max Return",
                        "Lookback": int(best_return["lookback"]),
                        "Entry Z": float(best_return["entry_z"]),
                        "Exit Z": float(best_return["exit_z"]),
                        "Annual Return (%)": float(best_return["annual_return"]),
                        "Annual Risk (%)": float(best_return["annual_vol"]),
                        "Max Drawdown (%)": float(best_return["max_drawdown"]),
                        "Risk-Adjusted Return (Sharpe)": float(best_return["sharpe"]),
                    },
                    {
                        "Profile": "Min Drawdown",
                        "Lookback": int(best_defensive["lookback"]),
                        "Entry Z": float(best_defensive["entry_z"]),
                        "Exit Z": float(best_defensive["exit_z"]),
                        "Annual Return (%)": float(best_defensive["annual_return"]),
                        "Annual Risk (%)": float(best_defensive["annual_vol"]),
                        "Max Drawdown (%)": float(best_defensive["max_drawdown"]),
                        "Risk-Adjusted Return (Sharpe)": float(best_defensive["sharpe"]),
                    },
                ]
            ).round(2)
            st.dataframe(summary_df, use_container_width=True)

            selected_profile = st.radio(
                "Performance view for",
                ["Max Return", "Min Drawdown"],
                horizontal=True,
                key=f"opt_profile_{symbol}",
            )
            chosen = best_return if selected_profile == "Max Return" else best_defensive
            strategy_df = run_vrp_strategy(
                implied_realized_df,
                price_series,
                lookback=int(chosen["lookback"]),
                entry_z=float(chosen["entry_z"]),
                exit_z=float(chosen["exit_z"]),
                cost_bps=cost_bps,
            )
            if strategy_df.empty:
                st.warning("Unable to build strategy series from selected optimal parameters.")
            else:
                strategy_stats = summarize_performance(strategy_df["Strategy_Return"], "Optimal VRP Strategy")
                benchmark_stats = summarize_performance(strategy_df["Underlying_Return"], "Buy & Hold")
                strategy_ir = information_ratio(strategy_df["Strategy_Return"], strategy_df["Underlying_Return"])

                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                with metric_col1:
                    st.metric("Return (%)", f"{strategy_stats['Annual Return (%)']:.2f}")
                with metric_col2:
                    st.metric("Risk (%)", f"{strategy_stats['Annual Vol (%)']:.2f}")
                with metric_col3:
                    st.metric("Drawdown (%)", f"{strategy_stats['Max Drawdown (%)']:.2f}")
                with metric_col4:
                    st.metric("Risk-Adjusted (Sharpe)", f"{strategy_stats['Sharpe']:.2f}")

                stats_df = pd.DataFrame([strategy_stats, benchmark_stats]).round(2)
                stats_df["Trades"] = [int((strategy_df["Turnover"] > 0).sum()), 0]
                stats_df["IR vs BuyHold"] = [strategy_ir, np.nan]
                st.dataframe(stats_df, use_container_width=True)

                # IRP (Implied - Realized Premium) vs Price with strategy entry/exit markers
                irp_plot_df = strategy_df.copy()
                position_change = irp_plot_df["Position"].diff().fillna(irp_plot_df["Position"])
                long_entries = irp_plot_df[position_change > 0]
                exits = irp_plot_df[position_change < 0]

                fig_irp_price = go.Figure()
                fig_irp_price.add_trace(
                    go.Scatter(
                        x=irp_plot_df.index,
                        y=irp_plot_df["VRP"],
                        name="IRP (Implied - Realized)",
                        line=dict(color="#00d4ff", width=2),
                        yaxis="y1",
                    )
                )
                fig_irp_price.add_trace(
                    go.Scatter(
                        x=irp_plot_df.index,
                        y=irp_plot_df["Price"],
                        name=f"{symbol} Price",
                        line=dict(color="#f5c542", width=2),
                        yaxis="y2",
                    )
                )
                if not long_entries.empty:
                    fig_irp_price.add_trace(
                        go.Scatter(
                            x=long_entries.index,
                            y=long_entries["Price"],
                            mode="markers",
                            name="Entry Long",
                            marker=dict(symbol="triangle-up", color="#00ff88", size=10),
                            yaxis="y2",
                        )
                    )
                if not exits.empty:
                    fig_irp_price.add_trace(
                        go.Scatter(
                            x=exits.index,
                            y=exits["Price"],
                            mode="markers",
                            name="Exit",
                            marker=dict(symbol="x", color="#ffffff", size=13, line=dict(color="#ff4b4b", width=1)),
                            yaxis="y2",
                        )
                    )
                fig_irp_price.update_layout(
                    title=f"IRP vs Price with Entry/Exit Points ({selected_profile})",
                    template="plotly_dark",
                    hovermode="x unified",
                    xaxis=dict(title=None),
                    yaxis=dict(title="IRP (%)", side="left"),
                    yaxis2=dict(title="Price", overlaying="y", side="right"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(fig_irp_price, use_container_width=True)

                pnl_col1, pnl_col2 = st.columns(2)
                with pnl_col1:
                    fig_daily_pnl = px.bar(
                        strategy_df,
                        x=strategy_df.index,
                        y="Daily_PnL",
                        title=f"Daily PnL (%) - {selected_profile}",
                        template="plotly_dark",
                        labels={"x": "Date", "Daily_PnL": "PnL (%)"},
                    )
                    fig_daily_pnl.update_layout(xaxis_title=None)
                    st.plotly_chart(fig_daily_pnl, use_container_width=True)

                with pnl_col2:
                    fig_cum_pnl = px.line(
                        strategy_df,
                        x=strategy_df.index,
                        y=["Strategy_CumPnL", "BuyHold_CumPnL"],
                        title=f"Cumulative PnL - {selected_profile}",
                        template="plotly_dark",
                        labels={"value": "Cumulative PnL", "variable": "Series"},
                    )
                    fig_cum_pnl.update_layout(xaxis_title=None, hovermode="x unified")
                    st.plotly_chart(fig_cum_pnl, use_container_width=True)
    with tab2:
        if POLARS_AVAILABLE and df_pl is not None:
            all_strikes = (
                df_pl.filter(pl.col("SYMBOL") == symbol)
                .select("STRIKE")
                .drop_nulls()
                .unique()
                .sort("STRIKE")
                .to_series()
                .to_list()
            )
        else:
            all_strikes = sorted(df_symbol['STRIKE'].unique())
        if not all_strikes:
            st.warning("No strikes available for this symbol/date.")
            st.stop()
        col_select, col_empty = st.columns([2, 4])
        with col_select:
            selected_strike = st.selectbox("Select Strike", all_strikes)
        st.divider()
        # 2. Final Filtered Data
        if POLARS_AVAILABLE and df_pl is not None:
            final_df = (
                df_pl.filter((pl.col("SYMBOL") == symbol) & (pl.col("STRIKE") == selected_strike))
                .sort("EXPIRY")
                .to_pandas()
            )
        else:
            final_df = df_symbol[df_symbol['STRIKE'] == selected_strike].sort_values("EXPIRY")
        st.subheader(f"📊 {symbol} Analysis - Expiry: {selected_strike}")
    
        # Create two equal-width columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Plot 1: Close Prices
            fig_price = px.line(
                final_df, 
                x="EXPIRY", 
                y="CLOSE", 
                color="TYPE",
                title="Option Closing Prices",
                labels={"EXPIRY": "EXPIRY", "CLOSE": "Price (₹)"},
                color_discrete_map={'CE': '#00ff00', 'PE': '#ff0000'},
                markers=True, 
                template="plotly_dark"
            )
            fig_price.update_layout(hovermode="x unified")
            st.plotly_chart(fig_price, width="stretch")
        
        # with col2:
        #     # Plot 2: Implied Volatility (The Smile)
        #     # 1. Create the base figure with Market IV
        #     fig_iv = px.line(
        #         smile_df, 
        #         x="STRIKE", 
        #         y="IV_pct", 
        #         title="IV Smile: Market vs SVI (using OTM CE and OT PE, Synthetic Forwrard)",
        #         markers=True,
        #         template="plotly_dark"
        #     )
        #     fig_iv.update_traces(mode='markers')
        #     # 2. Create a temporary figure for the SVI line
        #     fig_svi_line = px.line(
        #         pdf_df, 
        #         x="STRIKE", 
        #         y="IV_SVI"
        #     )
            
        #     # 3. Change the SVI line color so it stands out
        #     fig_svi_line.update_traces(line_color='#00d4ff', name='SVI Fit', showlegend=True)
            
        #     # 4. Add the traces from the SVI figure to the base figure
        #     for trace in fig_svi_line.data:
        #         fig_iv.add_trace(trace)
            
        #     # 5. Final updates and display
        #     fig_iv.update_layout(hovermode="x unified")
        #     st.plotly_chart(fig_iv, width="stretch")
    with tab1:
        # 1. Maturity Selector (Now that data exists)
        if POLARS_AVAILABLE and df_pl is not None:
            all_expiries = (
                df_pl.filter(pl.col("SYMBOL") == symbol)
                .select("EXPIRY")
                .drop_nulls()
                .unique()
                .sort("EXPIRY")
                .to_series()
                .to_list()
            )
        else:
            all_expiries = sorted(df_symbol['EXPIRY'].unique())
        if not all_expiries:
            st.warning("No expiries available for this symbol/date.")
            st.stop()
        col_select, col_empty = st.columns([2, 4])
        with col_select:
            selected_expiry = st.selectbox("Select Maturity", all_expiries)
        st.divider()
        # 2. Final Filtered Data
        if POLARS_AVAILABLE and df_pl is not None:
            final_df = (
                df_pl.filter((pl.col("SYMBOL") == symbol) & (pl.col("EXPIRY") == selected_expiry))
                .sort("STRIKE")
                .to_pandas()
            )
        else:
            final_df = df_symbol[df_symbol['EXPIRY'] == selected_expiry].sort_values("STRIKE")
    
        # Calculate IV for each row
        with st.spinner("Calculating Implied Volatility..."):
            if final_df.empty:
                st.warning("No contracts found for the selected expiry.")
                st.stop()
            
            tte = get_tte(str(trading_date), selected_expiry)
            strike_range = 0.2
            # min_volume = 10000
            spot_price = final_df['SPOT'].iloc[0]
            upper_bound = spot_price * (1 + strike_range)
            lower_bound = spot_price * (1 - strike_range)
            
            final_df = final_df[
                (final_df['STRIKE'] >= lower_bound) & 
                (final_df['STRIKE'] <= upper_bound)
            ].copy()
            final_df = final_df[final_df['OI'] > min_volume]
            if final_df.empty:
                st.warning("No contracts left after strike/OI filters. Reduce minimum volume or widen range.")
                st.stop()
            # 1. Identify the ATM Strike
            
            atm_strike = final_df.iloc[(final_df['STRIKE'] - spot_price).abs().argsort()[:1]]['STRIKE'].values[0]
            
            # 2. Filter using the ATM strike as the pivot
            otm_puts = final_df[(final_df['TYPE'] == 'PE') & (final_df['STRIKE'] <= atm_strike)]
            otm_calls = final_df[(final_df['TYPE'] == 'CE') & (final_df['STRIKE'] >= atm_strike)]
            atm_call = final_df[(final_df['STRIKE'] == atm_strike) & (final_df['TYPE'] == 'CE')]['CLOSE'].mean()
            atm_put = final_df[(final_df['STRIKE'] == atm_strike) & (final_df['TYPE'] == 'PE')]['CLOSE'].mean()
            
            synthetic_fwd = atm_strike + (atm_call - atm_put)
    
            # --- Inside your Data Processing block ---
            logger.info(f"Processing symbol: {symbol}")
            logger.info(f"Spot: {spot_price} | Synthetic Fwd: {synthetic_fwd:.2f}")
    
            # 3. Combine
            smile_df = pd.concat([otm_puts, otm_calls]).drop_duplicates(subset=['STRIKE', 'TYPE'])
    
            # final_df = pd.concat([final_df[(final_df['TYPE']=='PE')&(final_df['STRIKE']<final_df['SPOT'])],\
            #                       final_df[(final_df['TYPE']=='CE')&(final_df['STRIKE']>final_df['SPOT'])]])
            # smile_df['intrinsic'] = np.where(
            #     smile_df['TYPE'] == 'CE',
            #     (smile_df['SPOT'] - smile_df['STRIKE'] * np.exp(-risk_free * tte)), # Discounted Strike
            #     (smile_df['STRIKE'] * np.exp(-risk_free * tte) - smile_df['SPOT']))
            smile_df['intrinsic'] = np.where(smile_df['TYPE'] == 'CE', 
                                         (synthetic_fwd - smile_df['STRIKE']), 
                                         (smile_df['STRIKE'] - synthetic_fwd))
            
            smile_df['intrinsic'] = smile_df['intrinsic'].clip(lower=0)
            smile_df = smile_df[smile_df['CLOSE'] > smile_df['intrinsic']].copy()
            smile_df = smile_df[smile_df['CLOSE'] > 1.0]
            if smile_df.empty:
                st.warning("No OTM options left after intrinsic/price filters.")
                st.stop()
            smile_df['IV'] = smile_df.apply(
                lambda row: find_iv(row['CLOSE'], synthetic_fwd, row['STRIKE'], tte, row['TYPE']), axis=1
            )
            # Convert to percentage
            smile_df['IV_pct'] = smile_df['IV'] * 100
            smile_df = smile_df.dropna(subset=['IV_pct'])
            if smile_df.empty:
                st.warning("IV calculation returned no valid points for SVI fit.")
                st.stop()
            
            
            # --- RUNNING IT ---
            params = fit_svi(smile_df, synthetic_fwd, tte)
            strikes_fine, iv_fine, pdf_fine = get_svi_results(params, synthetic_fwd, tte, 
                                                             [smile_df['STRIKE'].min(), smile_df['STRIKE'].max()])
            
            pdf_df = pd.DataFrame({
                    'STRIKE': strikes_fine,
                    'IV_SVI': iv_fine,
                    'PDF': pdf_fine
                })
            logger.info(f"Spot: {pdf_df}")
        # 3. Visualization
    #    st.subheader(f"📈 {symbol} Close Prices - Expiry: {selected_expiry}")
    #    
    #    fig = px.line(final_df, x="STRIKE", y="CLOSE", color="TYPE",
    #                  labels={"STRIKE": "Strike Price", "CLOSE": "Closing Price"},
    #                  markers=True, template="plotly_dark")
    #    fig = px.line(final_df, x="STRIKE", y="IV_pct", color="TYPE",
    #                  labels={"STRIKE": "Strike Price", "IV_Pct": "Implied Volatility (BS)"},
    #                  markers=True, template="plotly_dark")
    #    st.plotly_chart(fig, width=True)
    # --- 3. SIDE-BY-SIDE VISUALIZATION ---
        
            rv_series, _ = get_realized_vol(yf_symbol, trading_date, 30)
            rv = rv_series.dropna().iloc[-1] if not rv_series.empty else np.nan
            # final_df['rv'] = rv
            logger.info(f"realized volatility: {rv}")
        st.subheader(f"📊 {symbol} Analysis - Expiry: {selected_expiry}")
    
        # Create two equal-width columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Plot 1: Close Prices
            fig_price = px.line(
                final_df, 
                x="STRIKE", 
                y="CLOSE", 
                color="TYPE",
                title="Option Closing Prices",
                labels={"STRIKE": "Strike", "CLOSE": "Price (₹)"},
                color_discrete_map={'CE': '#00ff00', 'PE': '#ff0000'},
                markers=True, 
                template="plotly_dark"
            )
            fig_price.update_layout(hovermode="x unified")
            st.plotly_chart(fig_price, width="stretch")
        
        with col2:
            # Plot 2: Implied Volatility (The Smile)
            # 1. Create the base figure with Market IV
            fig_iv = px.line(
                smile_df, 
                x="STRIKE", 
                y="IV_pct", 
                title="IV Smile: Market vs SVI (using OTM CE and OT PE, Synthetic Forwrard)",
                markers=True,
                template="plotly_dark"
            )
            fig_iv.update_traces(mode='markers')
            # 2. Create a temporary figure for the SVI line
            fig_svi_line = px.line(
                pdf_df, 
                x="STRIKE", 
                y="IV_SVI"
            )
            # 3. Change the SVI line color so it stands out
            fig_svi_line.update_traces(line_color='#00d4ff', name='SVI Fit', showlegend=True)
            
            
            # 4. Add the traces from the SVI figure to the base figure
            for trace in fig_svi_line.data: #, fig_rv_line.data
                fig_iv.add_trace(trace)
            
            if pd.notna(rv):
                fig_iv.add_hline(
                    y=rv,
                    line_dash="dash",
                    line_color="#ff4b4b",
                    annotation_text=f"30D Realized Vol ({rv:.2f}%)",
                    annotation_position="bottom right"
                )
            
            # 5. Final updates and display
            fig_iv.update_layout(hovermode="x unified")
            st.plotly_chart(fig_iv, width="stretch")
            
        with col3:
            fig_pdf = go.Figure()
    
    # Shaded Area for Probability
            fig_pdf.add_trace(go.Scatter(
                x=strikes_fine, 
                y=pdf_fine, 
                fill='tozeroy', 
                name='Risk-Neutral Density',
                line=dict(color='#ff7f0e', width=2),
                fillcolor='rgba(255, 127, 14, 0.2)'
            ))
            
            # Forward Price Reference
            fig_pdf.add_vline(x=synthetic_fwd, line_dash="dot", line_color="white", 
                              annotation_text="Expected Price (Forward)")
            
            fig_pdf.update_layout(
                title="Market-Implied Probability Distribution (at Expiry)",
                xaxis_title="Price at Expiry",
                yaxis_title="Probability Density",
                template="plotly_dark",
                hovermode="x unified",
                height=400 # Slightly taller for better readability
            )
            st.plotly_chart(fig_pdf, width="stretch")
            # 4. Data Table
        with st.expander("View Filtered Data Table"):
            st.dataframe(smile_df[['STRIKE', 'TYPE', 'CLOSE', 'IV_pct', 'OI']].sort_values(by=['TYPE','STRIKE']), width="stretch")
        with st.expander("View SVI IV and PDF"):
            st.dataframe(pdf_df[['STRIKE','IV_SVI','PDF']], width='stretch')
    st.divider()
    st.subheader("Best Performing Stocks (Optimizer Ranked)")
    rank_cost_bps = st.number_input(
        "Ranking Cost (bps/trade)",
        min_value=0.0,
        max_value=50.0,
        value=2.0,
        step=0.5,
        key="rank_cost_bps",
    )
    if st.button("Run Ranking", key="run_ranking_btn"):
        symbols_for_rank = sorted(df["SYMBOL"].dropna().astype(str).unique().tolist())
        with st.spinner("Ranking symbols by optimal strategy performance..."):
            st.session_state["ranked_stocks_df"] = rank_symbols_by_optimal_strategy(
                symbols_for_rank,
                trading_date,
                cost_bps=rank_cost_bps,
            )

    ranked_df = st.session_state.get("ranked_stocks_df", pd.DataFrame())
    if ranked_df.empty:
        st.info("Run ranking to identify top-performing stocks from the loaded symbol list.")
    else:
        ranked_display = ranked_df.copy().round(2)
        st.dataframe(ranked_display, use_container_width=True)
        top_symbol = ranked_df.iloc[0]["Symbol"]
        st.caption(f"Top stock by optimizer annual return: {top_symbol}")

        pick_col1, pick_col2 = st.columns([3, 1])
        with pick_col1:
            ranked_symbol_choice = st.selectbox(
                "Select a ranked stock to analyze",
                ranked_df["Symbol"].tolist(),
                key="ranked_symbol_choice",
            )
        with pick_col2:
            st.write("")
            st.write("")
            if st.button("Analyze Selected", key="analyze_ranked_symbol_btn", use_container_width=True):
                st.session_state.pending_selected_symbol = ranked_symbol_choice
                st.rerun()

        st.divider()
        st.subheader("Dynamic Portfolio: Daily Top 10 (All Business Dates)")
        port_c1, port_c2 = st.columns(2)
        with port_c1:
            portfolio_top_n = st.slider("Portfolio Top-N", min_value=3, max_value=20, value=10, step=1, key="portfolio_top_n")
        with port_c2:
            selection_lookback = st.slider(
                "Selection Lookback (days)", min_value=20, max_value=120, value=60, step=5, key="selection_lookback_days"
            )
        portfolio_df, portfolio_meta = build_rotating_top_portfolio_all_business_dates(
            ranked_df,
            trading_date,
            cost_bps=rank_cost_bps,
            top_n=portfolio_top_n,
            selection_lookback=selection_lookback,
        )
        if portfolio_df.empty:
            st.warning("Unable to build rotating portfolio with available data.")
        else:
            portfolio_stats = summarize_performance(portfolio_df["Portfolio_Return"], "Daily Top-N Optimal Portfolio")
            benchmark_stats = summarize_performance(portfolio_df["Benchmark_Return"], "Daily Top-N Buy & Hold")
            portfolio_ir = information_ratio(portfolio_df["Portfolio_Return"], portfolio_df["Benchmark_Return"])

            pcol1, pcol2, pcol3, pcol4 = st.columns(4)
            with pcol1:
                st.metric("Portfolio Return (%)", f"{portfolio_stats['Annual Return (%)']:.2f}")
            with pcol2:
                st.metric("Portfolio Risk (%)", f"{portfolio_stats['Annual Vol (%)']:.2f}")
            with pcol3:
                st.metric("Portfolio Drawdown (%)", f"{portfolio_stats['Max Drawdown (%)']:.2f}")
            with pcol4:
                st.metric("Portfolio Sharpe", f"{portfolio_stats['Sharpe']:.2f}")

            st.caption(
                f"Constituents used: {portfolio_meta.get('n_symbols', 0)} | "
                f"Total trades: {portfolio_meta.get('total_trades', 0)} | "
                f"Avg trades/symbol: {portfolio_meta.get('avg_trades_per_symbol', 0.0):.1f} | "
                f"Avg holdings/day: {portfolio_meta.get('avg_holdings', 0.0):.1f}"
            )
            daily_rank_df = portfolio_meta.get("daily_ranking_df", pd.DataFrame())
            if not daily_rank_df.empty:
                latest_rank_date = daily_rank_df["Date"].max()
                st.caption(f"Latest business-date ranking: {latest_rank_date}")
                latest_daily_top = (
                    daily_rank_df[daily_rank_df["Date"] == latest_rank_date]
                    .sort_values("Rank")
                    .reset_index(drop=True)
                )
                st.dataframe(
                    latest_daily_top[["Rank", "Symbol", "Score"]].round(4),
                    use_container_width=True
                )
                with st.expander("View Full Daily Top-N Ranking History"):
                    st.dataframe(
                        daily_rank_df.sort_values(["Date", "Rank"], ascending=[False, True]).round(4),
                        use_container_width=True
                    )

            portfolio_stats_df = pd.DataFrame([portfolio_stats, benchmark_stats]).round(2)
            portfolio_stats_df["Trades"] = [portfolio_meta.get("total_trades", 0), 0]
            portfolio_stats_df["IR vs BuyHold"] = [portfolio_ir, np.nan]
            st.dataframe(portfolio_stats_df, use_container_width=True)

            port_col1, port_col2 = st.columns(2)
            with port_col1:
                fig_port_daily = px.bar(
                    portfolio_df,
                    x=portfolio_df.index,
                    y="Daily_PnL",
                    title=f"Daily Top-{portfolio_top_n} Portfolio PnL (%)",
                    template="plotly_dark",
                    labels={"x": "Date", "Daily_PnL": "PnL (%)"},
                )
                fig_port_daily.update_layout(xaxis_title=None)
                st.plotly_chart(fig_port_daily, use_container_width=True)

            with port_col2:
                fig_port_cum = px.line(
                    portfolio_df,
                    x=portfolio_df.index,
                    y=["Portfolio_CumPnL", "Benchmark_CumPnL"],
                    title=f"Daily Top-{portfolio_top_n} Portfolio Cumulative PnL",
                    template="plotly_dark",
                    labels={"value": "Cumulative PnL", "variable": "Series"},
                )
                fig_port_cum.update_layout(xaxis_title=None, hovermode="x unified")
                st.plotly_chart(fig_port_cum, use_container_width=True)

else:
    st.info("Please select a date and click 'Get Data' in the sidebar to begin.")
