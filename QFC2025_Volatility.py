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

# --- PAGE SETUP ---
st.set_page_config(page_title="Nifty Maturity Hub", layout="wide")
CACHE_DIR = "data_cache"
if not os.path.exists(CACHE_DIR) : os.makedirs(CACHE_DIR)
    
# --- INITIALIZE SESSION STATE ---
# This prevents the AttributeError
if 'active_df' not in st.session_state: st.session_state.active_df = None


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
        return pd.read_csv(local_filename)

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
                    df = pd.read_csv(f)
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

# Example Usage:
# rv, price = get_realized_vol("^NSEI", "2026-01-05")
# print(f"Nifty RV: {rv:.2f}%, Close: {price:.2f}")
# --- SIDEBAR & SESSION STATE ---
if 'active_df' not in st.session_state:
    st.session_state.active_df = None

with st.sidebar:
    st.header("📅 Data Controls")
    trading_date = st.date_input("Trading Day", value=datetime(2025, 12, 31))
    symbol = st.selectbox("Index", ["NIFTY", "BANKNIFTY"])
    yf_tickers={'NIFTY':'^NSEI',
                'BANKNIFTY':'^NSEBANK'}
    yf_symbol = yf_tickers[symbol]
    # TRIGGER BUTTON
    if st.button("🚀 Get Data", width="stretch"):
        with st.spinner("Fetching from NSE Archives..."):
            result = get_nifty_data(trading_date)
            if isinstance(result, pd.DataFrame):
                st.session_state.active_df = result
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
    
    # Filter for Symbol
    df_symbol = df[df['SYMBOL'] == symbol].copy()
    
    with tab3:
        rv, _ = get_realized_vol('^NSEI', trading_date)
        vix = get_india_vix(trading_date)
        vrp_df = vix.join(rv.rename('^NSEI'), how='inner').dropna()
        st.subheader('NIFTY - Volatility Risk Premium (VIX vs 30D Realized Volatility)')
        col1, col2= st.columns(2)
        
        with col1:
            # Plot 1: Close Prices
            # Pass both columns as a list to y
            with st.container(border=True):
                if vrp_df.empty:
                    st.warning("Not enough VIX/price history to compute VRP for this date.")
                else:
                    fig_vol = px.line(
                        vrp_df,
                        x=vrp_df.index,
                        y=["^INDIAVIX", "^NSEI"],
                        title="Implied (VIX) vs Realized Volatility",
                        labels={"value": "Volatility (%)", "variable": "Type"},
                        template="plotly_dark"
                    )

                    # Customize the lines for professional look
                    fig_vol.update_traces(line=dict(width=2))
                    fig_vol.update_layout(xaxis_title=None, hovermode="x unified")

                    st.plotly_chart(fig_vol, use_container_width=True)
        with col2:
             # Plot 1: Close Prices
             # Pass both columns as a list to y
             with st.container(border=True):
                 if vrp_df.empty:
                     st.warning("VRP series is unavailable for the selected window.")
                 else:
                     fig_vol = px.line(
                         vrp_df,
                         x=vrp_df.index,
                         y=vrp_df["^INDIAVIX"] - vrp_df["^NSEI"],
                         title="Volatility Risk Premium",
                         labels={"value": "Volatility (%)", "variable": "Type"},
                         template="plotly_dark"
                     )

                     # Customize the lines for professional look
                     fig_vol.update_traces(line=dict(width=2))
                     fig_vol.update_layout(xaxis_title=None, hovermode="x unified", yaxis_title="Volatility (%)")
                     st.plotly_chart(fig_vol, use_container_width=True)
    with tab2:
        all_strikes = sorted(df_symbol['STRIKE'].unique())
        col_select, col_empty = st.columns([2, 4])
        with col_select:
            selected_strike = st.selectbox("Select Strike", all_strikes)
        st.divider()
        # 2. Final Filtered Data
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
        all_expiries = sorted(df_symbol['EXPIRY'].unique())
        col_select, col_empty = st.columns([2, 4])
        with col_select:
            selected_expiry = st.selectbox("Select Maturity", all_expiries)
        st.divider()
        # 2. Final Filtered Data
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
    with tab2:
        # --- Move your existing col1, col2, and fig_pdf code here ---
        st.write("Smile Analysis")
        # ... (Your existing code)

else:
    st.info("Please select a date and click 'Get Data' in the sidebar to begin.")
