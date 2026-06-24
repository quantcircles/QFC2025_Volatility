# QFC2025_Volatility
Streamlit portal to display volskew, term structure, risk neutral density of underlying, VRP of a particular option (European)

# Quantitative Derivatives Volatility Strategy Agent

An asynchronous Model Context Protocol (MCP) server built with `FastMCP` that dynamically analyzes volatility risk premium (VRP) strategies across Indian equities and indices (NSE). 

The agent automatically downloads derivatives market data (`Bhavcopy`), identifies highly liquid instruments by Open Interest (OI), models volatility profiles utilizing historical prices and `INDIAVIX`, optimizes historical backtest trading rules, and compiles an equal-weighted top-performing portfolio report.

---

## Architecture Overview

The system aggregates multiple data pipelines to construct, optimize, and serve daily performance reports:

* **Data Sourcing Pipeline:** Fetches tracking records from Yahoo Finance (`yfinance`) and downloads structural daily options market data directly from the National Stock Exchange of India (NSE).
* **Quantitative Engine:** Calculates a 30-day trailing Realized Volatility ($RV$) and pairs it alongside Implied Volatility ($IV$ from `^INDIAVIX`) to evaluate the Volatility Risk Premium (VRP):
    $$VRP = IV - RV$$
* **Grid Optimization:** Exhaustively tests lookback parameters and dynamic $Z$-score entry/exit thresholds per asset to minimize maximum drawdown while optimizing annualized returns.
* **MCP Infrastructure:** Implements a compliant fastmcp wrapper exposing specific backtesting engines as native Large Language Model tools and caching dynamic file reports.

---

## Directory Structure

Upon operational execution, the server establishes localized file states to prevent excessive upstream querying:

```text
├── data_cache/           # Cached daily NSE F&O Bhavcopy CSV files
├── reports/              # Performance summaries (Markdown format)
│   ├── daily_summary_YYYY-MM-DD.md
│   └── daily_summary_latest.md
├── main.py               # Application engine source code
└── README.md             # System documentation


## MCP Daily Summary Agent

Run MCP server:

```bash
python agents/daily_summary_mcp_agent.py
```

Exposed interfaces:

- Tool: `generate_daily_summary(trading_date, max_symbols, top_n, cost_bps)`
- Resource: `summary://latest`

Generated reports are saved in `reports/`:

- `reports/daily_summary_YYYY-MM-DD.md`
- `reports/daily_summary_latest.md`
