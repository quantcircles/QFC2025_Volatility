# QFC2025_Volatility
Streamlit portal to display volskew, term structure, risk neutral density of underlying, VRP of a particular option (European)

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
