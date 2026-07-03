# Portfolio Analysis NSE Data Pull

This file documents the portfolio data-pull workflow centered on `portfolioAnalysis_SM.py`
and `nse_nifty_pull.py`.

## Purpose

The current workflow pulls NIFTY 50 stock data from official NSE archive sources:

- NIFTY 50 constituents from the NSE index constituent CSV.
- Official NIFTY 50 index closes from the NSE index close archive.
- FRED India macro series for daily economy score and state checks.
- Daily cash-market bhavcopy files from NSE archive zip files.
- Cleaned OHLCV-style rows for selected NIFTY 50 `EQ` symbols.
- NSE Financial Results corporate filing records.
- NSE financial-result announcement records used as a freshness cross-check.
- NSE Shareholding Pattern corporate filing records.
- Daily NIFTY fundamental score records.
- Daily NIFTY technical score records using conventional value/mean-reversion
  and momentum indicators computed from cached OHLCV price history.
- Daily strategy backtests and rebalance holdings for Fundamental, Technical,
  and Average Score strategies.

The output is suitable for downstream portfolio analysis, volatility research, and
backtesting pipelines.

## Main Files

- `portfolioAnalysis_SM.py`
  - Thin command-line entry point.
  - Delegates execution to `nse_nifty_pull.main()`.

- `nse_nifty_pull.py`
  - Contains the NSE archive client.
  - Pulls NIFTY 50 constituents.
  - Pulls official NIFTY 50 index close history for benchmark and regime checks.
  - Downloads FRED India macro series and builds economy score history.
  - Downloads and caches daily cash-market bhavcopy files.
  - Filters bhavcopy rows to NIFTY 50 equity symbols.
  - Downloads NSE Financial Results and Shareholding Pattern records.
  - Downloads financial-result announcements to detect newer filings when the
    formal financial-results endpoint lags.
  - Builds daily fundamental scores and carries forward unchanged scores.
  - Builds historical daily technical scores from cached price files.
  - Builds strategy backtest CSVs and rebalance holdings from cached score
    histories.
  - Writes aggregate CSV outputs or one CSV per trading day.

## Dependencies

The NSE pull currently uses:

- `pandas`
- `requests`

Install project dependencies with:

```bash
pip install -r requirements.txt
```

Depending on your local Python setup, you may need `pip3` instead of `pip`.

## Usage

Pull all NIFTY 50 symbols for the default date window, which is the last 30 calendar
days ending yesterday:

```bash
python3 portfolioAnalysis_SM.py
```

Pull all NIFTY 50 symbols for a specific date range:

```bash
python3 portfolioAnalysis_SM.py --from 2026-06-01 --to 2026-06-25
```

Pull only selected NIFTY 50 symbols:

```bash
python3 portfolioAnalysis_SM.py --from 2026-06-22 --to 2026-06-26 --symbols RELIANCE TCS INFY
```

Limit the number of symbols pulled:

```bash
python3 portfolioAnalysis_SM.py --from 2026-06-22 --to 2026-06-26 --limit 5
```

Write outputs to a custom directory:

```bash
python3 portfolioAnalysis_SM.py --from 2026-06-22 --to 2026-06-26 --output-dir data_cache/my_pull
```

Create one price CSV and one constituent CSV for each available trading day in the
last one year:

```bash
python3 portfolioAnalysis_SM.py --last-year --daily-files
```

Seed or refresh the official NIFTY 50 index close history used by the dashboard
regime panel and benchmark overlay:

```bash
python3 portfolioAnalysis_SM.py --index-history-only --last-year
python3 portfolioAnalysis_SM.py --index-history-only --from 2025-06-02 --to 2026-06-25
```

Seed or refresh the FRED India macro economy score history used by the dashboard
economy state overlay:

```bash
python3 portfolioAnalysis_SM.py --economy-history-only
```

Run on a subsequent day to pull the latest available previous close. This skips
weekends and holidays by looking back up to 10 calendar days:

```bash
python3 portfolioAnalysis_SM.py --previous-close
```

Increase the holiday/weekend lookback window if needed:

```bash
python3 portfolioAnalysis_SM.py --previous-close --fallback-days 20
```

Download only fundamentals for the current NIFTY 50 list:

```bash
python3 portfolioAnalysis_SM.py --fundamentals-only
```

The Financial Results pull downloads both `Quarterly` and `Annual` filings. By
default it looks back 8 years:

```bash
python3 portfolioAnalysis_SM.py --fundamentals-only --fundamentals-years 10
```

Download previous-close price files and fundamentals in one run:

```bash
python3 portfolioAnalysis_SM.py --previous-close --fundamentals
```

Choose the fundamentals source workflow:

```bash
python3 portfolioAnalysis_SM.py --fundamentals --fundamentals-source auto
python3 portfolioAnalysis_SM.py --fundamentals --fundamentals-source screener
python3 portfolioAnalysis_SM.py --fundamentals --fundamentals-source moneycontrol
python3 portfolioAnalysis_SM.py --fundamentals --fundamentals-source economictimes
python3 portfolioAnalysis_SM.py --fundamentals --fundamentals-source cached
```

`auto` tries Screener fundamentals first, then Moneycontrol, then Economic
Times, and finally the latest cached fundamentals if live sources block the
runner. `screener` builds normalized quarterly, annual, and shareholding input
files from Screener tables. `moneycontrol` and `economictimes` remain fallback
financial-table sources. `cached` skips live fundamentals and uses the latest
cached provider files.

Generate daily fundamental scores after downloading fundamentals:

```bash
python3 portfolioAnalysis_SM.py --fundamentals-only --fundamental-scores
```

Download fundamentals and generate only the score file:

```bash
python3 portfolioAnalysis_SM.py --scores-only
```

Generate historical fundamental scores from cached fundamentals and cached
business-day price files:

```bash
python3 portfolioAnalysis_SM.py --score-history-only
```

Generate historical technical scores from cached daily price files:

```bash
python3 portfolioAnalysis_SM.py --technical-score-history-only
```

Generate strategy backtests from cached fundamental and technical score history:

```bash
python3 portfolioAnalysis_SM.py --backtests
```

Generate only the modelled NIFTY 50 buy-write backtest from cached official index
history:

```bash
python3 portfolioAnalysis_SM.py --buy-write-backtest
```

Daily end-to-end run, including latest price/fundamental refresh, score refresh,
strategy backtests, holdings, and standalone dashboard HTML:

```bash
python3 portfolioAnalysis_SM.py --previous-close --fallback-days 14 --index-history --economy-history --fundamentals --fundamentals-source auto --fundamental-scores --fundamental-score-history --technical-scores --backtests --dashboard-html
```

Generate only the latest and datestamped dashboard HTML from cached histories and
backtests:

```bash
python3 portfolioAnalysis_SM.py --dashboard-html
```

## GitHub Actions Automation

The daily GitHub Actions workflow in `.github/workflows/daily-summary.yml` runs on
Indian market weekdays after the regular NSE close. As part of that run, it calls:

```bash
python portfolioAnalysis_SM.py --previous-close --fallback-days 14 --index-history --economy-history --fundamentals --fundamentals-source auto --fundamental-scores --fundamental-score-history --technical-scores --backtests --dashboard-html
```

This writes the latest available previous-close NIFTY 50 files and official
NIFTY 50 index close into `data_cache/nse_equity/`, and writes Financial Results
and Shareholding Pattern CSVs into `data_cache/nse_equity/fundamentals/`. It also
writes financial-result announcement CSVs, a daily fundamental score CSV, and
technical score CSVs computed from the full cached price history. It then
generates the strategy backtest, holdings CSVs, and modelled NIFTY 30-day
buy-write CSVs under
`data_cache/nse_equity/backtests/`. The workflow
also writes `fundamental_score_dashboard.html` and a datestamped dashboard under
`dashboards/`. It then commits generated report files, NSE data files, and
dashboard files back to the repository when anything changed.

NSE can return `403 Forbidden` to GitHub-hosted runners for the live corporate
filing endpoints. When that happens, the script falls back to the latest cached
fundamentals under `data_cache/nse_equity/fundamentals/` so the daily price,
score, backtest, and dashboard pipeline can still complete.

## Outputs

By default, files are written under:

```text
data_cache/nse_equity/
```

Generated files include:

- `nifty50_constituents_YYYYMMDD_HHMMSS.csv`
  - Latest NIFTY 50 constituent list from NSE archives.

- `nifty50_prices_STARTDATE_ENDDATE_YYYYMMDD_HHMMSS.csv`
  - Cleaned price data for the selected symbols and date range.

- `bhavcopy_cache/bhavcopy_cm_YYYYMMDD.csv`
  - Cached raw cash-market bhavcopy data for each downloaded trading date.

When `--daily-files` is used, generated files also include:

- `prices_by_day/nifty50_prices_YYYYMMDD.csv`
  - One cleaned price file per available trading day.

- `constituents_by_day/nifty50_constituents_YYYYMMDD.csv`
  - One constituent file per available trading day.

When `--index-history` or `--index-history-only` is used, generated files include:

- `index_by_day/nifty50_index_YYYYMMDD.csv`
  - One official NIFTY 50 index close row per available trading day.

- `index_history/nifty50_index_history_STARTDATE_ENDDATE.csv`
  - Combined official NIFTY 50 index history used for dashboard regime
    classification and benchmark overlays.

When `--economy-history` or `--economy-history-only` is used, generated files
include:

- `economy/economic_variables_history/fred_india_economic_variables_YYYYMMDD.csv`
  - Raw FRED observations, transformed YoY/level values, component scores,
    weights, and source metadata for CPI, industrial production, exports,
    short-term interest rate, and GDP.

- `economy/economy_score_history/fred_india_economy_scores_history_STARTDATE_ENDDATE.csv`
  - Daily economy score history aligned to cached NIFTY index dates, including
    state labels, all component score values, observation dates, and component
    age in calendar days. Stale components are retained for auditability but
    excluded from the composite score after their configured freshness window.

When `--fundamentals` or `--fundamentals-only` is used, generated files include:

- `fundamentals/financial_results_by_day/screener_financial_results_YYYYMMDD.csv`
  - Screener quarterly and annual financial table records for the selected
    NIFTY 50 symbols.
  - If Screener fails and a fallback succeeds, the equivalent file is
    `moneycontrol_financial_results_YYYYMMDD.csv` or
    `economictimes_financial_results_YYYYMMDD.csv`.

- `fundamentals/financial_result_announcements_by_day/screener_financial_result_announcements_YYYYMMDD.csv`
  - Provider metadata rows used by the fundamental score recency logic.
  - If Screener fails and Economic Times succeeds, the equivalent file is
    `economictimes_financial_result_announcements_YYYYMMDD.csv`.

- `fundamentals/shareholding_by_day/screener_shareholding_pattern_YYYYMMDD.csv`
  - Screener shareholding table records with promoter and public holding values
    mapped into `pr_and_prgrp` and `public_val`.

- `fundamentals/screener_raw_by_day/screener_raw_YYYYMMDD.csv`
  - Raw Screener fetch metadata and parse diagnostics. Moneycontrol and
    Economic Times write equivalent files under their own raw directories.

- `fundamentals/fundamental_scores_by_day/screener_fundamental_scores_YYYYMMDD.csv`
  - One fundamental score row per selected NIFTY 50 symbol.

- `fundamentals/fundamental_scores_history/screener_fundamental_scores_history_STARTDATE_ENDDATE.csv`
  - Combined historical fundamental score file across cached business days.

When `--technical-score-history` or `--technical-score-history-only` is used,
generated files include:

- `technicals/technical_scores_by_day/nse_technical_scores_YYYYMMDD.csv`
  - One technical score row per selected symbol for each cached business day.

- `technicals/technical_scores_history/nse_technical_scores_history_STARTDATE_ENDDATE.csv`
  - Combined historical technical score file across cached business days.

When `--backtests` is used, generated files include:

- `backtests/fundamental_top10_score_weighted_backtest.csv`
  - Daily return, cumulative return, portfolio value, and benchmark columns for
    the Fundamental Score strategy.

- `backtests/technical_top10_score_weighted_backtest.csv`
  - Same fields for the Technical Score strategy.

- `backtests/average_top10_score_weighted_backtest.csv`
  - Same fields for the Average Score strategy, where strategy score is
    `(fundamental_score + technical_score) / 2`.

- `backtests/optimized_average_top10_score_weighted_backtest.csv`
  - Same fields for the Optimized Average strategy. At every rebalance date it
    selects the 10 highest Average Score names, then optimizes weights using
    trailing returns, covariance risk, and score alpha with long-only weight
    bounds.

- `backtests/*_top10_score_weighted_holdings.csv`
  - Rebalance holdings for each strategy, including score, weight, integer
    previous quantity, integer target quantity, integer quantity change, entry
    close, end close, holding calendar days, and holding-period return.

- `backtests/*_score_weighted_symbol_metrics.csv`
  - Per-symbol return/risk/Sharpe/drawdown metrics used to rank strategy
    candidates.

- `backtests/selected_top10_score_weighted_symbol_metrics.csv`
  - Combined top-10 symbol metrics across all strategies.

- `backtests/nifty_buy_write_30d_backtest.csv`
  - Closed 30-calendar-day NIFTY buy-write backtest periods, strategy value,
    benchmark value, and cumulative returns.

- `backtests/nifty_buy_write_30d_trades.csv`
  - Trade-level NIFTY buy-write replication details: entry/exit dates, NIFTY
    closes, strike, trailing volatility, modelled call premium, call payoff,
    index return, strategy return, and cumulative return.

When `--dashboard-html` is used, generated files include:

- `fundamental_score_dashboard.html`
  - Latest standalone dashboard HTML with embedded score history and strategy
    backtest data.

- `dashboards/fundamental_score_dashboard_YYYYMMDD.html`
  - Datestamped standalone dashboard HTML, where `YYYYMMDD` is the latest score
    history date embedded in the dashboard.

## Price Output Columns

The cleaned price file contains the available columns from the NSE cash-market
bhavcopy, normalized to readable names:

- `symbol`
- `trade_date`
- `business_date`
- `series`
- `isin`
- `instrument_name`
- `open`
- `high`
- `low`
- `close`
- `last`
- `previous_close`
- `settlement_price`
- `volume`
- `turnover`
- `trades`

## Fundamentals Output

The Financial Results CSV stores the raw fields returned by NSE's corporate
filings endpoint, plus:

- `symbol`
- `pulled_on`
- `from_date`
- `to_date`
- `pulled_at`

The Shareholding Pattern CSV follows the same convention. NSE fields can vary
over time as exchange filing formats change, so the downloader preserves all
available columns instead of forcing a narrow schema.

## Fundamental Score Output

The daily score file contains:

- `fundamental_score`
- `computed_score`
- `previous_score`
- `score_changed`
- `score_source`
- component scores for filing recency, disclosure, consistency, shareholding
  recency, and promoter/public ownership fields.

If a computed score is unchanged from the previous available score file, the
daily row carries forward the prior `fundamental_score` and marks
`score_changed` as `False`. If the computed score changes, the new score is used
and `score_changed` is `True`.

The first-pass score is based on NSE filing metadata, not fully parsed XBRL line
items. It currently weights:

- Quarterly filing recency, using financial-result announcements when they are
  newer than the formal Financial Results endpoint.
- Annual filing recency, using financial-result announcements when they are
  newer than the formal Financial Results endpoint.
- XBRL/new-format disclosure availability.
- Filing history consistency.
- Shareholding-pattern recency.
- Latest promoter/public ownership fields.

## Technical Score Output

The daily technical score file is designed to be independently reproducible from
the CSV itself. It includes the final score, component scores, raw indicators,
cross-sectional ranks, and weights used in the formula.

Top-level score columns:

- `technical_score`
- `value_score_0_50`
- `momentum_score_0_50`

Raw price and indicator columns include:

- OHLCV fields: `open`, `high`, `low`, `close`, `previous_close`, `volume`,
  `turnover`, `trades`
- Momentum returns: `return_21d_pct`, `return_63d_pct`, `return_126d_pct`,
  `return_252d_pct`
- Trend indicators: `sma_20`, `sma_50`, `sma_200`, `sma50_over_sma200_pct`,
  `close_vs_sma20_pct`
- Liquidity/participation: `volume_sma_20`, `volume_vs_sma20_pct`
- Value/mean-reversion indicators: `rolling_high_252`, `rolling_low_252`,
  `discount_to_252d_high_pct`, `premium_to_252d_low_pct`,
  `discount_to_sma200_pct`, `bollinger_pct_b`, `rsi_14`

The value half of the score contributes 0 to 50 points:

```text
value_score_0_50 =
  (0.40 * value_discount_high_rank
 + 0.25 * value_discount_sma200_rank
 + 0.20 * value_rsi_rank
 + 0.15 * value_bollinger_rank) / 2
```

The momentum half contributes 0 to 50 points:

```text
momentum_score_0_50 =
  (0.35 * momentum_21d_rank
 + 0.35 * momentum_63d_rank
 + 0.20 * momentum_126d_rank
 + 0.10 * momentum_trend_rank) / 2
```

The final score is:

```text
technical_score = value_score_0_50 + momentum_score_0_50
```

Rank columns are daily cross-sectional percentile ranks across the available
symbols for that business day. Higher ranks are better. Missing indicator ranks
are filled with a neutral value of 50 during score calculation; the raw missing
indicator values remain blank in the CSV.

## Strategy Backtests

`--backtests` reads the latest cached fundamental score history and technical
score history. It does not require NSE network access when those history files
already exist.

Generated strategies:

- `fundamental`
  - Selects the 10 symbols with the highest realized score-weighted Sharpe from
    the available history.
  - Daily exposure is based on `fundamental_score / 100`.

- `technical`
  - Selects the 10 symbols with the highest realized score-weighted Sharpe from
    the available history.
  - Daily exposure is based on `technical_score / 100`.

- `average`
  - Uses `(fundamental_score + technical_score) / 2` as the strategy score.
  - Selects the 10 symbols with the highest realized score-weighted Sharpe from
    the available history.

- `nifty_buy_write_30d`
  - Uses official cached NIFTY 50 index closes.
  - Buys the index and sells a modelled at-the-money call for each closed
    30-calendar-day holding period.
  - Uses Black-Scholes premium estimates from trailing 30-day realized
    volatility because historical option-chain premiums are not cached.

All strategy backtests start with `100000` initial capital and apply `0.01%`
transaction cost on exposure or weight turnover. Benchmark columns use the
official NSE NIFTY 50 index close history when cached, with an equal-weight
NIFTY constituent proxy fallback if index history is unavailable.
Portfolio holdings for Fundamental, Technical, and Average Score strategies are
rebalanced on 31-calendar-day windows. Optimized Average uses a 93-calendar-day
optimization/rebalance window so weights are refreshed roughly every three
months. Each block is priced on the last available cached trading date inside
the calendar window.
The buy-write backtest does not apply equity turnover transaction costs; its
CSV focuses on option premium, option payoff, index return, and strategy return
replication.

## HTML Dashboard

`fundamental_score_dashboard.html` is a standalone browser dashboard generated
from the cached price, fundamental score, and technical score history files. It
shows a top-level NIFTY 50 index panel using official NSE index closes, SMA50,
SMA200, 63-trading-day return, and RSI14 to label the market as bull, bear, or
range-bound.
It embeds the required data directly and lets you choose:

- NIFTY symbol
- Fundamental variable
- Technical indicator

The chart shows close price, fundamental score, technical score, the selected
fundamental variable, and the selected technical indicator on a normalized
0-100 comparison axis. Hovering the chart shows raw values for each series.

The second dashboard tab shows the modelled NIFTY 30-calendar-day buy-write
strategy. It plots buy-write and NIFTY period returns as bars, cumulative
buy-write and NIFTY returns as lines, summary performance/risk metrics, and the
full trade table.

The buy-write backtest uses cached official NIFTY 50 index closes. Because this
repository does not currently cache historical NIFTY option-chain prices, the
short call premium is modelled with Black-Scholes using trailing 30-day realized
NIFTY volatility, a 6.5% risk-free rate, and a 1.2% dividend yield. The CSV
includes all premium and payoff inputs so the strategy return can be replicated.

## Current Behavior

- Missing bhavcopy archive files, such as weekends or NSE holidays, are skipped.
- Raw daily bhavcopy files are cached locally to reduce repeated downloads.
- Only NIFTY 50 symbols listed in the constituent CSV are pulled.
- Only `EQ` series rows are retained from the cash-market bhavcopy.
- `--previous-close` writes daily files and selects the latest bhavcopy available
  on or before yesterday.
- `--last-year` uses the 365 calendar days ending on `--to`, or yesterday when
  `--to` is not supplied.
- Daily constituent files currently use the NSE archive constituent CSV available
  at run time. The workflow does not reconstruct historical index membership.
- Fundamentals files use NSE corporate filing APIs for Financial Results and
  Shareholding Pattern. They are filing records, not normalized accounting ratios.
- Financial-result announcements are used as a freshness overlay because the
  formal NSE Financial Results endpoint can lag for some symbols.
- Fundamental score files are generated daily. Unchanged scores are carried
  forward from the latest prior score file.
- Technical score history is generated from cached daily price files. It does
  not require NSE network access when `prices_by_day/` already exists.
- Technical-score rows follow the symbols available in each cached daily price
  file; a symbol absent from a source price file is absent from that day's
  technical score file.
- Price files use NSE archive CSVs. Fundamentals files use NSE corporate filing
  JSON endpoints because those records are not available in the same archive
  format.

## Verification

Basic syntax check:

```bash
python3 -m py_compile nse_nifty_pull.py portfolioAnalysis_SM.py
```

Small data-pull smoke test:

```bash
python3 portfolioAnalysis_SM.py --from 2026-06-22 --to 2026-06-26 --symbols RELIANCE TCS --limit 2
```

Daily-file smoke test:

```bash
python3 portfolioAnalysis_SM.py --from 2026-06-22 --to 2026-06-26 --symbols RELIANCE TCS --daily-files
```

Fundamentals smoke test:

```bash
python3 portfolioAnalysis_SM.py --fundamentals-only --symbols RELIANCE --limit 1
```

Fundamental score smoke test:

```bash
python3 portfolioAnalysis_SM.py --scores-only --symbols RELIANCE --limit 1
```

Technical score history smoke test:

```bash
python3 portfolioAnalysis_SM.py --technical-score-history-only --symbols RELIANCE TCS
```

NIFTY 50 index history smoke test:

```bash
python3 portfolioAnalysis_SM.py --index-history-only --previous-close --fallback-days 14
```

Backtest generation smoke test from cached score histories:

```bash
python3 portfolioAnalysis_SM.py --backtests
```

Dashboard HTML smoke test from cached histories and backtests:

```bash
python3 portfolioAnalysis_SM.py --dashboard-html
```

Expected result:

- A constituent CSV with 50 rows.
- A price CSV with rows for available trading dates in the requested range.
- With `--daily-files`, one price CSV and one constituent CSV for each available
  trading day.
- With `--fundamentals-only`, one Financial Results CSV and one Shareholding
  Pattern CSV, plus one financial-result announcements CSV.
- With `--scores-only`, one Financial Results CSV, one Shareholding Pattern CSV,
  one financial-result announcements CSV, and one Fundamental Score CSV.
- With `--technical-score-history-only`, one technical score CSV per cached
  business day and one combined technical score history CSV.
- With `--index-history-only`, one official NIFTY 50 index CSV per available
  trading day and one combined index history CSV.
- With `--backtests`, three strategy backtest CSVs, three strategy holdings CSVs,
  three per-strategy symbol metrics CSVs, and one combined selected top-10 metrics
  CSV under `data_cache/nse_equity/backtests/`, plus the NIFTY buy-write trade
  and backtest CSVs.
- With `--buy-write-backtest`, one modelled NIFTY 30-calendar-day buy-write
  trade CSV and one matching backtest CSV.
- With `--dashboard-html`, the latest standalone HTML dashboard and one
  datestamped copy under `dashboards/`.

## Documentation Maintenance

Keep this file updated whenever code changes affect:

- CLI arguments or example commands.
- Data sources or NSE URL patterns.
- Output directories, filenames, or schemas.
- Filtering rules, such as symbol selection or equity series handling.
- Required dependencies.
- Error handling, caching behavior, or holiday/weekend behavior.

When making future changes to `portfolioAnalysis_SM.py` or `nse_nifty_pull.py`,
update this README in the same change so the workflow description stays aligned
with the code.
