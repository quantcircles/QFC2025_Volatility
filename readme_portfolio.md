# Portfolio Analysis NSE Data Pull

This file documents the portfolio data-pull workflow centered on `portfolioAnalysis_SM.py`
and `nse_nifty_pull.py`.

## Purpose

The current workflow pulls NIFTY 50 stock data from official NSE archive sources:

- NIFTY 50 constituents from the NSE index constituent CSV.
- Daily cash-market bhavcopy files from NSE archive zip files.
- Cleaned OHLCV-style rows for selected NIFTY 50 `EQ` symbols.

The output is suitable for downstream portfolio analysis, volatility research, and
backtesting pipelines.

## Main Files

- `portfolioAnalysis_SM.py`
  - Thin command-line entry point.
  - Delegates execution to `nse_nifty_pull.main()`.

- `nse_nifty_pull.py`
  - Contains the NSE archive client.
  - Pulls NIFTY 50 constituents.
  - Downloads and caches daily cash-market bhavcopy files.
  - Filters bhavcopy rows to NIFTY 50 equity symbols.
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

Run on a subsequent day to pull the latest available previous close. This skips
weekends and holidays by looking back up to 10 calendar days:

```bash
python3 portfolioAnalysis_SM.py --previous-close
```

Increase the holiday/weekend lookback window if needed:

```bash
python3 portfolioAnalysis_SM.py --previous-close --fallback-days 20
```

## GitHub Actions Automation

The daily GitHub Actions workflow in `.github/workflows/daily-summary.yml` runs on
Indian market weekdays after the regular NSE close. As part of that run, it calls:

```bash
python portfolioAnalysis_SM.py --previous-close --fallback-days 14
```

This writes the latest available previous-close NIFTY 50 files into
`data_cache/nse_equity/`. The workflow then commits both generated report files
and NSE data files back to the repository when anything changed.

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
- The NSE live JSON API is not used for this workflow because archive files are
  more stable for repeatable research pulls.

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

Expected result:

- A constituent CSV with 50 rows.
- A price CSV with rows for available trading dates in the requested range.
- With `--daily-files`, one price CSV and one constituent CSV for each available
  trading day.

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
