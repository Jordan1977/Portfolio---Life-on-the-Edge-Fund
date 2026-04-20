# Life on the Hedge Fund

**Institutional Portfolio Analytics Terminal**  
Trinity College Dublin · Investment Analysis · Academic Portfolio

> Live dashboard: [jordan1977.github.io/life-on-the-hedge-fund-dashboard](https://jordan1977.github.io/life-on-the-hedge-fund-dashboard/)

---

## Portfolio Positioning

| Attribute | Value |
|---|---|
| **Name** | Life on the Hedge Fund |
| **Institution** | Trinity College Dublin · #1 in Ireland |
| **Context** | Academic portfolio analytics project |
| **Universe** | Concentrated US equity |
| **Initial AUM** | ~$50,000 USD |
| **Benchmark** | Nasdaq-100 via QQQ |
| **Secondary ref.** | S&P 500 via SPY |
| **Rebalancing** | None since inception (2025-03-06) |
| **Horizon** | 15 years |
| **Style** | High-conviction · high-beta · thematic growth |

---

## Holdings

| Ticker | Name | Sector | Risk Bucket |
|---|---|---|---|
| NVDA | Nvidia Corporation | AI / Semiconductors | GROWTH |
| GOOGL | Alphabet Inc. | AI / Tech Platform | CORE |
| PLTR | Palantir Technologies | AI / Defence Tech | GROWTH |
| APP | AppLovin Corporation | AI / AdTech | GROWTH |
| SOUN | SoundHound AI Inc. | AI / Voice | SPECULATIVE |
| RTX | RTX Corporation | Defense / Aerospace | CORE |
| RKLB | Rocket Lab USA | Space Economy | SPECULATIVE |
| GEV | GE Vernova Inc. | Energy Transition | GROWTH |
| COIN | Coinbase Global Inc. | Crypto Infrastructure | SPECULATIVE |
| MARA | MARA Holdings Inc. | Bitcoin Mining | SPECULATIVE |
| HOOD | Robinhood Markets Inc. | Fintech / Retail | SPECULATIVE |
| UBER | Uber Technologies Inc. | Mobility / Platform | GROWTH |
| RDDT | Reddit Inc. | Social / AI Data | SPECULATIVE |

---

## Architecture

```
Python = single source of truth
         ↓
build_dashboard.py
  ├── loads holdings.csv
  ├── downloads prices via yfinance
  ├── computes all metrics in Python
  ├── renders charts with Plotly
  └── writes docs/index.html  ← fully static HTML, no client-side fetching
         ↓
GitHub Actions (3× daily on weekdays)
         ↓
GitHub Pages → static HTML served publicly
```

No JavaScript data fetching. The browser only renders what Python already computed.

---

## What the Dashboard Includes

### KPI Overview
Current NAV · Total P&L · Total Return · Daily P&L · Alpha vs QQQ · Beta · Sharpe · Sortino · Max Drawdown · Annualised Volatility · Information Ratio · Hit Ratio

### Performance Charts
- Portfolio vs QQQ vs SPY — Base 100
- Drawdown from peak
- Monthly returns bar + QQQ overlay
- Rolling 30-day volatility
- Rolling 30-day beta
- Rolling 30-day Sharpe
- Monthly return heatmap

### Position Monitor
Full position table with: buy price · latest price · market value · P&L · return · weight · contribution · 1D / 5D / 1M performance · individual beta vs benchmark

### Full Metrics Table
Sharpe · Sortino · Calmar · Jensen Alpha · Treynor · Omega · Tracking Error · Information Ratio · VaR 95% · CVaR 95% · Upside / Downside Capture · Skewness · Kurtosis · Hit Ratio

### Structure & Allocation
Sector breakdown table · Sector donut · Thematic donut · Concentration (HHI · Effective N · Top 5 weight)

### Stress Test
Six scenario shocks mapped through portfolio beta and sector exposures, with P&L impact, return impact, and implied NAV after shock.

### Scenario Projections
600-path Monte Carlo using the portfolio's own historical return distribution. Bull / base / bear path overlays. Projection summary table for 3M / 6M / 12M / 15Y horizons. Clearly labelled as model-based envelopes, not forecasts.

### Portfolio Intelligence
Auto-generated narrative: Portfolio DNA · What worked · Risk lens · Benchmark lens · Concentration lens.

### News Flow
Latest headlines per ticker, sourced from Yahoo Finance at build time. Non-blocking — dashboard builds successfully even if news retrieval fails.

---

## Repository Structure

```
.
├── build_dashboard.py           ← single source of truth
├── holdings.csv                 ← portfolio definition
├── requirements.txt
├── README.md
├── docs/
│   └── index.html               ← generated static dashboard
├── data/
│   └── dashboard_snapshot.json  ← generated JSON snapshot
└── .github/
    └── workflows/
        └── update-dashboard.yml ← GitHub Actions workflow
```

---

## Local Setup

```bash
git clone https://github.com/Jordan1977/life-on-the-hedge-fund-dashboard.git
cd life-on-the-hedge-fund-dashboard
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python build_dashboard.py
# open docs/index.html in browser
```

---

## GitHub Pages Deployment

1. Push all files to the `main` branch.
2. Go to **Settings → Pages** in your repository.
3. Under **Source**, select **GitHub Actions**.
4. The workflow will build and deploy automatically on every push and on schedule.

Your public URL:

```
https://jordan1977.github.io/life-on-the-hedge-fund-dashboard/
```

---

## Automation Schedule

The workflow runs:
- On every `push` to `main`
- Three times daily on weekdays: **07:00 · 13:00 · 19:00 UTC**
- On manual trigger via **Actions → Run workflow**

---

## Disclaimer

This is a fictional academic portfolio. It is designed to demonstrate portfolio construction analysis, risk reporting, benchmark-relative thinking, and dashboard engineering. It is not investment advice.
