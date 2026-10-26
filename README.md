# Finmodel JS Public Edition

A static, JavaScript-only browser application intended for zero-monthly-cost deployment on a free static hosting tier.

## Deploy to Cloudflare Pages
1. Upload this folder's files to the root of a GitHub repository.
2. In Cloudflare Pages, create a project connected to that repository.
3. Framework preset: **None**.
4. Build command: leave blank.
5. Build output directory: `/` (repository root).
6. Deploy and use the assigned `pages.dev` URL.

## Current browser features
- Dashboard and multi-asset watchlist
- CSV and Excel import (Excel import uses SheetJS CDN)
- Price charting and ticker analysis
- SMA20/SMA50, RSI14, annualized return/volatility, max drawdown
- Beta, Jensen alpha, active return, tracking error, information ratio
- Browser-side long-only portfolio weight search
- CAPM / CAL / SML calculations
- Financial ratios and classic Altman Z-score input calculator
- Five-year FCFF/DCF model, bear/base/bull scenarios and WACC/terminal-growth sensitivity
- CSV exports and Excel workbook export when SheetJS CDN is available

## Important limits
This is a **functional browser-side conversion starter**, not feature-for-feature parity with the Python v15 engine. XGBoost has not been ported; the interface labels its forecast status as a JavaScript approximation and the DCF is a simplified editable model. The existing Python code's peer analysis, news/macro feeds, all valuation logic, detailed statement extraction, and some advanced modules are not all reproduced here yet.

No API keys or backend are included. Yahoo Finance direct browser fetch is optional and may fail due to CORS, throttling, or provider changes. The reliable path is importing CSV/Excel files. Sample prices are illustrative only and clearly identified. Do not treat sample values as live market data.

The app uses the public SheetJS CDN for Excel import/export. CSV import/export works without it. External CDNs and data sources may change, be blocked, or impose their own limits.

All financial calculations are educational and informational, not investment advice. Browser local data is not uploaded to a server by this app.
