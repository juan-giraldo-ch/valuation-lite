# Valuation Lite — phone-ready Streamlit app

This standalone mobile app reuses your original sector-specific stock valuation methodology.

## Deploy in your browser — no Python installation needed

1. Extract this ZIP to a folder on your computer.
2. On https://github.com, create a **private** repository, e.g. `valuation-lite`.
3. Use **Add file > Upload files** to upload all extracted files to the repository root (not the ZIP as a single file), then commit. Check that `mobile_app.py` and `requirements.txt` appear in the root.
4. Sign in at https://share.streamlit.io and authorize GitHub access.
5. Choose **Create app**, your repository and `main` branch, and enter **Main file path**: `mobile_app.py`.
6. Click **Deploy**. Streamlit installs the Python dependencies and hosts the service for you.
7. Keep app privacy set to **private**, and sign in on your Android browser to open the `.streamlit.app` URL.
8. Optionally use Chrome's menu to add a shortcut to your home screen.

## Configure market data

1. Claim a free API key at https://www.alphavantage.co/support/#api-key.
2. In Streamlit Community Cloud, open the app's settings and add this under **Secrets**:

	```toml
	ALPHA_VANTAGE_API_KEY = "your-key"
	```

3. For local use, put the same setting in `.streamlit/secrets.toml`. That file is gitignored; never commit or share the key.
4. The app uses the latest available daily close, not an intraday quote. Alpha Vantage's standard free allowance is 25 requests per day, shared by all users of the key. Responses are cached, but historical backfills still use several requests and monthly price data.
5. The free key may not provide the Company Overview endpoint. The app derives available ratios from statements and prices; forward estimates, beta, and industry details may remain unavailable. When sector classification is missing, choose a supported sector in the app or keep the generic defaults.
6. Enter symbols accepted by Alpha Vantage. International listing suffixes can differ by provider, so verify each symbol with your key before relying on its valuation.

## Data and security

- This isolated package has no personal portfolio or transaction files, and no Parquet migration is required.
- Market data comes from Alpha Vantage; the displayed quote is the latest available daily close.
- Historical valuations use ephemeral JSON caches on Community Cloud; a restart may require rebuilding history.
- Multi-year historical valuation backfills use monthly price candles and should be interpreted at that granularity.
- Run-time calculations remain the same as in your supplied valuation engine and sector configuration; keep these two files synchronized with your desktop version when you make changes.
- International ticker availability depends on Alpha Vantage's symbol coverage and must be checked per listing.
