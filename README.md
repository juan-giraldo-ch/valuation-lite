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

## Data and security

- This isolated package has no personal portfolio or transaction files, and no Parquet migration is required.
- Market data comes from Yahoo Finance; hosted access is not guaranteed and may encounter rate limits.
- Historical valuations use ephemeral JSON caches on Community Cloud; a restart may require rebuilding history.
- Run-time calculations remain the same as in your supplied valuation engine and sector configuration; keep these two files synchronized with your desktop version when you make changes.
- Static syntax and offline unit tests passed, but the app has not been tested with live Yahoo Finance responses or on a physical Android device.
