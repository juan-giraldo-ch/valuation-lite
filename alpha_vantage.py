"""Alpha Vantage client with Yahoo-shaped data frames for the valuation engine."""
from __future__ import annotations

import os
import threading
import time
from collections import defaultdict
from datetime import date, datetime
from typing import Any

import pandas as pd
import requests

API_URL = "https://www.alphavantage.co/query"
TIMEOUT_SECONDS = 15

_CACHE: dict[tuple[Any, ...], tuple[float, dict[str, Any]]] = {}
_CACHE_LOCK = threading.Lock()
_TTL_SECONDS = {
    "OVERVIEW": 3600,
    "INCOME_STATEMENT": 21600,
    "BALANCE_SHEET": 21600,
    "CASH_FLOW": 21600,
    "TIME_SERIES_DAILY": 600,
    "TIME_SERIES_MONTHLY": 21600,
    "FX_DAILY": 21600,
    "SYMBOL_SEARCH": 86400,
}

_FIELD_LABELS = {
    "totalRevenue": "Total Revenue",
    "operatingRevenue": "Operating Revenue",
    "grossProfit": "Gross Profit",
    "operatingIncome": "Operating Income",
    "ebit": "EBIT",
    "ebitda": "EBITDA",
    "netIncome": "Net Income",
    "netIncomeFromContinuingOperations": "Net Income From Continuing Operations",
    "eps": "EPS",
    "dilutedEPS": "Diluted EPS",
    "reportedEPS": "Reported EPS",
    "interestExpense": "Interest Expense",
    "depreciationAndAmortization": "Depreciation And Amortization",
    "totalAssets": "Total Assets",
    "totalLiabilities": "Total Liabilities",
    "totalShareholderEquity": "Stockholders Equity",
    "totalDebt": "Total Debt",
    "currentDebt": "Current Debt",
    "shortTermDebt": "Short Term Debt",
    "longTermDebt": "Long Term Debt",
    "cashAndCashEquivalentsAtCarryingValue": "Cash And Cash Equivalents",
    "cashAndShortTermInvestments": "Cash Cash Equivalents And Short Term Investments",
    "commonStockSharesOutstanding": "Common Stock Shares Outstanding",
    "operatingCashflow": "Operating Cash Flow",
    "capitalExpenditures": "Capital Expenditure",
    "freeCashFlow": "Free Cash Flow",
}


class AlphaVantageError(RuntimeError):
    """Raised when Alpha Vantage cannot provide a usable response."""


def _api_key() -> str | None:
    key = os.environ.get("ALPHA_VANTAGE_API_KEY")
    if key:
        return key.strip()
    try:
        import streamlit as st

        return str(st.secrets.get("ALPHA_VANTAGE_API_KEY", "")).strip() or None
    except Exception as exc:
        if type(exc).__name__ == "StreamlitSecretNotFoundError":
            return None
        raise AlphaVantageError(
            "Streamlit secrets could not be read. Check that .streamlit/secrets.toml "
            "contains valid TOML and a top-level ALPHA_VANTAGE_API_KEY entry."
        ) from exc


def _provider_error(payload: dict[str, Any]) -> str | None:
    message = payload.get("Error Message") or payload.get("Note") or payload.get("Information")
    if not message:
        return None
    message = str(message)
    lower = message.lower()
    if (
        "api call frequency" in lower
        or "requests per day" in lower
        or "call frequency" in lower
        or "standard api rate limit" in lower
    ):
        return "Alpha Vantage free API limit reached (25 requests per day). Try again tomorrow."
    if "premium" in lower or "subscription" in lower:
        return f"Alpha Vantage requires a premium endpoint: {message}"
    return f"Alpha Vantage: {message}"


class AlphaVantageClient:
    def __init__(self, api_key: str | None = None, session: Any = requests):
        self.api_key = api_key.strip() if api_key else _api_key()
        self.session = session

    def request(self, function: str, **params: str) -> dict[str, Any]:
        if not self.api_key:
            raise AlphaVantageError(
                "Alpha Vantage API key is not configured. Add ALPHA_VANTAGE_API_KEY "
                "to Streamlit Cloud Secrets or the local .streamlit/secrets.toml file."
            )
        cache_key = (function, tuple(sorted(params.items())), self.api_key)
        now = time.monotonic()
        with _CACHE_LOCK:
            cached = _CACHE.get(cache_key)
            if cached and cached[0] > now:
                return cached[1].copy()

        query = {"function": function, "apikey": self.api_key, **params}
        try:
            response = self.session.get(API_URL, params=query, timeout=TIMEOUT_SECONDS)
            response.raise_for_status()
            payload = response.json()
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status in {401, 403}:
                raise AlphaVantageError(
                    "Alpha Vantage rejected the API key or this endpoint is unavailable to the key. "
                    "Check the key in Streamlit Secrets."
                ) from exc
            raise AlphaVantageError(f"Alpha Vantage HTTP request failed (status {status or 'unknown'}).") from exc
        except requests.RequestException as exc:
            raise AlphaVantageError(
                f"Could not reach Alpha Vantage ({type(exc).__name__}). Check the app's network and retry."
            ) from exc
        except ValueError as exc:
            raise AlphaVantageError("Alpha Vantage returned invalid JSON.") from exc

        if not isinstance(payload, dict):
            raise AlphaVantageError("Alpha Vantage returned an unexpected response.")
        error = _provider_error(payload)
        if error:
            raise AlphaVantageError(error)

        ttl = _TTL_SECONDS.get(function, 3600)
        with _CACHE_LOCK:
            _CACHE[cache_key] = (now + ttl, payload)
        return payload.copy()


def _number(value: Any) -> float | None:
    if value in (None, "", "None", "null"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _statement_frame(payload: dict[str, Any], period: str) -> pd.DataFrame:
    reports = payload.get(period, [])
    if not isinstance(reports, list):
        return pd.DataFrame()
    rows: dict[str, dict[pd.Timestamp, float]] = defaultdict(dict)
    for report in reports:
        if not isinstance(report, dict):
            continue
        try:
            report_date = pd.Timestamp(report["fiscalDateEnding"])
        except (KeyError, TypeError, ValueError):
            continue
        for field, raw_value in report.items():
            if field in {"fiscalDateEnding", "reportedCurrency"}:
                continue
            value = _number(raw_value)
            if value is None:
                continue
            label = _FIELD_LABELS.get(field)
            if not label:
                label = " ".join(part.capitalize() for part in _split_camel_case(field))
            if field == "capitalExpenditures":
                value = -abs(value)
            rows[label][report_date] = value
    if not rows:
        return pd.DataFrame()
    if "Total Debt" not in rows and ("Short Term Debt" in rows or "Long Term Debt" in rows):
        short_term = rows.get("Short Term Debt") or rows.get("Current Debt") or {}
        long_term = rows.get("Long Term Debt") or {}
        dates = set(short_term) | set(long_term)
        rows["Total Debt"] = {
            report_date: max(short_term.get(report_date, 0.0), 0.0)
            + max(long_term.get(report_date, 0.0), 0.0)
            for report_date in dates
        }
    frame = pd.DataFrame.from_dict(rows, orient="index")
    frame = frame.reindex(sorted(frame.columns, reverse=True), axis=1)
    frame.columns = pd.DatetimeIndex(frame.columns)
    return frame.apply(pd.to_numeric, errors="coerce")


def _split_camel_case(value: str) -> list[str]:
    import re

    return re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value).split()


def _price_frame(payload: dict[str, Any], key: str) -> pd.DataFrame:
    series = payload.get(key, {})
    columns = ["Open", "High", "Low", "Close", "Volume"]
    if not isinstance(series, dict):
        return pd.DataFrame(columns=columns, index=pd.DatetimeIndex([]))
    rows = []
    for date_text, values in series.items():
        if not isinstance(values, dict):
            continue
        try:
            row_date = pd.Timestamp(date_text)
        except (TypeError, ValueError):
            continue
        rows.append(
            {
                "date": row_date,
                "Open": _number(values.get("1. open")),
                "High": _number(values.get("2. high")),
                "Low": _number(values.get("3. low")),
                "Close": _number(values.get("4. close")),
                "Volume": _number(values.get("5. volume")),
            }
        )
    if not rows:
        return pd.DataFrame(columns=columns, index=pd.DatetimeIndex([]))
    frame = pd.DataFrame(rows).set_index("date").sort_index()
    return frame[columns].apply(pd.to_numeric, errors="coerce")


class AlphaVantageTicker:
    """Small yfinance-compatible view over Alpha Vantage endpoints."""

    def __init__(self, symbol: str, client: AlphaVantageClient | None = None):
        self.symbol = symbol.strip().upper()
        self.client = client or AlphaVantageClient()

    @property
    def info(self) -> dict[str, Any]:
        daily = self.history()
        if daily.empty:
            raise AlphaVantageError(f"Alpha Vantage returned no daily prices for {self.symbol}.")
        latest = daily.iloc[-1]
        income = self._income_payload()
        balance = self._statement_payload("BALANCE_SHEET")
        cashflow = self._statement_payload("CASH_FLOW")
        q_income = income.get("quarterlyReports", [])
        annual_income = income.get("annualReports", [])
        q_balance = balance.get("quarterlyReports", [])
        annual_balance = balance.get("annualReports", [])
        q_cashflow = cashflow.get("quarterlyReports", [])
        annual_cashflow = cashflow.get("annualReports", [])
        latest_balance = _latest_report(q_balance or annual_balance)
        financial_currency = (
            _latest_report(q_income or annual_income).get("reportedCurrency")
            or latest_balance.get("reportedCurrency")
            or "USD"
        )
        currency = _quote_currency(self.symbol)
        revenue = _ttm_value(q_income, annual_income, "totalRevenue", "operatingRevenue")
        net_income = _ttm_value(q_income, annual_income, "netIncome", "netIncomeFromContinuingOperations")
        gross_profit = _ttm_value(q_income, annual_income, "grossProfit")
        operating_income = _ttm_value(q_income, annual_income, "operatingIncome", "ebit")
        ebitda = _ttm_value(q_income, annual_income, "ebitda")
        shares = _number(latest_balance.get("commonStockSharesOutstanding"))
        book_equity = _first_number(latest_balance, "totalShareholderEquity", "totalStockholderEquity")
        trailing_eps = _ttm_value(q_income, annual_income, "dilutedEPS", "reportedEPS", "eps")
        price = float(latest["Close"])
        if trailing_eps is None and net_income is not None and shares and shares > 0:
            trailing_eps = net_income / shares
        company_name, sector, industry = _symbol_classification(self.symbol)
        sector_source = "ticker_mapping" if sector != "Unknown" else "unavailable"
        warnings = [
            "The free API key cannot access Company Overview. Forward estimates, beta, and provider-supplied sector details are unavailable."
        ]
        if sector == "Unknown":
            warnings.append("No sector mapping is available for this ticker; generic valuation assumptions are being used.")
        info = {
            "symbol": self.symbol,
            "ticker": self.symbol,
            "shortName": company_name,
            "longName": company_name,
            "longBusinessSummary": "",
            "sector": sector,
            "industry": industry,
            "sectorSource": sector_source,
            "country": "",
            "currency": currency,
            "financialCurrency": financial_currency,
            "currentPrice": price,
            "quoteDate": daily.index[-1].date().isoformat(),
            "regularMarketChangePercent": _change_percent(daily["Close"]),
            "52WeekChange": None,
            "marketCap": price * shares if shares else None,
            "ebitda": ebitda,
            "trailingPE": price / trailing_eps if trailing_eps else None,
            "forwardPE": None,
            "pegRatio": None,
            "bookValue": book_equity / shares if book_equity is not None and shares else None,
            "trailingEps": trailing_eps,
            "forwardEps": None,
            "sharesOutstanding": shares,
            "totalRevenue": revenue,
            "grossMargins": _margin(gross_profit, revenue),
            "operatingMargins": _margin(operating_income, revenue),
            "profitMargins": _margin(net_income, revenue),
            "revenueGrowth": _ttm_growth(q_income, annual_income, "totalRevenue", "operatingRevenue"),
            "earningsGrowth": _ttm_growth(q_income, annual_income, "netIncome", "netIncomeFromContinuingOperations"),
            "earningsQuarterlyGrowth": None,
            "returnOnEquity": _margin(net_income, book_equity),
            "beta": None,
            "quoteType": "EQUITY",
            "providerWarnings": warnings,
        }
        info["totalDebt"] = _debt_total(latest_balance)
        info["totalCash"] = _first_number(latest_balance, "cashAndCashEquivalentsAtCarryingValue", "cashAndShortTermInvestments")
        info["freeCashflow"] = _ttm_fcf(q_cashflow, annual_cashflow)
        return info

    def _statement_payload(self, function: str) -> dict[str, Any]:
        return self.client.request(function, symbol=self.symbol)

    def _income_payload(self) -> dict[str, Any]:
        return self._statement_payload("INCOME_STATEMENT")

    @property
    def financials(self) -> pd.DataFrame:
        return _statement_frame(self._income_payload(), "annualReports")

    @property
    def quarterly_financials(self) -> pd.DataFrame:
        return _statement_frame(self._income_payload(), "quarterlyReports")

    @property
    def balance_sheet(self) -> pd.DataFrame:
        return _statement_frame(self._statement_payload("BALANCE_SHEET"), "annualReports")

    @property
    def quarterly_balance_sheet(self) -> pd.DataFrame:
        return _statement_frame(self._statement_payload("BALANCE_SHEET"), "quarterlyReports")

    @property
    def cashflow(self) -> pd.DataFrame:
        return _statement_frame(self._statement_payload("CASH_FLOW"), "annualReports")

    @property
    def quarterly_cashflow(self) -> pd.DataFrame:
        return _statement_frame(self._statement_payload("CASH_FLOW"), "quarterlyReports")

    def get_income_stmt(self, freq: str = "yearly") -> pd.DataFrame:
        frame = self.quarterly_financials if freq.lower().startswith("q") else self.financials
        if "Diluted EPS" in frame.index:
            frame = frame.rename(index={"Diluted EPS": "DilutedEPS"})
        return frame

    def get_earnings_dates(self, limit: int = 40) -> pd.DataFrame:
        return pd.DataFrame(columns=["EPS Estimate", "Reported EPS", "Surprise(%)"])

    def history(
        self,
        start: date | datetime | pd.Timestamp | None = None,
        end: date | datetime | pd.Timestamp | None = None,
        interval: str = "1d",
        period: str | None = None,
    ) -> pd.DataFrame:
        monthly = interval in {"1mo", "1m"}
        function = "TIME_SERIES_MONTHLY" if monthly else "TIME_SERIES_DAILY"
        key = "Monthly Time Series" if monthly else "Time Series (Daily)"
        payload = self.client.request(function, symbol=self.symbol)
        prices = _price_frame(payload, key)
        if start is not None:
            prices = prices.loc[prices.index >= pd.Timestamp(start).tz_localize(None)]
        if end is not None:
            prices = prices.loc[prices.index < pd.Timestamp(end).tz_localize(None)]
        return prices


def get_fx_rate(from_currency: str, to_currency: str, client: AlphaVantageClient | None = None) -> float | None:
    if not from_currency or not to_currency:
        return None
    if from_currency.upper() == to_currency.upper():
        return 1.0
    api = client or AlphaVantageClient()
    try:
        payload = api.request("FX_DAILY", from_symbol=from_currency.upper(), to_symbol=to_currency.upper())
    except AlphaVantageError:
        try:
            inverse = api.request("FX_DAILY", from_symbol=to_currency.upper(), to_symbol=from_currency.upper())
        except AlphaVantageError:
            return None
        series = inverse.get("Time Series FX (Daily)", {})
        last = _latest_series_value(series)
        rate = _number(last.get("4. close"))
        return 1 / rate if rate and rate > 0 else None
    series = payload.get("Time Series FX (Daily)", {})
    last = _latest_series_value(series)
    return _number(last.get("4. close"))


def clear_cache() -> None:
    with _CACHE_LOCK:
        _CACHE.clear()


def _first_number(data: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _number(data.get(key))
        if value is not None:
            return value
    return None


def _latest_series_value(series: Any) -> dict[str, Any]:
    if not isinstance(series, dict) or not series:
        return {}
    value = series[max(series)]
    return value if isinstance(value, dict) else {}


def _latest_report(reports: Any) -> dict[str, Any]:
    if not isinstance(reports, list) or not reports:
        return {}
    return max(reports, key=lambda report: str(report.get("fiscalDateEnding", "")))


def _report_value(report: dict[str, Any], *fields: str) -> float | None:
    return _first_number(report, *fields)


def _ttm_value(quarterly: Any, annual: Any, *fields: str) -> float | None:
    quarterly_reports = sorted(
        quarterly if isinstance(quarterly, list) else [],
        key=lambda report: str(report.get("fiscalDateEnding", "")),
        reverse=True,
    )
    quarter_values = [_report_value(report, *fields) for report in quarterly_reports]
    quarter_values = [value for value in quarter_values if value is not None]
    if len(quarter_values) >= 4:
        return sum(quarter_values[:4])
    annual_reports = sorted(
        annual if isinstance(annual, list) else [],
        key=lambda report: str(report.get("fiscalDateEnding", "")),
        reverse=True,
    )
    for report in annual_reports:
        value = _report_value(report, *fields)
        if value is not None:
            return value
    return sum(quarter_values) if quarter_values else None


def _ttm_growth(quarterly: Any, annual: Any, *fields: str) -> float | None:
    quarterly_reports = sorted(
        quarterly if isinstance(quarterly, list) else [],
        key=lambda report: str(report.get("fiscalDateEnding", "")),
        reverse=True,
    )
    quarter_values = [_report_value(report, *fields) for report in quarterly_reports]
    if len(quarter_values) >= 8 and all(value is not None for value in quarter_values[:8]):
        current, prior = sum(quarter_values[:4]), sum(quarter_values[4:8])
    else:
        annual_reports = sorted(
            annual if isinstance(annual, list) else [],
            key=lambda report: str(report.get("fiscalDateEnding", "")),
            reverse=True,
        )
        annual_values = [_report_value(report, *fields) for report in annual_reports]
        annual_values = [value for value in annual_values if value is not None]
        if len(annual_values) < 2:
            return None
        current, prior = annual_values[:2]
    return current / prior - 1 if prior else None


def _ttm_fcf(quarterly: Any, annual: Any) -> float | None:
    def report_fcf(report: dict[str, Any]) -> float | None:
        operating = _number(report.get("operatingCashflow"))
        capex = _number(report.get("capitalExpenditures"))
        return operating - abs(capex) if operating is not None and capex is not None else None

    quarterly_reports = sorted(
        quarterly if isinstance(quarterly, list) else [],
        key=lambda report: str(report.get("fiscalDateEnding", "")),
        reverse=True,
    )
    quarter_values = [report_fcf(report) for report in quarterly_reports]
    quarter_values = [value for value in quarter_values if value is not None]
    if len(quarter_values) >= 4:
        return sum(quarter_values[:4])
    for report in sorted(
        annual if isinstance(annual, list) else [],
        key=lambda report: str(report.get("fiscalDateEnding", "")),
        reverse=True,
    ):
        value = report_fcf(report)
        if value is not None:
            return value
    return sum(quarter_values) if quarter_values else None


def _quote_currency(symbol: str) -> str:
    suffix = symbol.rsplit(".", 1)[-1] if "." in symbol else ""
    return {
        "AS": "EUR", "PA": "EUR", "DE": "EUR", "F": "EUR", "MI": "EUR",
        "MC": "EUR", "BR": "EUR", "VI": "EUR", "HE": "EUR",
        "CO": "DKK", "ST": "SEK", "OL": "NOK", "L": "GBp",
        "TO": "CAD", "V": "CAD", "AX": "AUD", "HK": "HKD", "T": "JPY",
        "NZ": "NZD", "SI": "SGD", "KS": "KRW", "TW": "TWD",
    }.get(suffix, "USD")


def _symbol_classification(symbol: str) -> tuple[str, str, str]:
    base = symbol.upper().split(".", 1)[0]
    if base == "ASML":
        return "ASML Holding", "Technology", "Semiconductor Equipment"
    if base in {"NVO", "NOVO-B"}:
        return "Novo Nordisk", "Healthcare", "Pharmaceuticals"
    return symbol, "Unknown", "Unknown"


def _debt_total(report: dict[str, Any]) -> float | None:
    total = _number(report.get("totalDebt"))
    if total is not None:
        return max(total, 0.0)
    short_term = _first_number(report, "shortTermDebt", "currentDebt")
    long_term = _number(report.get("longTermDebt"))
    if short_term is not None or long_term is not None:
        return max(short_term or 0.0, 0.0) + max(long_term or 0.0, 0.0)
    return _number(report.get("totalDebt"))


def _margin(numerator: Any, denominator: Any) -> float | None:
    top, bottom = _number(numerator), _number(denominator)
    return top / bottom if top is not None and bottom else None


def _change_percent(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) < 2 or values.iloc[-2] == 0:
        return None
    return (float(values.iloc[-1]) / float(values.iloc[-2]) - 1) * 100


def _cashflow_fcf(report: dict[str, Any]) -> float | None:
    operating = _number(report.get("operatingCashflow"))
    capex = _number(report.get("capitalExpenditures"))
    return operating - abs(capex) if operating is not None and capex is not None else None


def _normalize_sector(sector: Any) -> str:
    value = str(sector or "Unknown").strip().upper()
    mapping = {
        "TECHNOLOGY": "Technology",
        "HEALTHCARE": "Healthcare",
        "FINANCE": "Financials",
        "FINANCIAL SERVICES": "Financial Services",
        "CONSUMER CYCLICAL": "Consumer",
        "CONSUMER DEFENSIVE": "Consumer",
        "BASIC MATERIALS": "Materials",
        "COMMUNICATION SERVICES": "Communication Services",
        "REAL ESTATE": "Real Estate",
        "INDUSTRIALS": "Industrials",
        "ENERGY": "Energy",
        "UTILITIES": "Utilities",
    }
    return mapping.get(value, value.title() if value != "UNKNOWN" else "Unknown")