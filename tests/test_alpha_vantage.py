import unittest
import os
import sys
from types import SimpleNamespace
from unittest.mock import Mock, PropertyMock, patch

import pandas as pd
import requests

from alpha_vantage import (
    AlphaVantageClient,
    AlphaVantageError,
    AlphaVantageTicker,
    _api_key,
    _price_frame,
    _statement_frame,
    _symbol_classification,
    _ttm_value,
    clear_cache,
)
from valuation_enhanced import fetch_market_metadata, resolve_valuation_sector


class AlphaVantageParsingTests(unittest.TestCase):
    def setUp(self):
        clear_cache()

    def test_statement_frames_normalize_labels_dates_and_capex_sign(self):
        payload = {
            "annualReports": [
                {
                    "fiscalDateEnding": "2024-12-31",
                    "reportedCurrency": "USD",
                    "totalRevenue": "1200",
                    "capitalExpenditures": "80",
                },
                {
                    "fiscalDateEnding": "2023-12-31",
                    "reportedCurrency": "USD",
                    "totalRevenue": "1000",
                    "capitalExpenditures": "60",
                },
            ]
        }

        frame = _statement_frame(payload, "annualReports")

        self.assertEqual(list(frame.columns), [pd.Timestamp("2024-12-31"), pd.Timestamp("2023-12-31")])
        self.assertEqual(frame.loc["Total Revenue"].tolist(), [1200.0, 1000.0])
        self.assertEqual(frame.loc["Capital Expenditure"].tolist(), [-80.0, -60.0])

    def test_balance_sheet_adds_total_debt_from_short_and_long_term(self):
        frame = _statement_frame(
            {
                "annualReports": [
                    {
                        "fiscalDateEnding": "2024-12-31",
                        "shortTermDebt": "30",
                        "longTermDebt": "70",
                    }
                ]
            },
            "annualReports",
        )

        self.assertEqual(frame.loc["Total Debt"].iloc[0], 100.0)

    def test_price_frame_returns_sorted_yahoo_style_ohlcv(self):
        payload = {
            "Time Series (Daily)": {
                "2025-01-02": {"1. open": "11", "2. high": "12", "3. low": "10", "4. close": "11.5", "5. volume": "20"},
                "2025-01-01": {"1. open": "10", "2. high": "11", "3. low": "9", "4. close": "10.5", "5. volume": "30"},
            }
        }

        frame = _price_frame(payload, "Time Series (Daily)")

        self.assertEqual(list(frame.columns), ["Open", "High", "Low", "Close", "Volume"])
        self.assertEqual(frame.iloc[-1]["Close"], 11.5)

    def test_provider_quota_response_becomes_actionable_error(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"Note": "Thank you for using Alpha Vantage! API call frequency is 5 calls per minute."}
        session = Mock()
        session.get.return_value = response

        with self.assertRaisesRegex(AlphaVantageError, "free API limit reached"):
            AlphaVantageClient(api_key="test", session=session).request("OVERVIEW", symbol="IBM")

    def test_http_errors_never_expose_api_key(self):
        response = Mock()
        response.status_code = 401
        session = Mock()
        session.get.side_effect = requests.HTTPError(
            "request failed for https://example.test/?apikey=private-test-key",
            response=response,
        )

        with self.assertRaises(AlphaVantageError) as context:
            AlphaVantageClient(api_key="private-test-key", session=session).request("OVERVIEW", symbol="IBM")

        self.assertNotIn("private-test-key", str(context.exception))
        self.assertIn("rejected the API key", str(context.exception))

    def test_provider_configuration_errors_are_not_retried(self):
        stock = Mock()
        type(stock).info = PropertyMock(
            side_effect=AlphaVantageError("Alpha Vantage API key is not configured.")
        )
        with patch("valuation_enhanced.AlphaVantageTicker", return_value=stock) as factory:
            fetched_stock, info, error = fetch_market_metadata("IBM", retry_delay=0)

        self.assertIsNone(fetched_stock)
        self.assertIsNone(info)
        self.assertIn("not configured", error)
        self.assertEqual(factory.call_count, 1)

    def test_invalid_streamlit_secrets_reports_toml_problem(self):
        fake_streamlit = SimpleNamespace(secrets=SimpleNamespace(get=Mock(side_effect=ValueError("invalid TOML"))))
        with patch.dict(os.environ, {"ALPHA_VANTAGE_API_KEY": ""}), patch.dict(
            sys.modules, {"streamlit": fake_streamlit}
        ):
            with self.assertRaisesRegex(AlphaVantageError, "valid TOML"):
                _api_key()

    def test_fx_pairs_do_not_share_cached_responses(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"Time Series FX (Daily)": {"2025-01-02": {"4. close": "0.9"}}}
        session = Mock()
        session.get.return_value = response
        client = AlphaVantageClient(api_key="test", session=session)

        client.request("FX_DAILY", from_symbol="USD", to_symbol="EUR")
        client.request("FX_DAILY", from_symbol="EUR", to_symbol="USD")

        self.assertEqual(session.get.call_count, 2)

    def test_ticker_history_filters_end_exclusively(self):
        client = Mock()
        client.request.return_value = {
            "Time Series (Daily)": {
                "2025-01-03": {"1. open": "12", "2. high": "13", "3. low": "11", "4. close": "12", "5. volume": "3"},
                "2025-01-02": {"1. open": "11", "2. high": "12", "3. low": "10", "4. close": "11", "5. volume": "2"},
                "2025-01-01": {"1. open": "10", "2. high": "11", "3. low": "9", "4. close": "10", "5. volume": "1"},
            }
        }

        frame = AlphaVantageTicker("IBM", client=client).history(start="2025-01-02", end="2025-01-03")

        self.assertEqual(frame["Close"].tolist(), [11.0])

    def test_ticker_info_maps_daily_close_and_financial_metadata(self):
        client = Mock()
        client.request.side_effect = [
            {"Time Series (Daily)": {"2025-01-03": {"1. open": "12", "2. high": "13", "3. low": "11", "4. close": "12", "5. volume": "3"}, "2025-01-02": {"1. open": "11", "2. high": "12", "3. low": "10", "4. close": "11", "5. volume": "2"}}},
            {"quarterlyReports": [{"reportedCurrency": "EUR", "totalRevenue": "1000", "netIncome": "100", "grossProfit": "400", "operatingIncome": "150", "ebitda": "180", "dilutedEPS": "2"}]},
            {"quarterlyReports": [{"shortTermDebt": "30", "longTermDebt": "70", "cashAndCashEquivalentsAtCarryingValue": "20"}]},
            {"quarterlyReports": [{"operatingCashflow": "50", "capitalExpenditures": "10"}]},
        ]

        info = AlphaVantageTicker("IBM", client=client).info

        self.assertEqual(info["currentPrice"], 12.0)
        self.assertEqual(info["quoteDate"], "2025-01-03")
        self.assertEqual(info["sector"], "Unknown")
        self.assertEqual(info["sectorSource"], "unavailable")
        self.assertEqual(info["financialCurrency"], "EUR")
        self.assertEqual(info["totalDebt"], 100.0)
        self.assertEqual(info["totalCash"], 20.0)
        self.assertEqual(info["freeCashflow"], 40.0)
        self.assertEqual(info["totalRevenue"], 1000.0)
        self.assertIsNone(info["forwardEps"])
        self.assertNotIn("OVERVIEW", [call.args[0] for call in client.request.call_args_list])

    def test_ttm_value_sums_latest_four_quarters(self):
        quarterly = [
            {"fiscalDateEnding": f"2025-0{quarter}-31", "totalRevenue": str(amount)}
            for quarter, amount in enumerate((400, 300, 200, 100), start=1)
        ]

        self.assertEqual(_ttm_value(quarterly, [], "totalRevenue"), 1000.0)

    def test_sector_mapping_covers_requested_companies(self):
        self.assertEqual(_symbol_classification("ASML.AS"), ("ASML Holding", "Technology", "Semiconductor Equipment"))
        self.assertEqual(_symbol_classification("NOVO-B.CO"), ("Novo Nordisk", "Healthcare", "Pharmaceuticals"))

    def test_manual_sector_override_uses_supported_sector(self):
        raw, sector, source, reason = resolve_valuation_sector(
            {"sector": "Unknown", "sectorSource": "unavailable"},
            "Semiconductors",
        )

        self.assertEqual((raw, sector, source), ("Unknown", "Semiconductors", "user"))
        self.assertEqual(reason, "manual sector selection")

    def test_manual_sector_override_rejects_unknown_values(self):
        with self.assertRaisesRegex(ValueError, "Unsupported manual sector"):
            resolve_valuation_sector({"sector": "Unknown"}, "Made Up Sector")


if __name__ == "__main__":
    unittest.main()