"""
Enhanced Valuation Module with Historical Tracking & Data Quality

Improvements:
- Sector-aware method selection and confidence scoring
- Adaptive growth assumptions based on sector
- Sector-specific outlier thresholds
- Historical growth rate calculation from last 5-10 years
- Sector-based EV/EBITDA multiples
- Data quality validation and fallback logic
- Fair value historical tracking (JSON-based cache)
- Margin of safety calculations
- More robust WACC calculation
- AUTO-CALIBRATING: Detects growth profile & adjusts assumptions automatically
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from pathlib import Path
import logging
import time
from typing import Any, Dict, List, Optional, Tuple
from sector_config import (
    SECTOR_CONFIGS,
    get_policy_metadata,
    get_sector_config,
    get_growth_assumption,
    get_sector_method_weights,
    SectorConfig,
)
from config import SUBUNIT_CURRENCIES

logger = logging.getLogger(__name__)

# Sector-specific metrics (based on industry averages)
SECTOR_MULTIPLES = {
    "Technology": {"ev_ebitda": 18, "pe": 28, "fcf_yield": 0.04},
    "Software": {"ev_ebitda": 22, "pe": 32, "fcf_yield": 0.035},          # Asset-light premium
    "Semiconductors": {"ev_ebitda": 20, "pe": 26, "fcf_yield": 0.045},    # Cyclical; mid-cycle reference
    "Communication Services": {"ev_ebitda": 18, "pe": 26, "fcf_yield": 0.035},
    "Healthcare": {"ev_ebitda": 16, "pe": 24, "fcf_yield": 0.035},
    "Managed Care": {"ev_ebitda": 13, "pe": 18, "fcf_yield": 0.045},
    "Pharmaceuticals": {"ev_ebitda": 15, "pe": 22, "fcf_yield": 0.04},
    "Medical Devices": {"ev_ebitda": 16, "pe": 24, "fcf_yield": 0.04},
    "Life Science Tools": {"ev_ebitda": 17, "pe": 25, "fcf_yield": 0.038},
    "Healthcare Providers": {"ev_ebitda": 9, "pe": 15, "fcf_yield": 0.06},
    "Pharmaceutical Distribution": {"ev_ebitda": 8, "pe": 14, "fcf_yield": 0.065},
    "Financials": {"ev_ebitda": 12, "pe": 14, "fcf_yield": 0.05},
    "Financial Services": {"ev_ebitda": 15, "pe": 20, "fcf_yield": 0.045},
    "Financial Data Services": {"ev_ebitda": 17, "pe": 24, "fcf_yield": 0.04},
    "Gov Defense IT": {"ev_ebitda": 12, "pe": 18, "fcf_yield": 0.05},
    "Gov Defense IT Hybrid": {"ev_ebitda": 13, "pe": 20, "fcf_yield": 0.048},
    "Industrials": {"ev_ebitda": 14, "pe": 18, "fcf_yield": 0.045},
    "Consumer": {"ev_ebitda": 15, "pe": 22, "fcf_yield": 0.04},
    "Energy": {"ev_ebitda": 10, "pe": 12, "fcf_yield": 0.06},
    "Utilities": {"ev_ebitda": 10, "pe": 14, "fcf_yield": 0.055},  # Lower multiples for stable, slow-growth
    "Materials": {"ev_ebitda": 11, "pe": 14, "fcf_yield": 0.05},
    "Default": {"ev_ebitda": 13, "pe": 18, "fcf_yield": 0.045},
}


# Growth Classification & Valuation Parameter Calibration
class GrowthProfile:
    """Classify company growth profile and suggest valuation parameters."""
    
    HIGH_GROWTH = "High-Growth"      # >15% historical CAGR
    MODERATE_GROWTH = "Moderate"     # 8-15% CAGR
    MATURE = "Mature"                # 0-8% CAGR
    VALUE = "Value"                  # Negative growth or low multiples
    
    def __init__(self, historical_growth, current_pe, sector, forward_pe=None):
        """
        Args:
            historical_growth: CAGR from last 5 years (0.0 - 1.0)
            current_pe: Current P/E multiple
            sector: Company sector
            forward_pe: Forward P/E if available
        """
        self.historical_growth = historical_growth or 0.05
        self.current_pe = current_pe or 15
        self.forward_pe = forward_pe or current_pe
        self.sector = sector
        self.profile = self._classify()
    
    def _classify(self):
        """Classify stock into growth category."""
        if self.historical_growth >= 0.15:
            return self.HIGH_GROWTH
        elif self.historical_growth >= 0.08:
            return self.MODERATE_GROWTH
        elif self.historical_growth > 0:
            return self.MATURE
        else:
            return self.VALUE
    
    def get_calibrated_params(self):
        """Return valuation parameters calibrated to this growth profile."""
        params = {
            "profile": self.profile,
            "historical_growth": self.historical_growth,
        }
        
        if self.profile == self.HIGH_GROWTH:
            # High-growth stocks: 12-18% growth, weight heavily on P/E
            params.update({
                "base_growth": min(0.18, self.historical_growth * 0.8),  # 80% of historical, capped at 18%
                "pessimistic_growth": max(0.08, self.historical_growth * 0.5),
                "optimistic_growth": min(0.25, self.historical_growth * 1.2),
                "method_weights": {"DCF": 0.25, "PE": 0.60, "EV_EBITDA": 0.15},
                "pe_expansion_factor": 1.10,  # 10% upside from P/E expansion
                "description": f"High-growth stock ({self.historical_growth*100:.1f}% historical CAGR)",
            })
        
        elif self.profile == self.MODERATE_GROWTH:
            # Moderate-growth: 8-12% growth, balanced methods
            params.update({
                "base_growth": min(0.12, self.historical_growth * 0.9),
                "pessimistic_growth": max(0.03, self.historical_growth * 0.5),
                "optimistic_growth": min(0.18, self.historical_growth * 1.3),
                "method_weights": {"DCF": 0.40, "PE": 0.35, "EV_EBITDA": 0.25},
                "pe_expansion_factor": 1.05,  # 5% upside
                "description": f"Moderate-growth stock ({self.historical_growth*100:.1f}% historical CAGR)",
            })
        
        elif self.profile == self.MATURE:
            # Mature: Conservative growth, weight on DCF/EBITDA
            params.update({
                "base_growth": min(0.08, self.historical_growth),
                "pessimistic_growth": max(0.00, self.historical_growth * 0.7),
                "optimistic_growth": min(0.12, self.historical_growth * 1.5),
                "method_weights": {"DCF": 0.50, "PE": 0.25, "EV_EBITDA": 0.25},
                "pe_expansion_factor": 1.02,  # 2% upside
                "description": f"Mature/stable stock ({self.historical_growth*100:.1f}% historical CAGR)",
            })
        
        else:  # VALUE
            # Value stocks: Low growth, focus on DCF and multiples
            params.update({
                "base_growth": 0.03,
                "pessimistic_growth": 0.00,
                "optimistic_growth": 0.08,
                "method_weights": {"DCF": 0.60, "PE": 0.20, "EV_EBITDA": 0.20},
                "pe_expansion_factor": 0.98,  # Discount for low growth
                "description": f"Value stock (negative/low growth)",
            })
        
        return params


class ValuationCache:
    """Simple JSON-based cache for historical fair values."""
    
    def __init__(self, cache_dir="cache/valuation_history"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_ticker_file(self, ticker):
        return self.cache_dir / f"{ticker.upper()}_valuation_history.json"
    
    def load_history(self, ticker):
        """Load historical valuations for a ticker."""
        file_path = self.get_ticker_file(ticker)
        if file_path.exists():
            try:
                with open(file_path, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def save_valuation(self, ticker, valuation_data, date=None, skip_date_filter=False, source="current"):
        """Save valuation to history.
        
        Args:
            skip_date_filter: If True, don't filter old dates (for historical backfill)
            source: Data source - "current" or "historical_backfill" for tracking
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        history = self.load_history(ticker)
        
        # Only filter if explicitly requested AND no historical backfill data exists
        # This preserves historical data that was already backfilled
        if not skip_date_filter:
            # Check if we have any historical_backfill entries
            has_historical_backfill = any(
                v.get("source") == "historical_backfill" 
                for v in history.values()
            )
            
            # Only apply time filter if NO historical backfill data exists
            if not has_historical_backfill:
                cutoff_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
                history = {k: v for k, v in history.items() if k >= cutoff_date}
        
        blended = valuation_data.get("blended", {}) or {}
        base_assumptions = valuation_data.get("base_assumptions", {}) or {}
        pe_assumptions = valuation_data.get("pe", {}) or base_assumptions.get("pe", {}) or {}
        forward_eps_snapshot = pe_assumptions.get("forward_eps", base_assumptions.get("forward_eps"))
        trailing_eps_snapshot = pe_assumptions.get("trailing_eps", base_assumptions.get("trailing_eps"))
        forward_pe_snapshot = pe_assumptions.get("forward_pe", base_assumptions.get("forward_pe"))
        trailing_pe_snapshot = pe_assumptions.get("trailing_pe", base_assumptions.get("trailing_pe"))
        history[date] = {
            "price": valuation_data.get("price"),
            "fair_value_base": valuation_data.get("fair_value_base", blended.get("Base")),
            "fair_value_low": valuation_data.get("fair_value_low", blended.get("Pessimistic")),
            "fair_value_high": valuation_data.get("fair_value_high", blended.get("Optimistic")),
            "discount_rate": valuation_data.get("discount_rate"),
            "forward_eps": forward_eps_snapshot,
            "trailing_eps": trailing_eps_snapshot,
            "forward_pe": forward_pe_snapshot,
            "trailing_pe": trailing_pe_snapshot,
            "peg_ratio": valuation_data.get("peg_ratio"),
            "lynch_ratio": valuation_data.get("lynch_ratio", valuation_data.get("peg_ratio")),
            "peg_growth_rate_pct": valuation_data.get("peg_growth_rate_pct"),
            "peg_growth_source": valuation_data.get("peg_growth_source"),
            "peg_pe_ratio_used": valuation_data.get("peg_pe_ratio_used"),
            "peg_pe_source": valuation_data.get("peg_pe_source"),
            "lynch_signal": valuation_data.get("lynch_signal"),
            "monthly_candle": valuation_data.get("monthly_candle"),
            "source": source,  # Track whether this came from backfill or current valuation
        }
        
        file_path = self.get_ticker_file(ticker)
        with open(file_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        return history
    
    def clear_history(self, ticker):
        """Clear all historical valuations for a ticker."""
        file_path = self.get_ticker_file(ticker)
        if file_path.exists():
            file_path.unlink()
        return {}


# =====================
# DATA QUALITY & HELPERS
# =====================

def validate_data(value, min_threshold=0.001):
    """Check if a metric is valid and non-zero."""
    try:
        if value is None:
            return False
        if isinstance(value, (int, float)):
            return value > min_threshold
        return False
    except:
        return False


def _implied_consensus_growth(valuation_info):
    """Back-calculate the consensus 5-year growth rate from Yahoo's pegRatio.
    
    Yahoo Finance computes pegRatio = forwardPE / (5-year consensus growth %).
    Inverting this gives the implied analyst consensus growth rate, which is
    a much better Lynch signal input than trailing historical growth.
    Returns a decimal (e.g. 0.168 for 16.8%) or None.
    """
    fpe = valuation_info.get("forwardPE")
    peg = valuation_info.get("pegRatio")
    if fpe and peg and fpe > 0 and peg > 0:
        implied_pct = fpe / peg  # e.g. 16.81 for 16.81%
        if 0.5 < implied_pct < 100:  # sanity bounds
            return implied_pct / 100.0  # convert to decimal
    return None


def calculate_lynch_peg_metrics(valuation_info, historical_eps_growth, historical_fcf_growth, assumptions_override=None):
    """Build a centralized PEG-style metric and metadata for Lynch-style display."""
    pe_candidates = [
        ("trailing_pe", valuation_info.get("trailingPE")),
        ("forward_pe", valuation_info.get("forwardPE")),
    ]
    growth_candidates = []
    if assumptions_override and assumptions_override.get("lynch_growth_rate") is not None:
        growth_candidates.append(("custom_lynch_growth", assumptions_override.get("lynch_growth_rate")))

    # Consensus 5-year growth back-calculated from Yahoo pegRatio (highest-quality automatic source)
    consensus_growth = _implied_consensus_growth(valuation_info)
    if consensus_growth is not None:
        growth_candidates.append(("yahoo_peg_implied_growth", consensus_growth))

    growth_candidates.extend([
        ("historical_eps_growth", historical_eps_growth),
        ("forward_earnings_growth", valuation_info.get("earningsGrowth")),
        ("historical_fcf_growth", historical_fcf_growth),
    ])

    pe_value = None
    pe_source = None
    for source_name, raw_value in pe_candidates:
        if validate_data(raw_value) and raw_value > 0:
            pe_value = float(raw_value)
            pe_source = source_name
            break

    growth_rate = None
    growth_source = None
    for source_name, raw_value in growth_candidates:
        if validate_data(raw_value) and raw_value > 0:
            growth_rate = float(raw_value)
            growth_source = source_name
            break

    growth_rate_pct = growth_rate * 100 if growth_rate is not None else None
    peg_ratio = None
    if pe_value is not None and growth_rate_pct and growth_rate_pct > 0:
        peg_ratio = pe_value / growth_rate_pct

    forward_eps = valuation_info.get("forwardEps")
    trailing_eps = valuation_info.get("trailingEps")
    eps_ratio = None
    normalization_risk = False
    normalization_reason = None
    if validate_data(forward_eps) and validate_data(trailing_eps) and trailing_eps > 0:
        eps_ratio = float(forward_eps) / float(trailing_eps)
        if eps_ratio > 1.50:
            normalization_risk = True
            normalization_reason = "forward EPS is materially above trailing EPS"
        elif eps_ratio < 0.70:
            normalization_risk = True
            normalization_reason = "forward EPS is materially below trailing EPS"

    earnings_quality = {
        "forward_to_trailing_eps_ratio": eps_ratio,
        "ratio_indicates_normalization_risk": normalization_risk,
        "reason": normalization_reason,
    }

    return {
        "peg_ratio": peg_ratio,
        "lynch_ratio": peg_ratio,
        "growth_rate_pct": growth_rate_pct,
        "growth_source": growth_source,
        "pe_ratio_used": pe_value,
        "pe_source": pe_source,
        "earnings_quality": earnings_quality,
    }


def describe_lynch_signal(peg_ratio):
    """Map a PEG ratio to a shared Lynch-style signal description and color."""
    if peg_ratio is None or not validate_data(peg_ratio, min_threshold=0.0) or peg_ratio < 0:
        return {
            "score": None,
            "label": "Unavailable",
            "description": "Missing positive growth or usable P/E",
            "color": "#6b7280",
        }

    stops = [
        (0.0, "#166534"),
        (1.0, "#a16207"),
        (1.5, "#c2410c"),
        (2.0, "#991b1b"),
    ]

    capped = min(float(peg_ratio), stops[-1][0])
    score = capped / stops[-1][0]

    if peg_ratio < 1.0:
        label = "Attractive"
        description = "Growth-adjusted valuation looks supportive"
    elif peg_ratio < 1.5:
        label = "Watch"
        description = "Proceed with caution and confirm fundamentals"
    elif peg_ratio < 2.0:
        label = "Stretched"
        description = "Growth-adjusted valuation is getting expensive"
    else:
        label = "Expensive"
        description = "Growth does not justify the current multiple"

    lower_stop = stops[0]
    upper_stop = stops[-1]
    for idx in range(len(stops) - 1):
        left = stops[idx]
        right = stops[idx + 1]
        if left[0] <= capped <= right[0]:
            lower_stop = left
            upper_stop = right
            break

    def _hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))

    def _rgb_to_hex(rgb):
        return "#" + "".join(f"{component:02x}" for component in rgb)

    if upper_stop[0] == lower_stop[0]:
        color = lower_stop[1]
    else:
        ratio = (capped - lower_stop[0]) / (upper_stop[0] - lower_stop[0])
        lower_rgb = _hex_to_rgb(lower_stop[1])
        upper_rgb = _hex_to_rgb(upper_stop[1])
        color = _rgb_to_hex(
            tuple(int(lower_rgb[i] + (upper_rgb[i] - lower_rgb[i]) * ratio) for i in range(3))
        )

    return {
        "score": score,
        "label": label,
        "description": description,
        "color": color,
    }


def build_historical_cache_payload(valuation_data):
    """Normalize a historical valuation point into the shared cache payload shape."""
    monthly_candle = valuation_data.get("monthly_candle")
    if isinstance(monthly_candle, dict):
        monthly_candle = {
            "open": monthly_candle.get("open"),
            "high": monthly_candle.get("high"),
            "low": monthly_candle.get("low"),
            "close": monthly_candle.get("close"),
            "average": monthly_candle.get("average"),
        }
    else:
        monthly_candle = None

    return {
        "price": valuation_data.get("price"),
        "fair_value_base": valuation_data.get("fair_value_base"),
        "fair_value_low": valuation_data.get("fair_value_low"),
        "fair_value_high": valuation_data.get("fair_value_high"),
        "discount_rate": valuation_data.get("discount_rate"),
        "forward_eps": valuation_data.get("forward_eps"),
        "trailing_eps": valuation_data.get("trailing_eps"),
        "forward_pe": valuation_data.get("forward_pe"),
        "trailing_pe": valuation_data.get("trailing_pe"),
        "peg_ratio": valuation_data.get("peg_ratio"),
        "lynch_ratio": valuation_data.get("lynch_ratio", valuation_data.get("peg_ratio")),
        "peg_growth_rate_pct": valuation_data.get("peg_growth_rate_pct"),
        "peg_growth_source": valuation_data.get("peg_growth_source"),
        "peg_pe_ratio_used": valuation_data.get("peg_pe_ratio_used"),
        "peg_pe_source": valuation_data.get("peg_pe_source"),
        "lynch_signal": valuation_data.get("lynch_signal"),
        "blended": {
            "Base": valuation_data.get("fair_value_base"),
            "Pessimistic": valuation_data.get("fair_value_low"),
            "Optimistic": valuation_data.get("fair_value_high"),
        },
        "monthly_candle": monthly_candle,
    }


def _apply_policy_range(value, field_name, field_range, enforcement_mode, quality_flags, assumption_deltas):
    """Apply field-level policy range using hard clamp or soft warning semantics."""
    if value is None or field_range is None:
        return value

    try:
        numeric = float(value)
    except Exception:
        return value

    low, high = field_range
    if low is None or high is None:
        return numeric

    mode = (enforcement_mode or "soft").lower()
    clamped = max(low, min(high, numeric))
    out_of_range = numeric < low or numeric > high
    if not out_of_range:
        return numeric

    if mode == "hard":
        quality_flags.append(
            f"Hard policy clamp applied for {field_name}: {numeric:.4f} -> {clamped:.4f} (range {low:.4f}-{high:.4f})"
        )
        assumption_deltas[field_name] = {
            "from": numeric,
            "to": clamped,
            "mode": "hard",
            "range": [low, high],
        }
        return clamped

    quality_flags.append(
        f"Soft policy warning for {field_name}: {numeric:.4f} outside range {low:.4f}-{high:.4f}"
    )
    assumption_deltas[field_name] = {
        "from": numeric,
        "to": numeric,
        "mode": "soft",
        "range": [low, high],
    }
    return numeric


def resolve_net_debt_components(stock, stock_info):
    """Resolve net debt with freshest available source and traceability metadata."""
    result = {
        "total_debt": 0.0,
        "total_cash": 0.0,
        "net_debt": 0.0,
        "source": "info_fallback",
        "freshness": "unknown",
    }

    try:
        bs = stock.balance_sheet
        if bs is not None and not bs.empty and len(bs.columns) > 0:
            latest_col = bs.columns[0]
            debt_candidates = [
                "Total Debt",
                "Current Debt And Capital Lease Obligation",
                "Current Debt",
                "Long Term Debt And Capital Lease Obligation",
                "Long Term Debt",
            ]
            cash_candidates = ["Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents", "Cash"]

            total_debt_bs = None
            for label in debt_candidates:
                if label in bs.index:
                    value = bs.loc[label, latest_col]
                    if pd.notna(value):
                        total_debt_bs = float(value)
                        break

            total_cash_bs = None
            for label in cash_candidates:
                if label in bs.index:
                    value = bs.loc[label, latest_col]
                    if pd.notna(value):
                        total_cash_bs = float(value)
                        break

            if total_debt_bs is not None or total_cash_bs is not None:
                result["total_debt"] = max(total_debt_bs or 0.0, 0.0)
                result["total_cash"] = max(total_cash_bs or 0.0, 0.0)
                result["net_debt"] = result["total_debt"] - result["total_cash"]
                result["source"] = "balance_sheet"
                result["freshness"] = "latest_reported"
                return result
    except Exception:
        pass

    total_debt = float(stock_info.get("totalDebt", 0) or 0)
    total_cash = float(stock_info.get("totalCash", 0) or 0)
    result["total_debt"] = max(total_debt, 0.0)
    result["total_cash"] = max(total_cash, 0.0)
    result["net_debt"] = result["total_debt"] - result["total_cash"]
    return result


def calculate_earnings_cagr(stock, years=5):
    """Calculate historical earnings CAGR from net income."""
    try:
        fin = stock.financials
        if fin is None or len(fin.columns) < 2:
            return None
        
        # Get last N years of net income
        ni_values = fin.loc["Net Income"].head(years).dropna()
        valid_ni = ni_values[ni_values > 0]
        
        if len(valid_ni) < 2:
            return None
        
        # CAGR = (ending / beginning) ^ (1 / years) - 1
        oldest = valid_ni.iloc[-1]
        newest = valid_ni.iloc[0]
        n = len(valid_ni) - 1
        
        if oldest <= 0:
            return None
        
        cagr = (newest / oldest) ** (1 / n) - 1
        # Cap at ±50% to avoid outliers
        return max(-0.50, min(0.50, cagr))
    except Exception as e:
        logger.debug(f"CAGR calculation failed: {e}")
        return None


def calculate_historical_growth(series, years=5):
    """Calculate CAGR from historical data (last N years)."""
    try:
        # Filter out non-positive values
        valid_series = series[series > 0].dropna()
        if len(valid_series) < 2:
            return None
        
        # Get oldest and newest values
        oldest = valid_series.iloc[-1]  # oldest (first)
        newest = valid_series.iloc[0]   # newest (last)
        n_years = len(valid_series) - 1
        
        if oldest <= 0:
            return None
        
        cagr = (newest / oldest) ** (1 / n_years) - 1
        # Cap growth between -50% and 50% to avoid outliers
        return max(-0.50, min(0.50, cagr))
    except:
        return None


def get_sector_defaults(stock_info):
    """Get sector-specific defaults."""
    sector, _ = get_valuation_sector_name(stock_info)
    return SECTOR_MULTIPLES.get(sector, SECTOR_MULTIPLES["Default"])


def _healthcare_text_blob(stock_info) -> str:
    """Build a lower-cased text blob for healthcare subtype classification."""
    fields = [
        stock_info.get("sector", ""),
        stock_info.get("industry", ""),
        stock_info.get("longBusinessSummary", ""),
        stock_info.get("shortName", ""),
        stock_info.get("longName", ""),
        stock_info.get("displayName", ""),
        stock_info.get("name", ""),
    ]
    return " ".join(str(value or "").strip().lower() for value in fields if value)


def _matches_any(text: str, tokens) -> bool:
    return any(token in text for token in tokens)


def _classify_healthcare_subtype(stock_info):
    """Classify Healthcare names into narrower valuation buckets."""
    text = _healthcare_text_blob(stock_info)
    sector_lower = str(stock_info.get("sector", "") or "").strip().lower()
    industry_lower = str(stock_info.get("industry", "") or "").strip().lower()

    if not any(token in text for token in ["health", "pharma", "medical", "biotech", "life science", "diagnostic"]):
        return None, None

    managed_care_tokens = [
        "managed care",
        "healthcare plans",
        "health care plans",
        "health insurance",
        "health plans",
        "health plan",
        "payer",
        "pbm",
        "pharmacy benefit",
        "pharmacy services",
        "evernorth",
        "aetna",
        "caremark",
        "unitedhealthcare",
        "optum",
        "health services",
        "medicare advantage",
        "medicaid",
    ]
    if _matches_any(text, managed_care_tokens) or (
        "health" in sector_lower
        and _matches_any(industry_lower, ["health insurance", "healthcare plans", "health care plans", "managed care", "payer", "pbm"])
        and _matches_any(text, ["health services", "health plan", "healthcare plans", "pharmacy benefit", "pbm", "caremark", "evernorth", "optum", "unitedhealthcare", "aetna", "medicare advantage", "medicaid"])
    ):
        return "Managed Care", "managed_care_pbm: health plans / PBM / services hybrid"

    biotech_tokens = ["biotechnology", "biopharmaceutical", "bio-tech", "biotech"]
    if _matches_any(text, biotech_tokens):
        return "Biotech", "biotech / biopharmaceutical model"

    pharma_industry_tokens = [
        "drug manufacturer",
        "drug manufacturers",
        "pharmaceutical",
        "pharma",
        "specialty & generic drug",
    ]
    if _matches_any(industry_lower, pharma_industry_tokens):
        return "Pharmaceuticals", "industry classification indicates pharmaceutical manufacturer"

    medical_device_tokens = [
        "medical device",
        "medical devices",
        "medtech",
        "surgical",
        "robotic",
        "implant",
        "cardiovascular devices",
        "diagnostic equipment",
    ]
    if _matches_any(text, medical_device_tokens):
        return "Medical Devices", "medical device / medtech business"

    life_science_tokens = [
        "life science tools",
        "life sciences tools",
        "laboratory",
        "lab products",
        "research tools",
        "diagnostics",
        "diagnostic services",
        "clinical laboratory",
        "laboratory services",
        "analytical instruments",
        "consumables",
        "instruments",
    ]
    if _matches_any(text, life_science_tokens):
        return "Life Science Tools", "life science tools / diagnostics services model"

    # Health IT: software/platform companies that serve healthcare via technology
    # (EMR, telehealth, digital health) rather than direct patient care.
    # Gross margin gate (≥30%) prevents pure clinic/hospital chains from matching.
    health_it_industry_tokens = [
        "health information", "healthcare technology", "health technology",
        "health information services", "healthcare it",
    ]
    health_it_description_tokens = [
        "electronic health record", "electronic medical record", "emr", "ehr",
        "digital health", "digital healthcare", "telehealth", "telemedicine",
        "health technology", "healthtech", "practice management",
        "health software", "health platform", "virtual care", "healthcare software",
    ]
    _gross_margin = stock_info.get("grossMargins") or 0.0
    if _matches_any(industry_lower, health_it_industry_tokens) or (
        _matches_any(text, health_it_description_tokens) and _gross_margin >= 0.30
    ):
        return "Healthcare IT", "health IT / digital health platform (EMR, telehealth, health software)"

    provider_tokens = [
        "hospital",
        "hospitals",
        "healthcare facilities",
        "health care facilities",
        "provider",
        "providers",
        "acute care",
        "medical center",
        "medical centres",
        "medical centers",
    ]
    if _matches_any(text, provider_tokens):
        return "Healthcare Providers", "hospital / provider model"

    distribution_tokens = [
        "pharmaceutical distribution",
        "drug distribution",
        "medical distribution",
        "wholesale distribution",
        "wholesale drug",
        "healthcare distribution",
        "pharmacy distribution",
    ]
    if _matches_any(text, distribution_tokens):
        return "Pharmaceutical Distribution", "low-margin pharmaceutical distribution model"

    pharma_tokens = ["drug manufacturer", "drug manufacturers", "pharmaceutical", "pharma", "therapeutic", "medicines", "generics"]
    if _matches_any(text, pharma_tokens):
        return "Pharmaceuticals", "pharmaceutical manufacturer / drug development"

    return None, None


def _classify_financial_subtype(stock_info):
    """Classify non-bank Financial names into narrower valuation buckets."""
    fields = [
        stock_info.get("sector", ""),
        stock_info.get("industry", ""),
        stock_info.get("longBusinessSummary", ""),
        stock_info.get("shortName", ""),
        stock_info.get("longName", ""),
        stock_info.get("displayName", ""),
        stock_info.get("name", ""),
    ]
    text = " ".join(str(value or "").strip().lower() for value in fields if value)
    industry_lower = str(stock_info.get("industry", "") or "").strip().lower()
    summary_lower = str(stock_info.get("longBusinessSummary", "") or "").strip().lower()

    if not any(token in text for token in ["financial", "market data", "ratings", "index", "exchange"]):
        return None, None

    bank_insurer_exclusion_tokens = [
        "banks",
        "regional banks",
        "diversified banks",
        "insurance",
        "reinsurance",
        "property & casualty",
        "life insurance",
        "deposits",
        "loan book",
        "net interest income",
        "underwriting",
        "mortgage",
        "lending",
    ]
    if _matches_any(industry_lower, bank_insurer_exclusion_tokens) or _matches_any(summary_lower, bank_insurer_exclusion_tokens):
        return None, None

    strong_industry_tokens = [
        "financial data & stock exchanges",
        "financial data and stock exchanges",
        "financial data",
        "credit ratings",
        "rating agency",
        "stock exchanges",
        "index provider",
        "market data",
        "information services",
    ]
    supporting_tokens = [
        "ratings",
        "credit ratings",
        "index",
        "indices",
        "benchmark",
        "market data",
        "pricing data",
        "reference data",
        "workflow solutions",
        "information services",
        "analytics",
        "research platform",
        "subscription",
        "exchange data",
    ]

    strong_signal = _matches_any(industry_lower, strong_industry_tokens)
    supporting_signal = _matches_any(text, supporting_tokens)
    if strong_signal and supporting_signal:
        return "Financial Data Services", "financial_data_services: index/ratings/market-data platform"

    return None, None


def _classify_government_defense_it_subtype(stock_info):
    """Classify government-services defense IT peers into direct and hybrid subtypes."""
    fields = [
        stock_info.get("sector", ""),
        stock_info.get("industry", ""),
        stock_info.get("longBusinessSummary", ""),
        stock_info.get("shortName", ""),
        stock_info.get("longName", ""),
        stock_info.get("displayName", ""),
        stock_info.get("name", ""),
        stock_info.get("symbol", ""),
        stock_info.get("ticker", ""),
    ]
    text = " ".join(str(value or "").strip().lower() for value in fields if value)
    industry_lower = str(stock_info.get("industry", "") or "").strip().lower()
    sector_lower = str(stock_info.get("sector", "") or "").strip().lower()

    ticker_hint = str(
        stock_info.get("symbol")
        or stock_info.get("ticker")
        or stock_info.get("shortName")
        or ""
    ).upper()

    direct_tickers = {"LDOS", "CACI", "SAIC", "BAH", "AMTM", "VVX"}
    hybrid_tickers = {"PSN", "KBR"}
    negative_control_tickers = {"LMT", "NOC", "RTX"}

    if ticker_hint in direct_tickers:
        return "Gov Defense IT", "gov_defense_it_direct: explicit peer mapping"
    if ticker_hint in hybrid_tickers:
        return "Gov Defense IT Hybrid", "gov_defense_it_hybrid: explicit peer mapping"
    if ticker_hint in negative_control_tickers:
        return None, None

    if not any(token in text for token in ["government", "federal", "defense", "public sector", "national security", "homeland"]):
        return None, None

    manufacturer_tokens = [
        "fighter aircraft",
        "aircraft manufacturer",
        "missile systems",
        "missiles",
        "engines",
        "aerosystems",
        "aerospace products",
        "propulsion",
    ]
    if any(token in text for token in manufacturer_tokens):
        return None, None

    direct_entity_tokens = [
        "leidos",
        "caci",
        "saic",
        "booz allen",
        "v2x",
        "amtm",
        "federal services",
        "defense it",
        "mission software",
        "digital modernization",
        "intelligence community",
    ]
    hybrid_entity_tokens = [
        "parsons",
        "kbr",
        "government and commercial",
        "commercial and government",
        "infrastructure services",
        "engineering services",
        "space and intelligence",
        "energy transition",
    ]

    has_direct = any(token in text for token in direct_entity_tokens)
    has_hybrid = any(token in text for token in hybrid_entity_tokens)
    has_gov_services_context = any(
        token in text
        for token in [
            "government services",
            "federal government",
            "government customers",
            "public sector",
            "defense",
            "national security",
            "homeland security",
        ]
    )

    if has_hybrid:
        return "Gov Defense IT Hybrid", "gov_defense_it_hybrid: mixed government/commercial services"

    if has_direct or (has_gov_services_context and ("information technology" in industry_lower or "it services" in industry_lower or "consulting" in industry_lower or "industrial" in sector_lower)):
        return "Gov Defense IT", "gov_defense_it_direct: government services and defense IT profile"

    return None, None


def get_valuation_sector_name(stock_info):
    """Map raw reported sector into a valuation-oriented sector when warranted."""
    raw_sector = stock_info.get("sector", "Default") or "Default"

    try:
        # Non-equity instruments: detect before any sector logic
        quote_type = str(stock_info.get("quoteType") or "").upper()
        if quote_type == "CRYPTOCURRENCY":
            return "Cryptocurrency", "cryptocurrency instrument"
        # Physical commodity ETPs may report quoteType=EQUITY with no sector.
        # Detect by name when there is no sector and the name indicates a commodity product.
        _all_names = (
            str(stock_info.get("longName") or "").lower()
            + " "
            + str(stock_info.get("shortName") or "").lower()
            + " "
            + str(stock_info.get("category") or "").lower()
        )
        if not stock_info.get("sector") and any(tok in _all_names for tok in
               ["gold", "silver", "platinum", "palladium", "precious metal",
                "xetra-gold", "physical metal"]):
            return "Physical Commodity ETF", "physical commodity ETP (no sector + commodity name)"

        if quote_type == "ETF" and any(tok in _all_names for tok in
               ["gold", "silver", "platinum", "palladium", "precious metal",
                "commodity", "physical metal"]):
            return "Physical Commodity ETF", "physical commodity ETF"

        sector_lower = str(raw_sector).lower()
        industry_lower = str(stock_info.get("industry", "") or "").lower()
        summary_lower = str(stock_info.get("longBusinessSummary", "") or "").lower()
        market_cap = stock_info.get("marketCap") or 0
        gross_margin = stock_info.get("grossMargins") or 0.0
        operating_margin = stock_info.get("operatingMargins") or 0.0
        revenue_growth = stock_info.get("revenueGrowth") or 0.0
        ad_platform_tokens = [
            "advertising platform",
            "advertising software",
            "digital advertising",
            "mobile advertising",
            "ad tech",
            "ad-tech",
            "demand-side platform",
            "supply-side platform",
            "programmatic advertising",
            "app monetization",
            "monetization platform",
            "marketing platform",
            "user acquisition",
            "bidding",
            "real-time bidding",
        ]
        has_ad_platform_model = any(tok in summary_lower for tok in ad_platform_tokens) or any(
            tok in industry_lower for tok in ["advertising", "marketing services"]
        )

        # Systematic split: banks/insurers stay in Financials policy;
        # non-bank financial-services businesses get Financial Services policy.
        if "financial" in sector_lower:
            # Reclassify crypto miners / compute-infrastructure companies
            # misclassified as Financial Services by yfinance
            compute_mining_tokens = [
                "bitcoin", "cryptocurrency", "crypto mining", "digital asset mining",
                "proof of work", "high performance computing", "hpc", "gpu cluster",
                "ai compute", "compute infrastructure", "data center computing",
            ]
            if any(tok in summary_lower for tok in compute_mining_tokens) or any(
                tok in industry_lower for tok in ["crypto", "bitcoin", "blockchain", "digital asset"]
            ):
                return "Technology", "compute infrastructure / crypto mining (reclassified from Financial Services)"

            strong_bank_industry_tokens = [
                "banks",
                "regional banks",
                "diversified banks",
                "banks - diversified",
                "banks - regional",
                "insurance",
                "reinsurance",
                "property & casualty",
                "life insurance",
            ]
            bank_tokens = [
                "bank",
                "banc",
                "savings",
                "trust",
                "mortgage",
                "consumer finance",
                "lending",
                "regional banks",
                "diversified banks",
            ]
            insurance_tokens = [
                "insurance",
                "reinsurance",
                "property & casualty",
                "life insurance",
            ]
            fs_tokens = [
                "financial data",
                "financial data & stock exchanges",
                "ratings",
                "credit ratings",
                "index",
                "exchange",
                "stock exchanges",
                "market data",
                "asset management",
                "investment management",
                "capital markets",
                "brokerage",
                "information services",
                "analytics",
                "payment processing",
                "credit card",
                "transaction services",
            ]

            is_bank_or_insurer = any(tok in industry_lower for tok in bank_tokens + insurance_tokens) or any(
                tok in summary_lower for tok in ["deposits", "loan book", "net interest income", "underwriting", "insurance"]
            )
            is_nonbank_fs = any(tok in industry_lower for tok in fs_tokens) or any(tok in summary_lower for tok in fs_tokens)

            is_strong_bank_or_insurer = any(tok in industry_lower for tok in strong_bank_industry_tokens)
            if is_strong_bank_or_insurer:
                return "Financials", "bank/insurance policy (industry gate)"

            # Payment network detection (platform businesses, not intermediaries)
            ticker_hint = str(stock_info.get("symbol") or stock_info.get("ticker") or "").upper()
            payment_network_tickers = {"V", "MA"}  # Visa, Mastercard
            
            payment_network_tokens = ["payment networks", "payment processing", "credit card networks", "transaction services"]
            name_tokens = ["visa", "mastercard"]
            is_payment_network = (
                ticker_hint in payment_network_tickers
                or any(tok in industry_lower for tok in payment_network_tokens)
                or any(tok in summary_lower for tok in ["payment processing", "transaction fees", "payment network", "credit card"])
                or any(tok in _all_names for tok in name_tokens)
            )
            if is_payment_network:
                return "Payment Networks", "platform business model: payment network operator"

            financial_subsector, financial_reason = _classify_financial_subtype(stock_info)
            if financial_subsector:
                return financial_subsector, financial_reason

            if is_nonbank_fs:
                return "Financial Services", "non-bank financial services policy"
            if is_bank_or_insurer:
                return "Financials", "bank/insurance policy"

        # Consumer internet platform / hyperscaler override.
        if "consumer" in sector_lower:
            internet_platform_industry = any(
                token in industry_lower
                for token in [
                    "internet retail",
                    "internet & direct marketing retail",
                    "e-commerce",
                    "online retail",
                ]
            )
            hyperscaler_or_platform = any(
                token in summary_lower
                for token in [
                    "amazon web services",
                    "aws",
                    "cloud",
                    "advertising",
                    "subscriptions",
                    "marketplace",
                    "platform",
                ]
            )

            if internet_platform_industry and market_cap >= 200_000_000_000 and (
                hyperscaler_or_platform or gross_margin >= 0.35 or operating_margin >= 0.08
            ):
                return "Communication Services", "consumer internet platform/hyperscaler override"

            # Tesla-like category: only promote auto makers when they actually look
            # like high-margin/high-growth innovation platforms, not traditional autos.
            auto_innovation = "auto manufacturer" in industry_lower and any(
                token in summary_lower
                for token in ["autonomous", "self-driving", "software", "energy storage", "robotaxi"]
            )
            if auto_innovation and market_cap >= 100_000_000_000 and (
                gross_margin >= 0.25 or operating_margin >= 0.10 or revenue_growth >= 0.15
            ):
                return "Technology", "innovative vehicle platform override"

        if has_ad_platform_model and (
            "technolog" in sector_lower
            or "communication" in sector_lower
            or "software" in industry_lower
            or "internet" in industry_lower
        ):
            return "Digital Advertising Platform", "ad-tech/platform business model"

        # Telecom detection: If labeled as Communication Services but is actually a
        # traditional telecom operator (not a tech platform), route to the appropriate
        # sub-sector config (Wireless Telecom / Integrated Telecom / Cable/Broadband).
        if "communication" in sector_lower:
            is_traditional_telecom = any(
                token in industry_lower
                for token in [
                    "telecom",
                    "telecommunications",
                    "wireless communications",
                    "telecom services",
                    "regional telecom",
                ]
            )
            is_not_platform = not any(
                token in summary_lower
                for token in [
                    "platform",
                    "streaming",
                    "advertising",
                    "social media",
                    "video streaming",
                    "content distribution",
                    "digital platform",
                ]
            ) and (gross_margin is None or gross_margin < 0.50) and (
                operating_margin is None or operating_margin < 0.15
            )

            # Traditional telecoms: high debt, lower growth expectations
            debt_to_equity = stock_info.get("debtToEquity") or 0
            is_high_leverage = debt_to_equity > 1.5

            if is_traditional_telecom and (is_not_platform or is_high_leverage):
                # Sub-classify by network type so each bucket gets appropriate multiples.
                is_wireless = any(token in industry_lower for token in [
                    "wireless communications", "wireless telecommunication",
                    "wireless services", "mobile",
                ])
                if is_wireless:
                    return "Wireless Telecom", "wireless/mobile network operator"
                return "Integrated Telecom", "integrated/wireline telecom operator"

            # Cable/Broadband: infrastructure-heavy, not labeled telecom (e.g. CMCSA).
            is_cable = any(token in industry_lower for token in [
                "cable & satellite", "cable and satellite", "cable television", "broadband",
            ])
            if is_cable:
                return "Cable/Broadband", "cable/broadband infrastructure operator"

        # Healthcare taxonomy: managed care, biotech, pharma, medtech, tools, providers, distribution.
        if "health" in sector_lower or "pharma" in sector_lower or "medical" in sector_lower or "biotech" in sector_lower:
            healthcare_sector, healthcare_reason = _classify_healthcare_subtype(stock_info)
            if healthcare_sector:
                return healthcare_sector, healthcare_reason

        # Materials: Specialty Chemicals detection.
        # Industrial gas / specialty chemical companies (LIN, APD, SHW, ECL) carry stable
        # long-term contracts and predictable FCF.  The commodity-mining policy (no DCF,
        # 5 % growth cap) is too restrictive and should not apply to them.
        if "material" in sector_lower:
            specialty_industries = [
                "specialty chemicals",
                "industrial gases",
                "diversified chemicals",
                "agricultural chemicals",
            ]
            if any(token in industry_lower for token in specialty_industries):
                return "Specialty Chemicals", "specialty/industrial chemicals (not commodity-exposed)"

        # Industrials: Airlines detection.
        # Airlines have extreme fleet leverage and cyclical revenue; DCF produces
        # unreliable results across the cycle.  Use EV/EBITDA and P/E only.
        if "industrial" in sector_lower:
            if "airlines" in industry_lower:
                return "Airlines", "airline operator (cyclical, high fleet leverage)"

        # Industrials: Government services defense IT detection.
        # Services-led government/defense IT contractors should not inherit
        # aerospace manufacturer assumptions.
        if "industrial" in sector_lower:
            gov_sector, gov_reason = _classify_government_defense_it_subtype(stock_info)
            if gov_sector:
                return gov_sector, gov_reason

        # Industrials: Aerospace & Defense detection.
        # Aircraft manufacturers and defense contractors (Boeing, Airbus, Lockheed,
        # Raytheon, Safran) operate on long government/program cycles with milestone-
        # based revenue.  EV/EBITDA is more stable than DCF or P/E across program
        # delays and delivery lumps.  The 'Aerospace' config already exists but was
        # only reachable via direct name match; detection is needed here.
        if "industrial" in sector_lower:
            if "aerospace" in industry_lower or "defense" in industry_lower:
                return "Aerospace", "aerospace/defense manufacturer (long-cycle program revenue)"

        # Industrials: Engineering & Construction detection.
        # Large construction and infrastructure contractors (Vinci, ACS, Ferrovial,
        # Skanska) carry project-based lumpiness and construction-cycle risk.
        # The 'Construction' config already exists but was unreachable.
        if "industrial" in sector_lower:
            construction_tokens = [
                "engineering & construction",
                "infrastructure operations",
                "engineering and construction",
            ]
            if any(token in industry_lower for token in construction_tokens):
                return "Construction", "engineering & construction contractor"

        # Technology: Software sub-sector.
        # Pure software companies (Application Software, Infrastructure Software)
        # are asset-light with subscription/recurring revenue and very clean FCF.
        # P/E and DCF are equally reliable; separating them from hardware and semis
        # avoids applying cyclical or capex-heavy assumptions to asset-light names.
        if "technolog" in sector_lower:
            if "software" in industry_lower:
                return "Software", "dedicated software company (asset-light, subscription model)"

        # Technology: Semiconductors sub-sector.
        # Chip designers (AMD, NVIDIA) and equipment makers (ASML) face capacity
        # cycles that make DCF extrapolation of upcycle growth unreliable.
        # EV/EBITDA is more cycle-stable and gets the highest weight.
        if "technolog" in sector_lower:
            semiconductor_tokens = [
                "semiconductors",
                "semiconductor equipment",
            ]
            if any(token in industry_lower for token in semiconductor_tokens):
                return "Semiconductors", "semiconductor company (capex-heavy, cyclical)"

    except Exception:
        pass

    return raw_sector, None


def get_normalized_fcf_per_share(stock, stock_info, raw_fcf_per_share):
    """Return a normalized FCF/share estimate for reinvestment-heavy platforms.

    This is intended for AMZN-like businesses where current reported FCF can be
    temporarily depressed by growth capex despite strong operating cash flow.
    """
    sector, override_reason = get_valuation_sector_name(stock_info)
    if override_reason != "consumer internet platform/hyperscaler override":
        return raw_fcf_per_share, None

    try:
        shares = stock_info.get("sharesOutstanding")
        total_revenue = stock_info.get("totalRevenue") or 0
        gross_margin = stock_info.get("grossMargins") or 0.0
        operating_margin = stock_info.get("operatingMargins") or 0.0
        if not shares or shares <= 0:
            return raw_fcf_per_share, None

        cashflow = stock.cashflow
        if cashflow is None:
            return raw_fcf_per_share, None

        if "Operating Cash Flow" not in cashflow.index or "Capital Expenditure" not in cashflow.index:
            return raw_fcf_per_share, None

        ocf_series = cashflow.loc["Operating Cash Flow"].dropna()
        capex_series = cashflow.loc["Capital Expenditure"].dropna()
        if ocf_series.empty or capex_series.empty:
            return raw_fcf_per_share, None

        current_ocf = float(ocf_series.iloc[0])
        current_capex = abs(float(capex_series.iloc[0]))
        if current_ocf <= 0:
            return raw_fcf_per_share, None

        capex_to_ocf = current_capex / current_ocf if current_ocf > 0 else 0.0

        # Only normalize when capex intensity is unusually high but the business
        # still has platform-like margins indicating a large growth-capex component.
        if capex_to_ocf < 0.65 or gross_margin < 0.35 or operating_margin < 0.08:
            return raw_fcf_per_share, None

        # Approximate maintenance capex conservatively as the smaller of:
        # - 45% of total capex
        # - 6% of revenue
        # This preserves a large growth-capex bucket instead of crediting all capex back.
        maintenance_capex = min(current_capex * 0.45, total_revenue * 0.06 if total_revenue > 0 else current_capex * 0.45)
        normalized_fcf_total = max(current_ocf - maintenance_capex, 0.0)

        # Keep the normalized baseline bounded so it does not exceed 75% of EBITDA.
        ebitda = stock_info.get("ebitda") or 0
        if ebitda and ebitda > 0:
            normalized_fcf_total = min(normalized_fcf_total, ebitda * 0.75)

        normalized_fcf_per_share = normalized_fcf_total / shares if shares > 0 else None
        if not validate_data(normalized_fcf_per_share):
            return raw_fcf_per_share, None

        if validate_data(raw_fcf_per_share):
            normalized_fcf_per_share = max(float(raw_fcf_per_share), float(normalized_fcf_per_share))

        details = {
            "applied": True,
            "reason": "reinvestment-heavy platform normalization",
            "raw_fcf_per_share": raw_fcf_per_share,
            "normalized_fcf_per_share": normalized_fcf_per_share,
            "operating_cash_flow": current_ocf,
            "capital_expenditure": current_capex,
            "capex_to_ocf": capex_to_ocf,
            "maintenance_capex": maintenance_capex,
            "valuation_sector": sector,
        }
        return normalized_fcf_per_share, details

    except Exception:
        return raw_fcf_per_share, None


def _normalize_telecom_fcf_per_share(stock, stock_info, raw_fcf_per_share):
    """Normalize FCF/share for capex-heavy telecom/cable operators.

    Uses the 3–5Y average capex/revenue ratio × current revenue rather than
    point-in-time capex, preventing lumpy spectrum auctions or network build
    cycles from distorting the DCF base.
    Returns (normalized_fcf_per_share, details_dict | None).
    """
    try:
        shares = stock_info.get("sharesOutstanding")
        total_revenue = stock_info.get("totalRevenue") or 0
        if not shares or shares <= 0 or total_revenue <= 0:
            return raw_fcf_per_share, None

        cashflow = stock.cashflow
        financials = stock.financials
        if cashflow is None or financials is None:
            return raw_fcf_per_share, None

        required = {"Operating Cash Flow", "Capital Expenditure"}
        if not required.issubset(set(cashflow.index)):
            return raw_fcf_per_share, None
        if "Total Revenue" not in financials.index:
            return raw_fcf_per_share, None

        ocf_series = cashflow.loc["Operating Cash Flow"].dropna()
        capex_series = cashflow.loc["Capital Expenditure"].dropna()
        rev_series = financials.loc["Total Revenue"].dropna()

        common = sorted(ocf_series.index.intersection(
            capex_series.index).intersection(rev_series.index))
        if len(common) < 2:
            return raw_fcf_per_share, None

        recent = common[-5:]  # up to 5 most recent annual periods
        ratios = [
            abs(float(capex_series[c])) / float(rev_series[c])
            for c in recent
            if float(rev_series[c]) > 0
        ]
        if not ratios:
            return raw_fcf_per_share, None

        avg_capex_ratio = sum(ratios) / len(ratios)
        normalized_capex = avg_capex_ratio * total_revenue
        current_ocf = float(ocf_series.iloc[0])
        if current_ocf <= 0:
            return raw_fcf_per_share, None

        normalized_fcf = current_ocf - normalized_capex
        normalized_fcf_ps = normalized_fcf / shares

        if not validate_data(normalized_fcf_ps):
            return raw_fcf_per_share, None

        # Safety: reject normalization if it moves FCF by more than 3× in either direction.
        if raw_fcf_per_share is not None and raw_fcf_per_share != 0:
            if abs(normalized_fcf_ps / raw_fcf_per_share) > 3.0:
                return raw_fcf_per_share, None

        details = {
            "applied": True,
            "reason": "capex-heavy telecom/cable: 3–5Y normalized capex/revenue",
            "raw_fcf_per_share": raw_fcf_per_share,
            "normalized_fcf_per_share": normalized_fcf_ps,
            "years_used": len(ratios),
            "avg_capex_to_revenue": avg_capex_ratio,
            "normalized_capex": normalized_capex,
            "current_ocf": current_ocf,
        }
        return normalized_fcf_ps, details

    except Exception:
        return raw_fcf_per_share, None


def is_mega_cap_software(stock_info):
    """Heuristic flag for mega-cap software/platform and AI infrastructure businesses.

    Used to apply less conservative DCF defaults than broad Technology averages.
    """
    try:
        sector = str(stock_info.get("sector", "") or "").lower()
        industry = str(stock_info.get("industry", "") or "").lower()
        market_cap = stock_info.get("marketCap") or 0
        gross_margin = stock_info.get("grossMargins")
        operating_margin = stock_info.get("operatingMargins")

        sector_ok = ("technolog" in sector) or ("communication" in sector)
        industry_ok = any(
            token in industry
            for token in [
                "software",
                "application",
                "infrastructure",
                "internet",
                "cloud",
                "platform",
                "semiconductor",
                "semiconductors",
                "chip",
                "gpu",
                "ai",
                "accelerator",
                "fabless",
            ]
        )
        market_cap_ok = market_cap >= 200_000_000_000  # 200B+

        # Quality gate: require at least one strong margin signal.
        quality_ok = (
            (gross_margin is not None and gross_margin >= 0.60)
            or (operating_margin is not None and operating_margin >= 0.20)
        )

        return bool(sector_ok and industry_ok and market_cap_ok and quality_ok)
    except Exception:
        return False


def is_mega_cap_ai_infra(stock_info):
    """Heuristic flag for mega-cap AI infrastructure / semiconductor leaders.

    This is a stricter subset used for H-model-like growth handling where
    near-term growth can materially exceed WACC before fading to steady state.
    """
    try:
        industry = str(stock_info.get("industry", "") or "").lower()
        market_cap = stock_info.get("marketCap") or 0
        gross_margin = stock_info.get("grossMargins")
        operating_margin = stock_info.get("operatingMargins")
        revenue_growth = stock_info.get("revenueGrowth")
        earnings_growth = stock_info.get("earningsGrowth")

        industry_ok = any(
            token in industry
            for token in [
                "semiconductor",
                "semiconductors",
                "chip",
                "gpu",
                "accelerator",
                "fabless",
            ]
        )
        market_cap_ok = market_cap >= 200_000_000_000
        quality_ok = (
            (gross_margin is not None and gross_margin >= 0.65)
            or (operating_margin is not None and operating_margin >= 0.25)
        )
        growth_ok = (
            (revenue_growth is not None and revenue_growth >= 0.30)
            or (earnings_growth is not None and earnings_growth >= 0.30)
        )

        return bool(industry_ok and market_cap_ok and quality_ok and growth_ok)
    except Exception:
        return False


def _resolve_subunit(currency: str):
    """Return (parent_currency, scale) where 1 unit of `currency` = scale units of parent."""
    if currency in SUBUNIT_CURRENCIES:
        sub = SUBUNIT_CURRENCIES[currency]
        return sub["parent"], sub["scale_to_parent"]
    return currency, 1.0


def get_fx_rate(from_currency: str, to_currency: str):
    """Get FX rate so that 1 unit of from_currency equals X units of to_currency.

    Handles sub-unit currencies (GBp, GBX, ZAC) defined in config.SUBUNIT_CURRENCIES
    by normalising to parent currencies before any yfinance lookup.
    E.g. get_fx_rate('GBP', 'GBp') -> 100.0  (1 GBP = 100 pence)
         get_fx_rate('GBp', 'GBP') -> 0.01
    """
    if not from_currency or not to_currency or from_currency == to_currency:
        return 1.0

    # Resolve sub-unit currencies to their parent
    from_parent, from_scale = _resolve_subunit(from_currency)
    to_parent, to_scale = _resolve_subunit(to_currency)

    # If both resolve to the same parent (e.g. GBP <-> GBp), compute algebraically.
    if from_parent == to_parent:
        # 1 from_currency = from_scale parent units
        # 1 to_currency   = to_scale   parent units
        # => 1 from_currency = (from_scale / to_scale) to_currency
        return from_scale / to_scale

    # Look up the parent-pair rate, then apply sub-unit scaling.
    parent_rate = _fetch_yf_fx(from_parent, to_parent)
    if parent_rate is None:
        return None
    # 1 from_currency = from_scale from_parent = from_scale * parent_rate to_parent
    # 1 to_currency   = to_scale   to_parent
    # => 1 from_currency = (from_scale * parent_rate / to_scale) to_currency
    return (from_scale * parent_rate) / to_scale


def _fetch_yf_fx(from_ccy: str, to_ccy: str):
    """Raw yfinance FX lookup between two standard ISO currency codes."""
    try:
        pair = f"{from_ccy}{to_ccy}=X"
        data = yf.download(pair, period="5d", progress=False, auto_adjust=False)
        if not data.empty and "Close" in data.columns:
            series = data["Close"]
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
            series = series.dropna()
            if not series.empty and series.iloc[-1] > 0:
                return float(series.iloc[-1])
    except Exception:
        pass

    try:
        pair = f"{to_ccy}{from_ccy}=X"
        data = yf.download(pair, period="5d", progress=False, auto_adjust=False)
        if not data.empty and "Close" in data.columns:
            series = data["Close"]
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
            series = series.dropna()
            if not series.empty and series.iloc[-1] > 0:
                return 1.0 / float(series.iloc[-1])
    except Exception:
        pass

    return None


# =====================
# CORE VALUATION LOGIC
# =====================

def calculate_fcf_per_share(stock, stock_info):
    """
    Extract FCF per share using the GAAP definition: Operating Cash Flow - CapEx.

    The 'Free Cash Flow' row that yfinance pre-computes (both in cashflow and
    info['freeCashflow']) uses a non-standard formula:
        NI + all non-cash add-backs - CapEx
    This OMITS working-capital changes, which can be massively negative for
    post-merger companies (content spend, large payables swings) or companies
    undergoing restructuring.  For WBD-type cases the non-standard value is 5-6×
    the GAAP value, leading to wildly inflated DCF outputs.

    Priority:
      1. OCF − CapEx from the cashflow statement  (GAAP, always preferred)
      2. Pre-computed 'Free Cash Flow' row if OCF-CapEx is unavailable AND the two
         values are within 2× of each other (they agree for most normal companies)
      3. info['freeCashflow'] as last resort when no cashflow statement is available
    """
    shares = stock_info.get("sharesOutstanding")
    provenance = {
        "source_method": "unresolved",
        "source_freshness": "unknown",
        "trust_level": 0.0,
        "quality_assessment": "insufficient_data",
        "ocf_latest": None,
        "capex_latest": None,
        "row_fcf_latest": None,
        "info_fcf": stock_info.get("freeCashflow"),
        "shares_outstanding": shares,
        "selected_fcf_total": None,
        "selected_fcf_per_share": None,
    }
    if not shares or shares <= 0:
        return None, provenance

    info_fcf = stock_info.get("freeCashflow")

    try:
        cashflow = stock.cashflow

        # ── Primary path: GAAP OCF − CapEx ──────────────────────────────────
        ocf_capex_fcf = None
        if "Operating Cash Flow" in cashflow.index and "Capital Expenditure" in cashflow.index:
            ocf_series   = cashflow.loc["Operating Cash Flow"].dropna()
            capex_series = cashflow.loc["Capital Expenditure"].dropna()
            if len(ocf_series) > 0 and len(capex_series) > 0:
                ocf   = float(ocf_series.iloc[0])
                capex = abs(float(capex_series.iloc[0]))  # stored as negative
                ocf_capex_fcf = ocf - capex
            provenance["ocf_latest"] = ocf
            provenance["capex_latest"] = capex

        # ── Cross-check: pre-computed 'Free Cash Flow' row ───────────────────
        row_fcf = None
        if "Free Cash Flow" in cashflow.index:
            fcf_values = cashflow.loc["Free Cash Flow"].dropna()
            if len(fcf_values) > 0:
                row_fcf = float(fcf_values.iloc[0])
                provenance["row_fcf_latest"] = row_fcf

        # ── Decision logic ────────────────────────────────────────────────────
        if ocf_capex_fcf is not None:
            # Always prefer GAAP OCF-CapEx.  Only fall back to the pre-computed
            # row if OCF-CapEx is non-positive AND the row value is positive.
            if ocf_capex_fcf > 0:
                selected = ocf_capex_fcf / shares
                provenance.update({
                    "source_method": "ocf_minus_capex",
                    "source_freshness": "latest_cashflow_period",
                    "trust_level": 1.0,
                    "quality_assessment": "gaap_primary",
                    "selected_fcf_total": ocf_capex_fcf,
                    "selected_fcf_per_share": selected,
                })
                return selected, provenance
            elif row_fcf is not None and row_fcf > 0:
                # OCF-CapEx ≤ 0 but the pre-computed row is positive.
                # Accept only if they are reasonably close (within 2×) —
                # large divergence signals the pre-computed row is inflated.
                if row_fcf <= abs(ocf_capex_fcf) * 2.0 + 1e6:  # also handles ocf_capex near 0
                    selected = row_fcf / shares
                    provenance.update({
                        "source_method": "free_cash_flow_row_crosscheck",
                        "source_freshness": "latest_cashflow_period",
                        "trust_level": 0.7,
                        "quality_assessment": "crosschecked_fallback",
                        "selected_fcf_total": row_fcf,
                        "selected_fcf_per_share": selected,
                    })
                    return selected, provenance
                # Big divergence: company is probably cash-negative; return None
                # so DCF is skipped and EV/EBITDA + P/E carry the valuation.
                provenance.update({
                    "source_method": "free_cash_flow_row_rejected",
                    "source_freshness": "latest_cashflow_period",
                    "trust_level": 0.2,
                    "quality_assessment": "divergence_rejected",
                })
                return None, provenance

        # ── Fallback: pre-computed row (no OCF-CapEx data available) ─────────
        if row_fcf is not None and row_fcf > 0:
            # Cross-check against info_fcf to catch stale/different-period rows
            if validate_data(info_fcf) and info_fcf > 0:
                # Prefer the lower (more conservative) of the two
                selected_total = min(row_fcf, info_fcf)
                selected = selected_total / shares
                provenance.update({
                    "source_method": "free_cash_flow_row_min_with_info",
                    "source_freshness": "latest_cashflow_period",
                    "trust_level": 0.5,
                    "quality_assessment": "row_info_crosscheck",
                    "selected_fcf_total": selected_total,
                    "selected_fcf_per_share": selected,
                })
                return selected, provenance
            selected = row_fcf / shares
            provenance.update({
                "source_method": "free_cash_flow_row_only",
                "source_freshness": "latest_cashflow_period",
                "trust_level": 0.45,
                "quality_assessment": "row_only_fallback",
                "selected_fcf_total": row_fcf,
                "selected_fcf_per_share": selected,
            })
            return selected, provenance

        # ── Last resort: info['freeCashflow'] ─────────────────────────────────
        if validate_data(info_fcf) and info_fcf > 0:
            selected = info_fcf / shares
            provenance.update({
                "source_method": "info_free_cashflow",
                "source_freshness": "point_in_time_info",
                "trust_level": 0.35,
                "quality_assessment": "provider_fallback",
                "selected_fcf_total": info_fcf,
                "selected_fcf_per_share": selected,
            })
            return selected, provenance

        # ── Historical average if all else fails ──────────────────────────────
        if "Free Cash Flow" in cashflow.index:
            fcf_values = cashflow.loc["Free Cash Flow"].dropna().head(10)
            valid_fcf  = fcf_values[fcf_values > 0]
            if len(valid_fcf) > 0:
                selected_total = float(valid_fcf.mean())
                selected = selected_total / shares
                provenance.update({
                    "source_method": "historical_free_cash_flow_average",
                    "source_freshness": "historical_average",
                    "trust_level": 0.25,
                    "quality_assessment": "historical_proxy",
                    "selected_fcf_total": selected_total,
                    "selected_fcf_per_share": selected,
                })
                return selected, provenance

    except Exception as e:
        logger.debug(f"FCF calculation failed: {e}")

    return None, provenance


def calculate_enhanced_wacc(stock, stock_info, net_debt_data=None):
    """
    Improved WACC calculation with adaptive parameters.
    For companies with net negative debt (more cash than debt), uses cost of equity only.
    """
    try:
        # Cost of Equity via CAPM
        beta = stock_info.get("beta", 1.0) or 1.0
        beta = max(0.5, min(2.5, beta))  # Clamp unrealistic betas
        
        # Adaptive risk-free rate based on treasury yields (simplified)
        rf = 0.045  # 10-year yield approximation
        market_risk_premium = 0.055
        Re = rf + beta * market_risk_premium
        
        # Net Debt calculation (prefer shared resolver output)
        net_debt_data = net_debt_data or resolve_net_debt_components(stock, stock_info)
        total_debt = net_debt_data.get("total_debt", 0) or 0
        net_debt = net_debt_data.get("net_debt", 0) or 0
        
        # If company is net-cash positive, WACC = Cost of Equity (no debt component)
        if net_debt <= 0:
            wacc = Re
        else:
            # Cost of Debt (only if company has actual net debt)
            Rd = 0.04  # Default corporate rate
            try:
                fin = stock.financials
                if fin is not None and "Interest Expense" in fin.index:
                    interest_expense = fin.loc["Interest Expense"].iloc[0]
                    # Only use if valid (not NaN and positive)
                    if pd.notna(interest_expense) and interest_expense > 0 and total_debt > 0:
                        Rd = abs(interest_expense) / total_debt
            except:
                pass
            
            # Market values
            shares = stock_info.get("sharesOutstanding", 0)
            price = stock_info.get("currentPrice", 0)
            market_cap = shares * price if shares and price else 0
            
            E = max(market_cap, 1)
            D = max(net_debt, 0)  # Only use positive net debt
            V = E + D
            
            # Adaptive tax rate
            Tc = 0.21  # US default
            
            # WACC formula
            if V > 0:
                wacc = (E / V) * Re + (D / V) * Rd * (1 - Tc)
            else:
                wacc = Re

            # Leverage guard: yfinance freeCashflow is post-interest (FCFE), so
            # the correct discount rate for DCF is the equity cost of capital
            # (Re), not WACC.  Using WACC with FCFE understates equity risk when
            # debt dominates EV.  When equity < 35% of EV, use Re as a floor so
            # thin-equity companies are not discounted at near-debt-cost rates.
            equity_share = E / V if V > 0 else 1.0
            if equity_share < 0.35:
                wacc = max(wacc, Re)

        # Clamp to reasonable range (7% - 20%) — ceiling raised from 15% to 20%
        # so extreme-leverage cases are not clipped at the top.
        return max(0.07, min(0.20, wacc))
        
    except Exception as e:
        logger.debug(f"WACC calculation error: {e}")
        return 0.08  # More reasonable default (was 0.10)


def two_stage_dcf(
    fcf_per_share,
    stage1_growth,
    stage2_growth,
    discount_rate,
    stage1_years=10,
    fade_years=5,
    terminal_fcf_yield=None,
    terminal_blend_weight=0.5,
    return_diagnostics=False,
):
    """Multi-stage DCF: high-growth stage, fade stage, then terminal stage."""

    # Enforce a minimum spread between discount_rate and terminal growth.
    # When WACC is very close to stage2_growth the Gordon-growth perpetuity
    # denominator approaches zero and inflates terminal value to nonsensical levels.
    # Clamp so that discount_rate >= stage2_growth + 0.025 (250 bps floor).
    MIN_SPREAD = 0.025
    discount_rate_input = float(discount_rate)
    spread_before = discount_rate_input - stage2_growth
    spread_adjusted = False
    if discount_rate < stage2_growth + MIN_SPREAD:
        discount_rate = stage2_growth + MIN_SPREAD
        spread_adjusted = True
    
    if not validate_data(fcf_per_share) or discount_rate <= stage2_growth:
        diagnostics = {
            "consistency_status": "invalid_inputs",
            "warning": "Invalid DCF inputs for terminal consistency diagnostics",
            "discount_rate_input": discount_rate_input,
            "discount_rate_used": discount_rate,
            "stage2_growth": stage2_growth,
            "spread_before": spread_before,
            "spread_after": discount_rate - stage2_growth,
            "spread_adjusted": spread_adjusted,
            "spread_floor": MIN_SPREAD,
            "gordon_terminal_multiple": None,
            "yield_terminal_multiple": None,
            "yield_gordon_divergence_pct": None,
        }
        if return_diagnostics:
            return None, diagnostics
        return None
    
    try:
        stage1_years = max(1, int(stage1_years))
        fade_years = max(0, int(fade_years))

        pv_fcf = 0.0
        year = 0
        current_fcf = float(fcf_per_share)

        # Stage 1: explicit high-growth years
        for _ in range(stage1_years):
            year += 1
            current_fcf = current_fcf * (1 + stage1_growth)
            pv_fcf += current_fcf / ((1 + discount_rate) ** year)

        # Stage 2: explicit fade to steady-state growth
        if fade_years > 0:
            for fade_idx in range(1, fade_years + 1):
                year += 1
                t = fade_idx / fade_years
                growth = stage1_growth + (stage2_growth - stage1_growth) * t
                current_fcf = current_fcf * (1 + growth)
                pv_fcf += current_fcf / ((1 + discount_rate) ** year)

        terminal_fcf = current_fcf

        # Gordon-growth terminal value
        terminal_value_gordon = (terminal_fcf * (1 + stage2_growth)) / (discount_rate - stage2_growth)

        terminal_fcf_yield_value = None
        if isinstance(terminal_fcf_yield, (int, float)) and np.isfinite(float(terminal_fcf_yield)) and terminal_fcf_yield > 0:
            terminal_fcf_yield_value = float(terminal_fcf_yield)

        # Exit-yield terminal value (helps avoid overly conservative perpetuity for quality businesses)
        terminal_value = terminal_value_gordon
        if terminal_fcf_yield_value is not None:
            terminal_value_exit = terminal_fcf / terminal_fcf_yield_value
            w = max(0.0, min(1.0, terminal_blend_weight))
            terminal_value = terminal_value_gordon * (1 - w) + terminal_value_exit * w

        pv_terminal = terminal_value / ((1 + discount_rate) ** year)

        gordon_terminal_multiple = None
        yield_terminal_multiple = None
        yield_gordon_divergence_pct = None
        consistency_status = "gordon_only"
        warning = None

        if abs(terminal_fcf) > 1e-9:
            gordon_terminal_multiple = terminal_value_gordon / terminal_fcf

        if terminal_fcf_yield_value is not None:
            yield_terminal_multiple = 1.0 / terminal_fcf_yield_value
            consistency_status = "consistent"
            if gordon_terminal_multiple and abs(gordon_terminal_multiple) > 1e-9:
                yield_gordon_divergence_pct = abs(yield_terminal_multiple - gordon_terminal_multiple) / abs(gordon_terminal_multiple)
                if yield_gordon_divergence_pct > 0.50:
                    consistency_status = "high_mismatch"
                    warning = "Terminal yield and Gordon terminal multiple diverge materially"
                elif yield_gordon_divergence_pct > 0.35:
                    consistency_status = "warning"
                    warning = "Terminal yield and Gordon terminal multiple are moderately divergent"

        diagnostics = {
            "consistency_status": consistency_status,
            "warning": warning,
            "discount_rate_input": discount_rate_input,
            "discount_rate_used": discount_rate,
            "stage2_growth": stage2_growth,
            "spread_before": spread_before,
            "spread_after": discount_rate - stage2_growth,
            "spread_adjusted": spread_adjusted,
            "spread_floor": MIN_SPREAD,
            "terminal_fcf_yield": terminal_fcf_yield_value,
            "gordon_terminal_multiple": gordon_terminal_multiple,
            "yield_terminal_multiple": yield_terminal_multiple,
            "yield_gordon_divergence_pct": yield_gordon_divergence_pct,
            "terminal_blend_weight": terminal_blend_weight,
        }

        value = pv_fcf + pv_terminal
        if return_diagnostics:
            return value, diagnostics
        return value
    except:
        diagnostics = {
            "consistency_status": "error",
            "warning": "Terminal consistency diagnostics failed due to exception",
            "discount_rate_input": discount_rate_input,
            "discount_rate_used": discount_rate,
            "stage2_growth": stage2_growth,
            "spread_before": spread_before,
            "spread_after": discount_rate - stage2_growth,
            "spread_adjusted": spread_adjusted,
            "spread_floor": MIN_SPREAD,
            "gordon_terminal_multiple": None,
            "yield_terminal_multiple": None,
            "yield_gordon_divergence_pct": None,
        }
        if return_diagnostics:
            return None, diagnostics
        return None


def dcf_scenarios_enhanced(
    stock,
    fcf_per_share,
    discount_rate,
    historical_growth=None,
    stock_info=None,
    sector=None,
    assumptions_override=None,
    growth_quality_adjustment=0.0,
    pessimistic_fcf_haircut=0.0,
):
    """DCF with historical growth-informed scenarios."""

    mega_cap_ai_infra_mode = bool(stock_info is not None and is_mega_cap_ai_infra(stock_info))
    mega_cap_software_mode = bool(stock_info is not None and is_mega_cap_software(stock_info))

    # Terminal growth by sector (conservative but less one-size-fits-all)
    sector_lower = str(sector).lower() if sector else ""
    sector_config = get_sector_config(sector) if sector else None
    if any(k in sector_lower for k in ["utilit", "energy", "material", "financial"]):
        stage2_growth = 0.020
    elif any(k in sector_lower for k in ["technolog", "health", "consumer"]):
        stage2_growth = 0.030
    else:
        stage2_growth = 0.025

    stage2_policy_adjustment = None
    terminal_yield_policy_adjustment = None

    # Apply sector policy range to terminal growth to avoid inflated perpetuity values.
    if sector_config is not None:
        tg_range = getattr(sector_config, "terminal_growth_range", None)
        if isinstance(tg_range, (list, tuple)) and len(tg_range) == 2:
            tg_low, tg_high = tg_range
            if tg_low is not None and tg_high is not None:
                prior = float(stage2_growth)
                stage2_growth = max(float(tg_low), min(float(tg_high), float(stage2_growth)))
                if abs(prior - stage2_growth) > 1e-12:
                    stage2_policy_adjustment = {
                        "from": prior,
                        "to": stage2_growth,
                        "range": [float(tg_low), float(tg_high)],
                    }
    
    # Use historical growth if available, else defaults
    if historical_growth is not None:
        if mega_cap_ai_infra_mode:
            # NVDA-like AI infrastructure leaders can sustain much higher near-term
            # growth than broad-tech defaults. Use an H-model style starting point.
            base_growth = min(0.45, max(0.40, historical_growth))
            pessimistic_growth = min(0.30, max(0.20, base_growth * 0.65))
            optimistic_growth = min(0.60, max(0.50, base_growth * 1.20))
        elif historical_growth < 0:
            # Company is declining: Pessimistic gets worse, Optimistic assumes recovery
            base_growth = historical_growth
            pessimistic_growth = historical_growth * 1.2  # Further decline
            optimistic_growth = max(0.01, historical_growth * 0.5)  # Recovery toward 0
        elif historical_growth > 0.25:
            # Very high growth: cap base at 25%, but maintain order
            base_growth = min(0.25, historical_growth)  # Use historical or cap at 25%
            pessimistic_growth = base_growth * 0.75  # 75% of base
            optimistic_growth = base_growth * 1.1   # 110% of base (but still realistic)
        else:
            # Normal positive growth: scale up/down from base
            base_growth = historical_growth
            pessimistic_growth = max(0.00, historical_growth * 0.6)
            optimistic_growth = min(0.25, historical_growth * 1.4)
    else:
        if mega_cap_ai_infra_mode:
            pessimistic_growth = 0.20
            base_growth = 0.40
            optimistic_growth = 0.55
        else:
            pessimistic_growth = 0.03
            base_growth = 0.08
            optimistic_growth = 0.12
    
    scenarios = {
        "Pessimistic": pessimistic_growth,
        "Base": base_growth,
        "Optimistic": optimistic_growth,
    }

    # Optional user overrides for scenario growth assumptions.
    if assumptions_override:
        override_growth = assumptions_override.get("dcf_growth_scenarios")
        if isinstance(override_growth, dict):
            custom = {}
            for scenario in ["Pessimistic", "Base", "Optimistic"]:
                value = override_growth.get(scenario)
                if value is None:
                    continue
                try:
                    value = float(value)
                    custom[scenario] = max(-0.30, min(0.40, value))
                except Exception:
                    continue
            if len(custom) == 3:
                scenarios = custom

        override_stage2 = assumptions_override.get("dcf_stage2_growth")
        if override_stage2 is not None:
            try:
                stage2_growth = max(0.00, min(0.05, float(override_stage2)))
            except Exception:
                pass
    
    # Sector-implied exit yield for hybrid terminal value
    terminal_fcf_yield = None
    terminal_blend_weight = 0.5
    mega_cap_software_mode = mega_cap_software_mode
    mega_cap_ai_infra_mode = mega_cap_ai_infra_mode
    if stock_info is not None:
        try:
            terminal_fcf_yield = get_sector_defaults(stock_info).get("fcf_yield")
            mega_cap_software_mode = is_mega_cap_software(stock_info)
            mega_cap_ai_infra_mode = is_mega_cap_ai_infra(stock_info)
            if mega_cap_software_mode:
                # For very high-quality mega-cap software, a lower terminal FCF yield
                # (higher terminal multiple) is generally more realistic than broad
                # Technology averages, while still bounded by DCF guardrails.
                terminal_fcf_yield = min(terminal_fcf_yield or 0.03, 0.03)
                terminal_blend_weight = 0.60
            if mega_cap_ai_infra_mode:
                terminal_fcf_yield = min(terminal_fcf_yield or 0.025, 0.025)
                terminal_blend_weight = 0.65
        except Exception:
            terminal_fcf_yield = None

    # Apply sector policy range to terminal yield assumptions.
    if sector_config is not None and terminal_fcf_yield is not None:
        yield_range = getattr(sector_config, "terminal_fcf_yield_range", None)
        if isinstance(yield_range, (list, tuple)) and len(yield_range) == 2:
            y_low, y_high = yield_range
            if y_low is not None and y_high is not None:
                prior = float(terminal_fcf_yield)
                terminal_fcf_yield = max(float(y_low), min(float(y_high), float(terminal_fcf_yield)))
                if abs(prior - terminal_fcf_yield) > 1e-12:
                    terminal_yield_policy_adjustment = {
                        "from": prior,
                        "to": terminal_fcf_yield,
                        "range": [float(y_low), float(y_high)],
                    }

    # Government-services defense IT guardrail: keep upside case conservative.
    optimistic_growth_cap = None
    if sector in {"Gov Defense IT", "Gov Defense IT Hybrid"}:
        optimistic_growth_cap = min(0.14, stage2_growth + 0.08)
        if scenarios.get("Optimistic") is not None:
            scenarios["Optimistic"] = min(scenarios["Optimistic"], optimistic_growth_cap)

    if mega_cap_ai_infra_mode:
        # For explicit H-model stage-1 growth we allow near-term growth to exceed WACC.
        # Guardrails are handled by the 3-year explicit high-growth phase and 7-year fade.
        max_growth_spread = {
            "Pessimistic": -0.10,
            "Base": -0.20,
            "Optimistic": -0.30,
        }
    elif mega_cap_software_mode:
        max_growth_spread = {
            "Pessimistic": 0.0150,
            "Base": 0.0075,
            "Optimistic": 0.0030,
        }
    else:
        max_growth_spread = {
            "Pessimistic": 0.0200,
            "Base": 0.0125,
            "Optimistic": 0.0050,
        }

    results = {}
    stage1_years_map = {}
    fade_years_map = {}
    effective_growth_scenarios = {}
    growth_caps_applied = {}
    terminal_consistency = {}

    for scenario, growth in scenarios.items():
        # If stage-1 growth is at/above WACC, cap it slightly below WACC
        # instead of dropping the scenario entirely.
        effective_growth = growth
        if growth >= discount_rate and not mega_cap_ai_infra_mode:
            spread = max_growth_spread.get(scenario, 0.01)
            capped_growth = discount_rate - spread - max(0.0, float(growth_quality_adjustment or 0.0))
            effective_growth = max(-0.30, min(0.40, capped_growth))
            growth_caps_applied[scenario] = {
                "from": growth,
                "to": effective_growth,
                "spread_below_discount_rate": spread,
            }
        elif mega_cap_ai_infra_mode:
            # H-model policy for AI infra leaders:
            # allow explosive stage-1 growth, but bound it to a realistic range.
            effective_growth = max(0.20, min(0.60, growth))
            if effective_growth != growth:
                growth_caps_applied[scenario] = {
                    "from": growth,
                    "to": effective_growth,
                    "spread_below_discount_rate": None,
                }

        effective_growth_scenarios[scenario] = effective_growth

        # Multi-stage horizon policy:
        # Stage 1 = explicit high-growth, then Stage 2 = fade-to-steady-state.
        if mega_cap_ai_infra_mode:
            stage1_years = 3
            fade_years = 7
        elif effective_growth >= 0.12:
            stage1_years = 5
            fade_years = 5
        elif effective_growth >= 0.06:
            stage1_years = 4
            fade_years = 4
        else:
            stage1_years = 3
            fade_years = 3
        stage1_years_map[scenario] = stage1_years
        fade_years_map[scenario] = fade_years

        scenario_fcf_per_share = fcf_per_share
        if scenario == "Pessimistic" and pessimistic_fcf_haircut > 0:
            scenario_fcf_per_share = fcf_per_share * (1.0 - min(0.20, max(0.0, pessimistic_fcf_haircut)))

        val, diag = two_stage_dcf(
            scenario_fcf_per_share,
            effective_growth,
            stage2_growth,
            discount_rate,
            stage1_years=stage1_years,
            fade_years=fade_years,
            terminal_fcf_yield=terminal_fcf_yield,
            terminal_blend_weight=terminal_blend_weight,
            return_diagnostics=True,
        )
        results[scenario] = val
        terminal_consistency[scenario] = diag

    terminal_consistency_warnings = []
    for scenario in ["Pessimistic", "Base", "Optimistic"]:
        diag = terminal_consistency.get(scenario) or {}
        warning = diag.get("warning")
        if warning:
            terminal_consistency_warnings.append(f"{scenario}: {warning}")

    results["_assumptions"] = {
        "fcf_per_share": fcf_per_share,
        "discount_rate": discount_rate,
        "stage2_growth": stage2_growth,
        "growth_scenarios": effective_growth_scenarios,
        "growth_scenarios_raw": scenarios,
        "growth_caps_applied": growth_caps_applied,
        "stage1_years": stage1_years_map,
        "fade_years": fade_years_map,
        "terminal_fcf_yield": terminal_fcf_yield,
        "stage2_policy_adjustment": stage2_policy_adjustment,
        "terminal_yield_policy_adjustment": terminal_yield_policy_adjustment,
        "optimistic_growth_cap": optimistic_growth_cap,
        "terminal_blend_weight": terminal_blend_weight,
        "terminal_consistency": terminal_consistency,
        "terminal_consistency_warnings": terminal_consistency_warnings,
        "mega_cap_software_mode": mega_cap_software_mode,
        "mega_cap_ai_infra_mode": mega_cap_ai_infra_mode,
    }
    
    return results


def compute_normalized_dcf_growth(
    stock_info,
    sector_config,
    historical_fcf_growth,
    historical_eps_growth,
    sector,
    fundamental_confirmation=None,
):
    """Build robust long-term DCF growth from multiple auto-extracted data signals."""
    signals = []

    def add_signal(name, value, weight, scale=1.0, low=-0.30, high=0.40):
        # Use a finite-number check instead of validate_data so that negative
        # growth signals (declining earnings, falling FCF) are accepted.
        # validate_data() requires value > 0.001, which silently drops every
        # negative observation and introduces an upward bias in growth estimates.
        if value is not None and isinstance(value, (int, float)) and np.isfinite(float(value)):
            v = float(value) * scale
            v = max(low, min(high, v))
            signals.append((name, v, weight))

    # Historical company-specific signals
    add_signal("hist_fcf", historical_fcf_growth, 0.35)
    add_signal("hist_eps", historical_eps_growth, 0.30)

    # Forward-looking signals from market data (short-term -> damped long-term)
    add_signal("earningsGrowth", stock_info.get("earningsGrowth"), 0.22, scale=0.55)
    add_signal("revenueGrowth", stock_info.get("revenueGrowth"), 0.18, scale=0.55)
    add_signal("earningsQuarterlyGrowth", stock_info.get("earningsQuarterlyGrowth"), 0.12, scale=0.35)

    # Forward EPS implied growth (usually next FY, so damp heavily)
    forward_eps = stock_info.get("forwardEps")
    trailing_eps = stock_info.get("trailingEps")
    if validate_data(forward_eps) and validate_data(trailing_eps) and trailing_eps > 0:
        implied = (forward_eps / trailing_eps) - 1.0
        add_signal("fwd_eps_implied", implied, 0.28, scale=0.40)

    # Weighted growth estimate across all available signals
    if signals:
        values = np.array([s[1] for s in signals], dtype=float)
        weights = np.array([s[2] for s in signals], dtype=float)
        raw_growth = float(np.average(values, weights=weights))
    else:
        raw_growth = 0.05

    # Gentle mean reversion to sector baseline (trust signals more)
    sector_mid = (sector_config.growth_min + sector_config.growth_max) / 2
    signal_count = len(signals)
    if signal_count >= 5:
        reversion = 0.05  # Keep 95% of signal
    elif signal_count >= 3:
        reversion = 0.10  # Keep 90% of signal
    else:
        reversion = 0.15  # Keep 85% of signal

    growth = raw_growth * (1 - reversion) + sector_mid * reversion

    # Base sector clamp from config policy
    growth = get_growth_assumption(sector_config, growth)

    sector_lower = str(sector or "").lower()
    industry_lower = str(stock_info.get("industry", "")).lower()
    ai_infra_mode = is_mega_cap_ai_infra(stock_info)

    # Structural caps by sector type for long-run DCF realism
    policy_cap = getattr(sector_config, "structural_growth_cap", None)
    if ai_infra_mode:
        cap = 0.40
    elif policy_cap is not None:
        cap = float(policy_cap)
    elif (
        sector_config.special_rules.get("use_normalized_commodity")
        or sector_config.special_rules.get("use_normalized_capex")
        or sector_config.special_rules.get("regulated")
    ):
        cap = 0.06
    elif "financial" in sector_lower:
        cap = 0.08
    elif "health" in sector_lower:
        cap = 0.12
    elif "communication services" in sector_lower or sector_config.special_rules.get("platform_business"):
        cap = 0.18  # Platform businesses can sustain higher growth
    elif "technolog" in sector_lower or "software" in sector_lower or "semiconductor" in industry_lower:
        cap = 0.15
    elif "consumer" in sector_lower:
        cap = 0.10
    else:
        cap = 0.12

    floor = -0.02
    growth = max(floor, min(cap, growth))

    # Keep FCF as the DCF anchor, but avoid treating unusually high cash
    # conversion or negative annual EPS as fully sustainable growth evidence.
    quality_adjustment = 0.0
    confirmation_breakdown = (fundamental_confirmation or {}).get("breakdown") or {}
    conversion_pct = confirmation_breakdown.get("fcf_conversion", {}).get("value")
    if isinstance(conversion_pct, (int, float)) and np.isfinite(float(conversion_pct)) and conversion_pct > 150.0:
        quality_adjustment += min(0.02, (float(conversion_pct) - 150.0) / 3200.0)
    latest_eps_status = confirmation_breakdown.get("latest_annual_eps", {}).get("status")
    if latest_eps_status == "negative":
        quality_adjustment += 0.015
    if quality_adjustment > 0.0:
        growth = max(floor, growth - quality_adjustment)

    diagnostics = {
        "signal_count": signal_count,
        "signals": [(n, round(v, 4), w) for n, v, w in signals],
        "raw_growth": raw_growth,
        "sector_mid": sector_mid,
        "reversion": reversion,
        "final_growth": growth,
        "ai_infra_mode": ai_infra_mode,
        "quality_adjustment": quality_adjustment,
        "fcf_conversion_pct": conversion_pct,
        "latest_annual_eps_status": latest_eps_status,
    }
    return growth, diagnostics


def pe_scenarios_enhanced(
    stock,
    stock_info,
    discount_rate,
    historical_pe=None,
    sector=None,
    sector_config=None,
    assumptions_override=None,
    quality_flags=None,
    assumption_deltas=None,
):
    """P/E valuation with historical and sector context."""
    
    eps = stock_info.get("forwardEps") or stock_info.get("trailingEps")
    if not validate_data(eps):
        return {"Pessimistic": None, "Base": None, "Optimistic": None}
    
    try:
        quality_flags = quality_flags if quality_flags is not None else []
        assumption_deltas = assumption_deltas if assumption_deltas is not None else {}

        # Get sector baseline. Avoid anchoring on extreme trailing P/E when forward P/E is available.
        sector_defaults = get_sector_defaults(stock_info)
        sector_pe = sector_defaults["pe"]
        sector_config = sector_config or get_sector_config(sector)

        # Prefer forward P/E, fallback to trailing, then sector default.
        base_pe = stock_info.get("forwardPE") or stock_info.get("trailingPE") or historical_pe or sector_pe
        raw_base_pe = base_pe
        
        years = 5
        results = {}
        
        # Sector-specific growth assumptions (capital-intensive sectors grow slower)
        sector_lower = str(sector).lower() if sector else ""
        if "utilit" in sector_lower or "energy" in sector_lower:
            # Utilities/Energy: Lower growth expectations (2-4% typically)
            growth_rates = {"Pessimistic": 0.02, "Base": 0.03, "Optimistic": 0.05}
        else:
            # Default growth assumptions for other sectors
            growth_rates = {"Pessimistic": 0.03, "Base": 0.08, "Optimistic": 0.12}
        
        # Apply field-level policy band for P/E (non-multiplicative handling).
        pe_range = getattr(sector_config, "pe_range", None)
        enforcement = (getattr(sector_config, "field_enforcement", {}) or {}).get("pe", "soft")
        base_pe = _apply_policy_range(base_pe, "pe", pe_range, enforcement, quality_flags, assumption_deltas)
        # Hard emergency clamp — wider than the soft prior; stops implausible multiples from passing silently.
        pe_sanity_range = getattr(sector_config, "pe_sanity_range", None)
        if pe_sanity_range is not None:
            base_pe = _apply_policy_range(base_pe, "pe_sanity", pe_sanity_range, "hard", quality_flags, assumption_deltas)

        # Scenario spread uses growth proxy only, not premium-ratio multiplication.
        forward_eps = stock_info.get("forwardEps")
        trailing_eps = stock_info.get("trailingEps")
        eps_growth_proxy = None
        if validate_data(forward_eps) and validate_data(trailing_eps) and trailing_eps > 0:
            eps_growth_proxy = (forward_eps / trailing_eps) - 1.0

        if eps_growth_proxy is None:
            pe_adjustments = [
                ("Pessimistic", -0.14),
                ("Base", 0.00),
                ("Optimistic", 0.14),
            ]
        elif eps_growth_proxy >= 0.25:
            pe_adjustments = [
                ("Pessimistic", -0.20),
                ("Base", 0.02),
                ("Optimistic", 0.20),
            ]
        elif eps_growth_proxy >= 0.10:
            pe_adjustments = [
                ("Pessimistic", -0.16),
                ("Base", 0.01),
                ("Optimistic", 0.16),
            ]
        elif eps_growth_proxy >= 0.00:
            pe_adjustments = [
                ("Pessimistic", -0.14),
                ("Base", 0.00),
                ("Optimistic", 0.14),
            ]
        else:
            pe_adjustments = [
                ("Pessimistic", -0.18),
                ("Base", -0.02),
                ("Optimistic", 0.10),
            ]

        premium_ratio = (base_pe / sector_pe) if validate_data(base_pe) and validate_data(sector_pe) and sector_pe > 0 else 1.0

        custom_target_multiples = None
        if assumptions_override:
            override_targets = assumptions_override.get("pe_target_multiples")
            if isinstance(override_targets, dict):
                candidate = {}
                for scenario in ["Pessimistic", "Base", "Optimistic"]:
                    value = override_targets.get(scenario)
                    if value is None:
                        continue
                    try:
                        value = float(value)
                        candidate[scenario] = max(1.0, min(120.0, value))
                    except Exception:
                        continue
                if len(candidate) == 3:
                    custom_target_multiples = candidate

        normalized_pessimistic_pe = None
        normalized_optimistic_pe = None

        if custom_target_multiples is not None:
            scenario_targets = dict(custom_target_multiples)
            target_source = "custom_override"
        else:
            pe_low = pe_range[0] if isinstance(pe_range, (list, tuple)) and len(pe_range) == 2 else None
            pe_high = pe_range[1] if isinstance(pe_range, (list, tuple)) and len(pe_range) == 2 else None

            # Keep base unchanged unless hard policy limits required clipping above.
            base_target = float(base_pe)

            pess_adj = next((adj for name, adj in pe_adjustments if name == "Pessimistic"), -0.14)
            opt_adj = next((adj for name, adj in pe_adjustments if name == "Optimistic"), 0.14)

            pess_from_base = base_target * (1 + pess_adj)
            opt_from_base = base_target * (1 + opt_adj)

            if validate_data(sector_pe):
                pess_ref = float(sector_pe) * 0.90
                opt_ref = float(sector_pe) * 1.10
            else:
                pess_ref = base_target * 0.90
                opt_ref = base_target * 1.10

            normalized_pessimistic_pe = min(base_target, (pess_from_base + pess_ref) / 2.0)
            normalized_optimistic_pe = max(base_target, (opt_from_base + opt_ref) / 2.0)

            if pe_low is not None:
                normalized_pessimistic_pe = max(float(pe_low), normalized_pessimistic_pe)
                normalized_optimistic_pe = max(float(pe_low), normalized_optimistic_pe)
            if pe_high is not None:
                normalized_pessimistic_pe = min(float(pe_high), normalized_pessimistic_pe)
                normalized_optimistic_pe = min(float(pe_high), normalized_optimistic_pe)

            # Guarantee monotonic ordering after normalization.
            normalized_pessimistic_pe = min(normalized_pessimistic_pe, base_target)
            normalized_optimistic_pe = max(normalized_optimistic_pe, base_target)

            scenario_targets = {
                "Pessimistic": normalized_pessimistic_pe,
                "Base": base_target,
                "Optimistic": normalized_optimistic_pe,
            }
            target_source = "normalized_sector_targets"

        for scenario in ["Pessimistic", "Base", "Optimistic"]:
            growth = growth_rates[scenario]
            eps_future = eps * ((1 + growth) ** years)
            pe_target = scenario_targets.get(scenario)

            # Re-apply policy range at target multiple level.
            pe_target = _apply_policy_range(
                pe_target,
                "pe",
                pe_range,
                enforcement,
                quality_flags,
                assumption_deltas,
            )
            if pe_sanity_range is not None:
                pe_target = _apply_policy_range(pe_target, "pe_sanity", pe_sanity_range, "hard", quality_flags, assumption_deltas)
            future_price = eps_future * pe_target
            fair_value = future_price / ((1 + discount_rate) ** years)
            results[scenario] = fair_value

        results["_assumptions"] = {
            "eps_used": eps,
            "years": years,
            "base_pe_raw": raw_base_pe,
            "base_pe_used": base_pe,
            "sector_pe": sector_pe,
            "premium_ratio": premium_ratio,
            "premium_ratio_influences_scenarios": False,
            "raw_base_pe": raw_base_pe,
            "normalized_pessimistic_pe": normalized_pessimistic_pe,
            "normalized_optimistic_pe": normalized_optimistic_pe,
            "pe_policy_bounds": list(pe_range) if isinstance(pe_range, (list, tuple)) and len(pe_range) == 2 else None,
            "target_source": target_source,
            "growth_rates": growth_rates,
            "pe_adjustments": {name: adj for name, adj in pe_adjustments},
            "custom_target_multiples": custom_target_multiples,
            "forward_pe": stock_info.get("forwardPE"),
            "trailing_pe": stock_info.get("trailingPE"),
            "forward_eps": stock_info.get("forwardEps"),
            "trailing_eps": stock_info.get("trailingEps"),
        }

        if validate_data(raw_base_pe) and validate_data(base_pe) and raw_base_pe > base_pe * 1.05:
            results["_pe_guardrail"] = {
                "raw_pe": raw_base_pe,
                "adjusted_pe": base_pe,
                "sector_pe": sector_pe,
                "premium_ratio": premium_ratio,
            }
        
        return results
    except:
        return {"Pessimistic": None, "Base": None, "Optimistic": None}


def ev_ebitda_scenarios_enhanced(
    stock,
    stock_info,
    net_debt_data=None,
    sector_config=None,
    quality_flags=None,
    assumption_deltas=None,
):
    """EV/EBITDA with sector-specific multiples."""
    
    ebitda = stock_info.get("ebitda")
    shares = stock_info.get("sharesOutstanding")
    
    if not validate_data(ebitda) or not validate_data(shares):
        return {"Pessimistic": None, "Base": None, "Optimistic": None}
    
    try:
        net_debt_data = net_debt_data or resolve_net_debt_components(stock, stock_info)
        total_debt = net_debt_data.get("total_debt", 0) or 0
        total_cash = net_debt_data.get("total_cash", 0) or 0
        net_debt = net_debt_data.get("net_debt", 0) or 0
        
        quality_flags = quality_flags if quality_flags is not None else []
        assumption_deltas = assumption_deltas if assumption_deltas is not None else {}

        sector_defaults = get_sector_defaults(stock_info)
        base_multiple = sector_defaults["ev_ebitda"]
        sector_config = sector_config or get_sector_config(stock_info.get("sector"))

        # Leverage discount on the sector EV/EBITDA reference multiple.
        # Companies with Net Debt/EBITDA > 3× trade at a discount to peers
        # because the equity residual is smaller and default risk is higher.
        # Apply a progressive haircut: linear from 0% at 3× to 25% at ≥ 5×.
        leverage_multiple_haircut = 0.0
        if ebitda > 0 and net_debt > 0:
            nd_to_ebitda = net_debt / ebitda
            if nd_to_ebitda >= 5.0:
                leverage_multiple_haircut = 0.25
            elif nd_to_ebitda > 3.0:
                leverage_multiple_haircut = (nd_to_ebitda - 3.0) / 2.0 * 0.25
            if leverage_multiple_haircut > 0:
                base_multiple *= (1.0 - leverage_multiple_haircut)

        ev_range = getattr(sector_config, "ev_ebitda_range", None)
        enforcement = (getattr(sector_config, "field_enforcement", {}) or {}).get("ev_ebitda", "soft")

        # Apply policy range at baseline before building scenarios.
        base_multiple = _apply_policy_range(
            base_multiple,
            "ev_ebitda",
            ev_range,
            enforcement,
            quality_flags,
            assumption_deltas,
        )
        # Hard emergency clamp on EV/EBITDA baseline.
        ev_sanity_range = getattr(sector_config, "ev_ebitda_sanity_range", None)
        if ev_sanity_range is not None:
            base_multiple = _apply_policy_range(base_multiple, "ev_ebitda_sanity", ev_sanity_range, "hard", quality_flags, assumption_deltas)

        results = {}
        for scenario, mult_adj in [
            ("Pessimistic", -0.20),
            ("Base", 0.00),
            ("Optimistic", 0.20),
        ]:
            target_mult = base_multiple * (1 + mult_adj)
            target_mult = _apply_policy_range(
                target_mult,
                "ev_ebitda",
                ev_range,
                enforcement,
                quality_flags,
                assumption_deltas,
            )
            if ev_sanity_range is not None:
                target_mult = _apply_policy_range(target_mult, "ev_ebitda_sanity", ev_sanity_range, "hard", quality_flags, assumption_deltas)
            ev = ebitda * target_mult
            equity_value = ev - net_debt
            fair_value = equity_value / shares if shares > 0 else None
            results[scenario] = fair_value

        results["_assumptions"] = {
            "ebitda": ebitda,
            "shares_outstanding": shares,
            "total_debt": total_debt,
            "total_cash": total_cash,
            "net_debt": net_debt,
            "net_debt_source": net_debt_data.get("source"),
            "net_debt_freshness": net_debt_data.get("freshness"),
            "base_multiple": base_multiple,
            "ev_ebitda_policy_bounds": list(ev_range) if isinstance(ev_range, (list, tuple)) and len(ev_range) == 2 else None,
            "leverage_multiple_haircut": leverage_multiple_haircut,
        }

        return results
    except:
        return {"Pessimistic": None, "Base": None, "Optimistic": None}


def null_scenarios():
    return {"Pessimistic": None, "Base": None, "Optimistic": None}


def pb_scenarios_enhanced(stock, stock_info, net_debt_data=None):
    """Price-to-Book valuation for financial institutions (banks, insurers).
    
    Banks and insurers are asset-based businesses where book value is the primary
    valuation anchor. Use P/B ratios with sector-based normalization.
    """
    try:
        price = stock_info.get("currentPrice")
        book_value_per_share = stock_info.get("bookValue")
        
        # Fallback: try multiple equity row names for international/HK banks
        if not validate_data(book_value_per_share):
            balance_sheet = stock.balance_sheet
            if balance_sheet is not None:
                equity_rows = [
                    "Stockholders Equity",
                    "Total Equity Gross Minority Interest",
                    "Common Stock Equity",
                ]
                for row_name in equity_rows:
                    if row_name in balance_sheet.index:
                        total_equity = balance_sheet.loc[row_name].iloc[0]
                        shares = stock_info.get("sharesOutstanding")
                        if validate_data(total_equity) and validate_data(shares) and shares > 0:
                            book_value_per_share = total_equity / shares
                            break
        
        if not validate_data(price) or not validate_data(book_value_per_share) or book_value_per_share <= 0:
            return null_scenarios()
        
        current_pb = price / book_value_per_share
        
        # Sector defaults for P/B targets (Financials: 0.8-1.5x typical)
        sector_defaults = get_sector_defaults(stock_info)
        
        # Conservative scenario: P/B contracts to 0.8x (low confidence in ROE)
        # Base scenario: P/B remains at sector median (e.g., 1.0-1.2x)
        # Optimistic scenario: P/B expands to 1.4x (strong ROE justifies premium)
        pb_targets = {
            "Pessimistic": 0.80,
            "Base": 1.00,
            "Optimistic": 1.40,
        }
        
        results = {}
        for scenario, target_pb in pb_targets.items():
            fair_value = book_value_per_share * target_pb
            results[scenario] = fair_value
        
        results["_assumptions"] = {
            "price": price,
            "book_value_per_share": book_value_per_share,
            "current_pb": current_pb,
            "pb_targets": pb_targets,
        }
        
        return results
    except Exception as e:
        logger.debug(f"P/B valuation failed: {e}")
        return null_scenarios()


def roe_scenarios_enhanced(stock, stock_info, net_debt_data=None):
    """ROE-based valuation for financial institutions.
    
    Uses Return on Equity to anchor fair value via justified P/B = ROE / cost_of_equity.
    Normalizes ROE to 5-year median for stability.
    """
    try:
        eps = stock_info.get("trailingEps")
        price = stock_info.get("currentPrice")
        book_value_per_share = stock_info.get("bookValue")
        roe = stock_info.get("returnOnEquity")
        
        # Fallback: derive ROE from statements using multiple equity row names
        if not validate_data(roe):
            try:
                quarterly_income = stock.quarterly_financials
                quarterly_balance = stock.quarterly_balance_sheet
                equity_rows = [
                    "Stockholders Equity",
                    "Total Equity Gross Minority Interest",
                    "Common Stock Equity",
                ]
                if (quarterly_income is not None and "Net Income" in quarterly_income.index
                        and quarterly_balance is not None):
                    recent_ni = quarterly_income.loc["Net Income"].iloc[0]
                    for row_name in equity_rows:
                        if row_name in quarterly_balance.index:
                            recent_se = quarterly_balance.loc[row_name].iloc[0]
                            if validate_data(recent_ni) and validate_data(recent_se) and recent_se > 0:
                                roe = recent_ni / recent_se
                                break
            except:
                pass
        
        if not validate_data(eps) or not validate_data(roe) or not validate_data(price) or roe <= 0:
            return null_scenarios()
        
        # Estimate cost of equity (use WACC as proxy, or sector default)
        # Conservative: 10%, Base: 8%, Optimistic: 6% for banks
        cost_of_equity_scenarios = {
            "Pessimistic": 0.10,
            "Base": 0.08,
            "Optimistic": 0.06,
        }
        
        results = {}
        for scenario, cost_of_equity in cost_of_equity_scenarios.items():
            # Justified P/E = ROE / cost_of_equity
            if cost_of_equity > 0:
                justified_pe = roe / cost_of_equity
                fair_value = eps * justified_pe
                results[scenario] = fair_value
            else:
                results[scenario] = None
        
        results["_assumptions"] = {
            "price": price,
            "eps": eps,
            "roe": roe,
            "cost_of_equity_scenarios": cost_of_equity_scenarios,
        }
        
        return results
    except Exception as e:
        logger.debug(f"ROE valuation failed: {e}")
        return null_scenarios()


def exclude_outlier_method(method_name, method_vals, price, quality_flags, min_ratio=0.20, max_ratio=8.0):
    """Exclude method entirely if its base valuation is implausibly far from current price."""
    try:
        base_val = method_vals.get("Base") if isinstance(method_vals, dict) else None
        if not validate_data(base_val) or not validate_data(price):
            return method_vals

        ratio = base_val / price
        if ratio < min_ratio or ratio > max_ratio:
            quality_flags.append(f"{method_name} excluded as outlier (Base/Price={ratio:.1f}x)")
            return null_scenarios()
        return method_vals
    except Exception:
        return method_vals


def calculate_confidence_score(methods_available: int, min_required: int, 
                              sector_base_confidence: float, data_issues: int) -> tuple:
    """
    Calculate confidence score (0-1) for valuation.
    
    Args:
        methods_available: Number of methods that produced valid results
        min_required: Minimum methods needed for sector
        sector_base_confidence: Base confidence from sector config (0-1)
        data_issues: Number of data quality problems found
        
    Returns:
        (confidence_score, confidence_description)
    """
    # Method availability factor (0-1)
    # Start with sector base confidence, then adjust based on method coverage
    if methods_available >= min_required:
        # We have enough methods - add bonus for extra methods
        # If min_required=1 and we have 1, use 1.0
        # If min_required=2 and we have 2, use 1.0
        # If min_required=2 and we have 3, use 1.05 (slight bonus)
        method_factor = min(1.05, 0.5 + (methods_available / 3.0) * 0.5) 
    else:
        # We have fewer methods than minimum - apply penalty
        # If min_required=2 and we have 1, use 0.6
        # If min_required=1 and we have 0, use 0.0
        method_factor = (methods_available / max(min_required, 1)) * 0.8
    
    # Data quality factor (0-1)
    # Each data issue reduces confidence by 15% (more severe than before)
    data_factor = max(0.0, 1.0 - (data_issues * 0.15))
    
    # Combine: sector confidence is the base, then adjust by method/data factors
    # This way, single-method valuations for banks get ~0.50*0.9 = 0.45 confidence
    confidence = sector_base_confidence * method_factor * data_factor
    confidence = max(0.0, min(1.0, confidence))  # Clamp 0-1
    
    description = _describe_confidence_level(confidence)
    
    return confidence, description


def _describe_confidence_level(confidence):
    """Map numeric confidence score to user-facing confidence label."""
    if confidence >= 0.80:
        return "High (80%+)"
    if confidence >= 0.65:
        return "Good (65%+)"
    if confidence >= 0.50:
        return "Fair (50%+)"
    if confidence >= 0.35:
        return "Low (35%+)"
    return "Very Low (<35%)"


def _to_naive_timestamp(value):
    """Convert input to naive pandas Timestamp, returning None on failure."""
    try:
        ts = pd.Timestamp(value)
        if pd.isna(ts):
            return None
        if ts.tzinfo is not None:
            ts = ts.tz_localize(None)
        return ts
    except Exception:
        return None


def _extract_reporting_dates(stock):
    """Best-effort extraction of filing and earnings-release dates from provider payload."""
    filing_dates = []
    release_dates = []

    try:
        sec_df = getattr(stock, "sec_filings", None)
        if sec_df is not None and not getattr(sec_df, "empty", True):
            for col in ["Date", "date", "Filing Date", "filingDate"]:
                if col in sec_df.columns:
                    series = pd.to_datetime(sec_df[col], errors="coerce")
                    filing_dates.extend([_to_naive_timestamp(v) for v in series.dropna().tolist()])
                    break
    except Exception:
        pass

    try:
        earnings_df = stock.get_earnings_dates(limit=40)
        if earnings_df is not None and not getattr(earnings_df, "empty", True):
            idx = pd.to_datetime(earnings_df.index, errors="coerce")
            release_dates.extend([_to_naive_timestamp(v) for v in idx.dropna().tolist()])
    except Exception:
        pass

    filing_dates = sorted([d for d in filing_dates if d is not None])
    release_dates = sorted([d for d in release_dates if d is not None])
    return filing_dates, release_dates


def _resolve_financial_availability_date(period_end, filing_dates, release_dates, fallback_days=45):
    """Resolve first likely-available date using filing -> earnings release -> fallback lag."""
    period_end = _to_naive_timestamp(period_end)
    if period_end is None:
        return None, "unknown"

    max_horizon = period_end + pd.Timedelta(days=200)

    filing_candidate = next((d for d in filing_dates if d >= period_end and d <= max_horizon), None)
    if filing_candidate is not None:
        return filing_candidate, "filing_date"

    release_candidate = next((d for d in release_dates if d >= period_end and d <= max_horizon), None)
    if release_candidate is not None:
        return release_candidate, "earnings_release"

    return period_end + pd.Timedelta(days=fallback_days), f"fallback_{int(fallback_days)}d"


def _compute_margin_outputs(intrinsic_base_value, anchored_base_value, price):
    """Return margin outputs with explicit basis and legacy-compatible field."""
    intrinsic_margin_of_safety = None
    anchored_margin_of_safety = None
    margin_of_safety_basis = "anchored"

    if price:
        if intrinsic_base_value:
            intrinsic_margin_of_safety = (intrinsic_base_value - price) / price * 100
        if anchored_base_value:
            anchored_margin_of_safety = (anchored_base_value - price) / price * 100

    return {
        "margin_of_safety": anchored_margin_of_safety,
        "margin_of_safety_basis": margin_of_safety_basis,
        "intrinsic_margin_of_safety": intrinsic_margin_of_safety,
        "anchored_margin_of_safety": anchored_margin_of_safety,
    }


def build_investment_score(valuation_result, fundamental_confirmation=None):
    """Summarize current valuation evidence without changing fair-value outputs."""
    components = {}

    def _number(value):
        try:
            value = float(value)
            return value if np.isfinite(value) else None
        except (TypeError, ValueError):
            return None

    def _add_component(key, label, weight, score, detail, unavailable_reason=None):
        components[key] = {
            "label": label,
            "weight": weight,
            "score": round(max(0.0, min(100.0, score))) if score is not None else None,
            "detail": detail,
            "unavailable_reason": unavailable_reason,
        }

    margin = _number(valuation_result.get("anchored_margin_of_safety"))
    if margin is None:
        _add_component(
            "valuation_opportunity", "Valuation Opportunity", 0.35, None, None,
            "Anchored fair value or current price is unavailable.",
        )
    else:
        opportunity_score = 50.0 + 45.0 * np.tanh(margin / 30.0)
        _add_component(
            "valuation_opportunity", "Valuation Opportunity", 0.35,
            opportunity_score,
            f"Anchored margin of safety: {margin:+.1f}%. Formula: 50 + 45 x tanh({margin:+.1f}% / 30%) = {opportunity_score:.1f}; asymptotic range: 5-95.",
        )

    confirmation_score = _number((fundamental_confirmation or {}).get("score"))
    confirmation_status = (fundamental_confirmation or {}).get("status")
    if confirmation_score is None or confirmation_status == "insufficient":
        _add_component(
            "fundamental_support", "Fundamental Support", 0.25, None, None,
            (fundamental_confirmation or {}).get("phrase")
            or "Fewer than four fundamental confirmation metrics are available.",
        )
    else:
        _add_component(
            "fundamental_support", "Fundamental Support", 0.25, confirmation_score,
            (fundamental_confirmation or {}).get("phrase")
            or f"Fundamental confirmation: {confirmation_status}.",
        )

    confidence = _number(valuation_result.get("model_confidence_score"))
    if confidence is None:
        confidence = _number(valuation_result.get("confidence_score"))
    methods_used = valuation_result.get("methods_used") or []
    quality_flags = valuation_result.get("quality_flags") or []
    if confidence is None:
        _add_component(
            "valuation_robustness", "Valuation Robustness", 0.25, None, None,
            "Valuation confidence is unavailable.",
        )
    else:
        reliability_flags = [
            flag for flag in quality_flags
            if "terminal consistency" not in str(flag).lower()
            and any(term in str(flag).lower() for term in ("dispersion", "normalization risk", "only ", "missing", "unavailable"))
        ]
        terminal_warning_text = {
            str(flag).strip().lower()
            for flag in quality_flags
            if "terminal consistency" in str(flag).lower()
        }
        terminal_penalty = min(6.0, len(terminal_warning_text) * 2.0)
        robustness = confidence * 100.0 - min(20.0, len(reliability_flags) * 4.0) - terminal_penalty
        _add_component(
            "valuation_robustness", "Valuation Robustness", 0.25, robustness,
            f"Pre-leverage model confidence: {confidence:.0%}; methods used: {len(methods_used)}; severe warnings: {len(reliability_flags)} x 4 points; distinct terminal warnings: {len(terminal_warning_text)} x 2 points (cap 6); result: {robustness:.1f}.",
        )

    price = _number(valuation_result.get("price"))
    pessimistic_value = _number((valuation_result.get("blended") or {}).get("Pessimistic"))
    base_value = _number((valuation_result.get("blended") or {}).get("Base"))
    leverage = _number(valuation_result.get("net_debt_to_ebitda"))
    sector = str(valuation_result.get("sector") or "")
    sector_leverage_profiles = {
        "Utilities": (6.0, "regulated utility"),
        "Telecom": (6.0, "telecom"),
        "Wireless Telecom": (6.0, "telecom"),
        "Integrated Telecom": (6.0, "telecom"),
        "Cable/Broadband": (6.0, "telecom"),
        "Energy": (7.0, "asset-heavy energy"),
        "Materials": (7.0, "asset-heavy materials"),
    }
    unsuitable_leverage_sectors = {"Financials", "Financial Services", "Financial Data Services", "Real Estate"}
    platform_business_whitelist = {"Payment Networks", "Financial Data Services"}  # leverage is optional for platform businesses
    if sector in unsuitable_leverage_sectors and sector not in platform_business_whitelist:
        _add_component(
            "downside_resilience", "Downside Resilience", 0.15, None, None,
            f"Net debt/EBITDA is not a comparable resilience measure for {sector}; a sector-specific balance-sheet metric is required.",
        )
    elif price is None or price <= 0 or pessimistic_value is None or base_value is None or leverage is None:
        _add_component(
            "downside_resilience", "Downside Resilience", 0.15, None, None,
            "Pessimistic and base fair values, current price, and Net Debt/EBITDA are required.",
        )
    else:
        pessimistic_margin = (pessimistic_value - price) / price * 100.0
        pessimistic_score = 50.0 + 45.0 * np.tanh(pessimistic_margin / 30.0)
        scenario_spread = max(0.0, (base_value - pessimistic_value) / price * 100.0)
        spread_score = max(0.0, min(100.0, 100.0 - scenario_spread * 2.5))
        leverage_slope, leverage_profile = sector_leverage_profiles.get(sector, (8.5, "standard"))
        leverage_score = (
            80.0
            - leverage_slope * max(0.0, leverage)
            + 5.0 * min(1.0, max(0.0, -leverage))
        )
        leverage_score = max(0.0, min(100.0, leverage_score))
        resilience = 0.50 * pessimistic_score + 0.30 * spread_score + 0.20 * leverage_score
        _add_component(
            "downside_resilience", "Downside Resilience", 0.15, resilience,
            f"D_pess={pessimistic_score:.1f} (pessimistic margin {pessimistic_margin:+.1f}%); D_spread={spread_score:.1f} (base-to-pessimistic spread {scenario_spread:.1f}%); D_leverage={leverage_score:.1f} ({leverage_profile}, ND/EBITDA {leverage:.1f}x; slope {leverage_slope:.1f}, net-cash bonus up to 5). Composite: 0.50 x D_pess + 0.30 x D_spread + 0.20 x D_leverage = {resilience:.1f}.",
        )

    available = [component for component in components.values() if component["score"] is not None]
    available_weight = sum(component["weight"] for component in available)
    required_available = (
        components["valuation_opportunity"]["score"] is not None
        and components["valuation_robustness"]["score"] is not None
        and len(available) >= 3
    )
    total_score = None
    if required_available and available_weight > 0:
        total_score = round(sum(component["score"] * component["weight"] for component in available) / available_weight)

    if total_score is None:
        tier = "Insufficient evidence"
    elif total_score >= 80:
        tier = "High conviction"
    elif total_score >= 65:
        tier = "Constructive"
    elif total_score >= 45:
        tier = "Mixed"
    else:
        tier = "Cautious"

    return {
        "total_score": total_score,
        "tier": tier,
        "components": components,
        "available_components": len(available),
        "available_weight": available_weight,
        "is_sufficient": required_available,
        "description": "Current valuation and evidence summary; not a price prediction or backtested trading-edge measure.",
    }


def build_sell_indicator(valuation_result, fundamental_confirmation=None, average_buy_price=None):
    """Build a current Sell Pressure indicator without changing valuation outputs."""
    components = {}

    def _number(value):
        try:
            value = float(value)
            return value if np.isfinite(value) else None
        except (TypeError, ValueError):
            return None

    def _add_component(key, label, weight, score, detail, unavailable_reason=None):
        components[key] = {
            "label": label,
            "weight": weight,
            "score": round(max(0.0, min(100.0, score))) if score is not None else None,
            "detail": detail,
            "unavailable_reason": unavailable_reason,
        }

    margin = _number(valuation_result.get("anchored_margin_of_safety"))
    if margin is None:
        _add_component("valuation_excess", "Valuation Excess", 0.35, None, None, "Anchored margin of safety is unavailable.")
    else:
        excess_score = 100.0 * (1.0 - np.exp(min(0.0, margin) / 35.0))
        _add_component(
            "valuation_excess", "Valuation Excess", 0.35, excess_score,
            f"Anchored margin: {margin:+.1f}%; pressure = 100 x (1 - exp(min(0, margin) / 35)) = {excess_score:.1f}.",
        )

    price = _number(valuation_result.get("price"))
    pessimistic = _number((valuation_result.get("blended") or {}).get("Pessimistic"))
    base = _number((valuation_result.get("blended") or {}).get("Base"))
    if price is None or price <= 0 or pessimistic is None or base is None:
        _add_component("downside_asymmetry", "Downside Asymmetry", 0.25, None, None, "Price and fair-value scenarios are required.")
    else:
        pessimistic_margin = (pessimistic - price) / price * 100.0
        downside_pressure = 50.0 + 50.0 * np.tanh(-pessimistic_margin / 30.0)
        spread = max(0.0, (base - pessimistic) / price * 100.0)
        spread_pressure = min(100.0, spread * 2.5)
        downside_score = 0.65 * downside_pressure + 0.35 * spread_pressure
        _add_component(
            "downside_asymmetry", "Downside Asymmetry", 0.25, downside_score,
            f"Pessimistic margin: {pessimistic_margin:+.1f}% -> pressure {downside_pressure:.1f}; scenario spread: {spread:.1f}% -> pressure {spread_pressure:.1f}; composite: {downside_score:.1f}.",
        )

    breakdown = (fundamental_confirmation or {}).get("breakdown") or {}
    available_signals = [item for item in breakdown.values() if isinstance(item, dict) and item.get("status") != "unavailable"]
    negative_signals = [item for item in available_signals if item.get("status") == "negative"]
    if not available_signals:
        _add_component("fundamental_deterioration", "Fundamental Deterioration", 0.25, None, None, "Fundamental confirmation data is unavailable.")
    else:
        deterioration = len(negative_signals) / len(available_signals) * 100.0
        _add_component(
            "fundamental_deterioration", "Fundamental Deterioration", 0.25, deterioration,
            f"Negative signals: {len(negative_signals)}/{len(available_signals)}; pressure: {deterioration:.1f}.",
        )

    quality_flags = valuation_result.get("quality_flags") or []
    warning_text = {str(flag).strip().lower() for flag in quality_flags if str(flag).strip()}
    terminal_warnings = {flag for flag in warning_text if "terminal consistency" in flag}
    severe_warnings = {
        flag for flag in warning_text
        if "terminal consistency" not in flag
        and any(term in flag for term in ("dispersion", "normalization risk", "missing", "unavailable", "only ", "extreme leverage", "high leverage"))
    }
    leverage = _number(valuation_result.get("net_debt_to_ebitda"))
    leverage_pressure = min(100.0, max(0.0, (leverage - 3.0) * 15.0)) if leverage is not None else None
    warning_pressure = min(100.0, len(terminal_warnings) * 12.0 + len(severe_warnings) * 20.0)
    risk_values = [warning_pressure] + ([leverage_pressure] if leverage_pressure is not None else [])
    if not risk_values:
        _add_component("model_risk", "Model/Risk Warning", 0.15, None, None, "Model risk inputs are unavailable.")
    else:
        risk_score = float(np.mean(risk_values))
        _add_component(
            "model_risk", "Model/Risk Warning", 0.15, risk_score,
            f"Distinct terminal warnings: {len(terminal_warnings)}; severe warnings: {len(severe_warnings)}; leverage pressure: {leverage_pressure if leverage_pressure is not None else 'n/a'}; result: {risk_score:.1f}.",
        )

    available = [component for component in components.values() if component["score"] is not None]
    available_weight = sum(component["weight"] for component in available)
    is_sufficient = components["valuation_excess"]["score"] is not None and len(available) >= 2
    total_score = None
    if is_sufficient and available_weight > 0:
        total_score = round(sum(component["score"] * component["weight"] for component in available) / available_weight)

    if total_score is None:
        tier = "Insufficient evidence"
    elif total_score >= 75:
        tier = "Strong Sell Candidate"
    elif total_score >= 55:
        tier = "Strong Trim"
    elif total_score >= 30:
        tier = "Consider Trim"
    else:
        tier = "Hold / Monitor"

    buy_price = _number(average_buy_price)
    position_gain = ((price / buy_price) - 1.0) * 100.0 if price is not None and buy_price is not None and buy_price > 0 else None
    return {
        "total_score": total_score,
        "tier": tier,
        "components": components,
        "available_components": len(available),
        "available_weight": available_weight,
        "is_sufficient": is_sufficient,
        "position_gain_pct": position_gain,
        "description": "Current sell-pressure evidence, not a personalized sell recommendation; excludes taxes, concentration, liquidity, and investor horizon.",
    }


def calculate_fundamental_confirmation(stock, info):
    """Score statement trends used by both individual and ETF Investment Scores."""
    def _row_series(frame, aliases):
        if frame is None or getattr(frame, "empty", True):
            return None
        for alias in aliases:
            matching = next((idx for idx in frame.index if alias.lower() in str(idx).lower()), None)
            if matching is not None:
                series = pd.to_numeric(frame.loc[matching], errors="coerce").dropna()
                series.index = pd.to_datetime(series.index, errors="coerce")
                return series[series.index.notna()].groupby(level=0).last().sort_index()
        return None

    def _combined(quarterly, annual):
        values = {}
        for series in (annual, quarterly):
            if series is not None:
                values.update({pd.Timestamp(date): float(value) for date, value in series.items()})
        return pd.Series(values).sort_index() if values else None

    def _recent_change(series):
        if series is None:
            return np.nan
        values = pd.to_numeric(series, errors="coerce").dropna().sort_index()
        if len(values) < 4:
            return np.nan
        window = values.tail(6)
        old_value = float(window.iloc[:3].median()) if len(window) >= 6 else float(window.iloc[0])
        new_value = float(window.iloc[-3:].median()) if len(window) >= 6 else float(window.iloc[-1])
        return (new_value - old_value) / abs(old_value) * 100.0 if abs(old_value) > 1e-9 else np.nan

    def _classify(value, label):
        if not np.isfinite(value):
            return {"value": None, "status": "unavailable", "label": label}
        status = "positive" if value > 3.0 else "negative" if value < -3.0 else "neutral"
        return {"value": round(value, 1), "status": status, "label": label}

    try:
        quarterly_financials = stock.quarterly_financials
        annual_financials = stock.financials
        quarterly_cashflow = stock.quarterly_cashflow
        annual_cashflow = stock.cashflow
        quarterly_balance = stock.quarterly_balance_sheet
        annual_balance = stock.balance_sheet

        annual_revenue = _row_series(annual_financials, ["Total Revenue", "Operating Revenue"])
        annual_net_income = _row_series(annual_financials, ["Net Income", "Net Earnings"])
        annual_shares = _row_series(annual_balance, ["Ordinary Shares Number", "Share Issued", "Common Stock Shares Outstanding", "Diluted Average Shares", "Basic Average Shares"])
        revenue = _combined(
            _row_series(quarterly_financials, ["Total Revenue", "Operating Revenue"]),
            annual_revenue,
        )
        operating_income = _combined(
            _row_series(quarterly_financials, ["Operating Income", "Ebit"]),
            _row_series(annual_financials, ["Operating Income", "Ebit"]),
        )
        net_income = _combined(
            _row_series(quarterly_financials, ["Net Income", "Net Earnings"]),
            annual_net_income,
        )
        operating_cashflow = _combined(
            _row_series(quarterly_cashflow, ["Operating Cash Flow"]),
            _row_series(annual_cashflow, ["Operating Cash Flow"]),
        )
        capex = _combined(
            _row_series(quarterly_cashflow, ["Capital Expenditure"]),
            _row_series(annual_cashflow, ["Capital Expenditure"]),
        )
        reported_fcf = _combined(
            _row_series(quarterly_cashflow, ["Free Cash Flow"]),
            _row_series(annual_cashflow, ["Free Cash Flow"]),
        )
        shares = _combined(
            _row_series(quarterly_balance, ["Ordinary Shares Number", "Share Issued", "Common Stock Shares Outstanding", "Diluted Average Shares", "Basic Average Shares"]),
            annual_shares,
        )
        fcf = None
        if operating_cashflow is not None and capex is not None:
            dates = operating_cashflow.index.intersection(capex.index)
            if len(dates) > 0:
                fcf = operating_cashflow.loc[dates] - capex.loc[dates].abs()
        fcf = fcf if fcf is not None else reported_fcf
        eps = None
        if net_income is not None:
            if shares is not None:
                aligned_shares = pd.to_numeric(shares.reindex(net_income.index), errors="coerce").ffill().bfill()
                valid = aligned_shares > 0
                if valid.any():
                    eps = net_income[valid] / aligned_shares[valid]
            if eps is None and info.get("sharesOutstanding"):
                eps = net_income / float(info["sharesOutstanding"])

        latest_annual_eps = None
        if annual_net_income is not None:
            annual_aligned_shares = (
                pd.to_numeric(annual_shares.reindex(annual_net_income.index), errors="coerce").ffill().bfill()
                if annual_shares is not None
                else pd.Series(float(info["sharesOutstanding"]), index=annual_net_income.index)
                if info.get("sharesOutstanding")
                else None
            )
            if annual_aligned_shares is not None:
                valid_annual = annual_aligned_shares > 0
                annual_eps = (annual_net_income[valid_annual] / annual_aligned_shares[valid_annual]).dropna()
                if not annual_eps.empty:
                    latest_annual_eps = float(annual_eps.sort_index().iloc[-1])

        breakdown = {
            "revenue_growth": _classify(_recent_change(revenue), "revenue growth"),
            "eps_growth": _classify(_recent_change(eps), "EPS growth"),
            "fcf_growth": _classify(_recent_change(fcf), "FCF growth"),
        }
        if latest_annual_eps is None or not np.isfinite(latest_annual_eps):
            breakdown["latest_annual_eps"] = {"value": None, "status": "unavailable", "label": "latest annual EPS"}
        else:
            breakdown["latest_annual_eps"] = {
                "value": round(latest_annual_eps, 3),
                "status": "positive" if latest_annual_eps > 0 else "negative" if latest_annual_eps < 0 else "neutral",
                "label": "latest annual EPS",
            }
        conversion = {"value": None, "status": "unavailable", "label": "cash conversion"}
        if fcf is not None and net_income is not None:
            dates = fcf.index.intersection(net_income.index)
            ratios = (fcf.loc[dates] / net_income.loc[dates]).replace([np.inf, -np.inf], np.nan).dropna()
            if not ratios.empty:
                ratio = float(ratios.tail(3).median())
                conversion = {"value": round(ratio * 100, 1), "status": "positive" if ratio >= 0.8 else "neutral" if ratio >= 0.5 else "negative", "label": "cash conversion"}
        breakdown["fcf_conversion"] = conversion
        margin = {"value": None, "status": "unavailable", "label": "operating margin"}
        if operating_income is not None and revenue is not None:
            dates = operating_income.index.intersection(revenue.index)
            margins = (operating_income.loc[dates] / revenue.loc[dates] * 100).replace([np.inf, -np.inf], np.nan).dropna()
            change = _recent_change(margins)
            if np.isfinite(change):
                margin = {"value": round(change, 2), "status": "positive" if change > 0.5 else "negative" if change < -0.5 else "neutral", "label": "operating margin"}
        breakdown["operating_margin_trend"] = margin
        share_change = _recent_change(shares)
        breakdown["share_dilution"] = {"value": round(share_change, 1) if np.isfinite(share_change) else None, "status": "positive" if np.isfinite(share_change) and share_change <= 0 else "negative" if np.isfinite(share_change) and share_change > 2 else "neutral" if np.isfinite(share_change) else "unavailable", "label": "share buybacks" if np.isfinite(share_change) and share_change <= 0 else "share dilution" if np.isfinite(share_change) and share_change > 2 else "share count"}
    except Exception as exc:
        logger.debug("Fundamental confirmation unavailable: %s", exc)
        return {"status": "insufficient", "score": None, "phrase": "Fundamental statement data is unavailable.", "breakdown": {}}

    available = [item for item in breakdown.values() if item["status"] != "unavailable"]
    if len(available) < 4:
        return {"status": "insufficient", "score": None, "phrase": "Insufficient confirmation data (fewer than 4 metrics available).", "breakdown": breakdown}
    score = round((sum(1 if item["status"] == "positive" else -1 if item["status"] == "negative" else 0 for item in available) + len(available)) / (2 * len(available)) * 100)
    status = "Supported" if score >= 70 else "Partly Supported" if score >= 40 else "Weakly Supported"
    return {"status": status, "score": score, "phrase": f"{status} - {score}/100 based on {len(available)} statement trends.", "breakdown": breakdown}


def fetch_yahoo_metadata(ticker, attempts=3, retry_delay=0.75):
    """Fetch Yahoo metadata with retries for transient crumb and rate-limit failures."""
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            if isinstance(info, dict) and info:
                return stock, info, None
            last_error = "Yahoo Finance returned empty metadata"
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"

        if attempt < attempts:
            logger.warning(
                "[%s] Yahoo metadata attempt %d/%d failed: %s; retrying",
                ticker,
                attempt,
                attempts,
                last_error,
            )
            time.sleep(retry_delay * attempt)

    logger.warning("[%s] Yahoo metadata unavailable after %d attempts: %s", ticker, attempts, last_error)
    return None, None, last_error


def compute_valuation_enhanced(ticker, use_historical_growth=True, assumptions_override=None):
    """
    Enhanced valuation with sector-aware method selection.
    

    Returns:
        dict with valuation metrics and quality indicators
    """
    
    stock, info, metadata_error = fetch_yahoo_metadata(ticker)
    if info is None:
        return {
            "error": (
                f"Yahoo Finance metadata is temporarily unavailable for {ticker}: {metadata_error}. "
                "Please retry shortly."
            )
        }
    
    # Basic info
    name = info.get("shortName", ticker)
    price = info.get("currentPrice")
    # Free price-performance fields already present on `info`, no extra request needed.
    price_change_1d = info.get("regularMarketChangePercent")
    price_change_1y = info.get("52WeekChange")
    currency = info.get("currency", "USD")
    financial_currency = info.get("financialCurrency", currency)
    raw_sector = info.get("sector", "Unknown")
    sector, sector_override_reason = get_valuation_sector_name(info)

    # Instruments with disable_all must exit before the price check so that
    # NOT_APPLICABLE / UNSUPPORTED_REGIME codes are always returned.
    _early_cfg = get_sector_config(sector)
    if _early_cfg and _early_cfg.special_rules.get("disable_all"):
        return {
            "error": _early_cfg.special_rules.get(
                "error_message", f"Valuation not supported for {sector}"
            ),
            "ticker": ticker,
            "name": name,
            "sector": sector,
            "confidence_score": 0.0,
            "confidence_description": "Sector unsupported",
        }

    if not validate_data(price):
        return {"error": f"No valid price data for {ticker}", "name": name}
    
    # Data quality check
    quality_flags = []
    assumption_deltas = {}
    policy_metadata = get_policy_metadata()
    if sector_override_reason:
        quality_flags.append(
            f"Valuation sector override applied: {raw_sector} -> {sector} ({sector_override_reason})"
        )

    # ===== SECTOR-AWARE CONTEXT (needed early for growth handling) =====
    sector_config = get_sector_config(sector)

    # Policy confidence with optional fallback blend.
    policy_confidence = float(sector_config.policy_confidence or sector_config.base_confidence)
    policy_fallback_sector = sector_config.fallback_parent_sector
    if policy_fallback_sector and policy_fallback_sector in SECTOR_CONFIGS:
        parent_conf = SECTOR_CONFIGS[policy_fallback_sector].base_confidence
        policy_confidence = 0.8 * policy_confidence + 0.2 * parent_conf

    # Preserve raw assumptions before any policy normalization.
    raw_assumptions = {
        "discount_rate": None,
        "revenue_growth": info.get("revenueGrowth"),
        "profit_margin": info.get("profitMargins"),
        "earnings_growth": info.get("earningsGrowth"),
    }

    # Apply field-level policy handling to assumptions used downstream.
    normalized_assumptions = {}
    normalized_assumptions["revenue_growth"] = _apply_policy_range(
        raw_assumptions["revenue_growth"],
        "revenue_growth",
        getattr(sector_config, "revenue_growth_range", None),
        sector_config.field_enforcement.get("revenue_growth"),
        quality_flags,
        assumption_deltas,
    )
    normalized_assumptions["profit_margin"] = _apply_policy_range(
        raw_assumptions["profit_margin"],
        "profit_margin",
        getattr(sector_config, "profit_margin_range", None),
        sector_config.field_enforcement.get("profit_margin"),
        quality_flags,
        assumption_deltas,
    )
    normalized_assumptions["earnings_growth"] = _apply_policy_range(
        raw_assumptions["earnings_growth"],
        "earnings_growth",
        getattr(sector_config, "earnings_growth_range", None),
        sector_config.field_enforcement.get("earnings_growth"),
        quality_flags,
        assumption_deltas,
    )

    # Feed normalized fundamentals into downstream growth logic without mutating provider payload.
    valuation_info = dict(info)
    valuation_info["revenueGrowth"] = normalized_assumptions["revenue_growth"]
    valuation_info["profitMargins"] = normalized_assumptions["profit_margin"]
    valuation_info["earningsGrowth"] = normalized_assumptions["earnings_growth"]
    fundamental_confirmation = calculate_fundamental_confirmation(stock, info)
    
    # FCF extraction with validation
    raw_fcf_per_share, fcf_provenance = calculate_fcf_per_share(stock, info)
    fcf_per_share, normalized_fcf_details = get_normalized_fcf_per_share(stock, info, raw_fcf_per_share)
    # For capex-heavy telecom/cable, replace point-in-time FCF with 3–5Y normalized capex baseline.
    if sector_config.special_rules.get("use_normalized_capex") and normalized_fcf_details is None:
        _norm_fcf_ps, _telecom_norm = _normalize_telecom_fcf_per_share(stock, info, raw_fcf_per_share)
        if _telecom_norm and _telecom_norm.get("applied"):
            fcf_per_share = _norm_fcf_ps
            normalized_fcf_details = _telecom_norm
            quality_flags.append(
                f"Telecom normalized capex: avg capex/rev={_telecom_norm['avg_capex_to_revenue']:.1%} "
                f"over {_telecom_norm['years_used']}Y "
                f"({(_telecom_norm.get('raw_fcf_per_share') or 0):.2f}/sh → {_telecom_norm['normalized_fcf_per_share']:.2f}/sh)"
            )
    if not validate_data(fcf_per_share):
        fcf_per_share = None
        quality_flags.append("Missing FCF data")
    elif normalized_fcf_details and normalized_fcf_details.get("applied"):
        quality_flags.append(
            "Normalized FCF baseline applied for reinvestment-heavy platform "
            f"({normalized_fcf_details['raw_fcf_per_share']:.2f}/sh -> {normalized_fcf_details['normalized_fcf_per_share']:.2f}/sh)"
        )

    # FCF credibility check: FCF > EBITDA is structurally impossible for any
    # normally-operating company after capex and taxes.  When it occurs it signals
    # large non-cash add-backs — typically amortisation of acquired intangibles
    # (e.g. post-merger goodwill), content library write-downs, or restructuring
    # charges being reversed through OCF.  Cap FCF at EBITDA × 0.65 / shares,
    # a conservative FCF conversion ratio (capex + taxes typically consume 30–40%
    # of EBITDA).  This prevents inflated FCF from dominating the DCF output.
    if fcf_per_share is not None and fcf_per_share > 0:
        _ebitda_fcf_chk = info.get("ebitda") or 0
        _shares_fcf_chk = info.get("sharesOutstanding") or 1
        if _ebitda_fcf_chk > 0 and _shares_fcf_chk > 0:
            _ebitda_ps = _ebitda_fcf_chk / _shares_fcf_chk
            if fcf_per_share > _ebitda_ps:
                _normalised_fcf = _ebitda_ps * 0.65
                quality_flags.append(
                    f"\u26a0 FCF/share ({fcf_per_share:.2f}) exceeds EBITDA/share ({_ebitda_ps:.2f}): "
                    f"likely inflated by non-cash add-backs (merger amortisation, restructuring). "
                    f"Normalised to {_normalised_fcf:.2f} for DCF."
                )
                fcf_per_share = _normalised_fcf

    pessimistic_fcf_haircut = 0.0
    confirmation_breakdown = (fundamental_confirmation or {}).get("breakdown") or {}
    conversion_pct = confirmation_breakdown.get("fcf_conversion", {}).get("value")
    if isinstance(conversion_pct, (int, float)) and np.isfinite(float(conversion_pct)) and conversion_pct > 150.0:
        pessimistic_fcf_haircut += min(0.20, (float(conversion_pct) - 150.0) / 320.0)
    if confirmation_breakdown.get("latest_annual_eps", {}).get("status") == "negative":
        pessimistic_fcf_haircut += 0.10
    pessimistic_fcf_haircut = min(0.20, pessimistic_fcf_haircut)

    if isinstance(fcf_provenance, dict):
        source_method = fcf_provenance.get("source_method")
        trust_level = fcf_provenance.get("trust_level")
        quality_assessment = fcf_provenance.get("quality_assessment")
        if source_method:
            trust_text = f", trust={trust_level:.0%}" if isinstance(trust_level, (int, float)) else ""
            quality_flags.append(
                f"FCF provenance: {source_method} ({quality_assessment}{trust_text})"
            )
    
    # Calculate historical growth if available
    historical_fcf_growth = None
    historical_eps_growth = None
    
    if use_historical_growth:
        try:
            # Historical FCF growth
            # Use GAAP OCF − CapEx series (same source as calculate_fcf_per_share).
            # CapEx is stored as negative in yfinance so OCF + CapEx = OCF − |CapEx|.
            # Fall back to the pre-computed 'Free Cash Flow' row only when OCF or
            # CapEx data is unavailable.
            cashflow = stock.cashflow
            if ("Operating Cash Flow" in cashflow.index
                    and "Capital Expenditure" in cashflow.index):
                ocf_hist   = cashflow.loc["Operating Cash Flow"].dropna()
                capex_hist = cashflow.loc["Capital Expenditure"].dropna()
                fcf_series = (ocf_hist + capex_hist).dropna()  # CapEx is negative
                historical_fcf_growth = calculate_historical_growth(fcf_series, years=5)
            elif "Free Cash Flow" in cashflow.index:
                fcf_series = cashflow.loc["Free Cash Flow"].dropna()
                historical_fcf_growth = calculate_historical_growth(fcf_series, years=5)
            
            # Historical EPS growth
            hist = stock.quarterly_financials
            if "Net Income" in hist.index:
                ni_series = hist.loc["Net Income"].dropna()
                shares = info.get("sharesOutstanding")
                if shares and len(ni_series) > 0:
                    eps_series = ni_series / shares
                    historical_eps_growth = calculate_historical_growth(eps_series, years=5)
        except:
            pass
    
    # Resolve debt once and reuse in WACC + EV for consistency.
    net_debt_data = resolve_net_debt_components(stock, info)
    quality_flags.append(
        f"Net debt source: {net_debt_data.get('source')} ({net_debt_data.get('freshness')})"
    )

    # WACC calculation
    discount_rate = calculate_enhanced_wacc(stock, info, net_debt_data=net_debt_data)

    # Leverage-aware WACC uplift for high-ND/EBITDA sectors (telecom, cable).
    # Applied once here; EV/EBITDA already has its own separate multiple haircut.
    if sector_config.special_rules.get("leverage_wacc_uplift"):
        _ebitda_lev = info.get("ebitda") or 0
        _nd_lev = net_debt_data.get("net_debt", 0) or 0
        if _ebitda_lev > 0 and _nd_lev > 0:
            _nd_ebitda = _nd_lev / _ebitda_lev
            # Linear 0→150 bps from 2× to 5×; capped at 150 bps above 5×.
            if _nd_ebitda >= 5.0:
                _uplift = 0.015
            elif _nd_ebitda > 2.0:
                _uplift = (_nd_ebitda - 2.0) / 3.0 * 0.015
            else:
                _uplift = 0.0
            if _uplift > 0:
                quality_flags.append(
                    f"Leverage WACC uplift: +{_uplift*100:.0f} bps (ND/EBITDA={_nd_ebitda:.1f}×)"
                )
                discount_rate = min(discount_rate + _uplift, 0.20)

    if assumptions_override and assumptions_override.get("discount_rate") is not None:
        try:
            custom_discount_rate = float(assumptions_override.get("discount_rate"))
            custom_discount_rate = max(0.005, min(0.20, custom_discount_rate))
            discount_rate = custom_discount_rate
            quality_flags.append(f"Custom discount rate applied: {discount_rate*100:.2f}%")
        except Exception:
            quality_flags.append("Ignored invalid custom discount rate")

    raw_assumptions["discount_rate"] = discount_rate
    normalized_assumptions["discount_rate"] = _apply_policy_range(
        discount_rate,
        "discount_rate",
        getattr(sector_config, "discount_rate_range", None),
        sector_config.field_enforcement.get("discount_rate"),
        quality_flags,
        assumption_deltas,
    )
    discount_rate = normalized_assumptions["discount_rate"]
    
    # Valuation models
    dcf_vals = {}
    pe_vals = {}
    ev_vals = {}
    growth_diag = {}
    
    if fcf_per_share and fcf_per_share > 0:
        # Apply robust multi-signal, sector-aware growth normalization for DCF
        dcf_growth_input, growth_diag = compute_normalized_dcf_growth(
            valuation_info,
            sector_config,
            historical_fcf_growth,
            historical_eps_growth,
            sector,
            fundamental_confirmation=fundamental_confirmation,
        )
        dcf_vals = dcf_scenarios_enhanced(
            stock,
            fcf_per_share,
            discount_rate,
            dcf_growth_input,
            stock_info=info,
            sector=sector,
            assumptions_override=assumptions_override,
            growth_quality_adjustment=growth_diag.get("quality_adjustment", 0.0),
            pessimistic_fcf_haircut=pessimistic_fcf_haircut,
        )
        quality_flags.append(
            f"Normalized DCF growth applied: {dcf_growth_input*100:.1f}% "
            f"(signals={growth_diag.get('signal_count', 0)}, reversion={growth_diag.get('reversion', 0)*100:.0f}%)"
        )
        if growth_diag.get("quality_adjustment", 0) > 0:
            quality_flags.append(
                "DCF growth sustainability adjustment applied: "
                f"-{growth_diag['quality_adjustment']*100:.1f} percentage points "
                f"(FCF conversion={growth_diag.get('fcf_conversion_pct')}, "
                f"latest annual EPS={growth_diag.get('latest_annual_eps_status')})"
            )
        if pessimistic_fcf_haircut > 0:
            quality_flags.append(
                "Pessimistic DCF FCF sustainability haircut applied: "
                f"-{pessimistic_fcf_haircut*100:.1f}% "
                f"(FCF conversion={conversion_pct}, "
                f"latest annual EPS={confirmation_breakdown.get('latest_annual_eps', {}).get('status')})"
            )
        signal_names = [s[0] for s in growth_diag.get("signals", [])]
        if signal_names:
            quality_flags.append(f"Expected growth sources: {', '.join(signal_names[:5])}")
    else:
        quality_flags.append("DCF unavailable (no FCF)")
    
    pe_vals = pe_scenarios_enhanced(
        stock,
        valuation_info,
        discount_rate,
        valuation_info.get("forwardPE"),
        sector=sector,
        sector_config=sector_config,
        assumptions_override=assumptions_override,
        quality_flags=quality_flags,
        assumption_deltas=assumption_deltas,
    )

    if assumptions_override:
        custom_dcf = assumptions_override.get("dcf_growth_scenarios")
        if isinstance(custom_dcf, dict) and all(k in custom_dcf for k in ["Pessimistic", "Base", "Optimistic"]):
            quality_flags.append("Custom DCF growth scenarios applied")
        custom_pe = assumptions_override.get("pe_target_multiples")
        if isinstance(custom_pe, dict) and all(k in custom_pe for k in ["Pessimistic", "Base", "Optimistic"]):
            quality_flags.append("Custom future P/E target multiples applied")
        if assumptions_override.get("dcf_stage2_growth") is not None:
            quality_flags.append("Custom DCF terminal growth applied")

    dcf_assumptions = dcf_vals.pop("_assumptions", None) if isinstance(dcf_vals, dict) else None
    pe_assumptions = pe_vals.pop("_assumptions", None) if isinstance(pe_vals, dict) else None

    if isinstance(dcf_assumptions, dict):
        caps = dcf_assumptions.get("growth_caps_applied") or {}
        if caps:
            cap_parts = []
            for scenario in ["Pessimistic", "Base", "Optimistic"]:
                cap_info = caps.get(scenario)
                if cap_info:
                    cap_parts.append(
                        f"{scenario}: {cap_info['from']*100:.1f}%→{cap_info['to']*100:.1f}%"
                    )
            if cap_parts:
                quality_flags.append(
                    "DCF growth capped below discount rate for stability (" + ", ".join(cap_parts) + ")"
                )
        if dcf_assumptions.get("mega_cap_software_mode"):
            quality_flags.append(
                "Mega-cap software calibration active (lower terminal FCF yield and softer DCF growth caps)"
            )
        if dcf_assumptions.get("mega_cap_ai_infra_mode"):
            quality_flags.append(
                "Mega-cap AI infrastructure H-model active (3-year explosive growth, 7-year fade, GDP-like terminal growth)"
            )
        for warning in dcf_assumptions.get("terminal_consistency_warnings", []):
            quality_flags.append(f"⚠️ DCF terminal consistency: {warning}")
        if dcf_assumptions.get("stage2_policy_adjustment"):
            adj = dcf_assumptions["stage2_policy_adjustment"]
            quality_flags.append(
                "DCF terminal growth normalized to sector policy "
                f"({adj['from']*100:.2f}% -> {adj['to']*100:.2f}%)"
            )
        if dcf_assumptions.get("terminal_yield_policy_adjustment"):
            adj = dcf_assumptions["terminal_yield_policy_adjustment"]
            quality_flags.append(
                "DCF terminal yield normalized to sector policy "
                f"({adj['from']*100:.2f}% -> {adj['to']*100:.2f}%)"
            )
        if dcf_assumptions.get("optimistic_growth_cap") is not None and sector in {"Gov Defense IT", "Gov Defense IT Hybrid"}:
            quality_flags.append(
                f"Gov defense IT optimistic DCF growth capped at {dcf_assumptions['optimistic_growth_cap']*100:.1f}%"
            )

    pe_guardrail = pe_vals.pop("_pe_guardrail", None) if isinstance(pe_vals, dict) else None
    if pe_guardrail:
        quality_flags.append(
            "P/E guardrail applied: "
            f"raw {pe_guardrail['raw_pe']:.1f}x → adjusted {pe_guardrail['adjusted_pe']:.1f}x "
            f"(sector baseline {pe_guardrail['sector_pe']:.1f}x)"
        )
    
    ev_vals = ev_ebitda_scenarios_enhanced(
        stock,
        valuation_info,
        net_debt_data=net_debt_data,
        sector_config=sector_config,
        quality_flags=quality_flags,
        assumption_deltas=assumption_deltas,
    )
    ev_assumptions = ev_vals.pop("_assumptions", None) if isinstance(ev_vals, dict) else None

    # P/B valuation (for financial institutions)
    pb_vals = pb_scenarios_enhanced(stock, valuation_info, net_debt_data=net_debt_data) if sector_config.method_weights.get("P/B", 0) > 0 else null_scenarios()
    pb_assumptions = pb_vals.pop("_assumptions", None) if isinstance(pb_vals, dict) else None

    # ROE valuation (for financial institutions)
    roe_vals = roe_scenarios_enhanced(stock, valuation_info, net_debt_data=net_debt_data) if sector_config.method_weights.get("ROE", 0) > 0 else null_scenarios()
    roe_assumptions = roe_vals.pop("_assumptions", None) if isinstance(roe_vals, dict) else None

    # Currency normalization: DCF/EV/P/B/ROE are often based on financial-statement currency.
    # Convert those method outputs to quote currency before blending.
    fx_fin_to_quote = 1.0
    if financial_currency and currency and financial_currency != currency:
        fx_fin_to_quote = get_fx_rate(financial_currency, currency)
        if fx_fin_to_quote and fx_fin_to_quote > 0:
            dcf_vals = {k: (v * fx_fin_to_quote if v is not None else None) for k, v in dcf_vals.items()}
            pe_vals  = {k: (v * fx_fin_to_quote if v is not None else None) for k, v in pe_vals.items()}
            ev_vals  = {k: (v * fx_fin_to_quote if v is not None else None) for k, v in ev_vals.items()}
            pb_vals  = {k: (v * fx_fin_to_quote if v is not None else None) for k, v in pb_vals.items()}
            roe_vals = {k: (v * fx_fin_to_quote if v is not None else None) for k, v in roe_vals.items()}
            quality_flags.append(
                f"Converted DCF/PE/EV/P/B/ROE from {financial_currency} to {currency} (fx={fx_fin_to_quote:.4f})"
            )
        else:
            quality_flags.append(
                f"Currency mismatch ({financial_currency}->{currency}) with no FX conversion available"
            )
            fx_fin_to_quote = None

    
    # ===== SECTOR-AWARE VALUATION =====
    # Check if sector is fundamentally unsupported (REITs, etc.)
    if sector_config.special_rules.get("disable_all"):
        return {
            "error": sector_config.special_rules.get("error_message", f"Valuation not supported for {sector}"),
            "ticker": ticker,
            "name": name,
            "sector": sector,
            "confidence_score": 0.0,
            "confidence_description": "Sector unsupported",
        }
    
    # Apply sector-specific method disabling
    # For Telecom (capex-heavy, regulated), DCF is unreliable; use P/E and EV/EBITDA instead
    if sector_config.special_rules.get("disable_dcf"):
        dcf_vals = null_scenarios()
        quality_flags.append(
            f"DCF disabled for {sector} sector (high capex intensity and leverage make FCF-based valuation unreliable)"
        )
    
    if sector_config.special_rules.get("disable_ev_ebitda"):
        ev_vals = null_scenarios()
        quality_flags.append(f"EV/EBITDA disabled for {sector} sector (EBITDA metrics don't apply)")

    method_validity_flags = {
        "DCF": {"status": "valid", "reason": None},
        "P/E": {"status": "valid", "reason": None},
        "EV/EBITDA": {"status": "valid", "reason": None},
        "P/B": {"status": "valid", "reason": None},
        "ROE": {"status": "valid", "reason": None},
    }

    if dcf_vals.get("Base") is None:
        method_validity_flags["DCF"] = {"status": "invalid_data", "reason": "No usable DCF output"}
    if pe_vals.get("Base") is None:
        method_validity_flags["P/E"] = {"status": "invalid_data", "reason": "No usable P/E output"}
    if ev_vals.get("Base") is None:
        reason = "policy_excluded" if sector_config.special_rules.get("disable_ev_ebitda") else "invalid_data"
        method_validity_flags["EV/EBITDA"] = {"status": reason, "reason": "No usable EV/EBITDA output"}
    if pb_vals.get("Base") is None:
        method_validity_flags["P/B"] = {"status": "invalid_data", "reason": "No usable P/B output"}
    if roe_vals.get("Base") is None:
        method_validity_flags["ROE"] = {"status": "invalid_data", "reason": "No usable ROE output"}
    
    # Apply sector-specific outlier thresholds
    dcf_vals = exclude_outlier_method(
        "DCF", dcf_vals, price, quality_flags,
        min_ratio=0.05,
        max_ratio=sector_config.outlier_max_ratio
    )
    pe_vals = exclude_outlier_method("P/E", pe_vals, price, quality_flags,
                                    max_ratio=sector_config.outlier_max_ratio)
    ev_vals = exclude_outlier_method("EV/EBITDA", ev_vals, price, quality_flags,
                                    max_ratio=sector_config.outlier_max_ratio)
    pb_vals = exclude_outlier_method("P/B", pb_vals, price, quality_flags,
                                    max_ratio=sector_config.outlier_max_ratio)
    roe_vals = exclude_outlier_method("ROE", roe_vals, price, quality_flags,
                                     max_ratio=sector_config.outlier_max_ratio)
    
    if sector_config.outlier_max_ratio < 8.0:
        quality_flags.append(f"Sector-specific outlier threshold: {sector_config.outlier_max_ratio}x (tighter for {sector})")

    # Keep DCF outputs intact: reliability is handled in method weighting,
    # not by rewriting DCF values toward peer-implied anchors.
    
    # Add low-confidence warning if applicable
    if sector_config.special_rules.get("low_confidence_warning"):
        quality_flags.append(f"⚠️ {sector_config.special_rules['low_confidence_warning']}")

    # ===== AUTO-CALIBRATION: Detect growth profile and adjust weights =====
    earnings_cagr = calculate_earnings_cagr(stock, years=5)
    growth_profile = GrowthProfile(
        historical_growth=earnings_cagr,
        current_pe=info.get("trailingPE"),
        sector=sector,
        forward_pe=info.get("forwardPE")
    )
    calibration = growth_profile.get_calibrated_params()
    
    # Build final weights: sector weights are the base policy; growth profile provides a modest tilt.
    profile_weights = calibration["method_weights"]
    profile_weights = {
        "DCF": profile_weights.get("DCF", 0.33),
        "P/E": profile_weights.get("PE", 0.33),
        "EV/EBITDA": profile_weights.get("EV_EBITDA", 0.33)
    }
    sector_weights = get_sector_method_weights(sector_config.name)
    is_cyclical_or_regulated = (
        sector_config.special_rules.get("use_normalized_commodity")
        or sector_config.special_rules.get("use_normalized_capex")
        or sector_config.special_rules.get("regulated")
    )

    if is_cyclical_or_regulated:
        weights = {"DCF": 0.0, "P/E": 0.0, "EV/EBITDA": 0.0, **sector_weights}
    else:
        # Use .get() so single-method sectors (e.g. Biotech: P/E only) don't raise
        # a KeyError when DCF or EV/EBITDA are absent from the weights dict.
        # Only blend profile_weights if sector_weights is non-zero (avoid false defaults)
        weights = {
            "DCF": sector_weights.get("DCF", 0.0) * 0.75 + profile_weights.get("DCF", 0.0) * 0.25 if sector_weights.get("DCF", 0.0) > 0 else 0.0,
            "P/E": sector_weights.get("P/E", 0.0) * 0.75 + profile_weights.get("P/E", 0.0) * 0.25 if sector_weights.get("P/E", 0.0) > 0 else 0.0,
            "EV/EBITDA": sector_weights.get("EV/EBITDA", 0.0) * 0.75 + profile_weights.get("EV/EBITDA", 0.0) * 0.25 if sector_weights.get("EV/EBITDA", 0.0) > 0 else 0.0,
            "P/B": sector_weights.get("P/B", 0.0),
            "ROE": sector_weights.get("ROE", 0.0),
        }

    total_w = sum(weights.values())
    if total_w > 0:
        weights = {k: v / total_w for k, v in weights.items()}

    # Reliability-only weight calibration (no price-distance or peer-distance rules).
    reliability_reasons = []
    data_issues_count = len([f for f in quality_flags if "Missing" in f or "unavailable" in f.lower()])
    if data_issues_count >= 2 and weights.get("DCF", 0) > 0:
        shift = min(0.07, weights.get("DCF", 0) * 0.20)
        weights["DCF"] -= shift
        weights["P/E"] = weights.get("P/E", 0.0) + shift * 0.60
        weights["EV/EBITDA"] = weights.get("EV/EBITDA", 0.0) + shift * 0.40
        reliability_reasons.append("data_quality")

    fcf_unstable = bool(normalized_fcf_details and normalized_fcf_details.get("applied"))
    if fcf_unstable and weights.get("DCF", 0) > 0:
        shift = min(0.10, weights.get("DCF", 0) * 0.25)
        weights["DCF"] -= shift
        weights["P/E"] = weights.get("P/E", 0.0) + shift * 0.55
        weights["EV/EBITDA"] = weights.get("EV/EBITDA", 0.0) + shift * 0.45
        reliability_reasons.append("fcf_stability")

    if is_cyclical_or_regulated and weights.get("DCF", 0) > 0:
        shift = min(0.08, weights.get("DCF", 0) * 0.20)
        weights["DCF"] -= shift
        weights["P/E"] = weights.get("P/E", 0.0) + shift * 0.50
        weights["EV/EBITDA"] = weights.get("EV/EBITDA", 0.0) + shift * 0.50
        reliability_reasons.append("sector_applicability")

    # For mega-cap software/platform names, reduce DCF dominance slightly
    # and lean more on market-implied multiple methods.
    mega_cap_software_mode = bool(dcf_assumptions and dcf_assumptions.get("mega_cap_software_mode"))
    if mega_cap_software_mode and weights["DCF"] > 0.35:
        shift = min(0.08, weights["DCF"] - 0.35)
        weights["DCF"] -= shift
        weights["P/E"] += shift * 0.65
        weights["EV/EBITDA"] += shift * 0.35

        total_w = sum(weights.values())
        if total_w > 0:
            weights = {k: v / total_w for k, v in weights.items()}

        quality_flags.append("Mega-cap software method-weight tilt applied (less DCF concentration)")

    preblend_method_values = [v for v in [dcf_vals.get("Base"), pe_vals.get("Base"), ev_vals.get("Base")] if v is not None]
    base_method_dispersion = None
    if len(preblend_method_values) >= 2:
        median_val = float(np.median(preblend_method_values))
        if abs(median_val) > 1e-9:
            preblend_dispersion = (max(preblend_method_values) - min(preblend_method_values)) / abs(median_val)
            base_method_dispersion = preblend_dispersion
            if preblend_dispersion > 1.00 and weights.get("DCF", 0) > 0:
                shift = min(0.05, weights.get("DCF", 0) * 0.15)
                weights["DCF"] -= shift
                weights["P/E"] = weights.get("P/E", 0.0) + shift * 0.50
                weights["EV/EBITDA"] = weights.get("EV/EBITDA", 0.0) + shift * 0.50
                reliability_reasons.append("method_dispersion")

    total_w = sum(weights.values())
    if total_w > 0:
        weights = {k: v / total_w for k, v in weights.items()}
    if reliability_reasons:
        quality_flags.append(
            "Reliability-based method-weight calibration applied (" + ", ".join(sorted(set(reliability_reasons))) + ")"
        )
    
    logger.info(f"[{ticker}] Auto-calibration: {calibration['profile']} ({calibration['description']})")
    logger.debug(f"[{ticker}] Adjusted weights - DCF: {weights['DCF']:.0%}, P/E: {weights['P/E']:.0%}, EV/EBITDA: {weights['EV/EBITDA']:.0%}")
    logger.debug(f"[{ticker}] Sector: {sector} | Config: {sector_config.name} | Methods: {sector_config.primary_methods}")
    
    # Count available methods and track which are used
    # Only include methods that have both data AND positive weight in sector config
    methods_available = []
    if dcf_vals.get("Base") is not None and weights.get("DCF", 0) > 0:
        methods_available.append("DCF")
    if pe_vals.get("Base") is not None and weights.get("P/E", 0) > 0:
        methods_available.append("P/E")
    if ev_vals.get("Base") is not None and weights.get("EV/EBITDA", 0) > 0:
        methods_available.append("EV/EBITDA")
    if pb_vals.get("Base") is not None and weights.get("P/B", 0) > 0:
        methods_available.append("P/B")
    if roe_vals.get("Base") is not None and weights.get("ROE", 0) > 0:
        methods_available.append("ROE")
    
    if len(methods_available) < sector_config.min_required_methods:
        quality_flags.append(f"Only {len(methods_available)}/{sector_config.min_required_methods} required methods available for {sector}")

    if methods_available == ["P/E"]:
        quality_flags.append("Valuation relies on P/E only; treat result as directional, not precise")

    trailing_pe = info.get("trailingPE")
    if validate_data(trailing_pe) and trailing_pe > 100:
        quality_flags.append(f"Extreme trailing P/E ({trailing_pe:.1f}x) detected; multiple-compression risk is elevated")
    
    blended = {}

    for scenario in ["Pessimistic", "Base", "Optimistic"]:
        dcf_val = dcf_vals.get(scenario)
        pe_val = pe_vals.get(scenario)
        ev_val = ev_vals.get(scenario)
        pb_val = pb_vals.get(scenario)
        roe_val = roe_vals.get(scenario)
        
        valid_values = [v for v in [dcf_val, pe_val, ev_val, pb_val, roe_val] if v is not None]
        
        if not valid_values:
            blended[scenario] = None
            continue
        
        # Weighted blend with sector-specific weights (reweighted by availability)
        w_dcf = weights.get("DCF", 0.33) if dcf_val is not None else 0
        w_pe = weights.get("P/E", 0.33) if pe_val is not None else 0
        w_ev = weights.get("EV/EBITDA", 0.34) if ev_val is not None else 0
        w_pb = weights.get("P/B", 0.0) if pb_val is not None else 0
        w_roe = weights.get("ROE", 0.0) if roe_val is not None else 0
        total = w_dcf + w_pe + w_ev + w_pb + w_roe

        if total > 0:
            blended_val = (
                (dcf_val or 0) * w_dcf / total +
                (pe_val or 0) * w_pe / total +
                (ev_val or 0) * w_ev / total +
                (pb_val or 0) * w_pb / total +
                (roe_val or 0) * w_roe / total
            )
            blended[scenario] = blended_val

    # Build weights string for quality flag
    weight_str_parts = []
    for method in ["DCF", "P/E", "EV/EBITDA", "P/B", "ROE"]:
        w = weights.get(method, 0)
        if w > 0:
            weight_str_parts.append(f"{method}: {w:.0%}")
    weight_str = ", ".join(weight_str_parts) if weight_str_parts else "No methods available"
    quality_flags.append(f"Sector-weighted blend active ({weight_str})")

    # Enforce scenario monotonicity: Pessimistic <= Base <= Optimistic.
    # Method exclusions (e.g. DCF dropped for high-growth scenarios) can invert the order;
    # clamping the extremes against Base restores interpretability without changing Base.
    _base = blended.get("Base")
    if _base is not None:
        if blended.get("Pessimistic") is not None:
            blended["Pessimistic"] = min(blended["Pessimistic"], _base)
        if blended.get("Optimistic") is not None:
            blended["Optimistic"] = max(blended["Optimistic"], _base)

    distressed_ev_only_no_fair_value = (
        methods_available == ["EV/EBITDA"]
        and blended.get("Base") is not None
        and blended["Base"] <= 0
    )
    if distressed_ev_only_no_fair_value:
        quality_flags.append(
            "No reliable fair value: EV/EBITDA-only implied equity value is <= 0 after net debt; "
            "treated as a distressed/high-leverage no-valuation case."
        )
        for scenario in ["Pessimistic", "Base", "Optimistic"]:
            blended[scenario] = None

    # Keep intrinsic blended result before realism adjustment (for transparency)
    intrinsic_blended = dict(blended)

    # Calculate confidence score
    confidence_score, confidence_description = calculate_confidence_score(
        len(methods_available),
        sector_config.min_required_methods,
        policy_confidence,
        len([f for f in quality_flags if "Missing" in f or "unavailable" in f.lower()])
    )

    # Explicit confidence penalty for stale net-debt source fidelity.
    net_debt_source = (net_debt_data.get("source") or "").lower()
    net_debt_freshness = (net_debt_data.get("freshness") or "").lower()
    if net_debt_source == "info_fallback":
        confidence_score = max(0.0, confidence_score * 0.93)
        quality_flags.append("Net debt uses point-in-time fallback fields; confidence slightly reduced")
    if net_debt_freshness not in {"latest_reported", ""}:
        confidence_score = max(0.0, confidence_score * 0.95)
        quality_flags.append(f"Net debt freshness is {net_debt_data.get('freshness')}; confidence reduced")

    # Stronger confidence response to cross-method disagreement.
    if isinstance(base_method_dispersion, (int, float)) and base_method_dispersion > 0.75:
        dispersion_penalty = min(0.22, max(0.0, (base_method_dispersion - 0.75) * 0.14))
        if dispersion_penalty > 0:
            confidence_score = max(0.0, confidence_score * (1.0 - dispersion_penalty))
            quality_flags.append(
                "High method dispersion reduced confidence "
                f"({base_method_dispersion:.2f}x spread vs median)"
            )

    # Preserve valuation-model reliability before applying capital-structure risk.
    # Investment Score uses this for Robustness; Downside Resilience owns leverage.
    model_confidence_score = confidence_score

    # Extreme leverage check: Net Debt/EBITDA > 3.5× significantly increases equity
    # risk (thin residual claim amplifies every model error).  Warn and reduce
    # confidence so the output is not treated as a reliable buy/sell signal.
    _ebitda_chk = info.get("ebitda") or 0
    _nd_chk = net_debt_data.get("net_debt", 0) or 0
    net_debt_to_ebitda = _nd_chk / _ebitda_chk if _ebitda_chk > 0 else None
    if _ebitda_chk > 0 and _nd_chk > 0:
        _nd_ebitda = net_debt_to_ebitda
        if _nd_ebitda >= 5.0:
            confidence_score = max(0.0, confidence_score * 0.75)
            quality_flags.append(
                f"\u26a0 Extreme leverage: Net Debt/EBITDA={_nd_ebitda:.1f}\u00d7 (\u22655\u00d7). "
                "Equity is a thin residual claim; all fair-value estimates carry "
                "very wide uncertainty bands."
            )
        elif _nd_ebitda >= 3.5:
            confidence_score = max(0.0, confidence_score * 0.88)
            quality_flags.append(
                f"High leverage: Net Debt/EBITDA={_nd_ebitda:.1f}\u00d7 (\u22653.5\u00d7). "
                "Valuation confidence reduced \u2014 leverage amplifies model errors."
            )
    confidence_score = min(1.0, confidence_score)
    confidence_description = _describe_confidence_level(confidence_score)

    # Realism adjustment layer:
    # keep intrinsic math, then apply confidence/coverage/dispersion-based anchoring to market.
    # This improves comparability with external valuation sources without ticker-specific rules.
    anchor_strength = 0.0
    method_shortfall = max(0, sector_config.min_required_methods - len(methods_available))
    if method_shortfall > 0:
        anchor_strength += 0.10 * method_shortfall

    # Only anchor for very low confidence cases
    if confidence_score < 0.50:
        anchor_strength += (0.50 - confidence_score) * 0.35

    # Penalize high disagreement between methods (unstable intrinsic estimate)
    base_method_values = [v for v in [dcf_vals.get("Base"), pe_vals.get("Base"), ev_vals.get("Base")] if v is not None]
    if len(base_method_values) >= 2:
        median_val = float(np.median(base_method_values))
        if abs(median_val) > 1e-9:
            dispersion = (max(base_method_values) - min(base_method_values)) / abs(median_val)
            if dispersion > 1.00:
                anchor_strength += min(0.20, (dispersion - 1.00) * 0.18)

    # Apply stronger realism anchoring for government-services defense IT sectors.
    if sector in {"Gov Defense IT", "Gov Defense IT Hybrid"} and anchor_strength > 0:
        anchor_strength *= 1.15

    anchor_strength = min(0.35, max(0.0, anchor_strength))

    if anchor_strength > 0 and validate_data(price) and any(blended.get(scenario) is not None for scenario in ["Pessimistic", "Base", "Optimistic"]):
        anchor_price = float(price) if price is not None else 0.0
        for scenario in ["Pessimistic", "Base", "Optimistic"]:
            scenario_value = blended.get(scenario)
            if scenario_value is not None:
                blended[scenario] = float(scenario_value) * (1 - anchor_strength) + anchor_price * anchor_strength
        quality_flags.append(
            f"Applied realism anchor ({anchor_strength:.0%}) toward market due to confidence/coverage/dispersion"
        )

    # Additional optimistic-case clipping when disagreement is very high.
    # This preserves monotonicity while reducing fragile upside tails.
    if (
        blended.get("Base") is not None
        and blended.get("Optimistic") is not None
        and isinstance(base_method_dispersion, (int, float))
        and base_method_dispersion > 1.10
    ):
        optimism_limit = 0.55
        if sector in {"Gov Defense IT", "Gov Defense IT Hybrid"}:
            optimism_limit = 0.35
        max_optimistic = blended["Base"] * (1.0 + optimism_limit)
        if blended["Optimistic"] > max_optimistic:
            blended["Optimistic"] = max_optimistic
            quality_flags.append(
                "Optimistic scenario clipped under high dispersion "
                f"(max {optimism_limit*100:.0f}% above Base)"
            )

    # Margin semantics: preserve legacy anchored field and expose explicit dual fields.
    intrinsic_base_value = intrinsic_blended.get("Base")
    anchored_base_value = blended.get("Base")
    margin_payload = _compute_margin_outputs(intrinsic_base_value, anchored_base_value, price)
    margin_of_safety = margin_payload["margin_of_safety"]
    margin_of_safety_basis = margin_payload["margin_of_safety_basis"]
    intrinsic_margin_of_safety = margin_payload["intrinsic_margin_of_safety"]
    anchored_margin_of_safety = margin_payload["anchored_margin_of_safety"]
    if margin_of_safety is not None and abs(margin_of_safety) > 500:
            quality_flags.append("Extreme margin outlier (>500%); verify ticker mapping and financial currency")

    lynch_metrics = calculate_lynch_peg_metrics(
        valuation_info=valuation_info,
        historical_eps_growth=historical_eps_growth,
        historical_fcf_growth=historical_fcf_growth,
        assumptions_override=assumptions_override,
    )
    lynch_signal = describe_lynch_signal(lynch_metrics.get("peg_ratio"))
    if lynch_metrics.get("peg_ratio") is None:
        quality_flags.append("PEG/Lynch signal unavailable (missing positive growth or usable P/E)")
    elif lynch_metrics.get("growth_source") == "custom_lynch_growth":
        quality_flags.append(
            f"Custom Lynch growth applied: {lynch_metrics.get('growth_rate_pct', 0):.1f}%"
        )

    earnings_quality = lynch_metrics.get("earnings_quality") or {}
    if earnings_quality.get("ratio_indicates_normalization_risk"):
        ratio = earnings_quality.get("forward_to_trailing_eps_ratio")
        ratio_text = f"{ratio:.2f}x" if isinstance(ratio, (int, float)) else "n/a"
        reason = earnings_quality.get("reason") or "forward/trailing EPS divergence"
        quality_flags.append(
            f"⚠️ Earnings normalization risk: {reason} (forward/trailing EPS={ratio_text})"
        )
        confidence_score = max(0.0, min(1.0, confidence_score * 0.92))
        confidence_description = _describe_confidence_level(confidence_score)

    base_assumptions = {
        "valuation_date": datetime.now().strftime("%Y-%m-%d"),
        "ticker": ticker,
        "sector": sector,
        "raw_sector": raw_sector,
        "industry": info.get("industry"),
        "price": price,
        "currency": currency,
        "financial_currency": financial_currency,
        "fx_financial_to_quote": fx_fin_to_quote,
        "discount_rate": discount_rate,
        "profit_margin": valuation_info.get("profitMargins"),
        "operating_margin": valuation_info.get("operatingMargins"),
        "gross_margin": valuation_info.get("grossMargins"),
        "revenue_growth": valuation_info.get("revenueGrowth"),
        "earnings_growth": valuation_info.get("earningsGrowth"),
        "historical_fcf_growth": historical_fcf_growth,
        "historical_eps_growth": historical_eps_growth,
        "raw_fcf_per_share": raw_fcf_per_share,
        "fcf_provenance": fcf_provenance,
        "normalized_fcf_details": normalized_fcf_details,
        "normalized_growth_diagnostics": growth_diag,
        "peg_ratio": lynch_metrics.get("peg_ratio"),
        "lynch_ratio": lynch_metrics.get("lynch_ratio"),
        "peg_growth_rate_pct": lynch_metrics.get("growth_rate_pct"),
        "peg_growth_source": lynch_metrics.get("growth_source"),
        "peg_pe_ratio_used": lynch_metrics.get("pe_ratio_used"),
        "peg_pe_source": lynch_metrics.get("pe_source"),
        "lynch_signal": lynch_signal,
        "earnings_quality": earnings_quality,
        "policy_metadata": policy_metadata,
        "policy_confidence": policy_confidence,
        "policy_fallback_sector": policy_fallback_sector,
        "assumption_deltas": assumption_deltas,
        "raw_assumptions": raw_assumptions,
        "normalized_assumptions": normalized_assumptions,
        "net_debt_source": net_debt_data.get("source"),
        "net_debt_freshness": net_debt_data.get("freshness"),
        "method_validity_flags": method_validity_flags,
        "method_weights": weights,
        "methods_used": methods_available,
        "distressed_ev_only_no_fair_value": distressed_ev_only_no_fair_value,
        "confidence_score": confidence_score,
        "model_confidence_score": model_confidence_score,
        "confidence_description": confidence_description,
        "margin_of_safety_basis": margin_of_safety_basis,
        "intrinsic_margin_of_safety": intrinsic_margin_of_safety,
        "anchored_margin_of_safety": anchored_margin_of_safety,
        "dcf": dcf_assumptions,
        "pe": pe_assumptions,
        "ev_ebitda": ev_assumptions,
    }
    
    return {
        "ticker": ticker,
        "name": name,
        "sector": sector,
        "raw_sector": raw_sector,
        "industry": info.get("industry"),
        "sector_override_reason": sector_override_reason,
        "currency": currency,
        "financial_currency": financial_currency,
        "fx_financial_to_quote": fx_fin_to_quote,
        "price": price,
        "price_change_1d": price_change_1d,
        "price_change_1y": price_change_1y,
        "discount_rate": discount_rate,
        "dcf": dcf_vals,
        "pe": pe_vals,
        "ev": ev_vals,
        "pb": pb_vals,
        "roe": roe_vals,
        "blended": blended,
        "blended_intrinsic": intrinsic_blended,
        "fcf_per_share": fcf_per_share,
        "historical_fcf_growth": historical_fcf_growth,
        "historical_eps_growth": historical_eps_growth,
        "peg_ratio": lynch_metrics.get("peg_ratio"),
        "lynch_ratio": lynch_metrics.get("lynch_ratio"),
        "peg_growth_rate_pct": lynch_metrics.get("growth_rate_pct"),
        "peg_growth_source": lynch_metrics.get("growth_source"),
        "peg_pe_ratio_used": lynch_metrics.get("pe_ratio_used"),
        "peg_pe_source": lynch_metrics.get("pe_source"),
        "lynch_signal": lynch_signal,
        "margin_of_safety": margin_of_safety,
        "margin_of_safety_basis": margin_of_safety_basis,
        "intrinsic_margin_of_safety": intrinsic_margin_of_safety,
        "anchored_margin_of_safety": anchored_margin_of_safety,
        "quality_flags": quality_flags,
        # ===== CONFIDENCE SCORING (NEW) =====
        "confidence_score": confidence_score,
        "model_confidence_score": model_confidence_score,
        "confidence_description": confidence_description,
        "methods_used": methods_available,
        "distressed_ev_only_no_fair_value": distressed_ev_only_no_fair_value,
        # ===== AUTO-CALIBRATION INFO =====
        "growth_profile": calibration["profile"],
        "growth_profile_description": calibration["description"],
        "policy_metadata": policy_metadata,
        "policy_confidence": policy_confidence,
        "policy_fallback_sector": policy_fallback_sector,
        "raw_assumptions": raw_assumptions,
        "normalized_assumptions": normalized_assumptions,
        "assumption_deltas": assumption_deltas,
        "method_validity_flags": method_validity_flags,
        "net_debt": net_debt_data,
        "net_debt_to_ebitda": net_debt_to_ebitda,
        "fundamental_confirmation": fundamental_confirmation,
        "method_weights": weights,
        "base_assumptions": base_assumptions,
        "error": None,
    }


def compute_historical_valuations(ticker, months_back=12, assumptions_override=None):
    """
    Backfill historical fair values using quarterly financial data.
    This lets you see how fair value has evolved over time immediately,
    without waiting to collect multiple valuation snapshots.
    
    Args:
        ticker: Stock ticker
        months_back: How far back to calculate (12-60 months typical)
    
    Returns:
        dict with historical fair value data points keyed by date
    """
    
    stock, info, metadata_error = fetch_yahoo_metadata(ticker)
    if info is None:
        return {
            "error": (
                f"Yahoo Finance metadata is temporarily unavailable for {ticker}: {metadata_error}. "
                "Please retry shortly."
            )
        }
    name = info.get("shortName", ticker)
    currency = info.get("currency", "USD")
    financial_currency = info.get("financialCurrency", currency)

    # Normalize statement-derived values into quote currency for comparability
    # against market price (important for ADRs like TSM: TWD financials, USD quote).
    fx_fin_to_quote = 1.0
    if financial_currency and currency and financial_currency != currency:
        fx_candidate = get_fx_rate(financial_currency, currency)
        if fx_candidate and fx_candidate > 0:
            fx_fin_to_quote = fx_candidate
            logger.info(
                f"[{ticker}] Historical valuation FX conversion active: "
                f"{financial_currency}->{currency} (fx={fx_fin_to_quote:.6f})"
            )
        else:
            logger.warning(
                f"[{ticker}] Historical valuation currency mismatch "
                f"({financial_currency}->{currency}) and no FX rate found"
            )
            fx_fin_to_quote = None
    
    # Initialize tracking variables
    historical_valuations = {}
    quarters_outside_lookback = 0
    quarters_no_financials = 0
    quarters_failed = 0
    lag_skipped_points = 0
    lag_source_counts = {
        "filing_date": 0,
        "earnings_release": 0,
        "fallback_45d": 0,
        "unknown": 0,
    }
    
    # Dynamic trailing-period aggregation for quarterly series.
    # Detect reporting cadence from statement dates so this works automatically
    # across tickers with quarterly/semi-annual/other rhythms.
    def detect_target_periods(fin_df):
        try:
            if fin_df is None or len(fin_df.columns) < 2:
                return 4
            dates = sorted([pd.Timestamp(c) for c in fin_df.columns], reverse=True)
            deltas = []
            for i in range(len(dates) - 1):
                gap = abs((dates[i] - dates[i + 1]).days)
                if 30 <= gap <= 400:
                    deltas.append(gap)
            if not deltas:
                return 4
            median_gap_days = float(np.median(deltas))
            inferred = int(round(365.25 / median_gap_days)) if median_gap_days > 0 else 4
            return max(1, min(inferred, 8))
        except Exception:
            return 4

    target_periods = 4

    def trailing_quarter_sum(fin_df, row_name, asof_date):
        if fin_df is None or row_name not in fin_df.index:
            return None, 0
        try:
            series = fin_df.loc[row_name]
            if isinstance(series, pd.DataFrame):
                series = series.iloc[0]
            series = series.dropna()
            if series.empty:
                return None, 0

            eligible = series[series.index <= asof_date]
            if eligible.empty:
                return None, 0

            recent = eligible.sort_index(ascending=False).head(target_periods)
            periods_used = len(recent)
            if periods_used == 0:
                return None, 0

            total = float(recent.sum())
            if periods_used < target_periods:
                total = total * (target_periods / periods_used)

            return total, periods_used
        except Exception:
            return None, 0

    # Fetch historical prices for the period
    end_date = datetime.now()
    start_date = end_date - timedelta(days=months_back * 30)
    logger.info(f"[{ticker}] Computing historical valuations for {months_back} months")
    
    try:
        prices_hist = stock.history(start=start_date, end=end_date)
        # Strip timezone to avoid naive/aware datetime mismatch
        if prices_hist.index.tz is not None:
            prices_hist.index = prices_hist.index.tz_localize(None)
        logger.debug(f"[{ticker}] Fetched {len(prices_hist)} historical price data points")
    except Exception as e:
        logger.error(f"[{ticker}] Cannot fetch historical prices: {e}")
        return {"error": f"Cannot fetch historical prices for {ticker}"}
    
    if len(prices_hist) == 0:
        logger.error(f"[{ticker}] No historical price data available")
        return {"error": f"No historical price data available for {ticker}"}
    
    # Fetch quarterly financial data
    logger.debug(f"[{ticker}] Fetching quarterly financial data...")
    quarterly_info = None
    quarterly_balance = None
    quarterly_cashflow = None
    
    try:
        quarterly_info = stock.quarterly_financials
        quarterly_balance = stock.quarterly_balance_sheet
        quarterly_cashflow = stock.quarterly_cashflow
        target_periods = detect_target_periods(quarterly_info)
        logger.debug(f"[{ticker}] Quarterly financials loaded: {len(quarterly_info.columns) if quarterly_info is not None else 0} records")
    except Exception as e:
        logger.debug(f"[{ticker}] Quarterly data failed: {e}")
    
    # Always fetch annual data for older periods (to extend lookback beyond 6 quarters)
    logger.debug(f"[{ticker}] Fetching annual financial data...")
    annual_info = None
    annual_balance = None
    annual_cashflow = None
    
    try:
        annual_info = stock.financials
        annual_balance = stock.balance_sheet
        annual_cashflow = stock.cashflow
        logger.debug(f"[{ticker}] Annual financials loaded: {len(annual_info.columns) if annual_info is not None else 0} records")
    except Exception as e:
        logger.debug(f"[{ticker}] Annual data failed: {e}")
    
    # Combine: use quarterly for recent periods, annual for older
    # This gives us up to 6 recent quarters + up to 20 years of annual data
    combined_dates = []
    financials_map = {}  # Map date -> (data, is_quarterly)
    
    if quarterly_info is not None and len(quarterly_info.columns) > 0:
        for col in quarterly_info.columns:
            combined_dates.append(col)
            financials_map[col] = (quarterly_info, quarterly_balance, quarterly_cashflow, True)
    
    if annual_info is not None and len(annual_info.columns) > 0:
        for col in annual_info.columns:
            if col not in financials_map:  # Don't duplicate if already in quarterly
                combined_dates.append(col)
                financials_map[col] = (annual_info, annual_balance, annual_cashflow, False)
    
    if not combined_dates:
        logger.error(f"[{ticker}] No financial data available (quarterly or annual)")
        return {"error": f"No financial data available for {ticker}. This ticker may not be supported."}
    
    logger.info(f"[{ticker}] Total financial data points available: {len(combined_dates)} (Quarterly: {len(quarterly_info.columns) if quarterly_info is not None else 0}, Annual: {len(annual_info.columns) if annual_info is not None else 0})")
    filing_dates, release_dates = _extract_reporting_dates(stock)
    
    # Precompute sector context and growth diagnostics once; per-period valuation
    # then reuses the same enhanced engine functions as current valuation.
    raw_sector = info.get("sector", "Unknown")
    sector, _ = get_valuation_sector_name(info)
    sector_config = get_sector_config(sector)

    policy_confidence = float(sector_config.policy_confidence or sector_config.base_confidence)
    policy_fallback_sector = sector_config.fallback_parent_sector
    if policy_fallback_sector and policy_fallback_sector in SECTOR_CONFIGS:
        parent_conf = SECTOR_CONFIGS[policy_fallback_sector].base_confidence
        policy_confidence = 0.8 * policy_confidence + 0.2 * parent_conf

    historical_fcf_growth = None
    historical_eps_growth = None
    try:
        cashflow = stock.cashflow
        if cashflow is not None:
            if ("Operating Cash Flow" in cashflow.index
                    and "Capital Expenditure" in cashflow.index):
                ocf_hist   = cashflow.loc["Operating Cash Flow"].dropna()
                capex_hist = cashflow.loc["Capital Expenditure"].dropna()
                fcf_series = (ocf_hist + capex_hist).dropna()  # CapEx is negative
                historical_fcf_growth = calculate_historical_growth(fcf_series, years=5)
            elif "Free Cash Flow" in cashflow.index:
                fcf_series = cashflow.loc["Free Cash Flow"].dropna()
                historical_fcf_growth = calculate_historical_growth(fcf_series, years=5)
    except Exception:
        pass
    try:
        hist_fin = stock.quarterly_financials
        shares = info.get("sharesOutstanding")
        if hist_fin is not None and "Net Income" in hist_fin.index and shares:
            ni_series = hist_fin.loc["Net Income"].dropna()
            if len(ni_series) > 0:
                eps_series = ni_series / shares
                historical_eps_growth = calculate_historical_growth(eps_series, years=5)
    except Exception:
        pass

    logger.info(f"[{ticker}] Historical engine unified with compute_valuation_enhanced policy stack")
    quarters_to_process = min(len(combined_dates), int(months_back / 3) + 2)
    logger.info(f"[{ticker}] Total data points available: {len(combined_dates)}")
    logger.info(f"[{ticker}] Requested lookback: {months_back} months")
    logger.info(f"[{ticker}] Will process: {quarters_to_process} periods")
    
    # Sort dates newest to oldest
    sorted_dates = sorted(combined_dates, reverse=True)
    
    for i, quarter_date in enumerate(sorted_dates[:quarters_to_process]):
        try:
            fin_info, fin_balance, fin_cashflow, is_quarterly = financials_map[quarter_date]
            logger.debug(f"[{ticker}] [P{i}] Processing: {quarter_date.strftime('%Y-%m-%d')} ({'Q' if is_quarterly else 'A'})")

            availability_date, lag_source = _resolve_financial_availability_date(
                quarter_date,
                filing_dates,
                release_dates,
                fallback_days=45,
            )
            lag_source = lag_source if lag_source in lag_source_counts else "unknown"
            lag_source_counts[lag_source] += 1
            if availability_date is None:
                lag_skipped_points += 1
                continue

            eligible_prices = prices_hist.loc[prices_hist.index >= availability_date]
            if eligible_prices.empty:
                logger.debug(f"[{ticker}] [Q{i}] No eligible price data after availability date {availability_date}")
                lag_skipped_points += 1
                quarters_failed += 1
                continue

            price_date = eligible_prices.index[0]
            price = eligible_prices.iloc[0]["Close"]
            monthly_candle = None
            required_ohlc = {"Open", "High", "Low", "Close"}
            if required_ohlc.issubset(set(prices_hist.columns)):
                month_start = pd.Timestamp(price_date).to_period("M").start_time
                month_end = pd.Timestamp(price_date).to_period("M").end_time
                month_slice = prices_hist.loc[(prices_hist.index >= month_start) & (prices_hist.index <= month_end)]
                if not month_slice.empty:
                    open_val = pd.to_numeric(month_slice["Open"].iloc[0], errors="coerce")
                    high_val = pd.to_numeric(month_slice["High"].max(), errors="coerce")
                    low_val = pd.to_numeric(month_slice["Low"].min(), errors="coerce")
                    close_val = pd.to_numeric(month_slice["Close"].iloc[-1], errors="coerce")
                    if all(pd.notna(v) for v in [open_val, high_val, low_val, close_val]):
                        monthly_candle = {
                            "open": float(open_val),
                            "high": float(high_val),
                            "low": float(low_val),
                            "close": float(close_val),
                            "average": float((open_val + high_val + low_val + close_val) / 4.0),
                        }
            days_back = (end_date - price_date.to_pydatetime()).days
            max_days_back = months_back * 30
            logger.debug(
                f"[{ticker}] [Q{i}] Price date: {price_date.strftime('%Y-%m-%d')} "
                f"({days_back} days ago, max: {max_days_back}, lag_source={lag_source}, "
                f"available_from={availability_date.strftime('%Y-%m-%d')})"
            )
            
            # Only process if within our lookback period
            if days_back > max_days_back:
                logger.debug(f"[{ticker}] [Q{i}] SKIPPED - outside lookback window ({days_back} > {max_days_back})")
                quarters_outside_lookback += 1
                continue
            
            # Calculate FCF for this quarter.
            # Priority mirrors calculate_fcf_per_share() for current valuations:
            #   1. GAAP OCF − CapEx (primary — immune to non-cash add-back inflation)
            #   2. Pre-computed 'Free Cash Flow' row only when OCF/CapEx unavailable
            fcf_per_share_q = None
            shares = info.get("sharesOutstanding")

            # ── Primary: GAAP OCF − CapEx ─────────────────────────────────
            if shares and "Operating Cash Flow" in fin_cashflow.index and "Capital Expenditure" in fin_cashflow.index:
                try:
                    if is_quarterly:
                        ocf, ocf_n = trailing_quarter_sum(fin_cashflow, "Operating Cash Flow", quarter_date)
                        capex, capex_n = trailing_quarter_sum(fin_cashflow, "Capital Expenditure", quarter_date)
                        if (
                            annual_cashflow is not None
                            and quarter_date in annual_cashflow.columns
                            and (
                                (ocf_n is not None and ocf_n < target_periods)
                                or (capex_n is not None and capex_n < target_periods)
                            )
                        ):
                            if "Operating Cash Flow" in annual_cashflow.index:
                                annual_ocf = annual_cashflow.loc["Operating Cash Flow", quarter_date]
                                if pd.notna(annual_ocf):
                                    ocf = annual_ocf
                            if "Capital Expenditure" in annual_cashflow.index:
                                annual_capex = annual_cashflow.loc["Capital Expenditure", quarter_date]
                                if pd.notna(annual_capex):
                                    capex = annual_capex
                    else:
                        ocf   = fin_cashflow.loc["Operating Cash Flow", quarter_date]
                        capex = fin_cashflow.loc["Capital Expenditure", quarter_date]

                    if pd.notna(ocf) and pd.notna(capex):
                        fcf_annualized = ocf - abs(capex)  # CapEx stored negative
                        if fcf_annualized > 0:
                            fcf_per_share_q = fcf_annualized / shares
                except:
                    pass

            # ── Fallback: pre-computed 'Free Cash Flow' row ────────────────────
            if not fcf_per_share_q and shares and "Free Cash Flow" in fin_cashflow.index:
                try:
                    fcf_periods_used = None
                    if is_quarterly:
                        fcf_annualized, fcf_periods_used = trailing_quarter_sum(fin_cashflow, "Free Cash Flow", quarter_date)
                        if (
                            fcf_periods_used is not None
                            and fcf_periods_used < target_periods
                            and annual_cashflow is not None
                            and "Free Cash Flow" in annual_cashflow.index
                            and quarter_date in annual_cashflow.columns
                        ):
                            annual_fcf = annual_cashflow.loc["Free Cash Flow", quarter_date]
                            if pd.notna(annual_fcf):
                                fcf_annualized = annual_fcf
                    else:
                        fcf_annualized = fin_cashflow.loc["Free Cash Flow", quarter_date]

                    if pd.notna(fcf_annualized) and fcf_annualized > 0:
                        fcf_per_share_q = fcf_annualized / shares
                except:
                    pass
            
            # Get EPS for this quarter (quarterly -> trailing-quarter aggregation for P/E model)
            eps_q = None
            try:
                if "Net Income" in fin_info.index:
                    ni_periods_used = None
                    if is_quarterly:
                        ni_annualized, ni_periods_used = trailing_quarter_sum(fin_info, "Net Income", quarter_date)
                        if (
                            ni_periods_used is not None
                            and ni_periods_used < target_periods
                            and annual_info is not None
                            and "Net Income" in annual_info.index
                            and quarter_date in annual_info.columns
                        ):
                            annual_ni = annual_info.loc["Net Income", quarter_date]
                            if pd.notna(annual_ni):
                                ni_annualized = annual_ni
                    else:
                        ni_annualized = fin_info.loc["Net Income", quarter_date]

                    if pd.notna(ni_annualized) and shares:
                        eps_q = ni_annualized / shares
            except:
                pass
            
            # Get EBITDA for this quarter (quarterly -> trailing-quarter aggregation for EV/EBITDA model)
            ebitda_q = None
            try:
                if "EBITDA" in fin_info.index:
                    ebitda_periods_used = None
                    if is_quarterly:
                        ebitda_q, ebitda_periods_used = trailing_quarter_sum(fin_info, "EBITDA", quarter_date)
                        if (
                            ebitda_periods_used is not None
                            and ebitda_periods_used < target_periods
                            and annual_info is not None
                            and "EBITDA" in annual_info.index
                            and quarter_date in annual_info.columns
                        ):
                            annual_ebitda = annual_info.loc["EBITDA", quarter_date]
                            if pd.notna(annual_ebitda):
                                ebitda_q = annual_ebitda
                    else:
                        ebitda_q = fin_info.loc["EBITDA", quarter_date]
            except:
                pass
            
            # Only value if we have meaningful data
            if not (fcf_per_share_q or eps_q or ebitda_q):
                continue

            # FCF credibility cap — identical to the guard in compute_valuation_enhanced().
            # FCF > EBITDA/share is structurally impossible; it signals non-cash add-backs
            # (merger amortisation, restructuring).  Cap at EBITDA × 0.65 / shares so
            # historical and current valuations are computed on the same methodology.
            if fcf_per_share_q is not None and fcf_per_share_q > 0 and shares:
                _ebitda_hist = float(ebitda_q or info.get("ebitda") or 0)
                if _ebitda_hist > 0:
                    _ebitda_ps_hist = _ebitda_hist / shares
                    if fcf_per_share_q > _ebitda_ps_hist:
                        fcf_per_share_q = _ebitda_ps_hist * 0.65
            
            # Calculate WACC for this period
            discount_rate = calculate_enhanced_wacc(stock, info)

            # Build period-specific valuation payload and run the same enhanced
            # method stack as current valuation.
            valuation_info = dict(info)
            valuation_info["currentPrice"] = float(price)
            valuation_info["trailingEps"] = float(eps_q) if eps_q and eps_q > 0 else info.get("trailingEps")
            valuation_info["forwardEps"] = None
            valuation_info["forwardPE"] = None
            valuation_info["trailingPE"] = (float(price) / float(eps_q)) if (eps_q and eps_q > 0) else info.get("trailingPE")
            valuation_info["ebitda"] = float(ebitda_q) if ebitda_q and ebitda_q > 0 else info.get("ebitda")

            dcf_vals = null_scenarios()
            growth_diag = {}
            if fcf_per_share_q and fcf_per_share_q > 0:
                dcf_growth_input, growth_diag = compute_normalized_dcf_growth(
                    valuation_info,
                    sector_config,
                    historical_fcf_growth,
                    historical_eps_growth,
                    sector,
                )
                dcf_vals = dcf_scenarios_enhanced(
                    stock,
                    fcf_per_share_q,
                    discount_rate,
                    dcf_growth_input,
                    stock_info=valuation_info,
                    sector=sector,
                    assumptions_override=assumptions_override,
                )

            pe_vals = pe_scenarios_enhanced(
                stock,
                valuation_info,
                discount_rate,
                valuation_info.get("forwardPE"),
                sector=sector,
                sector_config=sector_config,
                assumptions_override=assumptions_override,
            )

            # Prefer historical balance-sheet values for period EV net debt.
            net_debt_data = {
                "total_debt": float(info.get("totalDebt", 0) or 0),
                "total_cash": float(info.get("totalCash", 0) or 0),
                "net_debt": float(info.get("totalDebt", 0) or 0) - float(info.get("totalCash", 0) or 0),
                "source": "info_fallback",
                "freshness": "unknown",
            }
            try:
                if fin_balance is not None and quarter_date in fin_balance.columns:
                    debt_candidates = [
                        "Total Debt",
                        "Current Debt And Capital Lease Obligation",
                        "Current Debt",
                        "Long Term Debt And Capital Lease Obligation",
                        "Long Term Debt",
                    ]
                    cash_candidates = [
                        "Cash Cash Equivalents And Short Term Investments",
                        "Cash And Cash Equivalents",
                        "Cash",
                    ]
                    total_debt_bs = None
                    for label in debt_candidates:
                        if label in fin_balance.index:
                            v = fin_balance.loc[label, quarter_date]
                            if pd.notna(v):
                                total_debt_bs = float(v)
                                break
                    total_cash_bs = None
                    for label in cash_candidates:
                        if label in fin_balance.index:
                            v = fin_balance.loc[label, quarter_date]
                            if pd.notna(v):
                                total_cash_bs = float(v)
                                break
                    if total_debt_bs is not None or total_cash_bs is not None:
                        total_debt = max(total_debt_bs or 0.0, 0.0)
                        total_cash = max(total_cash_bs or 0.0, 0.0)
                        net_debt_data = {
                            "total_debt": total_debt,
                            "total_cash": total_cash,
                            "net_debt": total_debt - total_cash,
                            "source": "historical_balance_sheet",
                            "freshness": quarter_date.strftime("%Y-%m-%d"),
                        }
            except Exception:
                pass

            ev_vals = ev_ebitda_scenarios_enhanced(
                stock,
                valuation_info,
                net_debt_data=net_debt_data,
                sector_config=sector_config,
                quality_flags=[],
                assumption_deltas={},
            )

            dcf_assumptions = dcf_vals.pop("_assumptions", None) if isinstance(dcf_vals, dict) else None
            pe_vals.pop("_assumptions", None) if isinstance(pe_vals, dict) else None
            pe_vals.pop("_pe_guardrail", None) if isinstance(pe_vals, dict) else None
            ev_vals.pop("_assumptions", None) if isinstance(ev_vals, dict) else None

            # Convert statement-based valuation outputs into quote currency.
            # Historical EPS is statement-currency based, so include P/E conversion.
            if fx_fin_to_quote and fx_fin_to_quote > 0 and financial_currency != currency:
                dcf_vals = {k: (v * fx_fin_to_quote if v is not None else None) for k, v in dcf_vals.items()}
                pe_vals = {k: (v * fx_fin_to_quote if v is not None else None) for k, v in pe_vals.items()}
                ev_vals = {k: (v * fx_fin_to_quote if v is not None else None) for k, v in ev_vals.items()}

            # Apply same outlier gating as current valuation.
            dcf_vals = exclude_outlier_method(
                "DCF", dcf_vals, float(price), [], min_ratio=0.05, max_ratio=sector_config.outlier_max_ratio
            )
            pe_vals = exclude_outlier_method(
                "P/E", pe_vals, float(price), [], max_ratio=sector_config.outlier_max_ratio
            )
            ev_vals = exclude_outlier_method(
                "EV/EBITDA", ev_vals, float(price), [], max_ratio=sector_config.outlier_max_ratio
            )

            growth_profile = GrowthProfile(
                historical_growth=historical_eps_growth,
                current_pe=valuation_info.get("trailingPE"),
                sector=sector,
                forward_pe=valuation_info.get("forwardPE"),
            )
            calibration = growth_profile.get_calibrated_params()
            profile_weights = {
                "DCF": calibration["method_weights"].get("DCF", 0.33),
                "P/E": calibration["method_weights"].get("PE", 0.33),
                "EV/EBITDA": calibration["method_weights"].get("EV_EBITDA", 0.33),
            }
            sector_weights = get_sector_method_weights(sector_config.name)
            is_cyclical_or_regulated = (
                sector_config.special_rules.get("use_normalized_commodity")
                or sector_config.special_rules.get("use_normalized_capex")
                or sector_config.special_rules.get("regulated")
            )
            if is_cyclical_or_regulated:
                weights = {
                    "DCF": sector_weights.get("DCF", 0.0),
                    "P/E": sector_weights.get("P/E", 0.0),
                    "EV/EBITDA": sector_weights.get("EV/EBITDA", 0.0),
                }
            else:
                weights = {
                    "DCF": sector_weights.get("DCF", 0.0) * 0.75 + profile_weights.get("DCF", 0.0) * 0.25 if sector_weights.get("DCF", 0.0) > 0 else 0.0,
                    "P/E": sector_weights.get("P/E", 0.0) * 0.75 + profile_weights.get("P/E", 0.0) * 0.25 if sector_weights.get("P/E", 0.0) > 0 else 0.0,
                    "EV/EBITDA": sector_weights.get("EV/EBITDA", 0.0) * 0.75 + profile_weights.get("EV/EBITDA", 0.0) * 0.25 if sector_weights.get("EV/EBITDA", 0.0) > 0 else 0.0,
                }
            total_w = sum(weights.values())
            if total_w > 0:
                weights = {k: v / total_w for k, v in weights.items()}

            mega_cap_software_mode = bool(dcf_assumptions and dcf_assumptions.get("mega_cap_software_mode"))
            if mega_cap_software_mode and weights["DCF"] > 0.35:
                shift = min(0.08, weights["DCF"] - 0.35)
                weights["DCF"] -= shift
                weights["P/E"] += shift * 0.65
                weights["EV/EBITDA"] += shift * 0.35
                total_w = sum(weights.values())
                if total_w > 0:
                    weights = {k: v / total_w for k, v in weights.items()}

            blended = {}
            for scenario in ["Pessimistic", "Base", "Optimistic"]:
                dcf_val = dcf_vals.get(scenario)
                pe_val = pe_vals.get(scenario)
                ev_val = ev_vals.get(scenario)
                w_dcf = weights["DCF"] if dcf_val is not None else 0
                w_pe = weights["P/E"] if pe_val is not None else 0
                w_ev = weights["EV/EBITDA"] if ev_val is not None else 0
                total = w_dcf + w_pe + w_ev
                if total > 0:
                    blended[scenario] = (
                        (dcf_val or 0) * w_dcf / total
                        + (pe_val or 0) * w_pe / total
                        + (ev_val or 0) * w_ev / total
                    )
                else:
                    blended[scenario] = None

            base_val = blended.get("Base")
            if base_val is None:
                logger.debug(f"[{ticker}] [Q{i}] SKIPPED: No valid blended valuation")
                quarters_no_financials += 1
                continue

            # Same realism-anchor policy as current valuation.
            methods_available = []
            if dcf_vals.get("Base") is not None:
                methods_available.append("DCF")
            if pe_vals.get("Base") is not None:
                methods_available.append("P/E")
            if ev_vals.get("Base") is not None:
                methods_available.append("EV/EBITDA")

            confidence_score, _ = calculate_confidence_score(
                len(methods_available),
                sector_config.min_required_methods,
                policy_confidence,
                0,
            )
            anchor_strength = 0.0
            method_shortfall = max(0, sector_config.min_required_methods - len(methods_available))
            if method_shortfall > 0:
                anchor_strength += 0.10 * method_shortfall
            if confidence_score < 0.50:
                anchor_strength += (0.50 - confidence_score) * 0.35
            base_method_values = [
                v for v in [dcf_vals.get("Base"), pe_vals.get("Base"), ev_vals.get("Base")] if v is not None
            ]
            if len(base_method_values) >= 2:
                median_val = float(np.median(base_method_values))
                if abs(median_val) > 1e-9:
                    dispersion = (max(base_method_values) - min(base_method_values)) / abs(median_val)
                    if dispersion > 1.00:
                        anchor_strength += min(0.20, (dispersion - 1.00) * 0.18)
            anchor_strength = min(0.35, max(0.0, anchor_strength))
            if anchor_strength > 0:
                for scenario in ["Pessimistic", "Base", "Optimistic"]:
                    if blended.get(scenario) is not None:
                        blended[scenario] = blended[scenario] * (1 - anchor_strength) + float(price) * anchor_strength

            # Enforce monotonicity for interpretability.
            if blended.get("Base") is not None:
                if blended.get("Pessimistic") is not None:
                    blended["Pessimistic"] = min(blended["Pessimistic"], blended["Base"])
                if blended.get("Optimistic") is not None:
                    blended["Optimistic"] = max(blended["Optimistic"], blended["Base"])

            lynch_metrics = calculate_lynch_peg_metrics(
                valuation_info=valuation_info,
                historical_eps_growth=historical_eps_growth,
                historical_fcf_growth=historical_fcf_growth,
                assumptions_override=assumptions_override,
            )
            lynch_signal = describe_lynch_signal(lynch_metrics.get("peg_ratio"))

            base_blended = blended.get("Base")
            pessimistic_blended = blended.get("Pessimistic")
            optimistic_blended = blended.get("Optimistic")
            fair_value_base = float(base_blended) if base_blended is not None else None
            fair_value_low = float(pessimistic_blended) if pessimistic_blended is not None else fair_value_base
            fair_value_high = float(optimistic_blended) if optimistic_blended is not None else fair_value_base

            date_str = price_date.strftime("%Y-%m-%d")
            historical_valuations[date_str] = {
                "price": float(price),
                "fair_value_base": fair_value_base,
                "fair_value_low": fair_value_low,
                "fair_value_high": fair_value_high,
                "discount_rate": discount_rate,
                "peg_ratio": lynch_metrics.get("peg_ratio"),
                "lynch_ratio": lynch_metrics.get("lynch_ratio"),
                "peg_growth_rate_pct": lynch_metrics.get("growth_rate_pct"),
                "peg_growth_source": lynch_metrics.get("growth_source"),
                "peg_pe_ratio_used": lynch_metrics.get("pe_ratio_used"),
                "peg_pe_source": lynch_metrics.get("pe_source"),
                "lynch_signal": lynch_signal,
                "monthly_candle": monthly_candle,
            }
            logger.debug(f"[{ticker}] [Q{i}] SUCCESS: Added valuation {date_str} with fair value {blended.get('Base'):.2f}")
        
        except Exception as e:
            logger.debug(f"[{ticker}] [Q{i}] ERROR: {type(e).__name__}: {e}")
            quarters_failed += 1
            continue
    
    # Summary log
    logger.info(f"[{ticker}] ===== HISTORICAL VALUATION SUMMARY =====")
    logger.info(f"[{ticker}] Quarters processed: {quarters_to_process}")
    logger.info(f"[{ticker}] Quarters successfully valued: {len(historical_valuations)}")
    logger.info(f"[{ticker}] Quarters outside lookback window: {quarters_outside_lookback}")
    logger.info(f"[{ticker}] Quarters without financial data: {quarters_no_financials}")
    logger.info(f"[{ticker}] Quarters with calculation errors: {quarters_failed}")
    logger.info(f"[{ticker}] Points skipped by lag policy: {lag_skipped_points}")
    
    logger.info(f"[{ticker}] Successfully calculated {len(historical_valuations)} historical valuations")
    if not historical_valuations:
        logger.error(f"[{ticker}] Failed to calculate any valuations - no valid periods found")
        return {
            "error": f"No historical valuation data could be calculated for {ticker}. "
                    f"This ticker may have limited financial data available. "
                    f"Try using 'Current Valuation' mode instead, or try a different ticker."
        }
    
    return {
        "ticker": ticker,
        "name": name,
        "currency": currency,
        "sector": sector,
        "raw_sector": raw_sector,
        "historical_valuations": historical_valuations,
        "lag_policy": {
            "fallback_days": 45,
            "sources_used": lag_source_counts,
            "skipped_points": lag_skipped_points,
        },
        "error": None,
    }


def refresh_valuation_cache(
    ticker: str,
    months_back: int = 12,
    assumptions_override: dict | None = None,
) -> dict:
    """
    Refresh cached historical valuations for a ticker using the current (fixed) logic.
    This clears old cached data and re-computes everything from scratch.
    
    Args:
        ticker: Stock ticker
        months_back: How far back to recompute (12 months default)
    
    Returns:
        dict with success/error status and number of entries refreshed
    """
    cache = ValuationCache()
    
    try:
        # Step 1: Compute historical valuations with fixed logic
        hist_result = compute_historical_valuations(
            ticker,
            months_back=months_back,
            assumptions_override=assumptions_override,
        )
        
        if hist_result.get("error"):
            return {"error": hist_result.get("error"), "refreshed": 0}
        
        # Step 2: Clear old cache
        cache.clear_history(ticker)
        
        # Step 3: Save newly computed valuations to cache
        hist_valuations = hist_result.get("historical_valuations", {})
        for date_str, valuation_data in hist_valuations.items():
            cache.save_valuation(
                ticker,
                build_historical_cache_payload(valuation_data),
                date=date_str,
                skip_date_filter=True,
                source="historical_backfill",
            )
        
        return {
            "success": True,
            "refreshed": len(hist_valuations),
            "message": f"✅ Refreshed {len(hist_valuations)} historical valuation entries for {ticker}"
        }
    
    except Exception as e:
        return {
            "error": f"Failed to refresh cache: {str(e)}",
            "refreshed": 0
        }


if __name__ == "__main__":
    # Test
    result = compute_valuation_enhanced("ASML")
    import pprint
    pprint.pprint(result)

