"""
Sector Configuration System

Defines which valuation methods work for each sector and adaptive parameters.
This enables sector-aware valuation with confidence scoring.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


POLICY_VERSION = "v1.3.0"
POLICY_EFFECTIVE_DATE = "2026-08-12"


DEFAULT_METHOD_WEIGHTS: Dict[str, Dict[str, float]] = {
    "Technology": {"DCF": 0.45, "P/E": 0.35, "EV/EBITDA": 0.20},
    "Healthcare": {"DCF": 0.45, "P/E": 0.35, "EV/EBITDA": 0.20},
    "Healthcare IT": {"DCF": 0.35, "EV/EBITDA": 0.45, "P/E": 0.20},  # EV/EBITDA dominant; P/E down-weighted (GAAP distorted by M&A amortization)
    "Managed Care": {"P/E": 0.55, "EV/EBITDA": 0.45},
    "Pharmaceuticals": {"DCF": 0.45, "P/E": 0.35, "EV/EBITDA": 0.20},
    "Medical Devices": {"DCF": 0.40, "P/E": 0.35, "EV/EBITDA": 0.25},
    "Life Science Tools": {"DCF": 0.35, "P/E": 0.35, "EV/EBITDA": 0.30},
    "Healthcare Providers": {"EV/EBITDA": 0.60, "P/E": 0.40},
    "Pharmaceutical Distribution": {"EV/EBITDA": 0.61, "P/E": 0.39},
    "Consumer": {"DCF": 0.35, "P/E": 0.40, "EV/EBITDA": 0.25},
    "Industrials": {"DCF": 0.35, "P/E": 0.30, "EV/EBITDA": 0.35},
    "Financials": {"DCF": 0.0, "P/E": 0.50, "EV/EBITDA": 0.0, "P/B": 0.25, "ROE": 0.25},
    "Financial Services": {"DCF": 0.0, "P/E": 0.50, "EV/EBITDA": 0.0, "P/B": 0.25, "ROE": 0.25},
    "Financial Data Services": {"DCF": 0.35, "P/E": 0.35, "EV/EBITDA": 0.30, "P/B": 0.0, "ROE": 0.0},
    "Payment Networks": {"DCF": 0.45, "P/E": 0.40, "EV/EBITDA": 0.15},
    "Gov Defense IT": {"DCF": 0.20, "P/E": 0.40, "EV/EBITDA": 0.40},
    "Gov Defense IT Hybrid": {"DCF": 0.25, "P/E": 0.35, "EV/EBITDA": 0.40},
    "Energy": {"P/E": 0.76, "EV/EBITDA": 0.24},
    "Materials": {"P/E": 0.71, "EV/EBITDA": 0.29},
    "Utilities": {"P/E": 0.75, "EV/EBITDA": 0.25},
    "Telecom": {"DCF": 0.20, "P/E": 0.50, "EV/EBITDA": 0.30},            # DCF re-enabled at low weight
    "Wireless Telecom": {"DCF": 0.22, "P/E": 0.40, "EV/EBITDA": 0.38},   # growth mobile operators
    "Integrated Telecom": {"DCF": 0.20, "P/E": 0.45, "EV/EBITDA": 0.35}, # mature wireline/integrated
    "Cable/Broadband": {"DCF": 0.20, "P/E": 0.35, "EV/EBITDA": 0.45},    # EV/EBITDA dominant for cable transactions
    "Communication Services": {"DCF": 0.45, "P/E": 0.35, "EV/EBITDA": 0.20},
    "Real Estate": {},
    "Aerospace": {"P/E": 0.36, "EV/EBITDA": 0.64},
    "Construction": {"P/E": 0.50, "EV/EBITDA": 0.50},
    "Biotech": {"P/E": 1.0},                                             # DCF and EV/EBITDA disabled; P/E only
    "Specialty Chemicals": {"DCF": 0.45, "P/E": 0.30, "EV/EBITDA": 0.25},
    "Airlines": {"P/E": 0.55, "EV/EBITDA": 0.45},                       # No DCF (cyclical)
    "Software": {"DCF": 0.40, "P/E": 0.40, "EV/EBITDA": 0.20},         # Equal DCF+P/E (very clean FCF + stable earnings)
    "Digital Advertising Platform": {"DCF": 0.45, "P/E": 0.30, "EV/EBITDA": 0.25},  # Ad-tech platforms: FCF rich, but ad-cycle exposed
    "Semiconductors": {"EV/EBITDA": 0.45, "P/E": 0.30, "DCF": 0.25},   # EV/EBITDA dominant (cycle-stable)
}

DEFAULT_FIELD_ENFORCEMENT: Dict[str, str] = {
    "discount_rate": "hard",
    "pe": "soft",
    "ev_ebitda": "soft",
    "terminal_growth": "hard",
    "terminal_fcf_yield": "soft",
    "revenue_growth": "soft",
    "profit_margin": "soft",
    "earnings_growth": "soft",
}


@dataclass
class SectorConfig:
    """Configuration for valuation methods by sector."""
    
    name: str
    primary_methods: List[str]          # Methods to try first
    secondary_methods: List[str]        # Fallback methods if primary unavailable
    min_required_methods: int           # Minimum methods needed for valid valuation
    
    # Growth assumptions
    growth_mode: str                    # "use_historical", "conservative_override", "fixed_override"
    growth_min: float                   # Minimum growth to use
    growth_max: float                   # Maximum growth to use (for clamping)
    growth_override: Optional[float]    # Fixed growth if mode is "fixed_override"
    
    # Outlier thresholds (Base/Price ratio)
    outlier_min_ratio: float            # Minimum Base/Price (usually 0.2)
    outlier_max_ratio: float            # Maximum Base/Price before exclusion
    
    # Sector characteristics
    capex_impact: str                   # "low", "medium", "high", "very_high"
    fcf_reliability: str                # "reliable", "moderate", "unreliable"
    
    # Confidence
    base_confidence: float              # 0-1 scale, sector inherent reliability

    # Policy controls
    method_weights: Optional[Dict[str, float]] = None
    discount_rate_range: Tuple[float, float] = (0.07, 0.15)
    pe_range: Tuple[float, float] = (8.0, 45.0)
    ev_ebitda_range: Tuple[float, float] = (6.0, 28.0)
    terminal_growth_range: Tuple[float, float] = (0.01, 0.03)
    terminal_fcf_yield_range: Tuple[float, float] = (0.03, 0.08)
    revenue_growth_range: Tuple[float, float] = (-0.05, 0.35)
    profit_margin_range: Tuple[float, float] = (-0.10, 0.50)
    earnings_growth_range: Tuple[float, float] = (-0.25, 0.60)
    structural_growth_cap: Optional[float] = None
    # Two-tier multiple enforcement: pe_range/ev_ebitda_range = soft prior (warning only);
    # pe_sanity_range/ev_ebitda_sanity_range = hard emergency clamp (never exceeded).
    pe_sanity_range: Optional[Tuple[float, float]] = None
    ev_ebitda_sanity_range: Optional[Tuple[float, float]] = None
    field_enforcement: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_FIELD_ENFORCEMENT))
    policy_confidence: Optional[float] = None
    fallback_parent_sector: Optional[str] = None
    
    # Special rules
    special_rules: Optional[Dict[str, Any]] = None          # Override specific behaviors
    
    def __post_init__(self):
        if self.special_rules is None:
            self.special_rules = {}

        if self.policy_confidence is None:
            self.policy_confidence = self.base_confidence

        if self.method_weights is None:
            if self.special_rules.get("disable_all"):
                self.method_weights = {}
            else:
                self.method_weights = DEFAULT_METHOD_WEIGHTS.get(
                    self.name,
                    {"DCF": 0.35, "P/E": 0.40, "EV/EBITDA": 0.25},
                ).copy()

        total = sum(self.method_weights.values())
        if total > 0:
            self.method_weights = {k: v / total for k, v in self.method_weights.items()}

        merged_enforcement = dict(DEFAULT_FIELD_ENFORCEMENT)
        merged_enforcement.update(self.field_enforcement or {})
        self.field_enforcement = merged_enforcement


def get_policy_metadata() -> Dict[str, Any]:
    """Return policy-level metadata for traceability in valuation outputs."""
    return {
        "policy_version": POLICY_VERSION,
        "policy_effective_date": POLICY_EFFECTIVE_DATE,
    }


# ============================================================================
# SECTOR CONFIGURATIONS
# ============================================================================

SECTOR_CONFIGS = {
    "Technology": SectorConfig(
        name="Technology",
        primary_methods=["DCF", "P/E", "EV/EBITDA"],
        secondary_methods=["P/E", "EV/EBITDA"],
        min_required_methods=2,
        
        growth_mode="use_historical",
        growth_min=0.05,
        growth_max=0.25,
        growth_override=None,
        
        outlier_min_ratio=0.2,
        outlier_max_ratio=8.0,
        
        capex_impact="medium",
        fcf_reliability="reliable",
        base_confidence=0.90,
    ),
    
    "Healthcare": SectorConfig(
        name="Healthcare",
        primary_methods=["DCF", "P/E", "EV/EBITDA"],
        secondary_methods=["P/E", "EV/EBITDA"],
        min_required_methods=2,
        
        growth_mode="use_historical",
        growth_min=0.00,
        growth_max=0.20,
        growth_override=None,
        
        outlier_min_ratio=0.2,
        outlier_max_ratio=8.0,
        
        capex_impact="medium",
        fcf_reliability="reliable",
        base_confidence=0.85,
    ),

    "Managed Care": SectorConfig(
        name="Managed Care",
        primary_methods=["P/E", "EV/EBITDA"],
        secondary_methods=["P/E"],
        min_required_methods=2,

        growth_mode="conservative_override",
        growth_min=0.00,
        growth_max=0.08,
        growth_override=None,

        outlier_min_ratio=0.2,
        outlier_max_ratio=5.0,

        capex_impact="low",
        fcf_reliability="reliable",
        base_confidence=0.75,

        special_rules={
            "disable_dcf": True,
            "low_confidence_warning": "Managed care / PBM hybrid: DCF is unreliable; use P/E and EV/EBITDA",
        },

        fallback_parent_sector="Healthcare",
    ),

    "Pharmaceuticals": SectorConfig(
        name="Pharmaceuticals",
        primary_methods=["DCF", "P/E", "EV/EBITDA"],
        secondary_methods=["P/E", "EV/EBITDA"],
        min_required_methods=2,

        growth_mode="use_historical",
        growth_min=0.03,
        growth_max=0.18,
        growth_override=None,

        outlier_min_ratio=0.2,
        outlier_max_ratio=7.0,

        capex_impact="medium",
        fcf_reliability="reliable",
        base_confidence=0.80,

        fallback_parent_sector="Healthcare",
    ),

    "Medical Devices": SectorConfig(
        name="Medical Devices",
        primary_methods=["DCF", "P/E", "EV/EBITDA"],
        secondary_methods=["P/E", "EV/EBITDA"],
        min_required_methods=2,

        growth_mode="use_historical",
        growth_min=0.02,
        growth_max=0.18,
        growth_override=None,

        outlier_min_ratio=0.2,
        outlier_max_ratio=7.0,

        capex_impact="medium",
        fcf_reliability="reliable",
        base_confidence=0.78,

        fallback_parent_sector="Healthcare",
    ),

    "Life Science Tools": SectorConfig(
        name="Life Science Tools",
        primary_methods=["DCF", "EV/EBITDA", "P/E"],
        secondary_methods=["EV/EBITDA", "P/E"],
        min_required_methods=2,

        growth_mode="use_historical",
        growth_min=0.02,
        growth_max=0.16,
        growth_override=None,

        outlier_min_ratio=0.2,
        outlier_max_ratio=7.0,

        capex_impact="low",
        fcf_reliability="reliable",
        base_confidence=0.78,

        fallback_parent_sector="Healthcare",
    ),

    "Healthcare Providers": SectorConfig(
        name="Healthcare Providers",
        primary_methods=["EV/EBITDA", "P/E"],
        secondary_methods=["P/E"],
        min_required_methods=2,

        growth_mode="conservative_override",
        growth_min=0.00,
        growth_max=0.08,
        growth_override=None,

        outlier_min_ratio=0.2,
        outlier_max_ratio=5.0,

        capex_impact="very_high",
        fcf_reliability="moderate",
        base_confidence=0.70,

        special_rules={
            "disable_dcf": True,
            "regulated": True,
            "use_normalized_capex": True,
            "low_confidence_warning": "Hospital/provider model: EV/EBITDA and P/E preferred over DCF",
        },

        fallback_parent_sector="Healthcare",
    ),

    "Pharmaceutical Distribution": SectorConfig(
        name="Pharmaceutical Distribution",
        primary_methods=["EV/EBITDA", "P/E"],
        secondary_methods=["P/E"],
        min_required_methods=2,

        growth_mode="conservative_override",
        growth_min=0.00,
        growth_max=0.06,
        growth_override=None,

        outlier_min_ratio=0.2,
        outlier_max_ratio=5.0,

        capex_impact="medium",
        fcf_reliability="moderate",
        base_confidence=0.68,

        special_rules={
            "disable_dcf": True,
            "low_confidence_warning": "Low-margin pharmaceutical distribution: EV/EBITDA and P/E preferred",
        },

        fallback_parent_sector="Healthcare",
    ),
    
    "Consumer": SectorConfig(
        name="Consumer",
        primary_methods=["P/E", "EV/EBITDA", "DCF"],
        secondary_methods=["P/E", "EV/EBITDA"],
        min_required_methods=2,
        
        growth_mode="use_historical",
        growth_min=0.00,
        growth_max=0.15,
        growth_override=None,
        
        outlier_min_ratio=0.2,
        outlier_max_ratio=7.0,
        
        capex_impact="medium",
        fcf_reliability="reliable",
        base_confidence=0.80,
    ),
    
    "Industrials": SectorConfig(
        name="Industrials",
        primary_methods=["EV/EBITDA", "P/E", "DCF"],
        secondary_methods=["P/E", "EV/EBITDA"],
        min_required_methods=2,
        
        growth_mode="use_historical",
        growth_min=0.00,
        growth_max=0.12,
        growth_override=None,
        
        outlier_min_ratio=0.2,
        outlier_max_ratio=6.0,  # Tighter: lumpy capex
        
        capex_impact="high",
        fcf_reliability="moderate",
        base_confidence=0.75,
        
        special_rules={"normalize_capex": True},
    ),
    
    "Financials": SectorConfig(
        name="Financials",
        primary_methods=["P/E", "P/B", "ROE"],  # P/E + asset-based methods for banks/insurers
        secondary_methods=[],
        min_required_methods=2,  # Need at least 2 of 3 methods
        
        growth_mode="conservative_override",
        growth_min=0.00,
        growth_max=0.10,
        growth_override=None,
        
        outlier_min_ratio=0.2,
        outlier_max_ratio=4.0,  # Tight threshold for single-method outliers
        
        capex_impact="low",
        fcf_reliability="unreliable",
        base_confidence=0.60,  # Improved: 3-method blend more robust than 1-method
        
        special_rules={
            "disable_dcf": True,
            "low_confidence_warning": "P/B and ROE available; DCF doesn't apply to banks",
        },
    ),

    "Financial Services": SectorConfig(
        name="Financial Services",
        primary_methods=["P/E", "P/B", "ROE"],  # P/E + asset-based methods for banks/insurers
        secondary_methods=[],
        min_required_methods=2,  # Need at least 2 of 3 methods

        growth_mode="conservative_override",
        growth_min=0.00,
        growth_max=0.10,
        growth_override=None,

        outlier_min_ratio=0.2,
        outlier_max_ratio=4.0,  # Tight threshold for single-method outliers

        capex_impact="low",
        fcf_reliability="unreliable",
        base_confidence=0.60,  # Improved: 3-method blend more robust than 1-method

        special_rules={
            "disable_dcf": True,
            "low_confidence_warning": "P/B and ROE available; DCF doesn't apply to banks",
        },
    ),

    "Financial Data Services": SectorConfig(
        name="Financial Data Services",
        primary_methods=["DCF", "P/E", "EV/EBITDA"],
        secondary_methods=["P/E", "EV/EBITDA"],
        min_required_methods=2,

        growth_mode="use_historical",
        growth_min=0.00,
        growth_max=0.16,
        growth_override=None,

        outlier_min_ratio=0.2,
        outlier_max_ratio=6.5,

        capex_impact="low",
        fcf_reliability="reliable",
        base_confidence=0.75,

        fallback_parent_sector="Financial Services",
    ),

    "Payment Networks": SectorConfig(
        name="Payment Networks",
        primary_methods=["DCF", "P/E", "EV/EBITDA"],
        secondary_methods=["P/E", "EV/EBITDA"],
        min_required_methods=2,

        growth_mode="use_historical",
        growth_min=0.04,
        growth_max=0.15,
        growth_override=None,

        outlier_min_ratio=0.3,
        outlier_max_ratio=4.0,

        capex_impact="very_low",
        fcf_reliability="highly_reliable",
        base_confidence=0.85,

        pe_range=(28.0, 50.0),
        pe_sanity_range=(22.0, 65.0),
        ev_ebitda_range=(24.0, 38.0),
        ev_ebitda_sanity_range=(18.0, 50.0),
        terminal_growth_range=(0.025, 0.04),
        profit_margin_range=(0.35, 0.50),
        revenue_growth_range=(0.05, 0.20),

        field_enforcement={"pe": "soft", "ev_ebitda": "soft", "revenue_growth": "soft", "profit_margin": "soft"},

        special_rules={"platform_business": True},
        policy_confidence=0.88,
    ),

    "Gov Defense IT": SectorConfig(
        name="Gov Defense IT",
        primary_methods=["EV/EBITDA", "P/E", "DCF"],
        secondary_methods=["P/E", "EV/EBITDA"],
        min_required_methods=2,

        growth_mode="conservative_override",
        growth_min=0.00,
        growth_max=0.09,
        growth_override=None,

        outlier_min_ratio=0.2,
        outlier_max_ratio=4.5,

        capex_impact="medium",
        fcf_reliability="moderate",
        base_confidence=0.68,

        discount_rate_range=(0.08, 0.14),
        pe_range=(10.0, 24.0),
        ev_ebitda_range=(7.0, 16.0),
        terminal_growth_range=(0.01, 0.025),
        terminal_fcf_yield_range=(0.04, 0.07),
        revenue_growth_range=(-0.03, 0.15),
        earnings_growth_range=(-0.20, 0.25),

        special_rules={
            "normalize_capex": True,
            "low_confidence_warning": "Government services defense IT: keep conservative terminal and multiple assumptions",
        },
        field_enforcement={"ev_ebitda": "hard", "terminal_growth": "hard", "terminal_fcf_yield": "hard"},
        fallback_parent_sector="Industrials",
    ),

    "Gov Defense IT Hybrid": SectorConfig(
        name="Gov Defense IT Hybrid",
        primary_methods=["EV/EBITDA", "P/E", "DCF"],
        secondary_methods=["P/E", "EV/EBITDA"],
        min_required_methods=2,

        growth_mode="conservative_override",
        growth_min=0.00,
        growth_max=0.10,
        growth_override=None,

        outlier_min_ratio=0.2,
        outlier_max_ratio=5.0,

        capex_impact="medium",
        fcf_reliability="moderate",
        base_confidence=0.65,

        discount_rate_range=(0.08, 0.145),
        pe_range=(10.0, 26.0),
        ev_ebitda_range=(7.0, 18.0),
        terminal_growth_range=(0.01, 0.028),
        terminal_fcf_yield_range=(0.038, 0.07),
        revenue_growth_range=(-0.03, 0.18),
        earnings_growth_range=(-0.20, 0.30),

        special_rules={
            "normalize_cycles": True,
            "low_confidence_warning": "Government/commercial hybrid services: apply wider uncertainty bands",
        },
        field_enforcement={"ev_ebitda": "hard", "terminal_growth": "hard", "terminal_fcf_yield": "hard"},
        fallback_parent_sector="Industrials",
    ),
    
    "Energy": SectorConfig(
        name="Energy",
        primary_methods=["P/E", "EV/EBITDA"],  # NOT DCF (commodity cycles)
        secondary_methods=["EV/EBITDA"],
        min_required_methods=2,
        
        growth_mode="fixed_override",
        growth_min=0.00,
        growth_max=0.05,
        growth_override=0.04,  # Fixed 4% regardless of historical
        
        outlier_min_ratio=0.2,
        outlier_max_ratio=4.0,  # Tight: commodity extremes
        
        capex_impact="very_high",
        fcf_reliability="unreliable",
        base_confidence=0.65,
        
        special_rules={
            "disable_dcf": True,
            "use_normalized_commodity": True,
        },
    ),
    
    "Materials": SectorConfig(
        name="Materials",
        primary_methods=["P/E", "EV/EBITDA"],  # NOT DCF (commodity cycles)
        secondary_methods=["EV/EBITDA"],
        min_required_methods=2,
        
        growth_mode="fixed_override",
        growth_min=0.00,
        growth_max=0.05,
        growth_override=0.04,  # Fixed 4%
        
        outlier_min_ratio=0.2,
        outlier_max_ratio=4.0,
        
        capex_impact="very_high",
        fcf_reliability="unreliable",
        base_confidence=0.65,
        
        special_rules={
            "disable_dcf": True,
            "use_normalized_commodity": True,
        },
    ),
    
    "Utilities": SectorConfig(
        name="Utilities",
        primary_methods=["P/E", "EV/EBITDA"],  # NOT DCF (capex cycles)
        secondary_methods=["EV/EBITDA"],
        min_required_methods=2,
        
        growth_mode="conservative_override",
        growth_min=0.00,
        growth_max=0.04,
        growth_override=None,
        
        outlier_min_ratio=0.2,
        outlier_max_ratio=4.0,  # Tight: capex cycles
        
        capex_impact="very_high",
        fcf_reliability="moderate",
        base_confidence=0.65,
        
        special_rules={
            "disable_dcf": True,
            "use_normalized_capex": True,
            "regulated": True,
        },
    ),
    
    "Telecom": SectorConfig(
        name="Telecom",
        primary_methods=["DCF", "P/E", "EV/EBITDA"],
        secondary_methods=["P/E", "EV/EBITDA"],
        min_required_methods=2,

        growth_mode="use_historical",
        growth_min=0.00,
        growth_max=0.08,             # relaxed; terminal_growth_range still caps terminal at 1–2.5%
        growth_override=None,

        outlier_min_ratio=0.2,
        outlier_max_ratio=4.0,

        capex_impact="very_high",
        fcf_reliability="moderate",
        base_confidence=0.60,

        pe_range=(8.0, 15.0),                # soft prior
        pe_sanity_range=(6.0, 20.0),         # hard emergency clamp
        ev_ebitda_range=(4.5, 8.0),          # soft prior
        ev_ebitda_sanity_range=(3.0, 11.0),  # hard emergency clamp
        discount_rate_range=(0.08, 0.14),
        terminal_growth_range=(0.01, 0.025),

        special_rules={
            "use_normalized_capex": True,
            "regulated": True,
            "leverage_wacc_uplift": True,
        },
    ),

    "Wireless Telecom": SectorConfig(
        name="Wireless Telecom",
        primary_methods=["DCF", "P/E", "EV/EBITDA"],
        secondary_methods=["P/E", "EV/EBITDA"],
        min_required_methods=2,

        growth_mode="use_historical",
        growth_min=0.00,
        growth_max=0.12,             # wireless can sustain higher near-term growth
        growth_override=None,

        outlier_min_ratio=0.2,
        outlier_max_ratio=5.0,

        capex_impact="very_high",
        fcf_reliability="moderate",
        base_confidence=0.65,

        pe_range=(12.0, 22.0),
        pe_sanity_range=(8.0, 28.0),
        ev_ebitda_range=(6.0, 12.0),
        ev_ebitda_sanity_range=(4.0, 15.0),
        discount_rate_range=(0.08, 0.14),
        terminal_growth_range=(0.01, 0.025),

        special_rules={
            "use_normalized_capex": True,
            "leverage_wacc_uplift": True,
        },
        fallback_parent_sector="Telecom",
    ),

    "Integrated Telecom": SectorConfig(
        name="Integrated Telecom",
        primary_methods=["DCF", "P/E", "EV/EBITDA"],
        secondary_methods=["P/E", "EV/EBITDA"],
        min_required_methods=2,

        growth_mode="use_historical",
        growth_min=0.00,
        growth_max=0.08,
        growth_override=None,

        outlier_min_ratio=0.2,
        outlier_max_ratio=4.0,

        capex_impact="very_high",
        fcf_reliability="moderate",
        base_confidence=0.60,

        pe_range=(8.0, 15.0),
        pe_sanity_range=(6.0, 20.0),
        ev_ebitda_range=(4.5, 8.0),
        ev_ebitda_sanity_range=(3.0, 11.0),
        discount_rate_range=(0.08, 0.14),
        terminal_growth_range=(0.01, 0.025),

        special_rules={
            "use_normalized_capex": True,
            "regulated": True,
            "leverage_wacc_uplift": True,
        },
        fallback_parent_sector="Telecom",
    ),

    "Cable/Broadband": SectorConfig(
        name="Cable/Broadband",
        primary_methods=["DCF", "EV/EBITDA", "P/E"],
        secondary_methods=["EV/EBITDA", "P/E"],
        min_required_methods=2,

        growth_mode="use_historical",
        growth_min=0.00,
        growth_max=0.10,
        growth_override=None,

        outlier_min_ratio=0.2,
        outlier_max_ratio=4.5,

        capex_impact="high",
        fcf_reliability="moderate",
        base_confidence=0.65,

        pe_range=(10.0, 18.0),
        pe_sanity_range=(6.0, 22.0),
        ev_ebitda_range=(5.0, 10.0),
        ev_ebitda_sanity_range=(3.0, 13.0),
        discount_rate_range=(0.07, 0.13),
        terminal_growth_range=(0.01, 0.025),

        special_rules={
            "use_normalized_capex": True,
            "leverage_wacc_uplift": True,
        },
        fallback_parent_sector="Telecom",
    ),

    "Real Estate": SectorConfig(
        name="Real Estate",
        primary_methods=[],  # NONE OF OUR METHODS WORK FOR REITS
        secondary_methods=[],
        min_required_methods=0,
        
        growth_mode="n/a",
        growth_min=0.0,
        growth_max=0.0,
        growth_override=None,
        
        outlier_min_ratio=0.2,
        outlier_max_ratio=0.0,
        
        capex_impact="very_high",
        fcf_reliability="unreliable",
        base_confidence=0.0,  # ALWAYS unsupported
        
        special_rules={
            "disable_all": True,
            "requires_custom_metrics": "FFO/AFFO",
            "error_message": "UNSUPPORTED_REGIME: REIT \u2014 FFO/AFFO valuation planned",
        },
    ),

    "Cryptocurrency": SectorConfig(
        name="Cryptocurrency",
        primary_methods=[],
        secondary_methods=[],
        min_required_methods=0,
        growth_mode="n/a",
        growth_min=0.0,
        growth_max=0.0,
        growth_override=None,
        outlier_min_ratio=0.0,
        outlier_max_ratio=0.0,
        capex_impact="n/a",
        fcf_reliability="n/a",
        base_confidence=0.0,
        special_rules={
            "disable_all": True,
            "error_message": "NOT_APPLICABLE: CRYPTO \u2014 no equity cash flow to discount",
        },
    ),

    "Physical Commodity ETF": SectorConfig(
        name="Physical Commodity ETF",
        primary_methods=[],
        secondary_methods=[],
        min_required_methods=0,
        growth_mode="n/a",
        growth_min=0.0,
        growth_max=0.0,
        growth_override=None,
        outlier_min_ratio=0.0,
        outlier_max_ratio=0.0,
        capex_impact="n/a",
        fcf_reliability="n/a",
        base_confidence=0.0,
        special_rules={
            "disable_all": True,
            "error_message": "NOT_APPLICABLE: PHYSICAL_COMMODITY_ETF \u2014 no equity valuation applicable",
        },
    ),
    
    "Aerospace": SectorConfig(
        name="Aerospace",
        primary_methods=["EV/EBITDA", "P/E"],  # DCF possible but project cycles vary
        secondary_methods=["P/E"],
        min_required_methods=2,
        
        growth_mode="use_historical",
        growth_min=0.00,
        growth_max=0.12,
        growth_override=None,
        
        outlier_min_ratio=0.2,
        outlier_max_ratio=5.0,  # Tight: project cycles lumpy
        
        capex_impact="high",
        fcf_reliability="moderate",
        base_confidence=0.70,
        
        special_rules={"normalize_capex": True},
    ),
    
    "Construction": SectorConfig(
        name="Construction",
        primary_methods=["EV/EBITDA", "P/E"],
        secondary_methods=["P/E"],
        min_required_methods=2,
        
        growth_mode="use_historical",
        growth_min=0.00,
        growth_max=0.10,
        growth_override=None,
        
        outlier_min_ratio=0.2,
        outlier_max_ratio=5.0,
        
        capex_impact="medium",
        fcf_reliability="moderate",
        base_confidence=0.70,
        
        special_rules={"normalize_cycles": True},
    ),
    
    "Communication Services": SectorConfig(
        name="Communication Services",
        primary_methods=["DCF", "P/E", "EV/EBITDA"],
        secondary_methods=["P/E", "EV/EBITDA"],
        min_required_methods=2,
        
        growth_mode="use_historical",
        growth_min=0.00,  # Allows declining legacy media companies to reflect ~0% growth
        growth_max=0.25,  # Platform businesses can grow like tech
        growth_override=None,
        
        outlier_min_ratio=0.2,
        outlier_max_ratio=8.0,
        
        capex_impact="low",  # Platform businesses have low capex
        fcf_reliability="reliable",  # Strong FCF generation
        base_confidence=0.90,
        
        special_rules={"platform_business": True},
    ),

    # -----------------------------------------------------------------------
    # Sub-sector overrides: companies that share a raw yfinance sector with
    # the parent but require different valuation policies.
    # -----------------------------------------------------------------------

    "Biotech": SectorConfig(
        name="Biotech",
        # P/E only when profitable; DCF assumes growing FCF that doesn't yet
        # exist pre-revenue, and EV/EBITDA is meaningless for loss-makers.
        primary_methods=["P/E"],
        secondary_methods=[],
        min_required_methods=1,

        growth_mode="use_historical",
        growth_min=0.00,
        growth_max=0.30,   # Can be high once a drug is approved and selling
        growth_override=None,

        outlier_min_ratio=0.2,
        outlier_max_ratio=5.0,  # Tighter: binary pipeline outcomes inflate extremes

        capex_impact="low",              # R&D is OpEx, not CapEx
        fcf_reliability="unreliable",    # Pre-revenue or negative FCF
        base_confidence=0.35,            # Very low: binary FDA / pipeline risk

        special_rules={
            "disable_dcf": True,
            "disable_ev_ebitda": True,
            "low_confidence_warning": (
                "Biotech: pre-revenue / loss-making; P/E only when profitable; "
                "binary pipeline risk makes any estimate highly uncertain"
            ),
        },
        fallback_parent_sector="Healthcare",
    ),

    "Healthcare IT": SectorConfig(
        name="Healthcare IT",
        # EV/EBITDA dominant: GAAP earnings are frequently distorted by
        # acquisition-related intangible amortization; EV/EBITDA and DCF
        # are more reliable anchors for health IT roll-ups.
        primary_methods=["EV/EBITDA", "DCF", "P/E"],
        secondary_methods=["EV/EBITDA", "DCF"],
        min_required_methods=2,

        growth_mode="use_historical",
        growth_min=0.00,
        growth_max=0.25,
        growth_override=None,

        outlier_min_ratio=0.2,
        outlier_max_ratio=6.0,

        capex_impact="low",
        fcf_reliability="moderate",   # acquisitions create lumpy FCF
        base_confidence=0.65,         # earnings distortion lowers inherent reliability

        pe_range=(12.0, 35.0),
        pe_sanity_range=(8.0, 55.0),
        ev_ebitda_range=(8.0, 18.0),
        ev_ebitda_sanity_range=(5.0, 25.0),
        terminal_growth_range=(0.01, 0.03),

        special_rules={
            "low_confidence_warning": (
                "Healthcare IT: GAAP earnings often distorted by acquisition amortization; "
                "EV/EBITDA and DCF are more reliable than P/E here"
            ),
        },
        fallback_parent_sector="Healthcare",
    ),

    "Specialty Chemicals": SectorConfig(
        name="Specialty Chemicals",
        # Industrial gas and specialty chemical companies (LIN, APD, SHW, ECL)
        # have stable long-term contracts and predictable FCF — all three methods
        # are valid, unlike commodity miners where DCF is disabled.
        primary_methods=["DCF", "EV/EBITDA", "P/E"],
        secondary_methods=["EV/EBITDA", "P/E"],
        min_required_methods=2,

        growth_mode="use_historical",
        growth_min=0.00,
        growth_max=0.12,   # Higher ceiling than commodity materials (5 %)
        growth_override=None,

        outlier_min_ratio=0.2,
        outlier_max_ratio=6.0,

        capex_impact="medium",      # Process facilities; less extreme than mining
        fcf_reliability="reliable",  # Stable, contract-based revenue
        base_confidence=0.80,

        fallback_parent_sector="Materials",
    ),

    "Airlines": SectorConfig(
        name="Airlines",
        # Cyclical revenue, very high fleet leverage; DCF is unreliable across
        # the cycle.  EV/EBITDA and P/E provide a more stable relative picture.
        primary_methods=["EV/EBITDA", "P/E"],
        secondary_methods=["P/E"],
        min_required_methods=2,

        growth_mode="conservative_override",
        growth_min=0.00,
        growth_max=0.08,   # Mature, capacity-constrained industry
        growth_override=None,

        outlier_min_ratio=0.2,
        outlier_max_ratio=4.0,   # Tight: cyclical extremes distort multiples

        capex_impact="very_high",      # Aircraft fleet replacement / maintenance
        fcf_reliability="unreliable",  # Highly cyclical FCF
        base_confidence=0.55,          # Low: leverage + cycle sensitivity

        special_rules={
            "disable_dcf": True,
            "cyclical": True,
            "low_confidence_warning": (
                "Airlines: DCF unreliable (cyclical revenue, high fleet leverage); "
                "EV/EBITDA and P/E only"
            ),
        },
        fallback_parent_sector="Industrials",
    ),

    "Software": SectorConfig(
        name="Software",
        # Pure software companies (Adobe, ADP, Salesforce, SAP, Oracle, MSFT)
        # are asset-light with subscription/recurring revenue and very clean FCF.
        # P/E and DCF are equally reliable; EV/EBITDA is supplementary.
        primary_methods=["DCF", "P/E", "EV/EBITDA"],
        secondary_methods=["P/E", "EV/EBITDA"],
        min_required_methods=2,

        growth_mode="use_historical",
        growth_min=0.05,
        growth_max=0.22,   # Even fast-growing software rarely sustains > 22 % long term
        growth_override=None,

        outlier_min_ratio=0.20,
        outlier_max_ratio=8.0,

        capex_impact="low",              # Asset-light; R&D is OpEx not CapEx
        fcf_reliability="reliable",      # Subscription model = predictable FCF
        base_confidence=0.90,
        structural_growth_cap=0.15,

        fallback_parent_sector="Technology",
    ),

    "Digital Advertising Platform": SectorConfig(
        name="Digital Advertising Platform",
        # Ad-tech and app-monetization platforms are asset-light and FCF rich like
        # software, but revenue can be more cyclical because it depends on ad budgets.
        primary_methods=["DCF", "EV/EBITDA", "P/E"],
        secondary_methods=["EV/EBITDA", "P/E"],
        min_required_methods=2,

        growth_mode="use_historical",
        growth_min=0.03,
        growth_max=0.25,
        growth_override=None,

        outlier_min_ratio=0.20,
        outlier_max_ratio=8.0,

        capex_impact="low",
        fcf_reliability="reliable",
        base_confidence=0.85,
        structural_growth_cap=0.18,

        pe_range=(10.0, 50.0),
        ev_ebitda_range=(8.0, 32.0),
        revenue_growth_range=(-0.05, 0.45),
        profit_margin_range=(-0.10, 0.55),
        earnings_growth_range=(-0.25, 0.70),
        special_rules={"platform_business": True, "ad_cycle_exposed": True},
        fallback_parent_sector="Communication Services",
    ),

    "Semiconductors": SectorConfig(
        name="Semiconductors",
        # Chip designers and equipment makers face capacity cycles that make
        # boom-year revenue growth unreliable to extrapolate via DCF.
        # EV/EBITDA is cycle-stable and gets the highest weight; DCF weight
        # is reduced to limit upcycle over-extrapolation.
        primary_methods=["EV/EBITDA", "P/E", "DCF"],
        secondary_methods=["EV/EBITDA", "P/E"],
        min_required_methods=2,

        growth_mode="use_historical",
        growth_min=0.00,
        growth_max=0.20,   # Don't extrapolate upcycle peaks; cap at 20 %
        growth_override=None,

        outlier_min_ratio=0.15,    # Wider lower bound — cyclical troughs inflate ratios
        outlier_max_ratio=10.0,    # Wider upper bound — upcycle peaks inflate multiples

        capex_impact="high",              # Fab construction + equipment cycles
        fcf_reliability="moderate",       # Reliable mid-cycle; swings at extremes
        base_confidence=0.75,             # Cycle uncertainty = materially wider range

        fallback_parent_sector="Technology",
    ),
}


def get_sector_config(sector: str) -> SectorConfig:
    """
    Get sector configuration, with fallback to Consumer default.
    
    Args:
        sector: Sector name from yfinance
        
    Returns:
        SectorConfig object for the sector
    """
    if not sector:
        return SECTOR_CONFIGS["Consumer"]  # Default fallback
    
    sector_lower = sector.lower()
    
    # Direct match
    for config_name, config in SECTOR_CONFIGS.items():
        if config_name.lower() == sector_lower:
            return config
    
    # Fuzzy matching
    if (
        "financial data services" in sector_lower
        or "financial data & stock exchanges" in sector_lower
        or "financial data and stock exchanges" in sector_lower
    ):
        return SECTOR_CONFIGS["Financial Data Services"]
    elif "gov defense it hybrid" in sector_lower or "government services defense it hybrid" in sector_lower:
        return SECTOR_CONFIGS["Gov Defense IT Hybrid"]
    elif "gov defense it" in sector_lower or "government services defense it" in sector_lower:
        return SECTOR_CONFIGS["Gov Defense IT"]
    elif "financial services" in sector_lower:
        return SECTOR_CONFIGS["Financial Services"]
    elif "financial" in sector_lower or "bank" in sector_lower or "insurance" in sector_lower:
        return SECTOR_CONFIGS["Financials"]
    elif "energy" in sector_lower or "oil" in sector_lower or "gas" in sector_lower:
        return SECTOR_CONFIGS["Energy"]
    elif "utilit" in sector_lower or "power" in sector_lower or "water" in sector_lower:
        return SECTOR_CONFIGS["Utilities"]
    elif "communication services" in sector_lower:
        return SECTOR_CONFIGS["Communication Services"]
    elif "managed care" in sector_lower or "pbm" in sector_lower:
        return SECTOR_CONFIGS["Managed Care"]
    elif "pharmaceutical distribution" in sector_lower or "drug distribution" in sector_lower:
        return SECTOR_CONFIGS["Pharmaceutical Distribution"]
    elif "life science tools" in sector_lower or "diagnostic" in sector_lower:
        return SECTOR_CONFIGS["Life Science Tools"]
    elif "healthcare providers" in sector_lower or "hospital" in sector_lower:
        return SECTOR_CONFIGS["Healthcare Providers"]
    elif "medical devices" in sector_lower or "medtech" in sector_lower:
        return SECTOR_CONFIGS["Medical Devices"]
    elif "pharmaceutical" in sector_lower or "drug manufacturer" in sector_lower:
        return SECTOR_CONFIGS["Pharmaceuticals"]
    elif "wireless telecom" in sector_lower:
        return SECTOR_CONFIGS["Wireless Telecom"]
    elif "integrated telecom" in sector_lower:
        return SECTOR_CONFIGS["Integrated Telecom"]
    elif "cable" in sector_lower or "broadband" in sector_lower:
        return SECTOR_CONFIGS["Cable/Broadband"]
    elif "telecom" in sector_lower:
        return SECTOR_CONFIGS["Telecom"]
    elif "specialty chem" in sector_lower:
        return SECTOR_CONFIGS["Specialty Chemicals"]
    elif "material" in sector_lower or "mining" in sector_lower:
        return SECTOR_CONFIGS["Materials"]
    elif "real estate" in sector_lower or "reit" in sector_lower:
        return SECTOR_CONFIGS["Real Estate"]
    elif "advertising platform" in sector_lower or "ad tech" in sector_lower or "ad-tech" in sector_lower:
        return SECTOR_CONFIGS["Digital Advertising Platform"]
    elif "semiconductor" in sector_lower:
        return SECTOR_CONFIGS["Semiconductors"]
    elif "software" in sector_lower:
        return SECTOR_CONFIGS["Software"]
    elif "technolog" in sector_lower:
        return SECTOR_CONFIGS["Technology"]
    elif "biotech" in sector_lower:
        return SECTOR_CONFIGS["Biotech"]
    elif "healthcare it" in sector_lower or "health information technology" in sector_lower:
        return SECTOR_CONFIGS["Healthcare IT"]
    elif "health" in sector_lower:
        return SECTOR_CONFIGS["Healthcare"]
    elif "consumer" in sector_lower or "retail" in sector_lower:
        return SECTOR_CONFIGS["Consumer"]
    elif "airline" in sector_lower:
        return SECTOR_CONFIGS["Airlines"]
    elif "industrial" in sector_lower or "manufact" in sector_lower:
        return SECTOR_CONFIGS["Industrials"]
    elif "aerospace" in sector_lower or "defense" in sector_lower:
        return SECTOR_CONFIGS["Aerospace"]
    elif "construct" in sector_lower:
        return SECTOR_CONFIGS["Construction"]
    
    # Default fallback
    return SECTOR_CONFIGS["Consumer"]


def get_growth_assumption(sector_config: SectorConfig, historical_growth: float) -> float:
    """
    Get the growth rate to use for valuation based on sector config.
    
    Args:
        sector_config: SectorConfig object
        historical_growth: Historical CAGR from financial data
        
    Returns:
        Growth rate to use (0-1 scale)
    """
    if sector_config.growth_mode == "use_historical":
        # Clamp to sector's min/max range
        return max(sector_config.growth_min, min(historical_growth, sector_config.growth_max))
    
    elif sector_config.growth_mode == "conservative_override":
        # Use historical but don't exceed max (usually lower for utilities/telecom)
        return max(sector_config.growth_min, min(historical_growth, sector_config.growth_max))
    
    elif sector_config.growth_mode == "fixed_override":
        # Ignore historical, use fixed value (for energy/materials commodity override)
        return sector_config.growth_override or sector_config.growth_min
    
    else:
        # Fallback
        return max(sector_config.growth_min, min(historical_growth, sector_config.growth_max))


def get_sector_method_weights(sector_name: str) -> Dict[str, float]:
    """
    Get sector-specific valuation method weights.

    Policy:
    - Only methods declared in the sector policy can contribute.
    - Explicit disable rules always exclude a method.
    - Weights are renormalized after policy exclusions.
    """
    cfg = SECTOR_CONFIGS.get(sector_name)
    if cfg:
        raw_weights = cfg.method_weights or {}
        allowed_methods = set(cfg.primary_methods) | set(cfg.secondary_methods)
        disabled_methods = {
            method
            for method, rule in {
                "DCF": "disable_dcf",
                "P/E": "disable_pe",
                "EV/EBITDA": "disable_ev_ebitda",
                "P/B": "disable_pb",
                "ROE": "disable_roe",
            }.items()
            if cfg.special_rules.get(rule)
        }
        eligible_weights = {
            method: weight
            for method, weight in raw_weights.items()
            if weight > 0 and method in allowed_methods and method not in disabled_methods
        }
        total = sum(eligible_weights.values())
        if total > 0:
            return {method: weight / total for method, weight in eligible_weights.items()}
        return {}

    return dict(DEFAULT_METHOD_WEIGHTS.get(
        sector_name,
        {"DCF": 0.35, "P/E": 0.40, "EV/EBITDA": 0.25},
    ))


def get_sector_method_exclusions(sector_name: str) -> List[str]:
    """Return valuation methods excluded by the sector policy."""
    cfg = SECTOR_CONFIGS.get(sector_name)
    if not cfg or cfg.special_rules.get("disable_all"):
        return []

    allowed_methods = set(cfg.primary_methods) | set(cfg.secondary_methods)
    disabled_methods = {
        method
        for method, rule in {
            "DCF": "disable_dcf",
            "P/E": "disable_pe",
            "EV/EBITDA": "disable_ev_ebitda",
            "P/B": "disable_pb",
            "ROE": "disable_roe",
        }.items()
        if cfg.special_rules.get(rule)
    }
    configured_methods = set(cfg.method_weights or {})
    return sorted((configured_methods - allowed_methods) | disabled_methods)


if __name__ == "__main__":
    # Test sector config loading
    print("Sector Configuration System Test")
    print("=" * 80)
    
    for sector_name, config in SECTOR_CONFIGS.items():
        print(f"\n{sector_name}:")
        print(f"  Methods: {config.primary_methods}")
        print(f"  Confidence: {config.base_confidence:.0%}")
        print(f"  Growth mode: {config.growth_mode}")
        print(f"  Outlier threshold: {config.outlier_max_ratio}x")
        if config.special_rules:
            print(f"  Special rules: {config.special_rules}")
