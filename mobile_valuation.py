"""Mobile-first stock valuation, using the existing valuation engine unchanged.

Launch through mobile_app.py, not directly: the router hides existing desktop
pages from the mobile deployment.
"""
from __future__ import annotations

import math
import re
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from valuation_enhanced import (
    ValuationCache,
    build_historical_cache_payload,
    build_investment_score,
    compute_historical_valuations,
    compute_valuation_enhanced,
)
from mobile_valuation_history import (
    build_valuation_history_frame,
    compute_valuation_history_insight,
    get_fundamentals_payload,
)

TICKER_PATTERN = re.compile(r"^[A-Z0-9^._=\-]{1,20}$")
VIEWS = ("Overview", "Value", "Quality", "History")
METHODS = {
    "DCF": "dcf",
    "P/E": "pe",
    "EV/EBITDA": "ev",
    "P/B": "pb",
    "ROE": "roe",
}


def number(value):
    """Return finite float or None; zero remains a meaningful observation."""
    try:
        val = float(value)
        return val if math.isfinite(val) else None
    except (TypeError, ValueError):
        return None


def money(value, currency="USD"):
    val = number(value)
    return f"{currency} {val:,.2f}" if val is not None else "N/A"


def pct(value, decimals=1, signed=True):
    val = number(value)
    if val is None:
        return "N/A"
    return f"{val:+.{decimals}f}%" if signed else f"{val:.{decimals}f}%"


def ratio(value, decimals=2):
    val = number(value)
    return f"{val:.{decimals}f}×" if val is not None else "N/A"


def normalized_ticker(raw):
    ticker = (raw or "").strip().upper()
    return ticker if TICKER_PATTERN.fullmatch(ticker) else None


@st.cache_data(ttl=600, show_spinner=False)
def current_valuation(ticker: str, refresh_key: int):
    # refresh_key is only used to expire one session's cached result on request.
    return compute_valuation_enhanced(ticker)


@st.cache_data(ttl=3600, show_spinner=False)
def fundamental_history(ticker: str, lookback_months: int):
    return get_fundamentals_payload(ticker, lookback_months)


def history_cache():
    return ValuationCache()


def render_flag_messages(result):
    flags = result.get("quality_flags") or []
    if not flags:
        return
    with st.expander(f"Data quality and model notes ({len(flags)})"):
        for message in flags:
            st.caption(f"• {message}")


def render_overview(result, investment_score):
    currency = result.get("currency") or "USD"
    price = result.get("price")
    anchored = (result.get("blended") or {}).get("Base")
    intrinsic = (result.get("blended_intrinsic") or {}).get("Base")
    mos = result.get("anchored_margin_of_safety")
    score = investment_score.get("total_score")

    left, right = st.columns(2, gap="small")
    with left:
        with st.container(border=True):
            st.caption("Investment Score")
            st.metric("Score", f"{score}/100" if score is not None else "N/A", label_visibility="collapsed")
            st.caption(investment_score.get("tier") or "Insufficient evidence")
    with right:
        with st.container(border=True):
            st.caption("Anchored FV upside")
            st.metric("Upside", pct(mos), label_visibility="collapsed")
            st.caption(f"Base fair value: {money(anchored, currency)}")

    if score is None:
        st.info("Total investment score unavailable: the engine requires Valuation Opportunity, Valuation Robustness, and at least one additional component.")

    st.subheader("Score breakdown")
    st.caption(f"Evidence: {investment_score.get('available_components', 0)}/4 dimensions · {investment_score.get('available_weight', 0):.0%} of configured weight")
    for entry in (investment_score.get("components") or {}).values():
        label = entry.get("label", "Unknown")
        component_score = number(entry.get("score"))
        if component_score is None:
            st.caption(f"**{label}:** Unavailable — {entry.get('unavailable_reason', 'insufficient data')}")
        else:
            st.write(f"**{label}: {component_score:.0f}/100** · configured weight {entry.get('weight', 0):.0%}")
            st.progress(max(0.0, min(1.0, component_score / 100.0)))
            with st.expander(f"How is {label} calculated?"):
                st.write(entry.get("detail") or "No explanation supplied by the model.")

    st.divider()
    st.subheader("Model interpretation")
    confidence = number(result.get("confidence_score"))
    conf_msg = result.get("confidence_description") or "No confidence description supplied"
    st.write(f"**Model confidence:** {pct(confidence * 100, signed=False) if confidence is not None else 'N/A'} — {conf_msg}")
    st.caption("Model confidence is not the probability of a positive investment return.")

    confirmation = result.get("fundamental_confirmation") or {}
    st.write(f"**Fundamental support:** {confirmation.get('phrase') or 'Insufficient or unavailable statement evidence.'}")
    st.write(f"**Current quote:** {money(price, currency)} · **Intrinsic base estimate:** {money(intrinsic, currency)}")
    st.caption("The displayed upside follows the original engine's anchored fair-value basis. It is not an independently calculated return forecast.")
    render_flag_messages(result)


def render_valuation(result):
    currency = result.get("currency") or "USD"
    price = result.get("price")
    anchored = result.get("blended") or {}
    intrinsic = result.get("blended_intrinsic") or {}

    st.subheader("Fair-value estimates")
    with st.container(border=True):
        for label, value in (
            ("Current price", price),
            ("Final (confidence-adjusted) base", anchored.get("Base")),
            ("Intrinsic (unanchored) base", intrinsic.get("Base")),
        ):
            left, right = st.columns([3, 2])
            left.write(label)
            right.write(f"**{money(value, currency)}**")
        st.caption(f"Intrinsic FV upside: {pct(result.get('intrinsic_margin_of_safety'))} · Anchored FV upside: {pct(result.get('anchored_margin_of_safety'))}")

    st.subheader("Scenarios — final blended")
    cols = st.columns(3, gap="small")
    for col, scenario in zip(cols, ("Pessimistic", "Base", "Optimistic")):
        value = anchored.get(scenario)
        with col:
            with st.container(border=True):
                st.caption({"Pessimistic": "Bear", "Base": "Base", "Optimistic": "Bull"}[scenario])
                st.write(f"**{money(value, currency)}**")
                val = number(value)
                p = number(price)
                st.caption(pct((val / p - 1) * 100) if val is not None and p is not None and p > 0 else "N/A")

    st.subheader("Applicable valuation models")
    st.caption("Values and weights come directly from the existing sector-specific engine; unavailable methods are not treated as zero.")
    weights = result.get("method_weights") or {}
    any_method = False
    for label, key in METHODS.items():
        vals = result.get(key) or {}
        base = number(vals.get("Base"))
        weight = number(weights.get(label))
        # Excluded methods can still have diagnostic values, but they should
        # not be presented as if they contribute to the blended estimate.
        if weight is None or weight <= 0:
            continue
        any_method = True
        with st.container(border=True):
            left, right = st.columns([2, 3])
            left.write(f"**{label}**")
            right.write(f"{money(base, currency)}")
            st.caption(f"Configured/effective method weight: {pct(weight * 100, signed=False) if weight is not None else 'N/A'}")
    if not any_method:
        st.info("No applicable model values available.")

    with st.expander("Model parameters and caveats"):
        discount = number(result.get("discount_rate"))
        st.write(f"Discount rate: {pct(discount * 100, signed=False) if discount is not None else 'N/A'}")
        st.write(f"Valuation route: {result.get('sector') or 'Unknown'}")
        st.write(f"Models used: {', '.join(result.get('methods_used') or []) or 'N/A'}")
        st.write(f"Financial-reporting currency: {result.get('financial_currency') or 'N/A'}")
        if result.get("distressed_ev_only_no_fair_value"):
            st.warning("The engine identified an EV-only distressed valuation case; fair value may be unavailable.")
    render_flag_messages(result)


def render_fundamentals(result):
    st.subheader("Fundamental support")
    confirmation = result.get("fundamental_confirmation") or {}
    st.write(confirmation.get("phrase") or "No fundamental confirmation available.")
    breakdown = confirmation.get("breakdown") or {}
    if breakdown:
        # Display-only unit annotations. We do NOT change scores or metric values.
        percent_metrics = {
            "revenue_growth", "eps_growth", "fcf_growth",
            "fcf_conversion", "operating_margin_trend", "share_dilution",
        }
        for key, entry in breakdown.items():
            if not isinstance(entry, dict):
                continue
            label = entry.get("label") or key.replace("_", " ").title()
            value = entry.get("value")
            status = entry.get("status", "unavailable")
            numeric = number(value)
            if numeric is None:
                display = "N/A"
            elif key in percent_metrics:
                display = pct(numeric)
            elif key == "latest_annual_eps":
                display = money(numeric, result.get("financial_currency") or result.get("currency") or "USD") + "/share"
            else:
                display = str(value)
            with st.container(border=True):
                st.write(f"**{label.title()}**")
                st.write(f"{display} · {status}")
        st.caption("Operating margin trend is the relative change reported by the engine, not percentage-point change. Missing or inapplicable values stay N/A.")
    else:
        st.info("Fundamental time series are unavailable or do not meet the model's minimum coverage.")

    st.subheader("Growth and valuation checks")
    assumptions = result.get("base_assumptions") or {}
    items = (
        ("Historical EPS growth", pct(number(result.get("historical_eps_growth")) * 100, signed=False) if number(result.get("historical_eps_growth")) is not None else "N/A"),
        ("Historical FCF growth", pct(number(result.get("historical_fcf_growth")) * 100, signed=False) if number(result.get("historical_fcf_growth")) is not None else "N/A"),
        ("Revenue growth", pct(number(assumptions.get("revenue_growth")) * 100, signed=False) if number(assumptions.get("revenue_growth")) is not None else "N/A"),
        ("Operating margin", pct(number(assumptions.get("operating_margin")) * 100, signed=False) if number(assumptions.get("operating_margin")) is not None else "N/A"),
        ("PEG", ratio(result.get("peg_ratio"))),
        ("PEG source", (result.get("peg_growth_source") or "Unavailable").replace("_", " ").title()),
        ("Net debt / EBITDA", ratio(result.get("net_debt_to_ebitda"))),
    )
    with st.container(border=True):
        for label, display in items:
            col1, col2 = st.columns([3, 2])
            col1.write(label)
            col2.write(f"**{display}**")
    sector = result.get("sector") or ""
    if sector in {"Financials", "Financial Services", "Real Estate"}:
        st.caption("Net debt / EBITDA is not considered comparable for this sector by the investment-score engine.")
    st.caption("Missing or inapplicable metrics remain unavailable and are not scored as zero.")


def _history_chart(df, currency):
    fig = go.Figure()
    if not df.empty:
        fig.add_trace(go.Scatter(x=df["date"], y=df["price"], mode="lines+markers", name="Price", line=dict(width=2)))
        fig.add_trace(go.Scatter(x=df["date"], y=df["fair_value_base"], mode="lines+markers", name="Fair value", line=dict(width=2)))
        if "fair_value_low" in df and "fair_value_high" in df:
            fig.add_trace(go.Scatter(x=df["date"], y=df["fair_value_low"], mode="lines", name="Bear", line=dict(width=1, dash="dot")))
            fig.add_trace(go.Scatter(x=df["date"], y=df["fair_value_high"], mode="lines", name="Bull", line=dict(width=1, dash="dot")))
    fig.update_layout(
        height=315,
        margin=dict(t=8, l=2, r=2, b=4),
        yaxis_title=currency,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
    )
    return fig


def render_history(ticker, result):
    st.subheader("Historical valuation")
    df = build_valuation_history_frame(ticker)
    if df.empty:
        st.info("No saved historical valuations for this stock in this environment. Generate a backfill below; cloud-local cache can disappear after reboot.")
    else:
        st.plotly_chart(_history_chart(df, result.get("currency") or "USD"), use_container_width=True, config={"displayModeBar": False})
        st.caption("The chart uses the same saved historical anchors as the desktop page. Dates and financial observation availability follow the existing backfill logic.")
        base = (result.get("blended") or {}).get("Base")
        insight = compute_valuation_history_insight(ticker, result.get("price"), base)
        if insight.get("status") == "ok":
            with st.container(border=True):
                cols = st.columns(2)
                cols[0].metric("Current percentile", f"{insight['current_percentile']:.0f}th")
                cols[1].metric("Vs. historical median", f"{insight['delta_vs_median']:+.1f} pp")
                st.write(insight["summary"])
                st.caption(f"Median: {pct(insight['historical_median'])} · P10/P90: {pct(insight['p10'])} / {pct(insight['p90'])} · Observations: {insight['points']}")
            st.caption("Historical percentiles can use interpolated monthly fair-value points, exactly as in the desktop implementation; they are not all independent fundamental observations.")
            if st.button("Explain historical fundamental drivers", use_container_width=True):
                st.session_state[f"history_drivers_{ticker}"] = True
            if st.session_state.get(f"history_drivers_{ticker}"):
                with st.spinner("Loading fundamentals and the existing historical explanation…"):
                    fundamentals = fundamental_history(ticker, 60)
                    explained = compute_valuation_history_insight(ticker, result.get("price"), base, fundamentals)
                driver = explained.get("fundamental_drivers") or {}
                if driver.get("status") == "ok":
                    for key in ("current_state_text", "historical_context_text", "fundamental_explanation_text", "conclusion_text"):
                        if driver.get(key):
                            st.write(driver[key])
                else:
                    st.info(driver.get("message", "Not enough fundamental evidence to explain historical patterns."))
        else:
            st.info(insight.get("message", "Historical comparison unavailable."))

    with st.expander("Generate or refresh historical valuations"):
        st.warning("A historical backfill may trigger many Yahoo Finance requests and take significant time. It also writes local JSON history, which is temporary on Community Cloud.")
        months = st.select_slider("Lookback (months)", options=(6, 12, 24, 36, 48, 60), value=24)
        if st.button("Generate history", type="primary", use_container_width=True):
            with st.spinner(f"Calculating {months} months of historical valuations for {ticker}…"):
                hist = compute_historical_valuations(ticker, months_back=months)
                if not isinstance(hist, dict) or hist.get("error"):
                    st.error((hist or {}).get("error", "Historical backfill failed."))
                else:
                    observations = hist.get("historical_valuations") or {}
                    if observations:
                        cache = history_cache()
                        cache.clear_history(ticker)
                        for date, historical_result in observations.items():
                            cache.save_valuation(ticker, build_historical_cache_payload(historical_result), date=date, skip_date_filter=True, source="historical_backfill")
                        cache.save_valuation(ticker, result, source="current")
                        st.success(f"Saved {len(observations)} historical entries to the current environment.")
                        st.rerun()
                    else:
                        st.info("No historical points were returned by the valuation engine.")


def main():
    st.title("📈 Valuation Lite")
    st.caption("Your existing stock valuation methodology, optimized for Android.")

    requested = st.query_params.get("ticker", "")
    if "mobile_active_ticker" not in st.session_state:
        st.session_state.mobile_active_ticker = normalized_ticker(requested)
    selected = st.session_state.mobile_active_ticker

    with st.form("mobile_search", border=False):
        ticker_input = st.text_input("Stock ticker", value=selected or normalized_ticker(requested) or "", placeholder="e.g. ASML.AS, ACN, GOOGL")
        submitted = st.form_submit_button("Analyze stock", type="primary", use_container_width=True)
    if submitted:
        new_ticker = normalized_ticker(ticker_input)
        if new_ticker is None:
            st.error("Enter a valid Yahoo Finance ticker (letters, numbers, dot, dash or ^).")
            st.stop()
        st.session_state.mobile_active_ticker = new_ticker
        st.query_params["ticker"] = new_ticker
        st.session_state.mobile_view = "Overview"
        selected = new_ticker

    if not selected:
        st.info("Enter a ticker above to run the same valuation engine used on desktop.")
        st.stop()

    actions = st.columns([3, 2], gap="small")
    actions[0].caption(f"Selected: **{selected}**")
    refresh = actions[1].button("↻ Refresh data", use_container_width=True)
    if "mobile_refresh_key" not in st.session_state:
        st.session_state.mobile_refresh_key = 0
    if refresh:
        st.session_state.mobile_refresh_key += 1

    with st.spinner(f"Valuing {selected}…"):
        try:
            result = current_valuation(selected, st.session_state.mobile_refresh_key)
        except Exception as exc:
            st.error(f"Valuation could not be completed: {exc}")
            st.stop()
    if not isinstance(result, dict) or result.get("error"):
        st.error((result or {}).get("error", "The valuation engine did not return a result."))
        st.stop()

    st.subheader(f"{result.get('name') or selected}")
    st.caption(f"{selected} · {result.get('sector') or 'Unknown sector'} · Quote {money(result.get('price'), result.get('currency') or 'USD')}")
    as_of = (result.get("base_assumptions") or {}).get("valuation_date")
    st.caption(f"Model valuation date: {as_of or 'not available'} · Quote timestamp not supplied by the valuation engine")

    investment_score = build_investment_score(result, result.get("fundamental_confirmation"))
    view = st.segmented_control("Analysis section", VIEWS, default="Overview", key="mobile_view", label_visibility="collapsed")
    if view == "Value":
        render_valuation(result)
    elif view == "Quality":
        render_fundamentals(result)
    elif view == "History":
        render_history(selected, result)
    else:
        render_overview(result, investment_score)

    st.divider()
    st.caption("For research and decision support only. Scores summarize model evidence, not investment recommendations or return probabilities.")


main()
