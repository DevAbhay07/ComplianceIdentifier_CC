"""
result_card.py
==============
Component for rendering the prominent top-level product compliance result card.
Defensively handles all data types, missing fields, and custom uncertainty objects.
"""

import streamlit as st
from typing import Dict, Any


def render_result_card(result: Dict[str, Any]):
    """
    Renders the primary verdict hero card with distinct colors, confidence badge, and reasons.
    """
    if not isinstance(result, dict):
        return

    verdict = str(result.get("overall_prediction", "UNCERTAIN") or "UNCERTAIN").upper()
    try:
        confidence = float(result.get("confidence", 0.0) or 0.0)
    except (ValueError, TypeError):
        confidence = 0.0

    uncertainty = result.get("uncertainty", {})
    if isinstance(uncertainty, dict):
        unc_status = str(uncertainty.get("status", "LOW") or "LOW")
    elif isinstance(uncertainty, float):
        unc_status = "HIGH" if uncertainty > 0.4 else "LOW"
    elif isinstance(uncertainty, bool):
        unc_status = "HIGH" if uncertainty else "LOW"
    else:
        unc_status = "LOW"

    primary_reason = str(result.get("primary_reason", "") or "")
    decision_basis = str(result.get("decision_basis", "") or "")
    if not primary_reason and decision_basis:
        primary_reason = decision_basis.split(";")[0]
    if not primary_reason:
        primary_reason = "Statutory Rule 6 declaration assessment complete."

    # Visual Theme Styles
    if verdict == "COMPLIANT":
        bg_color = "rgba(16, 185, 129, 0.12)"
        border_color = "#10b981"
        text_color = "#059669"
        icon = "✅"
        sub_badge = "ALL MANDATORY STATUTORY DECLARATIONS SATISFIED"
    elif verdict == "NON_COMPLIANT":
        bg_color = "rgba(239, 68, 68, 0.12)"
        border_color = "#ef4444"
        text_color = "#dc2626"
        icon = "⚠️"
        sub_badge = "STATUTORY DEFICIENCY IDENTIFIED"
    else:  # UNCERTAIN
        bg_color = "rgba(245, 158, 11, 0.12)"
        border_color = "#f59e0b"
        text_color = "#d97706"
        icon = "❓"
        sub_badge = "INSUFFICIENT VISUAL / OCR EVIDENCE"

    st.markdown(
        f"""
        <div style="
            background: {bg_color};
            border: 2px solid {border_color};
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        ">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-size: 13px; font-weight: 700; letter-spacing: 1.2px; text-transform: uppercase; color: {text_color};">
                    PRODUCT COMPLIANCE VERDICT
                </span>
                <span style="background: {border_color}; color: white; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 600;">
                    Confidence: {confidence * 100:.1f}% (Uncertainty: {unc_status})
                </span>
            </div>
            <div style="display: flex; align-items: baseline; gap: 12px; margin-bottom: 12px;">
                <span style="font-size: 38px;">{icon}</span>
                <span style="font-size: 34px; font-weight: 800; color: {text_color}; letter-spacing: -0.5px;">
                    {verdict}
                </span>
                <span style="font-size: 13px; color: #64748b; font-weight: 500; margin-left: 8px;">
                    — {sub_badge}
                </span>
            </div>
            <div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid rgba(0,0,0,0.08);">
                <p style="margin: 0 0 6px 0; font-size: 14.5px; font-weight: 600; color: #1e293b;">
                    <strong>Primary Reason:</strong> {primary_reason}
                </p>
                <p style="margin: 0; font-size: 13.5px; color: #475569;">
                    <strong>Decision Basis:</strong> {decision_basis or 'Evaluated across Rule 6 requirements.'}
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
