"""
evidence_panel.py
=================
Component for rendering internal D01–D07 Statutory Declaration Evidence.
Defensively handles all missing fields, formats normalized actual values,
identifies source images, and flags conflicting evidence.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, List

DECLARATION_METADATA = {
    "D01": {"name": "Manufacturer / Packer / Importer", "rule": "Rule 6(1)(a)"},
    "D02": {"name": "Common / Generic Name", "rule": "Rule 6(1)(b)"},
    "D03": {"name": "Net Quantity in Standard Units", "rule": "Rule 6(1)(c)"},
    "D04": {"name": "Date of Manufacture / Packing", "rule": "Rule 6(1)(d)"},
    "D05": {"name": "Maximum Retail Price (MRP)", "rule": "Rule 6(1)(e)"},
    "D06": {"name": "Consumer Care Details / Helpline", "rule": "Rule 6(1)(n)"},
    "D07": {"name": "Country of Origin", "rule": "Rule 6(1)(g)"}
}


def _format_actual_values(normalized: Any, raw: str) -> str:
    """Formats normalized entity values into clean human-readable text."""
    if isinstance(normalized, dict) and normalized:
        if normalized.get("formatted"):
            return str(normalized["formatted"])
        if normalized.get("canonical_display"):
            return str(normalized["canonical_display"])
        parts = []
        for k, v in normalized.items():
            if k in ["observations", "unique_normalized_values", "canonical_key", "canonical_tokens", "base_value", "conflicting_values"]:
                continue
            if v and str(v).strip():
                clean_k = k.replace("_", " ").title()
                parts.append(f"{clean_k}: {v}")
        if parts:
            return ", ".join(parts)
    return str(raw).strip() if raw else "—"


def render_evidence_panel(result: Dict[str, Any]):
    """
    Renders an interactive, clean table and cards of D01-D07 declaration evidence.
    """
    if not isinstance(result, dict):
        return

    declarations = result.get("declarations", {}) or {}
    evidence_summary = result.get("evidence_summary", {}) or {}
    conflicts = evidence_summary.get("conflicts_detected", []) or []

    with st.expander("📋 Internal Declaration Evidence (D01 – D07 Statutory Layer)", expanded=True):
        st.caption("Internal explainability layer: Statutory Rule 6 assessment used as evidentiary signals for product-level decision.")

        # If conflicting evidence was detected across multi-image fusion
        if conflicts:
            st.warning(f"⚠️ **Conflicting Evidence Detected** in: {', '.join(conflicts)}. Different views of the package presented contradictory declarations.")

        table_rows = []
        for code, meta in DECLARATION_METADATA.items():
            d_data = declarations.get(code, {})
            if not isinstance(d_data, dict):
                d_data = {}

            status = d_data.get("status") or d_data.get("prediction", "CANNOT_DETERMINE")
            if hasattr(status, "value"):
                status = status.value
            status = str(status).upper()

            conf = float(d_data.get("confidence", d_data.get("extraction_confidence", d_data.get("ocr_confidence", 0.0))) or 0.0)
            raw_val = d_data.get("raw_value") or d_data.get("extracted_value") or d_data.get("raw_text") or ""
            norm_val = d_data.get("normalized_value", {})
            actual_display = _format_actual_values(norm_val, raw_val)

            source_img = d_data.get("source_image", "")
            source_type = d_data.get("source", "OCR")
            source_display = f"{source_type} ({source_img})" if source_img and source_img != "default_image" else str(source_type)

            # Check for conflict in this field
            field_conflicts = d_data.get("conflicting_values", [])
            has_conflict = (code in conflicts) or bool(field_conflicts) or (status == "CONFLICTING_EVIDENCE")

            # Status Badge Icon
            if has_conflict:
                status_display = "⚠️ CONFLICT"
            elif status == "PRESENT":
                status_display = "🟢 PRESENT"
            elif status == "ABSENT":
                status_display = "🔴 ABSENT"
            elif status == "PARTIAL":
                status_display = "🟡 PARTIAL"
            elif status == "NOT_APPLICABLE":
                status_display = "⚪ EXEMPT"
            else:
                status_display = "⚪ CANNOT_DETERMINE"

            table_rows.append({
                "Code": code,
                "Declaration Name": meta["name"],
                "Rule 6 Reference": meta["rule"],
                "Status": status_display,
                "Confidence": f"{conf * 100:.1f}%",
                "Extracted Actual Values / Evidence": actual_display,
                "Source Image / Channel": source_display
            })

        df = pd.DataFrame(table_rows)
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Code": st.column_config.TextColumn("Code", width="small"),
                "Declaration Name": st.column_config.TextColumn("Statutory Declaration", width="medium"),
                "Status": st.column_config.TextColumn("Evidence Status", width="small"),
                "Confidence": st.column_config.TextColumn("Confidence", width="small"),
                "Extracted Actual Values / Evidence": st.column_config.TextColumn("Extracted Actual Value / Evidence Text", width="large"),
                "Source Image / Channel": st.column_config.TextColumn("Source View", width="medium")
            }
        )
