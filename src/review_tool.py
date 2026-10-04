"""
review_tool.py
================
Streamlit-based review interface with AI-assisted drafting for the
36 inconsistent annotations identified in Phase 4.

IMPORTANT:
  - The drafting function is an ASSISTANT ONLY.
  - It must NEVER become the final annotator.
  - Machine Draft != Human Verified Annotation.
  - Original annotations are NEVER modified.
  - AI drafts are NOT ground truth and must NOT be used as training labels.
  - Human reviewer confirms or revises every declaration.
  - Reviews are stored separately from original annotations.

Usage:
    streamlit run src/review_tool.py
"""

import ast
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

import streamlit as st

# Import the AI drafting engine
from src.data.generate_review_draft import (
    DECLS,
    DECL_NAMES,
    DEFAULT_GEMINI_MODEL,
    SUPPORTED_DRAFT_MODELS,
    VALID_DECL_LABELS,
    VALID_EVIDENCE_SOURCES,
    VALID_OVERALL_LABELS,
    collect_product_evidence,
    generate_ai_draft,
)

# -------------------------------------------------------------------------
# Constants & Paths
# -------------------------------------------------------------------------
REVIEW_CSV_PATH = "annotations/review_36.csv"
REVIEW_RESULTS_PATH = "annotations/review_36_results.jsonl"
PARSED_DATA_PATH = "data/interim/parsed_annotation_candidates.jsonl"
ANNOTATIONS_PATH = "annotations/pilot_annotations.jsonl"

DECLARATION_LABELS = ["PRESENT", "ABSENT", "PARTIAL", "NOT_APPLICABLE", "CANNOT_DETERMINE"]
OVERALL_LABELS = ["COMPLIANT", "NON_COMPLIANT", "UNCERTAIN"]

EVIDENCE_SOURCES = [
    "PRODUCT_TITLE",
    "PRODUCT_DESCRIPTION",
    "PRODUCT_SPECIFICATIONS",
    "PRODUCT_IMAGE",
    "PRODUCT_PAGE",
    "MULTIPLE_SOURCES",
    "NO_EVIDENCE",
    "CANNOT_VERIFY",
]

DECL_REVIEW_GUIDANCE = {
    "D01": (
        "Is there a manufacturer/packer/importer name AND address?\n\n"
        "- **PRESENT**: Full name and complete postal address explicitly stated.\n"
        "- **PARTIAL**: Brand name exists but no complete address/pincode.\n"
        "- **ABSENT**: No manufacturer/brand information at all.\n"
        "- **CANNOT_DETERMINE**: Insufficient evidence to judge."
    ),
    "D02": (
        "Is the generic/common product name identifiable?\n\n"
        "- **PRESENT**: Clear generic commodity name (e.g., 'Lip Balm', 'Coffee Mug').\n"
        "- **PARTIAL**: Only vague identification.\n"
        "- **ABSENT**: No generic name identifiable."
    ),
    "D03": (
        "Is net quantity (weight/volume/count) explicitly stated?\n\n"
        "- **PRESENT**: Specific quantity in standard SI units (e.g., '250 ml', 'Pack of 2').\n"
        "- **PARTIAL**: Ambiguous quantity information.\n"
        "- **ABSENT**: No quantity information anywhere."
    ),
    "D04": (
        "Is the month and year of manufacture/packing/import stated?\n\n"
        "- **PRESENT**: Explicit month/year.\n"
        "- **CANNOT_DETERMINE**: No date info in listing/packaging (standard for e-commerce).\n"
        "- **ABSENT**: Omitted where expected."
    ),
    "D05": (
        "Is the MRP (retail price) available?\n\n"
        "- **PRESENT**: Retail price field is available.\n"
        "- **ABSENT**: No price field available."
    ),
    "D06": (
        "Are consumer care/complaint contact details available?\n\n"
        "- **PRESENT**: Customer care phone number, email, or helpline address.\n"
        "- **PARTIAL**: Generic website support link only.\n"
        "- **ABSENT**: No consumer care details found."
    ),
    "D07": (
        "Is the country of origin explicitly stated?\n\n"
        "- **PRESENT**: Country explicitly mentioned (e.g., 'Made in India').\n"
        "- **NOT_APPLICABLE**: Domestic product where specific origin declaration is exempt/evident.\n"
        "- **CANNOT_DETERMINE**: Not enough evidence to determine."
    ),
}

LABEL_EMOJI = {
    "PRESENT": "🟢",
    "ABSENT": "🔴",
    "PARTIAL": "🟡",
    "NOT_APPLICABLE": "⚪",
    "CANNOT_DETERMINE": "🔵",
}


# -------------------------------------------------------------------------
# Data Loading
# -------------------------------------------------------------------------
@st.cache_data
def load_review_products():
    """Load the 36 flagged products from the review CSV."""
    import pandas as pd
    return pd.read_csv(REVIEW_CSV_PATH)


@st.cache_data
def load_parsed_products():
    """Load full parsed product data."""
    products = {}
    with open(PARSED_DATA_PATH, "r") as f:
        for line in f:
            rec = json.loads(line)
            products[str(rec.get("product_id", ""))] = rec
    return products


@st.cache_data
def load_original_annotations():
    """Load deduplicated original annotations (read-only)."""
    all_anns = []
    with open(ANNOTATIONS_PATH) as f:
        for line in f:
            all_anns.append(json.loads(line))
    latest = {}
    for ann in all_anns:
        pid = ann["product_id"]
        if pid not in latest or ann["timestamp"] > latest[pid]["timestamp"]:
            latest[pid] = ann
    return latest


def load_existing_reviews():
    """Load previously saved reviews from review_36_results.jsonl."""
    reviews = {}
    if os.path.exists(REVIEW_RESULTS_PATH):
        with open(REVIEW_RESULTS_PATH, "r") as f:
            for line in f:
                if line.strip():
                    rev = json.loads(line)
                    pid = rev.get("product_id", "")
                    reviews[pid] = rev
    return reviews


def save_review(review: dict):
    """Append a review record to the JSONL file."""
    os.makedirs(os.path.dirname(REVIEW_RESULTS_PATH), exist_ok=True)
    with open(REVIEW_RESULTS_PATH, "a") as f:
        f.write(json.dumps(review, ensure_ascii=False) + "\n")


# -------------------------------------------------------------------------
# Main UI
# -------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="NitiMark — AI-Assisted Review Tool",
        page_icon="⚖️",
        layout="wide",
    )

    st.title("⚖️ NitiMark — AI-Assisted Compliance Review Tool")
    st.caption(
        "Phase 4.5B: Human review of inconsistent annotations with AI-assisted drafting. "
        "The drafting engine is an **assistant only** — all final annotations are verified by a human. "
        "Original annotations remain strictly preserved."
    )

    # ------------------------------------------------------------------
    # Sidebar
    # ------------------------------------------------------------------
    with st.sidebar:
        st.header("⚙️ Reviewer Settings")
        reviewer_id = st.text_input("Reviewer ID", value="reviewer_01")

        st.divider()
        st.header("🤖 AI Drafting Settings")

        env_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        user_key = st.text_input(
            "Gemini API Key (optional override)",
            type="password",
            value="",
            help="Enter your Gemini API key if not set in environment variables.",
        )
        active_api_key = user_key.strip() if user_key.strip() else env_key

        model_options = list(SUPPORTED_DRAFT_MODELS.keys())
        default_idx = 0
        for idx, (disp_name, m_id) in enumerate(SUPPORTED_DRAFT_MODELS.items()):
            if m_id == DEFAULT_GEMINI_MODEL:
                default_idx = idx
                break

        selected_model_label = st.selectbox(
            "AI Drafting Model",
            model_options,
            index=default_idx,
            help="Select the Gemini model to use for generating draft annotations.",
        )
        selected_model_id = SUPPORTED_DRAFT_MODELS[selected_model_label]
        st.caption(f"Model ID: `{selected_model_id}`")

        if active_api_key:
            st.success("🟢 API Key configured")
        else:
            st.warning("⚠️ No API Key configured. Manual annotation only.")

        st.divider()
        st.header("📊 Review Progress")

        review_df = load_review_products()
        products = load_parsed_products()
        original_anns = load_original_annotations()
        existing_reviews = load_existing_reviews()

        reviewed_count = len(existing_reviews)
        total_count = len(review_df)

        st.progress(reviewed_count / total_count if total_count > 0 else 0)
        st.write(f"**{reviewed_count} / {total_count}** products reviewed")

        st.divider()
        st.header("🔎 Navigate")

        nav_option = st.radio(
            "Show products:",
            ["All", "Not reviewed", "Already reviewed"],
        )

        all_pids = review_df["product_id"].tolist()

        if nav_option == "Not reviewed":
            visible_pids = [pid for pid in all_pids if pid not in existing_reviews]
        elif nav_option == "Already reviewed":
            visible_pids = [pid for pid in all_pids if pid in existing_reviews]
        else:
            visible_pids = all_pids

        if not visible_pids:
            st.warning("No products match this filter.")
            return

        product_options = {}
        for i, pid in enumerate(visible_pids):
            row = review_df[review_df["product_id"] == pid].iloc[0]
            name = str(row.get("product_name", ""))[:45]
            status = "✅" if pid in existing_reviews else "⏳"
            product_options[f"{status} #{i+1} | {name}"] = pid

        selected_label = st.selectbox("Select product:", list(product_options.keys()))
        current_pid = product_options[selected_label]

    # ------------------------------------------------------------------
    # Product Data & Context
    # ------------------------------------------------------------------
    row = review_df[review_df["product_id"] == current_pid].iloc[0]
    product = products.get(current_pid, {})
    ann = original_anns.get(current_pid, {})

    is_already_reviewed = current_pid in existing_reviews

    if is_already_reviewed:
        st.success(
            f"✅ **Previously Reviewed**: This product already has a saved review record in "
            f"`review_36_results.jsonl`. Saving again will append an updated review record."
        )

    # Inconsistency Banner
    st.error(
        "⚠️ **FLAGGED INCONSISTENCY**: Original annotation has all 7 declarations marked "
        "`PRESENT` but the overall label is `NON_COMPLIANT`."
    )

    # ------------------------------------------------------------------
    # SECTION 1: AI Drafting Action Bar
    # ------------------------------------------------------------------
    st.header("🤖 AI Drafting Assistant")
    st.caption(f"Active Model: **{selected_model_label}** (`{selected_model_id}`)")

    # Session State keys for current product
    draft_state_key = f"ai_draft_{current_pid}"
    draft_status_key = f"draft_status_{current_pid}"
    draft_hist_key = f"draft_history_{current_pid}"

    current_draft = st.session_state.get(draft_state_key, None)
    draft_history = st.session_state.get(draft_hist_key, [])

    draft_col1, draft_col2, draft_col3 = st.columns([1.5, 2, 1.5])

    with draft_col1:
        generate_clicked = st.button(
            f"⚡ Generate Draft ({selected_model_label})",
            type="primary",
            use_container_width=True,
            help=f"Analyzes listing text, specs, and packaging images using {selected_model_label}.",
        )

    with draft_col2:
        if current_draft and current_draft.get("draft_generated"):
            m_used = current_draft.get("model_name", selected_model_id)
            st.markdown(f"🏷️ **Status**: `AI draft ({m_used}) — not yet verified`")
        elif st.session_state.get(draft_status_key) == "quota_exhausted":
            st.error("⚠️ Quota exhausted. Select another model in sidebar.")
        elif st.session_state.get(draft_status_key) == "failed":
            st.warning("⚠️ AI drafting unavailable. Continue manual annotation.")
        else:
            st.info(f"💡 Click to generate draft using **{selected_model_label}**.")

    with draft_col3:
        if current_draft and current_draft.get("draft_generated"):
            if st.button("📋 Accept All Drafts", use_container_width=True, help="Copies all AI suggestions to human review fields for verification."):
                for did in DECLS:
                    d_draft = current_draft["declarations"].get(did, {})
                    st.session_state[f"rev_label_{current_pid}_{did}"] = d_draft.get("draft_label", "CANNOT_DETERMINE")
                    st.session_state[f"rev_source_{current_pid}_{did}"] = d_draft.get("evidence_source", "MULTIPLE_SOURCES")
                    st.session_state[f"rev_notes_{current_pid}_{did}"] = f"[AI Draft ({current_draft.get('model_name')})] {d_draft.get('evidence_text', '')}"
                    st.session_state[f"rev_action_{current_pid}_{did}"] = "ACCEPTED_AI_DRAFT"

                sug_overall = current_draft.get("suggested_overall", {}).get("suggested_label", "NON_COMPLIANT")
                st.session_state[f"rev_overall_{current_pid}"] = sug_overall
                st.session_state[f"overall_action_{current_pid}"] = "ACCEPTED_AI_DRAFT"
                st.rerun()

    # Handle Draft Generation Execution
    if generate_clicked:
        with st.spinner(f"Analyzing product metadata, specifications, and packaging images with {selected_model_label}..."):
            draft_result = generate_ai_draft(
                product=product,
                api_key=active_api_key,
                model=selected_model_id,
            )

        if draft_result.get("draft_generated"):
            st.session_state[draft_state_key] = draft_result
            st.session_state[draft_status_key] = "success"

            # Append to draft history
            if draft_hist_key not in st.session_state:
                st.session_state[draft_hist_key] = []
            st.session_state[draft_hist_key].append(draft_result)

            st.success(f"✅ AI Draft generated successfully using {selected_model_label}! Review recommendations below.")
            st.rerun()
        else:
            if draft_result.get("is_quota_error"):
                st.session_state[draft_status_key] = "quota_exhausted"
                st.error(f"❌ {draft_result.get('error')}")
                st.warning(
                    f"💡 **Action Required**: {selected_model_label} quota is currently exhausted. "
                    f"Please choose an alternate model from the sidebar (e.g. Gemini 3.6 Flash, Gemini 2.5 Pro, or Gemini 2.5 Flash-Lite) and try again."
                )
            else:
                st.session_state[draft_status_key] = "failed"
                err_msg = draft_result.get("error", "AI drafting unavailable.")
                st.error(f"❌ {err_msg}")
                st.warning("⚠️ AI drafting unavailable. Continue manual annotation.")

    # Show Draft History Expander if multiple models were tested for this product
    if draft_history and len(draft_history) > 1:
        with st.expander(f"📜 Model Comparison History ({len(draft_history)} drafts generated)", expanded=False):
            for h_idx, h_draft in enumerate(draft_history, 1):
                st.markdown(f"**Draft #{h_idx}** — Model: `{h_draft.get('model_name')}` at `{h_draft.get('draft_timestamp', '')[11:19]}` UTC")
                h_decls = h_draft.get("declarations", {})
                h_summary = " | ".join([f"{d}: `{h_decls.get(d, {}).get('draft_label', '?')}`" for d in DECLS])
                st.caption(f"Labels: {h_summary}")
                st.caption(f"Suggested Overall: `{h_draft.get('suggested_overall', {}).get('suggested_label', '?')}`")
                st.divider()

    st.divider()


    # ------------------------------------------------------------------
    # SECTION 2: Available Evidence
    # ------------------------------------------------------------------
    st.header("📦 Product Evidence")

    ev_col1, ev_col2 = st.columns(2)

    with ev_col1:
        st.subheader("Listing Metadata")
        st.markdown(f"**Product ID:** `{current_pid}`")
        st.markdown(f"**Product Name:** {product.get('product_name', 'N/A')}")
        st.markdown(f"**Category:** {product.get('top_category', 'N/A')}")
        st.markdown(f"**Category Tree:** {product.get('product_category_tree', 'N/A')}")
        st.markdown(f"**Brand:** {product.get('brand', '') or '❌ *Missing*'}")
        st.markdown(f"**Retail Price (MRP):** ₹{product.get('retail_price', 'N/A')}")
        st.markdown(f"**Discounted Price:** ₹{product.get('discounted_price', 'N/A')}")

    with ev_col2:
        st.subheader("Product Specifications")
        specs = product.get("parsed_specs", {})
        if specs:
            for k, v in specs.items():
                st.markdown(f"- **{k}:** {v}")
        else:
            st.warning("No specifications available in listing.")

    # Description
    with st.expander("📝 Full Description", expanded=False):
        desc = product.get("description", "")
        if desc:
            st.text(desc[:3000])
        else:
            st.warning("No description available")

    # Images
    raw_urls = product.get("image_urls", "")
    if raw_urls and str(raw_urls) != "nan":
        with st.expander("🖼️ Product Images", expanded=True):
            try:
                parsed_urls = ast.literal_eval(raw_urls)
                urls = [u.strip().strip('"\'') for u in parsed_urls] if isinstance(parsed_urls, list) else [str(parsed_urls)]
            except Exception:
                urls = [raw_urls.strip('[]"\'').split(",")[0].strip().strip('"\'')]

            img_cols = st.columns(min(len(urls), 4))
            for idx, url in enumerate(urls[:4]):
                with img_cols[idx]:
                    st.image(url, caption=f"Image {idx+1}", use_container_width=True)

    st.divider()

    # ------------------------------------------------------------------
    # SECTION 3: Original Annotation (Read-Only Baseline)
    # ------------------------------------------------------------------
    st.header("📋 Original Annotation (Read-Only)")
    st.caption("Preserved original human annotations for reference. These will NOT be overwritten.")

    orig_cols = st.columns(7)
    for i, did in enumerate(DECLS):
        with orig_cols[i]:
            orig_lbl = str(row.get(f"original_{did}", ""))
            st.markdown(f"**{did}**")
            st.markdown(f"{LABEL_EMOJI.get(orig_lbl, '❓')} `{orig_lbl}`")
            orig_ev = str(row.get(f"evidence_{did}", ""))
            if orig_ev and orig_ev != "nan":
                st.caption(orig_ev[:60])

    st.markdown(f"**Original Overall:** 🔴 `{row.get('original_overall', '')}`")
    st.markdown(f"**Original Notes:** {row.get('original_overall_notes', 'None')}")

    st.divider()

    # ------------------------------------------------------------------
    # SECTION 4: Declaration Review with AI Assistance
    # ------------------------------------------------------------------
    st.header("✍️ Human Review & Declaration Decisions")
    st.warning(
        "⚠️ **CRITICAL GROUND-TRUTH RULE**: Machine drafts are suggestions only. "
        "Examine the evidence and confirm each declaration."
    )

    revised_declarations = {}
    review_actions = {}

    for did in DECLS:
        st.subheader(f"{did}: {DECL_NAMES[did]}")

        orig_lbl = str(row.get(f"original_{did}", "PRESENT"))
        orig_ev = str(row.get(f"evidence_{did}", ""))

        # Initialize session state keys for this declaration
        lbl_key = f"rev_label_{current_pid}_{did}"
        src_key = f"rev_source_{current_pid}_{did}"
        notes_key = f"rev_notes_{current_pid}_{did}"
        action_key = f"rev_action_{current_pid}_{did}"

        if action_key not in st.session_state:
            st.session_state[action_key] = "MANUAL"

        # AI Draft Display Box
        has_draft = current_draft and current_draft.get("draft_generated") and did in current_draft.get("declarations", {})
        draft_item = current_draft["declarations"][did] if has_draft else None

        if has_draft and draft_item:
            d_lbl = draft_item.get("draft_label", "CANNOT_DETERMINE")
            d_src = draft_item.get("evidence_source", "NO_EVIDENCE")
            d_conf = draft_item.get("confidence", 0.5)
            d_ev = draft_item.get("evidence_text", "")
            d_reason = draft_item.get("reasoning", "")

            with st.container(border=True):
                st.markdown(
                    f"🤖 **AI Draft**: {LABEL_EMOJI.get(d_lbl, '❓')} `{d_lbl}` &nbsp;|&nbsp; "
                    f"**AI Draft Confidence**: `{d_conf:.2f}` &nbsp;|&nbsp; "
                    f"**Source**: `{d_src}`"
                )
                if d_ev:
                    st.markdown(f"📌 **Evidence Found**: *\"{d_ev}\"*")
                if d_reason:
                    st.markdown(f"💡 **Reasoning**: {d_reason}")

                btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 2])
                with btn_col1:
                    if st.button(f"Accept Draft", key=f"btn_accept_{current_pid}_{did}"):
                        st.session_state[lbl_key] = d_lbl
                        st.session_state[src_key] = d_src if d_src in EVIDENCE_SOURCES else "MULTIPLE_SOURCES"
                        st.session_state[notes_key] = f"[AI Draft] {d_ev}" if d_ev else f"[AI Draft] {d_reason}"
                        st.session_state[action_key] = "ACCEPTED_AI_DRAFT"
                        st.rerun()

                with btn_col2:
                    if st.button(f"Reject Draft", key=f"btn_reject_{current_pid}_{did}"):
                        st.session_state[action_key] = "REJECTED_AI_DRAFT"
                        st.rerun()

                with btn_col3:
                    current_action = st.session_state.get(action_key, "MANUAL")
                    if current_action == "ACCEPTED_AI_DRAFT":
                        st.success("✅ AI Draft accepted for this field")
                    elif current_action == "REJECTED_AI_DRAFT":
                        st.error("❌ AI Draft rejected (using manual entry)")

        with st.expander(f"Legal guidance for {did}", expanded=False):
            st.markdown(DECL_REVIEW_GUIDANCE[did])

        r_col1, r_col2, r_col3 = st.columns([1, 1, 1.5])

        with r_col1:
            default_lbl_idx = DECLARATION_LABELS.index(orig_lbl) if orig_lbl in DECLARATION_LABELS else 0
            revised_label = st.selectbox(
                f"Revised Label ({did})",
                DECLARATION_LABELS,
                index=default_lbl_idx,
                key=lbl_key,
            )

        with r_col2:
            revised_source = st.selectbox(
                f"Evidence Source ({did})",
                EVIDENCE_SOURCES,
                index=0,
                key=src_key,
            )

        with r_col3:
            revised_notes = st.text_input(
                f"Evidence Notes ({did})",
                key=notes_key,
                placeholder="What evidence supports this label?",
            )

        # Detect manual modifications
        current_action = st.session_state.get(action_key, "MANUAL")
        if has_draft and draft_item:
            if current_action == "ACCEPTED_AI_DRAFT" and (
                revised_label != draft_item.get("draft_label") or revised_source != draft_item.get("evidence_source")
            ):
                current_action = "MODIFIED_AI_DRAFT"
                st.session_state[action_key] = "MODIFIED_AI_DRAFT"

        revised_declarations[did] = {
            "original_label": orig_lbl,
            "revised_label": revised_label,
            "changed": revised_label != orig_lbl,
            "evidence_source": revised_source,
            "evidence_notes": revised_notes,
            "review_action": current_action,
        }

    st.divider()

    # ------------------------------------------------------------------
    # SECTION 5: Overall Compliance Assessment
    # ------------------------------------------------------------------
    st.header("📊 Revised Overall Assessment")

    # Declaration Summary Cards
    st.caption("Summary of revised declaration labels:")
    sum_cols = st.columns(7)
    for i, did in enumerate(DECLS):
        with sum_cols[i]:
            rev = revised_declarations[did]
            changed_marker = " ✏️" if rev["changed"] else ""
            st.markdown(f"**{did}**{changed_marker}")
            st.markdown(f"{LABEL_EMOJI.get(rev['revised_label'], '❓')} `{rev['revised_label']}`")
            st.caption(f"Action: `{rev['review_action']}`")

    # AI Suggested Overall Box
    if current_draft and current_draft.get("draft_generated"):
        sug_ov = current_draft.get("suggested_overall", {})
        st.info(
            f"🤖 **AI Suggested Overall**: `{sug_ov.get('suggested_label', 'UNCERTAIN')}`\n\n"
            f"💡 **AI Reasoning**: {sug_ov.get('reasoning', 'No reasoning provided')}"
        )

    orig_overall = str(row.get("original_overall", "NON_COMPLIANT"))
    default_overall_idx = OVERALL_LABELS.index(orig_overall) if orig_overall in OVERALL_LABELS else 0

    ov_col1, ov_col2 = st.columns([1, 1])

    with ov_col1:
        revised_overall = st.selectbox(
            "Human Verified Overall Label",
            OVERALL_LABELS,
            index=default_overall_idx,
            key=f"rev_overall_{current_pid}",
            help="Select the final human-verified overall compliance label.",
        )

    with ov_col2:
        reviewer_decision = st.selectbox(
            "Reviewer Decision",
            ["CONFIRMED", "REVISED", "NEEDS_MORE_EVIDENCE"],
            help="CONFIRMED = original labels were accurate. REVISED = one or more labels changed. NEEDS_MORE_EVIDENCE = incomplete listing.",
        )

    review_reason = st.text_area(
        "Review Reason / Justification",
        placeholder="Explain why declarations or overall label were confirmed or revised.",
    )

    reviewer_notes = st.text_input(
        "Additional Reviewer Notes",
        placeholder="Any edge cases, observations, or packaging peculiarities.",
    )

    review_confidence = st.select_slider(
        "Review Confidence",
        options=["LOW", "MEDIUM", "HIGH"],
        value="HIGH",
    )

    st.divider()

    # ------------------------------------------------------------------
    # SECTION 6: Save & Submit Review
    # ------------------------------------------------------------------
    if st.button("💾 Save Human-Verified Review", type="primary", use_container_width=True):
        # Determine overall action
        overall_action = st.session_state.get(f"overall_action_{current_pid}", "MANUAL")
        if current_draft and current_draft.get("draft_generated"):
            ai_sug = current_draft.get("suggested_overall", {}).get("suggested_label")
            if revised_overall == ai_sug and overall_action == "ACCEPTED_AI_DRAFT":
                overall_action = "ACCEPTED_AI_DRAFT"
            elif revised_overall != ai_sug and overall_action != "MANUAL":
                overall_action = "MODIFIED_AI_DRAFT"

        # Count declaration changes
        declarations_changed = sum(1 for rev in revised_declarations.values() if rev["changed"])

        # Construct comprehensive review record
        review_record = {
            "review_id": f"review_{current_pid}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            "product_id": current_pid,
            "reviewer_id": reviewer_id,
            "review_timestamp": datetime.now(timezone.utc).isoformat(),
            "review_phase": "4.5B",
            "reviewed_by_human": True,
            "human_confirmed": True,

            # 1. Original Baseline (Read-Only)
            "original_overall": orig_overall,
            "original_declarations": {
                did: str(row.get(f"original_{did}", "")) for did in DECLS
            },

            # 2. Machine Draft & History (Stored separately, NOT ground truth)
            "ai_draft": current_draft if (current_draft and current_draft.get("draft_generated")) else {
                "draft_generated": False,
                "model_name": None,
                "draft_timestamp": None,
                "declarations": {},
                "suggested_overall": {},
            },
            "ai_draft_history": draft_history if draft_history else [],

            # 3. Human Verified Values
            "revised_overall": revised_overall,
            "overall_review_action": overall_action,
            "revised_declarations": revised_declarations,

            # 4. Audit & Change Tracking
            "reviewer_decision": reviewer_decision,
            "review_reason": review_reason,
            "reviewer_notes": reviewer_notes,
            "review_confidence": review_confidence,
            "overall_changed": revised_overall != orig_overall,
            "declarations_changed_count": declarations_changed,
        }

        save_review(review_record)
        st.success(f"✅ Human-verified review saved successfully for product `{current_pid}`!")
        st.balloons()
        st.cache_data.clear()



if __name__ == "__main__":
    main()
