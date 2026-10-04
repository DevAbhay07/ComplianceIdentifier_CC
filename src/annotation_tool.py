"""
annotation_tool.py
====================
Streamlit-based human annotation tool for NitiMark compliance labeling.

Displays one product at a time and allows the annotator to:
  - Review product metadata (listing title, description, specs, price, brand)
  - Generate machine draft recommendations (semi-automated)
  - Label each legal declaration (PRESENT/ABSENT/PARTIAL/NOT_APPLICABLE/CANNOT_DETERMINE)
  - Specify evidence sources and notes
  - Assign an overall compliance label (COMPLIANT/NON_COMPLIANT/UNCERTAIN)
  - Save annotations as JSONL (append-only, with product snapshots)

Usage:
    streamlit run src/annotation_tool.py

CRITICAL: Machine-generated draft labels are NEVER ground truth.
The human annotator must review and explicitly save all annotations.
"""

import streamlit as st
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path so we can import from src/data/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.generate_annotation_draft import generate_draft

# -------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------
PILOT_DATA_PATH = "data/interim/parsed_annotation_candidates.jsonl"
PILOT_MANIFEST_PATH = "annotations/pilot_manifest.csv"
ANNOTATIONS_OUTPUT_PATH = "annotations/pilot_annotations.jsonl"

DECLARATION_LABELS = ["PRESENT", "ABSENT", "PARTIAL", "NOT_APPLICABLE", "CANNOT_DETERMINE"]
OVERALL_LABELS = ["COMPLIANT", "NON_COMPLIANT", "UNCERTAIN"]

EVIDENCE_SOURCES = [
    "LISTING_TITLE",
    "DESCRIPTION",
    "SPECIFICATION",
    "PRICE_FIELD",
    "BRAND_FIELD",
    "IMAGE",
    "OTHER",
    "NONE",
]

# The 7 mandatory declarations from the legal requirements schema
DECLARATIONS = {
    "D01": {
        "id": "D01",
        "name": "Manufacturer/Packer/Importer Name & Address",
        "rule": "Rule 6(1)(a), Rule 10",
        "guidance": (
            "Check for manufacturer name AND complete address. "
            "Brand name alone is PARTIAL (name but no address). "
            "Look in: brand field, description, product specifications."
        ),
    },
    "D02": {
        "id": "D02",
        "name": "Common/Generic Name of Commodity",
        "rule": "Rule 6(1)(b)",
        "guidance": (
            "Check whether the product's common/generic name is clearly identifiable. "
            "Usually present in the product title. "
            "A purely brand-name title without generic name is PARTIAL."
        ),
    },
    "D03": {
        "id": "D03",
        "name": "Net Quantity (Weight/Volume/Number)",
        "rule": "Rule 6(1)(c), Rules 11-13",
        "guidance": (
            "Check for net weight, volume, or number of items. "
            "Look in: product name, specifications (Weight, Pack of, Number of Contents). "
            "Must be in SI units."
        ),
    },
    "D04": {
        "id": "D04",
        "name": "Month & Year of Manufacture/Packing/Import",
        "rule": "Rule 6(1)(d)",
        "guidance": (
            "Check for manufacturing or packing date. "
            "This is ALMOST NEVER present in e-commerce listings. "
            "CANNOT_DETERMINE is the expected answer for most products."
        ),
    },
    "D05": {
        "id": "D05",
        "name": "Maximum Retail Price (MRP)",
        "rule": "Rule 6(1)(e)",
        "guidance": (
            "Check for retail_price field. "
            "Note: the listing price may be discounted price, not MRP. "
            "If retail_price exists and is labelled as MRP, mark PRESENT. "
            "If only discounted_price exists, mark PARTIAL."
        ),
    },
    "D06": {
        "id": "D06",
        "name": "Consumer Care Contact Details",
        "rule": "Rule 6(2)",
        "guidance": (
            "Check for customer care phone number, email, or address. "
            "Look in: description, specifications. "
            "This is RARELY present in e-commerce listings."
        ),
    },
    "D07": {
        "id": "D07",
        "name": "Country of Origin (Imported Products)",
        "rule": "Rule 6(1)(a)",
        "guidance": (
            "Check specifications for 'Country of Origin' or 'Imported by'. "
            "If the product appears to be domestic and no import info exists, "
            "NOT_APPLICABLE may be appropriate. "
            "If imported but no country stated, mark ABSENT."
        ),
    },
}

ANNOTATION_SCHEMA_VERSION = "1.1.0"  # Bumped for draft integration


# -------------------------------------------------------------------------
# Data Loading
# -------------------------------------------------------------------------
@st.cache_data
def load_pilot_products():
    """Load pilot manifest product IDs and the full parsed product data."""
    import pandas as pd

    # Load manifest to get pilot product IDs
    manifest = pd.read_csv(PILOT_MANIFEST_PATH)
    pilot_ids = set(manifest["product_id"].astype(str).tolist())

    # Load full parsed data
    products = {}
    with open(PILOT_DATA_PATH, "r") as f:
        for line in f:
            record = json.loads(line)
            pid = str(record.get("product_id", ""))
            if pid in pilot_ids:
                products[pid] = record

    # Maintain manifest order
    ordered_ids = [str(pid) for pid in manifest["product_id"]]
    return ordered_ids, products, manifest


def load_existing_annotations():
    """Load previously saved annotations to track progress."""
    annotations = {}
    if os.path.exists(ANNOTATIONS_OUTPUT_PATH):
        with open(ANNOTATIONS_OUTPUT_PATH, "r") as f:
            for line in f:
                ann = json.loads(line)
                pid = ann.get("product_id", "")
                annotations[pid] = ann
    return annotations


def save_annotation(annotation: dict):
    """Append an annotation to the JSONL file (append-only, never overwrites)."""
    os.makedirs(os.path.dirname(ANNOTATIONS_OUTPUT_PATH), exist_ok=True)
    with open(ANNOTATIONS_OUTPUT_PATH, "a") as f:
        f.write(json.dumps(annotation, ensure_ascii=False) + "\n")


# -------------------------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="NitiMark Annotation Tool",
        page_icon="📋",
        layout="wide",
    )

    st.title("📋 NitiMark — Compliance Annotation Tool")
    st.caption("Human-reviewed ground-truth annotation for Legal Metrology (PC) Rules, 2011")

    # ------------------------------------------------------------------
    # Initialize session state for draft prefill
    # ------------------------------------------------------------------
    if "draft" not in st.session_state:
        st.session_state.draft = None
    if "draft_product_id" not in st.session_state:
        st.session_state.draft_product_id = None

    # ------------------------------------------------------------------
    # Sidebar: Annotator config and navigation
    # ------------------------------------------------------------------
    with st.sidebar:
        st.header("⚙️ Annotator Settings")
        annotator_id = st.text_input("Annotator ID", value="annotator_01")

        st.divider()
        st.header("📊 Progress")

        ordered_ids, products, manifest = load_pilot_products()
        existing = load_existing_annotations()

        annotated_count = len(existing)
        total_count = len(ordered_ids)

        st.progress(annotated_count / total_count if total_count > 0 else 0)
        st.write(f"**{annotated_count} / {total_count}** products annotated")

        st.divider()
        st.header("🔎 Navigate")

        # Navigation options
        nav_option = st.radio(
            "Show products:",
            ["All", "Not annotated", "Already annotated"],
        )

        if nav_option == "Not annotated":
            visible_ids = [pid for pid in ordered_ids if pid not in existing]
        elif nav_option == "Already annotated":
            visible_ids = [pid for pid in ordered_ids if pid in existing]
        else:
            visible_ids = ordered_ids

        if not visible_ids:
            st.warning("No products match this filter.")
            return

        # Product selector
        product_options = {
            f"#{i+1} | {products[pid]['product_name'][:50]}": pid
            for i, pid in enumerate(visible_ids)
            if pid in products
        }

        selected_label = st.selectbox(
            "Select product:",
            list(product_options.keys()),
        )
        current_pid = product_options[selected_label]

    # Clear draft if product changed
    if st.session_state.draft_product_id != current_pid:
        st.session_state.draft = None
        st.session_state.draft_product_id = current_pid

    # ------------------------------------------------------------------
    # Main area: Product display and annotation
    # ------------------------------------------------------------------
    product = products.get(current_pid)
    if not product:
        st.error(f"Product {current_pid} not found in parsed data.")
        return

    # Check if already annotated
    is_reannotation = current_pid in existing

    if is_reannotation:
        st.info(
            "⚠️ This product has already been annotated. "
            "Submitting again will add a NEW annotation record (previous ones are preserved)."
        )

    # ------------------------------------------------------------------
    # SECTION 1: Available Evidence (read-only)
    # ------------------------------------------------------------------
    st.header("📦 Available Evidence")
    st.caption("This section shows the raw product information from the Flipkart listing. Review this evidence before annotating.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Listing Metadata")
        st.markdown(f"**Product ID:** `{product.get('product_id', 'N/A')}`")
        st.markdown(f"**Product Name:** {product.get('product_name', 'N/A')}")
        st.markdown(f"**Category:** {product.get('top_category', 'N/A')}")
        st.markdown(f"**Full Category Tree:** {product.get('product_category_tree', 'N/A')}")
        st.markdown(f"**Brand:** {product.get('brand', '') or '❌ *Missing*'}")
        st.markdown(f"**Retail Price:** ₹{product.get('retail_price', 'N/A')}")
        st.markdown(f"**Discounted Price:** ₹{product.get('discounted_price', 'N/A')}")

    with col2:
        st.subheader("Product Specifications")
        specs = product.get("parsed_specs", {})
        if specs:
            for key, value in specs.items():
                st.markdown(f"- **{key}:** {value}")
        else:
            st.warning("No specifications available")

        extras = product.get("extra_spec_values", [])
        if extras:
            st.subheader("Additional Spec Values")
            for v in extras:
                st.markdown(f"- {v}")

    # Description (can be long, use expander)
    with st.expander("📝 Full Description", expanded=False):
        desc = product.get("description", "")
        if desc:
            st.text(desc[:3000])
        else:
            st.warning("No description available")

    st.divider()

    # ------------------------------------------------------------------
    # SECTION 2: Draft Generation Controls
    # ------------------------------------------------------------------
    st.header("⚡ Semi-Automated Draft")

    draft_col1, draft_col2, draft_col3 = st.columns(3)

    with draft_col1:
        if st.button("⚡ Generate Draft Annotation", type="secondary", use_container_width=True):
            st.session_state.draft = generate_draft(product)
            st.session_state.draft_product_id = current_pid
            st.rerun()

    with draft_col2:
        if st.button("✓ Accept All Drafts", use_container_width=True, disabled=(st.session_state.draft is None)):
            # This just keeps the draft in session state — the form
            # fields below read from it. It does NOT save anything.
            pass

    with draft_col3:
        if st.button("↺ Reset Draft", use_container_width=True, disabled=(st.session_state.draft is None)):
            st.session_state.draft = None
            st.rerun()

    draft = st.session_state.draft

    if draft is not None:
        st.warning(
            "⚠️ **MACHINE-GENERATED DRAFT RECOMMENDATIONS** — "
            "Review and correct every label before saving. "
            "These are NOT ground truth. They are pre-filled suggestions only."
        )

    st.divider()

    # ------------------------------------------------------------------
    # SECTION 3: Annotator Decision (THIS IS THE HUMAN ANNOTATION)
    # ------------------------------------------------------------------
    st.header("✍️ Annotator Decision")
    st.caption("YOUR decisions below form the ground-truth annotation. Review the evidence above, then label each declaration.")

    # Per-declaration annotation
    declaration_annotations = {}

    for decl_id, decl_info in DECLARATIONS.items():
        st.subheader(f"{decl_id}: {decl_info['name']}")
        st.caption(f"*Legal Rule: {decl_info['rule']}*")

        # Show draft info if available
        draft_decl = None
        if draft is not None:
            draft_decl = draft.get("declarations", {}).get(decl_id)

        if draft_decl:
            with st.expander(f"🤖 Machine Draft for {decl_id}", expanded=True):
                dcol1, dcol2 = st.columns([1, 2])
                with dcol1:
                    label_emoji = {
                        "PRESENT": "🟢", "ABSENT": "🔴", "PARTIAL": "🟡",
                        "NOT_APPLICABLE": "⚪", "CANNOT_DETERMINE": "🔵",
                    }
                    st.markdown(f"**Draft Label:** {label_emoji.get(draft_decl['label'], '❓')} `{draft_decl['label']}`")
                    st.markdown(f"**Sources:** {', '.join(draft_decl.get('evidence_sources', []))}")
                with dcol2:
                    st.markdown(f"**Reasoning:** {draft_decl.get('reasoning', '')}")
                    if draft_decl.get('evidence_text'):
                        st.markdown(f"**Evidence:** {draft_decl['evidence_text'][:200]}")

        with st.expander(f"Annotation guidance for {decl_id}"):
            st.write(decl_info["guidance"])

        # Determine default values (from draft if available, else defaults)
        if draft_decl:
            default_label_idx = DECLARATION_LABELS.index(draft_decl["label"]) if draft_decl["label"] in DECLARATION_LABELS else 0
            default_sources = [s for s in draft_decl.get("evidence_sources", []) if s in EVIDENCE_SOURCES]
            default_notes = draft_decl.get("evidence_text", "")
        else:
            default_label_idx = 0
            default_sources = []
            default_notes = ""

        col_label, col_source, col_notes = st.columns([1, 1, 1.5])

        with col_label:
            label = st.selectbox(
                f"Label",
                DECLARATION_LABELS,
                index=default_label_idx,
                key=f"label_{decl_id}",
            )

        with col_source:
            sources = st.multiselect(
                f"Evidence Source",
                EVIDENCE_SOURCES,
                default=default_sources,
                key=f"source_{decl_id}",
            )

        with col_notes:
            evidence_text = st.text_input(
                f"Evidence / Notes",
                value=default_notes,
                key=f"notes_{decl_id}",
                placeholder="What specific text did you find?",
            )

        declaration_annotations[decl_id] = {
            "label": label,
            "evidence_sources": sources,
            "evidence_text": evidence_text,
        }

    st.divider()

    # ------------------------------------------------------------------
    # Overall assessment
    # ------------------------------------------------------------------
    st.header("📊 Overall Compliance Assessment")

    # Show summary of per-declaration labels
    st.caption("Summary of your declaration labels:")
    summary_cols = st.columns(7)
    for i, (decl_id, ann) in enumerate(declaration_annotations.items()):
        with summary_cols[i]:
            label_color = {
                "PRESENT": "🟢",
                "ABSENT": "🔴",
                "PARTIAL": "🟡",
                "NOT_APPLICABLE": "⚪",
                "CANNOT_DETERMINE": "🔵",
            }
            st.markdown(f"**{decl_id}**")
            st.markdown(f"{label_color.get(ann['label'], '❓')} {ann['label']}")

    # Determine overall defaults from draft
    draft_overall = draft.get("overall_assessment", {}) if draft else {}
    default_overall_idx = OVERALL_LABELS.index(draft_overall["label"]) if draft_overall.get("label") in OVERALL_LABELS else 0
    default_overall_notes = draft_overall.get("notes", "")
    default_confidence = draft_overall.get("confidence", "MEDIUM")

    if draft_overall:
        st.info(f"🤖 **Machine draft overall:** `{draft_overall.get('label', 'N/A')}` — {draft_overall.get('reasoning', '')}")

    overall_label = st.selectbox(
        "Overall Compliance Label",
        OVERALL_LABELS,
        index=default_overall_idx,
        help="COMPLIANT = all applicable declarations PRESENT. "
             "NON_COMPLIANT = one or more ABSENT. "
             "UNCERTAIN = some PARTIAL/CANNOT_DETERMINE, none clearly ABSENT.",
    )

    overall_notes = st.text_area(
        "Overall Notes",
        value=default_overall_notes,
        placeholder="Explain your overall assessment. Why did you choose this label?",
    )

    confidence = st.select_slider(
        "Annotation Confidence",
        options=["LOW", "MEDIUM", "HIGH"],
        value=default_confidence,
        help="How confident are you in this annotation given the available evidence?",
    )

    st.divider()

    # ------------------------------------------------------------------
    # Submit
    # ------------------------------------------------------------------
    if st.button("💾 Save Annotation", type="primary", use_container_width=True):
        # Build machine_guidance with modification tracking
        machine_guidance = {}

        if draft:
            machine_guidance["draft_generated"] = True
            machine_guidance["draft_timestamp"] = draft.get("draft_timestamp", "")

            for decl_id in DECLARATIONS:
                draft_decl = draft.get("declarations", {}).get(decl_id, {})
                final_label = declaration_annotations[decl_id]["label"]
                machine_label = draft_decl.get("label", "")

                machine_guidance[decl_id] = {
                    "machine_label": machine_label,
                    "machine_sources": draft_decl.get("evidence_sources", []),
                    "machine_evidence": draft_decl.get("evidence_text", ""),
                    "machine_reasoning": draft_decl.get("reasoning", ""),
                    "final_label": final_label,
                    "human_modified": final_label != machine_label,
                }

            # Overall tracking
            draft_overall_label = draft.get("overall_assessment", {}).get("label", "")
            machine_guidance["overall"] = {
                "machine_label": draft_overall_label,
                "final_label": overall_label,
                "human_modified": overall_label != draft_overall_label,
            }
        else:
            machine_guidance["draft_generated"] = False

        # Build the annotation record
        annotation = {
            "annotation_id": f"{current_pid}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            "schema_version": ANNOTATION_SCHEMA_VERSION,
            "product_id": current_pid,
            "annotator_id": annotator_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "product_snapshot": {
                "name": product.get("product_name", ""),
                "category": product.get("top_category", ""),
                "brand": product.get("brand", ""),
                "retail_price": product.get("retail_price"),
                "discounted_price": product.get("discounted_price"),
                "spec_key_count": product.get("spec_key_count", 0),
                "description_length": len(product.get("description", "")),
            },
            "declarations": {},
            "overall_assessment": {
                "label": overall_label,
                "notes": overall_notes,
                "confidence": confidence,
            },
            "machine_guidance": machine_guidance,
        }

        # Add declaration annotations (HUMAN decisions only)
        for decl_id, ann in declaration_annotations.items():
            annotation["declarations"][decl_id] = {
                "label": ann["label"],
                "evidence_sources": ann["evidence_sources"],
                "evidence_text": ann["evidence_text"],
            }

        # Save (append-only)
        save_annotation(annotation)

        st.success(f"✅ Annotation saved for product {current_pid}")
        st.balloons()

        # Clear draft and cache
        st.session_state.draft = None
        st.session_state.draft_product_id = None
        st.cache_data.clear()


if __name__ == "__main__":
    main()
