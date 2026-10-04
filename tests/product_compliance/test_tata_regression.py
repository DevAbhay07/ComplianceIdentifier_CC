"""
test_tata_regression.py
=======================
Comprehensive test suite verifying the resolution of the Tata 1mg Creatine regression:
1. D01 role semantics (Manufacturer + Marketer distinct roles do NOT conflict).
2. D01 corporate token normalization and formatting variation reconciliation.
3. D05 statutory MRP prioritization over unit sale prices and incidental numbers.
4. Multi-image fusion corroboration vs conflict detection.
5. Image order invariance ([front, back, side] == [side, front, back]).
6. Duplicate image invariance ([front, back] == [front, back, back]).
7. Actual statutory value preservation invariant (no silent ₹0.00 corruption).
8. Determinism audit across 5 repeated runs.
"""

import pytest
from src.extraction.declaration_schema import (
    DeclarationEvidence,
    DeclarationStatus,
    EvidenceSource,
    ProductEvidenceRecord
)
from src.extraction.normalization import (
    normalize_manufacturer_value,
    are_manufacturers_conflicting,
    normalize_price_value,
    are_prices_conflicting,
    normalize_quantity_value,
    are_quantities_conflicting,
    normalize_date_value,
    normalize_country_value,
    normalize_consumer_care_value,
    normalize_generic_name_value,
    extract_core_company_tokens
)
from src.extraction.declaration_extractor import DeclarationExtractor
from src.extraction.evidence_fusion import EvidenceFusionEngine
from src.inference.product_compliance_pipeline import ProductCompliancePipeline


# =========================================================================
# 1. D01 ROLE & FORMATTING TESTS
# =========================================================================

def test_d01_manufacturer_and_marketer_no_conflict():
    """Rule 6(1)(a): Manufacturer A + Marketer B are distinct roles and must NOT conflict."""
    m1 = normalize_manufacturer_value(
        role_entities={"MARKETER": {"name": "TATA 1MG Healthcare Solutions Pvt. Ltd.", "address_markers": ["Delhi"], "pincode": "110020"}}
    )
    m2 = normalize_manufacturer_value(
        role_entities={"MANUFACTURER": {"name": "BACFO Pharmaceuticals (India) Limited", "address_markers": ["Noida"], "pincode": "201305"}}
    )
    assert not are_manufacturers_conflicting(m1, m2), "Manufacturer and Marketer should NOT conflict!"


def test_d01_identical_manufacturer_no_conflict():
    """Same manufacturer across images must corroborate with no conflict."""
    m1 = normalize_manufacturer_value(
        entity_name="BACFO Pharmaceuticals (India) Limited",
        role_entities={"MANUFACTURER": {"name": "BACFO Pharmaceuticals (India) Limited", "pincode": "201305"}}
    )
    m2 = normalize_manufacturer_value(
        entity_name="BACFO Pharmaceuticals (India) Limited",
        role_entities={"MANUFACTURER": {"name": "BACFO Pharmaceuticals (India) Limited", "pincode": "201305"}}
    )
    assert not are_manufacturers_conflicting(m1, m2), "Identical manufacturer should NOT conflict!"


def test_d01_conflicting_manufacturer_same_role():
    """Different manufacturers for the same role with no overlap must be flagged as conflict."""
    m1 = normalize_manufacturer_value(
        role_entities={"MANUFACTURER": {"name": "Company Alpha Labs Limited", "pincode": "110001"}}
    )
    m2 = normalize_manufacturer_value(
        role_entities={"MANUFACTURER": {"name": "Company Beta Pharma Limited", "pincode": "400001"}}
    )
    assert are_manufacturers_conflicting(m1, m2), "Contradictory manufacturers on the same role MUST conflict!"


def test_d01_manufacturer_and_packer_no_conflict():
    """Manufacturer A + Packer B are distinct roles and must NOT automatically conflict."""
    m1 = normalize_manufacturer_value(
        role_entities={"MANUFACTURER": {"name": "Company Alpha Limited"}}
    )
    m2 = normalize_manufacturer_value(
        role_entities={"PACKER": {"name": "Company Beta Packaging Solutions Pvt. Ltd."}}
    )
    assert not are_manufacturers_conflicting(m1, m2), "Manufacturer and Packer should NOT conflict!"


def test_d01_manufacturer_and_importer_no_conflict():
    """Manufacturer A + Importer B are distinct roles and must NOT automatically conflict."""
    m1 = normalize_manufacturer_value(
        role_entities={"MANUFACTURER": {"name": "Global Wellness Inc."}}
    )
    m2 = normalize_manufacturer_value(
        role_entities={"IMPORTER": {"name": "India Import Solutions Pvt. Ltd."}}
    )
    assert not are_manufacturers_conflicting(m1, m2), "Manufacturer and Importer should NOT conflict!"


def test_d01_formatting_variations_no_conflict():
    """Safe formatting variations of the same legal entity must normalize without conflict."""
    name1 = "BACFO Pharmaceuticals (India) Limited"
    name2 = "BACFO Pharmaceuticals India Ltd."
    m1 = normalize_manufacturer_value(
        role_entities={"MANUFACTURER": {"name": name1, "pincode": "201305"}}
    )
    m2 = normalize_manufacturer_value(
        role_entities={"MANUFACTURER": {"name": name2, "pincode": "201305"}}
    )
    assert not are_manufacturers_conflicting(m1, m2), f"Formatting variations {name1} vs {name2} should not conflict!"


# =========================================================================
# 2. D05 MRP & PRICE EXTRACTION TESTS
# =========================================================================

def test_d05_extraction_explicit_mrp_variants():
    """Extracts MRP across statutory formats."""
    extractor = DeclarationExtractor()

    # 1. MRP ₹1063
    d1 = extractor.extract_from_text_and_vision(raw_ocr_text="MRP ₹1063")
    assert d1["D05"].status == DeclarationStatus.PRESENT.value
    assert d1["D05"].normalized_value["mrp"] == 1063.0

    # 2. MRP ₹1063.00
    d2 = extractor.extract_from_text_and_vision(raw_ocr_text="MRP ₹1063.00")
    assert d2["D05"].status == DeclarationStatus.PRESENT.value
    assert d2["D05"].normalized_value["mrp"] == 1063.0

    # 3. MRP (Incl. of all taxes): ₹1063.00
    d3 = extractor.extract_from_text_and_vision(raw_ocr_text="MRP (Incl. of all taxes): ₹1063.00")
    assert d3["D05"].status == DeclarationStatus.PRESENT.value
    assert d3["D05"].normalized_value["mrp"] == 1063.0
    assert d3["D05"].normalized_value["tax_inclusive"] is True


def test_d05_prioritizes_mrp_over_unit_sale_price():
    """MRP must be selected over Unit Sale Price ₹4.25/g."""
    extractor = DeclarationExtractor()
    text = "MRP ₹1063.00 inclusive of all taxes\nUnit Sale Price ₹4.25/g"
    decls = extractor.extract_from_text_and_vision(raw_ocr_text=text)
    assert decls["D05"].status == DeclarationStatus.PRESENT.value
    assert decls["D05"].normalized_value["mrp"] == 1063.0, f"Expected 1063.0, got {decls['D05'].normalized_value['mrp']}"


def test_d05_prioritizes_mrp_over_unrelated_prices():
    """Explicit MRP must be selected when unrelated prices exist in text."""
    extractor = DeclarationExtractor()
    text = "MRP ₹1063\nanother unrelated price ₹999"
    decls = extractor.extract_from_text_and_vision(raw_ocr_text=text)
    assert decls["D05"].status == DeclarationStatus.PRESENT.value
    assert decls["D05"].normalized_value["mrp"] == 1063.0


def test_d05_fusion_corroboration_vs_conflict():
    """₹1063 and ₹1063.00 corroborate; ₹1063 and ₹999 conflict."""
    p1 = normalize_price_value("1063")
    p2 = normalize_price_value("1063.00")
    p3 = normalize_price_value("999.00")

    assert not are_prices_conflicting(p1, p2), "₹1063 and ₹1063.00 must corroborate!"
    assert are_prices_conflicting(p1, p3), "₹1063 and ₹999.00 must conflict!"


# =========================================================================
# 3. MULTI-IMAGE FUSION & INVARIANCE TESTS
# =========================================================================

def _build_tata_mock_views():
    """Builds synthetic front, back, and side packaging views for Tata 1mg Creatine."""
    extractor = DeclarationExtractor()
    
    front_text = "Tata 1mg Creatine Monohydrate Net Quantity: 250 g"
    back_text = """
    Marketed By:
    TATA 1MG Healthcare Solutions Pvt. Ltd.
    B-16, Okhla Industrial Area, Phase-II, New Delhi-110020

    Manufactured By:
    BACFO Pharmaceuticals (India) Limited
    C-46, Phase-II, Noida, Distt. Gautam Buddha Nagar, U.P. - 201305

    Mfg Date: 02/2026
    MRP ₹1063.00 inclusive of all taxes
    Unit Sale Price ₹4.25/g
    Consumer Care: care@1mg.com, +91-995-8932200
    Country of Origin: India
    """
    side_text = "250 g Net Weight\nMRP ₹1063.00"

    v_front = {
        "image_name": "front.png",
        "declarations": extractor.extract_from_text_and_vision(raw_ocr_text=front_text, image_name="front.png"),
        "ocr_text": front_text,
        "ocr_confidence": 0.95,
        "visual_regions": []
    }
    v_back = {
        "image_name": "back.png",
        "declarations": extractor.extract_from_text_and_vision(raw_ocr_text=back_text, image_name="back.png"),
        "ocr_text": back_text,
        "ocr_confidence": 0.95,
        "visual_regions": []
    }
    v_side = {
        "image_name": "side.png",
        "declarations": extractor.extract_from_text_and_vision(raw_ocr_text=side_text, image_name="side.png"),
        "ocr_text": side_text,
        "ocr_confidence": 0.90,
        "visual_regions": []
    }
    return v_front, v_back, v_side


def test_image_order_invariance():
    """[front, back, side] and [side, front, back] must produce equivalent fused evidence."""
    engine = EvidenceFusionEngine()
    v_front, v_back, v_side = _build_tata_mock_views()

    fused_1 = engine.fuse_image_evidences([v_front, v_back, v_side], product_id="tata_order_1")
    fused_2 = engine.fuse_image_evidences([v_side, v_front, v_back], product_id="tata_order_2")

    for did in ["D01", "D02", "D03", "D04", "D05", "D06", "D07"]:
        assert fused_1.declarations[did].status == fused_2.declarations[did].status == "PRESENT"
    
    assert fused_1.declarations["D05"].normalized_value["mrp"] == fused_2.declarations["D05"].normalized_value["mrp"] == 1063.0
    assert fused_1.declarations["D03"].normalized_value["base_value"] == fused_2.declarations["D03"].normalized_value["base_value"] == 250.0
    assert not fused_1.conflicting_fields
    assert not fused_2.conflicting_fields


def test_duplicate_image_invariance():
    """[front, back] versus [front, back, back] must not introduce false conflicts."""
    engine = EvidenceFusionEngine()
    v_front, v_back, _ = _build_tata_mock_views()

    fused_clean = engine.fuse_image_evidences([v_front, v_back], product_id="tata_clean")
    fused_dup = engine.fuse_image_evidences([v_front, v_back, v_back], product_id="tata_dup")

    assert not fused_clean.conflicting_fields
    assert not fused_dup.conflicting_fields
    assert fused_dup.declarations["D01"].status == "PRESENT"
    assert fused_dup.declarations["D05"].status == "PRESENT"
    assert fused_dup.declarations["D05"].normalized_value["mrp"] == 1063.0


# =========================================================================
# 4. ACTUAL VALUE INVARIANT & REPEATED 5-RUN DETERMINISM AUDIT
# =========================================================================

def test_actual_value_invariant_and_determinism_5_runs():
    """
    Runs the complete Tata 1mg packaging inference pipeline 5 times repeatedly.
    Verifies that actual values (MRP=1063.0, Qty=250g) are preserved end-to-end
    and results are 100% deterministic across all 5 runs.
    """
    pipeline = ProductCompliancePipeline(use_mock_ocr_fallback=True)
    v_front, v_back, v_side = _build_tata_mock_views()

    combined_text = f"{v_front['ocr_text']}\n{v_back['ocr_text']}\n{v_side['ocr_text']}"

    results = []
    for run_idx in range(5):
        res = pipeline.analyze_product(
            image_paths=[],
            product_id=f"tata_run_{run_idx+1}",
            product_title="Tata 1mg Creatine Monohydrate",
            offline_reference_text=combined_text
        )
        results.append(res)

    # Validate Run 1 against statutory expectations
    r0 = results[0]
    decls = r0["declarations"]

    assert decls["D01"]["status"] == "PRESENT", "D01 must be PRESENT"
    assert "D01" not in r0["evidence_summary"]["conflicts_detected"], "D01 must not conflict"
    assert decls["D03"]["status"] == "PRESENT", "D03 must be PRESENT"
    assert decls["D04"]["status"] == "PRESENT", "D04 must be PRESENT"
    assert decls["D05"]["status"] == "PRESENT", "D05 must be PRESENT"
    assert decls["D06"]["status"] == "PRESENT", "D06 must be PRESENT"
    assert decls["D07"]["status"] == "PRESENT", "D07 must be PRESENT"

    # Actual value invariant: MRP must be ₹1063.00, NOT ₹0.00
    d05_norm = decls["D05"]["normalized_value"]
    assert d05_norm["mrp"] == 1063.0, f"D05 MRP must be 1063.0, got {d05_norm.get('mrp')}"
    assert "1063" in decls["D05"]["raw_value"] or "1063" in decls["D05"]["evidence_text"]

    # Final verdict must be COMPLIANT (all mandatory present, 0 conflicts)
    assert r0["overall_prediction"] == "COMPLIANT"
    assert not r0["abstention_applied"]

    # Verify 100% Determinism across all 5 runs
    for run_i, r in enumerate(results[1:], start=2):
        assert r["overall_prediction"] == r0["overall_prediction"], f"Run {run_i} prediction diverged!"
        assert r["confidence"] == r0["confidence"], f"Run {run_i} confidence diverged!"
        assert r["evidence_summary"]["conflicts_detected"] == r0["evidence_summary"]["conflicts_detected"]
        for did in ["D01", "D02", "D03", "D04", "D05", "D06", "D07"]:
            assert r["declarations"][did]["status"] == r0["declarations"][did]["status"], f"Run {run_i} {did} status diverged!"
            assert r["declarations"][did]["normalized_value"] == r0["declarations"][did]["normalized_value"], f"Run {run_i} {did} normalized value diverged!"
