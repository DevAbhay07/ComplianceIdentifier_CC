"""
test_evidence_fusion_regression.py
==================================
Comprehensive regression tests for Evidence Fusion & Normalization.
Verifies that corroborating observations across multiple images/views are merged without conflict,
while genuine discrepancies are accurately flagged as CONFLICTING_EVIDENCE.
"""

import pytest
from src.extraction.declaration_schema import DeclarationStatus, DeclarationEvidence, EvidenceSource
from src.extraction.declaration_extractor import DeclarationExtractor
from src.extraction.evidence_fusion import EvidenceFusionEngine
from src.extraction.normalization import (
    normalize_quantity_value, are_quantities_conflicting,
    normalize_price_value, are_prices_conflicting,
    normalize_date_value, are_dates_conflicting,
    normalize_country_value, are_countries_conflicting
)


@pytest.fixture
def extractor():
    return DeclarationExtractor()


@pytest.fixture
def fusion():
    return EvidenceFusionEngine()


class TestQuantityNormalizationAndFusion:
    """Covers TEST 1 through TEST 9 for D03 Net Quantity."""

    def test_test1_identical_quantity_no_conflict(self, extractor, fusion):
        """TEST 1: '250 g' and '250 g' -> PRESENT, no conflict."""
        rec1 = extractor.extract_declarations("Net Quantity: 250 g", source_image="image_1")
        rec2 = extractor.extract_declarations("Net Quantity: 250 g", source_image="image_2")
        fused = fusion.fuse_product_evidence([rec1, rec2])

        d03 = fused.declarations["D03"]
        assert d03.status == DeclarationStatus.PRESENT.value
        assert d03.normalized_value["canonical_key"] == "250 g"
        assert len(d03.normalized_value["observations"]) == 2
        assert d03.normalized_value["unique_normalized_values"] == ["250 g"]
        assert "D03" not in fused.conflicting_fields

    def test_test2_formatting_variation_no_conflict(self, extractor, fusion):
        """TEST 2: '250 g' and '250g' -> PRESENT, no conflict."""
        rec1 = extractor.extract_declarations("Net Quantity: 250 g", source_image="image_1")
        rec2 = extractor.extract_declarations("Net Quantity: 250g", source_image="image_2")
        fused = fusion.fuse_product_evidence([rec1, rec2])

        d03 = fused.declarations["D03"]
        assert d03.status == DeclarationStatus.PRESENT.value
        assert d03.normalized_value["canonical_key"] == "250 g"
        assert "D03" not in fused.conflicting_fields

    def test_test3_unit_equivalence_kg_to_g(self, extractor, fusion):
        """TEST 3: '0.25 kg' and '250 g' -> PRESENT, no conflict (canonical conversion)."""
        rec1 = extractor.extract_declarations("Net Quantity: 0.25 kg", source_image="image_1")
        rec2 = extractor.extract_declarations("Net Quantity: 250 g", source_image="image_2")
        fused = fusion.fuse_product_evidence([rec1, rec2])

        d03 = fused.declarations["D03"]
        assert d03.status == DeclarationStatus.PRESENT.value
        assert d03.normalized_value["canonical_key"] == "250 g"
        assert "D03" not in fused.conflicting_fields

    def test_test4_true_quantity_conflict(self, extractor, fusion):
        """TEST 4: '250 g' and '200 g' -> CONFLICTING_EVIDENCE."""
        rec1 = extractor.extract_declarations("Net Quantity: 250 g", source_image="image_1")
        rec2 = extractor.extract_declarations("Net Quantity: 200 g", source_image="image_2")
        fused = fusion.fuse_product_evidence([rec1, rec2])

        d03 = fused.declarations["D03"]
        assert d03.status == DeclarationStatus.CONFLICTING_EVIDENCE.value
        assert "D03" in fused.conflicting_fields
        assert len(d03.conflicting_values) >= 2

    def test_test5_three_images_corroborating_250g(self, extractor, fusion):
        """TEST 5: Three images all saying 250 g -> PRESENT, no conflict, 3 observations."""
        rec1 = extractor.extract_declarations("Net Quantity: 250 g", source_image="front.png")
        rec2 = extractor.extract_declarations("Net Wt: 250 gm", source_image="back.png")
        rec3 = extractor.extract_declarations("Pack size: 250g", source_image="side.png")
        fused = fusion.fuse_product_evidence([rec1, rec2, rec3])

        d03 = fused.declarations["D03"]
        assert d03.status == DeclarationStatus.PRESENT.value
        assert d03.normalized_value["canonical_key"] == "250 g"
        assert len(d03.normalized_value["observations"]) == 3
        assert d03.normalized_value["unique_normalized_values"] == ["250 g"]
        assert "Corroborated across 3 views" in d03.evidence_text

    def test_test6_one_readable_one_unreadable_image(self, extractor, fusion):
        """TEST 6: One image says 250 g, second image has no readable quantity -> PRESENT (no conflict)."""
        rec1 = extractor.extract_declarations("Net Quantity: 250 g", source_image="front.png")
        rec2 = extractor.extract_declarations("Brand Logo and generic pattern text without numbers", source_image="side.png")
        fused = fusion.fuse_product_evidence([rec1, rec2])

        d03 = fused.declarations["D03"]
        assert d03.status == DeclarationStatus.PRESENT.value
        assert d03.normalized_value["canonical_key"] == "250 g"
        assert "D03" not in fused.conflicting_fields

    def test_test7_yolo_zero_ocr_positive(self, extractor):
        """TEST 7: YOLO = 0, OCR = 250 g -> PRESENT."""
        rec = extractor.extract_declarations("Net Quantity: 250 g", visual_regions=[], source_image="front.png")
        d03 = rec.declarations["D03"]
        assert d03.status == DeclarationStatus.PRESENT.value
        assert d03.source == EvidenceSource.OCR.value
        assert d03.normalized_value["canonical_key"] == "250 g"

    def test_test8_yolo_positive_ocr_empty_is_cannot_determine(self, extractor):
        """TEST 8: YOLO detects Net Weight region, OCR returns empty -> CANNOT_DETERMINE."""
        yolo_dets = [{"class": "weight", "class_name": "weight", "confidence": 0.88, "bbox": [10, 10, 50, 50]}]
        rec = extractor.extract_declarations("", visual_regions=yolo_dets, source_image="front.png")
        d03 = rec.declarations["D03"]
        assert d03.status == DeclarationStatus.CANNOT_DETERMINE.value
        assert d03.source == EvidenceSource.VISUAL_YOLO.value

    def test_test9_different_field_names_same_value(self, extractor, fusion):
        """TEST 9: OCR extracts quantity from two different field names but same value -> PRESENT, no conflict."""
        rec1 = extractor.extract_declarations("quantity: 250 g", source_image="image_1")
        rec2 = extractor.extract_declarations("net_quantity: 250 g", source_image="image_2")
        fused = fusion.fuse_product_evidence([rec1, rec2])

        d03 = fused.declarations["D03"]
        assert d03.status == DeclarationStatus.PRESENT.value
        assert d03.normalized_value["canonical_key"] == "250 g"
        assert "D03" not in fused.conflicting_fields


class TestOtherDeclarationsFusionRegression:
    """Covers D01, D02, D04, D05, D06, D07 multi-image fusion."""

    def test_d05_mrp_identical_no_conflict(self, extractor, fusion):
        """D05: ₹999 and ₹999 -> PRESENT, no conflict."""
        rec1 = extractor.extract_declarations("MRP Rs. 999.00 (Incl of all taxes)", source_image="img1")
        rec2 = extractor.extract_declarations("MRP: ₹999 (incl. taxes)", source_image="img2")
        fused = fusion.fuse_product_evidence([rec1, rec2])

        d05 = fused.declarations["D05"]
        assert d05.status == DeclarationStatus.PRESENT.value
        assert d05.normalized_value["mrp"] == 999.0
        assert "D05" not in fused.conflicting_fields

    def test_d05_mrp_differing_is_conflict(self, extractor, fusion):
        """D05: ₹999 and ₹1099 -> CONFLICTING_EVIDENCE."""
        rec1 = extractor.extract_declarations("MRP Rs. 999.00", source_image="img1")
        rec2 = extractor.extract_declarations("MRP Rs. 1099.00", source_image="img2")
        fused = fusion.fuse_product_evidence([rec1, rec2])

        d05 = fused.declarations["D05"]
        assert d05.status == DeclarationStatus.CONFLICTING_EVIDENCE.value
        assert "D05" in fused.conflicting_fields

    def test_d04_date_identical_no_conflict(self, extractor, fusion):
        """D04: 02/2026 and 02/2026 -> PRESENT, no conflict."""
        rec1 = extractor.extract_declarations("Mfd Date: 02/2026", source_image="img1")
        rec2 = extractor.extract_declarations("Mfg: Feb 2026", source_image="img2")
        fused = fusion.fuse_product_evidence([rec1, rec2])

        d04 = fused.declarations["D04"]
        assert d04.status == DeclarationStatus.PRESENT.value
        assert d04.normalized_value["month"] == 2
        assert d04.normalized_value["year"] == 2026
        assert "D04" not in fused.conflicting_fields

    def test_d04_date_differing_is_conflict(self, extractor, fusion):
        """D04: 02/2026 and 05/2026 -> CONFLICTING_EVIDENCE."""
        rec1 = extractor.extract_declarations("Mfd Date: 02/2026", source_image="img1")
        rec2 = extractor.extract_declarations("Mfd Date: 05/2026", source_image="img2")
        fused = fusion.fuse_product_evidence([rec1, rec2])

        d04 = fused.declarations["D04"]
        assert d04.status == DeclarationStatus.CONFLICTING_EVIDENCE.value
        assert "D04" in fused.conflicting_fields

    def test_d07_country_identical_no_conflict(self, extractor, fusion):
        """D07: 'India' and 'Made in India' -> PRESENT, no conflict."""
        rec1 = extractor.extract_declarations("Country of Origin: India", source_image="img1")
        rec2 = extractor.extract_declarations("Made in India", source_image="img2")
        fused = fusion.fuse_product_evidence([rec1, rec2])

        d07 = fused.declarations["D07"]
        assert d07.status == DeclarationStatus.PRESENT.value
        assert d07.normalized_value["canonical_country"] == "india"
        assert "D07" not in fused.conflicting_fields

    def test_d07_country_differing_is_conflict(self, extractor, fusion):
        """D07: 'India' and 'China' -> CONFLICTING_EVIDENCE."""
        rec1 = extractor.extract_declarations("Country of Origin: India", source_image="img1")
        rec2 = extractor.extract_declarations("Country of Origin: China", source_image="img2")
        fused = fusion.fuse_product_evidence([rec1, rec2])

        d07 = fused.declarations["D07"]
        assert d07.status == DeclarationStatus.CONFLICTING_EVIDENCE.value
        assert "D07" in fused.conflicting_fields

    def test_d06_consumer_care_corroboration(self, extractor, fusion):
        """D06: Phone 18001021234 and email care@1mg.com across views -> PRESENT, no conflict."""
        rec1 = extractor.extract_declarations("Call toll free: 1800-102-1234", source_image="img1")
        rec2 = extractor.extract_declarations("Email: care@1mg.com", source_image="img2")
        fused = fusion.fuse_product_evidence([rec1, rec2])

        d06 = fused.declarations["D06"]
        assert d06.status == DeclarationStatus.PRESENT.value
        assert "18001021234" in d06.normalized_value["phone"]
        assert "care@1mg.com" in d06.normalized_value["email"]
        assert "D06" not in fused.conflicting_fields
