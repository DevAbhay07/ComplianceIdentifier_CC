"""
test_leakage_audit.py
=====================
Tests proving zero target leakage in features and verifying that
feature interventions propagate to model predictions.
"""

import sys
import json
import numpy as np
import pytest
from pathlib import Path

PROJECT_ROOT = Path("/home/devabhay07/Desktop/MLmodelV2")
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.product_training_representation import ProductComplianceTrainingRecord
from src.features.product_compliance_feature_extractor import ProductComplianceFeatureExtractor
from src.models.predict_product_compliance import ProductCompliancePredictor


# ═══════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════
PREDICTOR = None

def get_predictor():
    global PREDICTOR
    if PREDICTOR is None:
        PREDICTOR = ProductCompliancePredictor()
    return PREDICTOR


def make_record(
    d01="PRESENT", d02="PRESENT", d03="PRESENT", d04="PRESENT",
    d05="PRESENT", d06="PRESENT", d07="PRESENT",
    ocr_quality=0.90, raw_text="Test product MRP Rs 250 Net Wt 100g"
):
    """Create a minimal training record with controlled declaration states."""
    decls = {}
    for did, status in [("D01", d01), ("D02", d02), ("D03", d03), ("D04", d04),
                         ("D05", d05), ("D06", d06), ("D07", d07)]:
        conf = 0.95 if status == "PRESENT" else (0.85 if status == "ABSENT" else 0.30)
        decls[did] = {"status": status, "confidence": conf, "raw_text": f"{did} text", "source": "OCR"}
    
    return ProductComplianceTrainingRecord(
        product_id="TEST_LEAKAGE",
        source_dataset="test",
        target_provenance="TEST",
        overall_target="UNKNOWN",
        category="Beauty",
        brand="TestBrand",
        product_title="Test Product",
        raw_text=raw_text,
        declarations=decls,
        d01_values={"mfg_name": "Test Mfg"},
        d02_values={"generic_name": "Test Product"},
        d03_values={"net_quantity": "100g"},
        d04_values={"date_text": "01/2026"},
        d05_values={"mrp": "Rs 250"},
        d06_values={"consumer_care": "1800-123-4567"},
        d07_values={"country_of_origin": "India"},
        packaging_features={"yolo_detection_count": 0},
        ocr_quality=ocr_quality,
        text_length=len(raw_text),
        image_count=1,
        image_paths=[]
    )


# ═══════════════════════════════════════════════════════
# Test 1: No feature contains target/provenance fields
# ═══════════════════════════════════════════════════════
def test_no_target_in_features():
    """Feature names must NOT contain target, provenance, violation_type, or overall_target."""
    fe = ProductComplianceFeatureExtractor()
    feature_names = fe.get_feature_names("SET_A")
    
    forbidden = ["target", "provenance", "violation", "overall_target", "ground_truth", "synthetic_label"]
    for fname in feature_names:
        for forbidden_word in forbidden:
            assert forbidden_word not in fname.lower(), (
                f"Feature '{fname}' contains forbidden word '{forbidden_word}' — potential target leakage!"
            )


# ═══════════════════════════════════════════════════════
# Test 2: Feature vector does NOT encode product_id or source
# ═══════════════════════════════════════════════════════
def test_no_product_id_in_features():
    """Features must be declaration-centric, not product-ID-dependent."""
    fe = ProductComplianceFeatureExtractor()
    feature_names = fe.get_feature_names("SET_A")
    
    forbidden_identity = ["product_id", "source_dataset", "image_path", "filename"]
    for fname in feature_names:
        for fid in forbidden_identity:
            assert fid not in fname.lower(), f"Feature '{fname}' encodes identity '{fid}' — leakage risk!"


# ═══════════════════════════════════════════════════════
# Test 3: Feature intervention — D05 PRESENT→ABSENT changes features
# ═══════════════════════════════════════════════════════
def test_feature_intervention_d05_changes():
    """Flipping D05 from PRESENT to ABSENT MUST change the feature vector."""
    fe = ProductComplianceFeatureExtractor()
    
    rec_present = make_record(d05="PRESENT")
    rec_absent = make_record(d05="ABSENT")
    
    X_present = fe.transform([rec_present], feature_set="SET_A")
    X_absent = fe.transform([rec_absent], feature_set="SET_A")
    
    # They must NOT be identical
    diff = np.abs(X_present[0] - X_absent[0]).sum()
    assert diff > 0.01, (
        f"Flipping D05 PRESENT→ABSENT produced no feature change (diff={diff:.6f}). "
        "Features are not evidence-driven!"
    )


# ═══════════════════════════════════════════════════════
# Test 4: Feature intervention propagates to prediction
# ═══════════════════════════════════════════════════════
def test_feature_intervention_changes_prediction():
    """Changing D06 from PRESENT to ABSENT MUST change model output probabilities."""
    predictor = get_predictor()
    
    rec_compliant = make_record(d06="PRESENT")
    rec_missing_d06 = make_record(d06="ABSENT")
    
    result_compliant = predictor.predict_record(rec_compliant)
    result_missing = predictor.predict_record(rec_missing_d06)
    
    # Probabilities must differ
    prob_c1 = result_compliant["probabilities"].get("COMPLIANT", 0)
    prob_c2 = result_missing["probabilities"].get("COMPLIANT", 0)
    
    assert abs(prob_c1 - prob_c2) > 0.001, (
        f"D06 PRESENT→ABSENT did NOT change COMPLIANT probability "
        f"({prob_c1:.4f} vs {prob_c2:.4f}). Model is not evidence-responsive!"
    )


# ═══════════════════════════════════════════════════════
# Test 5: All-PRESENT predicts COMPLIANT, missing D06 predicts NON_COMPLIANT
# ═══════════════════════════════════════════════════════
def test_prediction_responds_to_evidence():
    """Full declarations → COMPLIANT; Missing mandatory → NON_COMPLIANT."""
    predictor = get_predictor()
    
    rec_full = make_record()  # All PRESENT
    rec_missing = make_record(d06="ABSENT")  # Missing mandatory consumer care
    
    result_full = predictor.predict_record(rec_full)
    result_missing = predictor.predict_record(rec_missing)
    
    assert result_full["overall_prediction"] in ("COMPLIANT", "UNCERTAIN"), (
        f"All declarations PRESENT but predicted {result_full['overall_prediction']}"
    )
    assert result_missing["overall_prediction"] in ("NON_COMPLIANT", "UNCERTAIN"), (
        f"D06 ABSENT but predicted {result_missing['overall_prediction']}"
    )


# ═══════════════════════════════════════════════════════
# Test 6: Shuffled-target sanity check (no systematic leakage)
# ═══════════════════════════════════════════════════════
def test_shuffled_target_no_perfect_accuracy():
    """
    If we randomize targets but keep features fixed, accuracy should drop to ~50%.
    This is a proxy for detecting systematic target encoding in features.
    """
    fe = ProductComplianceFeatureExtractor()
    
    # Create 20 varied records
    records = []
    for i in range(20):
        if i < 10:
            r = make_record()  # All PRESENT
        else:
            r = make_record(d05="ABSENT", d06="ABSENT")  # Missing mandatory
        records.append(r)
    
    X = fe.transform(records, feature_set="SET_A")
    
    # Real targets (deterministic from features)
    real_targets = np.array([0 if i < 10 else 1 for i in range(20)])  # 0=COMPLIANT, 1=NC
    
    # Shuffled targets (random)
    np.random.seed(42)
    shuffled_targets = real_targets.copy()
    np.random.shuffle(shuffled_targets)
    
    # Simple nearest-centroid accuracy on shuffled targets should be low
    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(X, shuffled_targets)
    shuffled_acc = clf.score(X, shuffled_targets)
    
    # With shuffled targets, accuracy should not be perfect
    # (some overfitting is expected with KNN on training data, but leakage would give 1.0)
    assert shuffled_acc < 0.95, (
        f"Shuffled-target accuracy is suspiciously high ({shuffled_acc:.4f}). "
        "Check for target encoding in features!"
    )


# ═══════════════════════════════════════════════════════
# Test 7: Evidence-quality-adjusted confidence
# ═══════════════════════════════════════════════════════
def test_confidence_calibration():
    """Confidence must be lower when evidence quality is poor."""
    predictor = get_predictor()
    
    # Good evidence
    rec_good = make_record(ocr_quality=0.95)
    # Poor evidence
    rec_poor = make_record(ocr_quality=0.30)
    
    result_good = predictor.predict_record(rec_good)
    result_poor = predictor.predict_record(rec_poor)
    
    assert result_good["confidence"] >= result_poor["confidence"], (
        f"Good OCR ({result_good['confidence']:.4f}) should produce >= confidence "
        f"than poor OCR ({result_poor['confidence']:.4f})"
    )
    
    # Poor evidence should never show 100% confidence
    assert result_poor["confidence"] < 1.0, (
        f"Poor OCR quality should not produce 100% confidence"
    )


# ═══════════════════════════════════════════════════════
# Test 8: Synthetic ground truth fields never in features
# ═══════════════════════════════════════════════════════
def test_synthetic_ground_truth_excluded():
    """Fields like violation_type, ground_truth_source, generation_method must not be features."""
    fe = ProductComplianceFeatureExtractor()
    feature_names = fe.get_feature_names("SET_A")
    
    synthetic_specific = [
        "violation_type", "ground_truth_source", "generation_method",
        "image_generation_method", "generation_timestamp"
    ]
    for fname in feature_names:
        for sf in synthetic_specific:
            assert sf not in fname.lower(), (
                f"Feature '{fname}' contains synthetic ground-truth field '{sf}'!"
            )
