"""
synthetic_extraction_validator.py
=================================
Validates declaration and actual-value extraction against known synthetic ground truth.
Isolates Extraction Failures from Feature Failures, ML Failures, and Decision Engine Failures.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import defaultdict

PROJECT_ROOT = Path("/home/devabhay07/Desktop/MLmodelV2")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.extraction.declaration_extractor import DeclarationExtractor
from src.features.product_compliance_feature_extractor import ProductComplianceFeatureExtractor
from src.features.product_training_representation import ProductComplianceTrainingRecord
from src.models.predict_product_compliance import ProductCompliancePredictor

SYNTHETIC_JSONL = PROJECT_ROOT / "data/processed/synthetic_physical_packaging.jsonl"
REPORTS_DIR = PROJECT_ROOT / "reports/product_compliance"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def run_synthetic_extraction_validation() -> Dict[str, Any]:
    print("=================================================================")
    print("   SYNTHETIC GROUND-TRUTH EXTRACTION & PIPELINE VALIDATION       ")
    print("=================================================================\n")

    if not SYNTHETIC_JSONL.exists():
        raise FileNotFoundError(f"Synthetic dataset missing: {SYNTHETIC_JSONL}")

    records: List[Dict[str, Any]] = []
    with open(SYNTHETIC_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    print(f"Loaded {len(records)} synthetic records with known ground truth.")

    extractor = DeclarationExtractor()
    feature_extractor = ProductComplianceFeatureExtractor()
    predictor = ProductCompliancePredictor()

    total_samples = len(records)
    state_matches = defaultdict(int)
    value_matches = defaultdict(int)
    total_decls_evaluated = defaultdict(int)

    failure_breakdown = {
        "EXTRACTION_FAILURES": 0,
        "FEATURE_FAILURES": 0,
        "ML_FAILURES": 0,
        "DECISION_ENGINE_FAILURES": 0
    }

    eval_details = []

    for r in records:
        sid = r["product_id"]
        gt_target = r["overall_target"]
        gt_decls = r["declarations"]
        raw_text = r["raw_text"]
        spec = r.get("metadata", {}).get("synthetic_spec", {})

        # Step 1: Run extractor on text
        extracted_decls = extractor.extract_from_text_and_vision(
            raw_ocr_text=raw_text,
            visual_regions=[]
        )

        decl_extraction_correct = True
        for did in ["D01", "D02", "D03", "D04", "D05", "D06", "D07"]:
            gt_st = gt_decls.get(did, {}).get("status", "CANNOT_DETERMINE")
            ex_st = extracted_decls[did].status.value if hasattr(extracted_decls[did].status, "value") else extracted_decls[did].status
            
            total_decls_evaluated[did] += 1
            if gt_st == ex_st or (gt_st == "PARTIAL" and ex_st in ["PRESENT", "PARTIAL"]):
                state_matches[did] += 1
            else:
                decl_extraction_correct = False

        # Value comparison
        # D05 MRP
        gt_mrp = r.get("d05_values", {}).get("mrp")
        ex_mrp = extracted_decls["D05"].normalized_value.get("mrp")
        if gt_mrp and ex_mrp and abs(float(gt_mrp) - float(ex_mrp)) < 1.0:
            value_matches["D05_MRP"] += 1

        # D03 Qty
        gt_qty = r.get("d03_values", {}).get("quantity")
        ex_qty = extracted_decls["D03"].normalized_value.get("quantity")
        if gt_qty and ex_qty and abs(float(gt_qty) - float(ex_qty)) < 0.1:
            value_matches["D03_QTY"] += 1

        # D04 Year
        gt_yr = r.get("d04_values", {}).get("year")
        ex_yr = extracted_decls["D04"].normalized_value.get("year")
        if gt_yr and ex_yr and int(gt_yr) == int(ex_yr):
            value_matches["D04_YEAR"] += 1

        # Step 2: Feature extraction check
        training_rec = ProductComplianceTrainingRecord.from_dict(r)
        feat_matrix = feature_extractor.transform([training_rec], feature_set="SET_A")
        feat_valid = (feat_matrix.shape == (1, 16) and not np.isnan(feat_matrix).any())

        # Step 3: Run predictor
        pred_res = predictor.predict_record(training_rec, apply_abstention=True)
        final_verdict = pred_res["overall_prediction"]
        raw_pred = pred_res["raw_prediction"]

        # Failure classification
        if not decl_extraction_correct and gt_target != "UNCERTAIN":
            failure_breakdown["EXTRACTION_FAILURES"] += 1
        elif not feat_valid:
            failure_breakdown["FEATURE_FAILURES"] += 1
        elif raw_pred != gt_target and gt_target in ["COMPLIANT", "NON_COMPLIANT"]:
            # Note: Before retraining on synthetic, ML may mispredict COMPLIANT
            failure_breakdown["ML_FAILURES"] += 1
        elif final_verdict != gt_target:
            failure_breakdown["DECISION_ENGINE_FAILURES"] += 1

    # Summarize Metrics
    state_accuracies = {did: round(state_matches[did] / total_decls_evaluated[did], 4) for did in sorted(state_matches.keys())}
    
    print("\n--- Declaration-State Extraction Accuracies ---")
    for did, acc in state_accuracies.items():
        print(f"  • {did} State Extraction Accuracy: {acc * 100:.2f}% ({state_matches[did]}/{total_decls_evaluated[did]})")

    print("\n--- Actual-Value Normalization Accuracies ---")
    for vname, cnt in value_matches.items():
        print(f"  • {vname} Exact Value Match: {cnt}/{total_samples}")

    print("\n--- Stage Failure Breakdown (Pre-Retraining Diagnostics) ---")
    for k, v in failure_breakdown.items():
        print(f"  • {k}: {v}")

    report = {
        "timestamp": "2026-09-05T20:32:00Z",
        "total_synthetic_evaluated": total_samples,
        "declaration_state_accuracies": state_accuracies,
        "actual_value_accuracies": {k: round(v / total_samples, 4) for k, v in value_matches.items()},
        "failure_breakdown": failure_breakdown
    }

    out_file = REPORTS_DIR / "synthetic_extraction_validation.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nValidation report saved to: {out_file}")
    return report


if __name__ == "__main__":
    import numpy as np
    run_synthetic_extraction_validation()
