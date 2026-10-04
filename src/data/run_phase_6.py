"""
run_phase_6.py
===============
Master execution pipeline for Phase 6: Automated Dataset Expansion via Weak Labels.
Executes all pipeline stages end-to-end with error handling, progress logging, and verification.

Usage:
    python src/data/run_phase_6.py
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import time
from datetime import datetime, timezone

from src.data.audit_flipkart_dataset import run_audit
from src.data.build_automated_candidate_pool import build_candidate_pool
from src.data.extract_product_evidence import run_evidence_extraction
from src.data.generate_weak_labels import run_weak_label_generation
from src.data.filter_weak_labels import run_filter_and_export
from src.data.analyze_weak_label_distribution import (
    analyze_weak_dataset,
    compare_human_vs_weak,
    run_diagnostic_against_human_reference
)
from src.data.weak_label_leakage_audit import run_leakage_audit
from src.data.phase_6_quality_gate import run_quality_gates


def main():
    start_time = time.time()
    print("=" * 70)
    print("NITIMARK ML — PHASE 6: AUTOMATED WEAK-LABEL DATASET EXPANSION PIPELINE")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 70)

    # Paths configuration
    RAW_CSV = "data/raw/flipkart/flipkart_com-ecommerce_sample.csv"
    RULES_YAML = "configs/weak_label_rules.yaml"
    PILOT_JSONL = "data/processed/pilot_ground_truth_75_v1.jsonl"
    
    CANDIDATES_CSV = "data/interim/automated_candidate_pool.csv"
    EVIDENCE_JSONL = "data/interim/product_evidence.jsonl"
    RAW_WEAK_JSONL = "data/interim/raw_weak_labels.jsonl"
    
    OUT_WEAK_JSONL = "data/processed/flipkart_weak_labels_v1.jsonl"
    OUT_WEAK_CSV = "data/processed/flipkart_weak_labels_v1.csv"
    MANIFEST_JSON = "reports/phase_6/weak_label_manifest.json"
    
    REPORTS_DIR = "reports/phase_6"
    LEAKAGE_REPORT_MD = os.path.join(REPORTS_DIR, "weak_label_leakage_audit.md")
    QUALITY_GATE_MD = os.path.join(REPORTS_DIR, "PHASE_6_QUALITY_GATE.md")

    # Step 1: Dataset Audit
    print("\n>>> STAGE 1: Flipkart Dataset Audit & Profiling...")
    run_audit(RAW_CSV, RULES_YAML, REPORTS_DIR)

    # Step 2: Candidate Pool Selection
    print("\n>>> STAGE 2: Diversity-Aware Candidate Pool Selection...")
    build_candidate_pool(RAW_CSV, RULES_YAML, PILOT_JSONL, CANDIDATES_CSV, target_pool_size=1500)

    # Step 3: Evidence Extraction
    print("\n>>> STAGE 3: Multi-Source Evidence Extraction...")
    run_evidence_extraction(CANDIDATES_CSV, RAW_CSV, EVIDENCE_JSONL, check_images=False)

    # Step 4: Weak Label Generation
    print("\n>>> STAGE 4: Automated Legal Declaration Weak Labeling...")
    run_weak_label_generation(EVIDENCE_JSONL, RULES_YAML, RAW_WEAK_JSONL)

    # Step 5: Quality Filtering & Deduplication Export
    print("\n>>> STAGE 5: Quality Filtering & Deduplication Export...")
    run_filter_and_export(RAW_WEAK_JSONL, OUT_WEAK_JSONL, OUT_WEAK_CSV, MANIFEST_JSON)

    # Step 6: Distribution Analysis & Reference Comparisons
    print("\n>>> STAGE 6: Weak-Label Distribution & Diagnostic Analysis...")
    analyze_weak_dataset(OUT_WEAK_JSONL, REPORTS_DIR)
    compare_human_vs_weak(
        OUT_WEAK_JSONL,
        PILOT_JSONL,
        os.path.join(REPORTS_DIR, "human_vs_weak_distribution_comparison.md")
    )
    run_diagnostic_against_human_reference(
        PILOT_JSONL,
        RULES_YAML,
        os.path.join(REPORTS_DIR, "weak_label_diagnostic_against_human_reference.md")
    )

    # Step 7: Leakage Audit
    print("\n>>> STAGE 7: 10-Point Data Leakage Audit...")
    leakage_status, _ = run_leakage_audit(
        OUT_WEAK_JSONL, PILOT_JSONL, EVIDENCE_JSONL, LEAKAGE_REPORT_MD
    )

    # Step 8: Master Quality Gate
    print("\n>>> STAGE 8: Master Quality Gate Verification...")
    qgate_status, _ = run_quality_gates(
        RAW_CSV,
        CANDIDATES_CSV,
        EVIDENCE_JSONL,
        OUT_WEAK_JSONL,
        OUT_WEAK_CSV,
        MANIFEST_JSON,
        PILOT_JSONL,
        QUALITY_GATE_MD
    )

    elapsed = round(time.time() - start_time, 2)
    print("\n" + "=" * 70)
    print(f"PHASE 6 PIPELINE COMPLETE IN {elapsed}s")
    print(f"Leakage Audit Status: {leakage_status}")
    print(f"Quality Gate Status:  {qgate_status}")
    print("=" * 70)


if __name__ == "__main__":
    main()
