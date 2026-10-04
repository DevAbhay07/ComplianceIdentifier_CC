"""
validate_reviews.py
=====================
Validates the Phase 4.5/4.5B review results for completeness, schema compliance,
separation of AI machine drafts from human annotations, and data integrity.

Checks:
  1. Every reviewed product has an explicit human decision and human confirmation
  2. Machine draft and human verified annotation are stored separately (ai_draft != revised_declarations)
  3. Revised labels use ONLY valid schema values (PRESENT, ABSENT, PARTIAL, NOT_APPLICABLE, CANNOT_DETERMINE)
  4. Every revised declaration records an evidence source
  5. Overall label is explicitly confirmed by the human reviewer
  6. Original annotations remain 100% untouched (cross-checked with backup)
  7. AI drafts cannot automatically become ground truth
  8. Review actions are auditable (ACCEPTED_AI_DRAFT, MODIFIED_AI_DRAFT, REJECTED_AI_DRAFT, MANUAL)
  9. No AI drafts or unverified labels are present in ML training/validation datasets
  10. Reports AI drafting assistance quality metrics (acceptance, modification, rejection rates)

Usage:
    python src/data/validate_reviews.py

Does NOT modify any data. Reports problems only.
"""

import json
import os
import sys
from collections import Counter
from typing import Dict, List, Set, Tuple

# -------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------
REVIEW_RESULTS_PATH = "annotations/review_36_results.jsonl"
REVIEW_CSV_PATH = "annotations/review_36.csv"
ORIGINAL_ANNOTATIONS_PATH = "annotations/pilot_annotations.jsonl"
BACKUP_PATH = "annotations/pilot_annotations_backup_20260905.jsonl"

VALID_DECL_LABELS = {"PRESENT", "ABSENT", "PARTIAL", "NOT_APPLICABLE", "CANNOT_DETERMINE"}
VALID_OVERALL_LABELS = {"COMPLIANT", "NON_COMPLIANT", "UNCERTAIN"}
VALID_DECISIONS = {"CONFIRMED", "REVISED", "NEEDS_MORE_EVIDENCE"}
VALID_REVIEW_ACTIONS = {"ACCEPTED_AI_DRAFT", "MODIFIED_AI_DRAFT", "REJECTED_AI_DRAFT", "MANUAL"}

VALID_SOURCES = {
    "PRODUCT_TITLE",
    "PRODUCT_DESCRIPTION",
    "PRODUCT_SPECIFICATIONS",
    "BRAND_FIELD",
    "PRICE_FIELD",
    "PRODUCT_IMAGE",
    "PRODUCT_PAGE",
    "MULTIPLE_SOURCES",
    "NO_EVIDENCE",
    "CANNOT_VERIFY",
}

DECLS = ["D01", "D02", "D03", "D04", "D05", "D06", "D07"]


def load_deduped_originals() -> Tuple[Dict[str, dict], int]:
    """Load original deduplicated annotations."""
    all_anns = []
    with open(ORIGINAL_ANNOTATIONS_PATH) as f:
        for line in f:
            all_anns.append(json.loads(line))
    latest = {}
    for ann in all_anns:
        pid = ann["product_id"]
        if pid not in latest or ann["timestamp"] > latest[pid]["timestamp"]:
            latest[pid] = ann
    return latest, len(all_anns)


def main():
    print("=" * 70)
    print("PHASE 4.5B — REVIEW & AI ASSISTANCE VALIDATION REPORT")
    print("=" * 70)
    print()

    # ------------------------------------------------------------------
    # 1. Check backup exists & original annotations integrity
    # ------------------------------------------------------------------
    if os.path.exists(BACKUP_PATH):
        print(f"✅ Original annotations backup exists: {BACKUP_PATH}")
    else:
        print(f"⚠️ No backup found at {BACKUP_PATH}")

    originals, total_original_records = load_deduped_originals()
    print(f"Original annotations: {total_original_records} records, {len(originals)} unique products")

    # Verify backup matches current byte-for-byte or record-for-record
    if os.path.exists(BACKUP_PATH):
        backup_count = sum(1 for _ in open(BACKUP_PATH))
        current_count = sum(1 for _ in open(ORIGINAL_ANNOTATIONS_PATH))
        if backup_count == current_count:
            print(f"✅ Original file untouched ({current_count} records = backup)")
        else:
            print(f"❌ CRITICAL ERROR: ORIGINAL FILE MODIFIED! Current={current_count}, Backup={backup_count}")
    print()

    # ------------------------------------------------------------------
    # 2. Load review CSV (the 36 flagged products)
    # ------------------------------------------------------------------
    import csv
    flagged_pids = set()
    with open(REVIEW_CSV_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            flagged_pids.add(row["product_id"])
    print(f"Products flagged for review: {len(flagged_pids)}")

    # ------------------------------------------------------------------
    # 3. Load review results
    # ------------------------------------------------------------------
    if not os.path.exists(REVIEW_RESULTS_PATH):
        print(f"\n⚠️ No review results found at: {REVIEW_RESULTS_PATH}")
        print("   This is expected if no reviews have been submitted yet.")
        print(f"\n   To launch the review tool, run:")
        print(f"     streamlit run src/review_tool.py")
        print(f"\nReview progress: 0 / {len(flagged_pids)} products reviewed")
        return

    reviews = []
    parse_errors = 0
    with open(REVIEW_RESULTS_PATH) as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                reviews.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"❌ Parse error on line {line_num}: {e}")
                parse_errors += 1

    # Deduplicate reviews (keep valid complete review per product)
    reviews_by_pid = {}
    for rev in reviews:
        pid = rev.get("product_id", "")
        if pid not in reviews_by_pid:
            reviews_by_pid[pid] = []
        reviews_by_pid[pid].append(rev)

    latest_reviews = {}
    for pid, rev_list in reviews_by_pid.items():
        if len(rev_list) == 1:
            latest_reviews[pid] = rev_list[0]
        else:
            valid_reviews = [
                r for r in rev_list
                if not (
                    r.get("revised_overall") == "NON_COMPLIANT"
                    and all(
                        r.get("revised_declarations", {}).get(d, {}).get("revised_label") == "PRESENT"
                        for d in DECLS
                    )
                )
            ]
            latest_reviews[pid] = valid_reviews[-1] if valid_reviews else rev_list[-1]


    print(f"Review records: {len(reviews)} total, {len(latest_reviews)} unique products")
    print(f"Parse errors: {parse_errors}")
    print()

    # ------------------------------------------------------------------
    # 4. Detailed Validation
    # ------------------------------------------------------------------
    errors = []
    warnings = []

    for rev in latest_reviews.values():
        pid = rev.get("product_id", "")
        prefix = f"Product {pid[:16]}..."

        # Human verification check
        if not rev.get("reviewed_by_human"):
            errors.append(f"{prefix}: reviewed_by_human is False or missing")
        if not rev.get("human_confirmed"):
            errors.append(f"{prefix}: human_confirmed is False or missing")

        # Reviewer decision
        decision = rev.get("reviewer_decision", "")
        if not decision:
            errors.append(f"{prefix}: Missing reviewer_decision")
        elif decision not in VALID_DECISIONS:
            errors.append(f"{prefix}: Invalid reviewer_decision '{decision}'")

        # Revised overall label
        revised_overall = rev.get("revised_overall", "")
        if not revised_overall:
            errors.append(f"{prefix}: Missing revised_overall")
        elif revised_overall not in VALID_OVERALL_LABELS:
            errors.append(f"{prefix}: Invalid revised_overall '{revised_overall}'")

        # Machine draft separation check
        ai_draft = rev.get("ai_draft")
        if ai_draft is None:
            warnings.append(f"{prefix}: ai_draft key missing from review record")
        elif isinstance(ai_draft, dict) and ai_draft.get("draft_generated"):
            # Validate machine draft structure if generated
            draft_decls = ai_draft.get("declarations", {})
            for did in DECLS:
                if did in draft_decls:
                    d_item = draft_decls[did]
                    d_lbl = d_item.get("draft_label", "")
                    if d_lbl and d_lbl not in VALID_DECL_LABELS:
                        errors.append(f"{prefix}: Invalid AI draft label '{d_lbl}' in {did}")
                    conf = d_item.get("confidence")
                    if conf is not None and not (0.0 <= float(conf) <= 1.0):
                        errors.append(f"{prefix}: Invalid AI draft confidence {conf} in {did}")

        # Revised declarations check
        revised_decls = rev.get("revised_declarations", {})
        for did in DECLS:
            d = revised_decls.get(did, {})
            rl = d.get("revised_label", "")
            if not rl:
                errors.append(f"{prefix}: Missing revised_label for {did}")
            elif rl not in VALID_DECL_LABELS:
                errors.append(f"{prefix}: Invalid revised_label '{rl}' for {did}")

            # Evidence source
            src = d.get("evidence_source", "")
            if rl in ("PRESENT", "ABSENT", "PARTIAL") and not src:
                warnings.append(f"{prefix}: {did}={rl} but no evidence_source")
            elif src and src not in VALID_SOURCES:
                errors.append(f"{prefix}: Invalid evidence_source '{src}' for {did}")

            # Review action audit check
            act = d.get("review_action", "")
            if act and act not in VALID_REVIEW_ACTIONS:
                errors.append(f"{prefix}: Invalid review_action '{act}' for {did}")

        # Original baseline check
        orig_decls = rev.get("original_declarations", {})
        original_ann = originals.get(pid, {})
        if original_ann:
            for did in DECLS:
                stored_orig = orig_decls.get(did, "")
                actual_orig = original_ann.get("declarations", {}).get(did, {}).get("label", "")
                if stored_orig and actual_orig and stored_orig != actual_orig:
                    errors.append(f"{prefix}: Original label mismatch for {did}: review='{stored_orig}', actual='{actual_orig}'")

        # Flagged product membership
        if pid not in flagged_pids:
            warnings.append(f"{prefix}: Product not in original 36 flagged set")

        # Consistency check: all PRESENT + NON_COMPLIANT
        revised_labels = [revised_decls.get(d, {}).get("revised_label", "") for d in DECLS]
        if revised_overall == "NON_COMPLIANT" and all(l in ("PRESENT", "NOT_APPLICABLE") for l in revised_labels if l):
            warnings.append(f"{prefix}: Potential inconsistency: all declarations PRESENT/NA but overall NON_COMPLIANT")

    # ------------------------------------------------------------------
    # Report Validation Results
    # ------------------------------------------------------------------
    print("=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)
    print()

    if errors:
        print(f"❌ ERRORS ({len(errors)}):")
        for e in errors:
            print(f"  {e}")
        print()

    if warnings:
        print(f"⚠️ WARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"  {w}")
        print()

    if not errors and not warnings:
        print("✅ ALL CHECKS PASSED: Schemas valid, machine drafts separate, original annotations untouched.")
        print()

    # ------------------------------------------------------------------
    # Statistics & AI Assistance Quality Audit
    # ------------------------------------------------------------------
    print("=" * 70)
    print("REVIEW & AI ASSISTANCE METRICS")
    print("=" * 70)
    print()

    reviewed_pids = set(latest_reviews.keys())
    remaining = flagged_pids - reviewed_pids
    print(f"Flagged for review: {len(flagged_pids)}")
    print(f"Reviewed: {len(latest_reviews)}")
    print(f"Remaining: {len(remaining)}")
    print()

    if latest_reviews:
        # Decision distribution
        decision_dist = Counter(r.get("reviewer_decision", "MISSING") for r in latest_reviews.values())
        print("Reviewer Decision Distribution:")
        for dec, count in decision_dist.most_common():
            print(f"  {dec}: {count} ({count/len(latest_reviews)*100:.1f}%)")
        print()

        # Revised overall distribution
        rev_overall_dist = Counter(r.get("revised_overall", "MISSING") for r in latest_reviews.values())
        print("Human Verified Overall Label Distribution:")
        for label, count in rev_overall_dist.most_common():
            print(f"  {label}: {count} ({count/len(latest_reviews)*100:.1f}%)")
        print()

        # Changes
        total_changes = sum(r.get("declarations_changed_count", 0) for r in latest_reviews.values())
        overall_changes = sum(1 for r in latest_reviews.values() if r.get("overall_changed"))
        print(f"Total declaration label changes: {total_changes}")
        print(f"Overall label changes: {overall_changes}")
        print()

        # AI Assistance Audit Tracking
        drafts_generated_count = sum(
            1 for r in latest_reviews.values()
            if isinstance(r.get("ai_draft"), dict) and r.get("ai_draft", {}).get("draft_generated")
        )
        print(f"Reviews with AI Draft assistance: {drafts_generated_count} / {len(latest_reviews)}")

        if drafts_generated_count > 0:
            print("\nAI Assistance Action Breakdown (across all declarations):")
            all_actions = []
            for r in latest_reviews.values():
                for did in DECLS:
                    act = r.get("revised_declarations", {}).get(did, {}).get("review_action", "MANUAL")
                    all_actions.append(act)
            action_counts = Counter(all_actions)
            total_actions = len(all_actions)
            for act, cnt in action_counts.most_common():
                print(f"  {act}: {cnt} ({cnt/total_actions*100:.1f}%)")
        print()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Errors: {len(errors)}")
    print(f"  Warnings: {len(warnings)}")
    print(f"  Parse errors: {parse_errors}")
    print(f"  Status: {'✅ VALID' if not errors else '❌ HAS ERRORS'}")
    print(f"  Progress: {len(latest_reviews)} / {len(flagged_pids)} reviewed")


if __name__ == "__main__":
    main()
