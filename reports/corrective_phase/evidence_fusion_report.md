# Multi-Image Evidence Fusion Report

## Executive Summary
Real physical packaging declarations rarely appear on a single panel.
Manufacturers distribute statutory markings across the front, back, top, bottom, and side panels.
The **Evidence Fusion Engine** (`src/extraction/evidence_fusion.py`) aggregates evidence across all photographs of a single physical product into **ONE unified, conflict-aware `ProductEvidenceRecord`** before feature engineering or ML inference is executed.

---

## Architectural Principle: The Product is the Unit of Inference

```
[Front View Image]  ──>  YOLO + OCR  ──>  Front Evidence (D02, D03, D05)
                                                    │
[Back View Image]   ──>  YOLO + OCR  ──>  Back Evidence  (D01, D04, D06, D07)
                                                    │
                                                    ▼
                                      ┌───────────────────────────┐
                                      │   EVIDENCE FUSION ENGINE  │
                                      │   • Evidence Union        │
                                      │   • Source Provenance     │
                                      │   • Conflict Detection    │
                                      └─────────────┬─────────────┘
                                                    │
                                                    ▼
                                      ONE ProductEvidenceRecord
                                                    │
                                                    ▼
                                      ONE Declaration Feature Vector (Set E)
                                                    │
                                                    ▼
                                      ONE Product Compliance Prediction
```

**Anti-Pattern Avoided**: The system never predicts compliance separately per photograph and averages prediction scores. Instead, packaging evidence is fused first.

---

## Multi-Image Regression Test Scenarios

### Scenario 1: Complementary Panel Evidence (MRP Retention)
- **Image 1 (Front)**: Contains product name and net quantity (200g). Lacks MRP.
- **Image 2 (Back)**: Contains MRP (₹299.00 incl. of taxes) and manufacturer address.
- **Fusion Behavior**: `D03` (200g) and `D05` (₹299.00) are merged into the final representation. Both declarations are marked `PRESENT`.

### Scenario 2: Distributed Contact & Identity Info
- **Image 1**: Contains manufacturing address only.
- **Image 2**: Contains customer helpline and grievance email only.
- **Fusion Behavior**: `D01` and `D06` are preserved with image source attribution (`source_image`: "image_1", "image_2").

### Scenario 3: Global Insufficient Evidence
- **All Images**: Blurry, illegible captures with OCR confidence < 0.20.
- **Fusion Behavior**: Missing declarations remain `CANNOT_DETERMINE` or `ABSENT`, correctly prompting the ML predictor to abstain to `UNCERTAIN`.

### Scenario 4: Cross-View Conflict Detection
- **Image 1**: Transcribes `MRP: Rs. 999.00`.
- **Image 2**: Transcribes `MRP: Rs. 1099.00`.
- **Fusion Behavior**:
  - The conflict is flagged: `conflicting_fields = ["D05"]`.
  - Both conflicting candidate values are preserved in `conflicting_values`.
  - The downstream predictor detects the conflict and triggers abstention to `UNCERTAIN` rather than silently picking an arbitrary value.
