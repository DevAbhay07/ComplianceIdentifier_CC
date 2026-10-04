# Inference Failure Modes & Fault Tolerance (Phase 9)

**Project:** NitiMark-ML — Standalone Legal Metrology Compliance Predictor  
**Phase:** Phase 9 — Standalone Inference & Real-Product Validation  
**Date:** September 2026  

---

## 1. Overview

This document analyzes all potential inference failure modes, their handling mechanisms, and the diagnostic messages returned by the system.

---

## 2. Failure Mode Catalog

| Failure Code | Trigger Condition | System Behavior | User Diagnostic Message |
| :--- | :--- | :--- | :--- |
| **`FILE_NOT_FOUND`** | Image path does not exist on disk | Fails gracefully before OCR | `"FILE_NOT_FOUND: Image file does not exist at <path>"` |
| **`UNSUPPORTED_FORMAT`** | File format is not JPEG, PNG, or WEBP (e.g. BMP, TIFF, GIF) | Fails gracefully | `"UNSUPPORTED_FORMAT: Format '<fmt>' is not supported. Use JPEG, PNG, or WEBP."` |
| **`CORRUPT_OR_UNREADABLE`** | Truncated or corrupted image bytes | Caught by `img.verify()` | `"CORRUPT_OR_UNREADABLE_IMAGE: <PIL error details>"` |
| **`INVALID_API_KEY`** | `VISION_API_KEY` missing or invalid | Skips OCR, raises warning | `"VISION_API_KEY is missing or empty. Configure VISION_API_KEY in environment or .env file."` |
| **`QUOTA_EXHAUSTED`** | Gemini API 429 quota reached | Catches exception, sets status | `"VISION_API_ERROR (QUOTA_EXHAUSTED): Rate limit exceeded."` |
| **`VISION_REQUIRED`** | Image-only input supplied in `--offline` mode | Exits cleanly without crashing | `"VISION_REQUIRED: Image provided in offline mode without text metadata."` |
| **`INFERENCE_SCHEMA_ERROR`** | Column count or names do not match frozen pipeline | Blocks model execution | `"Schema mismatch: DIMENSION_MISMATCH..."` |
| **`DATA_LEAKAGE_WARNING`** | Input product ID matches item in training split | Prediction runs; warning issued | `"DATA_LEAKAGE_WARNING: Product ID was part of Phase 8 training partition."` |

---

## 3. Resilience Guarantees

1. **Zero Unhandled Exceptions:** All CLI and API entry points wrap critical stages in try/catch blocks that return structured JSON errors rather than raw stack traces.
2. **Degraded Execution:** If Vision OCR fails due to a network timeout or quota exhaustion, but the user supplied product text/title, the system falls back gracefully to text-only ML prediction with an explicit warning.
