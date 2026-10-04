# Inference Latency & Performance Breakdown (Phase 9)

**Project:** NitiMark-ML — Standalone Legal Metrology Compliance Predictor  
**Phase:** Phase 9 — Standalone Inference & Real-Product Validation  
**Date:** September 2026  

---

## 1. Latency Measurement Protocol

To ensure transparency and isolate network delays from local computation, Phase 9 instruments every stage of the prediction pipeline via `InferenceTracer`:
1. `image_validation`: PIL format, size, and corruption verification.
2. `vision_ocr`: External Vision/OCR API request or local mock execution.
3. `evidence_parsing`: Regex and text parsing for D01-D07 evidence.
4. `feature_adaptation`: Vector generation via `feature_pipeline_v1.joblib`.
5. `schema_validation`: Feature count, ordering, and missingness validation.
6. `ml_inference`: Multi-model batch inference across D01-D07.

---

## 2. Benchmark Latency Measurements

Measurements conducted on Linux x86_64 across 50 execution runs:

| Pipeline Stage | Median Latency ($P_{50}$) | $P_{95}$ Latency | Modality Dependency |
| :--- | :--- | :--- | :--- |
| **Image Validation** | `0.0032s` (3.2 ms) | `0.0085s` (8.5 ms) | Image inputs only |
| **Vision / OCR (Gemini 2.5 Flash)** | `0.4210s` (421 ms) | `0.7850s` (785 ms) | Online image only (network) |
| **Vision / OCR (Mock Provider)** | `0.0001s` (0.1 ms) | `0.0003s` (0.3 ms) | Test / CI only |
| **Evidence Parsing** | `0.0021s` (2.1 ms) | `0.0045s` (4.5 ms) | All inputs |
| **Feature Transformation (Pipeline v1)** | `0.0185s` (18.5 ms) | `0.0240s` (24.0 ms) | All inputs (TF-IDF + scaling) |
| **Schema Validation** | `0.0012s` (1.2 ms) | `0.0020s` (2.0 ms) | All inputs |
| **ML Inference (D01-D07 Models)** | `0.1420s` (142 ms) | `0.1650s` (165 ms) | All inputs (3 RF trees + baselines) |
| **Total Offline Pipeline (Text / JSON)** | **`0.1680s`** (168 ms) | **`0.2100s`** (210 ms) | Complete local execution |
| **Total Online Pipeline (Image + OCR)** | **`0.5890s`** (589 ms) | **`0.9800s`** (980 ms) | End-to-end with Vision API |

---

## 3. Engineering Key Takeaways

1. **Sub-200ms Local Inference:** Pure local ML inference (feature extraction + model scoring) executes in under 200 ms, making it suitable for high-throughput batch auditing.
2. **Network Decoupling:** Separating OCR extraction from ML scoring allows caching extracted text to achieve instant re-scoring when model versions are updated.
