# Phase 11 Engineering & Dataset Roadmap Recommendations

> **Project**: NitiMark-ML — Standalone ML Prediction Layer (SIH25057)  
> **Milestone Goal**: Transition from Phase 10 Decision Framework to Phase 11 Packaging-Trained Production Predictor.

---

## 1. Top 5 Recommended Priorities for Phase 11

### Priority 1: Physical Packaging Dataset Collection ($N = 500\text{--}1,000$)
- **Problem**: Training on e-commerce catalog metadata creates an inescapable domain transfer ceiling because catalog listings omit batch manufacturing dates (`D04`) and origin stamps (`D07`).
- **Solution**: Curate a dedicated packaging label dataset across FMCG, cosmetics, pharmaceuticals, electronics, and food products with ground-truth packaging OCR transcripts.

### Priority 2: Replace Frozen Baselines for D04 and D07
- **Problem**: In Phase 8, extreme class imbalance in Flipkart metadata forced D04 and D07 to be represented by majority baselines predicting absence, causing 17 false positive alarms on compliant physical packages in Phase 10.
- **Solution**: Train dedicated supervised token classifiers and regex parsers for batch date stamps (`MFD`, `PKD`, `EXP`) and country of origin declarations.

### Priority 3: Spatial Layout-Aware OCR Embeddings (LayoutLMv3 / Donut)
- **Problem**: Flattening OCR text into 1D TF-IDF bag-of-words loses spatial hierarchy (e.g. principal display panel vs. back-label text blocks).
- **Solution**: Incorporate 2D spatial bounding box coordinates $[x_0, y_0, x_1, y_1]$ into the feature adaptation pipeline.

### Priority 4: Dual-Tier Local/Cloud OCR Sidecar
- **Problem**: Cloud vision REST calls introduce network latency ($1.2\text{s}\text{--}2.8\text{s}$) and quota limits.
- **Solution**: Deploy an embedded local OCR engine (PaddleOCR / Tesseract-v5) for fast sub-150ms processing, falling back to Google Gemini 3.6 Flash only for degraded packaging images.

### Priority 5: Preparation for Main NitiMark Application Integration
- **Solution**: Package `src/inference/` and `src/decision/` into a versioned Python wheel or containerized microservice ready for downstream integration into the NitiMark web dashboard and backend compliance engine.
