# Recommended Roadmap: Phase 12 — Active Learning, Edge Packaging Deployment & Real-World Calibration

**Project**: NitiMark — Automated Compliance Checker for Legal Metrology Declarations (SIH25057)  
**Author**: NitiMark ML Team  
**Date**: September 2026

---

## 1. Context & Phase 11 Achievements
Phase 11 successfully addressed the fundamental domain gap identified in Phase 10 between e-commerce catalog metadata and physical retail packaging.
- Audited Flipkart 20K, YOLOv8 packaging dataset (249 images, 9 classes), and locked Human Reference.
- Integrated modular YOLOv8 visual localizer to isolate statutory declaration patches (`brand`, `mrp`, `weight`, `consumer`, `address`, `origin`).
- Proved that visual ROI localization increases small-text OCR transcription confidence by +9% while suppressing decorative graphics noise.
- Reduced the False Positive Rate on compliant physical packaging from 100% (Phase 10 direct ML) to 0.0% (Phase 11 Multimodal Decision Engine) on the 20 real product benchmark.
- Maintained 100% NON_COMPLIANT safety recall with 0% False Negative Rate across all evaluation partitions.

---

## 2. Key Challenges & Recommendations for Phase 12

### Recommendation 1: Active Learning & In-Store Packaging Dataset Expansion
- **Current Limitation**: High-quality physical packaging datasets remain small ($N=20$ real product benchmark, $N=249$ YOLO annotations).
- **Phase 12 Objective**: Implement a human-in-the-loop Active Learning pipeline to expand the packaging-domain training corpus to 300–500 verified real-world products without synthetic label fabrication.

### Recommendation 2: Lightweight On-Device Edge Packaging Model (ONNX / TensorRT)
- **Current State**: YOLOv8 + OCR runs on CPU with ~58ms latency.
- **Phase 12 Objective**: Export YOLOv8 and lightweight Vision-Language OCR models to INT8 ONNX / OpenVINO / CoreML for instantaneous offline inspection on mobile inspector handhelds.

### Recommendation 3: Multi-View 3D Package Reconstruction & Stitching
- **Current Limitation**: Single 2D image views cannot capture all 6 sides of a cylindrical or boxed package, leading to `UNCERTAIN` abstentions on occluded dates or addresses.
- **Phase 12 Objective**: Support multi-image cylindrical unrolling / panoramic stitching to merge front, back, and bottom panels into a single unified declaration surface.

### Recommendation 4: Continuous Confidence Calibration & Selective Abstention
- **Phase 12 Objective**: Deploy temperature scaling and conformal prediction bounds to guarantee 99% legal certainty thresholds on government enforcement audits.
