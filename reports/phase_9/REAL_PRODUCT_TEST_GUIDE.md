# Real-Product Testing Guide (Phase 9)

**Project:** NitiMark-ML — Standalone Legal Metrology Compliance Predictor  
**Phase:** Phase 9 — Standalone Inference & Real-Product Validation  
**Date:** September 2026  

---

## 1. Quick Start Guide

The NitiMark-ML standalone prediction engine allows you to evaluate any real-world packaged commodity (packaged foods, personal care, electronics, apparel) using its image, text description, or structured JSON.

### Step 1: Set Up Your API Key (Optional for Online Vision)
If you want to extract text from product images using Google Gemini Vision, export your API key:
```bash
export VISION_API_KEY="your_actual_gemini_api_key"
```
*(Alternatively, copy `.env.example` to `.env` and set `VISION_API_KEY=your_key`).*

---

## 2. Supported CLI Usage Modes

### Mode A: Complete Product JSON (Offline / Online)
To evaluate a product with both text metadata and image path:
```bash
venv/bin/python src/inference/predict.py --input examples/sample_product.json --offline
```

### Mode B: Real Product Image (Online Vision OCR)
To extract packaging text from an image and run ML predictions:
```bash
venv/bin/python src/inference/predict.py --image path/to/your_product.jpg
```

### Mode C: Real Product Image with Mock OCR (No API Key Required)
To run image-based inference deterministically without an external API:
```bash
venv/bin/python src/inference/predict.py --image examples/sample_product.png --provider mock
```

### Mode D: Text-Only Listing (Offline)
To check compliance on e-commerce listing text:
```bash
venv/bin/python src/inference/predict.py \
    --title "Tata Tea Gold 500g" \
    --text "Mfd by Tata Consumer Products Ltd., Kolkata. Net Qty: 500g. MRP Rs. 310.00 incl. taxes. Care: care@tataconsumer.com. Made in India." \
    --offline
```

### Mode E: Machine-Readable JSON Output (For API pipelines)
```bash
venv/bin/python src/inference/predict.py --input examples/sample_product.json --offline --json
```

---

## 3. Interpreting Output

### Terminal Output Example:
```
======================================================================
      NITIMARK-ML — STANDALONE LEGAL METROLOGY PREDICTIONS
======================================================================

[Input Summary]
  * Product Image : examples/sample_product.png
  * Product Title : Colgate Total Clean Mint Toothpaste
  * Mode          : ONLINE

----------------------------------------------------------------------
DECLARATION                         | PREDICTION       | CONF   | MODEL TYPE
----------------------------------------------------------------------
D01 Name & Address of Manufacturer /  | PRESENT          | 0.44   | RandomForest
D02 Generic or Common Name of Commodi | PRESENT          | 1.00   | BASELINE
D03 Net Quantity in Standard Units  | PRESENT          | 0.96   | RandomForest
D04 Date of Manufacture / Packing / E | CANNOT_DETERMINE | 1.00   | BASELINE
D05 Maximum Retail Price (MRP)      | PRESENT          | 1.00   | BASELINE
D06 Consumer Care Contact Details   | ABSENT           | 0.92   | RandomForest
D07 Country of Origin               | CANNOT_DETERMINE | 1.00   | BASELINE
----------------------------------------------------------------------

[Performance & Latency]
  * Total Latency : 0.1730s (ML Inference: 0.1436s)
  * Trace ID      : trace_1788569246946_b86dd2
======================================================================
```

---

## 4. Important Legal & Operational Reminders

1. **Model Confidence $\ne$ Legal Certainty:**
   - Confidence reflects the statistical probability output by the trained model (e.g. Random Forest leaf consensus).
   - It is an automated screening score, **not a statutory legal certification**.
2. **Real-Product Logging:**
   - Inference runs automatically record cryptographic hashes and latency breakdowns in `data/interim/inference_logs/`.
   - No API keys, passwords, or personal credentials are ever logged.
