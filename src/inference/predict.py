"""
predict.py
==========
Unified Standalone CLI interface for NitiMark-ML predictions on real-world e-commerce products.
Supports image-only, text-only, multimodal, structured JSON, and offline inference.

Usage Examples:
    python src/inference/predict.py --image path/to/image.png
    python src/inference/predict.py --input examples/sample_product.json --offline
    python src/inference/predict.py --text "Colgate Toothpaste 100g MRP Rs 99" --offline
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import json
import argparse
from typing import Dict, Any, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from src.inference.image_validator import validate_product_image
from src.inference.vision_ocr import get_vision_provider
from src.inference.evidence_parser import parse_product_evidence
from src.inference.inference_features import build_inference_feature_vectors
from src.inference.schema_validator import validate_inference_feature_schema
from src.inference.model_loader import load_frozen_models
from src.inference.declaration_predictor import predict_all_declarations
from src.inference.real_product_test import check_product_novelty
from src.inference.trace import InferenceTracer


def run_standalone_prediction(
    image_path: Optional[str] = None,
    title: str = "",
    description: str = "",
    specifications: Optional[Dict[str, Any]] = None,
    retail_price: Optional[float] = None,
    discounted_price: Optional[float] = None,
    brand: Optional[str] = None,
    category: Optional[str] = None,
    offline_mode: bool = False,
    provider_name: str = "gemini",
    api_key_env: str = "VISION_API_KEY",
    model_name: Optional[str] = None,
    log_dir: str = "data/interim/inference_logs",
    enable_logging: bool = True
) -> Dict[str, Any]:
    """
    Executes the end-to-end standalone inference pipeline for a single product.
    """
    tracer = InferenceTracer(log_dir=log_dir, enabled=enable_logging)
    warnings = []

    input_desc = {
        "image_path": image_path,
        "title": title,
        "text": f"{title} {description}".strip()
    }

    # 1. Check Product Novelty / Collision
    novelty = check_product_novelty(title=title)
    if novelty.get("warning"):
        warnings.append(novelty["warning"])

    # 2. Vision / OCR Extraction
    raw_ocr_text = ""
    ocr_result = {
        "status": "NOT_ATTEMPTED",
        "raw_text": "",
        "provider": "NONE",
        "model": "NONE",
        "latency_sec": 0.0,
        "error": None
    }

    if image_path:
        tracer.mark_start("image_validation")
        img_val = validate_product_image(image_path)
        tracer.mark_end("image_validation")

        if not img_val["valid"]:
            tracer.finalize(input_desc, ocr_result, {}, {}, status="INVALID_IMAGE")
            return {
                "project": "NitiMark-ML",
                "phase": "phase_9",
                "status": "INVALID_IMAGE",
                "error": img_val["error"],
                "declarations": {},
                "warnings": warnings,
                "trace_id": tracer.trace_id
            }

        if offline_mode:
            ocr_result["status"] = "OFFLINE_MODE_SKIPPED"
            if not title and not description:
                tracer.finalize(input_desc, ocr_result, {}, {}, status="VISION_REQUIRED")
                return {
                    "project": "NitiMark-ML",
                    "phase": "phase_9",
                    "status": "VISION_REQUIRED",
                    "error": "VISION_REQUIRED: Image provided in offline mode without text metadata. Enable online mode or supply text metadata.",
                    "declarations": {},
                    "warnings": warnings,
                    "trace_id": tracer.trace_id
                }
        else:
            # Online Vision Extraction
            tracer.mark_start("vision_ocr")
            api_key = os.environ.get(api_key_env)
            provider = get_vision_provider(
                provider_type=provider_name,
                api_key=api_key,
                model_name=model_name
            )
            ocr_res = provider.extract_from_image(image_path)
            tracer.mark_end("vision_ocr")
            ocr_result = ocr_res

            if ocr_res["status"] == "SUCCESS":
                raw_ocr_text = ocr_res["raw_text"]
            else:
                warnings.append(f"OCR_NOTICE: OCR extraction status was '{ocr_res['status']}'. Reason: {ocr_res.get('error')}")

    # 2b. Visual Localization via YOLOv8 (Phase 11)
    visual_regions = []
    yolo_latency_ms = 0.0
    if image_path:
        tracer.mark_start("yolo_localization")
        try:
            from src.vision.yolo_detector import YOLOPackagingDetector
            detector = YOLOPackagingDetector()
            yolo_res = detector.detect(image_path)
            visual_regions = yolo_res.get("detections", [])
            yolo_latency_ms = yolo_res.get("latency_ms", 0.0)
        except Exception as e:
            visual_regions = []
        tracer.mark_end("yolo_localization")

    # 3. Evidence Extraction from Text & Metadata
    tracer.mark_start("evidence_parsing")
    evidence_res = parse_product_evidence(
        title=title,
        description=description,
        specifications=specifications,
        raw_ocr_text=raw_ocr_text
    )
    tracer.mark_end("evidence_parsing")

    # 4. Feature Adaptation
    tracer.mark_start("feature_adaptation")
    try:
        X_set_a, X_set_b, raw_feats = build_inference_feature_vectors(
            title=title,
            description=description,
            raw_ocr_text=raw_ocr_text,
            specifications=specifications,
            retail_price=retail_price,
            discounted_price=discounted_price,
            brand=brand,
            category=category,
            image_count=1 if image_path else 0
        )
    except Exception as e:
        tracer.finalize(input_desc, ocr_result, {}, {}, status="FEATURE_ADAPTER_ERROR")
        return {
            "project": "NitiMark-ML",
            "phase": "phase_11",
            "status": "FEATURE_ADAPTER_ERROR",
            "error": f"Inference feature adapter failure: {str(e)}",
            "declarations": {},
            "warnings": warnings,
            "trace_id": tracer.trace_id
        }
    tracer.mark_end("feature_adaptation")

    # 5. Feature Schema Validation
    tracer.mark_start("schema_validation")
    schema_val = validate_inference_feature_schema(X_set_a, X_set_b)
    tracer.mark_end("schema_validation")

    if not schema_val["valid"]:
        tracer.finalize(input_desc, ocr_result, {}, schema_val, status="INFERENCE_SCHEMA_ERROR")
        return {
            "project": "NitiMark-ML",
            "phase": "phase_11",
            "status": "INFERENCE_SCHEMA_ERROR",
            "error": f"Schema mismatch: {'; '.join(schema_val['errors'])}",
            "declarations": {},
            "warnings": warnings,
            "trace_id": tracer.trace_id
        }

    # 6. Model Loading & Inference
    tracer.mark_start("ml_inference")
    try:
        loaded_models = load_frozen_models()
        predictions = predict_all_declarations(loaded_models, X_set_a, X_set_b)
    except Exception as e:
        tracer.finalize(input_desc, ocr_result, {}, {}, status="INFERENCE_EXECUTION_ERROR")
        return {
            "project": "NitiMark-ML",
            "phase": "phase_11",
            "status": "INFERENCE_EXECUTION_ERROR",
            "error": f"Prediction error: {str(e)}",
            "overall_prediction": "UNCERTAIN",
            "confidence": 0.0,
            "uncertainty": {"status": "HIGH", "score": 1.0, "coverage": 0.0},
            "declarations": {},
            "warnings": warnings,
            "trace_id": tracer.trace_id
        }
    tracer.mark_end("ml_inference")

    # 7. Evidence-Aware Product Compliance Decision
    tracer.mark_start("product_decision")
    from src.decision.product_compliance_decision import EvidenceAwareComplianceDecisionEngine
    decision_engine = EvidenceAwareComplianceDecisionEngine()
    decision_res = decision_engine.evaluate_product(
        declaration_predictions=predictions,
        extracted_evidence=evidence_res.get("declarations"),
        category=category,
        visual_regions=visual_regions
    )
    tracer.mark_end("product_decision")

    # Finalize Trace
    trace_record = tracer.finalize(
        input_descriptors=input_desc,
        ocr_result=ocr_result,
        predictions=predictions,
        diagnostics=evidence_res["diagnostics"],
        status="SUCCESS"
    )

    return {
        "project": "NitiMark-ML",
        "phase": "phase_11",
        "status": "SUCCESS",
        "trace_id": trace_record["trace_id"],
        "overall_prediction": decision_res["overall_prediction"],
        "confidence": decision_res["confidence"],
        "uncertainty": decision_res["uncertainty"],
        "primary_reason": decision_res["primary_reason"],
        "decision_basis": decision_res["decision_basis"],
        "visual_regions": visual_regions,
        "model_id": "multimodal_evidence_decision_v2",
        "model_version": "2.0.0",
        "input": {
            "image_path": image_path,
            "title": title,
            "has_specs": bool(specifications),
            "offline_mode": offline_mode
        },
        "vision_ocr": {
            "status": ocr_result["status"],
            "provider": ocr_result.get("provider", "NONE"),
            "model": ocr_result.get("model", "NONE"),
            "latency_sec": ocr_result.get("latency_sec", 0.0),
            "extracted_text_preview": (raw_ocr_text[:150] + "...") if len(raw_ocr_text) > 150 else raw_ocr_text
        },
        "extracted_evidence": evidence_res["declarations"],
        "declarations": predictions,
        "latencies_sec": trace_record["latencies_sec"],
        "warnings": warnings,
        "disclaimer": "Automated Legal Metrology compliance screening. Not a statutory judicial determination."
    }


def render_terminal_output(result: Dict[str, Any]):
    """Renders human-readable formatted terminal output."""
    print("\n" + "=" * 70)
    print("      NITIMARK-ML — PRODUCT COMPLIANCE PREDICTION (PHASE 10)")
    print("=" * 70)

    if result["status"] != "SUCCESS":
        print(f"\n[ERROR] Inference failed with status: {result['status']}")
        print(f"Details: {result.get('error')}\n")
        return

    # Overall Compliance Verdict
    ov_pred = result.get("overall_prediction", "UNCERTAIN")
    conf = result.get("confidence", 0.0)
    is_unc = result.get("uncertainty", False)
    p_reason = result.get("primary_reason", "No primary reason provided.")

    print(f"\n[OVERALL COMPLIANCE VERDICT]")
    print(f"  * Status        : {ov_pred}")
    print(f"  * Model         : Evidence-Aware Decision Engine v1")
    print(f"  * Confidence    : {conf:.2f}")
    print(f"  * Uncertainty   : {'YES (Abstained)' if is_unc else 'NO'}")
    print(f"  * Primary Basis : {p_reason}")

    # Input summary
    inp = result["input"]
    print(f"\n[Input Summary]")
    if inp.get("image_path"):
        print(f"  * Product Image : {inp['image_path']}")
    if inp.get("title"):
        print(f"  * Product Title : {inp['title']}")
    print(f"  * Mode          : {'OFFLINE' if inp.get('offline_mode') else 'ONLINE'}")

    # Vision summary
    v = result["vision_ocr"]
    print(f"\n[Vision / OCR Status]")
    print(f"  * Status        : {v['status']}")
    print(f"  * Provider/Model: {v['provider']} ({v['model']})")
    if v.get("extracted_text_preview"):
        print(f"  * Extracted Text: {v['extracted_text_preview']}")

    # Internal Declarations Reasoning
    print("\n" + "-" * 70)
    print(f"{'INTERNAL DECLARATION (D01-D07)':<35} | {'PREDICTION':<16} | {'CONF':<6} | {'MODEL TYPE'}")
    print("-" * 70)

    for did, d_data in result["declarations"].items():
        name = d_data["declaration_name"][:33]
        pred = d_data["prediction"]
        c_val = f"{d_data['confidence']:.2f}"
        m_type = "BASELINE" if d_data["is_baseline"] else d_data["model_family"]
        print(f"{did} {name:<31} | {pred:<16} | {c_val:<6} | {m_type}")

    print("-" * 70)

    # Latencies
    lats = result.get("latencies_sec", {})
    total_lat = lats.get("total_pipeline", 0.0)
    ml_lat = lats.get("ml_inference", 0.0)
    print(f"\n[Performance & Latency]")
    print(f"  * Total Latency : {total_lat:.4f}s (ML Inference: {ml_lat:.4f}s)")
    print(f"  * Trace ID      : {result['trace_id']}")

    if result.get("warnings"):
        print("\n[Warnings / Notes]")
        for w in result["warnings"]:
            print(f"  ! {w}")

    print("\n" + "=" * 70)
    print("DISCLAIMER: " + result["disclaimer"])
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="NitiMark-ML Standalone Legal Metrology Inference CLI")
    parser.add_argument("--image", type=str, help="Path to product image file (JPEG, PNG, WEBP)")
    parser.add_argument("--text", type=str, help="Raw product text or description")
    parser.add_argument("--title", type=str, help="Product title/name")
    parser.add_argument("--description", type=str, help="Product description")
    parser.add_argument("--input", type=str, help="Path to product JSON input file")
    parser.add_argument("--output", type=str, help="Path to write JSON prediction output")
    parser.add_argument("--offline", action="store_true", help="Run in offline mode without Vision API calls")
    parser.add_argument("--provider", type=str, default="gemini", choices=["gemini", "mock"], help="OCR Provider")
    parser.add_argument("--api-key-env", type=str, default="VISION_API_KEY", help="Env variable containing API key")
    parser.add_argument("--model", type=str, default="gemini-3.6-flash", help="Vision model name")
    parser.add_argument("--json", action="store_true", help="Output only machine-readable JSON to stdout")

    args = parser.parse_args()

    image_path = args.image
    title = args.title or ""
    description = args.description or args.text or ""
    specifications = {}
    retail_price = None
    discounted_price = None
    brand = None
    category = None

    # Load JSON input if provided
    if args.input:
        if not os.path.exists(args.input):
            print(f"Error: Input JSON file not found at {args.input}", file=sys.stderr)
            sys.exit(1)
        with open(args.input, "r", encoding="utf-8") as f:
            data = json.load(f)
            title = data.get("title") or title
            description = data.get("description") or description
            image_path = data.get("image_path") or image_path
            specifications = data.get("specifications") or specifications
            retail_price = data.get("retail_price")
            discounted_price = data.get("discounted_price")
            brand = data.get("brand")
            category = data.get("category")

    if not image_path and not title and not description:
        print("Error: At least one input modality (--image, --text, --title, or --input) must be provided.", file=sys.stderr)
        sys.exit(1)

    result = run_standalone_prediction(
        image_path=image_path,
        title=title,
        description=description,
        specifications=specifications,
        retail_price=retail_price,
        discounted_price=discounted_price,
        brand=brand,
        category=category,
        offline_mode=args.offline,
        provider_name=args.provider,
        api_key_env=args.api_key_env,
        model_name=args.model
    )

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        render_terminal_output(result)


if __name__ == "__main__":
    main()
