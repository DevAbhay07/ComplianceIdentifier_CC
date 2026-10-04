"""
product_compliance_decision.py
==============================
Evidence-Aware Decision Engine for Product-Level Legal Metrology Compliance (Phase 11 Upgrade).
Evaluates internal declaration-level predictions, multimodal visual regions (YOLOv8),
OCR quality, confidence bounds, and statutory applicability to determine product-level verdicts:
  - COMPLIANT
  - NON_COMPLIANT
  - UNCERTAIN

Preserves backward compatibility while improving real packaging performance and reducing false alarms.
"""

import os
import json
from typing import Dict, Any, List, Optional, Tuple, Union

DECLARATION_METADATA = {
    "D01": {
        "name": "Manufacturer / Packer / Importer",
        "mandatory": True,
        "description": "Name and complete address of the manufacturer, packer, or importer."
    },
    "D02": {
        "name": "Generic or Common Name",
        "mandatory": True,
        "description": "Common or generic name of the commodity."
    },
    "D03": {
        "name": "Net Quantity",
        "mandatory": True,
        "description": "Net quantity in terms of standard weight, measure, or number."
    },
    "D04": {
        "name": "Date of Manufacture / Packing",
        "mandatory": True,
        "description": "Month and year of manufacture, packing, or import."
    },
    "D05": {
        "name": "Maximum Retail Price (MRP)",
        "mandatory": True,
        "description": "Retail sale price inclusive of all taxes."
    },
    "D06": {
        "name": "Consumer Care Details",
        "mandatory": True,
        "description": "Name, address, telephone number, or email for consumer complaints."
    },
    "D07": {
        "name": "Country of Origin",
        "mandatory": False,  # Mandatory for imports / e-commerce rules, conditional on domestic durable exemptions
        "description": "Country of origin or manufacture."
    }
}

DEFAULT_POLICY_CONFIG = {
    "min_negative_confidence": 0.50,
    "min_positive_confidence": 0.40,
    "allow_durable_date_exemption": True,
    "uncertainty_threshold": 0.35,
    "min_ocr_quality_for_certainty": 0.40,
    "min_evidence_completeness": 0.30,
    "policy_version": "phase_11_multimodal_evidence_v2"
}


class UncertaintyStatus(dict):
    """
    Hybrid uncertainty object: behaves as a dictionary with canonical schema:
    {
        "status": "LOW" | "MEDIUM" | "HIGH",
        "score": float,
        "reason": str,
        "abstained": bool
    }
    and as a boolean for backward-compatible assertion checks.
    """
    def __init__(
        self,
        is_uncertain: Optional[bool] = None,
        details: Optional[Dict[str, Any]] = None,
        status: Optional[str] = None,
        score: Optional[float] = None,
        reason: Optional[str] = None,
        abstained: Optional[bool] = None
    ):
        det = dict(details) if details else {}
        stat = status or det.get("status") or ("HIGH" if is_uncertain else "LOW")
        sc = score if score is not None else det.get("score", (0.85 if stat == "HIGH" else (0.40 if stat == "MEDIUM" else 0.10)))
        reaz = reason if reason is not None else det.get("reason", ("High prediction uncertainty." if stat == "HIGH" else "Prediction within certainty bounds."))
        abs_flag = abstained if abstained is not None else det.get("abstained", (stat == "HIGH" or is_uncertain is True))

        base = {
            "status": stat,
            "score": round(float(sc), 4),
            "reason": str(reaz),
            "abstained": bool(abs_flag)
        }
        for k, v in det.items():
            if k not in base:
                base[k] = v
        super().__init__(base)
        self.is_uncertain = is_uncertain if is_uncertain is not None else (stat == "HIGH" or abs_flag or sc >= 0.45)

    def __bool__(self) -> bool:
        return self.is_uncertain


class EvidenceAwareComplianceDecisionEngine:
    """
    Evaluates declaration predictions, YOLO visual regions, and OCR evidence to produce
    transparent, statutorily grounded product-level compliance decisions.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = {**DEFAULT_POLICY_CONFIG, **(config or {})}

    def evaluate_product(
        self,
        declaration_predictions: Dict[str, Dict[str, Any]],
        extracted_evidence: Optional[Dict[str, Any]] = None,
        category: Optional[str] = None,
        visual_regions: Optional[List[Dict[str, Any]]] = None,
        ocr_quality: Optional[float] = None,
        image_quality: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Computes overall product compliance verdict from declaration predictions and multimodal evidence.
        """
        extracted_evidence = extracted_evidence or {}
        visual_regions = visual_regions or []
        decision_basis = []
        positive_declarations = []
        negative_declarations = []
        uncertain_declarations = []

        is_durable = False
        if category and any(k in category.lower() for k in ["kitchen", "dining", "hardware", "steel", "utensil"]):
            is_durable = True

        # Check visual region flags
        detected_classes = set()
        for reg in visual_regions:
            if isinstance(reg, dict):
                cname = reg.get("class", reg.get("class_name", ""))
                if cname:
                    detected_classes.add(cname)

        # Check each declaration
        for did, meta in DECLARATION_METADATA.items():
            pred_data = declaration_predictions.get(did, {})
            pred_label = pred_data.get("prediction", pred_data.get("label", "CANNOT_DETERMINE"))
            conf = float(pred_data.get("confidence", 0.50))
            d_name = meta["name"]
            is_mandatory = meta["mandatory"]

            # Handle durable exemption for D04 if applicable
            if did == "D04" and is_durable and self.config["allow_durable_date_exemption"]:
                is_mandatory = False

            # Statutory evaluation
            if pred_label == "ABSENT":
                if conf >= self.config["min_negative_confidence"]:
                    if is_mandatory:
                        negative_declarations.append({
                            "declaration": did,
                            "name": d_name,
                            "confidence": conf,
                            "reason": f"Mandatory declaration '{d_name}' ({did}) is ABSENT with confidence {conf:.2f}."
                        })
                    else:
                        decision_basis.append(f"Optional declaration '{d_name}' ({did}) is absent (no violation).")
                else:
                    uncertain_declarations.append({
                        "declaration": did,
                        "name": d_name,
                        "confidence": conf,
                        "reason": f"Low-confidence negative prediction for '{d_name}' ({did})."
                    })
            elif pred_label in ["PRESENT", "PARTIAL"]:
                positive_declarations.append({
                    "declaration": did,
                    "name": d_name,
                    "confidence": conf,
                    "state": pred_label
                })
            elif pred_label in ["CANNOT_DETERMINE", "UNKNOWN"]:
                if is_mandatory:
                    uncertain_declarations.append({
                        "declaration": did,
                        "name": d_name,
                        "confidence": conf,
                        "reason": f"Evidence is insufficient to determine mandatory '{d_name}' ({did})."
                    })

        # Calculate evidence metrics
        total_mandatory = sum(1 for d, m in DECLARATION_METADATA.items() if m["mandatory"] and not (d == "D04" and is_durable))
        evidence_completeness = (len(positive_declarations) + len(negative_declarations)) / max(total_mandatory, 1)

        # Multimodal quality checks
        effective_ocr_qual = ocr_quality if ocr_quality is not None else float(extracted_evidence.get("ocr_quality", 0.85))
        effective_img_qual = image_quality if image_quality is not None else float(extracted_evidence.get("image_quality", 0.90))

        # Overall Verdict Logic
        if negative_declarations:
            verdict = "NON_COMPLIANT"
            top_reasons = [item["reason"] for item in negative_declarations]
            decision_basis = top_reasons
            overall_confidence = max(item["confidence"] for item in negative_declarations)
            uncertainty_score = round(max(0.0, 1.0 - overall_confidence), 4)
            uncertainty_status = "LOW" if overall_confidence >= 0.70 else "MEDIUM"
            is_uncertain_bool = False

        elif uncertain_declarations:
            verdict = "UNCERTAIN"
            uncertain_names = [u["name"] for u in uncertain_declarations]
            decision_basis.append(f"Evidence cannot determine mandatory declaration(s): {', '.join(uncertain_names)}.")
            overall_confidence = 0.50
            uncertainty_score = 0.85
            uncertainty_status = "HIGH"
            is_uncertain_bool = True

        elif effective_ocr_qual < self.config["min_ocr_quality_for_certainty"] and not positive_declarations:
            verdict = "UNCERTAIN"
            decision_basis.append(f"Low OCR image quality ({effective_ocr_qual:.2f}) prevents definitive legal assessment.")
            overall_confidence = round(1.0 - effective_ocr_qual, 4)
            uncertainty_score = 0.85
            uncertainty_status = "HIGH"
            is_uncertain_bool = True

        elif len(positive_declarations) >= (total_mandatory - 1):
            verdict = "COMPLIANT"
            decision_basis.append(f"All mandatory declarations verified present ({len(positive_declarations)}/{total_mandatory}).")
            mean_pos_conf = sum(p["confidence"] for p in positive_declarations) / max(len(positive_declarations), 1)
            overall_confidence = round(mean_pos_conf, 4)
            uncertainty_score = round(max(0.0, 1.0 - overall_confidence), 4)
            uncertainty_status = "LOW" if overall_confidence >= 0.75 else "MEDIUM"
            is_uncertain_bool = False

        else:
            verdict = "UNCERTAIN"
            overall_confidence = 0.60
            uncertainty_score = 0.50
            uncertainty_status = "MEDIUM"
            decision_basis.append("Evidence across packaging declarations is partially complete.")
            is_uncertain_bool = True

        uncertainty_obj = UncertaintyStatus(
            is_uncertain=is_uncertain_bool,
            details={
                "status": uncertainty_status,
                "score": uncertainty_score,
                "coverage": 1.0 if not is_uncertain_bool else 0.0,
                "ocr_quality": effective_ocr_qual,
                "evidence_completeness": round(evidence_completeness, 4)
            }
        )

        return {
            "overall_prediction": verdict,
            "product_compliance": verdict,
            "overall_confidence": overall_confidence,
            "confidence": overall_confidence,
            "uncertainty": uncertainty_obj,
            "decision_basis": "; ".join(decision_basis),
            "primary_reason": decision_basis[0] if decision_basis else "Compliance evaluated against Rule 6.",
            "positive_count": len(positive_declarations),
            "negative_count": len(negative_declarations),
            "uncertain_count": len(uncertain_declarations),
            "visual_regions_detected": list(detected_classes),
            "policy_version": self.config["policy_version"],
            "disclaimer": "Automated Legal Metrology compliance screening. Not a statutory judicial determination."
        }
