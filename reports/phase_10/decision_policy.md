# Phase 10 Evidence-Aware Compliance Decision Policy Specification

> **Project**: NitiMark-ML — Standalone ML Prediction Layer (SIH25057)  
> **Policy Module**: `src/decision/product_compliance_decision.py`  
> **Policy Version**: `evidence_aware_v1`

---

## 1. Statutory Grounding

Under the Legal Metrology (Packaged Commodities) Rules, 2011, pre-packaged retail commodities must carry mandatory declarations on the principal display panel or packaging container:
- **D01**: Complete name and address of manufacturer/packer/importer.
- **D02**: Common or generic name of the commodity.
- **D03**: Net quantity in standard SI units (weight, measure, or count).
- **D04**: Month and year of manufacture, packing, or import (mandatory on perishable/FMCG commodities; conditional exemption on durable non-perishable hardware).
- **D05**: Maximum Retail Price (MRP) inclusive of all applicable taxes.
- **D06**: Consumer care contact details (name, address, phone number, or email).
- **D07**: Country of origin (mandatory for all imported goods and under e-commerce display rules).

---

## 2. Formal Decision Logic

```
IF (Any Mandatory Declaration is ABSENT or PARTIAL with Confidence >= 0.50):
    RETURN "NON_COMPLIANT"
    PRIMARY_REASON = Highest-confidence negative declaration

ELSE IF (Any Mandatory Declaration is CANNOT_DETERMINE or Positive Confidence < 0.40):
    RETURN "UNCERTAIN"
    PRIMARY_REASON = Insufficient evidence for mandatory declaration

ELSE (All Applicable Mandatory Declarations are PRESENT with Confidence >= 0.40):
    RETURN "COMPLIANT"
    PRIMARY_REASON = All mandatory declarations affirmatively evidenced
```

---

## 3. Decision Trace Schema

Every decision outputs a transparent JSON structure:

```json
{
  "overall_prediction": "NON_COMPLIANT",
  "confidence": 0.81,
  "uncertainty": false,
  "primary_reason": "Statutory declaration D06 (Consumer Care Details) is ABSENT.",
  "decision_basis": [
    {
      "declaration": "D06",
      "name": "Consumer Care Details",
      "status": "ABSENT",
      "confidence": 0.81
    }
  ]
}
```
