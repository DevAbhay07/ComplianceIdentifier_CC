# Legal Metrology Packaging Annotation Guidelines (Phase 11)

**Project**: NitiMark — Automated Compliance Checker for Legal Metrology Declarations (SIH25057)  
**Standard**: Legal Metrology (Packaged Commodities) Rules, 2011 (Rule 6)  
**Version**: 2.0 (Packaging Domain Evidence-First Edition)

---

## 1. Core Principles: Evidence-First Annotation

1. **Physical Evidence Dominance**: Declarations must be assessed strictly from observable physical packaging images or validated OCR extractions.
2. **Missing View $\ne$ Absence**: If an image view does not show all sides of a 3D package (e.g. only front face visible, back/bottom not shown), do NOT label missing declarations as `ABSENT`. The correct label is `CANNOT_DETERMINE`.
3. **Absence Criteria**: Label `ABSENT` ONLY when the complete relevant declaration panel (e.g., Principal Display Panel or Information Panel) is fully visible, high resolution, and the mandatory declaration is demonstrably omitted.
4. **Zero Fabrication**: Annotators and AI agents must never infer or fabricate declaration contents not physically visible in the image.

---

## 2. Declaration States (D01–D07)

### D01 — Manufacturer, Packer, or Importer Name & Address
- **PRESENT**: Full legal name and complete physical address (street, city, state, PIN code) or imported details (importer name + address).
- **PARTIAL**: Brand name or company name present, but address is incomplete (e.g., city only without full address or PIN).
- **ABSENT**: Information panel visible and zero manufacturer/importer/packer details appear.
- **NOT_APPLICABLE**: Never (Mandatory for all packaged commodities).
- **CANNOT_DETERMINE**: Declaration panel obscured, blurry, or back/side panel missing from image set.

### D02 — Generic or Common Commodity Name
- **PRESENT**: Standard commodity identity clearly stated on the Principal Display Panel (e.g., "Face Wash", "Basmati Rice", "Wireless Earbuds").
- **PARTIAL**: Ambiguous marketing title without standard common commodity noun.
- **ABSENT**: Complete PDP visible, but generic name missing.
- **NOT_APPLICABLE**: Never (Mandatory).
- **CANNOT_DETERMINE**: Title/front panel cut off or illegible.

### D03 — Net Quantity in Standard Metric Units
- **PRESENT**: Clear quantity stated in SI units (g, kg, ml, l, m, cm) or piece count (N / U / units) conforming to Legal Metrology unit rules.
- **PARTIAL**: Non-standard units (e.g. ounces, lbs) or number without units.
- **ABSENT**: Declaration area visible but no net quantity declared.
- **NOT_APPLICABLE**: Exempt bulk goods or specific industrial items.
- **CANNOT_DETERMINE**: Bottom or front quantity zone truncated/blurry.

### D04 — Month and Year of Manufacture / Packing / Import
- **PRESENT**: Clear date of manufacture, packing, or import declared in standard notation (e.g., "Mfd: 11/2025", "Packed on: Oct 2025").
- **PARTIAL**: Incomplete date notation (e.g., year only, or illegible batch stamp).
- **ABSENT**: Packaging panel visible, batch/stamp zone clearly blank.
- **NOT_APPLICABLE**: Certain non-perishable hardware/tools where statutory exemption applies under Rule 26.
- **CANNOT_DETERMINE**: Date stamp zone outside camera view or overprinted unreadably.

### D05 — Maximum Retail Price (MRP) & Unit Sale Price
- **PRESENT**: Explicit MRP formatted with "MRP Rs. / ₹", "Inclusive of all taxes", and Unit Sale Price (USP) where mandated.
- **PARTIAL**: Numeric price present without "Incl. of all taxes" or without currency marker.
- **ABSENT**: Price panel visible, zero price stated.
- **NOT_APPLICABLE**: Commodities meant for institutional or industrial consumers.
- **CANNOT_DETERMINE**: Price stamp cropped out or obscured.

### D06 — Consumer Care / Grievance Helpline
- **PRESENT**: Name, address, phone number (or toll-free helpline), and email address of grievance officer / consumer care cell.
- **PARTIAL**: Email only or phone only (missing full contact combination).
- **ABSENT**: Back panel / info panel fully legible, but no consumer care contact exists.
- **NOT_APPLICABLE**: Never (Mandatory for all consumer packaged goods under Rule 6(1)(n)).
- **CANNOT_DETERMINE**: Contact panel not captured in image.

### D07 — Country of Origin
- **PRESENT**: Explicit "Made in [Country]", "Country of Origin: India", or "Manufactured in [Country]".
- **PARTIAL**: Implied origin without statutory declaration.
- **ABSENT**: Packaging fully legible, origin omitted.
- **NOT_APPLICABLE**: Domestic goods where manufacturer address explicitly identifies Indian location (under specific state exemptions) or imported goods where importer details suffice under older sub-rules.
- **CANNOT_DETERMINE**: Origin text panel not visible.

---

## 3. Product-Level Compliance Decision Policy

```
                             [ Physical Product ]
                                      │
                 ┌────────────────────┴────────────────────┐
                 ▼                                         ▼
   Any mandatory declaration                 All mandatory declarations
       verified ABSENT                      verified PRESENT or NOT_APPLICABLE
                 │                                         │
                 ▼                                         ▼
        ┌─────────────────┐                       ┌─────────────────┐
        │  NON_COMPLIANT  │                       │    COMPLIANT    │
        └─────────────────┘                       └─────────────────┘
                                      │
                                      ▼
                        Ambiguous evidence or key
                      panels CANNOT_DETERMINE / PARTIAL
                                      │
                                      ▼
                             ┌─────────────────┐
                             │    UNCERTAIN    │
                             └─────────────────┘
```

- **COMPLIANT**: All applicable mandatory declarations (D01–D07) have sufficient positive physical evidence (`PRESENT` or `NOT_APPLICABLE`), and zero mandatory declaration is `ABSENT`.
- **NON_COMPLIANT**: At least one applicable mandatory declaration is conclusively established as `ABSENT` on the visible packaging.
- **UNCERTAIN**: Missing declaration views (`CANNOT_DETERMINE`), poor OCR quality (<0.50), or unresolved partial evidence preventing a definitive legal finding.
