# Annotation Workflow — Step-by-Step Guide

## Prerequisites

You need:
- Python 3.12+ with the project virtual environment
- The project repository at `NitiMark-ML/`
- About 2-5 minutes per product (using draft acceleration)

---

## Draft Annotation Feature (NEW)

The tool now supports **semi-automated draft annotation**:
- Click **⚡ Generate Draft Annotation** to auto-fill all declarations
- Machine drafts are based on listing evidence (keyword/pattern matching)
- Review, correct, and save — **drafts are never auto-saved as ground truth**
- Click **↺ Reset Draft** to clear all prefilled values

**Critical**: Always review the machine draft before saving. The draft engine may miss context or misinterpret evidence.

---

## 1. Start the Environment

```bash
cd /home/devabhay07/Desktop/MLmodelV2
source venv/bin/activate
```

---

## 2. Start the Annotation Tool

```bash
streamlit run src/annotation_tool.py
```

The tool will open in your browser at `http://localhost:8501`.

To specify your annotator ID from the command line:
```bash
# Or set it in the sidebar once the tool opens
```

---

## 3. Annotate One Product

### 3.1 Select a Product

In the **sidebar**:
1. Enter your **Annotator ID** (e.g., `devabhay07`)
2. Set filter to **"Not annotated"** to see remaining products
3. Select a product from the dropdown

### 3.2 Review the Evidence

In the **main area**, read:
- **Product name**: What is this product?
- **Category**: Which product category?
- **Brand**: Is a brand listed? (Missing brands show ❌)
- **Retail Price**: Is a price available?
- **Product Specifications**: Read all key-value pairs carefully
- **Full Description**: Expand and read for additional details

### 3.3 (Optional) Review Machine Guidance

Expand the **"🤖 Machine Guidance"** section for automated hints.

> ⚠️ Machine guidance is NOT ground truth. It only highlights what keywords were found. You make the final decision.

### 3.4 Label Each Declaration

For each of the 7 declarations (D01 through D07):

| Declaration | What to Check |
|---|---|
| **D01** Manufacturer name/address | Is there a manufacturer name? Full address? Brand-only = PARTIAL |
| **D02** Generic commodity name | Is the product's generic name clear? (not just brand name) |
| **D03** Net quantity | Weight, volume, or count in product name or specifications? |
| **D04** Manufacture date | Month/year of manufacture? (Almost always CANNOT_DETERMINE) |
| **D05** MRP | Retail price listed? Is it explicitly MRP? |
| **D06** Consumer care | Phone/email/address for complaints? (Often ABSENT) |
| **D07** Country of origin | For imported products: is country stated? Domestic = NOT_APPLICABLE |

For each, select:

| Label | When to Use |
|---|---|
| `PRESENT` | Information is clearly and completely present |
| `ABSENT` | Information is definitely not present anywhere |
| `PARTIAL` | Some info exists but incomplete (e.g., brand but no address) |
| `NOT_APPLICABLE` | This requirement doesn't apply (e.g., D07 for domestic products) |
| `CANNOT_DETERMINE` | Not enough information to judge (e.g., D04 for listing data) |

Then select the **evidence source(s)** where you found (or didn't find) the information, and add any **notes** explaining your reasoning.

### 3.5 Set Overall Label

Review the colored summary of your per-declaration labels, then choose:

| Label | When to Use |
|---|---|
| `COMPLIANT` | ALL applicable declarations are PRESENT |
| `NON_COMPLIANT` | One or more applicable declarations are clearly ABSENT |
| `UNCERTAIN` | Some are PARTIAL/CANNOT_DETERMINE, but none clearly ABSENT |

Write overall notes explaining your reasoning.

Set your **confidence level** (LOW / MEDIUM / HIGH).

### 3.6 Save

Click **"💾 Save Annotation"**. The annotation is appended to `annotations/pilot_annotations.jsonl`.

Move to the next product.

---

## 4. Meaning of Every Label

### Per-Declaration Labels

| Label | Definition | Example |
|---|---|---|
| `PRESENT` | The required information is clearly stated in the listing | Net weight "250g" in specifications |
| `ABSENT` | The information is not present anywhere in available data | No customer care phone/email found anywhere |
| `PARTIAL` | Relevant but incomplete information exists | Brand name "Himalaya" exists but no manufacturer address |
| `NOT_APPLICABLE` | The requirement doesn't apply to this product type | Country of origin for a clearly domestic product |
| `CANNOT_DETERMINE` | Not enough data to make a judgment | Manufacturing date (never in e-commerce listings) |

### Overall Labels

| Label | Definition |
|---|---|
| `COMPLIANT` | All applicable mandatory declarations are fully present in the listing |
| `NON_COMPLIANT` | At least one applicable mandatory declaration is definitively absent |
| `UNCERTAIN` | Some declarations are partial or undeterminable; cannot confirm compliance |

---

## 5. How to Handle Missing Information

- **Missing brand field**: Mark D01 as `ABSENT` or `PARTIAL` depending on whether manufacturer info appears elsewhere (description, specs)
- **No price field**: Mark D05 as `ABSENT`
- **No date information**: Mark D04 as `CANNOT_DETERMINE` (this is expected and correct)
- **No customer care info**: Mark D06 as `ABSENT` (common for e-commerce listings)
- **Cannot tell if product is imported or domestic**: Mark D07 as `CANNOT_DETERMINE`

**Important**: "Cannot verify from available evidence" ≠ "legally non-compliant." Use `CANNOT_DETERMINE`, not `ABSENT`, when you genuinely don't know.

---

## 6. How to Handle Ambiguous Products

Some products will be hard to classify. Examples:

- **Kitchen utensils with no packaging info**: These are physically packaged but specifications may not reflect packaging declarations → mark what's verifiable, use `CANNOT_DETERMINE` for the rest, set overall to `UNCERTAIN`
- **Combo/gift sets**: Check if individual item details are listed → if not, use `PARTIAL`
- **Digital or service products**: These may have accidentally been included in the category → mark all as `NOT_APPLICABLE`, note in overall notes

When in doubt, choose `UNCERTAIN` and write a clear note.

---

## 7. How to Resume Interrupted Annotation

The annotation tool automatically tracks which products have been annotated.

1. Start the tool: `streamlit run src/annotation_tool.py`
2. In the sidebar, set filter to **"Not annotated"**
3. Continue from where you left off

Annotations are saved immediately on each submit. No progress is lost.

---

## 8. How to Validate Annotations

After annotating some products, run the validator:

```bash
source venv/bin/activate
python src/data/validate_annotations.py
```

This will report:
- Any schema violations
- Invalid labels
- Duplicate annotations
- Statistics (label distribution, confidence distribution)

**The validator never modifies your annotations.** It only reports issues.

---

## 9. How to Export Annotation Results

Annotations are stored in:
```
annotations/pilot_annotations.jsonl
```

Each line is a complete JSON annotation record including:
- Product ID and snapshot
- All 7 declaration labels with evidence
- Overall compliance label
- Annotator ID and timestamp
- Machine guidance (separately stored)

To view annotations:
```bash
# Count annotations
wc -l annotations/pilot_annotations.jsonl

# View a specific annotation
head -1 annotations/pilot_annotations.jsonl | python3 -m json.tool

# Extract all overall labels
python3 -c "
import json
with open('annotations/pilot_annotations.jsonl') as f:
    for line in f:
        ann = json.loads(line)
        print(f'{ann[\"product_id\"]}: {ann[\"overall_assessment\"][\"label\"]}')"
```

---

## Quick Reference

| Action | Command |
|---|---|
| Activate environment | `source venv/bin/activate` |
| Start annotation tool | `streamlit run src/annotation_tool.py` |
| Validate annotations | `python src/data/validate_annotations.py` |
| Count annotations | `wc -l annotations/pilot_annotations.jsonl` |
| Stop annotation tool | `Ctrl+C` in the terminal |
