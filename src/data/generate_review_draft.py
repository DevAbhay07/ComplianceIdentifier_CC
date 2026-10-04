"""
generate_review_draft.py
==========================
AI-assisted DRAFTING engine for NitiMark Legal Metrology review.

IMPORTANT:
The drafting function is an ASSISTANT ONLY.
It must NEVER become the final annotator.
Machine drafts != Human verified annotations.
AI drafts are NOT ground truth and must NOT be used as training labels.

Approved Legal Schema:
  D01: Manufacturer/Packer/Importer Name & Address
  D02: Common/Generic Commodity Name
  D03: Net Quantity
  D04: Month/Year of Manufacture/Packing
  D05: MRP (Retail Price)
  D06: Consumer Care Details
  D07: Country of Origin

Valid Declaration Values:
  PRESENT, ABSENT, PARTIAL, NOT_APPLICABLE, CANNOT_DETERMINE

Valid Overall Suggested Values:
  COMPLIANT, NON_COMPLIANT, UNCERTAIN

Valid Evidence Sources:
  PRODUCT_IMAGE, PRODUCT_DESCRIPTION, PRODUCT_SPECIFICATIONS,
  PRODUCT_TITLE, PRODUCT_PAGE, MULTIPLE_SOURCES, NO_EVIDENCE, CANNOT_VERIFY
"""

import ast
import base64
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

# Auto-load .env file if present
def _load_env_file():
    env_file = os.path.join(WORKSPACE_ROOT, ".env")
    if os.path.exists(env_file):
        try:
            with open(env_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip('"\'')
                        if k and not os.environ.get(k):
                            os.environ[k] = v
        except Exception:
            pass

_load_env_file()

import requests
from pydantic import BaseModel, Field, field_validator



# -------------------------------------------------------------------------
# Constants & Approved Legal Schema
# -------------------------------------------------------------------------
DECLS = ["D01", "D02", "D03", "D04", "D05", "D06", "D07"]

DECL_NAMES = {
    "D01": "Manufacturer/Packer/Importer Name & Address",
    "D02": "Common/Generic Name of Commodity",
    "D03": "Net Quantity",
    "D04": "Month/Year of Manufacture/Packing",
    "D05": "MRP (Retail Price)",
    "D06": "Consumer Care Details",
    "D07": "Country of Origin",
}

VALID_DECL_LABELS = {
    "PRESENT",
    "ABSENT",
    "PARTIAL",
    "NOT_APPLICABLE",
    "CANNOT_DETERMINE",
}

VALID_OVERALL_LABELS = {
    "COMPLIANT",
    "NON_COMPLIANT",
    "UNCERTAIN",
}

VALID_EVIDENCE_SOURCES = {
    "PRODUCT_IMAGE",
    "PRODUCT_DESCRIPTION",
    "PRODUCT_SPECIFICATIONS",
    "PRODUCT_TITLE",
    "PRODUCT_PAGE",
    "MULTIPLE_SOURCES",
    "NO_EVIDENCE",
    "CANNOT_VERIFY",
}

# Supported Gemini models verified against Google Generative Language API
SUPPORTED_DRAFT_MODELS = {
    "Gemini 3.6 Flash": "gemini-3.6-flash",
    "Gemini 2.5 Flash": "gemini-2.5-flash",
    "Gemini 2.5 Pro": "gemini-2.5-pro",
    "Gemini 2.5 Flash-Lite": "gemini-2.5-flash-lite",
    "Gemini 3.7 Flash": "gemini-3.7-flash",
}

DEFAULT_GEMINI_MODEL = os.environ.get("GEMINI_DRAFT_MODEL", "gemini-2.5-flash")
IMAGE_FETCH_TIMEOUT_SECONDS = 4
API_REQUEST_TIMEOUT_SECONDS = 30



# -------------------------------------------------------------------------
# Pydantic Schema for Strict Output Validation
# -------------------------------------------------------------------------
class DeclarationDraft(BaseModel):
    draft_label: str
    evidence_source: str
    evidence_text: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    reasoning: str = ""

    @field_validator("draft_label")
    @classmethod
    def validate_label(cls, v: str) -> str:
        clean = str(v).strip().upper()
        if clean not in VALID_DECL_LABELS:
            raise ValueError(f"Invalid declaration label: {v}. Must be one of {VALID_DECL_LABELS}")
        return clean

    @field_validator("evidence_source")
    @classmethod
    def validate_source(cls, v: str) -> str:
        clean = str(v).strip().upper()
        if clean not in VALID_EVIDENCE_SOURCES:
            # Map common variants gracefully
            mapping = {
                "IMAGE": "PRODUCT_IMAGE",
                "IMAGES": "PRODUCT_IMAGE",
                "DESCRIPTION": "PRODUCT_DESCRIPTION",
                "SPECIFICATIONS": "PRODUCT_SPECIFICATIONS",
                "SPECS": "PRODUCT_SPECIFICATIONS",
                "TITLE": "PRODUCT_TITLE",
                "NONE": "NO_EVIDENCE",
                "UNAVAILABLE": "CANNOT_VERIFY",
            }
            if clean in mapping:
                return mapping[clean]
            raise ValueError(f"Invalid evidence source: {v}. Must be one of {VALID_EVIDENCE_SOURCES}")
        return clean

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        try:
            val = float(v)
            return max(0.0, min(1.0, round(val, 3)))
        except Exception:
            return 0.5


class OverallDraft(BaseModel):
    suggested_label: str
    reasoning: str = ""

    @field_validator("suggested_label")
    @classmethod
    def validate_overall(cls, v: str) -> str:
        clean = str(v).strip().upper()
        if clean not in VALID_OVERALL_LABELS:
            raise ValueError(f"Invalid overall label: {v}. Must be one of {VALID_OVERALL_LABELS}")
        return clean


class AIDraftOutput(BaseModel):
    D01: DeclarationDraft
    D02: DeclarationDraft
    D03: DeclarationDraft
    D04: DeclarationDraft
    D05: DeclarationDraft
    D06: DeclarationDraft
    D07: DeclarationDraft
    overall: OverallDraft


# -------------------------------------------------------------------------
# Evidence Collection Utilities
# -------------------------------------------------------------------------
def parse_image_urls(raw_urls: Any) -> List[str]:
    """Safely parse image URL string or list into a list of clean URLs."""
    if not raw_urls or str(raw_urls).strip() in ("nan", "None", ""):
        return []

    if isinstance(raw_urls, list):
        return [str(u).strip().strip('"\'') for u in raw_urls if str(u).strip()]

    raw_str = str(raw_urls).strip()
    try:
        parsed = ast.literal_eval(raw_str)
        if isinstance(parsed, list):
            return [str(u).strip().strip('"\'') for u in parsed if str(u).strip()]
        return [str(parsed).strip().strip('"\'')]
    except Exception:
        # Fallback regex / split
        parts = [p.strip().strip('[]"\'') for p in raw_str.split(",") if p.strip()]
        return [p for p in parts if p.startswith("http")]


def fetch_image_as_base64(url: str, timeout: int = IMAGE_FETCH_TIMEOUT_SECONDS) -> Optional[Tuple[str, str]]:
    """
    Download an image and return (base64_data, mime_type).
    Returns None if download fails, times out, or returns non-image.
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) NitiMark/1.0"}
        resp = requests.get(url, headers=headers, timeout=timeout)
        if resp.status_code == 200 and resp.content:
            mime = resp.headers.get("Content-Type", "image/jpeg").split(";")[0].strip()
            if not mime.startswith("image/"):
                mime = "image/jpeg"
            b64 = base64.b64encode(resp.content).decode("utf-8")
            return b64, mime
    except Exception:
        pass
    return None


def collect_product_evidence(product: Dict[str, Any], max_images: int = 2) -> Dict[str, Any]:
    """
    Assemble all available listing metadata, specifications, description, and images.
    Identifies what sources are available vs unavailable.
    """
    available_sources = []
    unavailable_sources = []

    # 1. Title
    title = str(product.get("product_name", "")).strip()
    if title and title != "N/A":
        available_sources.append("PRODUCT_TITLE")
    else:
        unavailable_sources.append("PRODUCT_TITLE")

    # 2. Specifications
    specs = product.get("parsed_specs", {})
    if isinstance(specs, dict) and len(specs) > 0:
        available_sources.append("PRODUCT_SPECIFICATIONS")
    else:
        specs = {}
        unavailable_sources.append("PRODUCT_SPECIFICATIONS")

    # 3. Description
    description = str(product.get("description", "")).strip()
    if description and description != "N/A":
        available_sources.append("PRODUCT_DESCRIPTION")
    else:
        unavailable_sources.append("PRODUCT_DESCRIPTION")

    # 4. Brand & Price
    brand = str(product.get("brand", "")).strip()
    retail_price = product.get("retail_price")
    discounted_price = product.get("discounted_price")

    # 5. Images
    image_urls = parse_image_urls(product.get("image_urls", ""))
    fetched_images = []
    for u in image_urls[:max_images]:
        img_res = fetch_image_as_base64(u)
        if img_res:
            fetched_images.append({
                "url": u,
                "base64": img_res[0],
                "mime_type": img_res[1],
            })

    if fetched_images:
        available_sources.append("PRODUCT_IMAGE")
    else:
        unavailable_sources.append("PRODUCT_IMAGE")

    return {
        "product_id": str(product.get("product_id", "")),
        "product_title": title,
        "category": str(product.get("top_category", "N/A")),
        "category_tree": str(product.get("product_category_tree", "N/A")),
        "brand": brand or "Not specified",
        "retail_price": retail_price,
        "discounted_price": discounted_price,
        "specifications": specs,
        "description": description[:3500] if description else "",
        "product_url": str(product.get("product_url", "")),
        "available_sources": available_sources,
        "unavailable_sources": unavailable_sources,
        "fetched_images": fetched_images,
        "total_images_in_listing": len(image_urls),
    }


# -------------------------------------------------------------------------
# Gemini Prompt Construction (Evidence-First & Compliant with Schema)
# -------------------------------------------------------------------------
SYSTEM_INSTRUCTION = """You are an AI annotation assistant for the NitiMark Legal Metrology Compliance system (Rule 6 of the Legal Metrology Packaged Commodities Rules, 2011).

CRITICAL GROUND RULES:
1. You are an ASSISTANT ONLY, not the final annotator. Your output is a draft for human verification.
2. Use ONLY the provided product evidence (title, specs, description, images).
3. Do NOT invent or fabricate missing information.
4. If evidence is insufficient to verify a declaration, use CANNOT_DETERMINE. Do not guess or assume.
5. Do NOT create new legal requirements. Use strictly the approved declarations D01-D07.
6. Return concise, factual evidence text and reasoning supporting each proposed label.
7. AI Draft Confidence must be a float between 0.0 and 1.0 reflecting the clarity and directness of the evidence found.

APPROVED DECLARATION SCHEMA:
- D01: Name and complete address of Manufacturer / Packer / Importer.
  - PRESENT: Complete manufacturer/packer name AND full postal address found.
  - PARTIAL: Brand or company name present, but complete address/pincode missing.
  - ABSENT: No manufacturer or brand details found.
  - CANNOT_DETERMINE: Insufficient evidence to judge.
- D02: Common or generic name of commodity (e.g., 'Lip Balm', 'Coffee Mug', 'Shampoo').
  - PRESENT: Clear generic commodity name identified.
  - PARTIAL: Ambiguous or overly vague commodity name.
  - ABSENT: No generic name identifiable.
- D03: Net quantity in standard SI units (weight, volume, length, or count).
  - PRESENT: Explicit net quantity present (e.g., '500 g', '250 ml', 'Pack of 2', '1 Piece').
  - PARTIAL: Vague/non-standard quantity.
  - ABSENT: No quantity stated.
- D04: Month and year of manufacture / packing / import.
  - PRESENT: Explicit month/year stated.
  - CANNOT_DETERMINE: No manufacturing date found in listing/image (this is standard for e-commerce listings).
  - ABSENT: Explicitly omitted where expected.
- D05: Maximum Retail Price (MRP) inclusive of all taxes.
  - PRESENT: Retail price field is present.
  - ABSENT: No price provided.
- D06: Consumer care details (phone number, email, address for consumer complaints).
  - PRESENT: Explicit consumer helpline phone/email/address found.
  - PARTIAL: Generic website support link only.
  - ABSENT: No consumer care details found.
- D07: Country of origin.
  - PRESENT: Country of origin explicitly stated (e.g., 'Made in India', 'Country of Origin: China').
  - NOT_APPLICABLE: Domestic product where specific origin declaration is legally exempt or evident.
  - CANNOT_DETERMINE: Not stated and cannot be determined.

VALID DECLARATION LABELS:
- PRESENT
- ABSENT
- PARTIAL
- NOT_APPLICABLE
- CANNOT_DETERMINE

VALID EVIDENCE SOURCES:
- PRODUCT_IMAGE
- PRODUCT_DESCRIPTION
- PRODUCT_SPECIFICATIONS
- PRODUCT_TITLE
- PRODUCT_PAGE
- MULTIPLE_SOURCES
- NO_EVIDENCE
- CANNOT_VERIFY

OVERALL SUGGESTED LABELS:
- COMPLIANT: All mandatory declarations are PRESENT (or NOT_APPLICABLE).
- NON_COMPLIANT: One or more mandatory declarations are ABSENT or fail legal requirements.
- UNCERTAIN: Critical declarations are CANNOT_DETERMINE or PARTIAL.
"""


def build_user_prompt(evidence: Dict[str, Any]) -> str:
    """Construct structured text prompt with evidence breakdown."""
    specs_formatted = json.dumps(evidence["specifications"], indent=2) if evidence["specifications"] else "None"
    
    prompt = f"""Analyze the following product evidence and generate a structured draft annotation for D01-D07 and an overall suggestion.

PRODUCT EVIDENCE:
- Product ID: {evidence['product_id']}
- Product Title: {evidence['product_title']}
- Category: {evidence['category']} (Tree: {evidence['category_tree']})
- Brand: {evidence['brand']}
- Retail Price: {evidence['retail_price']} (Discounted: {evidence['discounted_price']})
- Product URL: {evidence['product_url']}
- Available Sources: {', '.join(evidence['available_sources'])}
- Unavailable Sources: {', '.join(evidence['unavailable_sources']) if evidence['unavailable_sources'] else 'None'}
- Product Images Attached: {len(evidence['fetched_images'])} of {evidence['total_images_in_listing']} in listing

PRODUCT SPECIFICATIONS:
{specs_formatted}

PRODUCT DESCRIPTION:
{evidence['description'] if evidence['description'] else 'No description available'}

REQUIRED OUTPUT FORMAT:
You must respond with a single valid JSON object strictly matching this schema:
{{
  "D01": {{
    "draft_label": "PRESENT|ABSENT|PARTIAL|NOT_APPLICABLE|CANNOT_DETERMINE",
    "evidence_source": "PRODUCT_IMAGE|PRODUCT_DESCRIPTION|PRODUCT_SPECIFICATIONS|PRODUCT_TITLE|PRODUCT_PAGE|MULTIPLE_SOURCES|NO_EVIDENCE|CANNOT_VERIFY",
    "evidence_text": "Exact snippet of text found in evidence (or empty if none)",
    "confidence": 0.85,
    "reasoning": "Concise factual reason"
  }},
  "D02": {{
    "draft_label": "...",
    "evidence_source": "...",
    "evidence_text": "...",
    "confidence": 0.95,
    "reasoning": "..."
  }},
  "D03": {{
    "draft_label": "...",
    "evidence_source": "...",
    "evidence_text": "...",
    "confidence": 0.90,
    "reasoning": "..."
  }},
  "D04": {{
    "draft_label": "...",
    "evidence_source": "...",
    "evidence_text": "...",
    "confidence": 0.80,
    "reasoning": "..."
  }},
  "D05": {{
    "draft_label": "...",
    "evidence_source": "...",
    "evidence_text": "...",
    "confidence": 0.95,
    "reasoning": "..."
  }},
  "D06": {{
    "draft_label": "...",
    "evidence_source": "...",
    "evidence_text": "...",
    "confidence": 0.85,
    "reasoning": "..."
  }},
  "D07": {{
    "draft_label": "...",
    "evidence_source": "...",
    "evidence_text": "...",
    "confidence": 0.85,
    "reasoning": "..."
  }},
  "overall": {{
    "suggested_label": "COMPLIANT|NON_COMPLIANT|UNCERTAIN",
    "reasoning": "Summary of overall compliance reasoning"
  }}
}}
"""
    return prompt


# -------------------------------------------------------------------------
# Gemini API Client with REST & Multimodal Support
# -------------------------------------------------------------------------
def call_gemini_api(
    prompt: str,
    images: List[Dict[str, Any]],
    api_key: Optional[str] = None,
    model: str = DEFAULT_GEMINI_MODEL,
) -> Tuple[bool, Optional[str], Optional[str], bool]:
    """
    Call Gemini REST endpoint directly using requests.
    Returns: (success: bool, raw_response_text: Optional[str], error_message: Optional[str], is_quota_error: bool)
    """
    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        return False, None, "Missing Gemini API Key. Please configure GEMINI_API_KEY or enter your key in the sidebar.", False

    # Clean model name (strip models/ prefix if present)
    clean_model = model.replace("models/", "").strip()
    if not clean_model:
        clean_model = DEFAULT_GEMINI_MODEL

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{clean_model}:generateContent?key={key}"

    # Build parts: text prompt + base64 images
    parts = [{"text": prompt}]
    for img in images:
        if img.get("base64") and img.get("mime_type"):
            parts.append({
                "inline_data": {
                    "mime_type": img["mime_type"],
                    "data": img["base64"],
                }
            })

    payload = {
        "system_instruction": {
            "parts": [{"text": SYSTEM_INSTRUCTION}]
        },
        "contents": [
            {
                "role": "user",
                "parts": parts,
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "response_mime_type": "application/json",
        },
    }

    try:
        headers = {"Content-Type": "application/json"}
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=API_REQUEST_TIMEOUT_SECONDS,
        )

        if response.status_code == 429:
            err_msg = (
                f"Quota exhausted / rate limit reached for '{clean_model}' (HTTP 429). "
                f"Please select an alternate drafting model (e.g., Gemini 3.6 Flash, Gemini 2.5 Pro) from the sidebar."
            )
            return False, None, err_msg, True

        if response.status_code != 200:
            err_msg = f"Gemini API error for '{clean_model}' (status {response.status_code}): {response.text[:300]}"
            return False, None, err_msg, False

        res_json = response.json()
        candidates = res_json.get("candidates", [])
        if not candidates:
            return False, None, f"Gemini ({clean_model}) returned empty candidates list.", False

        candidate = candidates[0]
        content = candidate.get("content", {})
        cand_parts = content.get("parts", [])
        if not cand_parts:
            return False, None, f"Gemini ({clean_model}) returned empty content parts.", False

        text_content = cand_parts[0].get("text", "")
        return True, text_content, None, False

    except requests.exceptions.Timeout:
        return False, None, f"Gemini ({clean_model}) API request timed out after {API_REQUEST_TIMEOUT_SECONDS}s.", False
    except requests.exceptions.ConnectionError:
        return False, None, "Network connection error while reaching Gemini API.", False
    except Exception as e:
        return False, None, f"Unexpected error during API call: {str(e)}", False


# -------------------------------------------------------------------------
# High-Level Drafting Function
# -------------------------------------------------------------------------
def generate_ai_draft(
    product: Dict[str, Any],
    api_key: Optional[str] = None,
    model: str = DEFAULT_GEMINI_MODEL,
) -> Dict[str, Any]:
    """
    Generate an AI draft recommendation for declarations D01-D07 using the specified model.

    Returns a standardized dictionary:
    {
        "draft_generated": bool,
        "model_name": str,
        "draft_timestamp": str,
        "evidence_sources_available": list,
        "evidence_sources_unavailable": list,
        "declarations": {
            "D01": {"draft_label": ..., "evidence_source": ..., "evidence_text": ..., "confidence": ..., "reasoning": ...},
            ...
        },
        "suggested_overall": {"suggested_label": ..., "reasoning": ...},
        "is_quota_error": bool,
        "error": Optional[str],
    }
    """
    clean_model = model.replace("models/", "").strip() if model else DEFAULT_GEMINI_MODEL
    evidence = collect_product_evidence(product)
    user_prompt = build_user_prompt(evidence)

    success, raw_text, error_msg, is_quota = call_gemini_api(
        prompt=user_prompt,
        images=evidence["fetched_images"],
        api_key=api_key,
        model=clean_model,
    )

    timestamp = datetime.now(timezone.utc).isoformat()

    if not success or not raw_text:
        return {
            "draft_generated": False,
            "model_name": clean_model,
            "draft_timestamp": timestamp,
            "product_id": evidence["product_id"],
            "evidence_sources_available": evidence["available_sources"],
            "evidence_sources_unavailable": evidence["unavailable_sources"],
            "declarations": {},
            "suggested_overall": {},
            "is_quota_error": is_quota,
            "error": error_msg or "Unknown failure during draft generation",
        }

    # Clean JSON text if wrapped in markdown code fence
    cleaned_text = raw_text.strip()
    if cleaned_text.startswith("```json"):
        cleaned_text = cleaned_text[7:]
    elif cleaned_text.startswith("```"):
        cleaned_text = cleaned_text[3:]
    if cleaned_text.endswith("```"):
        cleaned_text = cleaned_text[:-3]
    cleaned_text = cleaned_text.strip()

    try:
        parsed_json = json.loads(cleaned_text)
        validated = AIDraftOutput(**parsed_json)
    except Exception as parse_err:
        return {
            "draft_generated": False,
            "model_name": clean_model,
            "draft_timestamp": timestamp,
            "product_id": evidence["product_id"],
            "evidence_sources_available": evidence["available_sources"],
            "evidence_sources_unavailable": evidence["unavailable_sources"],
            "declarations": {},
            "suggested_overall": {},
            "is_quota_error": False,
            "error": f"Failed to parse / validate Gemini structured response: {str(parse_err)}",
            "raw_output": raw_text[:500],
        }

    # Format validated output
    decls_dict = {}
    for did in DECLS:
        d_val = getattr(validated, did)
        decls_dict[did] = {
            "draft_label": d_val.draft_label,
            "evidence_source": d_val.evidence_source,
            "evidence_text": d_val.evidence_text,
            "confidence": d_val.confidence,
            "reasoning": d_val.reasoning,
        }

    return {
        "draft_generated": True,
        "model_name": clean_model,
        "draft_timestamp": timestamp,
        "product_id": evidence["product_id"],
        "evidence_sources_available": evidence["available_sources"],
        "evidence_sources_unavailable": evidence["unavailable_sources"],
        "declarations": decls_dict,
        "suggested_overall": {
            "suggested_label": validated.overall.suggested_label,
            "reasoning": validated.overall.reasoning,
        },
        "is_quota_error": False,
        "error": None,
    }


# -------------------------------------------------------------------------
# Standalone CLI Test
# -------------------------------------------------------------------------
def main():
    """CLI runner to test drafting engine on review products."""
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(description="Test AI drafting on review products")
    parser.add_argument("--test_sample", type=int, default=5, help="Number of products to test (default: 5)")
    parser.add_argument("--product_id", type=str, default="", help="Specific product ID to test")
    parser.add_argument("--api_key", type=str, default="", help="Optional Gemini API key")
    parser.add_argument("--model", type=str, default=DEFAULT_GEMINI_MODEL, help=f"Gemini model ID (default: {DEFAULT_GEMINI_MODEL})")
    args = parser.parse_args()

    REVIEW_CSV = "annotations/review_36.csv"
    PARSED_DATA = "data/interim/parsed_annotation_candidates.jsonl"

    print("=" * 70)
    print("PHASE 4.5C — AI DRAFTING ENGINE STANDALONE TEST")
    print("=" * 70)

    if not os.path.exists(REVIEW_CSV) or not os.path.exists(PARSED_DATA):
        print("Required review dataset or parsed data not found.")
        sys.exit(1)

    # Load parsed products
    products = {}
    with open(PARSED_DATA, "r") as f:
        for line in f:
            r = json.loads(line)
            products[str(r.get("product_id", ""))] = r

    # Load review dataset
    review_df = pd.read_csv(REVIEW_CSV)
    pids = review_df["product_id"].tolist()

    if args.product_id:
        target_pids = [args.product_id] if args.product_id in products else []
    else:
        target_pids = pids[: args.test_sample]

    print(f"Target products to test: {len(target_pids)}")
    print(f"Selected Model: {args.model}")
    print(f"API Key present: {bool(args.api_key or os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY'))}\n")

    for i, pid in enumerate(target_pids, 1):
        product = products.get(pid, {})
        print(f"--- [{i}/{len(target_pids)}] Product: {pid} | {product.get('product_name', 'N/A')[:50]} ---")
        
        # Test evidence collection
        evidence = collect_product_evidence(product)
        print(f"  Available sources: {evidence['available_sources']}")
        print(f"  Unavailable sources: {evidence['unavailable_sources']}")
        print(f"  Fetched images: {len(evidence['fetched_images'])}")

        # Run draft generator
        draft_res = generate_ai_draft(product, api_key=args.api_key, model=args.model)

        if draft_res["draft_generated"]:
            print(f"  ✅ AI Draft generated successfully (model: {draft_res['model_name']})")
            for did in DECLS:
                d = draft_res["declarations"][did]
                print(f"     {did}: {d['draft_label']} [Conf: {d['confidence']:.2f}] [Src: {d['evidence_source']}]")
                if d["evidence_text"]:
                    print(f"        Evidence: {d['evidence_text'][:70]}")
                print(f"        Reason: {d['reasoning']}")
            ov = draft_res["suggested_overall"]
            print(f"  Suggested Overall: {ov['suggested_label']} — {ov['reasoning']}\n")
        else:
            print(f"  ⚠️ Draft not generated: {draft_res['error']}\n")

    print("=" * 70)
    print("Standalone test completed successfully. No dataset files were modified.")


if __name__ == "__main__":
    main()

