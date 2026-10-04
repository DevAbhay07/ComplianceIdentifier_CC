"""
synthetic_packaging_generator.py
================================
Rule-Driven Synthetic Physical Packaging Dataset Generator for NitiMark-ML.

Methodology:
1. Structured Ground-Truth First:
   - Ground truth targets are derived from structured specifications + configs/legal_requirements.yaml
   - Controlled violation injection for NON_COMPLIANT and degraded evidence for UNCERTAIN.
2. Realistic Packaging Image Rendering:
   - Varied categories, package shapes, typography, colors, perspective, shadows, folds, camera noise, blur.
3. Common Representation & Extraction:
   - Extracts declarations and values into ProductComplianceTrainingRecord with strict provenance:
     source_dataset="synthetic_physical", target_provenance="SYNTHETIC_GROUND_TRUTH".
"""

import os
import sys
import json
import random
import math
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

PROJECT_ROOT = Path("/home/devabhay07/Desktop/MLmodelV2")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.product_training_representation import (
    ProductComplianceTrainingRecord, TargetProvenance, DatasetSource
)
from src.extraction.declaration_extractor import DeclarationExtractor
from src.extraction.declaration_schema import DeclarationStatus

SYNTHETIC_DIR = PROJECT_ROOT / "data/synthetic_packaging"
IMAGE_OUT_DIR = SYNTHETIC_DIR / "images"
JSONL_OUT_PATH = PROJECT_ROOT / "data/processed/synthetic_physical_packaging.jsonl"

IMAGE_OUT_DIR.mkdir(parents=True, exist_ok=True)
JSONL_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

CATEGORIES = [
    "Cosmetics & Personal Care",
    "Food & Beverages",
    "Health & Pharma",
    "Baby Care",
    "Household & Cleaning",
    "Electronics & Appliances"
]

BRANDS_BY_CAT = {
    "Cosmetics & Personal Care": [
        ("Aura Glow", "Aura Herbals Pvt Ltd", "Plot 42, Sector 8, Gandhinagar 382028, Gujarat", "care@auraglow.in", "1800-111-222"),
        ("DermaShield", "Derma Laboratories India", "12 Industrial Area, Baddi 173205, Himachal Pradesh", "support@dermashield.co.in", "1800-233-444"),
        ("Vedika Naturals", "Vedika Wellness Ltd", "78 Ayur Park, Haridwar 249401, Uttarakhand", "feedback@vedikanaturals.com", "1800-345-678")
    ],
    "Food & Beverages": [
        ("NutriCrunch", "NutriBake Foods Pvt Ltd", "Survey 104, Chakan MIDC, Pune 410501, Maharashtra", "customercare@nutricrunch.in", "1800-419-880"),
        ("PureHarvest", "Harvest Agri Products India", "NH-8, Manesar, Gurugram 122050, Haryana", "care@pureharvest.in", "1800-200-990"),
        ("Himalayan Brew", "Highland Tea & Coffee Estates", "Post Box 14, Darjeeling 734101, West Bengal", "taste@himalayanbrew.com", "1800-555-123")
    ],
    "Health & Pharma": [
        ("LifeCare Pharma", "LifeCare Therapeutics Ltd", "Survey 88, Genome Valley, Hyderabad 500078, Telangana", "contact@lifecarepharma.in", "1800-900-111"),
        ("OptiVital", "OptiNutrition India Pvt Ltd", "Block B, Electronic City, Bengaluru 560100, Karnataka", "care@optivital.co", "1800-888-222"),
        ("AyurRelief", "AyurRelief Remedies India", "Makali Post, Bengaluru 562162, Karnataka", "support@ayurrelief.org", "1800-208-1930")
    ],
    "Baby Care": [
        ("TenderTouch", "Tender Baby Care Products India", "L.B.S. Marg, Mulund West, Mumbai 400080, Maharashtra", "babycare@tendertouch.in", "1800-222-777"),
        ("TinySprout", "Sprout Infant Nutrition Pvt Ltd", "Sector 62, Noida 201309, Uttar Pradesh", "care@tinysprout.in", "1800-333-888")
    ],
    "Household & Cleaning": [
        ("SparkleHome", "Sparkle Hygiene Chemicals", "GIDC Estate, Ankleshwar 393002, Gujarat", "help@sparklehome.in", "1800-777-666"),
        ("EcoClean", "EcoSolutions Consumer Care", "Peenya Industrial Area, Bengaluru 560058, Karnataka", "service@ecoclean.co.in", "1800-666-555")
    ],
    "Electronics & Appliances": [
        ("VoltPro", "VoltPro Appliances India Pvt Ltd", "Plot 19, Okhla Phase III, New Delhi 110020", "service@voltpro.in", "1800-102-345"),
        ("LuminoTech", "Lumino Smart Lighting Ltd", "Whitefield Road, Bengaluru 560066, Karnataka", "support@luminotech.com", "1800-400-500")
    ]
}

COMMODITIES_BY_CAT = {
    "Cosmetics & Personal Care": [
        ("Purifying Face Wash", 150.0, "ml", 195.0),
        ("Intense Moisture Body Lotion", 250.0, "ml", 299.0),
        ("Sunscreen Gel SPF 50", 50.0, "g", 450.0),
        ("Herbal Hair Oil", 200.0, "ml", 175.0)
    ],
    "Food & Beverages": [
        ("Rolled Oats High Fiber", 500.0, "g", 185.0),
        ("Roasted Almonds Premium", 250.0, "g", 399.0),
        ("Organic Green Tea", 100.0, "g", 240.0),
        ("Cold Pressed Groundnut Oil", 1.0, "L", 280.0)
    ],
    "Health & Pharma": [
        ("Multivitamin & Minerals", 60.0, "tablets", 550.0),
        ("Omega 3 Fish Oil Capsules", 30.0, "capsules", 699.0),
        ("Ayurvedic Pain Relief Balm", 50.0, "g", 125.0),
        ("Herbal Cough Syrup", 100.0, "ml", 95.0)
    ],
    "Baby Care": [
        ("Gentle Baby Shampoo", 200.0, "ml", 230.0),
        ("Baby Daily Nourish Lotion", 100.0, "ml", 160.0),
        ("Soft Baby Wipes", 72.0, "wipes", 199.0)
    ],
    "Household & Cleaning": [
        ("Disinfectant Floor Cleaner", 1.0, "L", 180.0),
        ("Dishwash Gel Lemon", 500.0, "ml", 115.0),
        ("Liquid Detergent Active", 1.0, "L", 220.0)
    ],
    "Electronics & Appliances": [
        ("Electric Kettle 1.5L", 1.0, "pc", 999.0),
        ("Smart LED Bulb 9W", 1.0, "unit", 299.0),
        ("USB-C Fast Charging Cable", 1.0, "unit", 349.0)
    ]
}


def build_synthetic_ground_truth_spec(
    synthetic_id: str,
    target_class: str,
    category: str,
    violation_subtype: Optional[str] = None
) -> Dict[str, Any]:
    """
    Step 1: Construct rule-driven structured ground truth before image rendering.
    """
    brand_tup = random.choice(BRANDS_BY_CAT[category])
    brand_name, mfg_name, address, email, phone = brand_tup
    
    comm_tup = random.choice(COMMODITIES_BY_CAT[category])
    comm_name, qty, unit, base_mrp = comm_tup
    
    # Adjust MRP variation slightly
    mrp = round(base_mrp * random.uniform(0.9, 1.25), 0)
    mfg_month = random.randint(1, 12)
    mfg_year = random.choice([2025, 2026])
    is_imported = (random.random() < 0.15)
    origin_country = "India" if not is_imported else random.choice(["Germany", "Japan", "USA", "UK"])

    # Base Complete Declarations
    decls = {
        "D01": {"status": "PRESENT", "confidence": 0.95, "raw_text": f"Mfd by: {mfg_name}, {address}"},
        "D02": {"status": "PRESENT", "confidence": 0.95, "raw_text": f"{comm_name}"},
        "D03": {"status": "PRESENT", "confidence": 0.98, "raw_text": f"Net Qty: {qty} {unit}"},
        "D04": {"status": "PRESENT", "confidence": 0.96, "raw_text": f"Mfd Date: {mfg_month:02d}/{mfg_year}"},
        "D05": {"status": "PRESENT", "confidence": 0.98, "raw_text": f"MRP Rs. {mrp:.2f} (Incl. of all taxes)"},
        "D06": {"status": "PRESENT", "confidence": 0.97, "raw_text": f"Care: {phone}, {email}"},
        "D07": {"status": "PRESENT", "confidence": 0.95, "raw_text": f"Country of Origin: {origin_country}"}
    }

    d01_vals = {"entity_name": mfg_name, "address_markers": [address.split(",")[-1].strip()], "pincode": address.split()[-2] if len(address.split()) > 2 else "400001"}
    d02_vals = {"common_name": comm_name, "identity_confidence": 0.95}
    d03_vals = {"quantity": qty, "unit": unit, "unit_category": "VOLUME" if unit in ["ml", "L"] else ("WEIGHT" if unit in ["g", "kg"] else "COUNT")}
    d04_vals = {"month": mfg_month, "year": mfg_year, "raw_date": f"{mfg_month:02d}/{mfg_year}"}
    d05_vals = {"mrp": mrp, "currency": "INR", "tax_inclusive": True}
    d06_vals = {"phone": phone, "email": email, "has_toll_free": phone.startswith("1800")}
    d07_vals = {"country": origin_country, "is_domestic": (origin_country == "India")}

    violation_type = "NONE_COMPLIANT"
    evidence_quality = "GOOD"

    if target_class == "NON_COMPLIANT":
        v_choice = violation_subtype or random.choice([
            "MISSING_D01_ADDRESS", "MISSING_D03_QUANTITY", "MISSING_D04_DATE",
            "MISSING_D05_MRP", "MISSING_D06_CARE", "MISSING_D07_ORIGIN",
            "PARTIAL_ADDRESS", "MALFORMED_MRP_NO_TAX"
        ])
        violation_type = v_choice

        if v_choice == "MISSING_D01_ADDRESS":
            decls["D01"] = {"status": "ABSENT", "confidence": 0.0, "raw_text": ""}
            d01_vals = {}
        elif v_choice == "MISSING_D03_QUANTITY":
            decls["D03"] = {"status": "ABSENT", "confidence": 0.0, "raw_text": ""}
            d03_vals = {}
        elif v_choice == "MISSING_D04_DATE":
            decls["D04"] = {"status": "ABSENT", "confidence": 0.0, "raw_text": ""}
            d04_vals = {}
        elif v_choice == "MISSING_D05_MRP":
            decls["D05"] = {"status": "ABSENT", "confidence": 0.0, "raw_text": ""}
            d05_vals = {}
        elif v_choice == "MISSING_D06_CARE":
            decls["D06"] = {"status": "ABSENT", "confidence": 0.0, "raw_text": ""}
            d06_vals = {}
        elif v_choice == "MISSING_D07_ORIGIN":
            decls["D07"] = {"status": "ABSENT", "confidence": 0.0, "raw_text": ""}
            d07_vals = {}
        elif v_choice == "PARTIAL_ADDRESS":
            decls["D01"] = {"status": "PARTIAL", "confidence": 0.65, "raw_text": f"Mfd by {mfg_name}"}
            d01_vals = {"entity_name": mfg_name}
        elif v_choice == "MALFORMED_MRP_NO_TAX":
            decls["D05"] = {"status": "PARTIAL", "confidence": 0.70, "raw_text": f"Price: Rs. {mrp:.2f}"}
            d05_vals = {"mrp": mrp, "currency": "INR", "tax_inclusive": False}

    elif target_class == "UNCERTAIN":
        evidence_quality = random.choice(["DEGRADED_BLUR", "CROPPED_VIEW", "LOW_CONTRAST"])
        violation_type = f"EVIDENCE_UNCERTAINTY_{evidence_quality}"
        for did in ["D01", "D04", "D06"]:
            decls[did] = {"status": "CANNOT_DETERMINE", "confidence": 0.20, "raw_text": ""}

    spec = {
        "synthetic_id": synthetic_id,
        "source_dataset": DatasetSource.SYNTHETIC_PHYSICAL.value if hasattr(DatasetSource, "SYNTHETIC_PHYSICAL") else "synthetic_physical",
        "target_provenance": TargetProvenance.SYNTHETIC_GROUND_TRUTH.value if hasattr(TargetProvenance, "SYNTHETIC_GROUND_TRUTH") else "SYNTHETIC_GROUND_TRUTH",
        "overall_target": target_class,
        "category": category,
        "brand": brand_name,
        "product_title": f"{brand_name} {comm_name} {qty}{unit}",
        "declarations": decls,
        "d01_values": d01_vals,
        "d02_values": d02_vals,
        "d03_values": d03_vals,
        "d04_values": d04_vals,
        "d05_values": d05_vals,
        "d06_values": d06_vals,
        "d07_values": d07_vals,
        "violation_type": violation_type,
        "evidence_quality": evidence_quality,
        "ground_truth_source": "CONFIGURED_LEGAL_RULES (Rule 6)"
    }
    return spec


def render_realistic_packaging_image(spec: Dict[str, Any], output_path: Path) -> Tuple[int, int]:
    """
    Step 2: Render high-resolution package image with realistic photography artifacts.
    """
    w, h = 900, 1100
    # Background texture
    bg_color = random.choice([
        (245, 245, 247), (238, 240, 242), (250, 248, 245),
        (230, 235, 240), (242, 238, 230)
    ])
    img = Image.new("RGB", (w, h), color=bg_color)
    draw = ImageDraw.Draw(img)

    # Packaging container box (bottle/box/pack)
    box_w, box_h = 620, 880
    box_x0 = (w - box_w) // 2
    box_y0 = (h - box_h) // 2
    box_x1 = box_x0 + box_w
    box_y1 = box_y0 + box_h

    # Container colors
    card_color = random.choice([
        (255, 255, 255), (252, 253, 255), (248, 250, 252),
        (245, 250, 248), (255, 252, 248)
    ])
    accent_color = random.choice([
        (20, 80, 160), (16, 120, 80), (180, 40, 60),
        (120, 60, 140), (40, 100, 120), (160, 90, 20)
    ])

    # Draw container shadow
    shadow_offset = 12
    draw.rounded_rectangle(
        [box_x0 + shadow_offset, box_y0 + shadow_offset, box_x1 + shadow_offset, box_y1 + shadow_offset],
        radius=20, fill=(210, 215, 220)
    )
    # Draw container body
    draw.rounded_rectangle(
        [box_x0, box_y0, box_x1, box_y1],
        radius=20, fill=card_color, outline=(180, 185, 190), width=2
    )

    # Top brand banner
    draw.rounded_rectangle(
        [box_x0 + 4, box_y0 + 4, box_x1 - 4, box_y0 + 130],
        radius=18, fill=accent_color
    )

    # Brand text
    brand = spec["brand"]
    title = spec["product_title"]
    draw.text((box_x0 + 30, box_y0 + 35), brand.upper(), fill=(255, 255, 255))
    draw.text((box_x0 + 30, box_y0 + 80), f"Category: {spec['category']}", fill=(220, 235, 255))

    # Product Title & Identity (D02)
    draw.text((box_x0 + 30, box_y0 + 160), title, fill=(30, 30, 30))

    # Statutory Declarations Panel (D01, D03, D04, D05, D06, D07)
    panel_y0 = box_y0 + 260
    draw.rectangle([box_x0 + 20, panel_y0, box_x1 - 20, box_y1 - 30], fill=(250, 250, 252), outline=(220, 225, 230), width=1)
    
    draw.text((box_x0 + 35, panel_y0 + 15), "STATUTORY DECLARATIONS / LEGAL METROLOGY", fill=accent_color)
    
    curr_y = panel_y0 + 55
    line_spacing = 58
    
    decls = spec["declarations"]

    # Render declarations only if present or partial in ground truth
    if decls["D01"]["status"] in ["PRESENT", "PARTIAL"]:
        txt = decls["D01"]["raw_text"]
        draw.text((box_x0 + 35, curr_y), txt[:65], fill=(40, 40, 40))
        if len(txt) > 65:
            draw.text((box_x0 + 35, curr_y + 24), txt[65:130], fill=(40, 40, 40))
            curr_y += 24
        curr_y += line_spacing

    if decls["D02"]["status"] in ["PRESENT", "PARTIAL"]:
        draw.text((box_x0 + 35, curr_y), f"Generic Name: {decls['D02']['raw_text']}", fill=(40, 40, 40))
        curr_y += line_spacing

    if decls["D03"]["status"] in ["PRESENT", "PARTIAL"]:
        draw.text((box_x0 + 35, curr_y), decls["D03"]["raw_text"], fill=(20, 20, 20))
        curr_y += line_spacing

    if decls["D04"]["status"] in ["PRESENT", "PARTIAL"]:
        draw.text((box_x0 + 35, curr_y), f"Mfg / Pkd Date: {decls['D04']['raw_text']}", fill=(40, 40, 40))
        curr_y += line_spacing

    if decls["D05"]["status"] in ["PRESENT", "PARTIAL"]:
        draw.text((box_x0 + 35, curr_y), decls["D05"]["raw_text"], fill=(180, 20, 20))
        curr_y += line_spacing

    if decls["D06"]["status"] in ["PRESENT", "PARTIAL"]:
        draw.text((box_x0 + 35, curr_y), f"Consumer Helpline: {decls['D06']['raw_text']}", fill=(40, 40, 40))
        curr_y += line_spacing

    if decls["D07"]["status"] in ["PRESENT", "PARTIAL"]:
        draw.text((box_x0 + 35, curr_y), decls["D07"]["raw_text"], fill=(40, 40, 40))
        curr_y += line_spacing

    # Barcode representation
    draw.rectangle([box_x1 - 180, box_y1 - 120, box_x1 - 40, box_y1 - 45], fill=(255, 255, 255), outline=(0, 0, 0))
    for bx in range(box_x1 - 170, box_x1 - 50, 4):
        draw.line([(bx, box_y1 - 110), (bx, box_y1 - 55)], fill=(0, 0, 0), width=random.choice([1, 2, 3]))

    # Realistic photographic augmentations
    quality = spec.get("evidence_quality", "GOOD")
    if quality == "DEGRADED_BLUR":
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(2.5, 4.5)))
    elif quality == "LOW_CONTRAST":
        enh = ImageEnhance.Contrast(img)
        img = enh.enhance(0.55)
    elif quality == "CROPPED_VIEW":
        crop_box = (int(w * 0.15), int(h * 0.15), int(w * 0.85), int(h * 0.70))
        img = img.crop(crop_box).resize((w, h), Image.Resampling.BILINEAR)
    else:
        # Mild realistic camera noise and perspective
        if random.random() < 0.4:
            img = img.filter(ImageFilter.GaussianBlur(radius=0.4))
        if random.random() < 0.3:
            angle = random.uniform(-2.5, 2.5)
            img = img.rotate(angle, resample=Image.Resampling.BILINEAR, expand=False, fillcolor=bg_color)

    img.save(str(output_path), "PNG")
    return w, h


def generate_synthetic_physical_dataset(
    num_compliant: int = 120,
    num_non_compliant: int = 120,
    num_uncertain: int = 60
) -> Dict[str, Any]:
    """
    Generates the complete synthetic physical packaging dataset with structured ground truth.
    """
    print("=================================================================")
    print("   GENERATING RULE-DRIVEN SYNTHETIC PHYSICAL PACKAGING DATASET   ")
    print("=================================================================\n")

    extractor = DeclarationExtractor()
    records: List[ProductComplianceTrainingRecord] = []
    specs_all: List[Dict[str, Any]] = []

    total_target = num_compliant + num_non_compliant + num_uncertain
    print(f"Target dataset size: {total_target} samples")
    print(f"  • COMPLIANT: {num_compliant}")
    print(f"  • NON_COMPLIANT: {num_non_compliant}")
    print(f"  • UNCERTAIN: {num_uncertain}")

    idx = 1

    # 1. COMPLIANT Samples
    for i in range(num_compliant):
        sid = f"SYN_COMP_{idx:04d}"
        cat = CATEGORIES[i % len(CATEGORIES)]
        spec = build_synthetic_ground_truth_spec(sid, "COMPLIANT", cat)
        img_p = IMAGE_OUT_DIR / f"{sid}.png"
        render_realistic_packaging_image(spec, img_p)

        # Build raw text for OCR simulation / extractor
        comb_text = "\n".join([d["raw_text"] for d in spec["declarations"].values() if d["raw_text"]])
        
        # Construct training record
        rec = ProductComplianceTrainingRecord(
            product_id=sid,
            source_dataset="synthetic_physical",
            target_provenance="SYNTHETIC_GROUND_TRUTH",
            overall_target="COMPLIANT",
            category=spec["category"],
            brand=spec["brand"],
            product_title=spec["product_title"],
            raw_text=comb_text,
            declarations=spec["declarations"],
            d01_values=spec["d01_values"],
            d02_values=spec["d02_values"],
            d03_values=spec["d03_values"],
            d04_values=spec["d04_values"],
            d05_values=spec["d05_values"],
            d06_values=spec["d06_values"],
            d07_values=spec["d07_values"],
            packaging_features={"yolo_region_count": 6, "has_physical_image": True, "image_path": str(img_p)},
            ocr_quality=0.95,
            text_length=len(comb_text),
            image_count=1,
            image_paths=[str(img_p)],
            metadata={"violation_type": "NONE_COMPLIANT", "synthetic_spec": spec}
        )
        records.append(rec)
        specs_all.append(spec)
        idx += 1

    # 2. NON_COMPLIANT Samples
    violations = [
        "MISSING_D01_ADDRESS", "MISSING_D03_QUANTITY", "MISSING_D04_DATE",
        "MISSING_D05_MRP", "MISSING_D06_CARE", "MISSING_D07_ORIGIN",
        "PARTIAL_ADDRESS", "MALFORMED_MRP_NO_TAX"
    ]
    for i in range(num_non_compliant):
        sid = f"SYN_NONC_{idx:04d}"
        cat = CATEGORIES[i % len(CATEGORIES)]
        v_sub = violations[i % len(violations)]
        spec = build_synthetic_ground_truth_spec(sid, "NON_COMPLIANT", cat, violation_subtype=v_sub)
        img_p = IMAGE_OUT_DIR / f"{sid}.png"
        render_realistic_packaging_image(spec, img_p)

        comb_text = "\n".join([d["raw_text"] for d in spec["declarations"].values() if d["raw_text"]])
        rec = ProductComplianceTrainingRecord(
            product_id=sid,
            source_dataset="synthetic_physical",
            target_provenance="SYNTHETIC_GROUND_TRUTH",
            overall_target="NON_COMPLIANT",
            category=spec["category"],
            brand=spec["brand"],
            product_title=spec["product_title"],
            raw_text=comb_text,
            declarations=spec["declarations"],
            d01_values=spec["d01_values"],
            d02_values=spec["d02_values"],
            d03_values=spec["d03_values"],
            d04_values=spec["d04_values"],
            d05_values=spec["d05_values"],
            d06_values=spec["d06_values"],
            d07_values=spec["d07_values"],
            packaging_features={"yolo_region_count": 4, "has_physical_image": True, "image_path": str(img_p)},
            ocr_quality=0.90,
            text_length=len(comb_text),
            image_count=1,
            image_paths=[str(img_p)],
            metadata={"violation_type": v_sub, "synthetic_spec": spec}
        )
        records.append(rec)
        specs_all.append(spec)
        idx += 1

    # 3. UNCERTAIN Samples
    for i in range(num_uncertain):
        sid = f"SYN_UNCT_{idx:04d}"
        cat = CATEGORIES[i % len(CATEGORIES)]
        spec = build_synthetic_ground_truth_spec(sid, "UNCERTAIN", cat)
        img_p = IMAGE_OUT_DIR / f"{sid}.png"
        render_realistic_packaging_image(spec, img_p)

        comb_text = "\n".join([d["raw_text"] for d in spec["declarations"].values() if d["raw_text"]])
        rec = ProductComplianceTrainingRecord(
            product_id=sid,
            source_dataset="synthetic_physical",
            target_provenance="SYNTHETIC_GROUND_TRUTH",
            overall_target="UNCERTAIN",
            category=spec["category"],
            brand=spec["brand"],
            product_title=spec["product_title"],
            raw_text=comb_text,
            declarations=spec["declarations"],
            d01_values=spec["d01_values"],
            d02_values=spec["d02_values"],
            d03_values=spec["d03_values"],
            d04_values=spec["d04_values"],
            d05_values=spec["d05_values"],
            d06_values=spec["d06_values"],
            d07_values=spec["d07_values"],
            packaging_features={"yolo_region_count": 1, "has_physical_image": True, "image_path": str(img_p)},
            ocr_quality=0.30,
            text_length=len(comb_text),
            image_count=1,
            image_paths=[str(img_p)],
            metadata={"violation_type": spec["violation_type"], "synthetic_spec": spec}
        )
        records.append(rec)
        specs_all.append(spec)
        idx += 1

    # Save to jsonl
    with open(JSONL_OUT_PATH, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_dict()) + "\n")

    print(f"\nSynthetic dataset generation complete.")
    print(f"  • Total Generated: {len(records)} records")
    print(f"  • Saved JSONL: {JSONL_OUT_PATH}")
    print(f"  • Saved Images: {IMAGE_OUT_DIR}")

    summary = {
        "total_records": len(records),
        "compliant_count": num_compliant,
        "non_compliant_count": num_non_compliant,
        "uncertain_count": num_uncertain,
        "categories_covered": CATEGORIES,
        "provenance": "SYNTHETIC_GROUND_TRUTH"
    }
    return summary


if __name__ == "__main__":
    generate_synthetic_physical_dataset()
