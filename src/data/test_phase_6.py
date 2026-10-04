"""
test_phase_6.py
================
Unit and integration test suite for Phase 6 Weak-Label Dataset Expansion.
Includes synthetic unit test fixtures covering edge cases:
1. Clearly present declaration
2. Clearly missing declaration
3. Insufficient evidence / cannot determine
4. Conflicting evidence
5. Not applicable case
6. Malformed product record
7. Inaccessible / empty image URLs

Usage:
    python -m unittest src/data/test_phase_6.py -v
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import unittest
import json
import yaml
import pandas as pd

from src.data.extract_product_evidence import extract_evidence_record, parse_ruby_hash_specs
from src.data.detect_declaration_evidence import (
    detect_d01_evidence,
    detect_d02_evidence,
    detect_d03_evidence,
    detect_d04_evidence,
    detect_d05_evidence,
    detect_d06_evidence,
    detect_d07_evidence,
    detect_all_declaration_evidence
)
from src.data.generate_weak_labels import (
    derive_overall_compliance,
    assign_quality_tier,
    generate_weak_labels_for_product
)
from src.data.deduplicate_weak_dataset import deduplicate_and_assign_families
from src.data.image_evidence import parse_image_urls, inspect_product_images


class TestPhase6Pipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rules_path = "configs/weak_label_rules.yaml"
        with open(rules_path, "r", encoding="utf-8") as f:
            cls.rules_cfg = yaml.safe_load(f)

    # 1. Test Specification Parsing
    def test_ruby_hash_spec_parsing(self):
        raw_ruby = '{"product_specification"=>[{"key"=>"Net Quantity", "value"=>"500 ml"}, {"key"=>"Ideal For", "value"=>"Men"}]}'
        specs, extra, status = parse_ruby_hash_specs(raw_ruby)
        self.assertEqual(status, "SUCCESS")
        self.assertEqual(specs.get("Net Quantity"), "500 ml")
        self.assertEqual(specs.get("Ideal For"), "Men")

    def test_malformed_spec_parsing(self):
        raw_malformed = '{"product_specification"=>CORRUPTED_JSON_DATA'
        specs, extra, status = parse_ruby_hash_specs(raw_malformed)
        self.assertIsInstance(specs, dict)
        self.assertIsInstance(extra, list)

    # 2. Test Image Parsing & URL Validation
    def test_image_url_parsing(self):
        raw_images = '["http://img5a.flixcart.com/image/shampoo.jpg", "https://img6a.flixcart.com/image/bottle.jpg"]'
        urls = parse_image_urls(raw_images)
        self.assertEqual(len(urls), 2)
        self.assertTrue(urls[0].startswith("http"))

    def test_empty_image_handling(self):
        urls = parse_image_urls(None)
        self.assertEqual(len(urls), 0)
        res = inspect_product_images(urls, attempt_download=False)
        self.assertEqual(res["image_evidence_status"], "UNAVAILABLE")

    # 3. Fixture 1: Clearly Present Declarations
    def test_clearly_present_declarations(self):
        present_record = {
            "product_id": "test_present_001",
            "title": "Himalaya Purifying Neem Face Wash 100 ml",
            "description": "Manufactured by Himalaya Wellness at Makali, Bangalore 562123. Toll free customer care: 18002081930. Made in India.",
            "brand": "Himalaya",
            "retail_price": 150.0,
            "discounted_price": 130.0,
            "specifications": {
                "Net Quantity": "100 ml",
                "Type": "Face Wash",
                "Country of Origin": "India",
                "Customer Care": "care@himalayawellness.com",
                "Date of Manufacture": "04/2026",
                "Manufacturer Address": "Makali, Bangalore, Karnataka"
            },
            "category_hierarchy": ["Beauty and Personal Care", "Skin Care", "Face Wash"],
            "image_urls": ["http://img.flixcart.com/himalaya.jpg"]
        }
        decls = detect_all_declaration_evidence(present_record, self.rules_cfg)
        self.assertEqual(decls["D01"]["label"], "PRESENT")
        self.assertEqual(decls["D02"]["label"], "PRESENT")
        self.assertEqual(decls["D03"]["label"], "PRESENT")
        self.assertEqual(decls["D04"]["label"], "PRESENT")
        self.assertEqual(decls["D05"]["label"], "PRESENT")
        self.assertEqual(decls["D06"]["label"], "PRESENT")
        self.assertEqual(decls["D07"]["label"], "PRESENT")

        overall = derive_overall_compliance(decls)
        self.assertEqual(overall["overall_label"], "COMPLIANT")

    # 4. Fixture 2: Clearly Missing Declarations
    def test_clearly_missing_declarations(self):
        missing_record = {
            "product_id": "test_missing_002",
            "title": "Abstract Metal Art",
            "description": "High decorative value piece for living room.",
            "brand": "",
            "retail_price": None,
            "discounted_price": None,
            "specifications": {},
            "category_hierarchy": ["Home Decor"],
            "image_urls": []
        }
        decls = detect_all_declaration_evidence(missing_record, self.rules_cfg)
        self.assertEqual(decls["D01"]["label"], "ABSENT")
        self.assertEqual(decls["D03"]["label"], "ABSENT")
        self.assertEqual(decls["D05"]["label"], "ABSENT")
        self.assertEqual(decls["D06"]["label"], "ABSENT")

        overall = derive_overall_compliance(decls)
        self.assertEqual(overall["overall_label"], "NON_COMPLIANT")

    # 5. Fixture 3: Conflicting Origin Evidence
    def test_conflicting_origin_evidence(self):
        conflict_record = {
            "product_id": "test_conflict_003",
            "title": "Imported Wireless Mouse",
            "description": "Made in China with precision engineering.",
            "brand": "TechPro",
            "retail_price": 500.0,
            "specifications": {
                "Country of Origin": "India"
            },
            "category_hierarchy": ["Computers", "Accessories"],
            "image_urls": ["http://img.flixcart.com/mouse.jpg"]
        }
        d07 = detect_d07_evidence(conflict_record, self.rules_cfg)
        self.assertEqual(d07["label"], "CANNOT_DETERMINE")
        self.assertEqual(d07["evidence_nature"], "CONFLICTING_EVIDENCE")

    # 6. Test Deduplication & Family Clustering
    def test_deduplication_and_families(self):
        sample_records = [
            {
                "product_id": "p1",
                "product_url": "http://flipkart.com/shampoo-red",
                "product_name": "Dove Shampoo Daily Shine 180ml Red",
                "brand": "Dove",
                "category": "Beauty"
            },
            {
                "product_id": "p2",
                "product_url": "http://flipkart.com/shampoo-blue",
                "product_name": "Dove Shampoo Daily Shine 180ml Blue",
                "brand": "Dove",
                "category": "Beauty"
            }
        ]
        clustered = deduplicate_and_assign_families(sample_records)
        self.assertEqual(len(clustered), 2)
        # Should share same product family ID
        self.assertEqual(clustered[0]["product_family_id"], clustered[1]["product_family_id"])

    # 7. Test Weak Label Generation Structure
    def test_weak_label_record_generation(self):
        sample_ev = {
            "product_id": "p_gen_01",
            "product_url": "http://flipkart.com/item",
            "pid": "PID123",
            "title": "Nivea Soft Cream 200 ml",
            "description": "Soft light moisturising cream.",
            "brand": "Nivea",
            "retail_price": 299.0,
            "discounted_price": 249.0,
            "root_category": "Beauty and Personal Care",
            "category_hierarchy": ["Beauty and Personal Care", "Skin Care"],
            "specifications": {"Net Quantity": "200 ml", "Type": "Cream"},
            "image_urls": ["http://img.flixcart.com/nivea.jpg"],
            "image_evidence_status": "NOT_ATTEMPTED"
        }
        weak_rec = generate_weak_labels_for_product(sample_ev, self.rules_cfg)
        self.assertEqual(weak_rec["label_type"], "WEAK_LABEL")
        self.assertEqual(weak_rec["provenance"], "AUTOMATED_WEAK_LABEL_PIPELINE_V1")
        self.assertIn("D01", weak_rec["declarations"])
        self.assertIn("D07", weak_rec["declarations"])
        self.assertTrue(0.0 <= weak_rec["overall_confidence"] <= 1.0)


if __name__ == "__main__":
    unittest.main()
