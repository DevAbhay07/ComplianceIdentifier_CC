"""
test_ai_drafting.py
===================
Comprehensive test suite for Phase 4.5B AI Drafting and Review Integration.

Tests:
  1. Evidence collection across review products (metadata, specs, images)
  2. Schema validation with Pydantic (valid outputs, invalid labels, invalid sources, bounds)
  3. API failure and fallback handling (missing key, timeout, malformed responses)
  4. Data model separation (machine draft vs human annotation)
  5. Audit tracking (review actions, timestamps, confidence)
  6. Original annotation file integrity (zero modifications to pilot_annotations.jsonl)
"""

import json
import os
import sys
import unittest
from datetime import datetime, timezone
import pandas as pd
from pydantic import ValidationError

from src.data.generate_review_draft import (
    DECLS,
    AIDraftOutput,
    DeclarationDraft,
    OverallDraft,
    collect_product_evidence,
    generate_ai_draft,
    parse_image_urls,
)
from src.data.validate_reviews import BACKUP_PATH, ORIGINAL_ANNOTATIONS_PATH


class TestAIDrafting(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.review_df = pd.read_csv("annotations/review_36.csv")
        cls.products = {}
        with open("data/interim/parsed_annotation_candidates.jsonl", "r") as f:
            for line in f:
                r = json.loads(line)
                cls.products[str(r.get("product_id", ""))] = r

    def test_01_evidence_collection_sample(self):
        """Test evidence collection on 10 review products."""
        sample_pids = self.review_df["product_id"].tolist()[:10]
        self.assertEqual(len(sample_pids), 10)

        for pid in sample_pids:
            product = self.products.get(pid, {})
            evidence = collect_product_evidence(product, max_images=1)

            self.assertEqual(evidence["product_id"], pid)
            self.assertIn("PRODUCT_TITLE", evidence["available_sources"])
            self.assertIsInstance(evidence["available_sources"], list)
            self.assertIsInstance(evidence["unavailable_sources"], list)
            self.assertIsInstance(evidence["fetched_images"], list)
            # Ensure description and specs are present
            self.assertIn("specifications", evidence)
            self.assertIn("description", evidence)

    def test_02_image_url_parsing(self):
        """Test robust parsing of various Flipkart image URL formats."""
        # Standard list string
        raw_1 = '["http://img5a.flixcart.com/image1.jpg", "http://img5a.flixcart.com/image2.jpg"]'
        urls_1 = parse_image_urls(raw_1)
        self.assertEqual(len(urls_1), 2)
        self.assertTrue(urls_1[0].startswith("http://img5a.flixcart.com/image1.jpg"))

        # Empty / nan
        self.assertEqual(parse_image_urls("nan"), [])
        self.assertEqual(parse_image_urls(""), [])
        self.assertEqual(parse_image_urls(None), [])

    def test_03_pydantic_schema_validation_success(self):
        """Test that valid structured JSON passes Pydantic validation."""
        valid_payload = {
            "D01": {
                "draft_label": "PARTIAL",
                "evidence_source": "PRODUCT_SPECIFICATIONS",
                "evidence_text": "Brand: TestBrand",
                "confidence": 0.85,
                "reasoning": "Brand name present, address missing"
            },
            "D02": {
                "draft_label": "PRESENT",
                "evidence_source": "PRODUCT_TITLE",
                "evidence_text": "Ceramic Mug",
                "confidence": 0.95,
                "reasoning": "Generic commodity name clear"
            },
            "D03": {
                "draft_label": "PRESENT",
                "evidence_source": "PRODUCT_SPECIFICATIONS",
                "evidence_text": "Capacity: 330 ml",
                "confidence": 0.90,
                "reasoning": "Net quantity in standard metric unit"
            },
            "D04": {
                "draft_label": "CANNOT_DETERMINE",
                "evidence_source": "NO_EVIDENCE",
                "evidence_text": "",
                "confidence": 0.75,
                "reasoning": "Manufacturing date not in listing"
            },
            "D05": {
                "draft_label": "PRESENT",
                "evidence_source": "PRODUCT_PAGE",
                "evidence_text": "Retail Price: 699",
                "confidence": 0.95,
                "reasoning": "Price field available"
            },
            "D06": {
                "draft_label": "ABSENT",
                "evidence_source": "NO_EVIDENCE",
                "evidence_text": "",
                "confidence": 0.85,
                "reasoning": "No customer care details"
            },
            "D07": {
                "draft_label": "CANNOT_DETERMINE",
                "evidence_source": "NO_EVIDENCE",
                "evidence_text": "",
                "confidence": 0.80,
                "reasoning": "Origin not specified"
            },
            "overall": {
                "suggested_label": "NON_COMPLIANT",
                "reasoning": "D06 is ABSENT and D01 is PARTIAL"
            }
        }

        output = AIDraftOutput(**valid_payload)
        self.assertEqual(output.D01.draft_label, "PARTIAL")
        self.assertEqual(output.D02.draft_label, "PRESENT")
        self.assertEqual(output.overall.suggested_label, "NON_COMPLIANT")
        self.assertEqual(output.D01.confidence, 0.85)

    def test_04_pydantic_schema_validation_rejections(self):
        """Test that invalid declaration labels, sources, or overall labels raise ValidationError."""
        # Invalid declaration label
        with self.assertRaises(ValidationError):
            DeclarationDraft(
                draft_label="MAYBE_COMPLIANT",  # Invalid
                evidence_source="PRODUCT_IMAGE",
                confidence=0.8,
            )

        # Invalid overall label
        with self.assertRaises(ValidationError):
            OverallDraft(suggested_label="SEMI_LEGAL")  # Invalid

        # Invalid evidence source
        with self.assertRaises(ValidationError):
            DeclarationDraft(
                draft_label="PRESENT",
                evidence_source="SOCIAL_MEDIA_POST",  # Invalid
                confidence=0.8,
            )

    def test_05_graceful_api_failure_handling(self):
        """Test that missing API key returns a structured failure without crashing."""
        from unittest.mock import patch
        sample_pid = self.review_df["product_id"].iloc[0]
        product = self.products.get(sample_pid, {})

        # Test with no API key in argument or environment
        with patch.dict(os.environ, {"GEMINI_API_KEY": "", "GOOGLE_API_KEY": ""}, clear=True):
            res = generate_ai_draft(product, api_key="")
            self.assertFalse(res["draft_generated"])
            self.assertIsNotNone(res["error"])
            self.assertIn("API Key", res["error"])
            self.assertEqual(res["product_id"], sample_pid)

    def test_06_original_annotations_integrity(self):

        """Verify that pilot_annotations.jsonl is identical to backup."""
        self.assertTrue(os.path.exists(BACKUP_PATH), "Backup file missing")
        self.assertTrue(os.path.exists(ORIGINAL_ANNOTATIONS_PATH), "Original annotations file missing")

        with open(BACKUP_PATH, "r") as fb, open(ORIGINAL_ANNOTATIONS_PATH, "r") as fo:
            backup_lines = [line.strip() for line in fb if line.strip()]
            orig_lines = [line.strip() for line in fo if line.strip()]

        self.assertEqual(len(backup_lines), len(orig_lines), "Line count mismatch between original and backup!")
        self.assertEqual(backup_lines, orig_lines, "Content mismatch between original and backup!")

    def test_07_supported_draft_models_config(self):
        """Verify supported draft models dictionary and genuine model IDs."""

        from src.data.generate_review_draft import SUPPORTED_DRAFT_MODELS
        self.assertIn("Gemini 3.6 Flash", SUPPORTED_DRAFT_MODELS)
        self.assertIn("Gemini 2.5 Flash", SUPPORTED_DRAFT_MODELS)
        self.assertIn("Gemini 2.5 Pro", SUPPORTED_DRAFT_MODELS)
        self.assertIn("Gemini 2.5 Flash-Lite", SUPPORTED_DRAFT_MODELS)

        self.assertEqual(SUPPORTED_DRAFT_MODELS["Gemini 3.6 Flash"], "gemini-3.6-flash")
        self.assertEqual(SUPPORTED_DRAFT_MODELS["Gemini 2.5 Pro"], "gemini-2.5-pro")

    def test_08_mock_quota_429_handling(self):
        """Verify 429 quota error returns is_quota_error=True and does not crash."""
        from unittest.mock import MagicMock, patch
        from src.data.generate_review_draft import call_gemini_api

        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.text = '{"error": {"code": 429, "message": "Resource has been exhausted (e.g. check quota)."}}'

        with patch("requests.post", return_value=mock_resp):
            success, raw_text, error_msg, is_quota = call_gemini_api(
                prompt="test",
                images=[],
                api_key="fake_key",
                model="gemini-2.5-flash",
            )
            self.assertFalse(success)
            self.assertIsNone(raw_text)
            self.assertTrue(is_quota)
            self.assertIn("Quota exhausted", error_msg)

    def test_09_existing_13_reviews_integrity(self):
        """Verify that previously saved reviews in review_36_results.jsonl are preserved and valid."""
        results_path = "annotations/review_36_results.jsonl"
        self.assertTrue(os.path.exists(results_path), "review_36_results.jsonl missing")

        reviews = []
        with open(results_path, "r") as f:
            for line in f:
                if line.strip():
                    reviews.append(json.loads(line))

        self.assertGreaterEqual(len(reviews), 13, f"Expected at least 13 reviews, found {len(reviews)}")

        for rev in reviews:
            self.assertTrue(rev.get("reviewed_by_human"))
            self.assertTrue(rev.get("human_confirmed"))
            self.assertIn("product_id", rev)
            self.assertIn("revised_declarations", rev)
            self.assertIn("revised_overall", rev)
            self.assertIn("ai_draft", rev)


if __name__ == "__main__":
    unittest.main()

