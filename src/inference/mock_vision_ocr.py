"""
mock_vision_ocr.py
==================
Deterministic Mock Vision / OCR provider for automated unit testing,
offline validation, and CI/CD smoke tests without live API calls.
"""

from typing import Dict, Any, Optional
from src.inference.vision_ocr import VisionOCRProvider
from src.inference.image_validator import validate_product_image

DEFAULT_MOCK_OCR_TEXT = """Colgate Total Clean Mint Toothpaste 120 g
Manufactured by: Colgate-Palmolive (India) Ltd., Plot No. 12, Main Street, Mumbai 400001, Maharashtra, India.
Generic Name: Toothpaste
Net Quantity: 120 g
Date of Mfg: 01/2026 | Best Before 24 months from mfg
Maximum Retail Price (MRP): Rs. 145.00 (Inclusive of all taxes)
Customer Care: 1800-225-599 or email: care@colgate.com
Country of Origin: India"""


class MockVisionOCRProvider(VisionOCRProvider):
    """Deterministic mock OCR provider for unit tests and offline testing."""

    def __init__(
        self,
        mock_text: Optional[str] = None,
        mock_status: str = "SUCCESS",
        mock_error: Optional[str] = None,
        latency_sec: float = 0.05
    ):
        self.mock_text = mock_text or DEFAULT_MOCK_OCR_TEXT
        self.mock_status = mock_status
        self.mock_error = mock_error
        self.latency_sec = latency_sec

    def extract_from_image(self, image_path: str) -> Dict[str, Any]:
        # Validate image structure
        val_res = validate_product_image(image_path)
        if not val_res["valid"]:
            return {
                "status": "FAILED",
                "raw_text": "",
                "image_metadata": val_res,
                "provider": "mock_vision_ocr",
                "model": "mock-vision-v1",
                "latency_sec": 0.0,
                "error": val_res["error"]
            }

        if self.mock_status != "SUCCESS":
            return {
                "status": self.mock_status,
                "raw_text": "",
                "image_metadata": val_res,
                "provider": "mock_vision_ocr",
                "model": "mock-vision-v1",
                "latency_sec": self.latency_sec,
                "error": self.mock_error or f"MOCK_ERROR ({self.mock_status})"
            }

        return {
            "status": "SUCCESS",
            "raw_text": self.mock_text.strip(),
            "image_metadata": val_res,
            "provider": "mock_vision_ocr",
            "model": "mock-vision-v1",
            "latency_sec": self.latency_sec,
            "error": None
        }
