"""
vision_ocr.py
=============
Vision and OCR provider abstraction layer for extracting visible packaging text
from product images without prompting LLMs for legal compliance verdicts.
"""

import os
import time
import base64
import hashlib
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from src.inference.image_validator import validate_product_image

# Global in-memory cache for deterministic repeated image extractions
_IMAGE_OCR_CACHE: Dict[str, Dict[str, Any]] = {}

EXTRACTION_PROMPT = """Extract all visible text on this product packaging, label, or container.
Accurately transcribe any visible details regarding:
1. Manufacturer, packer, or importer name and physical address
2. Generic or common product name
3. Net quantity, weight, volume, or unit count
4. Date of manufacture, packing, best before, or expiry date
5. Maximum Retail Price (MRP), retail price, and tax notations
6. Consumer care phone number, email, address, or website
7. Country of origin

Output the extracted text clearly and verbatim where legible. Do not assess legal compliance."""


SUPPORTED_GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.1-pro-preview",
    "gemini-2.5-flash-lite",
    "gemini-flash-latest",
    "gemini-pro-latest"
]
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


class VisionOCRProvider(ABC):
    """Abstract base class for Vision / OCR extraction providers."""

    @abstractmethod
    def extract_from_image(self, image_path: str) -> Dict[str, Any]:
        """Extract visible text and packaging information from an image."""
        pass


class GoogleGeminiVisionProvider(VisionOCRProvider):
    """
    Google Gemini Vision provider for extracting packaging text.
    Supports dynamic model selection across gemini-2.5-flash, gemini-3.7-flash, etc.
    Employs temperature=0.0 and SHA-256 caching for strict determinism.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout_sec: float = 30.0
    ):
        self.api_key = api_key or os.environ.get("VISION_API_KEY") or os.environ.get("GEMINI_API_KEY")
        self.model_name = model_name or os.environ.get("VISION_MODEL") or DEFAULT_GEMINI_MODEL
        self.timeout_sec = timeout_sec

    def set_model(self, model_name: str):
        """Dynamically switch the Gemini model."""
        self.model_name = model_name

    def extract_from_image(self, image_path: str) -> Dict[str, Any]:
        # 1. Validate image first
        val_res = validate_product_image(image_path)
        if not val_res["valid"]:
            return {
                "status": "FAILED",
                "raw_text": "",
                "image_metadata": val_res,
                "provider": "google_gemini",
                "model": self.model_name,
                "latency_sec": 0.0,
                "error": val_res["error"]
            }

        # 2. Check API key
        if not self.api_key or not self.api_key.strip():
            return {
                "status": "INVALID_API_KEY",
                "raw_text": "",
                "image_metadata": val_res,
                "provider": "google_gemini",
                "model": self.model_name,
                "latency_sec": 0.0,
                "error": "VISION_API_KEY is missing or empty. Configure VISION_API_KEY in environment or .env file."
            }

        # 3. Read image bytes and check SHA-256 cache
        try:
            with open(image_path, "rb") as f:
                img_bytes = f.read()
            img_hash = hashlib.sha256(img_bytes).hexdigest()
            cache_key = f"{img_hash}_{self.model_name}"

            if cache_key in _IMAGE_OCR_CACHE:
                cached = dict(_IMAGE_OCR_CACHE[cache_key])
                cached["latency_sec"] = 0.001
                return cached
        except Exception as e:
            return {
                "status": "FAILED",
                "raw_text": "",
                "image_metadata": val_res,
                "provider": "google_gemini",
                "model": self.model_name,
                "latency_sec": 0.0,
                "error": f"Failed to read image: {e}"
            }

        t0 = time.time()
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=self.api_key)

            mime_type = f"image/{val_res['format'].lower()}"
            if mime_type == "image/jpg":
                mime_type = "image/jpeg"

            config = types.GenerateContentConfig(
                temperature=0.0,
                top_p=0.95
            )

            response = client.models.generate_content(
                model=self.model_name,
                contents=[
                    types.Part.from_bytes(data=img_bytes, mime_type=mime_type),
                    EXTRACTION_PROMPT
                ],
                config=config
            )

            latency = round(time.time() - t0, 3)
            raw_text = response.text or ""

            result = {
                "status": "SUCCESS",
                "raw_text": raw_text.strip(),
                "image_metadata": val_res,
                "provider": "google_gemini",
                "model": self.model_name,
                "latency_sec": latency,
                "error": None
            }

            if raw_text.strip():
                _IMAGE_OCR_CACHE[cache_key] = result

            return result

        except Exception as e:
            latency = round(time.time() - t0, 3)
            err_msg = str(e)
            status = "FAILED"
            if "quota" in err_msg.lower() or "429" in err_msg:
                status = "QUOTA_EXHAUSTED"
            elif "api_key" in err_msg.lower() or "permission" in err_msg.lower() or "403" in err_msg or "401" in err_msg:
                status = "INVALID_API_KEY"
            elif "timeout" in err_msg.lower():
                status = "TIMEOUT"

            return {
                "status": status,
                "raw_text": "",
                "image_metadata": val_res,
                "provider": "google_gemini",
                "model": self.model_name,
                "latency_sec": latency,
                "error": f"VISION_API_ERROR ({status}): {err_msg}"
            }


def get_vision_provider(
    provider_type: str = "gemini",
    api_key: Optional[str] = None,
    model_name: Optional[str] = None
) -> VisionOCRProvider:
    """Factory function for instantiating OCR / Vision providers."""
    p_type = (provider_type or os.environ.get("VISION_PROVIDER", "gemini")).lower()
    
    if p_type in ("gemini", "google_gemini", "google"):
        return GoogleGeminiVisionProvider(api_key=api_key, model_name=model_name or DEFAULT_GEMINI_MODEL)
    elif p_type in ("mock", "mock_ocr", "test"):
        from src.inference.mock_vision_ocr import MockVisionOCRProvider
        return MockVisionOCRProvider()
    else:
        raise ValueError(f"Unknown Vision provider type: {provider_type}. Supported: 'gemini', 'mock'.")
