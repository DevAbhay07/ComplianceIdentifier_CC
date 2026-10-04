"""
image_validator.py
==================
Validates input product images for existence, supported formats, file size,
dimensions, and image integrity before passing to Vision/OCR providers.
"""

import os
from typing import Dict, Any
from PIL import Image

SUPPORTED_FORMATS = {"JPEG", "JPG", "PNG", "WEBP"}
MAX_IMAGE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB
MIN_IMAGE_DIMENSION = 50                 # Minimum 50px width/height


def validate_product_image(image_path: str) -> Dict[str, Any]:
    """
    Validate an image file for existence, readable format, size, and corruption.

    Returns:
        Dict[str, Any] containing validation diagnostics:
        {
            "valid": bool,
            "image_path": str,
            "width": int or None,
            "height": int or None,
            "format": str or None,
            "size_bytes": int,
            "error": str or None
        }
    """
    if not image_path or not isinstance(image_path, str):
        return {
            "valid": False,
            "image_path": str(image_path),
            "width": None,
            "height": None,
            "format": None,
            "size_bytes": 0,
            "error": "INVALID_IMAGE_PATH: Path must be a non-empty string."
        }

    abs_path = os.path.abspath(image_path)
    if not os.path.exists(abs_path):
        return {
            "valid": False,
            "image_path": abs_path,
            "width": None,
            "height": None,
            "format": None,
            "size_bytes": 0,
            "error": f"FILE_NOT_FOUND: Image file does not exist at {abs_path}"
        }

    if not os.path.isfile(abs_path):
        return {
            "valid": False,
            "image_path": abs_path,
            "width": None,
            "height": None,
            "format": None,
            "size_bytes": 0,
            "error": f"NOT_A_FILE: Specified path is not a file: {abs_path}"
        }

    # File size check
    size_bytes = os.path.getsize(abs_path)
    if size_bytes == 0:
        return {
            "valid": False,
            "image_path": abs_path,
            "width": None,
            "height": None,
            "format": None,
            "size_bytes": 0,
            "error": "EMPTY_FILE: Image file has 0 bytes."
        }

    if size_bytes > MAX_IMAGE_SIZE_BYTES:
        return {
            "valid": False,
            "image_path": abs_path,
            "width": None,
            "height": None,
            "format": None,
            "size_bytes": size_bytes,
            "error": f"FILE_TOO_LARGE: Image exceeds maximum allowed size of {MAX_IMAGE_SIZE_BYTES // (1024*1024)} MB."
        }

    # PIL verification & corruption check
    try:
        with Image.open(abs_path) as img:
            img_format = (img.format or "").upper()
            if img_format == "JPG":
                img_format = "JPEG"

            if img_format not in SUPPORTED_FORMATS:
                return {
                    "valid": False,
                    "image_path": abs_path,
                    "width": img.width,
                    "height": img.height,
                    "format": img_format,
                    "size_bytes": size_bytes,
                    "error": f"UNSUPPORTED_FORMAT: Format '{img_format}' is not supported. Use JPEG, PNG, or WEBP."
                }

            if img.width < MIN_IMAGE_DIMENSION or img.height < MIN_IMAGE_DIMENSION:
                return {
                    "valid": False,
                    "image_path": abs_path,
                    "width": img.width,
                    "height": img.height,
                    "format": img_format,
                    "size_bytes": size_bytes,
                    "error": f"DIMENSIONS_TOO_SMALL: Image dimensions {img.width}x{img.height} below minimum {MIN_IMAGE_DIMENSION}px."
                }

            # Verify integrity
            img.verify()

        # Re-open to read width & height safely after verify()
        with Image.open(abs_path) as img:
            width, height = img.size

        return {
            "valid": True,
            "image_path": abs_path,
            "width": width,
            "height": height,
            "format": img_format,
            "size_bytes": size_bytes,
            "error": None
        }

    except Exception as e:
        return {
            "valid": False,
            "image_path": abs_path,
            "width": None,
            "height": None,
            "format": None,
            "size_bytes": size_bytes,
            "error": f"CORRUPT_OR_UNREADABLE_IMAGE: {str(e)}"
        }
