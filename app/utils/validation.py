"""
Image validation utilities for NitiMark Streamlit application.
Validates file formats, file integrity, dimensions, and size constraints.
"""

import io
from typing import Dict, Any, Tuple
from PIL import Image

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024  # 15 MB
MIN_IMAGE_DIM = 64
MAX_IMAGE_DIM = 8192


def validate_uploaded_image(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    Validates uploaded image byte buffer.
    Returns dictionary with 'valid': bool, 'error': str, 'image': Optional[Image.Image], 'metadata': dict.
    """
    if not file_bytes or len(file_bytes) == 0:
        return {
            "valid": False,
            "error": "The uploaded file is empty (0 bytes).",
            "image": None,
            "metadata": {}
        }

    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        size_mb = len(file_bytes) / (1024 * 1024)
        return {
            "valid": False,
            "error": f"File size ({size_mb:.2f} MB) exceeds maximum allowed limit (15 MB).",
            "image": None,
            "metadata": {}
        }

    ext = filename.split(".")[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        return {
            "valid": False,
            "error": f"Unsupported format '.{ext}'. Supported formats: PNG, JPG, JPEG, WEBP.",
            "image": None,
            "metadata": {}
        }

    try:
        img = Image.open(io.BytesIO(file_bytes))
        img.verify()  # Verify integrity
        
        # Re-open after verify() because verify() mutates the stream
        img = Image.open(io.BytesIO(file_bytes))
        
        # Convert paletted or RGBA to RGB for standard vision pipeline
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
            
        w, h = img.size
        if w < MIN_IMAGE_DIM or h < MIN_IMAGE_DIM:
            return {
                "valid": False,
                "error": f"Image dimensions ({w}x{h}) are too small (minimum {MIN_IMAGE_DIM}x{MIN_IMAGE_DIM}).",
                "image": None,
                "metadata": {}
            }
            
        if w > MAX_IMAGE_DIM or h > MAX_IMAGE_DIM:
            return {
                "valid": False,
                "error": f"Image dimensions ({w}x{h}) exceed maximum allowed dimension ({MAX_IMAGE_DIM}px).",
                "image": None,
                "metadata": {}
            }

        return {
            "valid": True,
            "error": None,
            "image": img,
            "metadata": {
                "filename": filename,
                "format": img.format or ext.upper(),
                "width": w,
                "height": h,
                "mode": img.mode,
                "size_bytes": len(file_bytes)
            }
        }

    except Exception as e:
        return {
            "valid": False,
            "error": f"Corrupt or unreadable image file: {str(e)}",
            "image": None,
            "metadata": {}
        }
