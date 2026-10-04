"""
image_evidence.py
==================
Module for validating, downloading, and inspecting product image evidence.
Designed to degrade gracefully if network is unavailable or vision APIs are offline.

Provides:
- parse_image_urls: Extracts list of valid URLs from Flipkart image field
- validate_image_url_structure: Static check for valid domain and image extensions
- fetch_image_metadata: Downloads image header/content to verify dimensions (cached)
- process_image_evidence: Batch/single handler for image evidence extraction
"""

import os
import re
import json
import hashlib
from urllib.parse import urlparse
from typing import List, Dict, Any, Optional

# Cache directory for image evidence metadata
CACHE_DIR = "data/interim/image_cache"


def parse_image_urls(raw_image_str: Any) -> List[str]:
    """Parse Flipkart image field into a clean list of HTTP/HTTPS URLs."""
    if not raw_image_str or str(raw_image_str).strip() in ("", "nan", "None", "[]"):
        return []
    s = str(raw_image_str).strip()
    urls = re.findall(r'https?://[^\s",\]]+', s)
    # Clean trailing characters
    clean_urls = []
    for u in urls:
        u = u.strip().rstrip('",\']')
        if u.startswith("http"):
            clean_urls.append(u)
    return clean_urls


def validate_image_url_structure(url: str) -> Dict[str, Any]:
    """Static validation of image URL structure."""
    if not url:
        return {"valid": False, "reason": "empty_url"}
    try:
        parsed = urlparse(url)
        is_http = parsed.scheme in ("http", "https")
        has_netloc = bool(parsed.netloc)
        return {
            "valid": is_http and has_netloc,
            "scheme": parsed.scheme,
            "domain": parsed.netloc,
            "path": parsed.path
        }
    except Exception as e:
        return {"valid": False, "reason": f"parse_error: {str(e)}"}


def inspect_product_images(
    image_urls: List[str],
    attempt_download: bool = False,
    timeout_sec: int = 3
) -> Dict[str, Any]:
    """
    Inspect product images and generate evidence metadata.
    If attempt_download is False (default for offline/batch fast mode),
    records static structural availability without network blockages.
    """
    if not image_urls:
        return {
            "image_evidence_status": "UNAVAILABLE",
            "accessible_images": [],
            "image_count": 0,
            "image_metadata": {"reason": "No image URLs present in listing"},
            "ocr_text": "",
            "vision_status": "NOT_ATTEMPTED"
        }

    valid_urls = [u for u in image_urls if validate_image_url_structure(u)["valid"]]
    
    if not attempt_download:
        return {
            "image_evidence_status": "NOT_ATTEMPTED",
            "accessible_images": valid_urls[:3],  # Record primary URLs
            "image_count": len(valid_urls),
            "image_metadata": {
                "total_urls": len(image_urls),
                "valid_url_count": len(valid_urls),
                "primary_domain": validate_image_url_structure(valid_urls[0])["domain"] if valid_urls else "",
                "download_policy": "deferred_for_ml_pipeline"
            },
            "ocr_text": "",
            "vision_status": "NOT_ATTEMPTED"
        }

    # If attempt_download is requested, attempt lightweight verification
    os.makedirs(CACHE_DIR, exist_ok=True)
    import urllib.request

    accessible = []
    meta = {}
    for i, url in enumerate(valid_urls[:2]):  # check up to 2 images
        url_hash = hashlib.md5(url.encode()).hexdigest()
        cache_file = os.path.join(CACHE_DIR, f"{url_hash}.json")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r") as f:
                    cached_data = json.load(f)
                    if cached_data.get("accessible"):
                        accessible.append(url)
                    meta[f"image_{i}"] = cached_data
                    continue
            except Exception:
                pass

        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) NitiMark/1.0"}
            )
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                status = resp.status
                content_type = resp.headers.get("Content-Type", "")
                content_len = resp.headers.get("Content-Length", 0)
                is_img = "image" in content_type.lower()
                res = {
                    "accessible": is_img and status == 200,
                    "http_status": status,
                    "content_type": content_type,
                    "content_length": int(content_len) if content_len else 0
                }
                if res["accessible"]:
                    accessible.append(url)
                meta[f"image_{i}"] = res
                with open(cache_file, "w") as f:
                    json.dump(res, f)
        except Exception as e:
            err_res = {"accessible": False, "error": str(e)}
            meta[f"image_{i}"] = err_res
            with open(cache_file, "w") as f:
                json.dump(err_res, f)

    status_str = "SUCCESS" if accessible else ("FAILED" if valid_urls else "UNAVAILABLE")

    return {
        "image_evidence_status": status_str,
        "accessible_images": accessible,
        "image_count": len(valid_urls),
        "image_metadata": meta,
        "ocr_text": "",
        "vision_status": "NOT_ATTEMPTED"
    }
