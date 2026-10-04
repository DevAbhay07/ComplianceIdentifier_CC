"""
trace.py
========
Inference traceability and audit logging manager. Computes cryptographic hashes
of input media/text, records component-level latency breakdowns, and persists
audit logs without exposing sensitive credentials.
"""

import os
import json
import hashlib
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional

DEFAULT_LOG_DIR = "data/interim/inference_logs"


def compute_content_sha256(content: str | bytes) -> str:
    """Computes SHA-256 hash of a string or byte stream."""
    if isinstance(content, str):
        content = content.encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def compute_file_sha256(filepath: str) -> Optional[str]:
    """Computes SHA-256 hash of a local file."""
    if not os.path.exists(filepath) or not os.path.isfile(filepath):
        return None
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


class InferenceTracer:
    """Tracks latency, input hashes, and provenance for a single inference request."""

    def __init__(self, log_dir: str = DEFAULT_LOG_DIR, enabled: bool = True):
        self.log_dir = log_dir
        self.enabled = enabled
        self.start_time = time.time()
        self.timestamps: Dict[str, float] = {}
        self.latencies: Dict[str, float] = {}
        self.trace_id = f"trace_{int(time.time()*1000)}_{os.urandom(3).hex()}"

    def mark_start(self, stage_name: str):
        self.timestamps[stage_name] = time.time()

    def mark_end(self, stage_name: str):
        if stage_name in self.timestamps:
            self.latencies[stage_name] = round(time.time() - self.timestamps[stage_name], 4)

    def finalize(
        self,
        input_descriptors: Dict[str, Any],
        ocr_result: Dict[str, Any],
        predictions: Dict[str, Any],
        diagnostics: Dict[str, Any],
        status: str = "SUCCESS"
    ) -> Dict[str, Any]:
        total_latency = round(time.time() - self.start_time, 4)
        self.latencies["total_pipeline"] = total_latency

        # Calculate input hashes
        input_hashes = {}
        if "image_path" in input_descriptors and input_descriptors["image_path"]:
            input_hashes["image_sha256"] = compute_file_sha256(input_descriptors["image_path"])
        if "text" in input_descriptors and input_descriptors["text"]:
            input_hashes["text_sha256"] = compute_content_sha256(input_descriptors["text"])

        trace_record = {
            "trace_id": self.trace_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "input_hashes": input_hashes,
            "input_modalities": list(input_descriptors.keys()),
            "latencies_sec": self.latencies,
            "ocr_metadata": {
                "provider": ocr_result.get("provider", "NONE"),
                "model": ocr_result.get("model", "NONE"),
                "status": ocr_result.get("status", "NOT_ATTEMPTED")
            },
            "declarations_predicted": list(predictions.keys()),
            "overall_diagnostics": diagnostics
        }

        if self.enabled:
            os.makedirs(self.log_dir, exist_ok=True)
            log_path = os.path.join(self.log_dir, f"{self.trace_id}.json")
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(trace_record, f, indent=2)

        return trace_record
