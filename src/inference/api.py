"""
api.py
======
Optional lightweight standalone HTTP API server for NitiMark-ML inference.
Built using Python standard library (http.server) to avoid external framework dependencies.

Endpoints:
    GET  /health   -> Status healthcheck
    POST /predict  -> Run inference on JSON payload
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from src.inference.predict import run_standalone_prediction


class NitiMarkInferenceHandler(BaseHTTPRequestHandler):
    """HTTP request handler for standalone inference endpoints."""

    def _send_json(self, status_code: int, data: Dict[str, Any]):
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode("utf-8"))

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {
                "project": "NitiMark-ML",
                "phase": "phase_9",
                "status": "HEALTHY",
                "models_loaded": True
            })
        else:
            self._send_json(404, {"error": f"Endpoint {self.path} not found"})

    def do_POST(self):
        if self.path == "/predict":
            try:
                content_len = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_len).decode("utf-8")
                payload = json.loads(body) if body else {}

                result = run_standalone_prediction(
                    image_path=payload.get("image_path"),
                    title=payload.get("title", ""),
                    description=payload.get("description", ""),
                    specifications=payload.get("specifications"),
                    retail_price=payload.get("retail_price"),
                    discounted_price=payload.get("discounted_price"),
                    brand=payload.get("brand"),
                    category=payload.get("category"),
                    offline_mode=payload.get("offline", False),
                    provider_name=payload.get("provider", "gemini"),
                    api_key_env=payload.get("api_key_env", "VISION_API_KEY"),
                    model_name=payload.get("model", "gemini-3.6-flash")
                )
                self._send_json(200, result)

            except Exception as e:
                self._send_json(500, {
                    "project": "NitiMark-ML",
                    "status": "SERVER_ERROR",
                    "error": str(e)
                })
        else:
            self._send_json(404, {"error": f"Endpoint {self.path} not found"})


def run_api_server(host: str = "127.0.0.1", port: int = 8504):
    server = HTTPServer((host, port), NitiMarkInferenceHandler)
    print(f"NitiMark-ML Inference API running on http://{host}:{port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down API server.")
        server.server_close()


if __name__ == "__main__":
    run_api_server()
