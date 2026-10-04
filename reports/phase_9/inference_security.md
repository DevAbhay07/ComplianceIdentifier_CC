# Inference Security & Credential Protection Report (Phase 9)

**Project:** NitiMark-ML — Standalone Legal Metrology Compliance Predictor  
**Phase:** Phase 9 — Standalone Inference & Real-Product Validation  
**Date:** September 2026  

---

## 1. Threat Model & Security Objectives

Standalone inference workflows frequently interact with external Vision APIs, user-supplied image files, and unstructured strings. This document specifies the security controls implemented in Phase 9.

---

## 2. Key Security Protections

### 2.1 API Key Hygiene & Zero Hardcoded Secrets
- **No Hardcoded Keys:** Static code analysis verified that zero API keys, tokens, or plaintext credentials exist in any repository file.
- **Environment Resolution:** `GoogleGeminiVisionProvider` reads API keys exclusively through `VISION_API_KEY` or `--api-key-env` parameter.
- **Safe Template:** A sanitized `.env.example` is provided with dummy placeholders.
- **Git Exclusion:** `.env` and sensitive credential paths are excluded via `.gitignore`.

### 2.2 Input Sanitization & Path Traversal Defense
- **Image Paths:** All file paths are passed through `os.path.abspath()` and verified via `os.path.exists()` and `os.path.isfile()` before file handles are opened.
- **Memory Bomb Prevention:** Maximum image file size is capped at 25 MB (`MAX_IMAGE_SIZE_BYTES`), preventing memory exhaustion denial-of-service attacks.
- **Image Corruption Verification:** PIL image integrity is verified with `img.verify()` in a isolated block before full decompression.

### 2.3 Audit Logging & Privacy Preservation
- **Redacted Trace Logs:** Trace logs in `data/interim/inference_logs/` record only input SHA-256 cryptographic hashes, latency numbers, and model outputs.
- **Zero Sensitive Data:** HTTP request headers, authorization tokens, and user IP addresses are completely excluded from log payloads.
