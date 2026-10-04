# Security, Privacy & Input Hardening Report
> **Project**: NitiMark-ML — Standalone ML Prediction Layer  
> **SIH Problem Statement**: SIH25057 — Automated Compliance Checker for Legal Metrology Declarations on E-Commerce Platforms  
> **Phase**: Phase 12 — Streamlit Testing App  

---

## 1. Threat Modeling & Attack Surface Overview

The Phase 12 Streamlit application operates in a standalone, sandboxed testing environment. To ensure robustness against adversarial inputs, data leakage, and unauthorized credential exposure, multi-layered security controls are implemented.

```
+-----------------------------------------------------------------------------+
|                            EXTERNAL INPUT SURFACE                           |
|      [ Uploaded Image Payload ]             [ Live Gemini Vision API Key ]  |
+--------------------------+--------------------------------+-----------------+
                           |                                |
                           v                                v
+------------------------------------------+  +-------------------------------+
|  1. INPUT VALIDATION & INTEGRITY GUARD   |  | 2. CREDENTIAL PRIVACY GUARD   |
|  - Extension whitelist (.png, .jpg, etc) |  | - Masked password UI input    |
|  - Max 15MB file size ceiling            |  | - Never saved to session state|
|  - 100x100 to 6000x6000 px dimensions    |  | - Never written to disk/logs  |
|  - Binary byte validation via PIL        |  | - Redacted from JSON exports  |
+--------------------------+---------------+  +---------------+---------------+
                           |                                  |
                           v                                  v
+-----------------------------------------------------------------------------+
|  3. EXECUTION SANDBOXING & MEMORY ISOLATION                                 |
|  - Ephemeral NamedTemporaryFiles with guaranteed cleanup                    |
|  - Zero persistence of user uploaded images to dataset repositories         |
|  - Zero target leakage (No auto-training on user uploads)                   |
+------------------------------------------+----------------------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------+
|  4. SANITIZED AUDIT EXPORT                                                  |
|  - JSON sanitizer recursively strips paths, tokens, keys, and session IDs   |
+-----------------------------------------------------------------------------+
```

---

## 2. Security Controls & Implementations

### 2.1 Image Upload Hardening (`app/utils/validation.py`)
- **Format Whitelist**: Only `PNG`, `JPG`, `JPEG`, and `WEBP` MIME/file types are accepted. Executables (`.exe`, `.sh`), scripts (`.js`, `.html`), and document formats (`.pdf`, `.docx`) are rejected immediately before byte stream parsing.
- **Payload Size Enforcement**: Enforces a strict $15.0\text{ MB}$ upload ceiling to prevent Denial of Service (DoS) via memory exhaustion or large uncompressed bitmap bombs.
- **Dimension Clamping**: Rejects images below $100 \times 100\text{ px}$ (insufficient resolution for Legal Metrology OCR) and above $6000 \times 6000\text{ px}$ (prevents tensor allocation OOM errors).
- **Corrupt Byte Verification**: Executes `PIL.Image.open().verify()` on uploaded byte streams. Malformed, truncated, or obfuscated payloads are caught with a user-friendly error message rather than raising unhandled exceptions.

### 2.2 API Key Privacy & Credential Redaction (`app/components/export.py`, `app/streamlit_app.py`)
- **UI Password Masking**: API keys are accepted exclusively via Streamlit's `type="password"` input field, preventing shoulder surfing and browser DOM exposure.
- **Stateless Handling**: API keys are not stored in persistent session state caches across user sessions or logged to standard output (`stdout`/`stderr`).
- **Export Redaction Engine**: The `sanitize_export_data` utility recursively traverses all dictionary and list objects before exporting JSON analysis reports, scrubbing:
  - Any key matching `api_key`, `secret`, `token`, `password`, `credential`, `auth`.
  - Absolute local filesystem paths (e.g. `/home/devabhay07/...`).
  - Temporary system file identifiers.

### 2.3 Ephemeral File Lifecycle & Storage Isolation
- When image bytes must be passed to Vision/OCR providers requiring a filesystem path, ephemeral files are created using Python's `tempfile.NamedTemporaryFile` and deleted inside `finally:` blocks immediately upon extraction completion.
- Uploaded packaging images are **never persisted** to training directories (`data/processed/`, `data/raw/`), ensuring **zero target leakage** and strict benchmark integrity.

---

## 3. Security Verification Checklist

| Security Gate | Mechanism | Verification Method | Status |
|:---|:---|:---|:---:|
| **SG-01** | Image Format Whitelist | `test_disallowed_format` with `.pdf`, `.gif` | ✅ PASS |
| **SG-02** | Max Payload Size Clamping | `test_file_size_limit_exceeded` with 20MB buffer | ✅ PASS |
| **SG-03** | Corrupt File Rejection | `test_corrupt_image_bytes` with random byte stream | ✅ PASS |
| **SG-04** | API Key Redaction | `test_export_sanitization` with dummy credentials | ✅ PASS |
| **SG-05** | Ephemeral File Cleanup | `try...finally` deletion in `InferencePipelineBridge` | ✅ PASS |
| **SG-06** | Zero Training Data Leakage | Read-only model weights & locked ground truth SHA-256 | ✅ PASS |
| **SG-07** | Headless Import Safety | `GATE_X` module initialization check | ✅ PASS |
