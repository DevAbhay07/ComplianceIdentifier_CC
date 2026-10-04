# Phase 9.5 Vision API Failure Modes, Fallbacks & Error Taxonomy

> **Project**: NitiMark-ML — Standalone Prediction Layer (SIH25057)  
> **Evaluation Scope**: Analysis of vision/OCR external failure modes, quota management, network resilience, and structured fallback mechanics.

---

## 1. Overview & Architectural Boundaries

In accordance with Phase 9 / 9.5 architectural guidelines, the Vision/OCR extraction component operates strictly as an **evidence transducer** (converting packaging images to raw text) without prompting external APIs for legal compliance decisions.

Because remote vision APIs (such as Google Gemini Vision) are subject to external runtime factors (network dropouts, API quota exhaustion, HTTP timeouts, credential revocation), the inference pipeline provides strict fault isolation, explicit error taxonomy, and offline fallback degradation.

```
                  +-----------------------------------+
                  |      Product Image / Request      |
                  +-----------------+-----------------+
                                    |
                          [Image Validation]
                                    |
                    +---------------+---------------+
                    | (Valid Image)                 | (Corrupted / Too Big)
                    v                               v
        [Vision Provider Check]              [INVALID_IMAGE] (Early Return)
                    |
        +-----------+-----------+
        |                       |
   (Online Mode)          (Offline Mode)
        |                       |
  [Gemini Vision]       [OFFLINE_MODE_SKIPPED]
        |                       |
+-------+-------+               v
|               |       [Fallback to Text Metadata]
(Success)    (Failure)          |
v               v               v
[Raw Text]  [Classified Error]  [Evidence Parser]
```

---

## 2. Taxonomy of Vision / API Failure Modes

The inference pipeline standardizes external API failures into 6 distinct, observable error states:

| Error Code | HTTP / API Cause | Pipeline Behavior | User Impact / Remediation |
|:---|:---|:---|:---|
| `INVALID_API_KEY` | 401 Unauthorized / 403 Forbidden / Missing Key | Immediate fail-safe return with descriptive error. | Pipeline instructs operator to configure `VISION_API_KEY` in environment or `.env`. |
| `QUOTA_EXHAUSTED` | 429 Too Many Requests / Resource Exhausted | Non-retry fail-safe return; logs quota state. | Halts runaway API billing; operator can switch model or wait for reset window. |
| `TIMEOUT` | Socket Timeout / Network Drop (>30s) | Gracefully catches timeout exception. | Logs network latency warning; falls back to text metadata if available. |
| `IMAGE_CORRUPTED` | Unreadable bytes / Invalid Magic Header | Early rejection before API payload transmission. | Zero API token waste; rejects damaged image file immediately. |
| `IMAGE_TOO_LARGE` | File size > 10MB or dimension > 4096px | Image validator rejects or downsizes payload. | Prevents API upload rejections and memory crashes. |
| `OFFLINE_MODE` | Explicit CLI `--offline` or env flag | Bypasses all remote network calls entirely. | Guarantees deterministic, 100% offline local prediction. |

---

## 3. Fallback Mechanics & Resilience Matrix

The pipeline handles multimodal input combinations with predictable degradation:

| Provided Inputs | Vision API Status | Degradation Behavior | Resulting Prediction Quality |
|:---|:---:|:---|:---|
| **Image + Title + Description** | `SUCCESS` | Full Multimodal Fusion (OCR + Metadata) | Optimal ($D=719$ features activated) |
| **Image + Title + Description** | `FAILED` / `TIMEOUT` | Graceful fallback to text metadata only | High for D01, D02, D05; degraded for D04, D07 |
| **Image Only (No Text)** | `SUCCESS` | Vision OCR extracted and parsed | Optimal for physical packaging declarations |
| **Image Only (No Text)** | `FAILED` / `OFFLINE` | Pipeline returns `VISION_REQUIRED` error | Predictable failure; prevents synthetic hallucination |
| **Text Only (No Image)** | N/A (Bypassed) | Direct offline text inference | Fast (<10ms) deterministic execution |

---

## 4. API Security & Zero Hardcoding Guarantee

1. **Environment Isolation**:
   - Zero API keys are hardcoded in source files, tests, scripts, or markdown artifacts.
   - Dynamic resolution via `os.environ.get("VISION_API_KEY")` or `os.environ.get("GEMINI_API_KEY")`.
2. **Repository Guardrails**:
   - `.gitignore` explicitly filters `.env`, `*.key`, `secrets/`, and `credentials/`.
   - Automated git status verification guarantees no sensitive keys are tracked in version control.
3. **Trace Log Masking**:
   - Inference tracer logs (`data/interim/inference_logs/`) record only sanitized metadata (provider, model name, latency, text preview) and never persist authorization headers or raw key strings.
