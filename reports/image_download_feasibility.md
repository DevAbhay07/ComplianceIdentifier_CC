# Image Download Feasibility Report

**Date**: 2026-09-05
**Phase**: 4 — Pilot Analysis

---

## Test Parameters

| Parameter | Value |
|---|---|
| Products tested | 25 (from 75 pilot products) |
| Selection method | Random sample (seed=42) |
| CDN host | flixcart.com (Flipkart legacy CDN) |
| Dataset vintage | 2016 |

---

## Results

| Metric | Value |
|---|---|
| **Accessible (HTTP 200)** | **24 (96%)** |
| **Inaccessible (HTTP 404)** | **1 (4%)** |
| **Content types** | image/jpeg (23), image/webp (1) |
| **File sizes** | 24 KB — 231 KB (median: 129 KB) |
| **Image dimensions** | 220×551 — 1299×3149 |
| **URLs per product** | 1 — 4 (median: 2) |

---

## Assessment

### Positive Findings

1. **96% accessibility rate** — the Flipkart CDN is still serving images from 2016, which is unexpectedly strong for a dataset this old.

2. **High resolution** — most images are 800+ pixels on the longest side. This is sufficient for OCR text extraction from packaging labels.

3. **Multiple views available** — products typically have 2-4 image URLs, meaning multiple product angles may be available (potentially including back label / information panel).

4. **Reasonable file sizes** — at ~130 KB median, downloading images for 1,000-2,000 products would require approximately 130-260 MB total storage.

### Concerns

1. **Not all images show packaging labels** — e-commerce product images typically show the product front or marketing shot, not the legal information panel. Additional image angles may be needed.

2. **CDN stability unknown** — the 96% rate is a snapshot. The CDN could begin returning 404s at any time since these are legacy URLs.

3. **No guarantee of label readability** — images may be too zoomed out, blurry, or at an angle that prevents reliable OCR extraction of fine print (manufacturer address, consumer care details, etc.).

4. **One webp format detected** — the pipeline should handle both JPEG and WebP formats.

---

## Recommendation

**Images CAN realistically be collected** for the next annotation phase.

### Recommended Approach

1. **Download images for the expanded dataset** (Stage 1: ~300-500 products)
2. **Store locally** in `data/interim/images/{product_id}/` with one subdirectory per product
3. **Record download metadata** (URL, HTTP status, file size, dimensions, timestamp) for reproducibility
4. **Quality filter** — discard images < 200px on shortest side, flag non-JPEG/WebP formats
5. **Rate limit** — use 0.3-0.5s delay between downloads to avoid CDN throttling
6. **Build a second annotation pass** — annotators review images alongside listing metadata for declarations (D01, D04, D06, D07) that benefit most from packaging label visibility

### NOT Recommended

- Downloading all 20,000 product images at once
- Relying solely on images without listing metadata
- Assuming all images show packaging labels
- Running OCR without human verification of extraction quality
