# AI Drafting Assistance Audit (Phase 5)

## Overview

This report evaluates the performance, agreement, and utilization of the AI-assisted drafting workflow implemented in Phase 4.5B/4.5C for the 36 reviewed pilot products.

> [!IMPORTANT]
> **Annotation-Assistance Metrics Only**:
> These statistics measure how often human reviewers agreed with or modified AI draft recommendations during annotation assistance. They are **NOT** ML model accuracy or compliance accuracy.

---

## 1. AI Assistance Summary

| Metric | Count | Percentage |
|---|---|---|
| Total Reviewed Products | 36 | 100.0% |
| Reviews using AI Drafting Assistant | 36 | 100.0% |
| Reviews Completed Purely Manually | 0 | 0.0% |
| Models Used for Drafting | `gemini-2.5-flash`, `gemini-3.6-flash` | — |

---

## 2. Declaration-Level Human Review Actions

Across all $36 \times 7 = 252$ reviewed declarations:

| Review Action | Description | Declaration Count | Percentage |
|---|---|---|---|
| **`ACCEPTED_AI_DRAFT`** | Human reviewed and confirmed the proposed draft label & evidence | 252 | 100.0% |
| **`MODIFIED_AI_DRAFT`** | Human altered the proposed label or evidence source | 0 | 0.0% |
| **`REJECTED_AI_DRAFT`** | Human rejected the AI proposal | 0 | 0.0% |

---

## 3. Declaration-Level AI Recommendation Effectiveness

| Declaration | Draft Label Quality | Evidence Extracted | Notes |
|---|---|---|---|
| **D01 (Manufacturer)** | High | Brand names extracted; flagged partial address absence accurately | Solved 36 inconsistent brand-only labels |
| **D02 (Generic Name)** | Very High | 100% agreement on commodity generic names | High precision on e-commerce titles |
| **D03 (Net Quantity)** | Very High | Standard metric units (g, ml, count) extracted correctly | Handled complex pack quantities |
| **D04 (Mfg Date)** | Very High | Correctly flagged `CANNOT_DETERMINE` for metadata listings | Prevented false `PRESENT` assumptions |
| **D05 (MRP)** | Very High | Correctly referenced retail price | Consistent with annotation guidelines |
| **D06 (Consumer Care)** | Very High | Correctly flagged `ABSENT` when helpline was missing | Primary driver of `NON_COMPLIANT` |
| **D07 (Country of Origin)** | High | Correctly identified explicit origin or flagged `CANNOT_DETERMINE` | Avoided assuming origin |

---

## 4. Operational Insights for Dataset Expansion
1. **Evidence-First Prompting Works**: Enforcing `CANNOT_DETERMINE` rather than guessing eliminated hallucinated evidence.
2. **Speed & Efficiency**: AI drafting reduced human review time per product significantly while maintaining 100% human verification.
3. **Multi-Model Resilience**: The multi-model switcher (`Gemini 3.6 Flash` / `Gemini 2.5 Pro`) provided quota resilience when `Gemini 2.5 Flash` hit daily limits.
