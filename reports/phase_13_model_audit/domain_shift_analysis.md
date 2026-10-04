# Phase 13 — Domain Shift Analysis: Catalog Metadata vs Physical Packaging

## Domain Discrepancy Matrix

| Characteristic | Catalog Silver (Weak Test N=258) | Human Catalog (Human Ref N=75) | Physical Packaging (Benchmark N=20) |
|---|---|---|---|
| **Data Source** | E-commerce scraped HTML text | Scraped catalog product pages | High-resolution multi-view camera photos |
| **Class Balance** | 96% Non-Compliant / 4% Compliant | 84% Non-Compliant / 16% Compliant | 15% Non-Compliant / 85% Compliant |
| **D04 (Mfg Date) Presence** | <5% present in text | <10% present in text | **>90% printed on physical box/bottle** |
| **D06 (Helpline Phone/Email)**| <15% present in text | <20% present in text | **>85% printed on physical box/bottle** |
| **Optical Noise / Layout** | Structured text fields | Web text fields | 3D curves, lighting reflections, fonts |

## Core Conclusion
Models trained on Flipkart catalog text learn to associate the absence of catalog text with statutory non-compliance.
When presented with physical packaging images where dates and contacts are printed directly on the packaging, the catalog-domain model fails to generalize, causing the observed generalization drop.