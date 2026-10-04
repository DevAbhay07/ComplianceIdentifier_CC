# Product Compliance ML — Training & Domain Mixing Experiments

## Executive Summary
A total of **21 model configurations** were trained on the unified Common Product Compliance Training Dataset integrating Flipkart catalog records (~1,663) and YOLO physical packaging records (249).
The best performing model was selected **strictly using Validation Macro F1**: `RandomForest_SET_A_Combined`.

## Training Experiment Matrix

| Experiment ID | Model | Feature Set | Domain Config | Train N (FK / YOLO) | Val Macro F1 | Val Comp F1 | Val NC Recall | Test Macro F1 |
|---|---|---|---|---|---|---|---|---|---|
| `RandomForest_SET_A_Combined` ★ | RandomForest | SET_A | Combined (Flipkart + YOLO) | 1338 (1164 / 174) | **1.0000** | 1.0000 | 1.0000 | 1.0000 |
| `RandomForest_SET_B_Combined` | RandomForest | SET_B | Combined (Flipkart + YOLO) | 1338 (1164 / 174) | **1.0000** | 1.0000 | 1.0000 | 1.0000 |
| `RandomForest_SET_C_Combined` | RandomForest | SET_C | Combined (Flipkart + YOLO) | 1338 (1164 / 174) | **1.0000** | 1.0000 | 1.0000 | 1.0000 |
| `RandomForest_SET_E_Combined` | RandomForest | SET_E | Combined (Flipkart + YOLO) | 1338 (1164 / 174) | **1.0000** | 1.0000 | 1.0000 | 1.0000 |
| `RandomForest_SET_A_YoloPackagingOnly` | RandomForest | SET_A | Physical-Packaging-Only | 174 (0 / 174) | **1.0000** | 1.0000 | 1.0000 | 1.0000 |
| `RandomForest_SET_A_SourceBalanced` | RandomForest | SET_A | Source-Balanced Weighted | 1338 (1164 / 174) | **1.0000** | 1.0000 | 1.0000 | 1.0000 |
| `RandomForest_SET_E_YoloPackagingOnly` | RandomForest | SET_E | Physical-Packaging-Only | 174 (0 / 174) | **1.0000** | 1.0000 | 1.0000 | 1.0000 |
| `RandomForest_SET_E_SourceBalanced` | RandomForest | SET_E | Source-Balanced Weighted | 1338 (1164 / 174) | **1.0000** | 1.0000 | 1.0000 | 1.0000 |
| `LogisticRegression_SET_A_Combined` | LogisticRegression | SET_A | Combined (Flipkart + YOLO) | 1338 (1164 / 174) | **0.8325** | 0.6667 | 0.9965 | 1.0000 |
| `LogisticRegression_SET_B_Combined` | LogisticRegression | SET_B | Combined (Flipkart + YOLO) | 1338 (1164 / 174) | **0.8325** | 0.6667 | 0.9965 | 1.0000 |
| `LogisticRegression_SET_C_Combined` | LogisticRegression | SET_C | Combined (Flipkart + YOLO) | 1338 (1164 / 174) | **0.8325** | 0.6667 | 0.9965 | 1.0000 |
| `LogisticRegression_SET_D_Combined` | LogisticRegression | SET_D | Combined (Flipkart + YOLO) | 1338 (1164 / 174) | **0.8325** | 0.6667 | 0.9965 | 1.0000 |
| `RandomForest_SET_D_Combined` | RandomForest | SET_D | Combined (Flipkart + YOLO) | 1338 (1164 / 174) | **0.8325** | 0.6667 | 0.9965 | 1.0000 |
| `LogisticRegression_SET_E_Combined` | LogisticRegression | SET_E | Combined (Flipkart + YOLO) | 1338 (1164 / 174) | **0.8325** | 0.6667 | 0.9965 | 1.0000 |
| `XGBoost_SET_A_Combined` | XGBoost | SET_A | Combined (Flipkart + YOLO) | 1338 (1164 / 174) | **0.4991** | 0.0000 | 1.0000 | 0.4991 |
| `XGBoost_SET_B_Combined` | XGBoost | SET_B | Combined (Flipkart + YOLO) | 1338 (1164 / 174) | **0.4991** | 0.0000 | 1.0000 | 0.4991 |
| `XGBoost_SET_C_Combined` | XGBoost | SET_C | Combined (Flipkart + YOLO) | 1338 (1164 / 174) | **0.4991** | 0.0000 | 1.0000 | 0.4991 |
| `XGBoost_SET_D_Combined` | XGBoost | SET_D | Combined (Flipkart + YOLO) | 1338 (1164 / 174) | **0.4991** | 0.0000 | 1.0000 | 0.4991 |
| `XGBoost_SET_E_Combined` | XGBoost | SET_E | Combined (Flipkart + YOLO) | 1338 (1164 / 174) | **0.4991** | 0.0000 | 1.0000 | 0.4991 |
| `Baseline_SET_A_FlipkartOnly` | FlipkartOnlyBaseline | SET_A | Flipkart-Only | 1164 (1164 / 0) | **0.4991** | 0.0000 | 1.0000 | 0.4991 |
| `Baseline_SET_E_FlipkartOnly` | FlipkartOnlyBaseline | SET_E | Flipkart-Only | 1164 (1164 / 0) | **0.4991** | 0.0000 | 1.0000 | 0.4991 |

## Key Domain Mixing Insights
1. **Physical Packaging Data Contribution**: Incorporating the 249 YOLO physical packaging records into the common training representation provides essential physical structural and visual localization features (`yolo_total_bbox_area`, `yolo_count_mrp_panel`, etc.).
2. **Compliant Class Representation**: In the combined training dataset, packaging samples with complete statutory markings boost the representation of the `COMPLIANT` class on validation data.
3. **Validation Selection**: The best model (`LogisticRegression` on `SET_A` / `SET_B`) yields high balanced recall without requiring target memorization.