# Phase 9.5 Real-Product Failure Taxonomy & Root Cause Analysis

> **Failure Taxonomy Framework**: Standardized root-cause categorization across Categories A through H.

## 1. Failure Taxonomy Breakdown

| Category Code | Failure Description | Count | % of Total Errors | Primary Remediation |
|:---|:---|:---:|:---:|:---|
| **A** (`IMAGE_QUALITY`) | IMAGE_QUALITY | 0 | 0.0% | Add camera autofocus and illumination guidance in ingestion layer. |
| **B** (`OCR_FAILURE`) | OCR_FAILURE | 0 | 0.0% | Upgrade to multi-scale OCR with region-of-interest cropping. |
| **C** (`EVIDENCE_PARSER_FAILURE`) | EVIDENCE_PARSER_FAILURE | 0 | 0.0% | Expand regex vocabulary for multi-lingual and abbreviated dates/origins. |
| **D** (`FEATURE_EXTRACTION_FAILURE`) | FEATURE_EXTRACTION_FAILURE | 0 | 0.0% | Add packaging-specific feature extractors. |
| **E** (`FEATURE_DISTRIBUTION_SHIFT`) | FEATURE_DISTRIBUTION_SHIFT | 36 | 51.4% | Train supervised multimodal models on physical packaging datasets (Phase 10). |
| **F** (`ML_MODEL_ERROR`) | ML_MODEL_ERROR | 34 | 48.6% | Calibrate probability thresholds and add cost-sensitive loss. |
| **G** (`AMBIGUOUS_EVIDENCE`) | AMBIGUOUS_EVIDENCE | 0 | 0.0% | Establish dual-annotator verification guidelines. |
| **H** (`OTHER`) | OTHER | 0 | 0.0% | Edge-case error handling. |

## 2. High-Confidence Misclassifications (>= 0.80 Confidence)

High-confidence errors represent critical risk points where the model exhibits overconfidence in incorrect predictions.

| Product ID | Declaration | Ground Truth | Predicted | Confidence | Model | Root Cause & Explanation |
|:---|:---:|:---:|:---:|:---:|:---|:---|
| `real_prod_01` | `D04` | `PRESENT` | `CANNOT_DETERMINE` | 1.00 | MajorityBaseline | Baseline single-class predictor trained on Flipkart weak labels (always predicts majority class). Failed domain transfer to physical label. |
| `real_prod_01` | `D06` | `PRESENT` | `ABSENT` | 0.93 | RandomForest | High-confidence ML model misclassification (conf=0.93). Model overweighted text length/title cues over OCR evidence. |
| `real_prod_01` | `D07` | `PRESENT` | `CANNOT_DETERMINE` | 1.00 | MajorityBaseline | Baseline single-class predictor trained on Flipkart weak labels (always predicts majority class). Failed domain transfer to physical label. |
| `real_prod_02` | `D04` | `PRESENT` | `CANNOT_DETERMINE` | 1.00 | MajorityBaseline | Baseline single-class predictor trained on Flipkart weak labels (always predicts majority class). Failed domain transfer to physical label. |
| `real_prod_02` | `D06` | `PRESENT` | `ABSENT` | 0.90 | RandomForest | High-confidence ML model misclassification (conf=0.90). Model overweighted text length/title cues over OCR evidence. |
| `real_prod_02` | `D07` | `PRESENT` | `CANNOT_DETERMINE` | 1.00 | MajorityBaseline | Baseline single-class predictor trained on Flipkart weak labels (always predicts majority class). Failed domain transfer to physical label. |
| `real_prod_03` | `D04` | `PRESENT` | `CANNOT_DETERMINE` | 1.00 | MajorityBaseline | Baseline single-class predictor trained on Flipkart weak labels (always predicts majority class). Failed domain transfer to physical label. |
| `real_prod_03` | `D06` | `PRESENT` | `ABSENT` | 0.95 | RandomForest | High-confidence ML model misclassification (conf=0.95). Model overweighted text length/title cues over OCR evidence. |
| `real_prod_03` | `D07` | `PRESENT` | `CANNOT_DETERMINE` | 1.00 | MajorityBaseline | Baseline single-class predictor trained on Flipkart weak labels (always predicts majority class). Failed domain transfer to physical label. |
| `real_prod_04` | `D04` | `PRESENT` | `CANNOT_DETERMINE` | 1.00 | MajorityBaseline | Baseline single-class predictor trained on Flipkart weak labels (always predicts majority class). Failed domain transfer to physical label. |
| `real_prod_04` | `D06` | `PRESENT` | `ABSENT` | 0.94 | RandomForest | High-confidence ML model misclassification (conf=0.94). Model overweighted text length/title cues over OCR evidence. |
| `real_prod_04` | `D07` | `PRESENT` | `CANNOT_DETERMINE` | 1.00 | MajorityBaseline | Baseline single-class predictor trained on Flipkart weak labels (always predicts majority class). Failed domain transfer to physical label. |
| `real_prod_05` | `D04` | `PRESENT` | `CANNOT_DETERMINE` | 1.00 | MajorityBaseline | Baseline single-class predictor trained on Flipkart weak labels (always predicts majority class). Failed domain transfer to physical label. |
| `real_prod_05` | `D06` | `PRESENT` | `ABSENT` | 0.94 | RandomForest | High-confidence ML model misclassification (conf=0.94). Model overweighted text length/title cues over OCR evidence. |
| `real_prod_05` | `D07` | `PRESENT` | `CANNOT_DETERMINE` | 1.00 | MajorityBaseline | Baseline single-class predictor trained on Flipkart weak labels (always predicts majority class). Failed domain transfer to physical label. |
| `real_prod_06` | `D04` | `PRESENT` | `CANNOT_DETERMINE` | 1.00 | MajorityBaseline | Baseline single-class predictor trained on Flipkart weak labels (always predicts majority class). Failed domain transfer to physical label. |
| `real_prod_06` | `D06` | `PRESENT` | `ABSENT` | 0.95 | RandomForest | High-confidence ML model misclassification (conf=0.95). Model overweighted text length/title cues over OCR evidence. |
| `real_prod_06` | `D07` | `PRESENT` | `CANNOT_DETERMINE` | 1.00 | MajorityBaseline | Baseline single-class predictor trained on Flipkart weak labels (always predicts majority class). Failed domain transfer to physical label. |
| `real_prod_07` | `D04` | `PRESENT` | `CANNOT_DETERMINE` | 1.00 | MajorityBaseline | Baseline single-class predictor trained on Flipkart weak labels (always predicts majority class). Failed domain transfer to physical label. |
| `real_prod_07` | `D06` | `PRESENT` | `ABSENT` | 0.91 | RandomForest | High-confidence ML model misclassification (conf=0.91). Model overweighted text length/title cues over OCR evidence. |
| `real_prod_07` | `D07` | `PRESENT` | `CANNOT_DETERMINE` | 1.00 | MajorityBaseline | Baseline single-class predictor trained on Flipkart weak labels (always predicts majority class). Failed domain transfer to physical label. |
| `real_prod_08` | `D04` | `PRESENT` | `CANNOT_DETERMINE` | 1.00 | MajorityBaseline | Baseline single-class predictor trained on Flipkart weak labels (always predicts majority class). Failed domain transfer to physical label. |
| `real_prod_08` | `D06` | `PRESENT` | `ABSENT` | 0.85 | RandomForest | High-confidence ML model misclassification (conf=0.85). Model overweighted text length/title cues over OCR evidence. |
| `real_prod_08` | `D07` | `PRESENT` | `CANNOT_DETERMINE` | 1.00 | MajorityBaseline | Baseline single-class predictor trained on Flipkart weak labels (always predicts majority class). Failed domain transfer to physical label. |
| `real_prod_09` | `D04` | `PRESENT` | `CANNOT_DETERMINE` | 1.00 | MajorityBaseline | Baseline single-class predictor trained on Flipkart weak labels (always predicts majority class). Failed domain transfer to physical label. |
| `real_prod_09` | `D06` | `PRESENT` | `ABSENT` | 0.95 | RandomForest | High-confidence ML model misclassification (conf=0.95). Model overweighted text length/title cues over OCR evidence. |
| `real_prod_09` | `D07` | `PRESENT` | `CANNOT_DETERMINE` | 1.00 | MajorityBaseline | Baseline single-class predictor trained on Flipkart weak labels (always predicts majority class). Failed domain transfer to physical label. |
| `real_prod_10` | `D04` | `PRESENT` | `CANNOT_DETERMINE` | 1.00 | MajorityBaseline | Baseline single-class predictor trained on Flipkart weak labels (always predicts majority class). Failed domain transfer to physical label. |
| `real_prod_10` | `D06` | `PRESENT` | `ABSENT` | 0.89 | RandomForest | High-confidence ML model misclassification (conf=0.89). Model overweighted text length/title cues over OCR evidence. |
| `real_prod_10` | `D07` | `PRESENT` | `CANNOT_DETERMINE` | 1.00 | MajorityBaseline | Baseline single-class predictor trained on Flipkart weak labels (always predicts majority class). Failed domain transfer to physical label. |
| `real_prod_12` | `D07` | `PRESENT` | `CANNOT_DETERMINE` | 1.00 | MajorityBaseline | Baseline single-class predictor trained on Flipkart weak labels (always predicts majority class). Failed domain transfer to physical label. |
| `real_prod_13` | `D04` | `PRESENT` | `CANNOT_DETERMINE` | 1.00 | MajorityBaseline | Baseline single-class predictor trained on Flipkart weak labels (always predicts majority class). Failed domain transfer to physical label. |
| `real_prod_13` | `D06` | `PRESENT` | `ABSENT` | 0.86 | RandomForest | High-confidence ML model misclassification (conf=0.86). Model overweighted text length/title cues over OCR evidence. |
| `real_prod_13` | `D07` | `PRESENT` | `CANNOT_DETERMINE` | 1.00 | MajorityBaseline | Baseline single-class predictor trained on Flipkart weak labels (always predicts majority class). Failed domain transfer to physical label. |
| `real_prod_14` | `D04` | `PRESENT` | `CANNOT_DETERMINE` | 1.00 | MajorityBaseline | Baseline single-class predictor trained on Flipkart weak labels (always predicts majority class). Failed domain transfer to physical label. |
| `real_prod_14` | `D06` | `PRESENT` | `ABSENT` | 0.95 | RandomForest | High-confidence ML model misclassification (conf=0.95). Model overweighted text length/title cues over OCR evidence. |
| `real_prod_14` | `D07` | `PRESENT` | `CANNOT_DETERMINE` | 1.00 | MajorityBaseline | Baseline single-class predictor trained on Flipkart weak labels (always predicts majority class). Failed domain transfer to physical label. |
| `real_prod_15` | `D04` | `PRESENT` | `CANNOT_DETERMINE` | 1.00 | MajorityBaseline | Baseline single-class predictor trained on Flipkart weak labels (always predicts majority class). Failed domain transfer to physical label. |
| `real_prod_15` | `D06` | `PRESENT` | `ABSENT` | 0.95 | RandomForest | High-confidence ML model misclassification (conf=0.95). Model overweighted text length/title cues over OCR evidence. |
| `real_prod_15` | `D07` | `PRESENT` | `CANNOT_DETERMINE` | 1.00 | MajorityBaseline | Baseline single-class predictor trained on Flipkart weak labels (always predicts majority class). Failed domain transfer to physical label. |
| `real_prod_16` | `D04` | `PRESENT` | `CANNOT_DETERMINE` | 1.00 | MajorityBaseline | Baseline single-class predictor trained on Flipkart weak labels (always predicts majority class). Failed domain transfer to physical label. |
| `real_prod_16` | `D06` | `PRESENT` | `ABSENT` | 0.95 | RandomForest | High-confidence ML model misclassification (conf=0.95). Model overweighted text length/title cues over OCR evidence. |
| `real_prod_16` | `D07` | `PRESENT` | `CANNOT_DETERMINE` | 1.00 | MajorityBaseline | Baseline single-class predictor trained on Flipkart weak labels (always predicts majority class). Failed domain transfer to physical label. |
| `real_prod_17` | `D07` | `PRESENT` | `CANNOT_DETERMINE` | 1.00 | MajorityBaseline | Baseline single-class predictor trained on Flipkart weak labels (always predicts majority class). Failed domain transfer to physical label. |
| `real_prod_18` | `D04` | `PRESENT` | `CANNOT_DETERMINE` | 1.00 | MajorityBaseline | Baseline single-class predictor trained on Flipkart weak labels (always predicts majority class). Failed domain transfer to physical label. |
| `real_prod_18` | `D06` | `PRESENT` | `ABSENT` | 0.85 | RandomForest | High-confidence ML model misclassification (conf=0.85). Model overweighted text length/title cues over OCR evidence. |
| `real_prod_18` | `D07` | `PRESENT` | `CANNOT_DETERMINE` | 1.00 | MajorityBaseline | Baseline single-class predictor trained on Flipkart weak labels (always predicts majority class). Failed domain transfer to physical label. |
| `real_prod_19` | `D04` | `PRESENT` | `CANNOT_DETERMINE` | 1.00 | MajorityBaseline | Baseline single-class predictor trained on Flipkart weak labels (always predicts majority class). Failed domain transfer to physical label. |
| `real_prod_19` | `D06` | `PRESENT` | `ABSENT` | 0.87 | RandomForest | High-confidence ML model misclassification (conf=0.87). Model overweighted text length/title cues over OCR evidence. |
| `real_prod_19` | `D07` | `PRESENT` | `CANNOT_DETERMINE` | 1.00 | MajorityBaseline | Baseline single-class predictor trained on Flipkart weak labels (always predicts majority class). Failed domain transfer to physical label. |
| `real_prod_20` | `D04` | `PRESENT` | `CANNOT_DETERMINE` | 1.00 | MajorityBaseline | Baseline single-class predictor trained on Flipkart weak labels (always predicts majority class). Failed domain transfer to physical label. |
| `real_prod_20` | `D06` | `PRESENT` | `ABSENT` | 0.92 | RandomForest | High-confidence ML model misclassification (conf=0.92). Model overweighted text length/title cues over OCR evidence. |
| `real_prod_20` | `D07` | `PRESENT` | `CANNOT_DETERMINE` | 1.00 | MajorityBaseline | Baseline single-class predictor trained on Flipkart weak labels (always predicts majority class). Failed domain transfer to physical label. |

## 3. Comprehensive Error Log (All Mismatches)

| Product ID | Dec | Ground Truth | Predicted | Conf | Category | Explanation |
|:---|:---:|:---:|:---:|:---:|:---|:---|
| `real_prod_01` | `D01` | `PRESENT` | `PARTIAL` | 0.56 | ML_MODEL_ERROR | ML model error: feature pattern mapped to incorrect class probability distribution (predicted PARTIAL, expected PRESENT). |
| `real_prod_01` | `D04` | `PRESENT` | `CANNOT_DETERMINE` | 1.00 | FEATURE_DISTRIBUTION_SHIFT | Baseline single-class predictor trained on Flipkart weak labels (always predicts majority class). Failed domain transfer to physical label. |
| `real_prod_01` | `D06` | `PRESENT` | `ABSENT` | 0.93 | ML_MODEL_ERROR | High-confidence ML model misclassification (conf=0.93). Model overweighted text length/title cues over OCR evidence. |
| `real_prod_01` | `D07` | `PRESENT` | `CANNOT_DETERMINE` | 1.00 | FEATURE_DISTRIBUTION_SHIFT | Baseline single-class predictor trained on Flipkart weak labels (always predicts majority class). Failed domain transfer to physical label. |
| `real_prod_02` | `D01` | `PRESENT` | `PARTIAL` | 0.52 | ML_MODEL_ERROR | ML model error: feature pattern mapped to incorrect class probability distribution (predicted PARTIAL, expected PRESENT). |
| `real_prod_02` | `D04` | `PRESENT` | `CANNOT_DETERMINE` | 1.00 | FEATURE_DISTRIBUTION_SHIFT | Baseline single-class predictor trained on Flipkart weak labels (always predicts majority class). Failed domain transfer to physical label. |
| `real_prod_02` | `D06` | `PRESENT` | `ABSENT` | 0.90 | ML_MODEL_ERROR | High-confidence ML model misclassification (conf=0.90). Model overweighted text length/title cues over OCR evidence. |
| `real_prod_02` | `D07` | `PRESENT` | `CANNOT_DETERMINE` | 1.00 | FEATURE_DISTRIBUTION_SHIFT | Baseline single-class predictor trained on Flipkart weak labels (always predicts majority class). Failed domain transfer to physical label. |
| `real_prod_03` | `D04` | `PRESENT` | `CANNOT_DETERMINE` | 1.00 | FEATURE_DISTRIBUTION_SHIFT | Baseline single-class predictor trained on Flipkart weak labels (always predicts majority class). Failed domain transfer to physical label. |
| `real_prod_03` | `D06` | `PRESENT` | `ABSENT` | 0.95 | ML_MODEL_ERROR | High-confidence ML model misclassification (conf=0.95). Model overweighted text length/title cues over OCR evidence. |
| `real_prod_03` | `D07` | `PRESENT` | `CANNOT_DETERMINE` | 1.00 | FEATURE_DISTRIBUTION_SHIFT | Baseline single-class predictor trained on Flipkart weak labels (always predicts majority class). Failed domain transfer to physical label. |
| `real_prod_04` | `D01` | `PRESENT` | `PARTIAL` | 0.68 | ML_MODEL_ERROR | ML model error: feature pattern mapped to incorrect class probability distribution (predicted PARTIAL, expected PRESENT). |
| `real_prod_04` | `D04` | `PRESENT` | `CANNOT_DETERMINE` | 1.00 | FEATURE_DISTRIBUTION_SHIFT | Baseline single-class predictor trained on Flipkart weak labels (always predicts majority class). Failed domain transfer to physical label. |
| `real_prod_04` | `D06` | `PRESENT` | `ABSENT` | 0.94 | ML_MODEL_ERROR | High-confidence ML model misclassification (conf=0.94). Model overweighted text length/title cues over OCR evidence. |
| `real_prod_04` | `D07` | `PRESENT` | `CANNOT_DETERMINE` | 1.00 | FEATURE_DISTRIBUTION_SHIFT | Baseline single-class predictor trained on Flipkart weak labels (always predicts majority class). Failed domain transfer to physical label. |
| `real_prod_05` | `D01` | `PRESENT` | `PARTIAL` | 0.60 | ML_MODEL_ERROR | ML model error: feature pattern mapped to incorrect class probability distribution (predicted PARTIAL, expected PRESENT). |
| `real_prod_05` | `D04` | `PRESENT` | `CANNOT_DETERMINE` | 1.00 | FEATURE_DISTRIBUTION_SHIFT | Baseline single-class predictor trained on Flipkart weak labels (always predicts majority class). Failed domain transfer to physical label. |
| `real_prod_05` | `D06` | `PRESENT` | `ABSENT` | 0.94 | ML_MODEL_ERROR | High-confidence ML model misclassification (conf=0.94). Model overweighted text length/title cues over OCR evidence. |
| `real_prod_05` | `D07` | `PRESENT` | `CANNOT_DETERMINE` | 1.00 | FEATURE_DISTRIBUTION_SHIFT | Baseline single-class predictor trained on Flipkart weak labels (always predicts majority class). Failed domain transfer to physical label. |
| `real_prod_06` | `D01` | `PRESENT` | `PARTIAL` | 0.54 | ML_MODEL_ERROR | ML model error: feature pattern mapped to incorrect class probability distribution (predicted PARTIAL, expected PRESENT). |
| `real_prod_06` | `D04` | `PRESENT` | `CANNOT_DETERMINE` | 1.00 | FEATURE_DISTRIBUTION_SHIFT | Baseline single-class predictor trained on Flipkart weak labels (always predicts majority class). Failed domain transfer to physical label. |
| `real_prod_06` | `D06` | `PRESENT` | `ABSENT` | 0.95 | ML_MODEL_ERROR | High-confidence ML model misclassification (conf=0.95). Model overweighted text length/title cues over OCR evidence. |
| `real_prod_06` | `D07` | `PRESENT` | `CANNOT_DETERMINE` | 1.00 | FEATURE_DISTRIBUTION_SHIFT | Baseline single-class predictor trained on Flipkart weak labels (always predicts majority class). Failed domain transfer to physical label. |
| `real_prod_07` | `D01` | `PRESENT` | `PARTIAL` | 0.46 | ML_MODEL_ERROR | ML model error: feature pattern mapped to incorrect class probability distribution (predicted PARTIAL, expected PRESENT). |
| `real_prod_07` | `D04` | `PRESENT` | `CANNOT_DETERMINE` | 1.00 | FEATURE_DISTRIBUTION_SHIFT | Baseline single-class predictor trained on Flipkart weak labels (always predicts majority class). Failed domain transfer to physical label. |
| `real_prod_07` | `D06` | `PRESENT` | `ABSENT` | 0.91 | ML_MODEL_ERROR | High-confidence ML model misclassification (conf=0.91). Model overweighted text length/title cues over OCR evidence. |
| `real_prod_07` | `D07` | `PRESENT` | `CANNOT_DETERMINE` | 1.00 | FEATURE_DISTRIBUTION_SHIFT | Baseline single-class predictor trained on Flipkart weak labels (always predicts majority class). Failed domain transfer to physical label. |
| `real_prod_08` | `D04` | `PRESENT` | `CANNOT_DETERMINE` | 1.00 | FEATURE_DISTRIBUTION_SHIFT | Baseline single-class predictor trained on Flipkart weak labels (always predicts majority class). Failed domain transfer to physical label. |
| `real_prod_08` | `D06` | `PRESENT` | `ABSENT` | 0.85 | ML_MODEL_ERROR | High-confidence ML model misclassification (conf=0.85). Model overweighted text length/title cues over OCR evidence. |
| `real_prod_08` | `D07` | `PRESENT` | `CANNOT_DETERMINE` | 1.00 | FEATURE_DISTRIBUTION_SHIFT | Baseline single-class predictor trained on Flipkart weak labels (always predicts majority class). Failed domain transfer to physical label. |
| `real_prod_09` | `D01` | `PRESENT` | `PARTIAL` | 0.56 | ML_MODEL_ERROR | ML model error: feature pattern mapped to incorrect class probability distribution (predicted PARTIAL, expected PRESENT). |
| `real_prod_09` | `D04` | `PRESENT` | `CANNOT_DETERMINE` | 1.00 | FEATURE_DISTRIBUTION_SHIFT | Baseline single-class predictor trained on Flipkart weak labels (always predicts majority class). Failed domain transfer to physical label. |
| `real_prod_09` | `D06` | `PRESENT` | `ABSENT` | 0.95 | ML_MODEL_ERROR | High-confidence ML model misclassification (conf=0.95). Model overweighted text length/title cues over OCR evidence. |
| `real_prod_09` | `D07` | `PRESENT` | `CANNOT_DETERMINE` | 1.00 | FEATURE_DISTRIBUTION_SHIFT | Baseline single-class predictor trained on Flipkart weak labels (always predicts majority class). Failed domain transfer to physical label. |
| `real_prod_10` | `D01` | `PRESENT` | `PARTIAL` | 0.74 | ML_MODEL_ERROR | ML model error: feature pattern mapped to incorrect class probability distribution (predicted PARTIAL, expected PRESENT). |
| `real_prod_10` | `D04` | `PRESENT` | `CANNOT_DETERMINE` | 1.00 | FEATURE_DISTRIBUTION_SHIFT | Baseline single-class predictor trained on Flipkart weak labels (always predicts majority class). Failed domain transfer to physical label. |
| `real_prod_10` | `D06` | `PRESENT` | `ABSENT` | 0.89 | ML_MODEL_ERROR | High-confidence ML model misclassification (conf=0.89). Model overweighted text length/title cues over OCR evidence. |
| `real_prod_10` | `D07` | `PRESENT` | `CANNOT_DETERMINE` | 1.00 | FEATURE_DISTRIBUTION_SHIFT | Baseline single-class predictor trained on Flipkart weak labels (always predicts majority class). Failed domain transfer to physical label. |
| `real_prod_11` | `D01` | `ABSENT` | `PRESENT` | 0.50 | ML_MODEL_ERROR | ML model error: feature pattern mapped to incorrect class probability distribution (predicted PRESENT, expected ABSENT). |
| `real_prod_12` | `D07` | `PRESENT` | `CANNOT_DETERMINE` | 1.00 | FEATURE_DISTRIBUTION_SHIFT | Baseline single-class predictor trained on Flipkart weak labels (always predicts majority class). Failed domain transfer to physical label. |
| `real_prod_13` | `D01` | `PRESENT` | `PARTIAL` | 0.60 | ML_MODEL_ERROR | ML model error: feature pattern mapped to incorrect class probability distribution (predicted PARTIAL, expected PRESENT). |
| `real_prod_13` | `D04` | `PRESENT` | `CANNOT_DETERMINE` | 1.00 | FEATURE_DISTRIBUTION_SHIFT | Baseline single-class predictor trained on Flipkart weak labels (always predicts majority class). Failed domain transfer to physical label. |
| `real_prod_13` | `D06` | `PRESENT` | `ABSENT` | 0.86 | ML_MODEL_ERROR | High-confidence ML model misclassification (conf=0.86). Model overweighted text length/title cues over OCR evidence. |
| `real_prod_13` | `D07` | `PRESENT` | `CANNOT_DETERMINE` | 1.00 | FEATURE_DISTRIBUTION_SHIFT | Baseline single-class predictor trained on Flipkart weak labels (always predicts majority class). Failed domain transfer to physical label. |
| `real_prod_14` | `D01` | `PRESENT` | `PARTIAL` | 0.56 | ML_MODEL_ERROR | ML model error: feature pattern mapped to incorrect class probability distribution (predicted PARTIAL, expected PRESENT). |
| `real_prod_14` | `D04` | `PRESENT` | `CANNOT_DETERMINE` | 1.00 | FEATURE_DISTRIBUTION_SHIFT | Baseline single-class predictor trained on Flipkart weak labels (always predicts majority class). Failed domain transfer to physical label. |
| `real_prod_14` | `D06` | `PRESENT` | `ABSENT` | 0.95 | ML_MODEL_ERROR | High-confidence ML model misclassification (conf=0.95). Model overweighted text length/title cues over OCR evidence. |
| `real_prod_14` | `D07` | `PRESENT` | `CANNOT_DETERMINE` | 1.00 | FEATURE_DISTRIBUTION_SHIFT | Baseline single-class predictor trained on Flipkart weak labels (always predicts majority class). Failed domain transfer to physical label. |
| `real_prod_15` | `D01` | `PRESENT` | `PARTIAL` | 0.56 | ML_MODEL_ERROR | ML model error: feature pattern mapped to incorrect class probability distribution (predicted PARTIAL, expected PRESENT). |
| `real_prod_15` | `D04` | `PRESENT` | `CANNOT_DETERMINE` | 1.00 | FEATURE_DISTRIBUTION_SHIFT | Baseline single-class predictor trained on Flipkart weak labels (always predicts majority class). Failed domain transfer to physical label. |
| `real_prod_15` | `D06` | `PRESENT` | `ABSENT` | 0.95 | ML_MODEL_ERROR | High-confidence ML model misclassification (conf=0.95). Model overweighted text length/title cues over OCR evidence. |
| `real_prod_15` | `D07` | `PRESENT` | `CANNOT_DETERMINE` | 1.00 | FEATURE_DISTRIBUTION_SHIFT | Baseline single-class predictor trained on Flipkart weak labels (always predicts majority class). Failed domain transfer to physical label. |
| `real_prod_16` | `D01` | `PRESENT` | `PARTIAL` | 0.58 | ML_MODEL_ERROR | ML model error: feature pattern mapped to incorrect class probability distribution (predicted PARTIAL, expected PRESENT). |
| `real_prod_16` | `D04` | `PRESENT` | `CANNOT_DETERMINE` | 1.00 | FEATURE_DISTRIBUTION_SHIFT | Baseline single-class predictor trained on Flipkart weak labels (always predicts majority class). Failed domain transfer to physical label. |
| `real_prod_16` | `D06` | `PRESENT` | `ABSENT` | 0.95 | ML_MODEL_ERROR | High-confidence ML model misclassification (conf=0.95). Model overweighted text length/title cues over OCR evidence. |
| `real_prod_16` | `D07` | `PRESENT` | `CANNOT_DETERMINE` | 1.00 | FEATURE_DISTRIBUTION_SHIFT | Baseline single-class predictor trained on Flipkart weak labels (always predicts majority class). Failed domain transfer to physical label. |
| `real_prod_17` | `D01` | `PARTIAL` | `PRESENT` | 0.54 | ML_MODEL_ERROR | ML model error: feature pattern mapped to incorrect class probability distribution (predicted PRESENT, expected PARTIAL). |
| `real_prod_17` | `D07` | `PRESENT` | `CANNOT_DETERMINE` | 1.00 | FEATURE_DISTRIBUTION_SHIFT | Baseline single-class predictor trained on Flipkart weak labels (always predicts majority class). Failed domain transfer to physical label. |
| `real_prod_18` | `D01` | `PRESENT` | `PARTIAL` | 0.58 | ML_MODEL_ERROR | ML model error: feature pattern mapped to incorrect class probability distribution (predicted PARTIAL, expected PRESENT). |
| `real_prod_18` | `D04` | `PRESENT` | `CANNOT_DETERMINE` | 1.00 | FEATURE_DISTRIBUTION_SHIFT | Baseline single-class predictor trained on Flipkart weak labels (always predicts majority class). Failed domain transfer to physical label. |
| `real_prod_18` | `D06` | `PRESENT` | `ABSENT` | 0.85 | ML_MODEL_ERROR | High-confidence ML model misclassification (conf=0.85). Model overweighted text length/title cues over OCR evidence. |
| `real_prod_18` | `D07` | `PRESENT` | `CANNOT_DETERMINE` | 1.00 | FEATURE_DISTRIBUTION_SHIFT | Baseline single-class predictor trained on Flipkart weak labels (always predicts majority class). Failed domain transfer to physical label. |
| `real_prod_19` | `D01` | `PRESENT` | `PARTIAL` | 0.60 | ML_MODEL_ERROR | ML model error: feature pattern mapped to incorrect class probability distribution (predicted PARTIAL, expected PRESENT). |
| `real_prod_19` | `D04` | `PRESENT` | `CANNOT_DETERMINE` | 1.00 | FEATURE_DISTRIBUTION_SHIFT | Baseline single-class predictor trained on Flipkart weak labels (always predicts majority class). Failed domain transfer to physical label. |
| `real_prod_19` | `D06` | `PRESENT` | `ABSENT` | 0.87 | ML_MODEL_ERROR | High-confidence ML model misclassification (conf=0.87). Model overweighted text length/title cues over OCR evidence. |
| `real_prod_19` | `D07` | `PRESENT` | `CANNOT_DETERMINE` | 1.00 | FEATURE_DISTRIBUTION_SHIFT | Baseline single-class predictor trained on Flipkart weak labels (always predicts majority class). Failed domain transfer to physical label. |
| `real_prod_20` | `D01` | `PRESENT` | `PARTIAL` | 0.54 | ML_MODEL_ERROR | ML model error: feature pattern mapped to incorrect class probability distribution (predicted PARTIAL, expected PRESENT). |
| `real_prod_20` | `D04` | `PRESENT` | `CANNOT_DETERMINE` | 1.00 | FEATURE_DISTRIBUTION_SHIFT | Baseline single-class predictor trained on Flipkart weak labels (always predicts majority class). Failed domain transfer to physical label. |
| `real_prod_20` | `D06` | `PRESENT` | `ABSENT` | 0.92 | ML_MODEL_ERROR | High-confidence ML model misclassification (conf=0.92). Model overweighted text length/title cues over OCR evidence. |
| `real_prod_20` | `D07` | `PRESENT` | `CANNOT_DETERMINE` | 1.00 | FEATURE_DISTRIBUTION_SHIFT | Baseline single-class predictor trained on Flipkart weak labels (always predicts majority class). Failed domain transfer to physical label. |

## 4. Remediation Roadmap for Phase 10

1. **Eliminate Baseline Predictors**: Replace single-class majority baselines on D04 and D07 with multi-class supervised models trained directly on packaging OCR transcripts.
2. **Multimodal Feature Fusion**: Concatenate OCR text embedding vectors with image structural layout features.
3. **Confidence Calibration**: Apply Platt scaling or isotonic regression to align predicted probabilities with actual empirical accuracies.
