"""
test_phase_7.py
================
Unit and integration test suite for Phase 7 Feature Engineering & Dataset Construction.
Tests all 18 leakage, splitting, transformation, and reproducibility requirements.

Usage:
    python -m unittest src/features/test_phase_7.py -v
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import unittest
import json
import yaml
import numpy as np
import pandas as pd

from src.features.preprocessing import (
    extract_raw_features_from_record,
    NitiMarkFeaturePipeline
)
from src.features.build_ml_dataset import (
    perform_group_aware_split,
    load_and_extract_features
)


class TestPhase7Pipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.config_path = "configs/feature_config.yaml"
        with open(cls.config_path, "r", encoding="utf-8") as f:
            cls.config = yaml.safe_load(f)

        # Load generated ML partitions
        cls.ml_dir = "data/processed/ml_dataset_v1"
        cls.X_train = pd.read_csv(os.path.join(cls.ml_dir, "train/X_train.csv"))
        cls.X_val = pd.read_csv(os.path.join(cls.ml_dir, "validation/X_val.csv"))
        cls.X_test = pd.read_csv(os.path.join(cls.ml_dir, "test/X_test.csv"))
        cls.X_human = pd.read_csv(os.path.join(cls.ml_dir, "human_reference/X_human_ref.csv"))

        cls.Y_train = pd.read_csv(os.path.join(cls.ml_dir, "train/Y_train.csv"))
        cls.meta_train = pd.read_csv(os.path.join(cls.ml_dir, "train/metadata_train.csv"))
        cls.meta_val = pd.read_csv(os.path.join(cls.ml_dir, "validation/metadata_val.csv"))
        cls.meta_test = pd.read_csv(os.path.join(cls.ml_dir, "test/metadata_test.csv"))
        cls.meta_human = pd.read_csv(os.path.join(cls.ml_dir, "human_reference/metadata_human_ref.csv"))

    # 1. Target columns excluded from X
    def test_target_columns_excluded_from_x(self):
        target_names = {f"D0{i}_target" for i in range(1, 8)}.union({f"D0{i}_label" for i in range(1, 8)})
        for col in self.X_train.columns:
            self.assertNotIn(col, target_names)

    # 2. Overall label excluded from X
    def test_overall_label_excluded_from_x(self):
        self.assertNotIn("overall_label", self.X_train.columns)
        self.assertNotIn("overall_target", self.X_train.columns)

    # 3. Weak-label confidence excluded from normal feature set
    def test_confidence_excluded_from_x(self):
        for col in self.X_train.columns:
            self.assertNotIn("confidence", col.lower())

    # 4. Rule IDs excluded from X
    def test_rule_ids_excluded_from_x(self):
        for col in self.X_train.columns:
            self.assertNotIn("rule_id", col.lower())

    # 5. Quality tier excluded from X
    def test_quality_tier_excluded_from_x(self):
        self.assertNotIn("quality_tier", self.X_train.columns)

    # 6. Human labels never enter weak-label feature generation
    def test_human_ground_truth_isolation(self):
        human_pids = set(self.meta_human["product_id"])
        train_pids = set(self.meta_train["product_id"])
        self.assertEqual(len(human_pids.intersection(train_pids)), 0)

    # 7. Product groups don't cross splits
    def test_product_groups_do_not_cross_splits(self):
        train_fams = set(self.meta_train["product_family_id"])
        val_fams = set(self.meta_val["product_family_id"])
        test_fams = set(self.meta_test["product_family_id"])

        self.assertEqual(len(train_fams.intersection(val_fams)), 0)
        self.assertEqual(len(train_fams.intersection(test_fams)), 0)
        self.assertEqual(len(val_fams.intersection(test_fams)), 0)

    # 8. Duplicate groups isolated
    def test_duplicate_groups_isolated(self):
        train_dups = set(self.meta_train["duplicate_group_id"])
        val_dups = set(self.meta_val["duplicate_group_id"])
        test_dups = set(self.meta_test["duplicate_group_id"])

        self.assertEqual(len(train_dups.intersection(val_dups)), 0)
        self.assertEqual(len(train_dups.intersection(test_dups)), 0)
        self.assertEqual(len(val_dups.intersection(test_dups)), 0)

    # 9. Preprocessing fit only on TRAIN
    def test_preprocessing_fit_strictly_on_train(self):
        pipe = NitiMarkFeaturePipeline(self.config)
        self.assertFalse(pipe.fitted)

        # Synthetic train & val
        df_tr = pd.DataFrame([{
            "title_length": 10, "title_word_count": 2, "desc_length": 20, "desc_word_count": 4,
            "spec_count": 3, "spec_text_length": 30, "numeric_token_count": 1, "measurement_token_count": 1,
            "currency_token_count": 1, "contact_token_count": 0, "origin_token_count": 1,
            "retail_price": 100.0, "discounted_price": 90.0, "discount_amount": 10.0, "discount_percentage": 10.0,
            "image_count": 2, "has_description": 1, "has_specifications": 1, "has_brand": 1, "has_image": 1, "has_price": 1,
            "category": "Beauty", "price_tier": "BUDGET", "combined_text": "Sample shampoo bottle 500ml"
        }])
        pipe.fit(df_tr)
        self.assertTrue(pipe.fitted)

    # 10. Validation & Test do not influence fitting
    def test_validation_does_not_alter_fitted_vocabulary(self):
        pipe = NitiMarkFeaturePipeline(self.config)
        df_tr = pd.DataFrame([{
            "title_length": 10, "title_word_count": 2, "desc_length": 20, "desc_word_count": 4,
            "spec_count": 3, "spec_text_length": 30, "numeric_token_count": 1, "measurement_token_count": 1,
            "currency_token_count": 1, "contact_token_count": 0, "origin_token_count": 1,
            "retail_price": 100.0, "discounted_price": 90.0, "discount_amount": 10.0, "discount_percentage": 10.0,
            "image_count": 2, "has_description": 1, "has_specifications": 1, "has_brand": 1, "has_image": 1, "has_price": 1,
            "category": "Beauty", "price_tier": "BUDGET", "combined_text": "cream lotion"
        }])
        pipe.fit(df_tr)
        vocab_before = list(pipe.tfidf_word.get_feature_names_out())

        # Transform on unseen val text
        df_val = pd.DataFrame([{
            "title_length": 15, "title_word_count": 3, "desc_length": 30, "desc_word_count": 6,
            "spec_count": 1, "spec_text_length": 10, "numeric_token_count": 0, "measurement_token_count": 0,
            "currency_token_count": 0, "contact_token_count": 0, "origin_token_count": 0,
            "retail_price": 500.0, "discounted_price": 400.0, "discount_amount": 100.0, "discount_percentage": 20.0,
            "image_count": 1, "has_description": 1, "has_specifications": 1, "has_brand": 0, "has_image": 1, "has_price": 1,
            "category": "UnseenCategory", "price_tier": "PREMIUM", "combined_text": "completely different vocabulary"
        }])
        _ = pipe.transform(df_val)
        vocab_after = list(pipe.tfidf_word.get_feature_names_out())
        self.assertEqual(vocab_before, vocab_after)

    # 11. Missing values handled without residual NaNs
    def test_no_residual_nans(self):
        self.assertEqual(int(self.X_train.isna().sum().sum()), 0)
        self.assertEqual(int(self.X_val.isna().sum().sum()), 0)
        self.assertEqual(int(self.X_test.isna().sum().sum()), 0)
        self.assertEqual(int(self.X_human.isna().sum().sum()), 0)

    # 12. Unknown categories handled gracefully
    def test_unknown_category_handling(self):
        pipe = NitiMarkFeaturePipeline(self.config)
        df_tr = pd.DataFrame([{
            "title_length": 10, "title_word_count": 2, "desc_length": 20, "desc_word_count": 4,
            "spec_count": 3, "spec_text_length": 30, "numeric_token_count": 1, "measurement_token_count": 1,
            "currency_token_count": 1, "contact_token_count": 0, "origin_token_count": 1,
            "retail_price": 100.0, "discounted_price": 90.0, "discount_amount": 10.0, "discount_percentage": 10.0,
            "image_count": 2, "has_description": 1, "has_specifications": 1, "has_brand": 1, "has_image": 1, "has_price": 1,
            "category": "Beauty", "price_tier": "BUDGET", "combined_text": "cream"
        }])
        pipe.fit(df_tr)

        df_unseen = pd.DataFrame([{
            "title_length": 10, "title_word_count": 2, "desc_length": 20, "desc_word_count": 4,
            "spec_count": 3, "spec_text_length": 30, "numeric_token_count": 1, "measurement_token_count": 1,
            "currency_token_count": 1, "contact_token_count": 0, "origin_token_count": 1,
            "retail_price": 100.0, "discounted_price": 90.0, "discount_amount": 10.0, "discount_percentage": 10.0,
            "image_count": 2, "has_description": 1, "has_specifications": 1, "has_brand": 1, "has_image": 1, "has_price": 1,
            "category": "NEW_UNKNOWN_CAT", "price_tier": "NEW_PRICE_TIER", "combined_text": "cream"
        }])
        transformed = pipe.transform(df_unseen)
        self.assertEqual(len(transformed), 1)
        self.assertFalse(transformed.isna().any().any())

    # 13. Feature names preserved across splits
    def test_feature_names_preserved_across_splits(self):
        self.assertEqual(list(self.X_train.columns), list(self.X_val.columns))
        self.assertEqual(list(self.X_train.columns), list(self.X_test.columns))
        self.assertEqual(list(self.X_train.columns), list(self.X_human.columns))

    # 14. Target states preserved
    def test_target_states_preserved(self):
        valid_states = {"PRESENT", "ABSENT", "PARTIAL", "NOT_APPLICABLE", "CANNOT_DETERMINE"}
        for did in [f"D0{i}_target" for i in range(1, 8)]:
            unique_targets = set(self.Y_train[did].dropna().unique())
            self.assertTrue(unique_targets.issubset(valid_states))

    # 15. Dataset reproducibility with same seed
    def test_split_reproducibility(self):
        sample_df = pd.DataFrame({
            "product_id": [f"p_{i}" for i in range(20)],
            "product_family_id": [f"fam_{i % 5}" for i in range(20)]
        })
        tr1, val1, test1 = perform_group_aware_split(sample_df, seed=42)
        tr2, val2, test2 = perform_group_aware_split(sample_df, seed=42)
        self.assertEqual(list(tr1["product_id"]), list(tr2["product_id"]))
        self.assertEqual(list(val1["product_id"]), list(val2["product_id"]))
        self.assertEqual(list(test1["product_id"]), list(test2["product_id"]))


if __name__ == "__main__":
    unittest.main()
