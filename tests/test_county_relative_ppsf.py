from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from housing_climate_risk.modeling.county_relative_ppsf.data import (
    MODEL_FEATURE_PRIORITY,
)
from housing_climate_risk.modeling.county_relative_ppsf.train import (
    MODEL_GROUP_MEMBERS,
    MODEL_GROUP_ORDER,
    SAMPLE_INDICATOR,
    TrainingConfig,
    _available_features,
    _evaluate_group,
    _expand_group_interactions,
    _metrics,
    _prune_high_correlations,
)


class CountyRelativePpsfTests(unittest.TestCase):
    def test_requested_replacement_features_have_pruning_priority(self) -> None:
        self.assertEqual(
            MODEL_FEATURE_PRIORITY[:4],
            (
                "extreme_event_count",
                "homeowners_insurance_pct_income",
                "property_taxes_pct_income",
                "utilities_pct_income",
            ),
        )

    def test_high_and_very_high_share_one_model_group(self) -> None:
        self.assertEqual(
            MODEL_GROUP_ORDER,
            ["Very Low", "Low", "Medium", "High + Very High"],
        )
        self.assertEqual(
            MODEL_GROUP_MEMBERS["High + Very High"],
            ["High", "Very High"],
        )
        self.assertEqual(SAMPLE_INDICATOR, "is_very_high")

    def test_group_interaction_expansion_uses_sample_indicator(self) -> None:
        expanded = _expand_group_interactions(
            np.array(
                [
                    [2.0, 3.0, 0.0],
                    [4.0, 5.0, 1.0],
                ]
            )
        )
        np.testing.assert_allclose(
            expanded,
            np.array(
                [
                    [2.0, 3.0, 0.0, 0.0, 0.0],
                    [4.0, 5.0, 1.0, 4.0, 5.0],
                ]
            ),
        )

    def test_feature_filter_records_sparse_and_constant_features(self) -> None:
        feature_columns = ["constant", "sparse", "usable"]
        feature_meta = {
            feature: ("Test", feature.title()) for feature in feature_columns
        }
        frame = pd.DataFrame(
            {
                feature: np.linspace(0, 1, 10) for feature in feature_columns
            }
        )
        frame["constant"] = 1.0
        frame["sparse"] = [1.0, np.nan, np.nan, np.nan, np.nan] * 2

        selected, coverage = _available_features(
            frame, feature_columns, feature_meta, minimum_coverage=0.5
        )
        coverage_by_feature = {row["feature"]: row for row in coverage}

        self.assertNotIn("constant", selected)
        self.assertEqual(
            coverage_by_feature["constant"]["exclusion_reason"],
            "constant_or_single_value",
        )
        self.assertNotIn("sparse", selected)
        self.assertEqual(
            coverage_by_feature["sparse"]["exclusion_reason"],
            "insufficient_non_null_values",
        )

    def test_high_positive_and_negative_correlations_are_pruned(self) -> None:
        frame = pd.DataFrame(
            {
                "first": np.arange(20, dtype=float),
                "positive_duplicate": np.arange(20, dtype=float) * 2,
                "negative_duplicate": -np.arange(20, dtype=float),
            }
        )
        selected, exclusions = _prune_high_correlations(
            frame,
            ["first", "positive_duplicate", "negative_duplicate"],
            threshold=0.85,
        )
        self.assertEqual(selected, ["first"])
        self.assertIn("positive_duplicate", exclusions)
        self.assertIn("negative_duplicate", exclusions)

    def test_metrics_are_in_target_units(self) -> None:
        result = _metrics(
            np.array([-0.02, 0.00, 0.02]),
            np.array([-0.01, 0.00, 0.01]),
        )
        self.assertAlmostEqual(result["mae"], 0.0066666667)
        self.assertAlmostEqual(result["spearman"], 1.0)

    def test_both_model_families_produce_out_of_fold_predictions(self) -> None:
        rng = np.random.default_rng(42)
        row_count = 30
        frame = pd.DataFrame(
            {
                "fips": [f"{index:05d}" for index in range(row_count)],
                "relative_median_ppsf_yoy": rng.normal(0, 0.02, row_count),
            }
        )
        features = ["feature_a", "feature_b", "feature_c", "feature_d"]
        for index, feature in enumerate(features):
            frame[feature] = (
                frame["relative_median_ppsf_yoy"] * (index + 1)
                + rng.normal(0, 0.01, row_count)
            )
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TrainingConfig(
                output_dir=Path(temp_dir),
                outer_repeats=1,
                max_outer_splits=3,
                inner_splits=2,
                gradient_search_iterations=1,
                n_jobs=1,
            )
            folds, predictions = _evaluate_group(
                frame,
                features,
                "Low",
                config,
            )

        self.assertEqual(
            set(folds["model"]),
            {"median_baseline", "elastic_net", "gradient_boosted_trees"},
        )
        counts = predictions.groupby("model")["fips"].nunique().to_dict()
        self.assertEqual(counts["elastic_net"], row_count)
        self.assertEqual(counts["gradient_boosted_trees"], row_count)


if __name__ == "__main__":
    unittest.main()
