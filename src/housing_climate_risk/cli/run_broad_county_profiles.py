"""Run broad county profile search with PCA + Gaussian mixtures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from housing_climate_risk.data_sources.raw import load_profile_inputs
from housing_climate_risk.modeling.county_clustering.broad_profiles import run_broad_profile_search
from housing_climate_risk.paths import DATA_DIR, ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description="Run broad county profile GMM search.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DATA_DIR / "county_processed_data.feather",
        help="Processed county feather file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output" / "county_clustering" / "broad_profiles",
        help="Directory for broad profile artifacts.",
    )
    args = parser.parse_args()

    counties = pd.read_feather(args.data_path)
    result = run_broad_profile_search(
        counties=counties,
        output_dir=args.output_dir,
        profile_inputs=load_profile_inputs(),
    )
    scores = result["scores"]
    labels = result["labels"]
    summary = {
        "best_k": result["best_k"],
        "score_table": str(result["paths"]["scores"]),
        "labels": str(result["paths"]["labels"]),
        "profiles": str(result["paths"]["profiles"]),
        "candidate_scores": scores[
            [
                "k",
                "passes_balance_check",
                "min_cluster_size",
                "max_cluster_share",
                "silhouette_score",
                "davies_bouldin_index",
                "mean_assignment_confidence",
                "low_confidence_under_0_60_rate",
                "usable_rank_score",
            ]
        ].to_dict(orient="records"),
        "label_rows": int(len(labels)),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
