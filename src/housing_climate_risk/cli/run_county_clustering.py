"""Run first-pass county clustering experiments.

Example:
    python -m housing_climate_risk.cli.run_county_clustering --preset preferred
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from housing_climate_risk.modeling.county_clustering.experiments import preferred_experiments
from housing_climate_risk.modeling.county_clustering.features import build_feature_sets
from housing_climate_risk.modeling.county_clustering.mapping import build_cluster_geojsons
from housing_climate_risk.modeling.county_clustering.post_assignment import post_assign_hdbscan_outputs
from housing_climate_risk.modeling.county_clustering.profiling import build_profiles_for_all_outputs
from housing_climate_risk.modeling.county_clustering.runners import HDBSCAN, run_experiments
from housing_climate_risk.modeling.county_clustering.stability import estimate_stability
from housing_climate_risk.paths import DATA_DIR, GEOGRAPHIC_DIR, ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description="Run county clustering experiments.")
    parser.add_argument("--preset", choices=["preferred"], default="preferred", help="Experiment preset to run.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DATA_DIR / "county_processed_data.feather",
        help="Processed county feather file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output" / "county_clustering",
        help="Directory for feature, label, score, profile, model, and map artifacts.",
    )
    parser.add_argument(
        "--skip-maps",
        action="store_true",
        help="Skip GeoJSON map outputs. Useful when only model score tables are needed.",
    )
    parser.add_argument(
        "--map-limit",
        type=int,
        default=None,
        help="Optional limit on number of label files converted to GeoJSON.",
    )
    parser.add_argument(
        "--stability-runs",
        type=int,
        default=3,
        help="Repeated fits per experiment for stability scoring. Use 0 to skip.",
    )
    parser.add_argument(
        "--skip-post-assignment",
        action="store_true",
        help="Skip KNN assignment of HDBSCAN noise counties.",
    )
    parser.add_argument(
        "--knn-neighbors",
        type=int,
        default=7,
        help="Neighbors used when assigning HDBSCAN noise counties.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    for folder in ["features", "labels", "scores", "profiles", "models", "maps"]:
        (output_dir / folder).mkdir(parents=True, exist_ok=True)

    counties = pd.read_feather(args.data_path)
    feature_sets = build_feature_sets(counties)
    for name, feature_df in feature_sets.items():
        feature_df.to_parquet(output_dir / "features" / f"{name}.parquet", index=False)

    experiments = preferred_experiments(include_hdbscan=HDBSCAN is not None)
    scores = run_experiments(experiments, feature_sets, output_dir)
    profiles = build_profiles_for_all_outputs(feature_sets, output_dir / "labels", output_dir / "profiles")
    stability = pd.DataFrame()
    if args.stability_runs > 0:
        stability = estimate_stability(
            experiments=experiments,
            feature_sets=feature_sets,
            output_dir=output_dir,
            n_runs=args.stability_runs,
        )
    post_assignment = pd.DataFrame()
    if not args.skip_post_assignment:
        post_assignment = post_assign_hdbscan_outputs(
            feature_sets=feature_sets,
            output_dir=output_dir,
            k_neighbors=args.knn_neighbors,
        )

    map_paths = []
    if not args.skip_maps:
        map_paths = build_cluster_geojsons(
            labels_dir=output_dir / "labels",
            boundaries_path=GEOGRAPHIC_DIR / "us_counties_boundaries_shapefile.json",
            output_dir=output_dir / "maps",
            limit=args.map_limit,
        )

    summary = {
        "preset": args.preset,
        "feature_sets": {name: {"rows": len(df), "columns": len(df.columns)} for name, df in feature_sets.items()},
        "experiments": len(experiments),
        "hdbscan_enabled": HDBSCAN is not None,
        "score_table": str(output_dir / "scores" / "clustering_scores.csv"),
        "profile_table": str(output_dir / "profiles" / "cluster_profiles.csv"),
        "stability_table": str(output_dir / "scores" / "stability_scores.csv") if args.stability_runs > 0 else None,
        "post_assignment_table": (
            str(output_dir / "post_assignment" / "evaluation" / "hdbscan_knn_assignment_evaluation.csv")
            if not args.skip_post_assignment
            else None
        ),
        "map_files": len(map_paths),
        "top_scores": scores.head(10)[
            ["experiment", "n_clusters", "noise_rate", "silhouette_score", "combined_metric_rank"]
        ].to_dict(orient="records"),
        "profile_rows": int(len(profiles)),
        "stability_rows": int(len(stability)),
        "post_assignment_rows": int(len(post_assignment)),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
