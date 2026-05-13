"""Stability checks for county clustering experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from housing_climate_risk.modeling.county_clustering.experiments import ClusteringExperiment
from housing_climate_risk.modeling.county_clustering.features import ID_COLUMNS
from housing_climate_risk.modeling.county_clustering.runners import build_model, fit_predict_labels


def estimate_stability(
    experiments: list[ClusteringExperiment],
    feature_sets: dict[str, pd.DataFrame],
    output_dir: Path | str,
    n_runs: int = 3,
    sample_fraction: float = 0.85,
) -> pd.DataFrame:
    """Estimate repeatability of each experiment on repeated county samples.

    Each run draws one county sample and fits the same experiment twice with
    different random states.  The labels are compared on the shared sample using
    adjusted Rand index and normalized mutual information.  Deterministic models
    such as agglomerative clustering and HDBSCAN should score near 1.0 for this
    check; lower scores for KMeans or GMM indicate sensitivity to initialization.
    """

    rows = []
    for experiment in experiments:
        feature_df = feature_sets[experiment.feature_set]
        for run_index in range(n_runs):
            sample = feature_df.sample(frac=sample_fraction, random_state=run_index).reset_index(drop=True)
            labels_a = _fit_labels_in_memory(experiment, sample, random_state=run_index)
            labels_b = _fit_labels_in_memory(experiment, sample, random_state=run_index + 10_000)
            rows.append(
                {
                    "experiment": experiment.slug,
                    "feature_set": experiment.feature_set,
                    "model_name": experiment.model_name,
                    "run_index": run_index,
                    "sample_fraction": sample_fraction,
                    "adjusted_rand_index": adjusted_rand_score(labels_a, labels_b),
                    "normalized_mutual_info": normalized_mutual_info_score(labels_a, labels_b),
                }
            )

    stability_df = pd.DataFrame(rows)
    summary_df = (
        stability_df.groupby(["experiment", "feature_set", "model_name"], as_index=False)
        .agg(
            stability_runs=("run_index", "size"),
            adjusted_rand_mean=("adjusted_rand_index", "mean"),
            adjusted_rand_min=("adjusted_rand_index", "min"),
            normalized_mutual_info_mean=("normalized_mutual_info", "mean"),
            normalized_mutual_info_min=("normalized_mutual_info", "min"),
        )
        .sort_values(["adjusted_rand_mean", "normalized_mutual_info_mean"], ascending=False)
    )

    output_dir = Path(output_dir)
    scores_dir = output_dir / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    stability_df.to_csv(scores_dir / "stability_runs.csv", index=False)
    summary_df.to_csv(scores_dir / "stability_scores.csv", index=False)
    return summary_df


def _fit_labels_in_memory(
    experiment: ClusteringExperiment,
    feature_df: pd.DataFrame,
    random_state: int,
) -> np.ndarray:
    identity_columns = [column for column in ID_COLUMNS if column in feature_df.columns]
    x_df = (
        feature_df.drop(columns=identity_columns, errors="ignore")
        .select_dtypes(include=[np.number])
        .replace([np.inf, -np.inf], np.nan)
        .dropna(axis=1, how="all")
    )
    preprocess = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", RobustScaler()),
        ]
    )
    x = preprocess.fit_transform(x_df)

    if experiment.reducer == "pca":
        reducer = PCA(n_components=0.90, random_state=random_state)
        x = reducer.fit_transform(x)
    elif experiment.reducer is not None:
        raise ValueError(f"Unknown reducer: {experiment.reducer}")

    model = build_model(experiment, random_state=random_state)
    labels, _ = fit_predict_labels(model, x, experiment.model_name)
    return labels

