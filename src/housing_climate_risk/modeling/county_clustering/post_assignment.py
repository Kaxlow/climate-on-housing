"""Post-assign HDBSCAN noise counties and evaluate final labels.

HDBSCAN intentionally leaves ambiguous observations with label ``-1``.  For a
production county map, every county usually needs a group assignment, but the
post-assignment should preserve uncertainty.  This module assigns HDBSCAN noise
counties to the cluster most represented among their nearest clustered
neighbors, then saves confidence and quality diagnostics alongside the final
labels.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.neighbors import NearestNeighbors

from housing_climate_risk.modeling.county_clustering.features import ID_COLUMNS


def post_assign_hdbscan_outputs(
    feature_sets: dict[str, pd.DataFrame],
    output_dir: Path | str,
    k_neighbors: int = 7,
) -> pd.DataFrame:
    """Assign noise counties for every saved HDBSCAN output.

    Parameters
    ----------
    feature_sets:
        Feature matrices keyed by feature-set name.
    output_dir:
        Root county clustering output directory containing ``labels`` and
        ``models`` folders.  Final labels and evaluations are written under
        ``post_assignment`` and ``scores``.
    k_neighbors:
        Number of clustered neighbors used to vote on each noise county's final
        cluster.  If a HDBSCAN result has fewer clustered counties than this,
        the neighbor count is reduced automatically.
    """

    output_dir = Path(output_dir)
    labels_dir = output_dir / "labels"
    models_dir = output_dir / "models"
    final_labels_dir = output_dir / "post_assignment" / "labels"
    evaluation_dir = output_dir / "post_assignment" / "evaluation"
    final_labels_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    evaluation_rows = []
    drift_rows = []
    confidence_rows = []

    for labels_path in sorted(labels_dir.glob("*hdbscan*.parquet")):
        labels_df = pd.read_parquet(labels_path)
        if labels_df.empty:
            continue

        experiment = str(labels_df["experiment"].iloc[0])
        model_path = models_dir / f"{experiment}.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model artifact for {experiment}: {model_path}")

        model_artifact = load(model_path)
        feature_set = str(labels_df["feature_set"].iloc[0])
        feature_df = feature_sets[feature_set]
        x_model = transform_features_for_model(feature_df, model_artifact)

        final_labels_df, assignment_metrics = assign_noise_by_knn(
            labels_df=labels_df,
            x_model=x_model,
            k_neighbors=k_neighbors,
        )
        evaluation = evaluate_post_assignment(
            experiment=experiment,
            x_model=x_model,
            hdbscan_labels=final_labels_df["hdbscan_cluster"].to_numpy(),
            final_labels=final_labels_df["final_cluster"].to_numpy(),
            assignment_metrics=assignment_metrics,
        )
        drift_df = evaluate_profile_drift(
            experiment=experiment,
            feature_df=feature_df,
            final_labels_df=final_labels_df,
        )
        confidence_df = summarize_assignment_confidence(final_labels_df)

        final_labels_path = final_labels_dir / f"{experiment}__knn_assigned.parquet"
        final_labels_csv_path = final_labels_dir / f"{experiment}__knn_assigned.csv"
        final_labels_df.to_parquet(final_labels_path, index=False)
        final_labels_df.to_csv(final_labels_csv_path, index=False)

        evaluation["final_labels_path"] = str(final_labels_path)
        evaluation["final_labels_csv_path"] = str(final_labels_csv_path)
        evaluation_rows.append(evaluation)
        drift_rows.append(drift_df)
        confidence_rows.append(confidence_df)

    evaluation_df = pd.DataFrame(evaluation_rows)
    drift_all = pd.concat(drift_rows, ignore_index=True) if drift_rows else pd.DataFrame()
    confidence_all = pd.concat(confidence_rows, ignore_index=True) if confidence_rows else pd.DataFrame()

    evaluation_df.to_csv(evaluation_dir / "hdbscan_knn_assignment_evaluation.csv", index=False)
    drift_all.to_csv(evaluation_dir / "hdbscan_knn_profile_drift.csv", index=False)
    confidence_all.to_csv(evaluation_dir / "hdbscan_knn_assignment_confidence.csv", index=False)
    evaluation_df.to_csv(output_dir / "scores" / "hdbscan_knn_assignment_evaluation.csv", index=False)
    return evaluation_df


def transform_features_for_model(feature_df: pd.DataFrame, model_artifact: dict[str, Any]) -> np.ndarray:
    """Recreate the PCA feature space used by the saved clustering model."""

    feature_columns = model_artifact["feature_columns"]
    x_df = feature_df.reindex(columns=feature_columns).replace([np.inf, -np.inf], np.nan)
    x = model_artifact["preprocess"].transform(x_df)
    reducer = model_artifact.get("reducer")
    if reducer is not None:
        x = reducer.transform(x)
    return x


def assign_noise_by_knn(
    labels_df: pd.DataFrame,
    x_model: np.ndarray,
    k_neighbors: int = 7,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Assign label ``-1`` counties using majority vote among nearest members."""

    hdbscan_labels = labels_df["cluster"].to_numpy(dtype=int)
    clustered_mask = hdbscan_labels != -1
    noise_mask = hdbscan_labels == -1
    clustered_count = int(clustered_mask.sum())
    noise_count = int(noise_mask.sum())

    if clustered_count == 0:
        raise ValueError("Cannot post-assign noise because HDBSCAN produced no clustered counties.")

    effective_k = min(k_neighbors, clustered_count)
    final_labels = hdbscan_labels.copy()
    assignment_confidence = np.ones(len(labels_df), dtype=float)
    mean_neighbor_distance = np.zeros(len(labels_df), dtype=float)
    nearest_neighbor_distance = np.zeros(len(labels_df), dtype=float)
    assignment_method = np.full(len(labels_df), "hdbscan_core", dtype=object)

    if noise_count > 0:
        nn = NearestNeighbors(n_neighbors=effective_k)
        nn.fit(x_model[clustered_mask])
        distances, indices = nn.kneighbors(x_model[noise_mask])
        clustered_labels = hdbscan_labels[clustered_mask]
        neighbor_labels = clustered_labels[indices]

        assigned_labels = []
        confidences = []
        for row_labels in neighbor_labels:
            votes = pd.Series(row_labels).value_counts()
            assigned_labels.append(int(votes.index[0]))
            confidences.append(float(votes.iloc[0] / effective_k))

        final_labels[noise_mask] = np.asarray(assigned_labels, dtype=int)
        assignment_confidence[noise_mask] = np.asarray(confidences, dtype=float)
        mean_neighbor_distance[noise_mask] = distances.mean(axis=1)
        nearest_neighbor_distance[noise_mask] = distances.min(axis=1)
        assignment_method[noise_mask] = "knn_post_assigned"

    final_labels_df = labels_df.copy()
    final_labels_df = final_labels_df.rename(columns={"cluster": "hdbscan_cluster"})
    final_labels_df["final_cluster"] = final_labels
    final_labels_df["assignment_method"] = assignment_method
    final_labels_df["assignment_confidence"] = assignment_confidence
    final_labels_df["mean_neighbor_distance"] = mean_neighbor_distance
    final_labels_df["nearest_neighbor_distance"] = nearest_neighbor_distance
    final_labels_df["knn_neighbors"] = effective_k

    metrics = {
        "requested_knn_neighbors": float(k_neighbors),
        "effective_knn_neighbors": float(effective_k),
        "noise_count": float(noise_count),
        "core_count": float(clustered_count),
        "mean_assignment_confidence": float(assignment_confidence[noise_mask].mean()) if noise_count else np.nan,
        "median_assignment_confidence": float(np.median(assignment_confidence[noise_mask])) if noise_count else np.nan,
        "low_confidence_under_0_60_rate": float((assignment_confidence[noise_mask] < 0.60).mean()) if noise_count else np.nan,
        "very_low_confidence_under_0_50_rate": float((assignment_confidence[noise_mask] < 0.50).mean()) if noise_count else np.nan,
        "mean_neighbor_distance": float(mean_neighbor_distance[noise_mask].mean()) if noise_count else np.nan,
        "median_neighbor_distance": float(np.median(mean_neighbor_distance[noise_mask])) if noise_count else np.nan,
    }
    return final_labels_df, metrics


def evaluate_post_assignment(
    *,
    experiment: str,
    x_model: np.ndarray,
    hdbscan_labels: np.ndarray,
    final_labels: np.ndarray,
    assignment_metrics: dict[str, float],
) -> dict[str, Any]:
    """Evaluate core HDBSCAN labels and all-county final labels side by side."""

    core_mask = hdbscan_labels != -1
    noise_mask = hdbscan_labels == -1
    core_scores = _cluster_scores(x_model[core_mask], hdbscan_labels[core_mask])
    final_scores = _cluster_scores(x_model, final_labels)
    final_sizes = pd.Series(final_labels).value_counts().sort_index()

    evaluation = {
        "experiment": experiment,
        "n_rows": int(len(final_labels)),
        "hdbscan_n_clusters": int(len(np.unique(hdbscan_labels[core_mask]))),
        "hdbscan_noise_count": int(noise_mask.sum()),
        "hdbscan_noise_rate": float(noise_mask.mean()),
        "final_n_clusters": int(len(np.unique(final_labels))),
        "final_min_cluster_size": int(final_sizes.min()),
        "final_max_cluster_share": float(final_sizes.max() / len(final_labels)),
        "final_cluster_sizes": json.dumps({int(key): int(value) for key, value in final_sizes.items()}, sort_keys=True),
        **{f"core_{key}": value for key, value in core_scores.items()},
        **{f"final_{key}": value for key, value in final_scores.items()},
        **assignment_metrics,
    }
    evaluation["silhouette_drop"] = evaluation["core_silhouette_score"] - evaluation["final_silhouette_score"]
    return evaluation


def evaluate_profile_drift(
    *,
    experiment: str,
    feature_df: pd.DataFrame,
    final_labels_df: pd.DataFrame,
) -> pd.DataFrame:
    """Measure how much cluster medians change after KNN post-assignment."""

    identity_columns = [column for column in ID_COLUMNS if column in feature_df.columns]
    numeric_columns = [
        column
        for column in feature_df.select_dtypes(include=[np.number]).columns
        if column not in identity_columns
    ]
    merged = final_labels_df[["fips", "hdbscan_cluster", "final_cluster"]].merge(
        feature_df[["fips", *numeric_columns]], on="fips", how="left"
    )
    national_std = merged[numeric_columns].std().replace(0, np.nan)
    rows = []
    for cluster_id in sorted(merged["final_cluster"].dropna().unique()):
        core = merged.loc[merged["hdbscan_cluster"] == cluster_id, numeric_columns]
        final = merged.loc[merged["final_cluster"] == cluster_id, numeric_columns]
        if core.empty or final.empty:
            continue
        standardized_drift = ((final.median() - core.median()) / national_std).abs().dropna()
        largest = standardized_drift.sort_values(ascending=False).head(8)
        rows.append(
            {
                "experiment": experiment,
                "cluster": int(cluster_id),
                "core_count": int(len(core)),
                "final_count": int(len(final)),
                "added_count": int(len(final) - len(core)),
                "mean_abs_standardized_median_drift": float(standardized_drift.mean()),
                "max_abs_standardized_median_drift": float(standardized_drift.max()),
                "largest_drift_features": "; ".join(f"{feature} ({value:.2f} SD)" for feature, value in largest.items()),
            }
        )
    return pd.DataFrame(rows)


def summarize_assignment_confidence(final_labels_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize post-assignment confidence by final cluster."""

    assigned = final_labels_df.loc[final_labels_df["assignment_method"] == "knn_post_assigned"].copy()
    if assigned.empty:
        return pd.DataFrame()
    return (
        assigned.groupby(["experiment", "feature_set", "model_name", "final_cluster"], as_index=False)
        .agg(
            assigned_count=("fips", "size"),
            mean_assignment_confidence=("assignment_confidence", "mean"),
            median_assignment_confidence=("assignment_confidence", "median"),
            low_confidence_under_0_60_rate=("assignment_confidence", lambda values: float((values < 0.60).mean())),
            very_low_confidence_under_0_50_rate=("assignment_confidence", lambda values: float((values < 0.50).mean())),
            mean_neighbor_distance=("mean_neighbor_distance", "mean"),
            median_neighbor_distance=("mean_neighbor_distance", "median"),
        )
        .sort_values(["experiment", "final_cluster"])
    )


def _cluster_scores(x: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    unique = np.unique(labels)
    if len(unique) < 2 or len(labels) <= len(unique):
        return {
            "silhouette_score": np.nan,
            "calinski_harabasz_score": np.nan,
            "davies_bouldin_index": np.nan,
        }
    return {
        "silhouette_score": float(silhouette_score(x, labels, sample_size=min(10_000, len(labels)), random_state=42)),
        "calinski_harabasz_score": float(calinski_harabasz_score(x, labels)),
        "davies_bouldin_index": float(davies_bouldin_score(x, labels)),
    }
