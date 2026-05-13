"""Fit clustering experiments and write reusable artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from housing_climate_risk.modeling.county_clustering.experiments import ClusteringExperiment
from housing_climate_risk.modeling.county_clustering.features import ID_COLUMNS

try:
    from sklearn.cluster import HDBSCAN
except ImportError:  # pragma: no cover - depends on the installed sklearn version.
    HDBSCAN = None


def run_experiment(
    experiment: ClusteringExperiment,
    feature_df: pd.DataFrame,
    output_dir: Path | str,
    random_state: int = 42,
) -> dict[str, Any]:
    """Fit one experiment and persist labels, scores, and model state."""

    output_dir = Path(output_dir)
    for folder in ["labels", "models", "scores"]:
        (output_dir / folder).mkdir(parents=True, exist_ok=True)

    identity_columns = [column for column in ID_COLUMNS if column in feature_df.columns]
    x_df = _numeric_model_frame(feature_df, identity_columns)
    x_df = x_df.dropna(axis=1, how="all")
    if x_df.empty:
        raise ValueError(f"{experiment.feature_set} produced no numeric features.")

    preprocess = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", RobustScaler()),
        ]
    )
    x = preprocess.fit_transform(x_df)

    reducer = None
    x_model = x
    if experiment.reducer == "pca":
        reducer = PCA(n_components=0.90, random_state=random_state)
        x_model = reducer.fit_transform(x)
    elif experiment.reducer is not None:
        raise ValueError(f"Unknown reducer: {experiment.reducer}")

    model = build_model(experiment, random_state=random_state)
    labels, probabilities = fit_predict_labels(model, x_model, experiment.model_name)
    scores = score_clustering(x_model, labels)
    scores.update(
        {
            "experiment": experiment.slug,
            "feature_set": experiment.feature_set,
            "reducer": experiment.reducer or "none",
            "model_name": experiment.model_name,
            "model_params": json.dumps(experiment.model_params, sort_keys=True),
            "n_input_features": int(x_df.shape[1]),
            "n_model_features": int(x_model.shape[1]),
        }
    )

    labels_df = feature_df[identity_columns].copy()
    labels_df["experiment"] = experiment.slug
    labels_df["feature_set"] = experiment.feature_set
    labels_df["model_name"] = experiment.model_name
    labels_df["cluster"] = labels
    if probabilities is not None:
        labels_df["cluster_probability"] = probabilities

    labels_path = output_dir / "labels" / f"{experiment.slug}.parquet"
    model_path = output_dir / "models" / f"{experiment.slug}.joblib"
    scores_path = output_dir / "scores" / f"{experiment.slug}.json"

    labels_df.to_parquet(labels_path, index=False)
    with scores_path.open("w", encoding="utf-8") as file:
        json.dump(scores, file, indent=2)
    dump(
        {
            "experiment": experiment,
            "preprocess": preprocess,
            "reducer": reducer,
            "model": model,
            "feature_columns": x_df.columns.tolist(),
            "scores": scores,
        },
        model_path,
    )

    return {**scores, "labels_path": str(labels_path), "model_path": str(model_path)}


def build_model(experiment: ClusteringExperiment, random_state: int = 42) -> Any:
    """Instantiate a clustering model from an experiment definition."""

    params = dict(experiment.model_params)
    if experiment.model_name == "kmeans":
        return KMeans(random_state=random_state, n_init=20, **params)
    if experiment.model_name == "gmm":
        return GaussianMixture(random_state=random_state, **params)
    if experiment.model_name == "agglomerative":
        return AgglomerativeClustering(linkage="ward", **params)
    if experiment.model_name == "hdbscan":
        if HDBSCAN is None:
            raise RuntimeError("sklearn.cluster.HDBSCAN is unavailable in this scikit-learn installation.")
        return HDBSCAN(**params)
    raise ValueError(f"Unknown model name: {experiment.model_name}")


def fit_predict_labels(model: Any, x_model: np.ndarray, model_name: str) -> tuple[np.ndarray, np.ndarray | None]:
    """Fit a model and return labels plus optional soft-membership confidence."""

    if model_name == "gmm":
        labels = model.fit_predict(x_model)
        probabilities = model.predict_proba(x_model).max(axis=1)
        return labels.astype(int), probabilities.astype(float)

    labels = model.fit_predict(x_model)
    probabilities = getattr(model, "probabilities_", None)
    if probabilities is not None:
        probabilities = np.asarray(probabilities, dtype=float)
    return np.asarray(labels, dtype=int), probabilities


def score_clustering(x_model: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    """Compute cluster quality and balance metrics.

    Standard clustering metrics can reward a model that creates one very large
    cluster plus several one-county outlier clusters.  The balance diagnostics
    make that failure mode explicit so the score table can be filtered for
    groupings that are useful as county typologies.
    """

    labels = np.asarray(labels)
    valid = labels != -1
    valid_labels = labels[valid]
    unique = np.unique(valid_labels)
    valid_cluster_sizes = pd.Series(valid_labels).value_counts().sort_index()
    all_cluster_sizes = pd.Series(labels).value_counts().sort_index().to_dict()
    min_cluster_size = int(valid_cluster_sizes.min()) if not valid_cluster_sizes.empty else 0
    max_cluster_share = float(valid_cluster_sizes.max() / valid.sum()) if valid.sum() else np.nan
    singleton_clusters = int((valid_cluster_sizes == 1).sum()) if not valid_cluster_sizes.empty else 0
    small_clusters = int((valid_cluster_sizes < 30).sum()) if not valid_cluster_sizes.empty else 0
    scores: dict[str, Any] = {
        "n_rows": int(len(labels)),
        "n_clusters": int(len(unique)),
        "noise_rate": float((labels == -1).mean()),
        "min_cluster_size": min_cluster_size,
        "max_cluster_share": max_cluster_share,
        "singleton_clusters": singleton_clusters,
        "clusters_under_30_counties": small_clusters,
        "cluster_sizes": json.dumps({int(key): int(value) for key, value in all_cluster_sizes.items()}, sort_keys=True),
    }

    if len(unique) >= 2 and valid.sum() > len(unique):
        scores["silhouette_score"] = float(
            silhouette_score(x_model[valid], valid_labels, sample_size=min(10_000, valid.sum()), random_state=42)
        )
        scores["calinski_harabasz_score"] = float(calinski_harabasz_score(x_model[valid], valid_labels))
        scores["davies_bouldin_index"] = float(davies_bouldin_score(x_model[valid], valid_labels))
    else:
        scores["silhouette_score"] = np.nan
        scores["calinski_harabasz_score"] = np.nan
        scores["davies_bouldin_index"] = np.nan
    return scores


def run_experiments(
    experiments: list[ClusteringExperiment],
    feature_sets: dict[str, pd.DataFrame],
    output_dir: Path | str,
    random_state: int = 42,
) -> pd.DataFrame:
    """Run a list of experiments and return the combined score table."""

    scores = []
    for experiment in experiments:
        scores.append(
            run_experiment(
                experiment=experiment,
                feature_df=feature_sets[experiment.feature_set],
                output_dir=output_dir,
                random_state=random_state,
            )
        )
    score_df = pd.DataFrame(scores)
    score_df = add_score_ranks(score_df)
    scores_dir = Path(output_dir) / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    score_df.to_csv(scores_dir / "clustering_scores.csv", index=False)
    return score_df


def add_score_ranks(score_df: pd.DataFrame) -> pd.DataFrame:
    """Add numeric-quality and usability ranks for scanning candidate models."""

    score_df = score_df.copy()
    score_df["silhouette_rank"] = score_df["silhouette_score"].rank(ascending=False, method="min")
    score_df["calinski_harabasz_rank"] = score_df["calinski_harabasz_score"].rank(ascending=False, method="min")
    score_df["davies_bouldin_rank"] = score_df["davies_bouldin_index"].rank(ascending=True, method="min")
    score_df["combined_metric_rank"] = score_df[
        ["silhouette_rank", "calinski_harabasz_rank", "davies_bouldin_rank"]
    ].mean(axis=1)
    score_df["passes_balance_check"] = (
        (score_df["min_cluster_size"] >= 30)
        & (score_df["max_cluster_share"] <= 0.85)
        & (score_df["n_clusters"] >= 3)
    )
    score_df["balance_penalty"] = (
        score_df["singleton_clusters"] * 2
        + score_df["clusters_under_30_counties"]
        + (score_df["max_cluster_share"].fillna(1.0) * 5)
    )
    score_df["usable_metric_rank"] = score_df["combined_metric_rank"] + score_df["balance_penalty"]
    return score_df.sort_values(["usable_metric_rank", "combined_metric_rank"]).reset_index(drop=True)


def _numeric_model_frame(feature_df: pd.DataFrame, identity_columns: list[str]) -> pd.DataFrame:
    return (
        feature_df.drop(columns=identity_columns, errors="ignore")
        .select_dtypes(include=[np.number])
        .replace([np.inf, -np.inf], np.nan)
    )
