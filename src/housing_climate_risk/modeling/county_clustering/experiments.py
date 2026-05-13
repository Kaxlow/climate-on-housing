"""Experiment definitions for the first county clustering run."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ClusteringExperiment:
    """A concrete model run over one engineered feature set."""

    feature_set: str
    reducer: str | None
    model_name: str
    model_params: dict[str, object] = field(default_factory=dict)

    @property
    def slug(self) -> str:
        """Stable filesystem-safe name for model artifacts."""

        reducer = self.reducer or "none"
        params = "_".join(f"{key}-{value}" for key, value in sorted(self.model_params.items()))
        return "__".join(part for part in [self.feature_set, reducer, self.model_name, params] if part)


def preferred_experiments(include_hdbscan: bool = True) -> list[ClusteringExperiment]:
    """Return the first-run experiment grid.

    The grid is intentionally small enough for quick iteration while covering
    the requested broad, climate/insurance, and housing trajectory views.
    """

    experiments: list[ClusteringExperiment] = []

    for feature_set in ["all_features_without_size_variables", "climate_insurance_only"]:
        for k in [5, 6, 8, 10]:
            experiments.extend(
                [
                    ClusteringExperiment(feature_set, "pca", "kmeans", {"n_clusters": k}),
                    ClusteringExperiment(feature_set, "pca", "gmm", {"n_components": k}),
                    ClusteringExperiment(feature_set, "pca", "agglomerative", {"n_clusters": k}),
                ]
            )
        if include_hdbscan:
            experiments.append(
                ClusteringExperiment(feature_set, "pca", "hdbscan", {"min_cluster_size": 30, "min_samples": 10})
            )

    for k in [5, 6, 8, 10]:
        experiments.extend(
            [
                ClusteringExperiment("housing_time_series_features", "pca", "kmeans", {"n_clusters": k}),
                ClusteringExperiment("housing_time_series_features", "pca", "gmm", {"n_components": k}),
            ]
        )

    return experiments

