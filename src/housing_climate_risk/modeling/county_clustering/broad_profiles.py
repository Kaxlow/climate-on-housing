"""Broad county profiles with PCA + Gaussian mixtures.

The HDBSCAN pass is useful for dense subtypes, but it creates many small
profiles. This module builds a compressed feature view and searches Gaussian
mixture models with k=8..12 to produce broad, all-county profiles with soft
membership fields.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer

from housing_climate_risk.modeling.county_clustering.features import ID_COLUMNS, build_all_features
from housing_climate_risk.modeling.county_profiles import FEATURE_COLUMNS as ECONOMIC_POPULATION_COLUMNS
from housing_climate_risk.modeling.county_profiles import build_county_year_panel


K_VALUES = range(8, 13)
PCA_COMPONENTS = 12
BROAD_K12_LABELS = {
    0: "Coastal flood, hurricane, high-tax insurance-pressure counties",
    1: "Western wildfire and mountain-hazard insurance-pressure counties",
    2: "Temperature-variable severe-storm counties",
    3: "Warm severe-storm casualty counties",
    4: "Tornado and storm-injury temperature-variable counties",
    5: "Western wildfire and warm mountain-hazard counties",
    6: "High premium-growth insurance-pressure counties",
    7: "Volatile housing-market and temperature-variable counties",
    8: "High-premium, high-tax housing-market counties",
    9: "High-casualty severe-storm counties",
    10: "Mountain, landslide, and inland-flood low-premium counties",
    11: "Ice-storm and cold severe-storm casualty counties",
}
BROAD_K12_RATIONALES = {
    0: "High coastal flooding and hurricane risk, high storm-injury signal, elevated premiums/nonrenewal and property-tax percentiles.",
    1: "High wildfire, avalanche, volcanic/mountain hazard, warm-temperature, and premium-growth signals; low tornado/hurricane/wind exposure.",
    2: "High temperature volatility with severe-storm/winter/cold signals; lower wildfire and warmer baseline temperatures.",
    3: "High storm death/injury percentiles with warm stable temperatures and lower property-tax percentiles.",
    4: "High storm injury and tornado signal with temperature variability.",
    5: "High wildfire, volcanic, avalanche/mountain hazard, and warm-temperature signals with low cold-wave/hurricane exposure.",
    6: "High premium level/growth percentiles with low nonrenewal-rate and low major hazard exposure.",
    7: "High sale-to-list and sold-above-list volatility plus temperature variability and longer days-on-market.",
    8: "High premium growth, effective tax, and housing-market pressure with lower hurricane/earthquake/flood/lightning risk.",
    9: "Very high storm death/injury percentiles and event-count percentile, with otherwise moderate housing and tax signals.",
    10: "High avalanche, landslide, inland-flood, and storm-injury signals with low insurance premium levels.",
    11: "High storm death percentiles plus ice-storm/winter-weather/cold-wave signals.",
}
BROAD_K8_LABELS = {
    0: "Transfer-reliant, temperature-variable severe-storm counties",
    1: "High-tax northern housing-market storm-risk counties",
    2: "Lower-tax severe-storm casualty counties",
    3: "Older wildfire and mountain-hazard housing-market counties",
    4: "High-premium, high-tax low-hazard counties",
    5: "Cold-region severe-storm casualty counties",
    6: "Lower-wage tornado and storm-injury counties",
    7: "High-risk inland-flood and storm-injury counties",
}
BROAD_K8_RATIONALES = {
    0: "High temperature volatility and transfer-receipts share, with severe-storm exposure and cooler/lower-wildfire profile.",
    1: "High storm-injury, effective tax, median property tax, wage, and per-capita-income signals; examples skew northern and higher-tax.",
    2: "Very high storm death/injury percentiles with lower property-tax and lower White/proprietors-income-share signals.",
    3: "High senior share, wildfire/mountain-hazard exposure, longer days on market, and housing-market volatility.",
    4: "High insurance premium growth and tax pressure with comparatively low hurricane, earthquake, lightning, and inland-flood risk.",
    5: "High storm death percentiles plus ice-storm/cold-region signals and lower premium growth.",
    6: "High storm-injury and tornado signals with lower average weekly wage and cooler temperature profile.",
    7: "High NRI risk, inland-flood risk, storm-injury percentiles, and lower transfer-receipts/senior-share profile.",
}


def run_broad_profile_search(
    counties: pd.DataFrame,
    output_dir: Path | str,
    profile_inputs: dict[str, pd.DataFrame] | None = None,
    k_values: range | list[int] = K_VALUES,
    random_state: int = 42,
) -> dict[str, Any]:
    """Run k=8..12 PCA+GMM candidates and save the best broad profile result."""

    output_dir = Path(output_dir)
    for folder in ["features", "labels", "models", "scores", "profiles"]:
        (output_dir / folder).mkdir(parents=True, exist_ok=True)

    feature_df = build_broad_profile_features(counties, profile_inputs=profile_inputs)
    feature_df.to_parquet(output_dir / "features" / "broad_profile_features.parquet", index=False)

    identity_columns = [column for column in ID_COLUMNS if column in feature_df.columns]
    numeric_columns = [
        column for column in feature_df.select_dtypes(include=[np.number]).columns if column not in identity_columns
    ]
    x_df = feature_df[numeric_columns].replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
    x_df = winsorize_frame(x_df)

    preprocess = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "quantile",
                QuantileTransformer(
                    n_quantiles=min(1000, len(x_df)),
                    output_distribution="normal",
                    random_state=random_state,
                    subsample=None,
                ),
            ),
        ]
    )
    x_scaled = preprocess.fit_transform(x_df)
    x_scaled = apply_feature_family_weights(x_scaled, x_df.columns)
    pca = PCA(n_components=min(PCA_COMPONENTS, x_scaled.shape[1]), random_state=random_state)
    x_model = pca.fit_transform(x_scaled)

    candidate_rows = []
    candidate_artifacts = {}
    for k in k_values:
        model = GaussianMixture(
            n_components=int(k),
            covariance_type="diag",
            reg_covar=1e-3,
            n_init=20,
            random_state=random_state,
        )
        labels = model.fit_predict(x_model)
        probabilities = model.predict_proba(x_model)
        candidate = evaluate_gmm_candidate(x_model, labels, probabilities)
        candidate.update({"k": int(k), "model_name": "gmm", "model_params": json.dumps(model.get_params(), default=str)})
        candidate_rows.append(candidate)
        candidate_artifacts[int(k)] = {"model": model, "labels": labels, "probabilities": probabilities}

    score_df = add_broad_profile_ranks(pd.DataFrame(candidate_rows))
    score_df.to_csv(output_dir / "scores" / "broad_profile_gmm_scores.csv", index=False)

    best_row = score_df.iloc[0]
    best_k = int(best_row["k"])
    best = candidate_artifacts[best_k]
    profile_df, lift_df, plain_labels = build_broad_cluster_profiles(feature_df, best["labels"], best["probabilities"])
    curated_labels = BROAD_K8_LABELS if best_k == 8 else BROAD_K12_LABELS if best_k == 12 else None
    curated_rationales = BROAD_K8_RATIONALES if best_k == 8 else BROAD_K12_RATIONALES if best_k == 12 else None
    if curated_labels is not None and set(plain_labels) == set(curated_labels):
        plain_labels = curated_labels
        profile_df["plain_label"] = profile_df["final_cluster"].map(plain_labels)
        profile_df["label_rationale"] = profile_df["final_cluster"].map(curated_rationales)
        lift_df["plain_label"] = lift_df["final_cluster"].map(plain_labels)
    labels_df = build_broad_label_table(feature_df, best["labels"], best["probabilities"], plain_labels)

    labels_df.to_parquet(output_dir / "labels" / "broad_profiles_gmm_best.parquet", index=False)
    labels_df.to_csv(output_dir / "labels" / "broad_profiles_gmm_best.csv", index=False)
    profile_df.to_csv(output_dir / "profiles" / "broad_profile_cluster_profiles.csv", index=False)
    lift_df.to_csv(output_dir / "profiles" / "broad_profile_feature_lifts.csv", index=False)
    dump(
        {
            "preprocess": preprocess,
            "pca": pca,
            "model": best["model"],
            "feature_columns": x_df.columns.tolist(),
            "feature_family_weights": feature_family_weights(x_df.columns),
            "best_k": best_k,
            "scores": best_row.to_dict(),
            "plain_labels": plain_labels,
        },
        output_dir / "models" / "broad_profiles_gmm_best.joblib",
    )

    return {
        "best_k": best_k,
        "scores": score_df,
        "labels": labels_df,
        "profiles": profile_df,
        "paths": {
            "scores": output_dir / "scores" / "broad_profile_gmm_scores.csv",
            "labels": output_dir / "labels" / "broad_profiles_gmm_best.csv",
            "profiles": output_dir / "profiles" / "broad_profile_cluster_profiles.csv",
        },
    }


def build_broad_profile_features(
    counties: pd.DataFrame,
    profile_inputs: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Build a compressed matrix that favors scores, rates, percentiles, and trends."""

    all_features = build_all_features(counties)
    keep_columns = [column for column in ID_COLUMNS if column in all_features.columns]
    numeric_columns = list(all_features.select_dtypes(include=[np.number]).columns)
    selected = [column for column in numeric_columns if _keep_broad_feature(column)]
    out = all_features[keep_columns + selected].dropna(axis=1, how="all").copy()
    if profile_inputs is not None:
        out = out.merge(build_economic_population_features(profile_inputs), on="fips", how="left")
    return out


def build_economic_population_features(profile_inputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build normalized demographic/economic county attributes.

    This reuses the existing county profile panel so broad clusters can reflect
    economic position, age/race composition, migration, and labor-market context
    in addition to climate, housing, and insurance attributes.
    """

    max_year = int(pd.to_numeric(profile_inputs["population_estimates_df"]["Year"], errors="coerce").max())
    target_years = list(range(max_year - 9, max_year + 1))
    panel = build_county_year_panel(**profile_inputs, target_years=target_years)
    panel = panel.replace([np.inf, -np.inf], np.nan)

    average = panel.groupby("fips", as_index=False)[ECONOMIC_POPULATION_COLUMNS].mean()
    latest = (
        panel.sort_values(["fips", "Year"])
        .groupby("fips", as_index=False)
        .tail(1)[["fips", *ECONOMIC_POPULATION_COLUMNS]]
        .rename(columns={column: f"{column}_latest" for column in ECONOMIC_POPULATION_COLUMNS})
    )
    trends = panel.groupby("fips").apply(_economic_population_trends, include_groups=False).reset_index()

    average = average.rename(columns={column: f"econ_pop_{column}_avg" for column in ECONOMIC_POPULATION_COLUMNS})
    latest = latest.rename(columns={column: f"econ_pop_{column}" for column in latest.columns if column != "fips"})
    trends = trends.rename(columns={column: f"econ_pop_{column}" for column in trends.columns if column != "fips"})
    return average.merge(latest, on="fips", how="left").merge(trends, on="fips", how="left")


def _economic_population_trends(group: pd.DataFrame) -> pd.Series:
    trend_columns = [
        "log_population",
        "per_capita_income",
        "Average Weekly Wage",
        "employment_rate_proxy",
        "domestic_migration_rate",
        "senior_share",
        "hispanic_share",
    ]
    out = {}
    for column in trend_columns:
        if column in group.columns:
            out[f"{column}_slope"] = _linear_slope(group[column].to_numpy(dtype=float))
    return pd.Series(out)


def apply_feature_family_weights(x_scaled: np.ndarray, columns: pd.Index) -> np.ndarray:
    """Weight feature families so no block dominates by column count."""

    weights = feature_family_weights(columns)
    return x_scaled * np.asarray([weights[column] for column in columns], dtype=float)


def feature_family_weights(columns: pd.Index) -> dict[str, float]:
    groups = pd.Series([_feature_family(column) for column in columns], index=columns)
    counts = groups.value_counts()
    target = float(np.median(counts))
    return {column: float(np.sqrt(target / counts[groups[column]])) for column in columns}


def _feature_family(column: str) -> str:
    if column.startswith("econ_pop_"):
        return "economic_population"
    if column.startswith("housing_"):
        return "housing"
    if column.startswith("insurance_") or column.startswith("property_tax_"):
        return "insurance_tax"
    if column.startswith("nri_"):
        return "nri_hazards"
    if column.startswith("storm_") or column.startswith("fema_") or column.startswith("nfip_"):
        return "disaster_claims"
    if column.startswith("temperature_"):
        return "temperature"
    return "other"


def winsorize_frame(x_df: pd.DataFrame, lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    """Clip feature extremes before quantile scaling."""

    clipped = x_df.copy()
    for column in clipped.columns:
        values = clipped[column]
        if values.notna().sum() < 20:
            continue
        lo = values.quantile(lower)
        hi = values.quantile(upper)
        if pd.notna(lo) and pd.notna(hi) and lo < hi:
            clipped[column] = values.clip(lo, hi)
    return clipped


def _linear_slope(values: np.ndarray) -> float:
    mask = np.isfinite(values)
    if mask.sum() < 2:
        return np.nan
    x = np.arange(values.size, dtype=float)[mask]
    y = values[mask]
    return float(np.polyfit(x, y, 1)[0])


def evaluate_gmm_candidate(x_model: np.ndarray, labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    """Score one GMM candidate for separation, balance, and soft certainty."""

    sizes = pd.Series(labels).value_counts().sort_index()
    sorted_probs = np.sort(probabilities, axis=1)
    confidence = sorted_probs[:, -1]
    margin = sorted_probs[:, -1] - sorted_probs[:, -2]
    return {
        "n_clusters": int(len(sizes)),
        "min_cluster_size": int(sizes.min()),
        "max_cluster_share": float(sizes.max() / len(labels)),
        "cluster_size_cv": float(sizes.std(ddof=0) / sizes.mean()),
        "mean_assignment_confidence": float(confidence.mean()),
        "median_assignment_confidence": float(np.median(confidence)),
        "low_confidence_under_0_60_rate": float((confidence < 0.60).mean()),
        "mean_assignment_margin": float(margin.mean()),
        "median_assignment_margin": float(np.median(margin)),
        "silhouette_score": float(silhouette_score(x_model, labels, sample_size=min(10_000, len(labels)), random_state=42)),
        "calinski_harabasz_score": float(calinski_harabasz_score(x_model, labels)),
        "davies_bouldin_index": float(davies_bouldin_score(x_model, labels)),
        "cluster_sizes": json.dumps({int(key): int(value) for key, value in sizes.items()}, sort_keys=True),
    }


def add_broad_profile_ranks(score_df: pd.DataFrame) -> pd.DataFrame:
    """Rank candidates by usable broad-profile quality."""

    score_df = score_df.copy()
    score_df["passes_balance_check"] = (
        (score_df["min_cluster_size"] >= 100)
        & (score_df["max_cluster_share"] <= 0.30)
        & (score_df["low_confidence_under_0_60_rate"] <= 0.35)
    )
    score_df["silhouette_rank"] = score_df["silhouette_score"].rank(ascending=False, method="min")
    score_df["davies_bouldin_rank"] = score_df["davies_bouldin_index"].rank(ascending=True, method="min")
    score_df["confidence_rank"] = score_df["mean_assignment_confidence"].rank(ascending=False, method="min")
    score_df["balance_penalty"] = (
        (score_df["max_cluster_share"] * 10)
        + score_df["cluster_size_cv"]
        + np.maximum(0, 100 - score_df["min_cluster_size"]) / 25
        + np.maximum(0, score_df["low_confidence_under_0_60_rate"] - 0.25) * 5
    )
    score_df["usable_rank_score"] = (
        score_df[["silhouette_rank", "davies_bouldin_rank", "confidence_rank"]].mean(axis=1)
        + score_df["balance_penalty"]
    )
    return score_df.sort_values(["passes_balance_check", "usable_rank_score"], ascending=[False, True]).reset_index(drop=True)


def build_broad_label_table(
    feature_df: pd.DataFrame,
    labels: np.ndarray,
    probabilities: np.ndarray,
    plain_labels: dict[int, str],
) -> pd.DataFrame:
    """Build final county assignments with soft GMM membership fields."""

    sorted_prob_indices = np.argsort(probabilities, axis=1)
    second_best = sorted_prob_indices[:, -2]
    confidence = probabilities.max(axis=1)
    margin = confidence - probabilities[np.arange(len(probabilities)), second_best]

    id_columns = [column for column in ID_COLUMNS if column in feature_df.columns]
    out = feature_df[id_columns].copy()
    out["final_cluster"] = labels.astype(int)
    out["plain_label"] = out["final_cluster"].map(plain_labels)
    out["assignment_confidence"] = confidence
    out["assignment_margin"] = margin
    out["second_best_cluster"] = second_best.astype(int)
    out["second_best_label"] = out["second_best_cluster"].map(plain_labels)
    return out


def build_broad_cluster_profiles(
    feature_df: pd.DataFrame,
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, str]]:
    """Summarize clusters and derive plain labels from strongest feature lifts."""

    numeric_columns = [column for column in feature_df.select_dtypes(include=[np.number]).columns if column not in ID_COLUMNS]
    work = feature_df[["fips", "county_name", "state", *numeric_columns]].copy()
    work["final_cluster"] = labels
    work["assignment_confidence"] = probabilities.max(axis=1)
    national_median = work[numeric_columns].median()
    national_std = work[numeric_columns].std().replace(0, np.nan)
    cluster_median = work.groupby("final_cluster")[numeric_columns].median()
    lifts = (cluster_median - national_median) / national_std

    plain_labels = {int(cluster): derive_plain_label(lifts.loc[cluster]) for cluster in lifts.index}
    profile_rows = []
    lift_rows = []
    for cluster, zscores in lifts.iterrows():
        cluster_rows = work.loc[work["final_cluster"] == cluster]
        high = zscores.dropna().sort_values(ascending=False).head(10)
        low = zscores.dropna().sort_values(ascending=True).head(8)
        profile_rows.append(
            {
                "final_cluster": int(cluster),
                "plain_label": plain_labels[int(cluster)],
                "county_count": int(len(cluster_rows)),
                "mean_assignment_confidence": float(cluster_rows["assignment_confidence"].mean()),
                "top_high_features": "; ".join(f"{feature} ({value:.2f} SD)" for feature, value in high.items()),
                "top_low_features": "; ".join(f"{feature} ({value:.2f} SD)" for feature, value in low.items()),
                "example_counties": ", ".join(
                    cluster_rows[["county_name", "state"]]
                    .dropna()
                    .head(10)
                    .apply(lambda row: f"{row['county_name']}, {row['state']}", axis=1)
                ),
            }
        )
        for feature, value in zscores.dropna().items():
            lift_rows.append(
                {
                    "final_cluster": int(cluster),
                    "plain_label": plain_labels[int(cluster)],
                    "feature": feature,
                    "standardized_median_lift": float(value),
                }
            )
    return pd.DataFrame(profile_rows), pd.DataFrame(lift_rows), plain_labels


def derive_plain_label(zscores: pd.Series) -> str:
    """Generate a concise label from the strongest broad feature families."""

    top_features = zscores.dropna().sort_values(ascending=False).head(12).index.tolist()
    text = " ".join(top_features)
    parts = []
    if "hurricane" in text or "coastal_flooding" in text:
        parts.append("Coastal hurricane/flood")
    if "wildfire" in text:
        parts.append("Wildfire")
    if "flood" in text and not any("flood" in part.lower() for part in parts):
        parts.append("Flood-prone")
    if "hail" in text or "tornado" in text or "strong_wind" in text or "lightning" in text:
        parts.append("Severe-storm")
    if "winter" in text or "ice_storm" in text or "cold_wave" in text:
        parts.append("Winter/cold")
    if "heat_wave" in text or "temperature_tmax" in text or "temperature_tmin" in text:
        parts.append("temperature-variable" if "volatility" in text else "warm-stable")
    if "insurance_premium" in text or "nonrenewal" in text:
        parts.append("insurance-pressure")
    if "property_tax" in text:
        parts.append("high-tax")
    if "housing" in text:
        parts.append("housing-market")
    if "nfip" in text:
        parts.append("NFIP-claim")
    if "storm_total_injuries" in text or "storm_total_deaths" in text:
        parts.append("storm-casualty")

    deduped = []
    for part in parts:
        if part not in deduped:
            deduped.append(part)
    if not deduped:
        return "Mixed moderate-risk counties"
    return " / ".join(deduped[:4]) + " counties"


def _keep_broad_feature(column: str) -> bool:
    lowered = column.lower()
    if any(token in lowered for token in ["start_year", "end_year", "latest_year", "_year"]):
        return False
    if lowered.endswith("_missing_rate"):
        return False
    if any(token in lowered for token in ["coverage", "replacementcost", "propertyvalue", "buildvalue", "agrivalue"]):
        return False
    if any(token in lowered for token in ["policycount", "num_policies", "population", "area"]):
        return False
    if lowered.startswith("nfip_") and "per_policy" not in lowered:
        return False
    if lowered.startswith("storm_"):
        return "percentile" in lowered or "event_type" in lowered
    if lowered.startswith("temperature_"):
        return (
            "average_temp_f_mean" in lowered
            or "trend_slope_temp_f_mean" in lowered
            or lowered.endswith("_volatility")
        )
    if lowered.startswith("housing_"):
        if any(size_metric in lowered for size_metric in ["homes_sold", "new_listings", "inventory"]):
            return any(suffix in lowered for suffix in ["change_12", "change_36", "slope", "volatility"])
        return True
    if lowered.startswith("nri_hazard_"):
        return lowered.endswith("_score") or lowered.endswith("_pct_nation")
    if lowered.startswith("nri_"):
        return "score" in lowered or "per_capita" in lowered
    if lowered.startswith("fema_disaster_"):
        return lowered in {"fema_disaster_total_count", "fema_disaster_years_with_declarations"}
    if lowered.startswith(("insurance_", "property_tax_")):
        return True
    return False
