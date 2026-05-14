"""Economic profiles and NRI risk-group contrasts.

This module supports the Stormhouse page section that asks which county-economy
characteristics distinguish counties in each NRI risk rating group. It clusters
counties by normalized economic, employment, income-source, population-scale,
and migration features once, then cross-tabs those economic profiles against
NRI risk ratings. Demographic fields are summarized after clustering, but they
are not used to assign or label economic profiles.
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
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer

from housing_climate_risk.modeling.county_profiles import FEATURE_COLUMNS, build_county_year_panel


FEATURE_LABELS = {
    "log_population": "population scale",
    "per_capita_income": "per capita income",
    "employment_rate_proxy": "employment per resident",
    "Average Weekly Wage": "average weekly wage",
    "transfer_receipts_share": "transfer receipts share",
    "dividends_rent_share": "dividends, interest, and rent share",
    "proprietors_income_share": "proprietors income share",
    "youth_share": "youth share",
    "prime_working_age_share": "prime working-age share",
    "senior_share": "senior share",
    "male_share": "male share",
    "white_share": "White share",
    "black_share": "Black share",
    "asian_share": "Asian share",
    "hispanic_share": "Hispanic share",
    "natural_increase_rate": "natural increase rate",
    "international_migration_rate": "international migration rate",
    "domestic_migration_rate": "domestic migration rate",
}

DEMOGRAPHIC_BASE_COLUMNS = {
    "youth_share",
    "prime_working_age_share",
    "senior_share",
    "male_share",
    "white_share",
    "black_share",
    "asian_share",
    "hispanic_share",
}

ECONOMIC_MODEL_BASE_COLUMNS = [column for column in FEATURE_COLUMNS if column not in DEMOGRAPHIC_BASE_COLUMNS]


def build_economic_profile_outputs(
    *,
    profile_inputs: dict[str, pd.DataFrame],
    nri_ratings: pd.DataFrame,
    output_dir: Path | str,
    risk_order: list[str],
    candidate_ks: range | list[int] = range(4, 6),
    random_state: int = 42,
) -> dict[str, Any]:
    """Cluster counties economically and summarize those profiles by NRI rating."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_df = build_economic_feature_matrix(profile_inputs)
    feature_df = feature_df.merge(nri_ratings, on="fips", how="inner")
    model_features = economic_model_feature_columns(feature_df)
    x_df = feature_df[model_features].replace([np.inf, -np.inf], np.nan)

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
    pca = PCA(n_components=min(8, x_scaled.shape[1]), random_state=random_state)
    x_model = pca.fit_transform(x_scaled)

    scores = []
    artifacts = {}
    for k in candidate_ks:
        model = GaussianMixture(
            n_components=int(k),
            covariance_type="diag",
            reg_covar=1e-3,
            n_init=20,
            random_state=random_state,
        )
        labels = model.fit_predict(x_model)
        probabilities = model.predict_proba(x_model)
        row = score_candidate(x_model, labels, probabilities)
        row["k"] = int(k)
        scores.append(row)
        artifacts[int(k)] = {"model": model, "labels": labels, "probabilities": probabilities}

    score_df = rank_candidates(pd.DataFrame(scores))
    best_k = int(score_df.iloc[0]["k"])
    best = artifacts[best_k]
    profile_labels = label_profiles(feature_df, best["labels"], best["probabilities"], best_k)
    assignments = build_assignment_table(feature_df, best["labels"], best["probabilities"], profile_labels)
    risk_profile_summary = build_risk_profile_summary(assignments, risk_order)
    risk_lifts = build_nri_feature_lifts(feature_df, risk_order)
    profile_summary = build_profile_summary(feature_df, assignments)

    assignments.to_csv(output_dir / "stormhouse_economic_profile_assignments.csv", index=False)
    risk_profile_summary.to_csv(output_dir / "stormhouse_economic_profile_summary.csv", index=False)
    risk_lifts.to_csv(output_dir / "stormhouse_nri_rating_economic_lifts.csv", index=False)
    profile_summary.to_csv(output_dir / "stormhouse_economic_profile_labels.csv", index=False)
    score_df.to_csv(output_dir / "stormhouse_economic_profile_scores.csv", index=False)
    dump(
        {
            "preprocess": preprocess,
            "pca": pca,
            "model": best["model"],
            "feature_columns": model_features,
            "best_k": best_k,
            "profile_labels": profile_labels,
            "scores": score_df.iloc[0].to_dict(),
        },
        output_dir / "stormhouse_economic_profile_model.joblib",
    )

    return {
        "assignments": assignments,
        "risk_profile_summary": risk_profile_summary,
        "risk_lifts": risk_lifts,
        "profile_summary": profile_summary,
        "scores": score_df,
        "best_k": best_k,
    }


def build_economic_feature_matrix(profile_inputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Create averaged, latest, and trend economic/population features."""

    max_year = int(pd.to_numeric(profile_inputs["population_estimates_df"]["Year"], errors="coerce").max())
    target_years = list(range(max_year - 9, max_year + 1))
    panel = build_county_year_panel(**profile_inputs, target_years=target_years).replace([np.inf, -np.inf], np.nan)
    average = panel.groupby(["fips", "county_name"], as_index=False)[FEATURE_COLUMNS].mean()
    latest = (
        panel.sort_values(["fips", "Year"])
        .groupby("fips", as_index=False)
        .tail(1)[["fips", *FEATURE_COLUMNS]]
        .rename(columns={column: f"{column}_latest" for column in FEATURE_COLUMNS})
    )
    trends = panel.groupby("fips").apply(_trend_features, include_groups=False).reset_index()
    average = average.rename(columns={column: f"econ_{column}" for column in FEATURE_COLUMNS})
    latest = latest.rename(columns={column: f"econ_{column}" for column in latest.columns if column != "fips"})
    trends = trends.rename(columns={column: f"econ_{column}" for column in trends.columns if column != "fips"})
    return average.merge(latest, on="fips", how="left").merge(trends, on="fips", how="left")


def _trend_features(group: pd.DataFrame) -> pd.Series:
    trend_columns = [
        "log_population",
        "per_capita_income",
        "Average Weekly Wage",
        "employment_rate_proxy",
        "domestic_migration_rate",
    ]
    out = {}
    for column in trend_columns:
        out[f"{column}_slope"] = linear_slope(group[column].to_numpy(dtype=float))
    return pd.Series(out)


def economic_model_feature_columns(feature_df: pd.DataFrame) -> list[str]:
    """Return the non-demographic columns used for economic clustering."""

    return [
        column
        for column in feature_df.columns
        if column.startswith("econ_") and base_feature_name(column) in ECONOMIC_MODEL_BASE_COLUMNS
    ]


def demographic_feature_columns(feature_df: pd.DataFrame) -> list[str]:
    """Return demographic columns used only for post-cluster descriptions."""

    return [
        column
        for column in feature_df.columns
        if column.startswith("econ_") and base_feature_name(column) in DEMOGRAPHIC_BASE_COLUMNS
    ]


def base_feature_name(feature: str) -> str:
    stripped = feature.removeprefix("econ_")
    if stripped.endswith("_latest"):
        stripped = stripped.removesuffix("_latest")
    elif stripped.endswith("_slope"):
        stripped = stripped.removesuffix("_slope")
    return stripped


def score_candidate(x_model: np.ndarray, labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
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
        "low_confidence_under_0_60_rate": float((confidence < 0.60).mean()),
        "mean_assignment_margin": float(margin.mean()),
        "silhouette_score": float(silhouette_score(x_model, labels, sample_size=min(10_000, len(labels)), random_state=42)),
        "davies_bouldin_index": float(davies_bouldin_score(x_model, labels)),
        "cluster_sizes": json.dumps({int(key): int(value) for key, value in sizes.items()}, sort_keys=True),
    }


def rank_candidates(score_df: pd.DataFrame) -> pd.DataFrame:
    score_df = score_df.copy()
    score_df["passes_balance_check"] = (
        (score_df["min_cluster_size"] >= 150)
        & (score_df["max_cluster_share"] <= 0.45)
        & (score_df["low_confidence_under_0_60_rate"] <= 0.30)
    )
    score_df["silhouette_rank"] = score_df["silhouette_score"].rank(ascending=False, method="min")
    score_df["davies_bouldin_rank"] = score_df["davies_bouldin_index"].rank(ascending=True, method="min")
    score_df["confidence_rank"] = score_df["mean_assignment_confidence"].rank(ascending=False, method="min")
    score_df["balance_penalty"] = score_df["max_cluster_share"] * 8 + score_df["cluster_size_cv"]
    score_df["usable_rank_score"] = (
        score_df[["silhouette_rank", "davies_bouldin_rank", "confidence_rank"]].mean(axis=1)
        + score_df["balance_penalty"]
    )
    return score_df.sort_values(["passes_balance_check", "usable_rank_score"], ascending=[False, True]).reset_index(drop=True)


def label_profiles(
    feature_df: pd.DataFrame,
    labels: np.ndarray,
    probabilities: np.ndarray,
    best_k: int,
) -> dict[int, str]:
    """Assign plain labels to economic profile clusters."""

    feature_columns = economic_model_feature_columns(feature_df)
    label_seed = {cluster: str(cluster) for cluster in sorted(set(labels))}
    assignments = build_assignment_table(feature_df, labels, probabilities, label_seed)
    merged = assignments.merge(feature_df[["fips", *feature_columns]], on="fips", how="left")
    national_median = merged[feature_columns].median()
    national_std = merged[feature_columns].std().replace(0, np.nan)

    out = {}
    used_labels: set[str] = set()
    for profile, group in merged.groupby("economic_profile"):
        lifts = ((group[feature_columns].median() - national_median) / national_std).dropna()
        high_features = list(lifts.sort_values(ascending=False).head(8).index)
        low_features = list(lifts.sort_values().head(6).index)
        label = derive_economic_label(high_features, low_features)
        if label in used_labels:
            label = f"{label} ({int(profile)})"
        used_labels.add(label)
        out[int(profile)] = label
    return out


def build_assignment_table(
    feature_df: pd.DataFrame,
    labels: np.ndarray,
    probabilities: np.ndarray,
    profile_labels: dict[int, str],
) -> pd.DataFrame:
    sorted_indices = np.argsort(probabilities, axis=1)
    second_best = sorted_indices[:, -2]
    confidence = probabilities.max(axis=1)
    margin = confidence - probabilities[np.arange(len(probabilities)), second_best]
    out = feature_df[["fips", "county_name", "nri_risk_rating"]].copy()
    out["economic_profile"] = labels.astype(int)
    out["economic_profile_label"] = out["economic_profile"].map(profile_labels)
    out["assignment_confidence"] = confidence
    out["assignment_margin"] = margin
    out["second_best_profile"] = second_best.astype(int)
    out["second_best_profile_label"] = out["second_best_profile"].map(profile_labels)
    return out


def build_risk_profile_summary(assignments: pd.DataFrame, risk_order: list[str]) -> pd.DataFrame:
    counts = (
        assignments.groupby(["nri_risk_rating", "economic_profile", "economic_profile_label"], as_index=False)
        .agg(counties=("fips", "size"), mean_assignment_confidence=("assignment_confidence", "mean"))
    )
    totals = assignments.groupby("nri_risk_rating")["fips"].size().rename("risk_group_count")
    counts = counts.merge(totals, on="nri_risk_rating", how="left")
    counts["share"] = counts["counties"] / counts["risk_group_count"]
    counts["nri_risk_rating"] = pd.Categorical(counts["nri_risk_rating"], categories=risk_order, ordered=True)
    return counts.sort_values(["nri_risk_rating", "share"], ascending=[True, False]).reset_index(drop=True)


def build_nri_feature_lifts(feature_df: pd.DataFrame, risk_order: list[str], top_n: int = 5) -> pd.DataFrame:
    feature_columns = economic_model_feature_columns(feature_df)
    national_median = feature_df[feature_columns].median()
    national_std = feature_df[feature_columns].std().replace(0, np.nan)
    rows = []
    for rating in risk_order:
        group = feature_df.loc[feature_df["nri_risk_rating"] == rating]
        if group.empty:
            continue
        lifts = ((group[feature_columns].median() - national_median) / national_std).dropna()
        for direction, values in [("higher", lifts.sort_values(ascending=False).head(top_n)), ("lower", lifts.sort_values().head(top_n))]:
            for feature, lift in values.items():
                rows.append(
                    {
                        "nri_risk_rating": rating,
                        "direction": direction,
                        "feature": feature,
                        "feature_label": human_feature_label(feature),
                        "standardized_lift": float(lift),
                        "group_median": float(group[feature].median()),
                        "national_median": float(national_median[feature]),
                    }
                )
    return pd.DataFrame(rows)


def build_profile_summary(feature_df: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    feature_columns = economic_model_feature_columns(feature_df)
    demographic_columns = demographic_feature_columns(feature_df)
    merged = assignments.merge(feature_df[["fips", *feature_columns, *demographic_columns]], on="fips", how="left")
    national_median = merged[feature_columns].median()
    national_std = merged[feature_columns].std().replace(0, np.nan)
    demographic_median = merged[demographic_columns].median() if demographic_columns else pd.Series(dtype=float)
    demographic_std = merged[demographic_columns].std().replace(0, np.nan) if demographic_columns else pd.Series(dtype=float)
    rows = []
    for profile, group in merged.groupby("economic_profile"):
        lifts = ((group[feature_columns].median() - national_median) / national_std).dropna()
        high = lifts.sort_values(ascending=False).head(6)
        low = lifts.sort_values().head(5)
        demographic_lifts = ((group[demographic_columns].median() - demographic_median) / demographic_std).dropna()
        rows.append(
            {
                "economic_profile": int(profile),
                "economic_profile_label": group["economic_profile_label"].iloc[0],
                "county_count": int(len(group)),
                "mean_assignment_confidence": float(group["assignment_confidence"].mean()),
                "top_high_features": "; ".join(f"{human_feature_label(feature)} ({value:.2f} SD)" for feature, value in high.items()),
                "top_low_features": "; ".join(f"{human_feature_label(feature)} ({value:.2f} SD)" for feature, value in low.items()),
                "demographic_description": describe_demographics(demographic_lifts),
            }
        )
    return pd.DataFrame(rows).sort_values("economic_profile")


def derive_economic_label(high_features: list[str], low_features: list[str]) -> str:
    """Name a cluster from the dominant positive and negative feature contrasts."""

    high = set(high_features)
    low = set(low_features)

    def has_base(features: set[str], base: str) -> bool:
        return any(base_feature_name(feature) == base for feature in features)

    if has_base(high, "log_population") and has_base(high, "Average Weekly Wage"):
        return "Large high-wage metro counties"
    if has_base(high, "log_population"):
        return "Larger average-economy counties"
    if has_base(low, "log_population") and has_base(high, "transfer_receipts_share"):
        return "Small transfer-reliant counties"
    if has_base(low, "log_population") and (
        has_base(high, "proprietors_income_share") or has_base(high, "domestic_migration_rate")
    ):
        return "Smaller mixed-economy counties"
    if has_base(high, "Average Weekly Wage") and has_base(high, "dividends_rent_share"):
        return "High-wage investment-income counties"
    if has_base(high, "transfer_receipts_share") and (
        has_base(low, "log_population") or has_base(low, "Average Weekly Wage")
    ):
        return "Small transfer-reliant counties"
    if has_base(high, "per_capita_income") and has_base(low, "transfer_receipts_share"):
        return "Higher-income low-transfer counties"
    if has_base(high, "domestic_migration_rate") or has_base(high, "international_migration_rate"):
        return "In-migration growth counties"
    if has_base(high, "employment_rate_proxy") and has_base(low, "Average Weekly Wage"):
        return "High-employment lower-wage counties"
    if has_base(high, "dividends_rent_share") and has_base(low, "proprietors_income_share"):
        return "Wage and investment-income counties"
    return f"{human_feature_label(high_features[0])} counties"


def describe_demographics(lifts: pd.Series) -> str:
    """Summarize demographic differences without using them as cluster labels."""

    if lifts.empty:
        return "Demographic mix is close to the national county pattern."
    material = lifts.loc[lifts.abs() >= 0.35].sort_values(key=lambda series: series.abs(), ascending=False)
    if material.empty:
        return "Demographic mix is close to the national county pattern."
    parts = []
    seen_bases: set[str] = set()
    for feature, value in material.items():
        base = base_feature_name(feature)
        if base in seen_bases:
            continue
        seen_bases.add(base)
        direction = "higher" if value > 0 else "lower"
        parts.append(f"{plain_demographic_label(feature)} is {direction} than typical")
        if len(parts) >= 3:
            break
    return "; ".join(parts) + "."


def plain_demographic_label(feature: str) -> str:
    label = human_feature_label(feature)
    return (
        label.replace(" latest", "")
        .replace("youth share", "youth share")
        .replace("prime working-age share", "working-age adult share")
        .replace("senior share", "senior share")
        .replace("male share", "male share")
        .replace("White share", "White population share")
        .replace("Black share", "Black population share")
        .replace("Asian share", "Asian population share")
        .replace("Hispanic share", "Hispanic population share")
    )


def human_feature_label(feature: str) -> str:
    stripped = feature.removeprefix("econ_")
    suffix = ""
    if stripped.endswith("_latest"):
        stripped = stripped.removesuffix("_latest")
        suffix = " latest"
    elif stripped.endswith("_slope"):
        stripped = stripped.removesuffix("_slope")
        suffix = " trend"
    return FEATURE_LABELS.get(stripped, stripped.replace("_", " ")) + suffix


def linear_slope(values: np.ndarray) -> float:
    mask = np.isfinite(values)
    if mask.sum() < 2:
        return np.nan
    x = np.arange(values.size, dtype=float)[mask]
    y = values[mask]
    return float(np.polyfit(x, y, 1)[0])
