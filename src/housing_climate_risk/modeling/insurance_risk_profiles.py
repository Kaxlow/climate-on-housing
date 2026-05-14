"""Home-insurance premium and nonrenewal county profiles.

This module builds broad county groups from home-insurance premium level,
premium growth, nonrenewal-rate level, and nonrenewal-rate volatility/trend
attributes. It intentionally excludes property tax, NFIP flood claims, and raw
policy-count fields so the Stormhouse page can discuss housing insurance
characteristics directly.
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

from housing_climate_risk.modeling.county_clustering.features import (
    ID_COLUMNS,
    extract_insurance_nonrenewal_features,
    extract_insurance_premium_features,
)


def build_insurance_profile_outputs(
    *,
    counties: pd.DataFrame,
    nri_ratings: pd.DataFrame,
    output_dir: Path | str,
    risk_order: list[str],
    candidate_ks: range | list[int] = range(4, 6),
    random_state: int = 42,
) -> dict[str, Any]:
    """Cluster counties by insurance-related features and summarize by NRI rating."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_df = build_insurance_feature_matrix(counties).merge(nri_ratings, on="fips", how="inner")
    id_columns = [column for column in ID_COLUMNS if column in feature_df.columns]
    model_features = [column for column in feature_df.columns if column.startswith("ins_")]
    x_df = winsorize_frame(feature_df[model_features].replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all"))

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
    profile_labels = label_profiles(feature_df, best["labels"], best["probabilities"])
    assignments = build_assignment_table(feature_df, id_columns, best["labels"], best["probabilities"], profile_labels)
    risk_profile_summary = build_risk_profile_summary(assignments, risk_order)
    risk_lifts = build_nri_feature_lifts(feature_df, risk_order)
    profile_summary = build_profile_summary(feature_df, assignments)

    assignments.to_csv(output_dir / "stormhouse_insurance_profile_assignments.csv", index=False)
    risk_profile_summary.to_csv(output_dir / "stormhouse_insurance_profile_summary.csv", index=False)
    risk_lifts.to_csv(output_dir / "stormhouse_nri_rating_insurance_lifts.csv", index=False)
    profile_summary.to_csv(output_dir / "stormhouse_insurance_profile_labels.csv", index=False)
    score_df.to_csv(output_dir / "stormhouse_insurance_profile_scores.csv", index=False)
    dump(
        {
            "preprocess": preprocess,
            "pca": pca,
            "model": best["model"],
            "feature_columns": x_df.columns.tolist(),
            "best_k": best_k,
            "profile_labels": profile_labels,
            "scores": score_df.iloc[0].to_dict(),
        },
        output_dir / "stormhouse_insurance_profile_model.joblib",
    )

    return {
        "assignments": assignments,
        "risk_profile_summary": risk_profile_summary,
        "risk_lifts": risk_lifts,
        "profile_summary": profile_summary,
        "scores": score_df,
        "best_k": best_k,
    }


def build_insurance_feature_matrix(counties: pd.DataFrame) -> pd.DataFrame:
    """Create a county-level matrix from insurance-related nested fields."""

    rows = []
    for row in counties.itertuples(index=False):
        features = {column: getattr(row, column, None) for column in ID_COLUMNS if hasattr(row, column)}
        raw = {}
        raw.update(extract_insurance_premium_features(getattr(row, "insurance_premiums_14_to_24", None)))
        raw.update(extract_insurance_nonrenewal_features(getattr(row, "insurance_non_renewal_rates", None)))
        features.update({f"ins_{key}": value for key, value in raw.items() if keep_insurance_feature(key)})
        rows.append(features)
    out = pd.DataFrame(rows)
    for column in out.columns:
        if column not in ID_COLUMNS:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan)


def keep_insurance_feature(column: str) -> bool:
    """Keep premium and nonrenewal-rate signals; drop raw size and non-insurance fields."""

    lowered = column.lower()
    if lowered.endswith("_missing_rate") or any(token in lowered for token in ["start_year", "end_year", "latest_year"]):
        return False
    if any(token in lowered for token in ["policycount", "num_policies", "coverage", "propertyvalue"]):
        return False
    if lowered.startswith("insurance_premium_"):
        if "_change_12" in lowered or "_change_36" in lowered:
            return False
        return True
    if lowered.startswith("insurance_nonrenewal_"):
        if "_change_12" in lowered or "_change_36" in lowered:
            return False
        return "non_renewal_rate" in lowered
    return False


def winsorize_frame(x_df: pd.DataFrame, lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    """Clip extreme values so a few outlier counties do not define the clusters."""

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


def score_candidate(x_model: np.ndarray, labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    """Score one GMM candidate for balance, separation, and assignment certainty."""

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


def rank_candidates(score_df: pd.DataFrame) -> pd.DataFrame:
    """Rank 4-5 cluster candidates with a preference for broad, balanced profiles."""

    score_df = score_df.copy()
    score_df["passes_balance_check"] = (
        (score_df["min_cluster_size"] >= 150)
        & (score_df["max_cluster_share"] <= 0.45)
        & (score_df["low_confidence_under_0_60_rate"] <= 0.35)
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


def label_profiles(feature_df: pd.DataFrame, labels: np.ndarray, probabilities: np.ndarray) -> dict[int, str]:
    """Assign plain labels from each cluster's strongest feature contrasts."""

    feature_columns = [column for column in feature_df.columns if column.startswith("ins_")]
    seed = {cluster: str(cluster) for cluster in sorted(set(labels))}
    assignments = build_assignment_table(feature_df, [column for column in ID_COLUMNS if column in feature_df.columns], labels, probabilities, seed)
    merged = assignments.merge(feature_df[["fips", *feature_columns]], on="fips", how="left")
    national_median = merged[feature_columns].median()
    national_std = merged[feature_columns].std().replace(0, np.nan)

    out = {}
    used: set[str] = set()
    for profile, group in merged.groupby("insurance_profile"):
        lifts = ((group[feature_columns].median() - national_median) / national_std).dropna()
        high = lifts.sort_values(ascending=False).head(10)
        low = lifts.sort_values().head(8)
        label = derive_insurance_label(high, low)
        if label in used:
            label = f"{label} ({int(profile)})"
        used.add(label)
        out[int(profile)] = label
    return out


def derive_insurance_label(high_features: pd.Series, low_features: pd.Series) -> str:
    """Name an insurance cluster from its dominant high and low signals."""

    high = " ".join(high_features.index).lower()
    low = " ".join(low_features.index).lower()
    strongest_high = float(high_features.iloc[0]) if len(high_features) else 0.0
    strongest_low = float(low_features.iloc[0]) if len(low_features) else 0.0
    if strongest_high < 0.15 and strongest_low <= -0.50 and "volatility" in low:
        return "Stable low-volatility insurance counties"
    if strongest_high < 0.15 and "premium_latest" in low:
        return "Typical lower-premium counties"
    if strongest_high < 0.15 and "nonrenewal" in high and "volatility" in low:
        return "Steady nonrenewal-rate counties"
    if "premium_latest" in high and "nonrenewal" in high:
        return "High-premium high-nonrenewal counties"
    if "premium_latest" in high and "premium_growth" in low:
        return "High-premium slower-growth counties"
    if "premium_growth" in high and "premium_latest" in low:
        return "Fast-rising lower-premium counties"
    if "premium_growth" in high:
        return "Fast-rising premium counties"
    if "nonrenewal" in high and "premium" in high:
        return "High-cost nonrenewal-pressure counties"
    if "nonrenewal" in high:
        return "High nonrenewal-pressure counties"
    if "premium" in high:
        return "High-premium counties"
    if "premium" in low and "nonrenewal" in low:
        return "Lower insurance-pressure counties"
    return f"{human_insurance_feature_label(high_features.index[0])} counties"


def build_assignment_table(
    feature_df: pd.DataFrame,
    id_columns: list[str],
    labels: np.ndarray,
    probabilities: np.ndarray,
    profile_labels: dict[int, str],
) -> pd.DataFrame:
    """Build county assignments with soft GMM membership fields."""

    sorted_indices = np.argsort(probabilities, axis=1)
    second_best = sorted_indices[:, -2]
    confidence = probabilities.max(axis=1)
    margin = confidence - probabilities[np.arange(len(probabilities)), second_best]
    out = feature_df[[*id_columns, "nri_risk_rating"]].copy()
    out["insurance_profile"] = labels.astype(int)
    out["insurance_profile_label"] = out["insurance_profile"].map(profile_labels)
    out["assignment_confidence"] = confidence
    out["assignment_margin"] = margin
    out["second_best_profile"] = second_best.astype(int)
    out["second_best_profile_label"] = out["second_best_profile"].map(profile_labels)
    return out


def build_risk_profile_summary(assignments: pd.DataFrame, risk_order: list[str]) -> pd.DataFrame:
    counts = (
        assignments.groupby(["nri_risk_rating", "insurance_profile", "insurance_profile_label"], as_index=False)
        .agg(counties=("fips", "size"), mean_assignment_confidence=("assignment_confidence", "mean"))
    )
    totals = assignments.groupby("nri_risk_rating")["fips"].size().rename("risk_group_count")
    counts = counts.merge(totals, on="nri_risk_rating", how="left")
    counts["share"] = counts["counties"] / counts["risk_group_count"]
    counts["nri_risk_rating"] = pd.Categorical(counts["nri_risk_rating"], categories=risk_order, ordered=True)
    return counts.sort_values(["nri_risk_rating", "share"], ascending=[True, False]).reset_index(drop=True)


def build_nri_feature_lifts(feature_df: pd.DataFrame, risk_order: list[str], top_n: int = 6) -> pd.DataFrame:
    feature_columns = [column for column in feature_df.columns if column.startswith("ins_")]
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
                        "feature_label": human_insurance_feature_label(feature),
                        "standardized_lift": float(lift),
                        "group_median": float(group[feature].median()),
                        "national_median": float(national_median[feature]),
                    }
                )
    return pd.DataFrame(rows)


def build_profile_summary(feature_df: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    feature_columns = [column for column in feature_df.columns if column.startswith("ins_")]
    merged = assignments.merge(feature_df[["fips", *feature_columns]], on="fips", how="left")
    national_median = merged[feature_columns].median()
    national_std = merged[feature_columns].std().replace(0, np.nan)
    rows = []
    for profile, group in merged.groupby("insurance_profile"):
        lifts = ((group[feature_columns].median() - national_median) / national_std).dropna()
        high = lifts.sort_values(ascending=False).head(6)
        low = lifts.sort_values().head(5)
        rows.append(
            {
                "insurance_profile": int(profile),
                "insurance_profile_label": group["insurance_profile_label"].iloc[0],
                "county_count": int(len(group)),
                "mean_assignment_confidence": float(group["assignment_confidence"].mean()),
                "top_high_features": "; ".join(f"{human_insurance_feature_label(feature)} ({value:.2f} SD)" for feature, value in high.items()),
                "top_low_features": "; ".join(f"{human_insurance_feature_label(feature)} ({value:.2f} SD)" for feature, value in low.items()),
            }
        )
    return pd.DataFrame(rows).sort_values("insurance_profile")


def human_insurance_feature_label(feature: str) -> str:
    """Convert technical insurance feature names to readable labels."""

    label = feature.removeprefix("ins_")
    label = label.replace("insurance_premium", "home-insurance premium")
    label = label.replace("insurance_nonrenewal", "home-insurance nonrenewal")
    label = label.replace("percentiles", "percentile")
    label = label.replace("nationally", "national percentile")
    label = label.replace("within_state", "state percentile")
    label = label.replace("cagr", "growth")
    label = label.replace("non_renewal_rate", "nonrenewal rate")
    label = label.replace("_", " ")
    return " ".join(label.split())
