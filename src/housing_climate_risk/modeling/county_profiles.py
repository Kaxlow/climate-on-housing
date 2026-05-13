"""
Cluster counties by economic and demographic characteristics.

The page-data pipeline prepares the raw economic and population DataFrames, then
calls ``build_county_profile_clusters`` to select the best clustering model,
assign county profile labels, write profile artifacts, and return DataFrames
that can be joined back onto housing rows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


BEA_LINECODES = {
    10: "personal_income",
    30: "per_capita_income",
    46: "dividends_interest_rent",
    47: "transfer_receipts",
    70: "proprietors_income",
}

FEATURE_LABELS = {
    "log_population": "Population size",
    "per_capita_income": "Per capita income",
    "employment_rate_proxy": "Employment per resident",
    "Average Weekly Wage": "Average weekly wage",
    "transfer_receipts_share": "Transfer receipts share",
    "dividends_rent_share": "Dividends, interest, and rent share",
    "proprietors_income_share": "Proprietors income share",
    "youth_share": "Youth share",
    "prime_working_age_share": "Prime working-age share",
    "senior_share": "Senior share",
    "male_share": "Male share",
    "white_share": "White share",
    "black_share": "Black share",
    "asian_share": "Asian share",
    "hispanic_share": "Hispanic share",
    "natural_increase_rate": "Natural increase rate",
    "international_migration_rate": "International migration rate",
    "domestic_migration_rate": "Domestic migration rate",
}

FEATURE_COLUMNS = list(FEATURE_LABELS)


def build_latest_available_panel(df: pd.DataFrame, value_cols: list[str], years: list[int]) -> pd.DataFrame:
    df = df[["fips", "Year"] + value_cols].sort_values(["fips", "Year"]).drop_duplicates(["fips", "Year"])
    idx = pd.MultiIndex.from_product(
        [df["fips"].unique(), range(int(df["Year"].min()), years[-1] + 1)],
        names=["fips", "Year"],
    )
    out = df.set_index(["fips", "Year"]).reindex(idx)
    out = out.groupby(level=0).ffill().groupby(level=0).bfill().reset_index()
    return out[out["Year"].isin(years)].copy()


def summarize_cluster(z_scores: pd.Series, county_count: int) -> str:
    high = z_scores.sort_values(ascending=False).head(3)
    low = z_scores.sort_values().head(3)

    def phrase(feature: str, value: float) -> str:
        label = FEATURE_LABELS[feature]
        direction = "above" if value > 0 else "below"
        return f"{label} ({abs(value):.1f} SD {direction} average)"

    high_text = "; ".join(phrase(feature, value) for feature, value in high.items())
    low_text = "; ".join(phrase(feature, value) for feature, value in low.items())
    return (
        f"This cluster contains {county_count:,} counties. "
        f"It is distinguished by {high_text}. "
        f"It is relatively low on {low_text}."
    )


def build_county_year_panel(
    *,
    bea_income_df: pd.DataFrame,
    cew_total_df: pd.DataFrame,
    population_change_df: pd.DataFrame,
    population_age_sex_df: pd.DataFrame,
    population_race_df: pd.DataFrame,
    population_estimates_df: pd.DataFrame,
    target_years: list[int],
) -> pd.DataFrame:
    county_lookup = (
        population_estimates_df.sort_values(["fips", "Year"])
        .drop_duplicates("fips", keep="last")[["fips", "Description"]]
        .rename(columns={"Description": "county_name"})
    )

    bea_features = (
        bea_income_df.pivot_table(index=["fips", "Year"], columns="metric", values="Data", aggfunc="first")
        .reset_index()
    )
    age_features = population_age_sex_df[
        [
            "fips",
            "Year",
            "Total Population",
            "Population 0-4",
            "Population 5-17",
            "Population 18-24",
            "Population 25-44",
            "Population 45-64",
            "Population 65+",
            "Male Population",
            "Female Population",
        ]
    ].copy()
    race_features = population_race_df[
        ["fips", "Year", "Total Population", "White Alone", "Black Alone", "Asian Alone", "Hispanic"]
    ].copy()
    change_features = population_change_df[
        ["fips", "Year", "Births", "Deaths", "Net International Migration", "Net Domestic Migration", "Residual"]
    ].copy()
    estimate_features = population_estimates_df[["fips", "Year", "Population"]].copy()
    cew_features = cew_total_df[["fips", "Year", "Employment", "Wages", "Average Wage", "Average Weekly Wage"]].copy()

    county_year_panel = (
        county_lookup.assign(key=1)
        .merge(pd.DataFrame({"Year": target_years, "key": 1}), on="key")
        .drop(columns="key")
    )

    for source_df, value_cols in [
        (estimate_features, ["Population"]),
        (
            age_features,
            [
                "Total Population",
                "Population 0-4",
                "Population 5-17",
                "Population 18-24",
                "Population 25-44",
                "Population 45-64",
                "Population 65+",
                "Male Population",
                "Female Population",
            ],
        ),
        (race_features, ["Total Population", "White Alone", "Black Alone", "Asian Alone", "Hispanic"]),
        (change_features, ["Births", "Deaths", "Net International Migration", "Net Domestic Migration", "Residual"]),
        (bea_features, list(BEA_LINECODES.values())),
        (cew_features, ["Employment", "Wages", "Average Wage", "Average Weekly Wage"]),
    ]:
        county_year_panel = county_year_panel.merge(
            build_latest_available_panel(source_df, value_cols, target_years),
            on=["fips", "Year"],
            how="left",
        )

    county_year_panel = county_year_panel.rename(
        columns={"Total Population_x": "age_total_population", "Total Population_y": "race_total_population"}
    )
    county_year_panel["log_population"] = np.log1p(county_year_panel["Population"])
    county_year_panel["employment_rate_proxy"] = county_year_panel["Employment"] / county_year_panel["Population"]
    county_year_panel["transfer_receipts_share"] = county_year_panel["transfer_receipts"] / county_year_panel["personal_income"]
    county_year_panel["dividends_rent_share"] = county_year_panel["dividends_interest_rent"] / county_year_panel["personal_income"]
    county_year_panel["proprietors_income_share"] = county_year_panel["proprietors_income"] / county_year_panel["personal_income"]
    county_year_panel["youth_share"] = (
        county_year_panel["Population 0-4"] + county_year_panel["Population 5-17"]
    ) / county_year_panel["age_total_population"]
    county_year_panel["prime_working_age_share"] = (
        county_year_panel["Population 25-44"] + county_year_panel["Population 45-64"]
    ) / county_year_panel["age_total_population"]
    county_year_panel["senior_share"] = county_year_panel["Population 65+"] / county_year_panel["age_total_population"]
    county_year_panel["male_share"] = county_year_panel["Male Population"] / county_year_panel["age_total_population"]
    county_year_panel["white_share"] = county_year_panel["White Alone"] / county_year_panel["race_total_population"]
    county_year_panel["black_share"] = county_year_panel["Black Alone"] / county_year_panel["race_total_population"]
    county_year_panel["asian_share"] = county_year_panel["Asian Alone"] / county_year_panel["race_total_population"]
    county_year_panel["hispanic_share"] = county_year_panel["Hispanic"] / county_year_panel["race_total_population"]
    county_year_panel["natural_increase_rate"] = (
        county_year_panel["Births"] - county_year_panel["Deaths"]
    ) / county_year_panel["Population"]
    county_year_panel["international_migration_rate"] = county_year_panel["Net International Migration"] / county_year_panel["Population"]
    county_year_panel["domestic_migration_rate"] = county_year_panel["Net Domestic Migration"] / county_year_panel["Population"]
    return county_year_panel


def build_county_profile_clusters(
    *,
    bea_income_df: pd.DataFrame,
    cew_total_df: pd.DataFrame,
    population_change_df: pd.DataFrame,
    population_age_sex_df: pd.DataFrame,
    population_race_df: pd.DataFrame,
    population_estimates_df: pd.DataFrame,
    target_years: list[int],
    output_dir: Path | str,
    candidate_ks: range | list[int] = range(3, 10),
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    county_year_panel = build_county_year_panel(
        bea_income_df=bea_income_df,
        cew_total_df=cew_total_df,
        population_change_df=population_change_df,
        population_age_sex_df=population_age_sex_df,
        population_race_df=population_race_df,
        population_estimates_df=population_estimates_df,
        target_years=target_years,
    )
    county_average_df = county_year_panel.groupby(["fips", "county_name"], as_index=False)[FEATURE_COLUMNS].mean()
    cluster_input = county_average_df[["fips", "county_name"] + FEATURE_COLUMNS].copy()
    cluster_input = cluster_input.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLUMNS, how="all").copy()

    county_profile_preprocess = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", RobustScaler()),
        ]
    )
    x = county_profile_preprocess.fit_transform(cluster_input[FEATURE_COLUMNS])

    cluster_quality = []
    cluster_fitted_models = {}
    for k in candidate_ks:
        for algorithm, model in [
            ("kmeans", KMeans(n_clusters=k, random_state=42, n_init=20)),
            ("ward_agglomerative", AgglomerativeClustering(n_clusters=k, linkage="ward")),
        ]:
            labels = model.fit_predict(x)
            unique_labels = np.unique(labels)
            if len(unique_labels) < 2 or len(unique_labels) >= len(cluster_input):
                continue
            cluster_key = (algorithm, int(len(unique_labels)))
            cluster_fitted_models[cluster_key] = model
            cluster_quality.append(
                {
                    "algorithm": algorithm,
                    "k": int(len(unique_labels)),
                    "silhouette_score": silhouette_score(x, labels, sample_size=min(10_000, len(cluster_input)), random_state=42),
                    "calinski_harabasz_score": calinski_harabasz_score(x, labels),
                    "davies_bouldin_index": davies_bouldin_score(x, labels),
                    "cluster_sizes": dict(pd.Series(labels).value_counts().sort_index()),
                }
            )

    cluster_quality_df = pd.DataFrame(cluster_quality)
    cluster_quality_df["silhouette_rank"] = cluster_quality_df["silhouette_score"].rank(ascending=False, method="min")
    cluster_quality_df["calinski_harabasz_rank"] = cluster_quality_df["calinski_harabasz_score"].rank(ascending=False, method="min")
    cluster_quality_df["davies_bouldin_rank"] = cluster_quality_df["davies_bouldin_index"].rank(ascending=True, method="min")
    cluster_quality_df["combined_metric_rank"] = cluster_quality_df[
        ["silhouette_rank", "calinski_harabasz_rank", "davies_bouldin_rank"]
    ].mean(axis=1)
    cluster_quality_df = cluster_quality_df.sort_values(
        ["combined_metric_rank", "silhouette_rank", "davies_bouldin_rank", "algorithm", "k"]
    ).reset_index(drop=True)

    best_cluster_row = cluster_quality_df.iloc[0]
    best_algorithm = best_cluster_row["algorithm"]
    best_k = int(best_cluster_row["k"])
    best_cluster_model = cluster_fitted_models[(best_algorithm, best_k)]
    cluster_input["cluster"] = (
        best_cluster_model.labels_ if hasattr(best_cluster_model, "labels_") else best_cluster_model.predict(x)
    )

    cluster_sizes = cluster_input.groupby("cluster").agg(counties=("fips", "size")).sort_index()
    cluster_feature_means = cluster_input.groupby("cluster")[FEATURE_COLUMNS].mean()
    overall_feature_means = cluster_input[FEATURE_COLUMNS].mean()
    overall_feature_std = cluster_input[FEATURE_COLUMNS].std().replace(0, np.nan)
    cluster_zscores = (cluster_feature_means - overall_feature_means) / overall_feature_std

    cluster_interpretations = []
    for cluster_id in cluster_zscores.index:
        county_count = int(cluster_sizes.loc[cluster_id, "counties"])
        examples = ", ".join(
            cluster_input.loc[cluster_input["cluster"] == cluster_id, "county_name"].drop_duplicates().head(8).tolist()
        )
        cluster_interpretations.append(
            {
                "county_profile": cluster_id,
                "county_profile_desc": summarize_cluster(cluster_zscores.loc[cluster_id], county_count),
                "example_counties": examples,
            }
        )

    cluster_interpretations_df = pd.DataFrame(cluster_interpretations)
    county_profiles_summary_df = cluster_interpretations_df[["county_profile", "county_profile_desc"]].copy()
    county_profiles_summary_df["county_profile_algorithm"] = best_algorithm
    county_profiles_summary_df["county_profile_k"] = best_k
    county_profiles_summary_df["county_profile_silhouette"] = float(best_cluster_row["silhouette_score"])
    county_profiles_summary_df["county_profile_calinski_harabasz"] = float(best_cluster_row["calinski_harabasz_score"])
    county_profiles_summary_df["county_profile_davies_bouldin"] = float(best_cluster_row["davies_bouldin_index"])
    county_profiles_summary_df["county_profile_combined_metric_rank"] = float(best_cluster_row["combined_metric_rank"])

    county_profile_df = (
        cluster_input[["fips", "county_name", "cluster"]]
        .rename(columns={"cluster": "county_profile"})
        .merge(county_profiles_summary_df, on="county_profile", how="left")
    )

    paths = {
        "profiles": output_dir / "county_profiles.csv",
        "assignments": output_dir / "county_profile_assignments.csv",
        "model": output_dir / "county_profile_model.joblib",
        "quality": output_dir / "county_profile_model_quality.csv",
    }
    county_profiles_summary_df.to_csv(paths["profiles"], index=False)
    county_profile_df.to_csv(paths["assignments"], index=False)
    cluster_quality_df.to_csv(paths["quality"], index=False)
    dump(
        {
            "preprocess_pipeline": county_profile_preprocess,
            "model": best_cluster_model,
            "algorithm": best_algorithm,
            "k": best_k,
            "feature_columns": FEATURE_COLUMNS,
            "feature_labels": FEATURE_LABELS,
            "selection_metrics": best_cluster_row.to_dict(),
            "cluster_interpretations": cluster_interpretations_df.to_dict(orient="records"),
        },
        paths["model"],
    )

    return {
        "county_year_panel": county_year_panel,
        "county_average_df": county_average_df,
        "cluster_input": cluster_input,
        "cluster_quality_df": cluster_quality_df,
        "county_profiles_summary_df": county_profiles_summary_df,
        "county_profile_df": county_profile_df,
        "cluster_interpretations_df": cluster_interpretations_df,
        "best_cluster_row": best_cluster_row,
        "best_algorithm": best_algorithm,
        "best_k": best_k,
        "paths": paths,
    }
