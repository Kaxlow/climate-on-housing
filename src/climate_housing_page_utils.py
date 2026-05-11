from __future__ import annotations

import json
import re
import socket
import sys
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd


ROOT = Path.cwd()
if not (ROOT / "data").exists():
    ROOT = ROOT.parent

SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

OUTPUT_DIR = ROOT / "output" / "visualizations"
ECONOMIC_DIR = ROOT / "data" / "economic"
POPULATION_DIR = ROOT / "data" / "population"
CACHE_DIR = ROOT / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

from cluster_county_econ_demographics import BEA_LINECODES, build_county_profile_clusters
from cluster_housing_yoy_responses import (
    PPSF_RESPONSE_METRIC_LABELS,
    build_all_housing_market_response_clusters,
)
from cluster_pre_incident_market_strength import build_all_pre_incident_market_strength_tiers
try:
    from polars_cached_io import (
        read_cew_employment_wages_cached,
        read_cew_totals_cached,
        read_fema_disasters_cached,
        read_redfin_county_cached,
    )
except ModuleNotFoundError:
    FEMA_DISASTER_COLUMNS = [
        "disasterNumber",
        "state",
        "declarationDate",
        "incidentType",
        "declarationTitle",
        "ihProgramDeclared",
        "iaProgramDeclared",
        "paProgramDeclared",
        "hmProgramDeclared",
        "incidentBeginDate",
        "incidentEndDate",
        "designatedArea",
        "designatedIncidentTypes",
        "fipsStateCode",
        "fipsCountyCode",
    ]
    REDFIN_COUNTY_COLUMNS = [
        "REGION",
        "STATE_CODE",
        "PARENT_METRO_REGION",
        "PROPERTY_TYPE",
        "PERIOD_BEGIN",
        "PERIOD_END",
        "MEDIAN_PPSF",
        "MEDIAN_PPSF_YOY",
        "MEDIAN_LIST_PPSF",
        "MEDIAN_LIST_PPSF_YOY",
        "HOMES_SOLD",
        "HOMES_SOLD_YOY",
        "PENDING_SALES",
        "PENDING_SALES_YOY",
        "INVENTORY",
        "INVENTORY_YOY",
        "MONTHS_OF_SUPPLY",
        "MONTHS_OF_SUPPLY_YOY",
        "MEDIAN_DOM",
        "MEDIAN_DOM_YOY",
        "AVG_SALE_TO_LIST",
        "AVG_SALE_TO_LIST_YOY",
        "PRICE_DROPS",
        "PRICE_DROPS_YOY",
    ]

    def _fresh_cache(cache_path: Path, source_path: Path) -> bool:
        return cache_path.exists() and cache_path.stat().st_mtime >= source_path.stat().st_mtime

    def _read_cache_or_none(source_path: str | Path, cache_dir: str | Path, suffix: str) -> pd.DataFrame | None:
        source_path = Path(source_path)
        cache_path = Path(cache_dir) / f"{source_path.stem}_{suffix}.parquet"
        if _fresh_cache(cache_path, source_path):
            return pd.read_parquet(cache_path)
        return None

    def read_fema_disasters_cached(source_path: str | Path, *, cache_dir: str | Path = CACHE_DIR) -> pd.DataFrame:
        cached = _read_cache_or_none(source_path, cache_dir, "projected_fema")
        if cached is not None:
            return cached
        return pd.read_csv(source_path, usecols=FEMA_DISASTER_COLUMNS, low_memory=False)

    def read_redfin_county_cached(source_path: str | Path, *, cache_dir: str | Path = CACHE_DIR) -> pd.DataFrame:
        cached = _read_cache_or_none(source_path, cache_dir, "county_all_residential_projected")
        if cached is not None:
            return cached
        df = pd.read_csv(source_path, usecols=REDFIN_COUNTY_COLUMNS, low_memory=False)
        return df[df["PROPERTY_TYPE"] == "All Residential"].copy()

    def read_cew_totals_cached(source_path: str | Path, *, cache_dir: str | Path = CACHE_DIR) -> pd.DataFrame:
        cached = _read_cache_or_none(source_path, cache_dir, "county_totals")
        if cached is not None:
            return cached
        df = pd.read_csv(
            source_path,
            usecols=[
                "Statefips",
                "Countyfips",
                "Description",
                "Year",
                "Ownership Code",
                "NAICS Code",
                "Employment",
                "Wages",
                "Average Wage",
                "Average Weekly Wage",
            ],
            low_memory=False,
        )
        df["Ownership Code"] = pd.to_numeric(df["Ownership Code"], errors="coerce")
        df["NAICS Code"] = pd.to_numeric(df["NAICS Code"], errors="coerce")
        df["Statefips"] = pd.to_numeric(df["Statefips"], errors="coerce")
        df["Countyfips"] = pd.to_numeric(df["Countyfips"], errors="coerce")
        df = df[(df["Ownership Code"] == 0) & (df["NAICS Code"] == 0)].dropna(subset=["Statefips", "Countyfips"]).copy()
        df["state_fips"] = df["Statefips"].astype(int).astype(str).str.zfill(2)
        df["county_fips"] = df["Countyfips"].astype(int).astype(str).str.zfill(3)
        df = df[(df["state_fips"] != "00") & (df["county_fips"] != "000")].copy()
        df["fips"] = df["state_fips"] + df["county_fips"]
        return df[["fips", "Description", "Year", "Employment", "Wages", "Average Wage", "Average Weekly Wage"]].drop_duplicates(["fips", "Year"])

    def read_cew_employment_wages_cached(source_path: str | Path, *, cache_dir: str | Path = CACHE_DIR) -> pd.DataFrame:
        cached = _read_cache_or_none(source_path, cache_dir, "county_employment_wages")
        if cached is not None:
            return cached
        df = pd.read_csv(
            source_path,
            usecols=["IBRC_GEO_ID", "Year", "Employment", "Wages", "Average Wage", "Ownership Code Description", "NAICS Code"],
            low_memory=False,
        )
        df["county_fips"] = df["IBRC_GEO_ID"].astype(str).str.replace(".0", "", regex=False).str.zfill(5)
        df["cew_year"] = pd.to_numeric(df["Year"], errors="coerce")
        df["employment"] = pd.to_numeric(df["Employment"], errors="coerce")
        df["wages"] = pd.to_numeric(df["Wages"], errors="coerce")
        df["average_wage"] = pd.to_numeric(df["Average Wage"], errors="coerce")
        df["ownership_desc"] = df["Ownership Code Description"].astype(str).str.strip()
        df["naics_code"] = df["NAICS Code"].astype(str).str.strip()
        df = df[
            df["county_fips"].str.len().eq(5)
            & (~df["county_fips"].eq("00000"))
            & df["ownership_desc"].eq("All")
            & df["naics_code"].isin(["00", "0"])
        ].copy()
        df["average_wage_per_job"] = df["average_wage"].where(df["average_wage"].notna(), df["wages"] / df["employment"].replace(0, pd.NA))
        return df[["county_fips", "cew_year", "employment", "average_wage_per_job"]].drop_duplicates(["county_fips", "cew_year"])


EXCLUDED_INCIDENT_TYPES = {
    "Biological",
    "Chemical",
    "Other",
    "Human Cause",
    "Terrorist",
    "Fishing Losses",
    "Dam/Levee Break",
    "Toxic Substances",
}
REQUESTED_HOUSING_METRICS = [
    "AVG_SALE_TO_LIST",
    "HOMES_SOLD",
    "INVENTORY",
    "MEDIAN_DOM",
    "MEDIAN_LIST_PPSF",
    "MEDIAN_PPSF",
    "MONTHS_OF_SUPPLY",
    "PENDING_SALES",
    "PRICE_DROPS",
]
YOY_12_TO_24_METRICS = {
    "HOUSING_MARKET_INDEX": "HOUSING_MARKET_INDEX_MOM",
    "AVG_SALE_TO_LIST": "AVG_SALE_TO_LIST_YOY_MOM",
    "HOMES_SOLD": "HOMES_SOLD_YOY_MOM",
    "INVENTORY": "INVENTORY_YOY_MOM",
    "MEDIAN_PPSF": "MEDIAN_PPSF_YOY_MOM",
}
YOY_12_TO_24_DROP_COLUMNS = [
    "employment",
    "average_wage_per_job",
    "employment_bin",
    "average_wage_per_job_bin",
    "nri_risk_score",
    "nri_risk_rating",
    "nri_risk_rating_date",
]
YOY_12_TO_24_BIN_LABELS = {
    0: "significant_decrease",
    1: "moderate_to_no_change",
    2: "significant_increase",
}
INDEX_HOUSING_COLUMNS = [
    "fips",
    "REGION",
    "county_name",
    "STATE_CODE",
    "MONTH",
    "housing_year",
    "PERIOD_BEGIN",
    "PERIOD_END",
    "per_capita_income",
    "per_capita_income_bin",
    "employment",
    "average_wage_per_job",
    *REQUESTED_HOUSING_METRICS,
    "MEDIAN_PPSF_YOY",
    "AVG_SALE_TO_LIST_YOY",
    "HOMES_SOLD_YOY",
    "INVENTORY_YOY",
    "HOUSING_MARKET_INDEX",
    "HOUSING_MARKET_INDEX_MOM",
    "month_offset_from_incident",
    "incident_num",
    "incident_type",
    "county_profile",
    "nri_risk_rating",
]
INDEX_24M_COLUMNS = [
    "fips",
    "REGION",
    "county_name",
    "STATE_CODE",
    "MONTH",
    "housing_year",
    "PERIOD_BEGIN",
    "PERIOD_END",
    "per_capita_income",
    "per_capita_income_bin",
    *REQUESTED_HOUSING_METRICS,
    "MEDIAN_PPSF_YOY",
    "AVG_SALE_TO_LIST_YOY",
    "HOMES_SOLD_YOY",
    "INVENTORY_YOY",
    "HOUSING_MARKET_INDEX",
    *[f"{metric}_YOY_MOM" for metric in REQUESTED_HOUSING_METRICS],
    "HOUSING_MARKET_INDEX_MOM",
    "month_offset_from_incident",
    "incident_num",
    "incident_type",
    "county_profile",
]
STORY_1_COLUMNS = [
    "fips",
    "REGION",
    "county_name",
    "MONTH",
    "PERIOD_BEGIN",
    "per_capita_income",
    "per_capita_income_bin",
    "population_growth_yoy",
    "HOUSING_MARKET_INDEX",
    "HOUSING_MARKET_INDEX_MOM",
    "month_offset_from_incident",
    "incident_num",
    "county_profile",
    "nri_risk_rating",
    "nri_risk_rating_date",
]
STORY_2_COLUMNS = [
    "fips",
    "REGION",
    "county_name",
    "MONTH",
    "housing_year",
    "PERIOD_BEGIN",
    "per_capita_income",
    "per_capita_income_bin",
    "HOUSING_MARKET_INDEX",
    "HOUSING_MARKET_INDEX_MOM",
    "month_offset_from_incident",
    "incident_num",
]
STORY_5_COLUMNS = [
    "fips",
    "county_name",
    "STATE_CODE",
    "incident_type",
    "income_group",
    "income_group_label",
    "avg_per_capita_income",
    "income_bin_min",
    "income_bin_max",
    "month_offset_from_incident",
    "weighted_housing_market_index_mom",
    "weighted_housing_market_index",
    "incident_count",
    "total_weight",
    "latest_incident_year",
]
CLUSTER_COLUMNS = [
    "median_ppsf_response_cluster",
    "median_ppsf_response_cluster_name",
    "median_ppsf_response_cluster_interpretation",
    "median_ppsf_response_cluster_algorithm",
    "median_ppsf_response_cluster_k",
    "median_ppsf_response_cluster_silhouette",
    "median_ppsf_response_incident_count",
]
COUNTY_SUMMARY_METRICS = {
    "HOUSING_MARKET_INDEX": "HOUSING_MARKET_INDEX_change_in_yoy_12_to_24",
    "AVG_SALE_TO_LIST": "AVG_SALE_TO_LIST_change_in_yoy_12_to_24",
    "HOMES_SOLD": "HOMES_SOLD_change_in_yoy_12_to_24",
    "INVENTORY": "INVENTORY_change_in_yoy_12_to_24",
    "MEDIAN_PPSF": "MEDIAN_PPSF_change_in_yoy_12_to_24",
}


_data_cache: dict[str, object] = {}
_server: ThreadingHTTPServer | None = None


def _add_county_fips(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Statefips"] = pd.to_numeric(out["Statefips"], errors="coerce")
    out["Countyfips"] = pd.to_numeric(out["Countyfips"], errors="coerce")
    out = out.dropna(subset=["Statefips", "Countyfips"])
    out["state_fips"] = out["Statefips"].astype(int).astype(str).str.zfill(2)
    out["county_fips"] = out["Countyfips"].astype(int).astype(str).str.zfill(3)
    out = out[(out["state_fips"] != "00") & (out["county_fips"] != "000")].copy()
    out["fips"] = out["state_fips"] + out["county_fips"]
    return out


def _dedupe_by_key(df: pd.DataFrame, key_cols: list[str], strategy: str = "first") -> pd.DataFrame:
    if not df.duplicated(key_cols).any():
        return df
    if strategy == "largest_magnitude":
        numeric_cols = [col for col in df.columns if col not in key_cols and pd.api.types.is_numeric_dtype(df[col])]
        score = df[numeric_cols].fillna(0).abs().sum(axis=1)
        return (
            df.assign(_score=score)
            .sort_values(key_cols + ["_score"], ascending=[True] * len(key_cols) + [False])
            .drop_duplicates(key_cols)
            .drop(columns="_score")
        )
    if strategy == "count_over_estimate":
        rank = df["Count or Estimate"].map({"Count": 0, "Estimate": 1}).fillna(2)
        return df.assign(_rank=rank).sort_values(key_cols + ["_rank"]).drop_duplicates(key_cols).drop(columns="_rank")
    return df.sort_values(key_cols).drop_duplicates(key_cols)


def _clean_strict(text: object) -> str:
    if pd.isna(text):
        return ""
    text = str(text).lower().split(",")[0].replace("city county", "city")
    text = re.sub(r"[^a-z0-9 ]", "", text)
    return " ".join(text.split())


def _clean_relaxed(text: object) -> str:
    if pd.isna(text):
        return ""
    text = str(text).lower().split(",")[0]
    text = re.sub(r"\b(county|parish|city|borough|municipality|census area|and|city county)\b", "", text)
    text = re.sub(r"[^a-z0-9 ]", "", text)
    return " ".join(text.split())


def load_raw_inputs() -> dict[str, pd.DataFrame]:
    if "raw_inputs" in _data_cache:
        return _data_cache["raw_inputs"]  # type: ignore[return-value]

    natural_disasters_df = read_fema_disasters_cached(
        ROOT / "data" / "climate" / "FEMA_Disaster_Declarations.csv",
        cache_dir=CACHE_DIR,
    )
    nri_county_df = pd.read_csv(ROOT / "data" / "climate" / "NRI_Table_Counties.csv")
    housing_county_df = read_redfin_county_cached(
        ROOT / "data" / "housing" / "Redfin-Housing-Market-By-County.csv",
        cache_dir=CACHE_DIR,
    )
    fips_df = pd.read_csv(ROOT / "data" / "geographic" / "fips_master_v2.csv")
    personal_income_df = pd.read_csv(
        ECONOMIC_DIR / "BEA - US, States, Counties - Personal Income.csv",
        usecols=["IBRC_GEO_ID", "Year", "Data", "Linecode Description"],
        low_memory=False,
    )

    nri_county_df = (
        nri_county_df.assign(
            fips=nri_county_df["STCOFIPS"].astype(str).str.zfill(5),
            nri_risk_score=pd.to_numeric(nri_county_df["RISK_SCORE"], errors="coerce"),
            nri_risk_rating=nri_county_df["RISK_RATNG"],
            nri_risk_rating_date=nri_county_df["NRI_VER"],
        )[["fips", "nri_risk_score", "nri_risk_rating", "nri_risk_rating_date"]]
        .drop_duplicates(subset=["fips"])
    )

    raw_inputs = {
        "natural_disasters_df": natural_disasters_df,
        "nri_county_df": nri_county_df,
        "housing_county_df": housing_county_df,
        "fips_df": fips_df,
        "personal_income_df": personal_income_df,
    }
    _data_cache["raw_inputs"] = raw_inputs
    return raw_inputs


def load_profile_inputs() -> dict[str, pd.DataFrame]:
    if "profile_inputs" in _data_cache:
        return _data_cache["profile_inputs"]  # type: ignore[return-value]

    bea_income_path = ECONOMIC_DIR / "BEA - US, States, Counties - Personal Income.csv"
    cew_total_path = ECONOMIC_DIR / "CEW - US, States, Counties - Total Ownership.csv"
    pop_change_path = POPULATION_DIR / "Components of Population Change - U.S., States, and Counties.csv"
    pop_age_sex_path = POPULATION_DIR / "Population by Age and Sex - US, States, Counties.csv"
    pop_race_path = POPULATION_DIR / "Population by Race - US, States, Counties.csv"
    pop_estimates_path = POPULATION_DIR / "Population Estimates - U.S., States, and Counties.csv"

    bea_income_raw = pd.read_csv(
        bea_income_path,
        usecols=["Statefips", "Countyfips", "Description", "Year", "Linecode", "Linecode Description", "Data"],
        low_memory=False,
    )
    bea_income_df = _add_county_fips(bea_income_raw)
    bea_income_df = bea_income_df[bea_income_df["Linecode"].isin(BEA_LINECODES)].copy()
    bea_income_df["metric"] = bea_income_df["Linecode"].map(BEA_LINECODES)

    profile_inputs = {
        "bea_income_df": bea_income_df,
        "cew_total_df": read_cew_totals_cached(cew_total_path, cache_dir=CACHE_DIR),
        "population_change_df": _dedupe_by_key(_add_county_fips(pd.read_csv(pop_change_path, low_memory=False)), ["fips", "Year"], "largest_magnitude"),
        "population_age_sex_df": _dedupe_by_key(_add_county_fips(pd.read_csv(pop_age_sex_path, low_memory=False)), ["fips", "Year"]),
        "population_race_df": _dedupe_by_key(_add_county_fips(pd.read_csv(pop_race_path, low_memory=False)), ["fips", "Year"]),
    }
    profile_inputs["population_estimates_df"] = load_population_estimates_df()

    _data_cache["profile_inputs"] = profile_inputs
    return profile_inputs


def load_population_estimates_df() -> pd.DataFrame:
    if "population_estimates_df" in _data_cache:
        return _data_cache["population_estimates_df"]  # type: ignore[return-value]
    pop_estimates_path = POPULATION_DIR / "Population Estimates - U.S., States, and Counties.csv"
    population_estimates_df = _add_county_fips(pd.read_csv(pop_estimates_path, low_memory=False))
    population_estimates_df = population_estimates_df[population_estimates_df["State or County Release"].eq("County")].copy()
    population_estimates_df = _dedupe_by_key(
        population_estimates_df,
        ["fips", "Year"],
        "count_over_estimate",
    )
    _data_cache["population_estimates_df"] = population_estimates_df
    return population_estimates_df


def prepare_natural_disasters_df() -> pd.DataFrame:
    if "natural_disasters_df" in _data_cache:
        return _data_cache["natural_disasters_df"]  # type: ignore[return-value]

    raw = load_raw_inputs()
    natural_disasters_df = raw["natural_disasters_df"].apply(lambda col: col.str.strip() if col.dtype == "object" else col)
    fips_df = raw["fips_df"].apply(lambda col: col.str.strip() if col.dtype == "object" else col)
    fips_df["fips"] = fips_df["fips"].astype(str).str.zfill(5)
    kalawao_alias = fips_df.loc[fips_df["fips"] == "15009"].copy()
    kalawao_alias["fips"] = "15005"
    kalawao_alias["county_name"] = "Kalawao County"
    fips_df = pd.concat([fips_df, kalawao_alias], ignore_index=True)

    natural_disasters_df["fipsStateCode"] = natural_disasters_df["fipsStateCode"].astype(str).str.zfill(2)
    natural_disasters_df["fipsCountyCode"] = natural_disasters_df["fipsCountyCode"].astype(str).str.zfill(3)
    natural_disasters_df["fips_code_full"] = natural_disasters_df["fipsStateCode"] + natural_disasters_df["fipsCountyCode"]
    for col in ["declarationDate", "incidentBeginDate", "incidentEndDate"]:
        natural_disasters_df[col] = pd.to_datetime(natural_disasters_df[col])

    natural_disasters_df = pd.merge(
        natural_disasters_df,
        fips_df,
        how="left",
        left_on="fips_code_full",
        right_on="fips",
    )
    natural_disasters_df = natural_disasters_df.drop(columns=["state_x", "fips_code_full"], errors="ignore")
    natural_disasters_df = natural_disasters_df.rename(columns={"state_y": "state"})
    natural_disasters_df = natural_disasters_df[
        [
            "fips",
            "county_name",
            "state",
            "disasterNumber",
            "declarationDate",
            "incidentType",
            "declarationTitle",
            "ihProgramDeclared",
            "iaProgramDeclared",
            "paProgramDeclared",
            "hmProgramDeclared",
            "incidentBeginDate",
            "incidentEndDate",
            "designatedArea",
            "designatedIncidentTypes",
        ]
    ]
    _data_cache["natural_disasters_df"] = natural_disasters_df
    _data_cache["fips_df_prepared"] = fips_df
    return natural_disasters_df


def _prepare_housing_county_df() -> pd.DataFrame:
    if "housing_county_df" in _data_cache:
        return _data_cache["housing_county_df"]  # type: ignore[return-value]

    raw = load_raw_inputs()
    population_estimates_df = load_population_estimates_df()
    fips_df = _data_cache.get("fips_df_prepared")
    if fips_df is None:
        prepare_natural_disasters_df()
        fips_df = _data_cache["fips_df_prepared"]
    fips_df = fips_df.copy()  # type: ignore[union-attr]

    housing_county_df = raw["housing_county_df"].apply(lambda col: col.str.strip() if col.dtype == "object" else col)
    housing_county_df["PERIOD_BEGIN"] = pd.to_datetime(housing_county_df["PERIOD_BEGIN"])
    housing_county_df["PERIOD_END"] = pd.to_datetime(housing_county_df["PERIOD_END"])
    housing_county_df["MONTH"] = housing_county_df["PERIOD_BEGIN"].dt.to_period("M")
    housing_county_df = housing_county_df[housing_county_df["PROPERTY_TYPE"] == "All Residential"].copy()

    fips_df["clean_strict"] = fips_df["county_name"].apply(_clean_strict)
    fips_df["clean_relaxed"] = fips_df["county_name"].apply(_clean_relaxed)
    fips_df["clean_msa"] = fips_df["msa_name"].apply(_clean_relaxed)
    fips_df["clean_csa"] = fips_df["csa_name"].apply(_clean_relaxed)
    housing_county_df["clean_strict"] = housing_county_df["REGION"].apply(_clean_strict)
    housing_county_df["clean_relaxed"] = housing_county_df["REGION"].apply(_clean_relaxed)
    housing_county_df["clean_parent_metro"] = housing_county_df["PARENT_METRO_REGION"].apply(_clean_relaxed)
    fips_cols = ["fips", "county_name", "state", "msa_name", "csa_name"]

    stage1 = pd.merge(housing_county_df, fips_df[fips_cols + ["clean_strict"]], left_on=["clean_strict", "STATE_CODE"], right_on=["clean_strict", "state"], how="left")
    matched = stage1[stage1["fips"].notna()].copy()
    unmatched = stage1[stage1["fips"].isna()].drop(columns=[c for c in fips_cols if c in stage1.columns], errors="ignore")
    stage2 = pd.merge(unmatched, fips_df[fips_cols + ["clean_relaxed"]], left_on=["clean_relaxed", "STATE_CODE"], right_on=["clean_relaxed", "state"], how="left")
    matched = pd.concat([matched, stage2[stage2["fips"].notna()]])
    unmatched = stage2[stage2["fips"].isna()].drop(columns=[c for c in fips_cols if c in stage2.columns], errors="ignore")
    stage3 = pd.merge(unmatched, fips_df[fips_cols + ["clean_msa"]], left_on=["clean_parent_metro", "STATE_CODE"], right_on=["clean_msa", "state"], how="left")
    matched = pd.concat([matched, stage3[stage3["fips"].notna()]])
    unmatched = stage3[stage3["fips"].isna()].drop(columns=[c for c in fips_cols if c in stage3.columns], errors="ignore")
    stage4 = pd.merge(unmatched, fips_df[fips_cols + ["clean_csa"]], left_on=["clean_parent_metro", "STATE_CODE"], right_on=["clean_csa", "state"], how="left")
    housing_county_df = pd.concat([matched, stage4], ignore_index=True)
    housing_county_df = housing_county_df.drop(columns=["clean_strict", "clean_relaxed", "clean_parent_metro", "state"], errors="ignore")
    housing_county_df.loc[housing_county_df["REGION"] == "Maui County, HI", "fips"] = "15009"
    housing_county_df.loc[housing_county_df["REGION"] == "La Salle Parish, LA", "fips"] = "22059"
    housing_county_df["fips"] = housing_county_df["fips"].astype(str).str.zfill(5)
    housing_county_df["housing_year"] = housing_county_df["PERIOD_BEGIN"].dt.year.astype("Int64")

    personal_income_county_df = (
        raw["personal_income_df"]
        .assign(
            county_fips=raw["personal_income_df"]["IBRC_GEO_ID"].astype(str).str.replace(".0", "", regex=False).str.zfill(5),
            Year=pd.to_numeric(raw["personal_income_df"]["Year"], errors="coerce").astype("Int64"),
            Data=pd.to_numeric(raw["personal_income_df"]["Data"], errors="coerce"),
            line_desc=raw["personal_income_df"]["Linecode Description"].astype(str).str.strip(),
        )
        .loc[
            lambda df: df["county_fips"].str.len().eq(5)
            & (~df["county_fips"].eq("00000"))
            & (df["line_desc"] == "Per capita personal income (dollars)"),
            ["county_fips", "Year", "Data"],
        ]
        .rename(columns={"Year": "bea_year", "Data": "per_capita_income"})
        .drop_duplicates(subset=["county_fips", "bea_year"])
    )
    employment_wages_county_df = read_cew_employment_wages_cached(
        ECONOMIC_DIR / "CEW - US, States, Counties - Total Ownership.csv",
        cache_dir=CACHE_DIR,
    )
    population_growth_county_df = (
        population_estimates_df[["fips", "Year", "Population"]]
        .assign(
            population_year=lambda df: pd.to_numeric(df["Year"], errors="coerce").astype("Int64"),
            population=lambda df: pd.to_numeric(df["Population"], errors="coerce"),
        )
        .sort_values(["fips", "population_year"])
        .drop_duplicates(subset=["fips", "population_year"])
    )
    population_growth_county_df["population_growth_yoy"] = population_growth_county_df.groupby("fips")["population"].pct_change()
    population_growth_county_df = population_growth_county_df[["fips", "population_year", "population", "population_growth_yoy"]]

    housing_county_df["bea_year"] = housing_county_df["housing_year"].clip(upper=int(personal_income_county_df["bea_year"].max()))
    housing_county_df["cew_year"] = housing_county_df["housing_year"].clip(upper=int(employment_wages_county_df["cew_year"].max()))
    housing_county_df["population_year"] = housing_county_df["housing_year"].clip(upper=int(population_growth_county_df["population_year"].max()))
    housing_county_df = (
        housing_county_df.merge(personal_income_county_df, left_on=["fips", "bea_year"], right_on=["county_fips", "bea_year"], how="left")
        .drop(columns=["county_fips"])
        .merge(employment_wages_county_df, left_on=["fips", "cew_year"], right_on=["county_fips", "cew_year"], how="left")
        .drop(columns=["county_fips"])
        .merge(population_growth_county_df, on=["fips", "population_year"], how="left")
    )
    housing_county_df = housing_county_df[
        [
            "fips",
            "REGION",
            "county_name",
            "STATE_CODE",
            "MONTH",
            "housing_year",
            "PERIOD_BEGIN",
            "PERIOD_END",
            "per_capita_income",
            "employment",
            "average_wage_per_job",
            "population",
            "population_growth_yoy",
            "MEDIAN_PPSF",
            "MEDIAN_PPSF_YOY",
            "MEDIAN_LIST_PPSF",
            "MEDIAN_LIST_PPSF_YOY",
            "HOMES_SOLD",
            "HOMES_SOLD_YOY",
            "PENDING_SALES",
            "PENDING_SALES_YOY",
            "INVENTORY",
            "INVENTORY_YOY",
            "MONTHS_OF_SUPPLY",
            "MONTHS_OF_SUPPLY_YOY",
            "MEDIAN_DOM",
            "MEDIAN_DOM_YOY",
            "AVG_SALE_TO_LIST",
            "AVG_SALE_TO_LIST_YOY",
            "PRICE_DROPS",
            "PRICE_DROPS_YOY",
        ]
    ]
    _data_cache["housing_county_df"] = housing_county_df
    return housing_county_df


def prepare_housing_df(*, include_profiles: bool = True) -> pd.DataFrame:
    cache_key = "housing_df_with_profiles" if include_profiles else "housing_df"
    if cache_key in _data_cache:
        return _data_cache[cache_key]  # type: ignore[return-value]

    housing_df = _prepare_housing_county_df().copy().sort_values(["fips", "MONTH"])
    components = ["MEDIAN_PPSF_YOY", "AVG_SALE_TO_LIST_YOY", "HOMES_SOLD_YOY", "INVENTORY_YOY"]
    inverted_components = {"INVENTORY_YOY"}
    z_cols = []
    for component_col in components:
        z_col = f"{component_col}_Z"
        values = pd.to_numeric(housing_df[component_col], errors="coerce")
        std = values.std(skipna=True)
        housing_df[z_col] = (values - values.mean(skipna=True)) / std if pd.notna(std) and std else pd.NA
        if component_col in inverted_components:
            housing_df[z_col] = -housing_df[z_col]
        z_cols.append(z_col)
    housing_df["HOUSING_MARKET_INDEX"] = housing_df[z_cols].mean(axis=1, skipna=True)
    housing_df["HOUSING_MARKET_INDEX_MOM"] = housing_df.groupby("fips")["HOUSING_MARKET_INDEX"].diff()
    housing_df = housing_df.drop(columns=z_cols)
    for metric in REQUESTED_HOUSING_METRICS:
        yoy_col = f"{metric}_YOY"
        if yoy_col in housing_df.columns:
            housing_df[f"{metric}_YOY_MOM"] = housing_df.groupby("fips")[yoy_col].diff()

    _data_cache["housing_df"] = housing_df
    if include_profiles:
        county_profile_df = build_county_profiles()["county_profile_df"]
        housing_df = (
            housing_df.drop(
                columns=[
                    "county_profile",
                    "county_profile_desc",
                    "county_profile_algorithm",
                    "county_profile_k",
                    "county_profile_silhouette",
                    "county_profile_calinski_harabasz",
                    "county_profile_davies_bouldin",
                    "county_profile_combined_metric_rank",
                ],
                errors="ignore",
            )
            .merge(county_profile_df.drop(columns="county_name"), on="fips", how="left")
        )
        _data_cache["housing_df_with_profiles"] = housing_df
    return housing_df


def build_county_profiles() -> dict[str, object]:
    if "county_profile_outputs" in _data_cache:
        return _data_cache["county_profile_outputs"]  # type: ignore[return-value]

    profile_inputs = load_profile_inputs()
    last_complete_year = pd.Timestamp.today().year - 1
    target_years = list(range(last_complete_year - 9, last_complete_year + 1))
    outputs = build_county_profile_clusters(
        **profile_inputs,
        target_years=target_years,
        output_dir=OUTPUT_DIR,
    )
    _data_cache["county_profile_outputs"] = outputs
    return outputs


def _add_quantile_bins(
    df: pd.DataFrame,
    source_col: str,
    target_col: str,
    bins: int,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    numeric_values = pd.to_numeric(df[source_col], errors="coerce")
    result = pd.Series(pd.NA, index=df.index, dtype="Int64")

    if group_cols:
        groups = df.groupby(group_cols, dropna=False).groups.values()
    else:
        groups = [df.index]

    for group_index in groups:
        group_values = numeric_values.loc[group_index]
        valid_mask = group_values.notna()
        if valid_mask.sum() == 0:
            continue
        try:
            binned = pd.qcut(group_values.loc[valid_mask], q=bins, labels=False, duplicates="drop")
        except ValueError:
            continue
        result.loc[group_values.loc[valid_mask].index] = (
            pd.Series(binned, index=group_values.loc[valid_mask].index).astype("Int64") + 1
        )

    df[target_col] = result
    return df


def _housing_lookup(housing_df: pd.DataFrame) -> dict[str, object]:
    housing_with_keys = housing_df.copy()
    housing_with_keys["fips_normalized"] = housing_with_keys["fips"].astype(str).str.zfill(5)
    housing_with_keys["state_prefix"] = housing_with_keys["fips_normalized"].str[:2]
    housing_with_keys["is_county"] = ~housing_with_keys["fips_normalized"].str.endswith("000")
    housing_with_keys = _add_quantile_bins(
        housing_with_keys,
        "per_capita_income",
        "per_capita_income_bin",
        3,
        group_cols=["housing_year"],
    )
    housing_with_keys = _add_quantile_bins(housing_with_keys, "employment", "employment_bin", 5)
    housing_with_keys = _add_quantile_bins(housing_with_keys, "average_wage_per_job", "average_wage_per_job_bin", 5)
    county_rows = housing_with_keys[housing_with_keys["is_county"]].copy()
    return {
        "county_rows": county_rows,
        "county_months": county_rows.groupby("fips_normalized", sort=False)["MONTH"].agg(lambda values: set(values)),
        "state_months": county_rows.groupby("state_prefix", sort=False)["MONTH"].agg(lambda values: set(values)),
        "empty": housing_with_keys.iloc[0:0].copy(),
    }


def _prepare_incident_df(natural_disasters_df: pd.DataFrame, incident_type: str | None = None) -> pd.DataFrame:
    if incident_type in EXCLUDED_INCIDENT_TYPES:
        raise ValueError("incident_type is not valid.")
    if incident_type is not None and not (natural_disasters_df["incidentType"] == incident_type).any():
        raise ValueError("incident_type is not valid.")
    cutoff_date = pd.Timestamp(year=pd.Timestamp.now().year - 10, month=1, day=1)
    incident_df = natural_disasters_df.loc[natural_disasters_df["incidentBeginDate"] >= cutoff_date].copy()
    incident_df = incident_df.loc[~incident_df["incidentType"].isin(EXCLUDED_INCIDENT_TYPES)].copy()
    if incident_type is not None:
        incident_df = incident_df.loc[incident_df["incidentType"] == incident_type].copy()
    duration = incident_df["incidentEndDate"] - incident_df["incidentBeginDate"]
    median_duration = duration[duration.notna()].median()
    incident_df["incidentEndDate"] = incident_df["incidentEndDate"].fillna(incident_df["incidentBeginDate"] + median_duration)
    incident_df["incident_begin_month"] = incident_df["incidentBeginDate"].dt.to_period("M")
    incident_df["incident_end_month"] = incident_df["incidentEndDate"].dt.to_period("M")
    return incident_df


def build_incident_housing_subset(
    natural_disasters_df: pd.DataFrame,
    housing_df: pd.DataFrame,
    *,
    incident_type: str,
    months_before: int,
    months_after: int,
    complete_after_anchor: bool = False,
) -> pd.DataFrame:
    incident_df = _prepare_incident_df(natural_disasters_df, incident_type=incident_type)
    lookup = _housing_lookup(housing_df)
    county_rows = lookup["county_rows"]
    county_months = lookup["county_months"]
    state_months = lookup["state_months"]
    event_month_rows = []

    for incident_num, event in enumerate(incident_df.itertuples(index=False), start=1):
        fips = str(event.fips).zfill(5)
        event_start = event.incident_begin_month
        event_end = event.incident_end_month
        available_months = state_months.get(fips[:2]) if fips.endswith("000") else county_months.get(fips)
        if available_months is None:
            continue
        if complete_after_anchor:
            required_after_months = {event_end + offset for offset in range(1, months_after + 1)}
            if not required_after_months.issubset(available_months):
                continue
        offset_lookup = {event_start + offset: offset for offset in range(-months_before, 0)}
        offset_lookup[event_end] = 0
        offset_lookup.update({event_end + offset: offset for offset in range(1, months_after + 1)})
        for month, offset in offset_lookup.items():
            event_month_rows.append(
                {
                    "event_fips": fips,
                    "event_state_prefix": fips[:2],
                    "is_statewide_event": fips.endswith("000"),
                    "MONTH": month,
                    "month_offset_from_incident": offset,
                    "incident_num": incident_num,
                    "incident_type": event.incidentType,
                }
            )

    if not event_month_rows:
        result = lookup["empty"].copy()
        result["month_offset_from_incident"] = pd.Series(dtype="Int64")
        result["incident_num"] = pd.Series(dtype="Int64")
        result["incident_type"] = pd.Series(dtype="object")
    else:
        event_months = pd.DataFrame(event_month_rows)
        matched_frames = []
        county_events = event_months.loc[~event_months["is_statewide_event"]]
        state_events = event_months.loc[event_months["is_statewide_event"]]
        if not county_events.empty:
            matched_frames.append(county_rows.merge(county_events, left_on=["fips_normalized", "MONTH"], right_on=["event_fips", "MONTH"], how="inner"))
        if not state_events.empty:
            matched_frames.append(county_rows.merge(state_events, left_on=["state_prefix", "MONTH"], right_on=["event_state_prefix", "MONTH"], how="inner"))
        result = pd.concat(matched_frames, ignore_index=True) if matched_frames else lookup["empty"].copy()

    result = result.drop(
        columns=["fips_normalized", "state_prefix", "is_county", "event_fips", "event_state_prefix", "is_statewide_event"],
        errors="ignore",
    )
    return result.merge(load_raw_inputs()["nri_county_df"], on="fips", how="left")


def _existing_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in df.columns]


def _write_csv_subset(df: pd.DataFrame, path: Path, columns: list[str]) -> None:
    subset = df.loc[:, _existing_columns(df, columns)].copy()
    if "fips" in subset.columns:
        subset["fips"] = subset["fips"].astype(str).str.zfill(5)
    subset.to_csv(path, index=False)


def _is_county_fips(series: pd.Series) -> pd.Series:
    fips = series.astype(str).str.zfill(5)
    return fips.str.len().eq(5) & fips.str.slice(2).ne("000")


def _weighted_incident_average(incidents: list[dict[str, float]]) -> tuple[float | None, int]:
    if not incidents:
        return None, 0
    incidents = sorted(incidents, key=lambda item: item["incident_num"])
    total_weight = len(incidents) * (len(incidents) + 1) / 2
    value = sum((idx + 1) * item["value"] for idx, item in enumerate(incidents)) / total_weight
    return value, len(incidents)


def _story_5_county_income_groups(housing_df: pd.DataFrame, county_fips: set[str] | None = None) -> pd.DataFrame:
    county_income = housing_df.copy()
    county_income["fips"] = county_income["fips"].astype(str).str.zfill(5)
    county_income = county_income.loc[_is_county_fips(county_income["fips"])].copy()
    county_income["per_capita_income"] = pd.to_numeric(county_income["per_capita_income"], errors="coerce")
    county_income = (
        county_income.dropna(subset=["per_capita_income"])
        .groupby("fips", as_index=False)
        .agg(
            county_name=("REGION", "last"),
            STATE_CODE=("STATE_CODE", "last"),
            avg_per_capita_income=("per_capita_income", "mean"),
        )
    )
    if county_fips is not None:
        normalized_fips = {str(fips).zfill(5) for fips in county_fips}
        county_income = county_income.loc[county_income["fips"].isin(normalized_fips)].copy()
    if county_income.empty:
        county_income["income_group"] = pd.Series(dtype="Int64")
        county_income["income_group_label"] = pd.Series(dtype="object")
        county_income["income_bin_min"] = pd.Series(dtype="float64")
        county_income["income_bin_max"] = pd.Series(dtype="float64")
        return county_income

    ranked_income = county_income["avg_per_capita_income"].rank(method="first")
    if len(county_income) == 1:
        county_income["income_group"] = pd.Series([2], index=county_income.index, dtype="Int64")
    elif len(county_income) == 2:
        county_income["income_group"] = ranked_income.map({1.0: 1, 2.0: 3}).astype("Int64")
    else:
        county_income["income_group"] = pd.qcut(ranked_income, q=3, labels=[1, 2, 3]).astype("Int64")
    labels = {1: "Lower income", 2: "Middle income", 3: "Higher income"}
    county_income["income_group_label"] = county_income["income_group"].map(labels)
    ranges = county_income.groupby("income_group")["avg_per_capita_income"].agg(["min", "max"]).rename(
        columns={"min": "income_bin_min", "max": "income_bin_max"}
    )
    return county_income.merge(ranges, on="income_group", how="left")


def _story_5_incident_dates(natural_disasters_df: pd.DataFrame, incident_type: str) -> pd.DataFrame:
    incident_df = _prepare_incident_df(natural_disasters_df, incident_type=incident_type).copy()
    incident_df = incident_df.reset_index(drop=True)
    incident_df["incident_num"] = range(1, len(incident_df) + 1)
    incident_df["incident_begin_date"] = pd.to_datetime(incident_df["incidentBeginDate"], errors="coerce")
    incident_df["incident_year"] = incident_df["incident_begin_date"].dt.year
    return incident_df[["incident_num", "incident_begin_date", "incident_year"]]


def build_story_5_income_response_df(
    natural_disasters_df: pd.DataFrame,
    housing_df: pd.DataFrame,
    *,
    incident_type: str,
) -> pd.DataFrame:
    housing_subset = build_incident_housing_subset(
        natural_disasters_df,
        housing_df,
        incident_type=incident_type,
        months_before=12,
        months_after=24,
        complete_after_anchor=True,
    )
    if housing_subset.empty:
        return pd.DataFrame(columns=STORY_5_COLUMNS)

    subset_fips = set(housing_subset["fips"].astype(str).str.zfill(5))
    income_groups = _story_5_county_income_groups(housing_df, subset_fips)
    if income_groups.empty:
        return pd.DataFrame(columns=STORY_5_COLUMNS)

    incident_dates = _story_5_incident_dates(natural_disasters_df, incident_type)
    work = housing_subset.copy()
    work["fips"] = work["fips"].astype(str).str.zfill(5)
    work = work.loc[_is_county_fips(work["fips"])].copy()
    work["incident_num"] = pd.to_numeric(work["incident_num"], errors="coerce")
    work["month_offset_from_incident"] = pd.to_numeric(work["month_offset_from_incident"], errors="coerce")
    work["HOUSING_MARKET_INDEX_MOM"] = pd.to_numeric(work["HOUSING_MARKET_INDEX_MOM"], errors="coerce")
    work["HOUSING_MARKET_INDEX"] = pd.to_numeric(work["HOUSING_MARKET_INDEX"], errors="coerce")
    work = work.dropna(subset=["incident_num", "month_offset_from_incident"])
    work = work.merge(incident_dates, on="incident_num", how="left")
    work = work.merge(income_groups, on="fips", how="inner", suffixes=("", "_income"))
    work = work.loc[work["month_offset_from_incident"].between(-12, 24)].copy()

    rows = []
    for (fips, offset), group in work.groupby(["fips", "month_offset_from_incident"], dropna=False):
        group = group.sort_values(["incident_begin_date", "incident_num"]).copy()
        group["_weight"] = range(1, len(group) + 1)
        mom = group.dropna(subset=["HOUSING_MARKET_INDEX_MOM"])
        index = group.dropna(subset=["HOUSING_MARKET_INDEX"])
        weighted_mom = (
            (mom["HOUSING_MARKET_INDEX_MOM"] * mom["_weight"]).sum() / mom["_weight"].sum()
            if not mom.empty
            else pd.NA
        )
        weighted_index = (
            (index["HOUSING_MARKET_INDEX"] * index["_weight"]).sum() / index["_weight"].sum()
            if not index.empty
            else pd.NA
        )
        latest = group.iloc[-1]
        rows.append(
            {
                "fips": fips,
                "county_name": latest.get("county_name_income") or latest.get("county_name") or latest.get("REGION") or fips,
                "STATE_CODE": latest.get("STATE_CODE_income") or latest.get("STATE_CODE"),
                "incident_type": incident_type,
                "income_group": latest["income_group"],
                "income_group_label": latest["income_group_label"],
                "avg_per_capita_income": latest["avg_per_capita_income"],
                "income_bin_min": latest["income_bin_min"],
                "income_bin_max": latest["income_bin_max"],
                "month_offset_from_incident": int(offset),
                "weighted_housing_market_index_mom": weighted_mom,
                "weighted_housing_market_index": weighted_index,
                "incident_count": int(group["incident_num"].nunique()),
                "total_weight": float(group["_weight"].sum()),
                "latest_incident_year": latest.get("incident_year", pd.NA),
            }
        )
    return pd.DataFrame(rows, columns=STORY_5_COLUMNS)


def _build_yoy_summary_payload(df: pd.DataFrame) -> dict[str, object]:
    payload: dict[str, object] = {"metrics": {}}
    if df.empty:
        return payload
    work = df.copy()
    work["fips"] = work["fips"].astype(str).str.zfill(5)
    work = work.loc[_is_county_fips(work["fips"])]
    work["incident_num"] = pd.to_numeric(work["incident_num"], errors="coerce")
    work["month_offset_from_incident"] = pd.to_numeric(work["month_offset_from_incident"], errors="coerce")
    work = work.dropna(subset=["incident_num", "month_offset_from_incident"])
    period_configs = {"months_1_12": range(1, 13), "months_13_24": range(13, 25)}

    for metric in REQUESTED_HOUSING_METRICS:
        metric_col = f"{metric}_YOY_MOM"
        if metric_col not in work.columns:
            continue
        metric_payload = {}
        metric_work = work.assign(_metric_value=pd.to_numeric(work[metric_col], errors="coerce")).dropna(subset=["_metric_value"])
        for period_key, required_offsets in period_configs.items():
            required_offsets = list(required_offsets)
            period_payload = {"all": {}, "complete": {}}
            incidents_by_county: dict[str, list[dict[str, float]]] = {}
            complete_incidents_by_county: dict[str, list[dict[str, float]]] = {}
            for (fips, incident_num), group in metric_work.groupby(["fips", "incident_num"], dropna=False):
                period_rows = group.loc[group["month_offset_from_incident"].isin(required_offsets)]
                if period_rows.empty:
                    continue
                by_offset = period_rows.groupby("month_offset_from_incident")["_metric_value"].agg(["mean", "count"])
                incident = {"incident_num": float(incident_num), "value": float(by_offset["mean"].mean())}
                incidents_by_county.setdefault(fips, []).append(incident)
                if set(by_offset.index.astype(int)) == set(required_offsets) and bool((by_offset["count"] == 1).all()):
                    complete_incidents_by_county.setdefault(fips, []).append(incident)
            for fips, incidents in incidents_by_county.items():
                value, count = _weighted_incident_average(incidents)
                if value is not None:
                    period_payload["all"][fips] = {"value": value, "incidentCount": count}
            for fips, incidents in complete_incidents_by_county.items():
                value, count = _weighted_incident_average(incidents)
                if value is not None:
                    period_payload["complete"][fips] = {"value": value, "incidentCount": count}
            metric_payload[period_key] = period_payload
        payload["metrics"][metric] = metric_payload
    return payload


def _build_county_summary_df(df: pd.DataFrame) -> pd.DataFrame:
    fieldnames = ["fips", "county_name", "county_profile", *CLUSTER_COLUMNS]
    for metric in COUNTY_SUMMARY_METRICS:
        fieldnames.extend([f"{metric}_change_all", f"{metric}_incident_count_all", f"{metric}_change_complete", f"{metric}_incident_count_complete"])
    if df.empty:
        return pd.DataFrame(columns=fieldnames)

    work = df.copy()
    work["fips"] = work["fips"].astype(str).str.zfill(5)
    work = work.loc[_is_county_fips(work["fips"])]
    work["incident_num"] = pd.to_numeric(work["incident_num"], errors="coerce")
    work["month_offset_from_incident"] = pd.to_numeric(work["month_offset_from_incident"], errors="coerce")
    work = work.dropna(subset=["incident_num"])
    incident_summaries_by_county = {}
    for (fips, incident_num), group in work.groupby(["fips", "incident_num"], dropna=False):
        latest = group.iloc[-1]
        summary = {
            "fips": fips,
            "incident_num": float(incident_num),
            "county_name": latest.get("REGION") or latest.get("county_name") or fips,
            "county_profile": latest.get("county_profile", pd.NA),
            "cluster_fields": {
                column: latest.get(column, "")
                for column in CLUSTER_COLUMNS
                if column in work.columns and pd.notna(latest.get(column, pd.NA)) and latest.get(column, "") != ""
            },
            "metrics": {},
        }
        for metric, change_col in COUNTY_SUMMARY_METRICS.items():
            if change_col not in group.columns or metric not in group.columns:
                continue
            change_values = pd.to_numeric(group[change_col], errors="coerce").dropna()
            if change_values.empty:
                continue
            metric_values = pd.to_numeric(group[metric], errors="coerce")
            metric_offsets = set(group.loc[metric_values.notna(), "month_offset_from_incident"].dropna().astype(int))
            summary["metrics"][metric] = {
                "value": float(change_values.iloc[0]),
                "complete": set(range(-12, 0)).issubset(metric_offsets) and set(range(1, 25)).issubset(metric_offsets),
            }
        incident_summaries_by_county.setdefault(fips, []).append(summary)

    rows = []
    for fips, incidents in sorted(incident_summaries_by_county.items()):
        incidents = sorted(incidents, key=lambda item: item["incident_num"])
        row = {
            "fips": fips,
            "county_name": incidents[-1]["county_name"],
            "county_profile": incidents[-1]["county_profile"],
        }
        row.update(incidents[-1]["cluster_fields"])
        has_any_metric = False
        for metric in COUNTY_SUMMARY_METRICS:
            all_incidents = [{"incident_num": incident["incident_num"], "value": incident["metrics"][metric]["value"]} for incident in incidents if metric in incident["metrics"]]
            complete_incidents = [
                {"incident_num": incident["incident_num"], "value": incident["metrics"][metric]["value"]}
                for incident in incidents
                if metric in incident["metrics"] and incident["metrics"][metric]["complete"]
            ]
            all_value, all_count = _weighted_incident_average(all_incidents)
            complete_value, complete_count = _weighted_incident_average(complete_incidents)
            row[f"{metric}_change_all"] = "" if all_value is None else all_value
            row[f"{metric}_incident_count_all"] = all_count
            row[f"{metric}_change_complete"] = "" if complete_value is None else complete_value
            row[f"{metric}_incident_count_complete"] = complete_count
            has_any_metric = has_any_metric or all_count > 0 or complete_count > 0
        if has_any_metric:
            rows.append(row)
    return pd.DataFrame(rows, columns=fieldnames)


def _assign_three_way_change_bins(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    result = pd.Series(pd.NA, index=series.index, dtype="object")
    valid = values.dropna()
    if valid.empty:
        return result
    if valid.nunique() == 1:
        result.loc[valid.index] = YOY_12_TO_24_BIN_LABELS[1]
        return result
    try:
        codes = pd.qcut(valid, q=3, labels=[0, 1, 2], duplicates="drop")
    except ValueError:
        result.loc[valid.index] = YOY_12_TO_24_BIN_LABELS[1]
        return result
    if getattr(codes, "cat", None) is not None and len(codes.cat.categories) < 3:
        result.loc[valid.index] = YOY_12_TO_24_BIN_LABELS[1]
        return result
    result.loc[valid.index] = codes.astype("Int64").map(YOY_12_TO_24_BIN_LABELS)
    return result


def _add_change_in_yoy_12_to_24_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        for metric in YOY_12_TO_24_METRICS:
            base_col = f"{metric}_change_in_yoy_12_to_24"
            df[base_col] = pd.Series(dtype="float64")
            df[f"{base_col}_bin"] = pd.Series(dtype="object")
        return df
    out = df.copy()
    group_cols = ["fips", "incident_num"]
    for metric, yoy_mom_col in YOY_12_TO_24_METRICS.items():
        base_col = f"{metric}_change_in_yoy_12_to_24"
        if yoy_mom_col not in out.columns:
            out[base_col] = pd.NA
            out[f"{base_col}_bin"] = pd.NA
            continue
        grouped = (
            out.groupby(group_cols, dropna=False)
            .apply(
                lambda group: pd.Series(
                    {
                        "months_1_12": pd.to_numeric(group.loc[group["month_offset_from_incident"].between(1, 12), yoy_mom_col], errors="coerce").mean(),
                        "months_13_24": pd.to_numeric(group.loc[group["month_offset_from_incident"].between(13, 24), yoy_mom_col], errors="coerce").mean(),
                    }
                ),
                include_groups=False,
            )
            .reset_index()
        )
        grouped[base_col] = grouped["months_13_24"] - grouped["months_1_12"]
        grouped[f"{base_col}_bin"] = _assign_three_way_change_bins(grouped[base_col])
        out = out.merge(grouped[group_cols + [base_col, f"{base_col}_bin"]], on=group_cols, how="left")
    return out


def _incident_type_slug(incident_type: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(incident_type).strip().lower())
    return slug.strip("_") or "unknown"


def _incident_types(natural_disasters_df: pd.DataFrame) -> list[str]:
    prepared = _prepare_incident_df(natural_disasters_df)
    return sorted(prepared["incidentType"].dropna().unique())


def _existing_manifest_assets(slug: str) -> dict[str, str]:
    assets = {
        "story_1_housing_csv": f"{slug}_story_1_housing.csv",
        "story_2_housing_24mths_csv": f"{slug}_story_2_housing_24mths.csv",
        "story_5_income_response_csv": f"{slug}_story_5_income_response.csv",
        "story_4_pre_market_tiers_csv": f"{slug}_story_4_pre_market_tiers.csv",
        "index_housing_csv": f"{slug}_index_housing.csv",
        "housing_csv": f"{slug}_index_housing.csv",
        "index_housing_24mths_csv": f"{slug}_index_housing_24mths.csv",
        "housing_24mths_csv": f"{slug}_index_housing_24mths.csv",
        "index_yoy_summary_json": f"{slug}_index_housing_24mths_yoy_summary.json",
        "county_summary_csv": f"{slug}_county_summary.csv",
    }
    present = {key: name for key, name in assets.items() if (OUTPUT_DIR / name).exists()}
    if (OUTPUT_DIR / "ppsf_response_cluster_summaries.json").exists() and "county_summary_csv" in present:
        present["ppsf_response_cluster_summary_json"] = "ppsf_response_cluster_summaries.json"
    if (OUTPUT_DIR / "pre_market_strength_tier_summaries.json").exists() and "story_4_pre_market_tiers_csv" in present:
        present["pre_market_strength_tier_summary_json"] = "pre_market_strength_tier_summaries.json"
    return present


def _write_manifest(entries: list[dict[str, str]]) -> dict[str, object]:
    manifest_path = OUTPUT_DIR / "incident_housing_manifest.json"
    existing_by_slug: dict[str, dict[str, str]] = {}
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            existing_by_slug = {
                entry["slug"]: entry
                for entry in existing.get("incident_types", [])
                if isinstance(entry, dict) and entry.get("slug")
            }
        except json.JSONDecodeError:
            existing_by_slug = {}

    merged_entries = []
    for entry in entries:
        merged = {
            **_existing_manifest_assets(entry["slug"]),
            **existing_by_slug.get(entry["slug"], {}),
            **entry,
        }
        merged_entries.append(merged)

    default_incident_type = "Fire" if any(e["incident_type"] == "Fire" for e in entries) else (entries[0]["incident_type"] if entries else None)
    manifest = {"default_incident_type": default_incident_type, "incident_types": merged_entries}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _maybe_add_cluster_annotations(housing_subset_24m: pd.DataFrame, annotations: pd.DataFrame | None, incident_type: str) -> pd.DataFrame:
    if annotations is None or annotations.empty:
        return housing_subset_24m
    current = annotations.loc[annotations["incident_type"] == incident_type, ["fips"] + CLUSTER_COLUMNS].copy()
    if current.empty:
        return housing_subset_24m
    current["fips"] = current["fips"].astype(str).str.zfill(5)
    frame = housing_subset_24m.drop(columns=CLUSTER_COLUMNS, errors="ignore").copy()
    frame["fips"] = frame["fips"].astype(str).str.zfill(5)
    return frame.merge(current, on="fips", how="left")


def _write_cluster_summaries(interpretations_df: pd.DataFrame) -> None:
    if interpretations_df is None or interpretations_df.empty:
        return
    cluster_summaries = {}
    for incident, group in interpretations_df.groupby("incident_type"):
        first_row = group.iloc[0]
        cluster_summaries[incident] = {
            "incident_type": incident,
            "status": "clustered",
            "algorithm": first_row["algorithm"],
            "k": int(first_row["k"]),
            "silhouette_score": float(first_row["silhouette_score"]),
            "metrics_used": first_row.get("metrics_used", ", ".join(PPSF_RESPONSE_METRIC_LABELS.values())),
            "counties_clustered": int(group["counties"].sum()),
            "clusters": [
                {
                    "cluster": int(row.cluster),
                    "name": row.cluster_name,
                    "interpretation": row.interpretation,
                    "counties": int(row.counties),
                }
                for row in group.sort_values("cluster").itertuples(index=False)
            ],
        }
    (OUTPUT_DIR / "ppsf_response_cluster_summaries.json").write_text(json.dumps(cluster_summaries, indent=2), encoding="utf-8")


def export_page_data(page: str) -> dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    natural_disasters_df = prepare_natural_disasters_df()
    page = page.lower().strip()
    needs_profiles = page in {"index", "climate-on-housing", "story-1", "story-3"}
    housing_df = prepare_housing_df(include_profiles=needs_profiles)
    ppsf_annotations = None
    ppsf_interpretations = None
    pre_market_story_files = {}

    if page == "story-3":
        cluster_outputs = build_all_housing_market_response_clusters(
            natural_disasters_df=natural_disasters_df,
            housing_df=housing_df,
            output_dir=OUTPUT_DIR,
        )
        ppsf_annotations = cluster_outputs["ppsf_response_cluster_annotations_df"]
        ppsf_interpretations = cluster_outputs["ppsf_response_cluster_interpretations_df"]
        _write_cluster_summaries(ppsf_interpretations)
    if page == "story-4":
        tier_outputs = build_all_pre_incident_market_strength_tiers(
            natural_disasters_df=natural_disasters_df,
            housing_df=housing_df,
            output_dir=OUTPUT_DIR,
        )
        pre_market_story_files = tier_outputs["story_files"]

    manifest_entries = []
    written_paths = []
    for incident_type in _incident_types(natural_disasters_df):
        slug = _incident_type_slug(incident_type)
        entry = {"incident_type": incident_type, "slug": slug}

        if page in {"index", "climate-on-housing", "story-1"}:
            housing_subset = build_incident_housing_subset(
                natural_disasters_df,
                housing_df,
                incident_type=incident_type,
                months_before=12,
                months_after=12,
            )
            path = OUTPUT_DIR / f"{slug}_story_1_housing.csv"
            _write_csv_subset(housing_subset, path, STORY_1_COLUMNS)
            entry["story_1_housing_csv"] = path.name
            written_paths.append(path)
            if page in {"index", "climate-on-housing"}:
                index_path = OUTPUT_DIR / f"{slug}_index_housing.csv"
                _write_csv_subset(housing_subset, index_path, INDEX_HOUSING_COLUMNS)
                entry["index_housing_csv"] = index_path.name
                entry["housing_csv"] = index_path.name
                written_paths.append(index_path)

        if page in {"index", "climate-on-housing", "story-2", "story-3", "story-4"}:
            housing_subset_24m = build_incident_housing_subset(
                natural_disasters_df,
                housing_df,
                incident_type=incident_type,
                months_before=12,
                months_after=24,
                complete_after_anchor=True,
            )
            housing_subset_24m = _add_change_in_yoy_12_to_24_columns(housing_subset_24m)
            housing_subset_24m = housing_subset_24m.drop(columns=YOY_12_TO_24_DROP_COLUMNS, errors="ignore")
            housing_subset_24m = _maybe_add_cluster_annotations(housing_subset_24m, ppsf_annotations, incident_type)
            if page in {"story-2", "story-4"}:
                story_2_path = OUTPUT_DIR / f"{slug}_story_2_housing_24mths.csv"
                _write_csv_subset(housing_subset_24m, story_2_path, STORY_2_COLUMNS)
                entry["story_2_housing_24mths_csv"] = story_2_path.name
                written_paths.append(story_2_path)

            if page in {"index", "climate-on-housing"}:
                index_24m_path = OUTPUT_DIR / f"{slug}_index_housing_24mths.csv"
                yoy_summary_path = OUTPUT_DIR / f"{slug}_index_housing_24mths_yoy_summary.json"
                _write_csv_subset(housing_subset_24m, index_24m_path, INDEX_24M_COLUMNS)
                yoy_summary_path.write_text(json.dumps(_build_yoy_summary_payload(housing_subset_24m), separators=(",", ":")), encoding="utf-8")
                entry["index_housing_24mths_csv"] = index_24m_path.name
                entry["index_yoy_summary_json"] = yoy_summary_path.name
                entry["housing_24mths_csv"] = index_24m_path.name
                written_paths.extend([index_24m_path, yoy_summary_path])

            if page == "story-3":
                summary_path = OUTPUT_DIR / f"{slug}_county_summary.csv"
                _build_county_summary_df(housing_subset_24m).to_csv(summary_path, index=False)
                entry["county_summary_csv"] = summary_path.name
                entry["ppsf_response_cluster_summary_json"] = "ppsf_response_cluster_summaries.json"
                written_paths.append(summary_path)

        if page == "story-4":
            story_4_path = Path(pre_market_story_files.get(incident_type, OUTPUT_DIR / f"{slug}_story_4_pre_market_tiers.csv"))
            if story_4_path.exists():
                entry["story_4_pre_market_tiers_csv"] = story_4_path.name
                entry["pre_market_strength_tier_summary_json"] = "pre_market_strength_tier_summaries.json"
                written_paths.append(story_4_path)

        if page == "story-5":
            story_5_path = OUTPUT_DIR / f"{slug}_story_5_income_response.csv"
            story_5_df = build_story_5_income_response_df(
                natural_disasters_df,
                housing_df,
                incident_type=incident_type,
            )
            _write_csv_subset(story_5_df, story_5_path, STORY_5_COLUMNS)
            entry["story_5_income_response_csv"] = story_5_path.name
            written_paths.append(story_5_path)

        manifest_entries.append(entry)

    manifest = _write_manifest(manifest_entries)
    return {"page": page, "manifest": manifest, "written_paths": written_paths}


def _find_open_port(host: str = "127.0.0.1", start: int = 8000, end: int = 8100) -> int:
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex((host, port)) != 0:
                return port
    raise RuntimeError("Could not find an open port for the local visualization server.")


def serve_visualization(html_file: str, *, host: str = "127.0.0.1", port: int | None = None) -> str:
    global _server
    if _server is not None:
        _server.shutdown()
        _server.server_close()
    html_path = OUTPUT_DIR / html_file
    if not html_path.exists():
        raise FileNotFoundError(f"Visualization file not found: {html_path.resolve()}")
    selected_port = port or _find_open_port(host=host)
    handler = partial(SimpleHTTPRequestHandler, directory=str(OUTPUT_DIR.resolve()))
    _server = ThreadingHTTPServer((host, selected_port), handler)
    thread = threading.Thread(target=_server.serve_forever, daemon=True)
    thread.start()
    url = f"http://{host}:{selected_port}/{html_file}"
    print(f"Open {html_file} at: {url}")
    return url


def build_and_serve(page: str, html_file: str, *, host: str = "127.0.0.1", port: int | None = None) -> dict[str, object]:
    result = export_page_data(page)
    url = serve_visualization(html_file, host=host, port=port)
    result["url"] = url
    print(f"Wrote {len(result['written_paths'])} page data files plus incident_housing_manifest.json")
    return result
