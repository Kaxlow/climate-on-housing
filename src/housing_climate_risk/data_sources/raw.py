from __future__ import annotations

from pathlib import Path

import pandas as pd

from housing_climate_risk.data_sources.cached_io import (
    read_cew_employment_wages_cached,
    read_cew_totals_cached,
    read_fema_disasters_cached,
    read_redfin_county_cached,
)
from housing_climate_risk.modeling.county_profiles import BEA_LINECODES
from housing_climate_risk.paths import (
    CACHE_DIR,
    CLIMATE_DIR,
    ECONOMIC_DIR,
    FEMA_DIR,
    FIPSGEO_DIR,
    GEOGRAPHIC_DIR,
    HOUSING_DIR,
    POPULATION_DIR,
)


_CACHE: dict[str, object] = {}


def clear_cache() -> None:
    _CACHE.clear()


def add_county_fips(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["state_fips"] = pd.to_numeric(out["Statefips"], errors="coerce")
    out["county_fips"] = pd.to_numeric(out["Countyfips"], errors="coerce")
    out = out.dropna(subset=["state_fips", "county_fips"]).copy()
    out["fips"] = out["state_fips"].astype(int).astype(str).str.zfill(2) + out["county_fips"].astype(int).astype(str).str.zfill(3)
    return out[(out["fips"].str[:2] != "00") & (out["fips"].str[2:] != "000")].copy()


def dedupe_by_key(df: pd.DataFrame, key_cols: list[str], strategy: str = "first") -> pd.DataFrame:
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


def load_fema_disasters() -> pd.DataFrame:
    path = CLIMATE_DIR / "FEMA_Disaster_Declarations.csv"
    if not path.exists():
        path = FEMA_DIR / "FEMA_Disaster_Declarations.csv"
    return read_fema_disasters_cached(path, cache_dir=CACHE_DIR)


def load_nri_counties() -> pd.DataFrame:
    path = CLIMATE_DIR / "NRI_Table_Counties.csv"
    if not path.exists():
        path = FEMA_DIR / "NRI_Table_Counties.csv"
    return pd.read_csv(path)


def load_redfin_county() -> pd.DataFrame:
    return read_redfin_county_cached(HOUSING_DIR / "Redfin-Housing-Market-By-County.csv", cache_dir=CACHE_DIR)


def load_fips_master() -> pd.DataFrame:
    path = GEOGRAPHIC_DIR / "fips_master_v2.csv"
    if not path.exists():
        path = FIPSGEO_DIR / "fips_master_v2.csv"
    return pd.read_csv(path)


def load_personal_income(usecols: list[str] | None = None) -> pd.DataFrame:
    return pd.read_csv(
        ECONOMIC_DIR / "BEA - US, States, Counties - Personal Income.csv",
        usecols=usecols,
        low_memory=False,
    )


def load_raw_inputs() -> dict[str, pd.DataFrame]:
    if "raw_inputs" in _CACHE:
        return _CACHE["raw_inputs"]  # type: ignore[return-value]

    nri_county_df = load_nri_counties()
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
        "natural_disasters_df": load_fema_disasters(),
        "nri_county_df": nri_county_df,
        "housing_county_df": load_redfin_county(),
        "fips_df": load_fips_master(),
        "personal_income_df": load_personal_income(
            usecols=["IBRC_GEO_ID", "Year", "Data", "Linecode Description"],
        ),
    }
    _CACHE["raw_inputs"] = raw_inputs
    return raw_inputs


def load_population_estimates_df() -> pd.DataFrame:
    if "population_estimates_df" in _CACHE:
        return _CACHE["population_estimates_df"]  # type: ignore[return-value]
    population_estimates_df = add_county_fips(pd.read_csv(POPULATION_DIR / "Population Estimates - U.S., States, and Counties.csv", low_memory=False))
    population_estimates_df = population_estimates_df[population_estimates_df["State or County Release"].eq("County")].copy()
    population_estimates_df = dedupe_by_key(population_estimates_df, ["fips", "Year"], "count_over_estimate")
    _CACHE["population_estimates_df"] = population_estimates_df
    return population_estimates_df


def load_profile_inputs() -> dict[str, pd.DataFrame]:
    if "profile_inputs" in _CACHE:
        return _CACHE["profile_inputs"]  # type: ignore[return-value]

    bea_income_raw = load_personal_income(
        usecols=["Statefips", "Countyfips", "Description", "Year", "Linecode", "Linecode Description", "Data"],
    )
    bea_income_df = add_county_fips(bea_income_raw)
    bea_income_df = bea_income_df[bea_income_df["Linecode"].isin(BEA_LINECODES)].copy()
    bea_income_df["metric"] = bea_income_df["Linecode"].map(BEA_LINECODES)

    profile_inputs = {
        "bea_income_df": bea_income_df,
        "cew_total_df": read_cew_totals_cached(ECONOMIC_DIR / "CEW - US, States, Counties - Total Ownership.csv", cache_dir=CACHE_DIR),
        "population_change_df": dedupe_by_key(add_county_fips(pd.read_csv(POPULATION_DIR / "Components of Population Change - U.S., States, and Counties.csv", low_memory=False)), ["fips", "Year"], "largest_magnitude"),
        "population_age_sex_df": dedupe_by_key(add_county_fips(pd.read_csv(POPULATION_DIR / "Population by Age and Sex - US, States, Counties.csv", low_memory=False)), ["fips", "Year"]),
        "population_race_df": dedupe_by_key(add_county_fips(pd.read_csv(POPULATION_DIR / "Population by Race - US, States, Counties.csv", low_memory=False)), ["fips", "Year"]),
        "population_estimates_df": load_population_estimates_df(),
    }
    _CACHE["profile_inputs"] = profile_inputs
    return profile_inputs


def load_cew_employment_wages() -> pd.DataFrame:
    return read_cew_employment_wages_cached(ECONOMIC_DIR / "CEW - US, States, Counties - Total Ownership.csv", cache_dir=CACHE_DIR)
