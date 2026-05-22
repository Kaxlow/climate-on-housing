"""Build the stormhouse county housing response visualization."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from housing_climate_risk.data_sources.raw import load_profile_inputs
from housing_climate_risk.data_sources.processed import prepare_housing_df, prepare_natural_disasters_df
from housing_climate_risk.modeling.economic_risk_profiles import build_economic_profile_outputs
from housing_climate_risk.modeling.insurance_risk_profiles import build_insurance_profile_outputs
from housing_climate_risk.modeling.migration_trend_profiles import build_migration_trend_profile_outputs
from housing_climate_risk.paths import CLIMATE_DIR, DATA_DIR, GEOGRAPHIC_DIR, OUTPUT_DIR

DATA_JS = OUTPUT_DIR / "stormhouse_data.js"
HTML_PATH = OUTPUT_DIR / "stormhouse.html"

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

VALID_RISK_RATINGS = ["Very Low", "Low", "Moderate", "High", "Very High"]
RISK_RATING_MAP = {
    "Very Low": "Very Low",
    "Relatively Low": "Low",
    "Relatively Moderate": "Moderate",
    "Relatively High": "High",
    "Very High": "Very High",
}
ECONOMIC_PROFILE_DIR = OUTPUT_DIR / "stormhouse_economic_profiles"
INSURANCE_PROFILE_DIR = OUTPUT_DIR / "stormhouse_insurance_profiles"
MIGRATION_TREND_PROFILE_DIR = OUTPUT_DIR / "stormhouse_migration_trends"
OFFSETS = list(range(-12, 25))
US_STATE_FIPS = {
    "01",
    "02",
    "04",
    "05",
    "06",
    "08",
    "09",
    "10",
    "11",
    "12",
    "13",
    "15",
    "16",
    "17",
    "18",
    "19",
    "20",
    "21",
    "22",
    "23",
    "24",
    "25",
    "26",
    "27",
    "28",
    "29",
    "30",
    "31",
    "32",
    "33",
    "34",
    "35",
    "36",
    "37",
    "38",
    "39",
    "40",
    "41",
    "42",
    "44",
    "45",
    "46",
    "47",
    "48",
    "49",
    "50",
    "51",
    "53",
    "54",
    "55",
    "56",
}


def _num(value: object, digits: int = 4) -> float | None:
    if pd.isna(value):
        return None
    return round(float(value), digits)


def _round_coords(coords: object, digits: int = 3) -> object:
    if isinstance(coords, (list, tuple)) and coords and isinstance(coords[0], (int, float)):
        return [round(float(coords[0]), digits), round(float(coords[1]), digits)]
    if isinstance(coords, (list, tuple)):
        return [_round_coords(item, digits) for item in coords]
    return coords


def build_county_risk_map() -> dict[str, object]:
    from shapely.geometry import mapping, shape

    nri = pd.read_csv(CLIMATE_DIR / "NRI_Table_Counties.csv", usecols=["STCOFIPS", "RISK_RATNG"])
    risk_by_fips = (
        nri.assign(
            fips=nri["STCOFIPS"].astype(str).str.zfill(5),
            riskRating=nri["RISK_RATNG"].map(RISK_RATING_MAP),
        )
        .drop_duplicates("fips")
        .set_index("fips")["riskRating"]
        .to_dict()
    )

    with (GEOGRAPHIC_DIR / "us_counties_boundaries_shapefile.json").open(encoding="utf-8") as file:
        counties = json.load(file)

    features: list[dict[str, object]] = []
    counts = {rating: 0 for rating in VALID_RISK_RATINGS}
    for feature in counties["features"]:
        props = feature["properties"]
        state_fips = str(props.get("STATEFP", "")).zfill(2)
        if state_fips not in US_STATE_FIPS:
            continue
        fips = str(props.get("GEOID", "")).zfill(5)
        risk_rating = risk_by_fips.get(fips)
        if risk_rating not in counts:
            continue
        counts[risk_rating] += 1

        tolerance = 0.08 if state_fips == "02" else 0.025
        geom = shape(feature["geometry"]).simplify(tolerance, preserve_topology=True)
        if geom.is_empty:
            continue
        geom_mapping = mapping(geom)
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": geom_mapping["type"],
                    "coordinates": _round_coords(geom_mapping["coordinates"]),
                },
                "properties": {
                    "fips": fips,
                    "stateFips": state_fips,
                    "name": props.get("NAMELSAD") or props.get("NAME") or fips,
                    "riskRating": risk_rating,
                },
            }
        )

    return {
        "type": "FeatureCollection",
        "features": features,
        "counts": counts,
    }


def build_complete_windows() -> pd.DataFrame:
    disasters = prepare_natural_disasters_df()
    housing = prepare_housing_df(include_profiles=False).copy()

    nri = pd.read_csv(CLIMATE_DIR / "NRI_Table_Counties.csv", usecols=["STCOFIPS", "RISK_RATNG"])
    nri = (
        nri.assign(
            fips=nri["STCOFIPS"].astype(str).str.zfill(5),
            nri_risk_rating=nri["RISK_RATNG"].map(RISK_RATING_MAP),
        )[["fips", "nri_risk_rating"]]
        .drop_duplicates("fips")
    )

    housing["fips"] = housing["fips"].astype(str).str.zfill(5)
    housing["fips_normalized"] = housing["fips"]
    housing["state_prefix"] = housing["fips"].str[:2]
    housing["MONTH"] = housing["MONTH"].astype("period[M]")
    housing = housing[
        [
            "fips",
            "fips_normalized",
            "state_prefix",
            "REGION",
            "county_name",
            "STATE_CODE",
            "MONTH",
            "HOUSING_MARKET_INDEX",
        ]
    ].copy()

    events = disasters.loc[~disasters["incidentType"].isin(EXCLUDED_INCIDENT_TYPES)].copy()
    events = events.dropna(subset=["fips", "incidentBeginDate"])
    duration = events["incidentEndDate"] - events["incidentBeginDate"]
    median_duration = duration[duration.notna()].median()
    events["incidentEndDate"] = events["incidentEndDate"].fillna(events["incidentBeginDate"] + median_duration)
    events = events.dropna(subset=["incidentEndDate"])
    events["fips"] = events["fips"].astype(str).str.zfill(5)
    events["event_id"] = np.arange(len(events), dtype=int) + 1
    events["incident_begin_month"] = events["incidentBeginDate"].dt.to_period("M")
    events["incident_end_month"] = events["incidentEndDate"].dt.to_period("M")

    event_rows: list[dict[str, object]] = []
    for event in events.itertuples(index=False):
        event_fips = str(event.fips).zfill(5)
        months = {event.incident_begin_month + offset: offset for offset in range(-12, 0)}
        months[event.incident_end_month] = 0
        months.update({event.incident_end_month + offset: offset for offset in range(1, 25)})
        for month, offset in months.items():
            event_rows.append(
                {
                    "event_fips": event_fips,
                    "event_state_prefix": event_fips[:2],
                    "is_statewide_event": event_fips.endswith("000"),
                    "MONTH": month,
                    "month_offset": int(offset),
                    "incident_event_id": int(event.event_id),
                    "incident_type": event.incidentType,
                    "incident_title": event.declarationTitle,
                    "incident_begin": event.incidentBeginDate.date().isoformat(),
                    "incident_end": event.incidentEndDate.date().isoformat(),
                }
            )

    event_months = pd.DataFrame(event_rows)
    county_events = event_months.loc[~event_months["is_statewide_event"]]
    state_events = event_months.loc[event_months["is_statewide_event"]]
    frames = []
    if not county_events.empty:
        frames.append(housing.merge(county_events, left_on=["fips_normalized", "MONTH"], right_on=["event_fips", "MONTH"], how="inner"))
    if not state_events.empty:
        frames.append(housing.merge(state_events, left_on=["state_prefix", "MONTH"], right_on=["event_state_prefix", "MONTH"], how="inner"))
    matched = pd.concat(frames, ignore_index=True) if frames else housing.iloc[0:0].copy()

    matched = matched.merge(nri, on="fips", how="left")
    matched = matched.dropna(subset=["HOUSING_MARKET_INDEX"])
    matched["series_id"] = (
        matched["fips"].astype(str)
        + "|"
        + matched["incident_type"].astype(str)
        + "|"
        + matched["incident_event_id"].astype(str)
    )
    expected = set(OFFSETS)
    complete_ids = (
        matched.groupby("series_id", sort=False)["month_offset"]
        .agg(lambda values: set(pd.to_numeric(values, errors="coerce").dropna().astype(int)) == expected)
    )
    complete_ids = complete_ids[complete_ids].index
    matched = matched.loc[matched["series_id"].isin(complete_ids)].copy()
    return matched


def build_nri_rating_lookup() -> pd.DataFrame:
    nri = pd.read_csv(CLIMATE_DIR / "NRI_Table_Counties.csv", usecols=["STCOFIPS", "RISK_RATNG"])
    return (
        nri.assign(
            fips=nri["STCOFIPS"].astype(str).str.zfill(5),
            nri_risk_rating=nri["RISK_RATNG"].map(RISK_RATING_MAP),
        )[["fips", "nri_risk_rating"]]
        .dropna(subset=["nri_risk_rating"])
        .drop_duplicates("fips")
    )


def load_county_processed_data() -> pd.DataFrame:
    """Load the nested county feature file used by county clustering workflows."""

    return pd.read_feather(DATA_DIR / "county_processed_data.feather")


def build_economic_profile_payload() -> dict[str, object]:
    outputs = build_economic_profile_outputs(
        profile_inputs=load_profile_inputs(),
        nri_ratings=build_nri_rating_lookup(),
        output_dir=ECONOMIC_PROFILE_DIR,
        risk_order=VALID_RISK_RATINGS,
    )
    summary = outputs["risk_profile_summary"].copy()
    lifts = outputs["risk_lifts"].copy()
    profile_summary = outputs["profile_summary"].copy()
    assignments = outputs["assignments"].copy()
    scores = outputs["scores"].copy()

    profile_rows = [
        {
            "profile": int(row.economic_profile),
            "label": row.economic_profile_label,
            "countyCount": int(row.county_count),
            "meanAssignmentConfidence": _num(row.mean_assignment_confidence, 3),
            "topHighFeatures": row.top_high_features,
            "topLowFeatures": row.top_low_features,
            "demographicDescription": row.demographic_description,
        }
        for row in profile_summary.itertuples(index=False)
    ]

    by_risk: dict[str, list[dict[str, object]]] = {}
    cards: list[dict[str, object]] = []
    for rating in VALID_RISK_RATINGS:
        rows = summary.loc[summary["nri_risk_rating"] == rating].copy()
        by_risk[rating] = [
            {
                "profile": int(row.economic_profile),
                "label": row.economic_profile_label,
                "counties": int(row.counties),
                "share": _num(row.share, 4),
                "meanAssignmentConfidence": _num(row.mean_assignment_confidence, 3),
            }
            for row in rows.itertuples(index=False)
        ]
        top_profiles = by_risk[rating][:3]
        high_lifts = lifts.loc[(lifts["nri_risk_rating"] == rating) & (lifts["direction"] == "higher")].head(3)
        low_lifts = lifts.loc[(lifts["nri_risk_rating"] == rating) & (lifts["direction"] == "lower")].head(2)
        cards.append(
            {
                "riskRating": rating,
                "countyCount": int(rows["risk_group_count"].max()) if not rows.empty else 0,
                "topProfiles": top_profiles,
                "higherTraits": [
                    {"label": label, "lift": lift}
                    for label, lift in serialize_plain_traits(high_lifts, limit=3)
                ],
                "lowerTraits": [
                    {"label": label, "lift": lift}
                    for label, lift in serialize_plain_traits(low_lifts, limit=2)
                ],
            }
        )

    contrast_rows = []
    for rating in VALID_RISK_RATINGS:
        higher = lifts.loc[(lifts["nri_risk_rating"] == rating) & (lifts["direction"] == "higher")].head(4)
        lower = lifts.loc[(lifts["nri_risk_rating"] == rating) & (lifts["direction"] == "lower")].head(3)
        contrast_rows.append(
            {
                "riskRating": rating,
                "higher": [
                    {"label": label, "lift": lift}
                    for label, lift in serialize_plain_traits(higher, limit=4)
                ],
                "lower": [
                    {"label": label, "lift": lift}
                    for label, lift in serialize_plain_traits(lower, limit=3)
                ],
            }
        )

    return {
        "bestK": int(outputs["best_k"]),
        "modelScores": [
            {
                "k": int(row.k),
                "silhouette": _num(row.silhouette_score, 3),
                "daviesBouldin": _num(row.davies_bouldin_index, 3),
                "meanAssignmentConfidence": _num(row.mean_assignment_confidence, 3),
                "minClusterSize": int(row.min_cluster_size),
                "maxClusterShare": _num(row.max_cluster_share, 3),
            }
            for row in scores.itertuples(index=False)
        ],
        "profiles": profile_rows,
        "assignments": [
            {
                "fips": str(row.fips).zfill(5),
                "countyName": row.county_name,
                "profile": int(row.economic_profile),
                "label": row.economic_profile_label,
                "assignmentConfidence": _num(row.assignment_confidence, 3),
            }
            for row in assignments.itertuples(index=False)
        ],
        "byRiskRating": by_risk,
        "cards": cards,
        "featureContrasts": contrast_rows,
        "commentary": build_economic_profile_commentary(summary, lifts),
    }


def build_economic_profile_commentary(summary: pd.DataFrame, lifts: pd.DataFrame) -> list[str]:
    """Write short, data-driven notes on how economic profiles differ by NRI rating."""

    return [
        "Lower-risk counties are more often smaller mixed-economy or average-economy counties, while higher-risk counties are more often large high-wage metro counties. The very-high-risk group is small, but it leans toward high-wage investment-income counties.",
    ]


def plain_economic_feature_label(label: str) -> str:
    """Convert model feature labels into page-friendly wording."""

    out = str(label).replace(" latest", "").replace(" trend", " change")
    replacements = {
        "population scale": "population size",
        "average weekly wage": "weekly wages",
        "per capita income": "income per person",
        "dividends, interest, and rent share": "investment and rent income share",
        "transfer receipts share": "public transfer income share",
        "proprietors income share": "business-owner income share",
        "prime working-age share": "working-age adult share",
        "natural increase rate": "births minus deaths rate",
        "international migration rate": "international migration",
        "domestic migration rate": "domestic migration",
        "White share": "White population share",
        "Black share": "Black population share",
        "Asian share": "Asian population share",
        "Hispanic share": "Hispanic population share",
        "senior share": "senior population share",
        "youth share": "youth population share",
        "male share": "male population share",
    }
    return replacements.get(out, out)


def serialize_plain_traits(rows: pd.DataFrame, limit: int) -> list[tuple[str, float | None]]:
    """Return deduplicated plain feature labels and rounded standardized lifts."""

    traits: list[tuple[str, float | None]] = []
    seen: set[str] = set()
    for row in rows.itertuples(index=False):
        label = plain_economic_feature_label(row.feature_label)
        if label in seen:
            continue
        seen.add(label)
        traits.append((label, _num(row.standardized_lift, 2)))
        if len(traits) >= limit:
            break
    return traits


def build_insurance_profile_payload() -> dict[str, object]:
    outputs = build_insurance_profile_outputs(
        counties=load_county_processed_data(),
        nri_ratings=build_nri_rating_lookup(),
        output_dir=INSURANCE_PROFILE_DIR,
        risk_order=VALID_RISK_RATINGS,
    )
    summary = outputs["risk_profile_summary"].copy()
    lifts = outputs["risk_lifts"].copy()
    profile_summary = outputs["profile_summary"].copy()
    assignments = outputs["assignments"].copy()
    scores = outputs["scores"].copy()

    profile_rows = [
        {
            "profile": int(row.insurance_profile),
            "label": row.insurance_profile_label,
            "countyCount": int(row.county_count),
            "meanAssignmentConfidence": _num(row.mean_assignment_confidence, 3),
            "topHighFeatures": row.top_high_features,
            "topLowFeatures": row.top_low_features,
        }
        for row in profile_summary.itertuples(index=False)
    ]

    by_risk: dict[str, list[dict[str, object]]] = {}
    cards: list[dict[str, object]] = []
    for rating in VALID_RISK_RATINGS:
        rows = summary.loc[summary["nri_risk_rating"] == rating].copy()
        by_risk[rating] = [
            {
                "profile": int(row.insurance_profile),
                "label": row.insurance_profile_label,
                "counties": int(row.counties),
                "share": _num(row.share, 4),
                "meanAssignmentConfidence": _num(row.mean_assignment_confidence, 3),
            }
            for row in rows.itertuples(index=False)
        ]
        top_profiles = by_risk[rating][:3]
        high_lifts = lifts.loc[(lifts["nri_risk_rating"] == rating) & (lifts["direction"] == "higher")].head(12)
        low_lifts = lifts.loc[(lifts["nri_risk_rating"] == rating) & (lifts["direction"] == "lower")].head(10)
        cards.append(
            {
                "riskRating": rating,
                "countyCount": int(rows["risk_group_count"].max()) if not rows.empty else 0,
                "topProfiles": top_profiles,
                "higherTraits": [
                    {"label": label, "lift": lift}
                    for label, lift in serialize_plain_insurance_traits(high_lifts, limit=3)
                ],
                "lowerTraits": [
                    {"label": label, "lift": lift}
                    for label, lift in serialize_plain_insurance_traits(low_lifts, limit=2)
                ],
            }
        )

    contrast_rows = []
    for rating in VALID_RISK_RATINGS:
        higher = lifts.loc[(lifts["nri_risk_rating"] == rating) & (lifts["direction"] == "higher")].head(14)
        lower = lifts.loc[(lifts["nri_risk_rating"] == rating) & (lifts["direction"] == "lower")].head(12)
        contrast_rows.append(
            {
                "riskRating": rating,
                "higher": [
                    {"label": label, "lift": lift}
                    for label, lift in serialize_plain_insurance_traits(higher, limit=4)
                ],
                "lower": [
                    {"label": label, "lift": lift}
                    for label, lift in serialize_plain_insurance_traits(lower, limit=3)
                ],
            }
        )

    return {
        "bestK": int(outputs["best_k"]),
        "modelScores": [
            {
                "k": int(row.k),
                "silhouette": _num(row.silhouette_score, 3),
                "daviesBouldin": _num(row.davies_bouldin_index, 3),
                "calinskiHarabasz": _num(row.calinski_harabasz_score, 1),
                "meanAssignmentConfidence": _num(row.mean_assignment_confidence, 3),
                "minClusterSize": int(row.min_cluster_size),
                "maxClusterShare": _num(row.max_cluster_share, 3),
                "lowConfidenceRate": _num(row.low_confidence_under_0_60_rate, 3),
            }
            for row in scores.itertuples(index=False)
        ],
        "profiles": profile_rows,
        "assignments": [
            {
                "fips": str(row.fips).zfill(5),
                "countyName": row.county_name,
                "profile": int(row.insurance_profile),
                "label": row.insurance_profile_label,
                "assignmentConfidence": _num(row.assignment_confidence, 3),
            }
            for row in assignments.itertuples(index=False)
        ],
        "byRiskRating": by_risk,
        "cards": cards,
        "featureContrasts": contrast_rows,
        "commentary": build_insurance_profile_commentary(summary, lifts),
    }


def build_insurance_profile_commentary(summary: pd.DataFrame, lifts: pd.DataFrame) -> list[str]:
    """Write short, plain notes on insurance differences by NRI rating."""

    return [
        "Higher-risk counties generally have higher premium levels, while lower-risk counties generally have lower premium levels but face faster premium growth.",
    ]


def plain_insurance_feature_label(label: str) -> str:
    """Convert insurance model labels into short page wording."""

    out = str(label).lower()
    if "premium growth" in out:
        return "premium growth"
    if "premium historical" in out and "slope" in out:
        return "premium trend"
    if "premium historical" in out and "volatility" in out:
        return "premium volatility"
    if (
        "premium latest" in out
        or "premium average" in out
        or ("premium historical" in out and "latest" in out)
        or ("premium historical" in out and "mean last 12" in out)
    ):
        return "premium level"
    if "nonrenewal growth" in out:
        return "nonrenewal-rate growth"
    if "nonrenewal historical" in out and "slope" in out:
        return "nonrenewal-rate trend"
    if "nonrenewal historical" in out and "volatility" in out:
        return "nonrenewal-rate volatility"
    if (
        "nonrenewal latest" in out
        or "nonrenewal average" in out
        or ("nonrenewal historical" in out and "mean last 12" in out)
    ):
        return "nonrenewal rate"
    replacements = {
        "home-insurance premium latest percentile mean national percentile": "premium level",
        "home-insurance premium latest percentile median national percentile": "premium level",
        "home-insurance premium growth percentile mean growth national percentile": "premium growth",
        "home-insurance premium growth percentile median growth national percentile": "premium growth",
        "home-insurance premium latest mean": "premium level",
        "home-insurance premium latest median": "premium level",
        "home-insurance premium growth mean growth": "premium growth",
        "home-insurance premium growth median growth": "premium growth",
        "home-insurance nonrenewal latest percentile nonrenewal rate national percentile": "nonrenewal rate",
        "home-insurance nonrenewal average nonrenewal rate": "nonrenewal rate",
        "home-insurance nonrenewal latest nonrenewal rate": "nonrenewal rate",
    }
    out = replacements.get(out, out)
    out = out.replace("home-insurance", "home insurance")
    out = out.replace(" national percentile", "")
    out = out.replace(" state percentile", "")
    return " ".join(out.split())


def serialize_plain_insurance_traits(rows: pd.DataFrame, limit: int) -> list[tuple[str, float | None]]:
    """Return deduplicated plain insurance labels and rounded standardized lifts."""

    traits: list[tuple[str, float | None]] = []
    seen: set[str] = set()
    for row in rows.itertuples(index=False):
        label = plain_insurance_feature_label(row.feature_label)
        if label in seen:
            continue
        seen.add(label)
        traits.append((label, _num(row.standardized_lift, 2)))
        if len(traits) >= limit:
            break
    return traits


def build_migration_trend_profile_payload(county_windows: pd.DataFrame) -> dict[str, object]:
    outputs = build_migration_trend_profile_outputs(
        county_windows=county_windows,
        profile_inputs=load_profile_inputs(),
        output_dir=MIGRATION_TREND_PROFILE_DIR,
        risk_order=VALID_RISK_RATINGS,
    )
    profile_summary = outputs["profile_summary"].copy()
    risk_summary = outputs["risk_profile_summary"].copy()
    series_summary = outputs["series_summary"].copy()
    scores = outputs["scores"].copy()

    profiles = [
        {
            "profile": int(row.migration_trend_profile),
            "label": row.migration_trend_profile_label,
            "countyCount": int(row.county_count),
            "meanAssignmentConfidence": _num(row.mean_assignment_confidence, 3),
            "preAvg": _num(row.pre_avg, 2),
            "postAvg": _num(row.post_avg, 2),
            "overallChange": _num(row.overall_change, 2),
            "firstYearChange": _num(row.first_year_change, 2),
            "secondYearChange": _num(row.second_year_change, 2),
            "volatility": _num(row.volatility, 2),
            "interpretation": row.interpretation,
        }
        for row in profile_summary.itertuples(index=False)
    ]

    by_risk: dict[str, list[dict[str, object]]] = {}
    for rating in VALID_RISK_RATINGS:
        rows = risk_summary.loc[risk_summary["nri_risk_rating"] == rating]
        by_risk[rating] = [
            {
                "profile": int(row.migration_trend_profile),
                "label": row.migration_trend_profile_label,
                "counties": int(row.counties),
                "share": _num(row.share, 4),
                "meanAssignmentConfidence": _num(row.mean_assignment_confidence, 3),
            }
            for row in rows.itertuples(index=False)
        ]

    by_profile: dict[str, list[dict[str, object]]] = {}
    for profile_id in profile_summary["migration_trend_profile"].astype(int).sort_values():
        rows = series_summary.loc[series_summary["migration_trend_profile"] == profile_id].sort_values("year_offset")
        by_profile[str(profile_id)] = [
            {
                "offset": int(row.year_offset),
                "median": _num(row.median),
                "q1": _num(row.q1),
                "q3": _num(row.q3),
                "n": int(row.n),
            }
            for row in rows.itertuples(index=False)
        ]

    return {
        "bestK": int(outputs["best_k"]),
        "modelScores": [
            {
                "k": int(row.k),
                "silhouette": _num(row.silhouette_score, 3),
                "daviesBouldin": _num(row.davies_bouldin_index, 3),
                "meanAssignmentConfidence": _num(row.mean_assignment_confidence, 3),
                "minClusterSize": int(row.min_cluster_size),
                "maxClusterShare": _num(row.max_cluster_share, 3),
            }
            for row in scores.itertuples(index=False)
        ],
        "profiles": profiles,
        "assignments": [
            {
                "fips": str(row.fips).zfill(5),
                "profile": int(row.migration_trend_profile),
                "label": row.migration_trend_profile_label,
                "assignmentConfidence": _num(row.assignment_confidence, 3),
            }
            for row in outputs["assignments"].itertuples(index=False)
        ],
        "byRiskRating": by_risk,
        "byProfile": by_profile,
    }


def build_profile_response_payload(weighted: pd.DataFrame, profile_payload: dict[str, object]) -> dict[str, list[dict[str, object]]]:
    """Summarize housing response trajectories by NRI rating and assigned profile."""

    assignments = pd.DataFrame(profile_payload.get("assignments", []))
    if assignments.empty:
        return {rating: [] for rating in VALID_RISK_RATINGS}

    assignments = assignments[["fips", "profile", "label"]].copy()
    assignments["fips"] = assignments["fips"].astype(str).str.zfill(5)
    profile_windows = weighted.merge(assignments, on="fips", how="inner")
    if profile_windows.empty:
        return {rating: [] for rating in VALID_RISK_RATINGS}

    stats = (
        profile_windows.groupby(["riskRating", "profile", "label", "month_offset"], sort=True)["housing_market_yoy_index"]
        .agg(
            median=lambda values: values.quantile(0.5),
            q1=lambda values: values.quantile(0.25),
            q3=lambda values: values.quantile(0.75),
            n="count",
        )
        .reset_index()
    )
    counts = (
        profile_windows.groupby(["riskRating", "profile"], sort=False)["fips"]
        .nunique()
        .rename("county_count")
        .reset_index()
    )
    stats = stats.merge(counts, on=["riskRating", "profile"], how="left")

    response: dict[str, list[dict[str, object]]] = {}
    for rating in VALID_RISK_RATINGS:
        rating_rows = stats.loc[stats["riskRating"] == rating]
        profile_rows: list[dict[str, object]] = []
        for (profile, label), rows in rating_rows.groupby(["profile", "label"], sort=True):
            rows = rows.sort_values("month_offset")
            profile_rows.append(
                {
                    "profile": int(profile),
                    "label": str(label),
                    "countyCount": int(rows["county_count"].max()) if not rows.empty else 0,
                    "rows": [
                        {
                            "offset": int(row.month_offset),
                            "median": _num(row.median),
                            "q1": _num(row.q1),
                            "q3": _num(row.q3),
                            "n": int(row.n),
                        }
                        for row in rows.itertuples(index=False)
                    ],
                }
            )
        response[rating] = sorted(profile_rows, key=lambda row: row["countyCount"], reverse=True)
    return response


def movement_phrase(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "a change that could not be measured"
    direction = "strengthening" if value > 0 else "weakening" if value < 0 else "little net movement"
    return f"{direction} of {abs(float(value)):.3f}"


def build_risk_takeaway(group_summaries: dict[str, dict[str, object]]) -> str:
    rows = []
    for rating, summary in group_summaries.items():
        if summary.get("countyCount", 0) <= 0:
            continue
        year_1 = summary.get("avgPreToMonths1To12")
        year_2 = summary.get("avgPreToMonths13To24")
        if year_1 is None or year_2 is None:
            continue
        values = [0, year_1, year_2]
        rows.append(
            {
                "rating": rating,
                "year1": year_1,
                "year2": year_2,
                "range": max(values) - min(values),
                "lateWeakening": year_2 < year_1,
            }
        )
    if not rows:
        return "The post-incident housing market movement could not be summarized for the risk groups."

    high_risk = [row for row in rows if row["rating"] in {"Moderate", "High", "Very High"}]
    high_risk_weakening = sum(1 for row in high_risk if row["lateWeakening"])
    high_risk_comment = (
        "Higher-risk counties generally show weakening markets over time, with the later post-incident year often softer than the first year."
        if high_risk_weakening >= max(1, len(high_risk) // 2)
        else "Higher-risk counties show more uneven post-incident movement, but the largest swings are still concentrated in the higher-risk groups."
    )
    strongest_rows = sorted(rows, key=lambda row: row["range"], reverse=True)[:2]

    def risk_trend_text(row: dict[str, object]) -> str:
        year_1 = float(row["year1"])
        year_2 = float(row["year2"])
        threshold = 0.01
        if year_1 < -threshold and year_2 < year_1:
            trend = "weakened after the incident and softened further in the second year"
        elif year_1 < -threshold and year_2 > year_1:
            trend = "weakened at first, then partly recovered"
        elif year_1 > threshold and year_2 < year_1:
            trend = "strengthened at first, then lost momentum"
        elif year_1 > threshold and year_2 > year_1:
            trend = "strengthened and kept improving"
        elif year_2 < -threshold:
            trend = "looked stable at first, then weakened in the second year"
        elif year_2 > threshold:
            trend = "looked stable at first, then improved in the second year"
        else:
            trend = "stayed comparatively stable"
        return f"{row['rating']} risk counties {trend}"

    return (
        f"{high_risk_comment} "
        f"The biggest movement trends were: {'; '.join(risk_trend_text(row) for row in strongest_rows)}."
    )


def build_profile_response_takeaway(weighted: pd.DataFrame, profile_payload: dict[str, object]) -> str:
    assignments = pd.DataFrame(profile_payload.get("assignments", []))
    if assignments.empty:
        return "No profile-specific housing market movement could be measured."

    assignments = assignments[["fips", "profile", "label"]].copy()
    assignments["fips"] = assignments["fips"].astype(str).str.zfill(5)
    profile_windows = weighted.merge(assignments, on="fips", how="inner")
    if profile_windows.empty:
        return "No profile-specific housing market movement could be measured."

    period_means = (
        profile_windows.assign(
            period=np.select(
                [
                    profile_windows["month_offset"].between(-12, -1),
                    profile_windows["month_offset"].between(1, 12),
                    profile_windows["month_offset"].between(13, 24),
                ],
                ["pre", "months_1_12", "months_13_24"],
                default="other",
            )
        )
        .loc[lambda df: df["period"].isin(["pre", "months_1_12", "months_13_24"])]
        .groupby(["riskRating", "profile", "label", "fips", "period"], sort=False)["housing_market_yoy_index"]
        .mean()
        .unstack("period")
        .reset_index()
    )
    summary = (
        period_means.dropna(subset=["pre", "months_1_12", "months_13_24"])
        .groupby(["riskRating", "profile", "label"], as_index=False)
        .agg(
            avg_pre=("pre", "mean"),
            avg_months_1_12=("months_1_12", "mean"),
            avg_months_13_24=("months_13_24", "mean"),
            county_count=("fips", "nunique"),
        )
    )
    if summary.empty:
        return "No profile-specific housing market movement could be measured."

    minimum_count = min(10, max(1, int(summary["county_count"].max())))
    eligible = summary.loc[summary["county_count"] >= minimum_count].copy()
    if eligible.empty:
        eligible = summary.copy()
    eligible["change_pre_to_1_12"] = eligible["avg_months_1_12"] - eligible["avg_pre"]
    eligible["change_1_12_to_13_24"] = eligible["avg_months_13_24"] - eligible["avg_months_1_12"]
    eligible["change_pre_to_13_24"] = eligible["avg_months_13_24"] - eligible["avg_pre"]
    eligible["movement_range"] = eligible[["avg_pre", "avg_months_1_12", "avg_months_13_24"]].max(axis=1) - eligible[
        ["avg_pre", "avg_months_1_12", "avg_months_13_24"]
    ].min(axis=1)
    strongest_rows = eligible.sort_values(["movement_range", "county_count"], ascending=[False, False]).head(3)

    def trend_text(row: pd.Series) -> str:
        first_change = float(row.change_pre_to_1_12)
        second_change = float(row.change_1_12_to_13_24)
        threshold = 0.01
        if abs(first_change) < threshold and abs(second_change) < threshold:
            trend = "stayed close to its pre-incident level"
        elif first_change < -threshold and second_change < -threshold:
            trend = "weakened in year 1 and kept weakening in year 2"
        elif first_change > threshold and second_change > threshold:
            trend = "strengthened in year 1 and kept strengthening in year 2"
        elif first_change > threshold and second_change < -threshold:
            trend = "strengthened in year 1, then gave back ground in year 2"
        elif first_change < -threshold and second_change > threshold:
            trend = "weakened in year 1, then recovered in year 2"
        elif first_change < -threshold:
            trend = "weakened mainly in year 1"
        elif second_change < -threshold:
            trend = "weakened mainly in year 2"
        elif first_change > threshold:
            trend = "strengthened mainly in year 1"
        else:
            trend = "strengthened mainly in year 2"
        return f"{row.label} in the {row.riskRating} risk group {trend}"

    details = "; ".join(trend_text(row) for _, row in strongest_rows.iterrows())
    return f"The largest peak-to-trough movement trends appeared in these clusters: {details}."


def build_migration_trend_payload(county_windows: pd.DataFrame, economic_profiles: dict[str, object]) -> dict[str, object]:
    """Summarize annual county net migration per 1,000 residents around incident years."""
    profile_inputs = load_profile_inputs()
    change = profile_inputs["population_change_df"][
        ["fips", "Year", "Net International Migration", "Net Domestic Migration"]
    ].copy()
    population = profile_inputs["population_estimates_df"][["fips", "Year", "Population"]].copy()
    change["fips"] = change["fips"].astype(str).str.zfill(5)
    change["Year"] = pd.to_numeric(change["Year"], errors="coerce").astype("Int64")
    population["fips"] = population["fips"].astype(str).str.zfill(5)
    population["Year"] = pd.to_numeric(population["Year"], errors="coerce").astype("Int64")
    population["population"] = pd.to_numeric(population["Population"], errors="coerce")
    change["net_migration"] = (
        pd.to_numeric(change["Net International Migration"], errors="coerce")
        + pd.to_numeric(change["Net Domestic Migration"], errors="coerce")
    )
    change = change.merge(population[["fips", "Year", "population"]], on=["fips", "Year"], how="left")
    change["net_migration_per_1000"] = np.where(
        change["population"] > 0,
        change["net_migration"] / change["population"] * 1000,
        np.nan,
    )
    events = (
        county_windows[["fips", "incident_event_id", "incident_begin_dt", "nri_risk_rating"]]
        .drop_duplicates(["fips", "incident_event_id"])
        .dropna(subset=["incident_begin_dt"])
        .copy()
    )
    events["incident_year"] = events["incident_begin_dt"].dt.year.astype("Int64")
    year_rows: list[dict[str, object]] = []
    for event in events.itertuples(index=False):
        for year_offset in range(-2, 3):
            year_rows.append(
                {
                    "fips": event.fips,
                    "incident_event_id": event.incident_event_id,
                    "riskRating": event.nri_risk_rating,
                    "year_offset": year_offset,
                    "Year": int(event.incident_year) + year_offset,
                }
            )
    work = pd.DataFrame(year_rows)
    work = work.merge(change[["fips", "Year", "net_migration_per_1000"]], on=["fips", "Year"], how="left")
    work = work.dropna(subset=["net_migration_per_1000"])
    lower_cap = float(work["net_migration_per_1000"].quantile(0.01)) if not work.empty else -10.0
    upper_cap = float(work["net_migration_per_1000"].quantile(0.99)) if not work.empty else 10.0
    work["plot_rate"] = work["net_migration_per_1000"].clip(lower=lower_cap, upper=upper_cap)
    work = work.sort_values(["fips", "year_offset", "incident_event_id"]).copy()
    work["recency_weight"] = work.groupby(["fips", "year_offset"]).cumcount() + 1
    work["weighted_rate"] = work["plot_rate"] * work["recency_weight"]
    weighted = (
        work.groupby(["fips", "year_offset"], as_index=False)
        .agg(
            net_migration_per_1000=("weighted_rate", "sum"),
            total_weight=("recency_weight", "sum"),
            incident_count=("incident_event_id", "nunique"),
            riskRating=("riskRating", "last"),
        )
        .assign(net_migration_per_1000=lambda df: df["net_migration_per_1000"] / df["total_weight"])
    )
    assignments = pd.DataFrame(economic_profiles.get("assignments", []))
    profile_labels = {
        int(profile["profile"]): str(profile["label"])
        for profile in economic_profiles.get("profiles", [])
        if profile.get("profile") is not None
    }
    if not assignments.empty:
        assignments = assignments[["fips", "profile", "label"]].copy()
        assignments["fips"] = assignments["fips"].astype(str).str.zfill(5)
        assignments["profile"] = pd.to_numeric(assignments["profile"], errors="coerce").astype("Int64")
        weighted = weighted.merge(assignments, on="fips", how="left")
    else:
        weighted["profile"] = pd.NA
        weighted["label"] = pd.NA

    def serialize_stats(stats_df: pd.DataFrame, key_column: str, keys: list[object]) -> dict[str, list[dict[str, object]]]:
        out: dict[str, list[dict[str, object]]] = {}
        for key in keys:
            rows = stats_df.loc[stats_df[key_column] == key].sort_values("year_offset")
            out[str(key)] = [
                {
                    "offset": int(row.year_offset),
                    "median": _num(row.median),
                    "q1": _num(row.q1),
                    "q3": _num(row.q3),
                    "n": int(row.n),
                }
                for row in rows.itertuples(index=False)
            ]
        return out

    def summarize_rows(source: pd.DataFrame, key_column: str, keys: list[object]) -> dict[str, dict[str, object]]:
        out: dict[str, dict[str, object]] = {}
        for key in keys:
            rows = source.loc[source[key_column] == key]
            offsets = rows.groupby("year_offset")["net_migration_per_1000"].median()
            out[str(key)] = {
                "countyCount": int(rows["fips"].nunique()),
                "minus2": _num(offsets.get(-2)),
                "minus1": _num(offsets.get(-1)),
                "incidentYear": _num(offsets.get(0)),
                "plus1": _num(offsets.get(1)),
                "plus2": _num(offsets.get(2)),
            }
        return out

    def serialize_overall_stats(source: pd.DataFrame) -> list[dict[str, object]]:
        stats_df = (
            source.groupby("year_offset", sort=True)["net_migration_per_1000"]
            .agg(
                median=lambda values: values.quantile(0.5),
                q1=lambda values: values.quantile(0.25),
                q3=lambda values: values.quantile(0.75),
                n="count",
            )
            .reset_index()
        )
        return [
            {
                "offset": int(row.year_offset),
                "median": _num(row.median),
                "q1": _num(row.q1),
                "q3": _num(row.q3),
                "n": int(row.n),
            }
            for row in stats_df.itertuples(index=False)
        ]

    def summarize_overall(source: pd.DataFrame) -> dict[str, object]:
        offsets = source.groupby("year_offset")["net_migration_per_1000"].median()
        return {
            "countyCount": int(source["fips"].nunique()),
            "minus2": _num(offsets.get(-2)),
            "minus1": _num(offsets.get(-1)),
            "incidentYear": _num(offsets.get(0)),
            "plus1": _num(offsets.get(1)),
            "plus2": _num(offsets.get(2)),
        }

    overall = serialize_overall_stats(weighted)
    overall_summary = summarize_overall(weighted)
    stats = (
        weighted.groupby(["riskRating", "year_offset"], sort=True)["net_migration_per_1000"]
        .agg(
            median=lambda values: values.quantile(0.5),
            q1=lambda values: values.quantile(0.25),
            q3=lambda values: values.quantile(0.75),
            n="count",
        )
        .reset_index()
    )
    by_risk = serialize_stats(stats, "riskRating", VALID_RISK_RATINGS)
    summaries = summarize_rows(weighted, "riskRating", VALID_RISK_RATINGS)
    profile_order = [
        int(profile["profile"])
        for profile in economic_profiles.get("profiles", [])
        if profile.get("profile") is not None
    ]
    profiled = weighted.dropna(subset=["profile"]).copy()
    profiled["profile"] = profiled["profile"].astype(int)
    profile_stats = (
        profiled.groupby(["profile", "year_offset"], sort=True)["net_migration_per_1000"]
        .agg(
            median=lambda values: values.quantile(0.5),
            q1=lambda values: values.quantile(0.25),
            q3=lambda values: values.quantile(0.75),
            n="count",
        )
        .reset_index()
    )
    profile_risk_stats = (
        profiled.groupby(["profile", "riskRating", "year_offset"], sort=True)["net_migration_per_1000"]
        .agg(
            median=lambda values: values.quantile(0.5),
            q1=lambda values: values.quantile(0.25),
            q3=lambda values: values.quantile(0.75),
            n="count",
        )
        .reset_index()
    )
    by_profile = serialize_stats(profile_stats, "profile", profile_order)
    profile_summaries = summarize_rows(profiled, "profile", profile_order)
    by_profile_risk: dict[str, dict[str, list[dict[str, object]]]] = {}
    profile_risk_summaries: dict[str, dict[str, dict[str, object]]] = {}
    for profile_id in profile_order:
        profile_rows = profile_risk_stats.loc[profile_risk_stats["profile"] == profile_id]
        by_profile_risk[str(profile_id)] = serialize_stats(profile_rows, "riskRating", VALID_RISK_RATINGS)
        profile_weighted = profiled.loc[profiled["profile"] == profile_id]
        profile_risk_summaries[str(profile_id)] = summarize_rows(profile_weighted, "riskRating", VALID_RISK_RATINGS)
    return {
        "overall": overall,
        "byRiskRating": by_risk,
        "byEconomicProfile": by_profile,
        "byEconomicProfileRiskRating": by_profile_risk,
        "overallSummary": overall_summary,
        "summaries": summaries,
        "profileSummaries": profile_summaries,
        "profileRiskSummaries": profile_risk_summaries,
        "economicProfiles": [
            {
                "profile": profile_id,
                "label": profile_labels.get(profile_id, f"Profile {profile_id}"),
            }
            for profile_id in profile_order
        ],
        "meta": {
            "offsets": [-2, -1, 0, 1, 2],
            "lowerCap": _num(lower_cap),
            "upperCap": _num(upper_cap),
            "metric": "net_migration_per_1000",
            "unit": "per_1000_residents",
        },
    }


def build_migration_housing_relationship_payload(
    county_windows: pd.DataFrame,
    housing_weighted: pd.DataFrame,
    economic_profiles: dict[str, object],
) -> list[dict[str, object]]:
    """Compare county migration change with housing-market change around incidents."""
    profile_inputs = load_profile_inputs()
    change = profile_inputs["population_change_df"][
        ["fips", "Year", "Net International Migration", "Net Domestic Migration"]
    ].copy()
    population = profile_inputs["population_estimates_df"][["fips", "Year", "Population"]].copy()
    change["fips"] = change["fips"].astype(str).str.zfill(5)
    change["Year"] = pd.to_numeric(change["Year"], errors="coerce").astype("Int64")
    population["fips"] = population["fips"].astype(str).str.zfill(5)
    population["Year"] = pd.to_numeric(population["Year"], errors="coerce").astype("Int64")
    population["population"] = pd.to_numeric(population["Population"], errors="coerce")
    change["net_migration"] = (
        pd.to_numeric(change["Net International Migration"], errors="coerce")
        + pd.to_numeric(change["Net Domestic Migration"], errors="coerce")
    )
    change = change.merge(population[["fips", "Year", "population"]], on=["fips", "Year"], how="left")
    change["net_migration_per_1000"] = np.where(
        change["population"] > 0,
        change["net_migration"] / change["population"] * 1000,
        np.nan,
    )
    events = (
        county_windows[["fips", "incident_event_id", "incident_begin_dt", "nri_risk_rating"]]
        .drop_duplicates(["fips", "incident_event_id"])
        .dropna(subset=["incident_begin_dt"])
        .copy()
    )
    events["incident_year"] = events["incident_begin_dt"].dt.year.astype("Int64")
    year_rows: list[dict[str, object]] = []
    for event in events.itertuples(index=False):
        for year_offset in range(-2, 3):
            year_rows.append(
                {
                    "fips": event.fips,
                    "incident_event_id": event.incident_event_id,
                    "riskRating": event.nri_risk_rating,
                    "year_offset": year_offset,
                    "Year": int(event.incident_year) + year_offset,
                }
            )
    migration = pd.DataFrame(year_rows)
    migration = migration.merge(change[["fips", "Year", "net_migration_per_1000"]], on=["fips", "Year"], how="left")
    migration = migration.dropna(subset=["net_migration_per_1000"])
    if migration.empty:
        return []
    lower_cap = float(migration["net_migration_per_1000"].quantile(0.01))
    upper_cap = float(migration["net_migration_per_1000"].quantile(0.99))
    migration["plot_rate"] = migration["net_migration_per_1000"].clip(lower=lower_cap, upper=upper_cap)
    migration = migration.sort_values(["fips", "year_offset", "incident_event_id"]).copy()
    migration["recency_weight"] = migration.groupby(["fips", "year_offset"]).cumcount() + 1
    migration["weighted_rate"] = migration["plot_rate"] * migration["recency_weight"]
    migration_weighted = (
        migration.groupby(["fips", "year_offset"], as_index=False)
        .agg(
            net_migration_per_1000=("weighted_rate", "sum"),
            total_weight=("recency_weight", "sum"),
            riskRating=("riskRating", "last"),
        )
        .assign(net_migration_per_1000=lambda df: df["net_migration_per_1000"] / df["total_weight"])
    )
    migration_periods = (
        migration_weighted.assign(
            period=np.select(
                [
                    migration_weighted["year_offset"].isin([-2, -1]),
                    migration_weighted["year_offset"].isin([1, 2]),
                ],
                ["pre", "post"],
                default="other",
            )
        )
        .loc[lambda df: df["period"].isin(["pre", "post"])]
        .groupby(["fips", "riskRating", "period"], sort=False)["net_migration_per_1000"]
        .mean()
        .unstack("period")
        .reset_index()
        .dropna(subset=["pre", "post"])
    )
    migration_periods["migrationChange"] = migration_periods["post"] - migration_periods["pre"]

    housing_periods = (
        housing_weighted.assign(
            period=np.select(
                [
                    housing_weighted["month_offset"].between(-12, -1),
                    housing_weighted["month_offset"].between(1, 24),
                ],
                ["pre", "post"],
                default="other",
            )
        )
        .loc[lambda df: df["period"].isin(["pre", "post"])]
        .groupby(["fips", "period"], sort=False)["housing_market_yoy_index"]
        .mean()
        .unstack("period")
        .reset_index()
        .dropna(subset=["pre", "post"])
    )
    housing_periods["housingChange"] = housing_periods["post"] - housing_periods["pre"]
    relationship = migration_periods.merge(housing_periods[["fips", "housingChange"]], on="fips", how="inner")
    assignments = pd.DataFrame(economic_profiles.get("assignments", []))
    if not assignments.empty:
        assignments = assignments[["fips", "profile", "label"]].copy()
        assignments["fips"] = assignments["fips"].astype(str).str.zfill(5)
        assignments["profile"] = pd.to_numeric(assignments["profile"], errors="coerce").astype("Int64")
        relationship = relationship.merge(assignments, on="fips", how="left")
    else:
        relationship["profile"] = pd.NA
        relationship["label"] = pd.NA
    relationship = relationship.replace({np.nan: None})
    return [
        {
            "fips": row.fips,
            "riskRating": row.riskRating,
            "profile": None if row.profile is None or pd.isna(row.profile) else int(row.profile),
            "profileLabel": None if row.label is None or pd.isna(row.label) else str(row.label),
            "migrationChange": _num(row.migrationChange, 2),
            "housingChange": _num(row.housingChange, 4),
        }
        for row in relationship.itertuples(index=False)
    ]


def build_county_housing_series_payload(
    housing_weighted: pd.DataFrame,
    economic_profiles: dict[str, object],
    migration_trend: dict[str, object],
    insurance_profiles: dict[str, object],
) -> list[dict[str, object]]:
    """Serialize county-level housing index series with profile assignments for browser filtering."""
    out = housing_weighted[["fips", "riskRating", "month_offset", "housing_market_yoy_index"]].copy()
    econ_assignments = pd.DataFrame(economic_profiles.get("assignments", []))
    if not econ_assignments.empty:
        econ_assignments = econ_assignments[["fips", "profile", "label"]].rename(
            columns={"profile": "economicProfile", "label": "economicProfileLabel"}
        )
        econ_assignments["fips"] = econ_assignments["fips"].astype(str).str.zfill(5)
        out = out.merge(econ_assignments, on="fips", how="left")
    migration_assignments = pd.DataFrame(migration_trend.get("trendProfiles", {}).get("assignments", []))
    if not migration_assignments.empty:
        migration_assignments = migration_assignments[["fips", "profile", "label"]].rename(
            columns={"profile": "migrationProfile", "label": "migrationProfileLabel"}
        )
        migration_assignments["fips"] = migration_assignments["fips"].astype(str).str.zfill(5)
        out = out.merge(migration_assignments, on="fips", how="left")
    insurance_assignments = pd.DataFrame(insurance_profiles.get("assignments", []))
    if not insurance_assignments.empty:
        insurance_assignments = insurance_assignments[["fips", "profile", "label"]].rename(
            columns={"profile": "insuranceProfile", "label": "insuranceProfileLabel"}
        )
        insurance_assignments["fips"] = insurance_assignments["fips"].astype(str).str.zfill(5)
        out = out.merge(insurance_assignments, on="fips", how="left")
    out = out.loc[out["month_offset"].between(-12, 24)].replace({np.nan: None})
    return [
        {
            "fips": row.fips,
            "riskRating": row.riskRating,
            "economicProfile": None if row.economicProfile is None or pd.isna(row.economicProfile) else int(row.economicProfile),
            "economicProfileLabel": None if row.economicProfileLabel is None or pd.isna(row.economicProfileLabel) else str(row.economicProfileLabel),
            "migrationProfile": None if row.migrationProfile is None or pd.isna(row.migrationProfile) else int(row.migrationProfile),
            "migrationProfileLabel": None if row.migrationProfileLabel is None or pd.isna(row.migrationProfileLabel) else str(row.migrationProfileLabel),
            "insuranceProfile": None if row.insuranceProfile is None or pd.isna(row.insuranceProfile) else int(row.insuranceProfile),
            "insuranceProfileLabel": None if row.insuranceProfileLabel is None or pd.isna(row.insuranceProfileLabel) else str(row.insuranceProfileLabel),
            "offset": int(row.month_offset),
            "value": _num(row.housing_market_yoy_index, 4),
        }
        for row in out.itertuples(index=False)
    ]


def _array_like_values(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Series):
        return value.tolist()
    if isinstance(value, list | tuple):
        return list(value)
    return []


def build_insurance_trend_payload(
    county_windows: pd.DataFrame,
    economic_profiles: dict[str, object],
    migration_trend: dict[str, object] | None = None,
) -> dict[str, object]:
    """Summarize annual county insurance series around incident years by economic profile."""

    counties = load_county_processed_data()[["fips", "insurance_premiums_14_to_24", "insurance_non_renewal_rates"]].copy()
    counties["fips"] = counties["fips"].astype(str).str.zfill(5)
    rows: list[dict[str, object]] = []
    for county in counties.itertuples(index=False):
        premium = county.insurance_premiums_14_to_24 if isinstance(county.insurance_premiums_14_to_24, dict) else {}
        premium_hist = premium.get("historical") if isinstance(premium.get("historical"), dict) else {}
        for year, value in zip(_array_like_values(premium_hist.get("years")), _array_like_values(premium_hist.get("median")), strict=False):
            rows.append({"fips": county.fips, "Year": int(year), "metric": "premiumLevel", "value": _num(value)})

        nonrenewal = county.insurance_non_renewal_rates if isinstance(county.insurance_non_renewal_rates, dict) else {}
        nonrenewal_hist = nonrenewal.get("historical") if isinstance(nonrenewal.get("historical"), dict) else {}
        for year, value in zip(_array_like_values(nonrenewal_hist.get("years")), _array_like_values(nonrenewal_hist.get("non_renewal_rate")), strict=False):
            rows.append({"fips": county.fips, "Year": int(year), "metric": "nonrenewalRate", "value": _num(value)})

    annual = pd.DataFrame(rows).dropna(subset=["value"])
    metric_meta = [
        {"key": "premiumLevel", "label": "Insurance premium levels", "format": "currency", "centerZero": False},
        {"key": "premiumYoy", "label": "Insurance YOY", "format": "percent", "centerZero": True},
        {"key": "nonrenewalRate", "label": "Non-renewal rates", "format": "percent", "centerZero": False},
    ]
    if annual.empty:
        return {
            "metrics": metric_meta,
            "byEconomicProfileMetric": {},
            "byRiskRatingMetric": {},
            "summaries": {},
            "riskSummaries": {},
            "domains": {},
            "overallCommentary": "",
        }

    annual = annual.sort_values(["fips", "metric", "Year"]).copy()
    annual["prior_value"] = annual.groupby(["fips", "metric"])["value"].shift(1)
    denom = annual["prior_value"].abs()
    annual["yoy"] = np.where(denom > 0.001, (annual["value"] - annual["prior_value"]) / denom, np.nan)
    value_rows = annual[["fips", "Year", "metric", "value"]].copy()
    yoy_rows = annual.dropna(subset=["yoy"])[["fips", "Year", "metric", "yoy"]].rename(columns={"yoy": "value"}).copy()
    yoy_rows["metric"] = yoy_rows["metric"].map({"premiumLevel": "premiumYoy", "nonrenewalRate": None})
    metric_values = pd.concat([value_rows, yoy_rows], ignore_index=True).dropna(subset=["metric", "value"])

    events = (
        county_windows[["fips", "incident_event_id", "incident_begin_dt", "nri_risk_rating"]]
        .drop_duplicates(["fips", "incident_event_id"])
        .dropna(subset=["incident_begin_dt"])
        .copy()
    )
    events["incident_year"] = events["incident_begin_dt"].dt.year.astype("Int64")
    year_rows: list[dict[str, object]] = []
    for event in events.itertuples(index=False):
        for year_offset in range(-2, 3):
            year_rows.append(
                {
                    "fips": event.fips,
                    "incident_event_id": event.incident_event_id,
                    "riskRating": event.nri_risk_rating,
                    "year_offset": year_offset,
                    "Year": int(event.incident_year) + year_offset,
                }
            )
    work = pd.DataFrame(year_rows).merge(metric_values, on=["fips", "Year"], how="inner")
    assignments = pd.DataFrame(economic_profiles.get("assignments", []))
    if not assignments.empty:
        assignments = assignments[["fips", "profile", "label"]].copy()
        assignments["fips"] = assignments["fips"].astype(str).str.zfill(5)
        assignments["profile"] = pd.to_numeric(assignments["profile"], errors="coerce").astype("Int64")
        work = work.merge(assignments, on="fips", how="left").dropna(subset=["profile"])
    else:
        work["profile"] = pd.NA
        work["label"] = pd.NA
    work["profile"] = work["profile"].astype(int)

    domains: dict[str, dict[str, object]] = {}
    capped_parts: list[pd.DataFrame] = []
    for metric in metric_meta:
        metric_key = str(metric["key"])
        metric_work = work.loc[work["metric"] == metric_key].copy()
        if metric_work.empty:
            continue
        lower = float(metric_work["value"].quantile(0.01))
        upper = float(metric_work["value"].quantile(0.99))
        metric_work["plot_value"] = metric_work["value"].clip(lower=lower, upper=upper)
        capped_parts.append(metric_work)
        min_value = float(metric_work["plot_value"].min())
        max_value = float(metric_work["plot_value"].max())
        if metric["centerZero"]:
            bound = max(abs(min_value), abs(max_value), 0.01)
            domain = [-bound, bound]
        else:
            pad = (max_value - min_value) * 0.12 if max_value != min_value else max(1.0, abs(max_value) * 0.12)
            domain = [min(0.0, min_value - pad), max_value + pad]
        domains[metric_key] = {"lowerCap": _num(lower), "upperCap": _num(upper), "domain": [_num(domain[0]), _num(domain[1])]}

    if not capped_parts:
        return {
            "metrics": metric_meta,
            "byEconomicProfileMetric": {},
            "byRiskRatingMetric": {},
            "summaries": {},
            "riskSummaries": {},
            "domains": domains,
            "overallCommentary": "",
        }
    capped = pd.concat(capped_parts, ignore_index=True).sort_values(["fips", "metric", "year_offset", "incident_event_id"])
    capped["recency_weight"] = capped.groupby(["fips", "metric", "year_offset"]).cumcount() + 1
    capped["weighted_value"] = capped["plot_value"] * capped["recency_weight"]
    weighted = (
        capped.groupby(["fips", "profile", "label", "riskRating", "metric", "year_offset"], as_index=False)
        .agg(value=("weighted_value", "sum"), total_weight=("recency_weight", "sum"))
        .assign(value=lambda df: df["value"] / df["total_weight"])
    )
    migration_assignments = pd.DataFrame((migration_trend or {}).get("trendProfiles", {}).get("assignments", []))
    if not migration_assignments.empty:
        migration_assignments = migration_assignments[["fips", "profile", "label"]].rename(
            columns={"profile": "migrationProfile", "label": "migrationProfileLabel"}
        )
        migration_assignments["fips"] = migration_assignments["fips"].astype(str).str.zfill(5)
        weighted = weighted.merge(migration_assignments, on="fips", how="left")
    else:
        weighted["migrationProfile"] = pd.NA
        weighted["migrationProfileLabel"] = pd.NA
    stats = (
        weighted.groupby(["profile", "label", "metric", "year_offset"], sort=True)["value"]
        .agg(median=lambda values: values.quantile(0.5), q1=lambda values: values.quantile(0.25), q3=lambda values: values.quantile(0.75), n="count")
        .reset_index()
    )
    risk_stats = (
        weighted.groupby(["riskRating", "metric", "year_offset"], sort=True)["value"]
        .agg(median=lambda values: values.quantile(0.5), q1=lambda values: values.quantile(0.25), q3=lambda values: values.quantile(0.75), n="count")
        .reset_index()
    )
    overall_stats = (
        weighted.groupby(["metric", "year_offset"], sort=True)["value"]
        .agg(median=lambda values: values.quantile(0.5), q1=lambda values: values.quantile(0.25), q3=lambda values: values.quantile(0.75), n="count")
        .reset_index()
    )
    overall_by_metric: dict[str, list[dict[str, object]]] = {}
    overall_summaries: dict[str, dict[str, object]] = {}
    county_metric_changes: list[dict[str, object]] = []
    for metric in metric_meta:
        metric_key = str(metric["key"])
        stat_rows = overall_stats.loc[overall_stats["metric"] == metric_key].sort_values("year_offset")
        overall_by_metric[metric_key] = [
            {"offset": int(row.year_offset), "median": _num(row.median), "q1": _num(row.q1), "q3": _num(row.q3), "n": int(row.n)}
            for row in stat_rows.itertuples(index=False)
        ]
        metric_rows = weighted.loc[weighted["metric"] == metric_key]
        offsets = metric_rows.groupby("year_offset")["value"].median()
        overall_summaries[metric_key] = {
            "countyCount": int(metric_rows["fips"].nunique()),
            "minus2": _num(offsets.get(-2)),
            "minus1": _num(offsets.get(-1)),
            "incidentYear": _num(offsets.get(0)),
            "plus1": _num(offsets.get(1)),
            "plus2": _num(offsets.get(2)),
        }
        periods = (
            metric_rows.assign(
                period=np.select(
                    [metric_rows["year_offset"].isin([-2, -1]), metric_rows["year_offset"].isin([1, 2])],
                    ["pre", "post"],
                    default="other",
                )
            )
            .loc[lambda df: df["period"].isin(["pre", "post"])]
            .groupby(["fips", "riskRating", "profile", "label", "period"], sort=False)["value"]
            .mean()
            .unstack("period")
            .reset_index()
            .dropna(subset=["pre", "post"])
        )
        periods["change"] = periods["post"] - periods["pre"]
        county_metric_changes.extend(
            {
                "fips": row.fips,
                "riskRating": row.riskRating,
                "economicProfile": int(row.profile),
                "economicProfileLabel": row.label,
                "metric": metric_key,
                "change": _num(row.change, 4),
            }
            for row in periods.itertuples(index=False)
        )
    profile_order = [int(profile["profile"]) for profile in economic_profiles.get("profiles", []) if profile.get("profile") is not None]
    by_profile_metric: dict[str, dict[str, list[dict[str, object]]]] = {}
    summaries: dict[str, dict[str, dict[str, object]]] = {}
    for profile_id in profile_order:
        profile_key = str(profile_id)
        by_profile_metric[profile_key] = {}
        summaries[profile_key] = {}
        for metric in metric_meta:
            metric_key = str(metric["key"])
            stat_rows = stats.loc[(stats["profile"] == profile_id) & (stats["metric"] == metric_key)].sort_values("year_offset")
            by_profile_metric[profile_key][metric_key] = [
                {"offset": int(row.year_offset), "median": _num(row.median), "q1": _num(row.q1), "q3": _num(row.q3), "n": int(row.n)}
                for row in stat_rows.itertuples(index=False)
            ]
            metric_rows = weighted.loc[(weighted["profile"] == profile_id) & (weighted["metric"] == metric_key)]
            offsets = metric_rows.groupby("year_offset")["value"].median()
            summaries[profile_key][metric_key] = {
                "countyCount": int(metric_rows["fips"].nunique()),
                "minus2": _num(offsets.get(-2)),
                "minus1": _num(offsets.get(-1)),
                "incidentYear": _num(offsets.get(0)),
                "plus1": _num(offsets.get(1)),
                "plus2": _num(offsets.get(2)),
            }
    by_risk_metric: dict[str, dict[str, list[dict[str, object]]]] = {}
    risk_summaries: dict[str, dict[str, dict[str, object]]] = {}
    for rating in VALID_RISK_RATINGS:
        by_risk_metric[rating] = {}
        risk_summaries[rating] = {}
        for metric in metric_meta:
            metric_key = str(metric["key"])
            stat_rows = risk_stats.loc[(risk_stats["riskRating"] == rating) & (risk_stats["metric"] == metric_key)].sort_values("year_offset")
            by_risk_metric[rating][metric_key] = [
                {"offset": int(row.year_offset), "median": _num(row.median), "q1": _num(row.q1), "q3": _num(row.q3), "n": int(row.n)}
                for row in stat_rows.itertuples(index=False)
            ]
            metric_rows = weighted.loc[(weighted["riskRating"] == rating) & (weighted["metric"] == metric_key)]
            offsets = metric_rows.groupby("year_offset")["value"].median()
            risk_summaries[rating][metric_key] = {
                "countyCount": int(metric_rows["fips"].nunique()),
                "minus2": _num(offsets.get(-2)),
                "minus1": _num(offsets.get(-1)),
                "incidentYear": _num(offsets.get(0)),
                "plus1": _num(offsets.get(1)),
                "plus2": _num(offsets.get(2)),
            }
    county_series = [
        {
            "fips": row.fips,
            "riskRating": row.riskRating,
            "economicProfile": int(row.profile),
            "economicProfileLabel": row.label,
            "migrationProfile": None if row.migrationProfile is None or pd.isna(row.migrationProfile) else int(row.migrationProfile),
            "migrationProfileLabel": None if row.migrationProfileLabel is None or pd.isna(row.migrationProfileLabel) else str(row.migrationProfileLabel),
            "metric": row.metric,
            "offset": int(row.year_offset),
            "value": _num(row.value, 6),
        }
        for row in weighted.replace({np.nan: None}).itertuples(index=False)
    ]
    return {
        "metrics": metric_meta,
        "overallByMetric": overall_by_metric,
        "byEconomicProfileMetric": by_profile_metric,
        "byRiskRatingMetric": by_risk_metric,
        "overallSummaries": overall_summaries,
        "summaries": summaries,
        "riskSummaries": risk_summaries,
        "countyMetricChanges": county_metric_changes,
        "countySeries": county_series,
        "domains": domains,
        "overallCommentary": "Premiums generally rise over the incident window, while non-renewal pressure is more uneven and concentrated in specific economic profiles.",
    }


def build_payload(windows: pd.DataFrame) -> dict[str, object]:
    county_risk_map = build_county_risk_map()
    county_windows = windows.loc[windows["nri_risk_rating"].isin(VALID_RISK_RATINGS)].copy()
    county_windows["incident_begin_dt"] = pd.to_datetime(county_windows["incident_begin"], errors="coerce")
    county_windows = county_windows.sort_values(["fips", "month_offset", "incident_begin_dt", "incident_event_id"]).copy()
    county_windows["recency_weight"] = county_windows.groupby(["fips", "month_offset"]).cumcount() + 1
    county_windows["weighted_index"] = county_windows["HOUSING_MARKET_INDEX"] * county_windows["recency_weight"]

    weighted = (
        county_windows.groupby(["fips", "month_offset"], as_index=False)
        .agg(
            housing_market_yoy_index=("weighted_index", "sum"),
            total_weight=("recency_weight", "sum"),
            incident_count=("incident_event_id", "nunique"),
        )
        .assign(housing_market_yoy_index=lambda df: df["housing_market_yoy_index"] / df["total_weight"])
    )
    meta = (
        county_windows.sort_values(["fips", "incident_begin_dt", "incident_event_id"])
        .groupby("fips", as_index=False)
        .agg(
            county=("county_name", "last"),
            state=("STATE_CODE", "last"),
            riskRating=("nri_risk_rating", "last"),
            latestIncident=("incident_begin", "last"),
            incidentCount=("incident_event_id", "nunique"),
        )
    )
    weighted = weighted.merge(meta, on="fips", how="left")

    group_stats = (
        weighted.groupby(["riskRating", "month_offset"], sort=True)["housing_market_yoy_index"]
        .agg(
            median=lambda values: values.quantile(0.5),
            q1=lambda values: values.quantile(0.25),
            q3=lambda values: values.quantile(0.75),
            n="count",
        )
        .reset_index()
    )

    risk_ratings = VALID_RISK_RATINGS
    group_payload: dict[str, list[dict[str, object]]] = {}
    for rating in risk_ratings:
        rows = group_stats.loc[group_stats["riskRating"] == rating]
        group_payload[rating] = [
            {
                "offset": int(row.month_offset),
                "median": _num(row.median),
                "q1": _num(row.q1),
                "q3": _num(row.q3),
                "n": int(row.n),
            }
            for row in rows.itertuples(index=False)
        ]

    period_means = (
        weighted.assign(
            period=np.select(
                [
                    weighted["month_offset"].between(-12, -1),
                    weighted["month_offset"].between(1, 12),
                    weighted["month_offset"].between(13, 24),
                ],
                ["pre", "months_1_12", "months_13_24"],
                default="other",
            )
        )
        .loc[lambda df: df["period"].isin(["pre", "months_1_12", "months_13_24"])]
        .groupby(["riskRating", "fips", "period"], sort=False)["housing_market_yoy_index"]
        .mean()
        .unstack("period")
        .reset_index()
    )
    period_means["change_pre_to_1_12"] = period_means["months_1_12"] - period_means["pre"]
    period_means["change_1_12_to_13_24"] = period_means["months_13_24"] - period_means["months_1_12"]
    period_means["change_pre_to_13_24"] = period_means["months_13_24"] - period_means["pre"]
    group_summaries: dict[str, dict[str, object]] = {}
    for rating in risk_ratings:
        group = period_means.loc[period_means["riskRating"] == rating]
        group_summaries[rating] = {
            "avgPreToMonths1To12": _num(group["change_pre_to_1_12"].mean()),
            "avgPreToMonths13To24": _num(group["change_pre_to_13_24"].mean()),
            "avgMonths1To12To13To24": _num(group["change_1_12_to_13_24"].mean()),
            "countyCount": int(group["fips"].nunique()),
        }

    commentary = [build_risk_takeaway(group_summaries)]
    economic_profiles = build_economic_profile_payload()
    migration_trend = build_migration_trend_payload(county_windows, economic_profiles)
    migration_trend["trendProfiles"] = build_migration_trend_profile_payload(county_windows)
    insurance_profiles = build_insurance_profile_payload()
    migration_trend["housingRelationship"] = build_migration_housing_relationship_payload(
        county_windows, weighted, economic_profiles
    )
    migration_trend["housingSeries"] = build_county_housing_series_payload(
        weighted, economic_profiles, migration_trend, insurance_profiles
    )
    insurance_trends = build_insurance_trend_payload(county_windows, economic_profiles, migration_trend)
    economic_profiles["responseByRiskRating"] = build_profile_response_payload(weighted, economic_profiles)
    insurance_profiles["responseByRiskRating"] = build_profile_response_payload(weighted, insurance_profiles)
    economic_profiles["responseTakeaway"] = build_profile_response_takeaway(weighted, economic_profiles)
    insurance_profiles["responseTakeaway"] = build_profile_response_takeaway(weighted, insurance_profiles)

    incident_types = sorted(windows["incident_type"].dropna().astype(str).unique())
    return {
        "meta": {
            "completeCountyIncidentWindows": int(windows["series_id"].nunique()),
            "countyCount": int(windows["fips"].nunique()),
            "plottedCountyCount": int(weighted["fips"].nunique()),
            "incidentTypes": incident_types,
            "excludedIncidentTypes": sorted(EXCLUDED_INCIDENT_TYPES),
            "riskRatings": risk_ratings,
            "offsets": OFFSETS,
        },
        "byRiskRating": group_payload,
        "groupSummaries": group_summaries,
        "commentary": commentary,
        "countyRiskMap": county_risk_map,
        "migrationTrend": migration_trend,
        "economicProfiles": economic_profiles,
        "insuranceProfiles": insurance_profiles,
        "insuranceTrends": insurance_trends,
    }


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Stormhouse: Housing Market Index Around FEMA Incidents</title>
  <script src="stormhouse_data.js?v=housing-insurance-premium-only"></script>
  <script src="https://d3js.org/d3.v7.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/topojson/3.0.2/topojson.min.js"></script>
  <style>
    :root {
      --bg: #f7f5ef;
      --ink: #172026;
      --muted: #5e6872;
      --line: #c7c1b5;
      --panel: #ffffff;
      --accent: #0f766e;
      --band: rgba(15, 118, 110, 0.18);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--ink);
    }
    main { width: min(1180px, calc(100vw - 32px)); margin: 0 auto; padding: 28px 0 42px; }
    header { margin-bottom: 18px; }
    h1 { margin: 0; font-size: 30px; line-height: 1.1; letter-spacing: 0; }
    .dek { margin: 10px 0 0; color: var(--muted); max-width: none; font-size: 15px; line-height: 1.45; }
    section { background: var(--panel); border: 1px solid #ddd8cf; border-radius: 8px; padding: 16px; margin-top: 16px; box-shadow: 0 1px 2px rgba(23, 32, 38, 0.04); }
    .story-section { background: transparent; border: 0; border-radius: 0; padding: 0; box-shadow: none; }
    .story-panel { transition: opacity 260ms ease; }
    .story-panel[hidden] { display: none; }
    .story-panel.fading-out { opacity: 0; }
    .story-panel.fading-in { animation: panel-fade-in 280ms ease both; }
    @keyframes panel-fade-in { from { opacity: 0; } to { opacity: 1; } }
    .visual-container { background: var(--panel); border: 1px solid #ddd8cf; border-radius: 8px; padding: 16px; margin-top: 10px; box-shadow: 0 1px 2px rgba(23, 32, 38, 0.04); }
    .section-head { display: flex; justify-content: space-between; gap: 16px; align-items: start; margin-bottom: 8px; }
    h2 { margin: 0; font-size: 18px; line-height: 1.25; }
    .sub { color: var(--muted); font-size: 13px; margin-top: 4px; max-width: none; }
    .chart { width: 100%; height: 470px; }
    .chart text { fill: var(--muted); font-size: 12px; }
    .axis path, .axis line, .grid line { stroke: #d8d3ca; }
    .grid path { display: none; }
    .period-band-early { fill: rgba(15, 118, 110, 0.045); stroke: none; }
    .period-band-late { fill: rgba(15, 118, 110, 0.085); stroke: none; }
    .period-label { fill: var(--muted); font-size: 11px; }
    .zero-line { stroke: #111827; stroke-width: 2.25; }
    .risk-line { fill: none; stroke-width: 2.6; stroke-linejoin: round; stroke-linecap: round; }
    .risk-hit-line { fill: none; stroke: transparent; stroke-width: 14; stroke-linejoin: round; stroke-linecap: round; cursor: pointer; }
    .risk-band { stroke: none; opacity: 0; pointer-events: none; transition: opacity 120ms ease; }
    .risk-band.active { opacity: 0.2; }
    .risk-band.background { opacity: 0.055; }
    .risk-line.dimmed { opacity: 0.16; }
    .risk-line.focused { opacity: 1; stroke-width: 3.2; }
    .incident-line { stroke: #2f3941; stroke-width: 1.5; stroke-dasharray: 5 5; }
    .incident-label { fill: #2f3941; font-size: 12px; font-weight: 700; }
    .legend { display: flex; flex-wrap: wrap; gap: 10px 16px; color: var(--muted); font-size: 12px; margin-top: 8px; }
    .plot-with-legend { display: grid; grid-template-columns: 150px minmax(0, 1fr); gap: 12px; align-items: stretch; margin-top: 10px; }
    .plot-with-two-legends { grid-template-columns: 150px minmax(0, 1fr) 170px; }
    .plot-with-legend .legend { align-content: start; align-items: stretch; flex-direction: column; flex-wrap: nowrap; gap: 8px; margin-top: 0; }
    .legend[hidden], .migration-filter-row [hidden] { display: none !important; }
    .plot-with-legend .risk-toggle-button { width: 100%; justify-content: flex-start; }
    .legend-item { display: inline-flex; align-items: center; gap: 6px; }
    .swatch { width: 18px; height: 3px; border-radius: 2px; display: inline-block; }
    .control-row { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 10px; margin: 10px 0 8px; }
    .button-row { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
    .control-button, .risk-toggle-button, .icon-button { border: 1px solid #d8d3ca; border-radius: 6px; background: #fffdf8; color: var(--muted); padding: 7px 10px; font: inherit; font-size: 12px; font-weight: 700; cursor: pointer; }
    .risk-toggle-button { display: inline-flex; align-items: center; gap: 6px; }
    .control-button.active, .risk-toggle-button.active, .icon-button.active { background: #172026; color: #fff; border-color: #172026; }
    .icon-button { min-width: 34px; padding-inline: 9px; }
    .paused-actions { display: none; }
    .paused-actions.visible { display: flex; }
    .view-panel[hidden] { display: none; }
    .drilldown-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; margin-top: 12px; }
    .drilldown-panel { min-width: 0; }
    .drilldown-panel h3 { font-size: 13px; margin-bottom: 6px; }
    .section-jump-row { display: flex; justify-content: flex-end; margin-bottom: 8px; }
    .section-jump-row.top-center, .section-jump-row.bottom-center { justify-content: center; }
    .section-jump-row.bottom-center { margin: 14px 0 0; }
    .edge-jump { position: fixed; left: 50%; transform: translateX(-50%); z-index: 20; opacity: 0; pointer-events: none; transition: opacity 180ms ease; margin: 0; }
    .edge-jump.visible { opacity: 1; pointer-events: auto; }
    .edge-jump.bottom-center { bottom: 18px; }
    .edge-jump.top-center { top: 18px; }
    .edge-jump .icon-button { width: 48px; height: 48px; border-radius: 999px; font-size: 24px; line-height: 1; box-shadow: 0 8px 22px rgba(23, 32, 38, 0.22); background: #172026; color: #fff; border-color: #172026; }
    .map-wrap { border-top: 1px solid #e6e0d6; margin-top: 16px; padding-top: 14px; }
    h3 { margin: 0; font-size: 16px; line-height: 1.25; }
    .map { width: 100%; height: 560px; margin-top: 10px; display: block; background: #fbfaf7; border: 1px solid #ebe5dc; border-radius: 6px; }
    .county { stroke: #ffffff; stroke-width: 0.22; vector-effect: non-scaling-stroke; cursor: default; }
    .county:hover { stroke: #172026; stroke-width: 0.8; }
    .map-label { fill: var(--muted); font-size: 12px; font-weight: 700; }
    .map-frame { fill: none; stroke: #d8d3ca; stroke-width: 1; }
    .note { color: var(--muted); font-size: 12px; line-height: 1.4; margin-top: 10px; max-width: none; }
    .commentary { border-top: 1px solid #e6e0d6; margin-top: 12px; padding-top: 10px; color: var(--muted); font-size: 13px; line-height: 1.45; }
    .commentary p { margin: 0 0 6px; }
    .takeaway-banner { margin-top: 12px; border-left: 5px solid var(--accent); background: #eef7f4; color: #172026; padding: 12px 14px; font-size: 15px; line-height: 1.45; font-weight: 700; }
    .takeaway-banner p { margin: 0; }
    .tooltip {
      position: fixed;
      display: none;
      max-width: 300px;
      background: #172026;
      color: #fff;
      border-radius: 6px;
      padding: 9px 10px;
      font-size: 12px;
      line-height: 1.35;
      pointer-events: none;
      box-shadow: 0 6px 18px rgba(23, 32, 38, 0.22);
      z-index: 10;
    }
    .tooltip strong { display: block; margin-bottom: 4px; }
    .profile-grid { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 10px; margin-top: 12px; }
    .profile-card { border: 1px solid #e6e0d6; border-radius: 6px; padding: 10px; background: #fbfaf7; min-height: 150px; }
    .profile-card h3 { font-size: 13px; margin-bottom: 6px; }
    .profile-card .count { font-size: 12px; color: var(--muted); margin-bottom: 8px; }
    .profile-card ul { margin: 0; padding-left: 16px; color: var(--muted); font-size: 12px; line-height: 1.35; }
    .profile-card li + li { margin-top: 4px; }
    .profile-card .trait { margin-top: 8px; font-size: 12px; line-height: 1.35; color: var(--muted); }
    .profile-card .trait strong { color: #2f3941; }
    .econ-commentary { margin-top: 12px; display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }
    .econ-commentary p { margin: 0; border-left: 3px solid #2f3941; padding: 8px 10px; background: #fbfaf7; color: var(--muted); font-size: 13px; line-height: 1.45; }
    .key-takeaway { grid-template-columns: 1fr; }
    .key-takeaway p { border-left-width: 5px; background: #f6f1e7; color: #2f3941; font-size: 15px; font-weight: 700; padding: 12px 14px; }
    .dominance-grid { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 10px; margin-top: 14px; }
    .dominance-tile { border: 1px solid #e6e0d6; border-radius: 6px; padding: 10px; background: #fffdf8; min-height: 118px; }
    .dominance-tile h3 { font-size: 13px; margin-bottom: 8px; }
    .dominance-tile .dominant-label { font-size: 14px; font-weight: 700; color: #2f3941; line-height: 1.25; }
    .dominance-tile .dominant-share { font-size: 22px; font-weight: 800; margin-top: 6px; color: #111827; }
    .dominance-tile .dominant-next { margin-top: 8px; font-size: 12px; color: var(--muted); line-height: 1.35; }
    .heatmap-wrap { margin-top: 14px; overflow-x: auto; }
    .profile-heatmap { display: grid; gap: 4px; min-width: 760px; }
    .heatmap-cell { min-height: 54px; border: 1px solid #fffaf0; border-radius: 4px; padding: 6px; color: #111827; font-size: 11px; line-height: 1.2; }
    .heatmap-cell strong { display: block; font-size: 13px; margin-bottom: 2px; }
    .heatmap-label { min-height: 54px; display: flex; align-items: center; color: #2f3941; font-size: 12px; font-weight: 700; }
    .heatmap-col-label { color: var(--muted); font-size: 11px; font-weight: 700; line-height: 1.2; align-self: end; }
    .response-matrix { display: grid; gap: 4px; min-width: 760px; }
    .response-matrix .heatmap-cell { min-height: 58px; }
    .metric-explainer { margin-top: 8px; color: var(--muted); font-size: 12px; line-height: 1.45; }
    .metric-explainer p { margin: 0 0 4px; }
    .profile-intro { margin: 10px 0 14px; color: var(--muted); font-size: 13px; line-height: 1.45; }
    .profile-intro p { margin: 0 0 7px; }
    .profile-intro ul { margin: 0; padding-left: 18px; display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 3px 16px; }
    .profile-response-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; margin-top: 14px; }
    .profile-response-panel { border-top: 1px solid #e6e0d6; padding-top: 10px; min-width: 0; }
    .profile-response-panel h3 { font-size: 13px; margin-bottom: 6px; }
    .profile-response-chart { width: 100%; height: 250px; display: block; }
    .economic-response-chart { height: 520px; }
    .migration-paired-plots { display: grid; grid-template-columns: minmax(138px, 0.75fr) minmax(0, 2.2fr) minmax(0, 2.2fr) minmax(138px, 0.75fr); gap: 14px; align-items: start; margin-top: 12px; }
    .migration-paired-plots h3 { font-size: 13px; margin: 0 0 6px; color: #2f3941; }
    .migration-single-layout { display: grid; grid-template-columns: minmax(150px, 0.8fr) minmax(0, 3fr) minmax(170px, 0.9fr); gap: 14px; align-items: start; margin-top: 12px; }
    .migration-single-layout h3 { font-size: 15px; margin: 0 0 6px; color: #2f3941; }
    .migration-filter-panel { display: grid; gap: 8px; margin-top: 12px; }
    .migration-filter-row { display: grid; grid-template-columns: 150px minmax(0, 1fr); gap: 10px; align-items: start; }
    .migration-filter-row > .control-button { width: 150px; min-height: 42px; }
    .legend.horizontal { flex-direction: row; flex-wrap: wrap; align-items: stretch; }
    .legend.horizontal .risk-toggle-button { width: auto; min-width: 116px; }
    .migration-chart-only { margin-top: 12px; }
    .migration-chart-only h3 { font-size: 15px; margin: 0 0 6px; color: #2f3941; }
    .migration-relationship-wrap { margin-top: 12px; }
    .insurance-trend-plots { margin-bottom: 12px; }
    .migration-small-chart { height: 430px; }
    .insurance-followup { border-top: 1px solid #e6e0d6; margin-top: 16px; padding-top: 14px; }
    .insurance-followup h3 { font-size: 14px; margin: 0 0 8px; color: #2f3941; }
    .insurance-relationship-grid { display: grid; grid-template-columns: minmax(0, 1fr) minmax(150px, 0.38fr); gap: 14px; align-items: start; }
    .insurance-risk-trend-grid { display: grid; grid-template-columns: minmax(138px, 0.65fr) minmax(0, 2.1fr) minmax(138px, 0.65fr); gap: 14px; align-items: start; }
    .migration-cluster-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; margin-top: 10px; }
    .migration-cluster-card { border: 1px solid #e6e0d6; border-radius: 6px; background: #fffdf8; padding: 12px; font-size: 12px; line-height: 1.35; }
    .migration-cluster-card h4 { margin: 0 0 6px; font-size: 13px; color: #172026; }
    .migration-cluster-card .meta { color: var(--muted); margin-bottom: 8px; }
    .migration-cluster-card .stats { display: grid; grid-template-columns: 1fr 1fr; gap: 4px 8px; color: var(--muted); margin-top: 8px; }
    .migration-cluster-card .stats strong { color: #2f3941; }
    .insurance-left-visuals { display: grid; grid-template-columns: minmax(0, 1fr) 132px; gap: 10px; align-items: center; }
    .insurance-risk-pie { width: 132px; height: 180px; display: block; }
    .pie-label { fill: #2f3941; font-size: 10px; font-weight: 700; }
    .pie-note { fill: var(--muted); font-size: 9px; }
    .migration-share-tile { border: 1px solid #e6e0d6; border-radius: 6px; background: #fffdf8; padding: 10px; margin-bottom: 10px; color: var(--muted); font-size: 12px; line-height: 1.3; }
    .migration-share-tile strong { display: block; color: #111827; font-size: 26px; line-height: 1; margin: 4px 0; }
    .migration-share-tile .label { color: #2f3941; font-weight: 700; }
    .profile-response-chart text { font-size: 10px; }
    .profile-response-band { stroke: none; opacity: 0; pointer-events: none; transition: opacity 120ms ease; }
    .profile-response-band.active { opacity: 0.18; }
    .profile-response-line { fill: none; stroke-width: 2.3; stroke-linejoin: round; stroke-linecap: round; }
    .economic-profile-band { opacity: 1; fill-opacity: 0.08; stroke-opacity: 0.9; stroke-width: 2.1; stroke-dasharray: 7 5; }
    .economic-profile-band.active { opacity: 1; fill-opacity: 0.10; }
    .economic-profile-line { stroke-width: 3.5; stroke-dasharray: 9 5; }
    .profile-response-hit-line { fill: none; stroke: transparent; stroke-width: 12; stroke-linejoin: round; stroke-linecap: round; cursor: pointer; }
    .profile-map { width: 100%; height: 520px; margin-top: 14px; display: block; }
    .profile-table { width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 12px; }
    .profile-table th, .profile-table td { border-top: 1px solid #e6e0d6; padding: 8px 6px; text-align: left; vertical-align: top; }
    .profile-table th { color: #2f3941; font-size: 11px; text-transform: uppercase; letter-spacing: 0; }
    .trait-list { margin: 0; padding-left: 16px; color: var(--muted); }
    .trait-list li + li { margin-top: 3px; }
    .bar-label { fill: #2f3941; font-size: 11px; font-weight: 700; }
    .bar-axis-label { fill: var(--muted); font-size: 11px; }
    .section-subheading { margin: 22px 0 8px; font-size: 16px; line-height: 1.25; }
    @media (max-width: 820px) {
      main { width: min(100vw - 20px, 1180px); padding-top: 18px; }
      .chart { height: 390px; }
      .map { height: 430px; }
      .plot-with-legend { grid-template-columns: 1fr; }
      .plot-with-two-legends { grid-template-columns: 1fr; }
      .plot-with-legend .legend { flex-direction: row; flex-wrap: wrap; }
      .plot-with-legend .risk-toggle-button { width: auto; }
      .drilldown-grid { grid-template-columns: 1fr; }
      .profile-grid { grid-template-columns: 1fr; }
      .dominance-grid { grid-template-columns: 1fr; }
      .migration-cluster-grid { grid-template-columns: 1fr; }
      .migration-single-layout { grid-template-columns: 1fr; }
      .migration-filter-row { grid-template-columns: 1fr; }
      .migration-filter-row > .control-button { width: 100%; }
      .profile-intro ul { grid-template-columns: 1fr; }
      .econ-commentary { grid-template-columns: 1fr; }
      .profile-response-grid { grid-template-columns: 1fr; }
      .profile-map { height: 430px; }
    }
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>Climate Risk to Housing Markets</h1>
      <p class="dek">Housing markets across the US are affected by extreme climate events. The impact is uneven across different regions of the country. What climate risk does your home face?</p>
    </div>
  </header>

  <section id="riskResponseSection" class="story-section story-panel">
    <div class="visual-container">
      <div class="section-head">
        <div>
          <h2>Housing Market Response by FEMA Risk Level</h2>
          <div id="riskViewSubhead" class="sub">This chart compares counties by FEMA risk level. Each line shows the typical housing market path for counties in that risk group, from one year before an incident to two years after it ends.</div>
        </div>
      </div>
      <div class="control-row">
        <div id="riskViewToggle" class="button-row" aria-label="Toggle risk response view">
          <button type="button" class="control-button active" data-risk-view="chart">Line plot</button>
          <button type="button" class="control-button" data-risk-view="map">Map view</button>
        </div>
        <div id="riskPausedActions" class="button-row paused-actions" aria-label="Paused risk chart actions">
          <button id="riskChartResume" type="button" class="control-button">Resume</button>
        </div>
      </div>
      <div class="plot-with-legend">
        <div id="riskLegend" class="legend" aria-label="Select FEMA risk group"></div>
        <div>
          <div id="riskChartPanel" class="view-panel">
            <svg id="riskChart" class="chart" role="img" aria-label="Housing market index around FEMA incidents grouped by NRI risk rating"></svg>
          </div>
          <div id="riskMapPanel" class="view-panel" hidden>
            <svg id="riskMap" class="map" role="img" aria-label="US county map colored by FEMA National Risk Index rating"></svg>
          </div>
        </div>
      </div>
      <div id="riskCommentary" class="takeaway-banner" aria-label="Takeaway on NRI risk group responses"></div>
      <p id="riskIndexNote" class="note">Housing market index: this score combines prices, sale-to-list ratios, homes sold, and inventory into one number. The inputs use year-over-year changes so normal seasonal swings, like busier spring markets, have less influence.</p>
      <p id="riskWeightNote" class="note">Incident weighting: when a county was hit by multiple incidents, its values are averaged by month, with more recent incidents counted more heavily.</p>
      <div id="riskBottomJump" class="section-jump-row bottom-center edge-jump">
        <button id="riskDrillDown" type="button" class="icon-button" aria-label="Go to Economic Profile">&darr;</button>
      </div>
    </div>
  </section>
  <section id="economicProfileSection" class="story-panel" hidden>
    <div class="section-head">
      <div>
        <h2>Economic Profile</h2>
        <div class="sub">Counties do not all have the same local economy. This section groups similar counties together, then compares how those county types respond after climate-related incidents.</div>
      </div>
    </div>
    <div id="economicProfileIntro" class="profile-intro" aria-label="Economic profile section introduction"></div>
    <div id="economicProfileSummaryCards" class="profile-grid" aria-label="Plain-language economic profile summaries"></div>
    <div id="riskUpJump" class="section-jump-row top-center edge-jump">
      <button id="riskDrillUp" type="button" class="icon-button" aria-label="Return to FEMA risk response">&uarr;</button>
    </div>
    <div class="control-row">
      <div></div>
      <div id="economicFrameActions" class="button-row" aria-label="Economic profile animation actions">
        <button id="economicProfilePlayToggle" type="button" class="control-button">Pause</button>
      </div>
    </div>
    <div class="plot-with-legend plot-with-two-legends">
      <div id="economicRiskLegend" class="legend" aria-label="Select FEMA risk group for economic profile response"></div>
      <div aria-label="Selected risk group and economic profile housing response">
        <h3 id="economicProfileResponseTitle">Economic profiles within selected risk group</h3>
        <svg id="economicProfileResponseChart" class="chart economic-response-chart" role="img" aria-label="Housing response by economic profile within selected risk group"></svg>
      </div>
      <div id="economicProfileResponseLegend" class="legend" aria-label="Select economic profile frame"></div>
    </div>
    <div id="economicProfileResponseTakeaway" class="takeaway-banner" aria-label="Takeaway on economic profile housing response"></div>
    <h3 class="section-subheading">Risk x Economic Profile Response</h3>
    <div class="control-row">
      <div class="sub">Cells compare the median housing market YOY index shift for each risk group and economic profile pair.</div>
      <div id="economicMatrixMetricToggle" class="button-row" aria-label="Select economic response matrix metric">
        <button type="button" class="control-button active" data-economic-matrix-metric="later">Later response</button>
        <button type="button" class="control-button" data-economic-matrix-metric="early">Early response</button>
        <button type="button" class="control-button" data-economic-matrix-metric="momentum">Momentum</button>
      </div>
    </div>
    <div class="metric-explainer">
      <p><strong>Later response:</strong> months 13-24 after the incident minus the 12 months before the incident.</p>
      <p><strong>Early response:</strong> months 1-12 after the incident minus the 12 months before the incident.</p>
      <p><strong>Momentum:</strong> months 13-24 after the incident minus months 1-12 after the incident.</p>
    </div>
    <div class="heatmap-wrap">
      <div id="economicResponseMatrix" class="response-matrix" aria-label="Heatmap of housing response by NRI risk rating and economic profile"></div>
    </div>
    <div id="economicResponseMatrixTakeaway" class="takeaway-banner" aria-label="Takeaway on risk and economic profile housing response"></div>
    <h3 class="section-subheading">Economic Profile Mix by Risk Group</h3>
    <div class="heatmap-wrap">
      <div id="economicProfileHeatmap" class="profile-heatmap" aria-label="Heatmap of economic profile shares by NRI risk rating"></div>
    </div>
    <div id="economicProfileCommentary" class="econ-commentary key-takeaway" aria-label="Commentary on economic profile differences"></div>
    <p class="note">Economic profiles are derived through clustering of county income, wage, employment, income-source, population-size, and migration features. Assignment confidence is the model's probability that a county belongs to its assigned profile; higher confidence means the county is a clearer match.</p>
    <div id="economicBottomJump" class="section-jump-row bottom-center edge-jump">
      <button id="economicDrillDown" type="button" class="icon-button" aria-label="Go to Migration Trend">&darr;</button>
    </div>
  </section>
  <section id="migrationTrendSection" class="story-panel" hidden>
    <div class="section-head">
      <div>
        <h2>Migration Trend</h2>
        <div class="sub">Annual county net migration is converted to net migrants per 1,000 residents for the two years before through two years after incident occurrence. Extreme rates are capped for plotting so the middle pattern remains readable. Use the NRI Risk Rating and Economic Profile toggles to drill into either grouping, both groupings together, or neither.</div>
      </div>
    </div>
    <div id="migrationUpJump" class="section-jump-row top-center edge-jump">
      <button id="migrationDrillUp" type="button" class="icon-button" aria-label="Return to Economic Profile">&uarr;</button>
    </div>
    <div id="migrationFilterToggles" class="migration-filter-panel" aria-label="Migration trend drilldown filters">
      <div class="migration-filter-row">
        <button type="button" class="control-button" data-migration-filter-toggle="risk">NRI Risk Rating</button>
        <div id="migrationRiskFilterLegend" class="legend horizontal" aria-label="Select NRI risk rating for migration trend" hidden></div>
      </div>
      <div class="migration-filter-row">
        <button type="button" class="control-button" data-migration-filter-toggle="profile">Economic Profile</button>
        <div id="migrationProfileFilterLegend" class="legend horizontal" aria-label="Select economic profile for migration trend" hidden></div>
      </div>
    </div>
    <div class="migration-chart-only">
      <div>
        <h3 id="migrationChartTitle">Net migration per capita around incidents</h3>
        <svg id="migrationChart" class="chart migration-small-chart" role="img" aria-label="County net migration per 1,000 residents around FEMA incidents"></svg>
      </div>
    </div>
    <div id="migrationOverallTakeaway" class="takeaway-banner" aria-label="Overall takeaway on migration trend"></div>
    <p class="note">Net migration combines domestic and international migration from annual county population estimates. The rate is shown per 1,000 residents and capped at the 1st and 99th percentiles for the plot.</p>
    <h3 class="section-subheading">Migration Trend Clusters</h3>
    <div id="migrationClusterSummary" class="takeaway-banner" aria-label="Summary of migration trend clusters"></div>
    <div id="migrationClusterLegend" class="legend horizontal" aria-label="Select migration trend cluster"></div>
    <div class="migration-chart-only">
      <h3 id="migrationClusterChartTitle">Net migration per capita by migration trend cluster</h3>
      <svg id="migrationClusterChart" class="chart migration-small-chart" role="img" aria-label="County net migration per 1,000 residents around FEMA incidents by migration trend cluster"></svg>
    </div>
    <div id="migrationClusterCards" class="migration-cluster-grid" aria-label="Migration trend cluster interpretations"></div>
    <h3 class="section-subheading">Migration and Housing Market Relationship</h3>
    <div id="migrationRelationshipRiskLegend" class="legend horizontal" aria-label="Select NRI risk rating for migration and housing relationship"></div>
    <div class="migration-relationship-wrap">
      <svg id="migrationHousingScatter" class="chart" role="img" aria-label="County-level relationship between net migration change and housing market YOY index change"></svg>
    </div>
    <h3 class="section-subheading">Housing Market Movement by Migration Trend</h3>
    <div id="migrationHousingFilterToggles" class="migration-filter-panel" aria-label="Housing response drilldown filters">
      <div class="migration-filter-row">
        <button type="button" class="control-button" data-housing-filter-toggle="risk">NRI Risk Rating</button>
        <div id="migrationHousingRiskLegend" class="legend horizontal" aria-label="Select NRI risk rating for housing response" hidden></div>
      </div>
      <div class="migration-filter-row">
        <button type="button" class="control-button" data-housing-filter-toggle="economic">Economic Profile</button>
        <div id="migrationHousingEconomicLegend" class="legend horizontal" aria-label="Select economic profile for housing response" hidden></div>
      </div>
      <div class="migration-filter-row">
        <button type="button" class="control-button" data-housing-filter-toggle="migration">Migration Trend Group</button>
        <div id="migrationHousingClusterLegend" class="legend horizontal" aria-label="Select migration trend group for housing response" hidden></div>
      </div>
    </div>
    <div class="migration-chart-only">
      <h3 id="migrationHousingTitle">Housing market YOY index around incidents</h3>
      <svg id="migrationHousingTrendChart" class="chart migration-small-chart" role="img" aria-label="Housing market YOY index around FEMA incidents by selected migration drilldowns"></svg>
    </div>
    <div id="migrationHousingTakeaway" class="takeaway-banner" aria-label="Takeaway on housing movement by migration trend"></div>
    <div id="migrationBottomJump" class="section-jump-row bottom-center edge-jump">
      <button id="migrationDrillDown" type="button" class="icon-button" aria-label="Go to Insurance Profile">&darr;</button>
    </div>
  </section>
  <section id="insuranceProfileSection" class="story-panel" hidden>
    <div class="section-head">
      <div>
        <h2>Housing Insurance Trend</h2>
        <div class="sub">This section compares insurance conditions around incident timing, then connects insurance movement with housing-market response and migration. Use the metric buttons to switch the insurance y-axis between premium levels, premium growth, and non-renewal rates.</div>
      </div>
    </div>
    <div id="insuranceUpJump" class="section-jump-row top-center edge-jump">
      <button id="insuranceDrillUp" type="button" class="icon-button" aria-label="Return to Migration Trend">&uarr;</button>
    </div>
    <div id="insuranceMetricLegend" class="legend horizontal" aria-label="Select insurance trend metric"></div>
    <div id="insuranceTrendFilterToggles" class="migration-filter-panel" aria-label="Insurance trend drilldown filters">
      <div class="migration-filter-row">
        <button type="button" class="control-button" data-insurance-trend-filter-toggle="risk">NRI Risk Rating</button>
        <div id="insuranceTrendRiskLegend" class="legend horizontal" hidden></div>
      </div>
      <div class="migration-filter-row">
        <button type="button" class="control-button" data-insurance-trend-filter-toggle="economic">Economic Profile</button>
        <div id="insuranceTrendEconomicLegend" class="legend horizontal" hidden></div>
      </div>
      <div class="migration-filter-row">
        <button type="button" class="control-button" data-insurance-trend-filter-toggle="migration">Migration Trend Group</button>
        <div id="insuranceTrendMigrationLegend" class="legend horizontal" hidden></div>
      </div>
    </div>
    <div class="migration-chart-only">
      <h3 id="insuranceMetricTitle">Insurance trend around incidents</h3>
      <svg id="insuranceTrendChart" class="chart migration-small-chart" role="img" aria-label="County insurance trend around FEMA incidents"></svg>
    </div>
    <div id="insuranceTrendTakeaway" class="takeaway-banner" aria-label="Takeaway on selected insurance trend"></div>
    <h3 class="section-subheading">Insurance Profile Clusters</h3>
    <div id="insuranceClusterSummary" class="takeaway-banner" aria-label="Insurance profile cluster summary"></div>
    <div id="insuranceClusterCards" class="migration-cluster-grid" aria-label="Insurance profile cluster interpretations"></div>
    <h3 class="section-subheading">Housing Market Movement by Insurance Profile</h3>
    <div id="insuranceHousingFilterToggles" class="migration-filter-panel" aria-label="Insurance housing response drilldown filters">
      <div class="migration-filter-row">
        <button type="button" class="control-button" data-insurance-housing-filter-toggle="risk">NRI Risk Rating</button>
        <div id="insuranceHousingRiskLegend" class="legend horizontal" hidden></div>
      </div>
      <div class="migration-filter-row">
        <button type="button" class="control-button" data-insurance-housing-filter-toggle="economic">Economic Profile</button>
        <div id="insuranceHousingEconomicLegend" class="legend horizontal" hidden></div>
      </div>
      <div class="migration-filter-row">
        <button type="button" class="control-button" data-insurance-housing-filter-toggle="migration">Migration Trend Group</button>
        <div id="insuranceHousingMigrationLegend" class="legend horizontal" hidden></div>
      </div>
      <div class="migration-filter-row">
        <button type="button" class="control-button" data-insurance-housing-filter-toggle="insurance">Insurance Profile Type</button>
        <div id="insuranceHousingProfileLegend" class="legend horizontal" hidden></div>
      </div>
    </div>
    <div class="migration-chart-only">
      <h3 id="insuranceHousingTitle">Housing market YOY index around incidents</h3>
      <svg id="insuranceHousingTrendChart" class="chart migration-small-chart" role="img" aria-label="Housing market YOY index around incidents by selected insurance drilldowns"></svg>
    </div>
    <h3 class="section-subheading">Insurance, Housing, and Migration Relationship</h3>
    <div id="insuranceRelationshipMetricLegend" class="legend horizontal" aria-label="Select variables for relationship plot"></div>
    <div id="insuranceRelationshipRiskLegend" class="legend horizontal" aria-label="NRI risk rating colors for relationship plot"></div>
    <p class="note">Each point is a county. Point colors show the county's NRI risk rating.</p>
    <div class="migration-relationship-wrap">
      <svg id="insuranceRelationshipScatter" class="chart" role="img" aria-label="Relationship between selected insurance, housing, and migration changes"></svg>
    </div>
  </section>
  <div id="riskTooltip" class="tooltip" role="status" aria-live="polite"></div>
  <div id="mapTooltip" class="tooltip" role="status" aria-live="polite"></div>
</main>

<script>
const data = window.STORMHOUSE_DATA;
let usAtlasPromise = null;
let usCountyFeatures = null;
let activeRiskRating = data.meta.riskRatings[0];
let activeEconomicProfile = data.economicProfiles?.profiles?.[0]?.profile ?? null;
let activeEconomicMatrixMetric = "later";
let activeMigrationRating = data.meta.riskRatings[0];
let activeMigrationEconomicProfile = data.migrationTrend?.economicProfiles?.[0]?.profile ?? null;
let migrationRiskFilterEnabled = false;
let migrationProfileFilterEnabled = false;
let activeMigrationClusterProfile = data.migrationTrend?.trendProfiles?.profiles?.[0]?.profile ?? null;
let activeMigrationHousingRiskRating = data.meta.riskRatings[0];
let housingRiskFilterEnabled = false;
let housingEconomicFilterEnabled = false;
let housingMigrationFilterEnabled = false;
let activeHousingRiskRating = data.meta.riskRatings[0];
let activeHousingEconomicProfile = data.migrationTrend?.economicProfiles?.[0]?.profile ?? null;
let activeHousingMigrationProfile = data.migrationTrend?.trendProfiles?.profiles?.[0]?.profile ?? null;
let activeInsuranceEconomicProfile = data.migrationTrend?.economicProfiles?.[0]?.profile ?? null;
let activeInsuranceMetric = data.insuranceTrends?.metrics?.[0]?.key ?? null;
let activeInsuranceRiskMetric = data.insuranceTrends?.metrics?.[0]?.key ?? null;
let activeInsuranceRiskRating = data.meta.riskRatings[0];
let insuranceTrendRiskFilterEnabled = false;
let insuranceTrendEconomicFilterEnabled = false;
let insuranceTrendMigrationFilterEnabled = false;
let activeInsuranceTrendRiskRating = data.meta.riskRatings[0];
let activeInsuranceTrendEconomicProfile = data.migrationTrend?.economicProfiles?.[0]?.profile ?? null;
let activeInsuranceTrendMigrationProfile = data.migrationTrend?.trendProfiles?.profiles?.[0]?.profile ?? null;
let insuranceHousingRiskFilterEnabled = false;
let insuranceHousingEconomicFilterEnabled = false;
let insuranceHousingMigrationFilterEnabled = false;
let insuranceHousingProfileFilterEnabled = false;
let activeInsuranceHousingRiskRating = data.meta.riskRatings[0];
let activeInsuranceHousingEconomicProfile = data.migrationTrend?.economicProfiles?.[0]?.profile ?? null;
let activeInsuranceHousingMigrationProfile = data.migrationTrend?.trendProfiles?.profiles?.[0]?.profile ?? null;
let activeInsuranceHousingProfile = data.insuranceProfiles?.profiles?.[0]?.profile ?? null;
let activeInsuranceRelationshipPair = "insurance-housing";
let riskFrameIndex = 0;
let riskChartPaused = false;
let riskChartTimer = null;
let economicProfilePaused = false;
let economicProfileTimer = null;
let migrationPaused = false;
let migrationTimer = null;
let insuranceTrendPaused = false;
let insuranceTrendTimer = null;
let insuranceRiskTrendPaused = false;
let insuranceRiskTrendTimer = null;
let riskViewMode = "chart";
let activeStoryPanel = "riskResponseSection";
const riskChartFrameMs = 5000;
const economicProfileFrameMs = 3600;
const migrationFrameMs = 4200;
const insuranceTrendFrameMs = 4200;
const insuranceRiskTrendFrameMs = 4200;
const colors = {
  "Very Low": "#16a34a",
  "Low": "#84cc16",
  "Moderate": "#facc15",
  "High": "#f97316",
  "Very High": "#dc2626"
};
const profileColors = ["#005AB5", "#DC3220", "#009E73", "#F0E442", "#7A3E9D", "#8C564B", "#00A1C9", "#111111"];
const economicLineColors = ["#6D28D9", "#0891B2", "#BE185D", "#4338CA", "#0E7490", "#A21CAF", "#0369A1", "#831843"];
const insuranceMetricColors = {
  premiumLevel: "#EA580C",
  premiumYoy: "#0F766E",
  nonrenewalRate: "#B91C1C"
};
const allOffsets = data.meta.offsets;
function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function fmtPct(value) {
  if (value == null || Number.isNaN(value)) return "n/a";
  return `${Math.round(value * 100)}%`;
}

function fmtLift(value) {
  if (value == null || Number.isNaN(value)) return "n/a";
  return `${value > 0 ? "+" : ""}${Number(value).toFixed(2)} SD`;
}

function fmtConfidence(value) {
  if (value == null || Number.isNaN(value)) return "n/a";
  return `${Math.round(Number(value) * 100)}%`;
}

function liftPhrase(value, label = "") {
  if (value == null || Number.isNaN(value)) return "near typical";
  const abs = Math.abs(Number(value));
  const lowerLabel = String(label).toLowerCase();
  if (lowerLabel.includes("volatility")) {
    const direction = value > 0 ? "more volatile" : "less volatile";
    if (abs >= 1) return `much ${direction} than typical`;
    if (abs >= 0.5) return `${direction} than typical`;
    return `slightly ${direction} than typical`;
  }
  if (lowerLabel.includes("growth") || lowerLabel.includes("trend")) {
    const direction = value > 0 ? "faster" : "slower";
    if (abs >= 1) return `much ${direction} than typical`;
    if (abs >= 0.5) return `${direction} than typical`;
    return `slightly ${direction} than typical`;
  }
  const direction = value > 0 ? "higher" : "lower";
  if (abs >= 1) return `much ${direction} than typical`;
  if (abs >= 0.5) return `${direction} than typical`;
  return `slightly ${direction} than typical`;
}

function traitPhrase(trait) {
  return `${escapeHtml(trait.label)} is ${liftPhrase(trait.lift, trait.label)}`;
}

function economicLineColor(profileId) {
  return economicLineColors[Math.abs(Number(profileId) || 0) % economicLineColors.length] || "#6D28D9";
}

function extent(values) {
  let min = Infinity, max = -Infinity;
  values.forEach(v => {
    if (v == null || Number.isNaN(v)) return;
    min = Math.min(min, v);
    max = Math.max(max, v);
  });
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [-1, 1];
  if (min === max) return [min - 1, max + 1];
  const pad = (max - min) * 0.12;
  return [min - pad, max + pad];
}

function makeScale(domain, range) {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  return value => r0 + ((value - d0) / (d1 - d0)) * (r1 - r0);
}

function symmetricExtent(values) {
  const [min, max] = extent(values);
  const bound = Math.max(Math.abs(min), Math.abs(max), 0.01);
  return [-bound, bound];
}

function domainFromValues(values, { symmetric = false, includeZero = false } = {}) {
  const filtered = values.filter(value => value != null && !Number.isNaN(value));
  if (!filtered.length) return symmetric ? [-1, 1] : [0, 1];
  const source = includeZero ? [...filtered, 0] : filtered;
  return symmetric ? symmetricExtent(source) : extent(source);
}

function focusedDomainFromValues(values, { symmetric = false, includeZero = false, quantile = 0.92 } = {}) {
  const filtered = values.filter(value => value != null && !Number.isNaN(value)).map(Number);
  if (!filtered.length) return symmetric ? [-1, 1] : [0, 1];
  if (symmetric) {
    const absValues = filtered.map(value => Math.abs(value)).sort((a, b) => a - b);
    const index = Math.min(absValues.length - 1, Math.max(0, Math.floor((absValues.length - 1) * quantile)));
    const bound = Math.max(absValues[index] * 1.18, 0.01);
    return [-bound, bound];
  }
  const sorted = [...filtered].sort((a, b) => a - b);
  const lo = sorted[Math.min(sorted.length - 1, Math.max(0, Math.floor((sorted.length - 1) * (1 - quantile))))];
  const hi = sorted[Math.min(sorted.length - 1, Math.max(0, Math.floor((sorted.length - 1) * quantile)))];
  const pad = Math.max((hi - lo) * 0.18, 0.001);
  const min = includeZero ? Math.min(0, lo - pad) : lo - pad;
  const max = includeZero ? Math.max(0, hi + pad) : hi + pad;
  return min === max ? [min - 1, max + 1] : [min, max];
}

function fullDomainFromValues(values, { symmetric = false, includeZero = false, padRatio = 0.06 } = {}) {
  const filtered = values.filter(value => value != null && !Number.isNaN(value)).map(Number);
  if (!filtered.length) return symmetric ? [-1, 1] : [0, 1];
  if (symmetric) {
    const bound = Math.max(...filtered.map(value => Math.abs(value)), 0.01) * (1 + padRatio);
    return [-bound, bound];
  }
  const source = includeZero ? [...filtered, 0] : filtered;
  const minValue = Math.min(...source);
  const maxValue = Math.max(...source);
  const pad = Math.max((maxValue - minValue) * padRatio, 0.001);
  return minValue === maxValue ? [minValue - 1, maxValue + 1] : [minValue - pad, maxValue + pad];
}

function clampValue(value, domain) {
  if (value == null || Number.isNaN(value)) return value;
  const span = Math.max(domain[1] - domain[0], 0.0001);
  const inset = span * 0.006;
  return Math.min(domain[1] - inset, Math.max(domain[0] + inset, Number(value)));
}

function cappedSeriesRows(rows, yDomain) {
  return (rows || []).map(row => ({
    ...row,
    median: clampValue(row.median, yDomain),
    q1: clampValue(row.q1, yDomain),
    q3: clampValue(row.q3, yDomain),
  }));
}

function seriesExceedsDomain(rows, yDomain) {
  return (rows || []).some(row => [row.median, row.q1, row.q3].some(value => {
    const numeric = Number(value);
    return Number.isFinite(numeric) && (numeric < yDomain[0] || numeric > yDomain[1]);
  }));
}

function featureList(value) {
  if (Array.isArray(value)) return value;
  if (value == null) return [];
  if (typeof value === "string") {
    const trimmed = value.trim();
    if (!trimmed) return [];
    try {
      const parsed = JSON.parse(trimmed);
      if (Array.isArray(parsed)) return parsed;
    } catch (_) {}
    return trimmed.split(/[,;|]/).map(item => item.trim()).filter(Boolean);
  }
  if (typeof value === "object") return Object.values(value).filter(item => item != null);
  return [value];
}

function plainProfileFeature(value) {
  return String(value ?? "")
    .replace(/\([^)]*\)/g, "")
    .replace(/\b(latest|mean|median|average|percentile|national|state)\b/gi, "")
    .replace(/[_-]/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .toLowerCase();
}

function uniquePlainFeatures(value, limit = 3) {
  const out = [];
  featureList(value).forEach(item => {
    const text = plainProfileFeature(item);
    if (!text) return;
    const duplicate = out.some(existing => existing.includes(text) || text.includes(existing));
    if (!duplicate) out.push(text);
  });
  return out.slice(0, limit);
}

function economicFeatureMagnitude(value) {
  const match = String(value ?? "").match(/([-+]?\d*\.?\d+)\s*SD/i);
  return match ? Math.abs(Number(match[1])) : 0;
}

function economicFeatureKey(value) {
  const raw = String(value ?? "").toLowerCase();
  const isTrend = raw.includes("trend");
  const base = raw
    .replace(/\([^)]*\)/g, "")
    .replace(/\b(latest|mean|median|average|percentile|national|state|trend)\b/gi, "")
    .replace(/[_-]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  return `${base}${isTrend ? " trend" : ""}`;
}

function economicFeaturePhrase(value) {
  const raw = String(value ?? "").toLowerCase();
  const isTrend = raw.includes("trend");
  if (raw.includes("per capita income")) return isTrend ? "rising per-person incomes" : "higher per-person incomes";
  if (raw.includes("dividends") || raw.includes("interest") || raw.includes("rent")) return "more income from dividends, interest, and rent";
  if (raw.includes("average weekly wage")) return isTrend ? "faster wage growth" : "higher wages";
  if (raw.includes("transfer receipts")) return "more household income from transfer payments";
  if (raw.includes("domestic migration")) return isTrend ? "improving domestic migration" : "stronger domestic in-migration";
  if (raw.includes("international migration")) return isTrend ? "improving international migration" : "stronger international migration";
  if (raw.includes("employment per resident")) return isTrend ? "changing jobs per resident" : "more jobs per resident";
  if (raw.includes("proprietors income")) return "more proprietor and self-employment income";
  if (raw.includes("natural increase")) return "more population growth from births minus deaths";
  if (raw.includes("population scale")) return isTrend ? "population growth" : "larger county populations";
  return plainProfileFeature(value);
}

function standoutEconomicFeatures(value, limit = 4) {
  const seen = new Set();
  const out = [];
  featureList(value)
    .filter(item => economicFeatureMagnitude(item) >= 0.05)
    .forEach(item => {
      const key = economicFeatureKey(item);
      const phrase = economicFeaturePhrase(item);
      if (!phrase || seen.has(key)) return;
      seen.add(key);
      out.push(phrase);
    });
  return out.slice(0, limit);
}

function economicResponseDomain() {
  const profileValues = Object.values(data.economicProfiles?.responseByRiskRating || {})
    .flatMap(series => series.flatMap(item => (item.rows || []).flatMap(row => [row.q1, row.q3, row.median])));
  const riskValues = Object.values(data.byRiskRating || {})
    .flatMap(rows => (rows || []).flatMap(row => [row.q1, row.q3, row.median]));
  return focusedDomainFromValues([...profileValues, ...riskValues], { symmetric: true, includeZero: true, quantile: 0.9 });
}

function countyHousingDomain() {
  const values = (data.migrationTrend?.housingSeries || []).map(row => row.value);
  return focusedDomainFromValues(values, { symmetric: true, includeZero: true, quantile: 0.9 });
}

function migrationDomain() {
  const overall = (data.migrationTrend?.overall || [])
    .flatMap(row => [row.q1, row.q3, row.median]);
  const byProfile = Object.values(data.migrationTrend?.byEconomicProfile || {})
    .flatMap(rows => (rows || []).flatMap(row => [row.q1, row.q3, row.median]));
  const byProfileRisk = Object.values(data.migrationTrend?.byEconomicProfileRiskRating || {})
    .flatMap(byRisk => Object.values(byRisk || {}).flatMap(rows => (rows || []).flatMap(row => [row.q1, row.q3, row.median])));
  const byRisk = Object.values(data.migrationTrend?.byRiskRating || {})
    .flatMap(rows => (rows || []).flatMap(row => [row.q1, row.q3, row.median]));
  return focusedDomainFromValues([...overall, ...byProfile, ...byProfileRisk, ...byRisk], { symmetric: true, includeZero: true, quantile: 0.9 });
}

function migrationClusterDomain() {
  const clusterValues = Object.values(data.migrationTrend?.trendProfiles?.byProfile || {})
    .flatMap(rows => (rows || []).flatMap(row => [row.q1, row.q3, row.median]));
  return focusedDomainFromValues(clusterValues, { symmetric: true, includeZero: true, quantile: 0.9 });
}

function insuranceMetricDomain(metricKey) {
  const metric = insuranceMetrics().find(item => item.key === metricKey);
  const overallValues = (data.insuranceTrends?.overallByMetric?.[metricKey] || [])
    .flatMap(row => [row.q1, row.q3, row.median]);
  const profileValues = Object.values(data.insuranceTrends?.byEconomicProfileMetric || {})
    .flatMap(byMetric => (byMetric?.[metricKey] || []).flatMap(row => [row.q1, row.q3, row.median]));
  const riskValues = Object.values(data.insuranceTrends?.byRiskRatingMetric || {})
    .flatMap(byMetric => (byMetric?.[metricKey] || []).flatMap(row => [row.q1, row.q3, row.median]));
  const values = [...overallValues, ...profileValues, ...riskValues];
  return focusedDomainFromValues(values, { symmetric: !!metric?.centerZero, includeZero: !!metric?.centerZero, quantile: 0.9 });
}

function metricFormat(meta, value) {
  if (value == null || Number.isNaN(value)) return "n/a";
  if (meta?.format === "currency") return `$${Math.round(Number(value)).toLocaleString()}`;
  if (meta?.key === "nonrenewalRate") return `${(Number(value) * 100).toFixed(2)}%`;
  return `${(Number(value) * 100).toFixed(0)}%`;
}

function pathFor(rows, x, y, key = "median") {
  return rows.map((d, i) => `${i === 0 ? "M" : "L"}${x(d.offset).toFixed(2)},${y(d[key]).toFixed(2)}`).join(" ");
}

function bandPath(rows, x, y) {
  const top = rows.map((d, i) => `${i === 0 ? "M" : "L"}${x(d.offset).toFixed(2)},${y(d.q3).toFixed(2)}`);
  const bottom = rows.slice().reverse().map(d => `L${x(d.offset).toFixed(2)},${y(d.q1).toFixed(2)}`);
  return `${top.join(" ")} ${bottom.join(" ")} Z`;
}

function seriesEndpoint(rows, offset) {
  return (rows || []).find(row => Number(row.offset) === Number(offset));
}

function showLineSeriesTooltip(event, title, rows, formatter = formatChange, countyCount = null) {
  const tooltip = document.getElementById("riskTooltip");
  const sorted = (rows || []).slice().sort((a, b) => Number(a.offset) - Number(b.offset));
  const first = sorted[0];
  const last = sorted[sorted.length - 1];
  const trend = first?.median != null && last?.median != null
    ? (last.median > first.median ? "moves higher across the window" : last.median < first.median ? "moves lower across the window" : "stays broadly level")
    : "has an unclear window trend";
  const middle = seriesEndpoint(sorted, 0);
  tooltip.innerHTML = `<strong>${escapeHtml(title)}</strong>` +
    `<div>Counties: ${countyCount == null ? "n/a" : Number(countyCount).toLocaleString()}</div>` +
    `<div>${trend}</div>` +
    `<div>Start: ${formatter(first?.median)}</div>` +
    `<div>Incident: ${formatter(middle?.median)}</div>` +
    `<div>End: ${formatter(last?.median)}</div>` +
    `<div>Typical spread: ${first?.q1 != null && first?.q3 != null ? `${formatter(first.q1)} to ${formatter(first.q3)}` : "n/a"}</div>`;
  tooltip.style.display = "block";
  moveTooltip(event);
}

function uniqueCountyCount(rows) {
  return new Set((rows || []).map(row => row.fips).filter(Boolean)).size;
}

function clear(svg) {
  while (svg.firstChild) svg.removeChild(svg.firstChild);
}

function add(tag, parent, attrs = {}) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));
  parent.appendChild(node);
  return node;
}

function quantileSorted(values, q) {
  if (!values.length) return null;
  const pos = (values.length - 1) * q;
  const base = Math.floor(pos);
  const rest = pos - base;
  return values[base + 1] == null ? values[base] : values[base] + rest * (values[base + 1] - values[base]);
}

function summarizeSeriesRows(rows, offsetField = "offset", valueField = "value") {
  const byOffset = new Map();
  rows.forEach(row => {
    const offset = Number(row[offsetField]);
    const value = Number(row[valueField]);
    if (Number.isNaN(offset) || Number.isNaN(value)) return;
    if (!byOffset.has(offset)) byOffset.set(offset, []);
    byOffset.get(offset).push(value);
  });
  return Array.from(byOffset.entries()).sort((a, b) => a[0] - b[0]).map(([offset, values]) => {
    const sorted = values.slice().sort((a, b) => a - b);
    return {
      offset,
      median: quantileSorted(sorted, 0.5),
      q1: quantileSorted(sorted, 0.25),
      q3: quantileSorted(sorted, 0.75),
      n: sorted.length
    };
  });
}

function drawAxes(svg, dims, x, y, xDomain, yDomain) {
  const g = add("g", svg);
  const yTicks = 5;
  for (let i = 0; i <= yTicks; i++) {
    const value = yDomain[0] + ((yDomain[1] - yDomain[0]) * i / yTicks);
    const yy = y(value);
    add("line", g, { x1: dims.left, x2: dims.width - dims.right, y1: yy, y2: yy, stroke: "#e5e0d7" });
    add("text", g, { x: dims.left - 8, y: yy + 4, "text-anchor": "end" }).textContent = value.toFixed(2);
  }
  const tickValues = [-12, -6, -1, 1, 6, 12, 18, 24].filter(d => d >= xDomain[0] && d <= xDomain[1]);
  tickValues.forEach(value => {
    const xx = x(value);
    add("line", g, { x1: xx, x2: xx, y1: dims.top, y2: dims.height - dims.bottom, stroke: "#eee9df" });
    add("text", g, { x: xx, y: dims.height - dims.bottom + 22, "text-anchor": "middle" }).textContent = value;
  });
  add("line", g, { x1: dims.left, x2: dims.width - dims.right, y1: y(0), y2: y(0), class: "zero-line" });
  add("text", g, { x: dims.left, y: dims.height - 8 }).textContent = "Month offset";
  add("text", g, { x: 4, y: dims.top - 8 }).textContent = "Housing market YOY index";
}

function drawCompactAxes(svg, dims, x, y, xDomain, yDomain) {
  const g = add("g", svg);
  const yTicks = 3;
  for (let i = 0; i <= yTicks; i++) {
    const value = yDomain[0] + ((yDomain[1] - yDomain[0]) * i / yTicks);
    const yy = y(value);
    add("line", g, { x1: dims.left, x2: dims.width - dims.right, y1: yy, y2: yy, stroke: "#e5e0d7" });
    add("text", g, { x: dims.left - 6, y: yy + 3, "text-anchor": "end" }).textContent = value.toFixed(2);
  }
  [-12, 0, 12, 24].filter(d => d >= xDomain[0] && d <= xDomain[1]).forEach(value => {
    const xx = x(value);
    add("line", g, { x1: xx, x2: xx, y1: dims.top, y2: dims.height - dims.bottom, stroke: "#eee9df" });
    add("text", g, { x: xx, y: dims.height - dims.bottom + 17, "text-anchor": "middle" }).textContent = value;
  });
  add("line", g, { x1: dims.left, x2: dims.width - dims.right, y1: y(0), y2: y(0), class: "zero-line" });
  add("text", g, { x: dims.width - dims.right, y: dims.height - 7, "text-anchor": "end" }).textContent = "Month offset";
}

function drawMigrationAxes(svg, dims, x, y, xDomain, yDomain) {
  const g = add("g", svg);
  const yTicks = 5;
  for (let i = 0; i <= yTicks; i++) {
    const value = yDomain[0] + ((yDomain[1] - yDomain[0]) * i / yTicks);
    const yy = y(value);
    add("line", g, { x1: dims.left, x2: dims.width - dims.right, y1: yy, y2: yy, stroke: "#e5e0d7" });
    add("text", g, { x: dims.left - 8, y: yy + 4, "text-anchor": "end" }).textContent = Number(value).toFixed(1);
  }
  const tickValues = [];
  for (let value = Math.ceil(xDomain[0]); value <= Math.floor(xDomain[1]); value += 1) tickValues.push(value);
  tickValues.forEach(value => {
    const xx = x(value);
    add("line", g, { x1: xx, x2: xx, y1: dims.top, y2: dims.height - dims.bottom, stroke: "#eee9df" });
    add("text", g, { x: xx, y: dims.height - dims.bottom + 22, "text-anchor": "middle" }).textContent = value;
  });
  add("line", g, { x1: dims.left, x2: dims.width - dims.right, y1: y(0), y2: y(0), class: "zero-line" });
  add("text", g, { x: dims.left, y: dims.height - 8 }).textContent = "Year offset";
  add("text", g, { x: 4, y: dims.top - 8 }).textContent = "Net migration per 1,000 residents";
}

function chartDims(svg) {
  const box = svg.getBoundingClientRect();
  return { width: Math.max(320, box.width), height: Math.max(300, box.height), top: 26, right: 22, bottom: 46, left: 58 };
}

function drawPeriodShading(svg, dims, x) {
  const y = dims.top;
  const height = dims.height - dims.bottom - dims.top;
  add("rect", svg, { x: x(1), y, width: x(12) - x(1), height, class: "period-band-early" });
  add("rect", svg, { x: x(13), y, width: x(24) - x(13), height, class: "period-band-late" });
  add("text", svg, { x: x(6.5), y: dims.top + 13, class: "period-label", "text-anchor": "middle" }).textContent = "Months 1-12";
  add("text", svg, { x: x(18.5), y: dims.top + 13, class: "period-label", "text-anchor": "middle" }).textContent = "Months 13-24";
}

function drawCompactPeriodShading(svg, dims, x) {
  const y = dims.top;
  const height = dims.height - dims.bottom - dims.top;
  add("rect", svg, { x: x(1), y, width: x(12) - x(1), height, class: "period-band-early" });
  add("rect", svg, { x: x(13), y, width: x(24) - x(13), height, class: "period-band-late" });
}

function formatChange(value) {
  if (value == null || Number.isNaN(value)) return "n/a";
  const fixed = Math.abs(value).toFixed(3);
  return `${value > 0 ? "+" : value < 0 ? "-" : ""}${fixed}`;
}

function formatMigration(value) {
  if (value == null || Number.isNaN(value)) return "n/a";
  return `${Number(value).toFixed(1)} per 1,000`;
}

function showTooltip(event, rating) {
  const tooltip = document.getElementById("riskTooltip");
  const summary = data.groupSummaries[rating] || {};
  tooltip.innerHTML = `<strong>${rating} risk</strong>` +
    `<div>Counties: ${summary.countyCount ?? "n/a"}</div>` +
    `<div>Avg. shift from before to year 1: ${formatChange(summary.avgPreToMonths1To12)}</div>` +
    `<div>Avg. shift from year 1 to year 2: ${formatChange(summary.avgMonths1To12To13To24)}</div>`;
  tooltip.style.display = "block";
  moveTooltip(event);
}

function moveTooltip(event) {
  const tooltip = document.getElementById("riskTooltip");
  const offset = 14;
  tooltip.style.left = `${Math.min(event.clientX + offset, window.innerWidth - tooltip.offsetWidth - offset)}px`;
  tooltip.style.top = `${Math.min(event.clientY + offset, window.innerHeight - tooltip.offsetHeight - offset)}px`;
}

function hideTooltip() {
  document.getElementById("riskTooltip").style.display = "none";
}

function riskLineGroups() {
  return data.meta.riskRatings.map(rating => ({
    rating,
    rows: (data.byRiskRating[rating] || []).filter(d => allOffsets.includes(d.offset))
  })).filter(group => group.rows.length);
}

function migrationLineGroups() {
  const byRisk = data.migrationTrend?.byRiskRating || {};
  const offsets = data.migrationTrend?.meta?.offsets || [-2, -1, 0, 1, 2];
  return data.meta.riskRatings.map(rating => ({
    rating,
    rows: (byRisk[rating] || []).filter(d => offsets.includes(d.offset))
  })).filter(group => group.rows.length);
}

function migrationEconomicProfiles() {
  return data.migrationTrend?.economicProfiles || [];
}

function migrationEconomicGroups() {
  const byProfile = data.migrationTrend?.byEconomicProfile || {};
  const offsets = data.migrationTrend?.meta?.offsets || [-2, -1, 0, 1, 2];
  return migrationEconomicProfiles().map(profile => ({
    profile: profile.profile,
    label: profile.label,
    rows: (byProfile[String(profile.profile)] || []).filter(d => offsets.includes(d.offset))
  })).filter(group => group.rows.length);
}

function migrationRiskGroupsForProfile(profileId = activeMigrationEconomicProfile) {
  const offsets = data.migrationTrend?.meta?.offsets || [-2, -1, 0, 1, 2];
  return data.meta.riskRatings.map(rating => ({
    rating,
    rows: migrationRowsForFilter({ riskEnabled: true, profileEnabled: true, rating, profileId }).filter(d => offsets.includes(d.offset))
  })).filter(group => group.rows.length);
}

function activeMigrationEconomicGroup() {
  const groups = migrationEconomicGroups();
  if (!groups.length) return null;
  const current = groups.find(group => Number(group.profile) === Number(activeMigrationEconomicProfile));
  if (current) return current;
  activeMigrationEconomicProfile = groups[0].profile;
  return groups[0];
}

function availableMigrationRiskRatings(profileId = activeMigrationEconomicProfile) {
  return migrationRiskGroupsForProfile(profileId).map(group => group.rating);
}

function ensureActiveMigrationRiskRating() {
  const ratings = availableMigrationRiskRatings(activeMigrationEconomicProfile);
  if (!ratings.length) {
    activeMigrationRating = null;
    return null;
  }
  if (!ratings.includes(activeMigrationRating)) {
    activeMigrationRating = ratings[0];
  }
  return activeMigrationRating;
}

function drawRiskChart(svgId = "riskChart", focusRating = activeRiskRating, compact = false) {
  const svg = document.getElementById(svgId);
  if (!svg) return;
  clear(svg);
  const dims = compact ? profileResponseDims(svg) : chartDims(svg);
  svg.setAttribute("viewBox", `0 0 ${dims.width} ${dims.height}`);
  const offsets = allOffsets;
  const groups = riskLineGroups();
  const yValues = groups.flatMap(group => group.rows.flatMap(d => [d.q1, d.q3, d.median]));
  const xDomain = [Math.min(...offsets), Math.max(...offsets)];
  const yDomain = focusedDomainFromValues(yValues, { symmetric: true, includeZero: true, quantile: 0.9 });
  const x = makeScale(xDomain, [dims.left, dims.width - dims.right]);
  const y = makeScale(yDomain, [dims.height - dims.bottom, dims.top]);
  if (compact) {
    drawCompactPeriodShading(svg, dims, x);
    drawCompactAxes(svg, dims, x, y, xDomain, yDomain);
  } else {
    drawPeriodShading(svg, dims, x);
    drawAxes(svg, dims, x, y, xDomain, yDomain);
  }
  const incidentX = x(0);
  add("line", svg, { x1: incidentX, x2: incidentX, y1: dims.top, y2: dims.height - dims.bottom, class: "incident-line" });
  if (!compact) {
    add("text", svg, { x: incidentX + 7, y: dims.top + 14, class: "incident-label" }).textContent = "Incident";
  }
  const visibleGroups = compact ? groups : groups.filter((group, index) => index <= riskFrameIndex);
  visibleGroups.forEach(group => {
    const isFocused = focusRating === group.rating;
    const isDimmed = focusRating && !isFocused;
    const drawRows = cappedSeriesRows(group.rows, yDomain);
    const linePath = pathFor(drawRows, x, y);
    const bandClass = `risk-band${isFocused ? " active" : " background"}`;
    const lineClass = `risk-line${isFocused ? " focused" : ""}${isDimmed ? " dimmed" : ""}`;
    const band = add("path", svg, { d: bandPath(drawRows, x, y), class: bandClass, fill: colors[group.rating] || "#5e6872", "data-rating": group.rating });
    add("path", svg, { d: linePath, class: lineClass, stroke: colors[group.rating] || "#5e6872", "data-rating": group.rating });
    const hitLine = add("path", svg, { d: linePath, class: "risk-hit-line", "data-rating": group.rating });
    hitLine.addEventListener("mouseenter", event => {
      band.classList.remove("background");
      band.classList.add("active");
      showTooltip(event, group.rating);
    });
    hitLine.addEventListener("mousemove", moveTooltip);
    hitLine.addEventListener("mouseleave", () => {
      if (!isFocused) {
        band.classList.remove("active");
        band.classList.add("background");
      }
      hideTooltip();
    });
  });
}

function drawMigrationSeriesChart(svgId, groups, activeKey, keyField, colorForGroup, tooltipForGroup) {
  const svg = document.getElementById(svgId);
  if (!svg) return;
  clear(svg);
  const dims = chartDims(svg);
  svg.setAttribute("viewBox", `0 0 ${dims.width} ${dims.height}`);
  const offsets = data.migrationTrend?.meta?.offsets || [-2, -1, 0, 1, 2];
  const xDomain = [Math.min(...offsets), Math.max(...offsets)];
  const yDomain = migrationDomain();
  const x = makeScale(xDomain, [dims.left, dims.width - dims.right]);
  const y = makeScale(yDomain, [dims.height - dims.bottom, dims.top]);
  drawMigrationAxes(svg, dims, x, y, xDomain, yDomain);
  const incidentX = x(0);
  add("line", svg, { x1: incidentX, x2: incidentX, y1: dims.top, y2: dims.height - dims.bottom, class: "incident-line" });
  add("text", svg, { x: incidentX + 7, y: dims.top + 14, class: "incident-label" }).textContent = "Incident";
  groups.forEach(group => {
    const key = group[keyField];
    const isFocused = activeKey === key;
    const isDimmed = activeKey != null && !isFocused;
    const drawRows = cappedSeriesRows(group.rows, yDomain);
    const linePath = pathFor(drawRows, x, y);
    const bandClass = `risk-band${isFocused ? " active" : " background"}`;
    const lineClass = `risk-line${isFocused ? " focused" : ""}${isDimmed ? " dimmed" : ""}`;
    const color = colorForGroup(group);
    const band = add("path", svg, { d: bandPath(drawRows, x, y), class: bandClass, fill: color });
    add("path", svg, { d: linePath, class: lineClass, stroke: color });
    const hitLine = add("path", svg, { d: linePath, class: "risk-hit-line" });
    hitLine.addEventListener("mouseenter", event => {
      band.classList.remove("background");
      band.classList.add("active");
      tooltipForGroup(event, group);
    });
    hitLine.addEventListener("mousemove", event => tooltipForGroup(event, group));
    hitLine.addEventListener("mouseleave", () => {
      if (!isFocused) {
        band.classList.remove("active");
        band.classList.add("background");
      }
      hideTooltip();
    });
  });
}

function drawMigrationEconomicChart() {
  const activeGroup = activeMigrationEconomicGroup();
  const title = document.getElementById("migrationEconomicTitle");
  if (title) title.textContent = activeGroup ? `Migration by economic profile: ${activeGroup.label}` : "Migration by economic profile";
  drawMigrationSeriesChart(
    "migrationEconomicChart",
    activeGroup ? [activeGroup] : [],
    activeGroup?.profile,
    "profile",
    group => economicLineColor(group.profile),
    showMigrationEconomicTooltip
  );
}

function drawMigrationRiskBreakdownChart() {
  const activeGroup = activeMigrationEconomicGroup();
  const activeRating = ensureActiveMigrationRiskRating();
  const title = document.getElementById("migrationRiskTitle");
  if (title) title.textContent = activeGroup ? `Risk groups within ${activeGroup.label}` : "Risk groups within selected economic profile";
  drawMigrationSeriesChart(
    "migrationChart",
    migrationRiskGroupsForProfile(activeMigrationEconomicProfile).filter(group => group.rating === activeRating),
    activeRating,
    "rating",
    group => colors[group.rating] || "#5e6872",
    showMigrationTooltip
  );
}

function migrationProfileLabel(profileId) {
  const profile = migrationEconomicProfiles().find(item => Number(item.profile) === Number(profileId));
  return profile?.label || `Profile ${profileId}`;
}

function migrationSelectedSeries() {
  const profileId = activeMigrationEconomicProfile;
  const rating = activeMigrationRating;
  if (migrationRiskFilterEnabled && migrationProfileFilterEnabled) {
    const rows = data.migrationTrend?.byEconomicProfileRiskRating?.[String(profileId)]?.[rating] || [];
    return {
      rows,
      label: `${rating} risk, ${migrationProfileLabel(profileId)}`,
      color: economicLineColor(profileId),
      summary: data.migrationTrend?.profileRiskSummaries?.[String(profileId)]?.[rating] || {},
      type: "risk-profile"
    };
  }
  if (migrationRiskFilterEnabled) {
    return {
      rows: data.migrationTrend?.byRiskRating?.[rating] || [],
      label: `${rating} risk counties`,
      color: colors[rating] || "#5e6872",
      summary: data.migrationTrend?.summaries?.[rating] || {},
      type: "risk"
    };
  }
  if (migrationProfileFilterEnabled) {
    return {
      rows: data.migrationTrend?.byEconomicProfile?.[String(profileId)] || [],
      label: migrationProfileLabel(profileId),
      color: economicLineColor(profileId),
      summary: data.migrationTrend?.profileSummaries?.[String(profileId)] || {},
      type: "profile"
    };
  }
  return {
    rows: data.migrationTrend?.overall || [],
    label: "All counties",
    color: "#2f3941",
    summary: data.migrationTrend?.overallSummary || {},
    type: "overall"
  };
}

function migrationRowsForFilter({ riskEnabled = migrationRiskFilterEnabled, profileEnabled = migrationProfileFilterEnabled, rating = activeMigrationRating, profileId = activeMigrationEconomicProfile } = {}) {
  if (riskEnabled && profileEnabled) return data.migrationTrend?.byEconomicProfileRiskRating?.[String(profileId)]?.[rating] || [];
  if (riskEnabled) return data.migrationTrend?.byRiskRating?.[rating] || [];
  if (profileEnabled) return data.migrationTrend?.byEconomicProfile?.[String(profileId)] || [];
  return data.migrationTrend?.overall || [];
}

function showMigrationSeriesTooltip(event, series) {
  const summary = series.summary || {};
  const count = Number(summary.countyCount) || 0;
  const start = summary.minus2;
  const end = summary.plus2;
  const trend = start != null && end != null ? migrationShiftPhrase(start, end) : "has an unclear before-to-after pattern";
  const tooltip = document.getElementById("riskTooltip");
  tooltip.innerHTML = `<strong>${escapeHtml(series.label)}</strong>` +
    `${count.toLocaleString()} counties<br>` +
    `Two years before: ${formatMigration(start)}<br>` +
    `Two years after: ${formatMigration(end)}<br>` +
    `Across the window: ${trend}`;
  tooltip.style.display = "block";
  moveTooltip(event);
}

function drawMigrationSingleChart() {
  const svg = document.getElementById("migrationChart");
  if (!svg) return;
  clear(svg);
  const dims = chartDims(svg);
  svg.setAttribute("viewBox", `0 0 ${dims.width} ${dims.height}`);
  const offsets = data.migrationTrend?.meta?.offsets || [-2, -1, 0, 1, 2];
  const xDomain = [Math.min(...offsets), Math.max(...offsets)];
  const yDomain = migrationDomain();
  const x = makeScale(xDomain, [dims.left, dims.width - dims.right]);
  const y = makeScale(yDomain, [dims.height - dims.bottom, dims.top]);
  drawMigrationAxes(svg, dims, x, y, xDomain, yDomain);
  const incidentX = x(0);
  add("line", svg, { x1: incidentX, x2: incidentX, y1: dims.top, y2: dims.height - dims.bottom, class: "incident-line" });
  add("text", svg, { x: incidentX + 7, y: dims.top + 14, class: "incident-label" }).textContent = "Incident";
  const series = migrationSelectedSeries();
  const title = document.getElementById("migrationChartTitle");
  if (title) title.textContent = `Net migration per capita: ${series.label}`;
  if (!series.rows.length) {
    add("text", svg, { x: dims.left, y: dims.top + 28, class: "axis-label" }).textContent = "No measured counties for this filter combination.";
    return;
  }
  const drawRows = cappedSeriesRows(series.rows, yDomain);
  const linePath = pathFor(drawRows, x, y);
  const band = add("path", svg, { d: bandPath(drawRows, x, y), class: "risk-band active", fill: series.color });
  add("path", svg, { d: linePath, class: "risk-line focused", stroke: series.color });
  const hitLine = add("path", svg, { d: linePath, class: "risk-hit-line" });
  hitLine.addEventListener("mouseenter", event => {
    band.classList.add("active");
    showMigrationSeriesTooltip(event, series);
  });
  hitLine.addEventListener("mousemove", event => showMigrationSeriesTooltip(event, series));
  hitLine.addEventListener("mouseleave", hideTooltip);
}

function drawMigrationChart() {
  drawMigrationFilterControls();
  drawMigrationSingleChart();
  updateMigrationOverallTakeaway();
  drawMigrationHousingResponseModule();
  drawMigrationTrendClusters();
  drawMigrationHousingScatter();
}

function migrationClusterProfiles() {
  return data.migrationTrend?.trendProfiles?.profiles || [];
}

function insuranceProfiles() {
  return data.insuranceProfiles?.profiles || [];
}

function filterHousingSeries(config) {
  return (data.migrationTrend?.housingSeries || []).filter(row => {
    if (config.riskEnabled && row.riskRating !== config.riskRating) return false;
    if (config.economicEnabled && Number(row.economicProfile) !== Number(config.economicProfile)) return false;
    if (config.migrationEnabled && Number(row.migrationProfile) !== Number(config.migrationProfile)) return false;
    if (config.insuranceEnabled && Number(row.insuranceProfile) !== Number(config.insuranceProfile)) return false;
    return row.value != null;
  });
}

function housingSelectionLabel(config) {
  const parts = [];
  if (config.riskEnabled) parts.push(`${config.riskRating} risk`);
  if (config.economicEnabled) parts.push(migrationProfileLabel(config.economicProfile));
  if (config.migrationEnabled) {
    const profile = migrationClusterProfiles().find(item => Number(item.profile) === Number(config.migrationProfile));
    parts.push(profile?.label || `Migration profile ${config.migrationProfile}`);
  }
  if (config.insuranceEnabled) {
    const profile = insuranceProfiles().find(item => Number(item.profile) === Number(config.insuranceProfile));
    parts.push(profile?.label || `Insurance profile ${config.insuranceProfile}`);
  }
  return parts.length ? parts.join(", ") : "All counties";
}

function drawProfileButtonLegend(legendId, profiles, activeValue, onSelect, colorForProfile, hasRows = null) {
  const legend = document.getElementById(legendId);
  if (!legend) return;
  legend.innerHTML = "";
  profiles.filter(profile => !hasRows || hasRows(profile)).forEach(profile => {
    const item = document.createElement("button");
    item.type = "button";
    item.className = `risk-toggle-button${Number(profile.profile) === Number(activeValue) ? " active" : ""}`;
    item.setAttribute("aria-pressed", Number(profile.profile) === Number(activeValue) ? "true" : "false");
    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.background = colorForProfile(profile);
    item.appendChild(swatch);
    item.appendChild(document.createTextNode(profile.label));
    item.addEventListener("click", () => onSelect(profile.profile));
    legend.appendChild(item);
  });
}

function drawRiskButtonLegend(legendId, activeValue, onSelect, hasRows = null) {
  const legend = document.getElementById(legendId);
  if (!legend) return;
  legend.innerHTML = "";
  data.meta.riskRatings.filter(rating => !hasRows || hasRows(rating)).forEach(rating => {
    const item = document.createElement("button");
    item.type = "button";
    item.className = `risk-toggle-button${rating === activeValue ? " active" : ""}`;
    item.setAttribute("aria-pressed", rating === activeValue ? "true" : "false");
    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.background = colors[rating] || "#5e6872";
    item.appendChild(swatch);
    item.appendChild(document.createTextNode(rating));
    item.addEventListener("click", () => onSelect(rating));
    legend.appendChild(item);
  });
}

function drawHousingTrendChart(svgId, titleId, rows, label, color = "#7c3aed") {
  const svg = document.getElementById(svgId);
  if (!svg) return;
  clear(svg);
  const title = document.getElementById(titleId);
  if (title) title.textContent = `Housing market YOY index: ${label}`;
  const dims = chartDims(svg);
  svg.setAttribute("viewBox", `0 0 ${dims.width} ${dims.height}`);
  const series = summarizeSeriesRows(rows);
  const xDomain = [-12, 24];
  const yDomain = countyHousingDomain();
  const x = makeScale(xDomain, [dims.left, dims.width - dims.right]);
  const y = makeScale(yDomain, [dims.height - dims.bottom, dims.top]);
  drawAxes(svg, dims, x, y, xDomain, yDomain);
  const incidentX = x(0);
  add("line", svg, { x1: incidentX, x2: incidentX, y1: dims.top, y2: dims.height - dims.bottom, class: "incident-line" });
  add("text", svg, { x: incidentX + 7, y: dims.top + 14, class: "incident-label" }).textContent = "Incident";
  if (!series.length) {
    add("text", svg, { x: dims.left, y: dims.top + 28, class: "axis-label" }).textContent = "No counties match this drilldown.";
    return;
  }
  const drawRows = cappedSeriesRows(series, yDomain);
  add("path", svg, { d: bandPath(drawRows, x, y), class: "risk-band active", fill: color });
  const linePath = pathFor(drawRows, x, y);
  add("path", svg, { d: linePath, class: "risk-line focused", stroke: color });
  const hitLine = add("path", svg, { d: linePath, class: "risk-hit-line" });
  const count = uniqueCountyCount(rows);
  hitLine.addEventListener("mouseenter", event => showLineSeriesTooltip(event, label, series, formatChange, count));
  hitLine.addEventListener("mousemove", event => showLineSeriesTooltip(event, label, series, formatChange, count));
  hitLine.addEventListener("mouseleave", hideTooltip);
}

function drawMigrationHousingControls() {
  const toggles = [
    ["risk", housingRiskFilterEnabled, "migrationHousingRiskLegend"],
    ["economic", housingEconomicFilterEnabled, "migrationHousingEconomicLegend"],
    ["migration", housingMigrationFilterEnabled, "migrationHousingClusterLegend"],
  ];
  document.querySelectorAll("[data-housing-filter-toggle]").forEach(button => {
    const filter = button.getAttribute("data-housing-filter-toggle");
    const active = filter === "risk" ? housingRiskFilterEnabled : filter === "economic" ? housingEconomicFilterEnabled : housingMigrationFilterEnabled;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", active ? "true" : "false");
  });
  toggles.forEach(([, active, legendId]) => {
    const legend = document.getElementById(legendId);
    if (legend) {
      legend.hidden = !active;
      if (!active) legend.innerHTML = "";
    }
  });
  if (housingRiskFilterEnabled) {
    drawRiskButtonLegend("migrationHousingRiskLegend", activeHousingRiskRating, rating => {
      activeHousingRiskRating = rating;
      drawMigrationHousingResponseModule();
    }, rating => filterHousingSeries({
      riskEnabled: true,
      economicEnabled: housingEconomicFilterEnabled,
      migrationEnabled: housingMigrationFilterEnabled,
      insuranceEnabled: false,
      riskRating: rating,
      economicProfile: activeHousingEconomicProfile,
      migrationProfile: activeHousingMigrationProfile,
    }).length > 0);
  }
  if (housingEconomicFilterEnabled) {
    drawProfileButtonLegend("migrationHousingEconomicLegend", migrationEconomicProfiles(), activeHousingEconomicProfile, profile => {
      activeHousingEconomicProfile = Number(profile);
      drawMigrationHousingResponseModule();
    }, profile => economicLineColor(profile.profile), profile => filterHousingSeries({
      riskEnabled: housingRiskFilterEnabled,
      economicEnabled: true,
      migrationEnabled: housingMigrationFilterEnabled,
      insuranceEnabled: false,
      riskRating: activeHousingRiskRating,
      economicProfile: profile.profile,
      migrationProfile: activeHousingMigrationProfile,
    }).length > 0);
  }
  if (housingMigrationFilterEnabled) {
    drawProfileButtonLegend("migrationHousingClusterLegend", migrationClusterProfiles(), activeHousingMigrationProfile, profile => {
      activeHousingMigrationProfile = Number(profile);
      drawMigrationHousingResponseModule();
    }, profile => profileColors[Number(profile.profile) % profileColors.length] || "#5e6872", profile => filterHousingSeries({
      riskEnabled: housingRiskFilterEnabled,
      economicEnabled: housingEconomicFilterEnabled,
      migrationEnabled: true,
      insuranceEnabled: false,
      riskRating: activeHousingRiskRating,
      economicProfile: activeHousingEconomicProfile,
      migrationProfile: profile.profile,
    }).length > 0);
  }
}

function housingLineColor(config) {
  if (config.insuranceEnabled) return profileColors[(Number(config.insuranceProfile) + 2) % profileColors.length] || "#5e6872";
  if (config.migrationEnabled) return profileColors[Number(config.migrationProfile) % profileColors.length] || "#5e6872";
  if (config.economicEnabled) return economicLineColor(config.economicProfile);
  if (config.riskEnabled) return colors[config.riskRating] || "#5e6872";
  return "#7c3aed";
}

function updateMigrationHousingTakeaway(rows, label) {
  const container = document.getElementById("migrationHousingTakeaway");
  if (!container) return;
  const series = summarizeSeriesRows(rows);
  const selectedCountyCount = uniqueCountyCount(rows);
  const clipped = seriesExceedsDomain(series, countyHousingDomain());
  const ranked = [];
  const source = data.migrationTrend?.housingSeries || [];
  const groups = [
    ...data.meta.riskRatings.map(rating => ({ label: `${rating} risk`, rows: source.filter(row => row.riskRating === rating) })),
    ...migrationEconomicProfiles().map(profile => ({ label: profile.label, rows: source.filter(row => Number(row.economicProfile) === Number(profile.profile)) })),
    ...migrationClusterProfiles().map(profile => ({ label: profile.label, rows: source.filter(row => Number(row.migrationProfile) === Number(profile.profile)) })),
  ];
  groups.forEach(group => {
    const pre = group.rows.filter(row => row.offset >= -12 && row.offset <= -1).map(row => Number(row.value)).filter(Number.isFinite);
    const post = group.rows.filter(row => row.offset >= 13 && row.offset <= 24).map(row => Number(row.value)).filter(Number.isFinite);
    if (!pre.length || !post.length) return;
    const change = quantileSorted(post.sort((a, b) => a - b), 0.5) - quantileSorted(pre.sort((a, b) => a - b), 0.5);
    ranked.push({ label: group.label, change });
  });
  ranked.sort((a, b) => Math.abs(b.change) - Math.abs(a.change));
  const top = ranked.slice(0, 3).map(item => `${escapeHtml(item.label)} (${item.change > 0 ? "strengthening or recovery" : "softening"})`).join(", ");
  const sampleNote = !rows.length
    ? "No matching counties."
    : selectedCountyCount <= 3
      ? `This drilldown is based on ${selectedCountyCount.toLocaleString()} ${selectedCountyCount === 1 ? "county" : "counties"}, so the median and IQR can collapse to a single path.`
      : "The selected counties have enough observations to compare the before and after path.";
  const capNote = clipped
    ? " Some values are outside the focused y-axis range and are pinned to the plot edge so the rest of the chart remains readable."
    : "";
  container.innerHTML = `<p><strong>${escapeHtml(label)}:</strong> ${sampleNote}${capNote} The most visible housing-market movement is concentrated in ${top || "groups with enough observations to compare"}.</p>`;
}

function drawMigrationHousingResponseModule() {
  drawMigrationHousingControls();
  const config = {
    riskEnabled: housingRiskFilterEnabled,
    economicEnabled: housingEconomicFilterEnabled,
    migrationEnabled: housingMigrationFilterEnabled,
    insuranceEnabled: false,
    riskRating: activeHousingRiskRating,
    economicProfile: activeHousingEconomicProfile,
    migrationProfile: activeHousingMigrationProfile,
  };
  const rows = filterHousingSeries(config);
  const label = housingSelectionLabel(config);
  drawHousingTrendChart("migrationHousingTrendChart", "migrationHousingTitle", rows, label, housingLineColor(config));
  updateMigrationHousingTakeaway(rows, label);
}

function migrationTrendClusterRiskMix(profileId) {
  const byRisk = data.migrationTrend?.trendProfiles?.byRiskRating || {};
  return data.meta.riskRatings
    .map(rating => ({ rating, row: (byRisk[rating] || []).find(item => Number(item.profile) === Number(profileId)) }))
    .filter(item => item.row)
    .map(item => ({ rating: item.rating, counties: item.row.counties || 0, share: item.row.share || 0 }));
}

function activeMigrationCluster() {
  const profiles = data.migrationTrend?.trendProfiles?.profiles || [];
  if (!profiles.length) return null;
  const current = profiles.find(profile => Number(profile.profile) === Number(activeMigrationClusterProfile));
  if (current) return current;
  activeMigrationClusterProfile = profiles[0].profile;
  return profiles[0];
}

function showMigrationClusterTooltip(event, profile) {
  const tooltip = document.getElementById("riskTooltip");
  tooltip.innerHTML = `<strong>${escapeHtml(profile.label)}</strong>` +
    `<div>Counties: ${profile.countyCount ?? "n/a"}</div>` +
    `<div>Before: ${formatMigration(profile.preAvg)}</div>` +
    `<div>After: ${formatMigration(profile.postAvg)}</div>` +
    `<div>Overall movement: ${formatMigration(profile.overallChange)}</div>`;
  tooltip.style.display = "block";
  moveTooltip(event);
}

function drawMigrationClusterLegend() {
  const legend = document.getElementById("migrationClusterLegend");
  if (!legend) return;
  const profiles = data.migrationTrend?.trendProfiles?.profiles || [];
  legend.innerHTML = "";
  profiles.forEach(profile => {
    const color = profileColors[Number(profile.profile) % profileColors.length] || "#5e6872";
    const item = document.createElement("button");
    item.type = "button";
    item.className = `risk-toggle-button${Number(profile.profile) === Number(activeMigrationClusterProfile) ? " active" : ""}`;
    item.setAttribute("aria-pressed", Number(profile.profile) === Number(activeMigrationClusterProfile) ? "true" : "false");
    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.background = color;
    item.appendChild(swatch);
    item.appendChild(document.createTextNode(profile.label));
    item.addEventListener("click", () => {
      activeMigrationClusterProfile = profile.profile;
      drawMigrationTrendClusters();
    });
    legend.appendChild(item);
  });
}

function drawMigrationClusterChart() {
  const svg = document.getElementById("migrationClusterChart");
  if (!svg) return;
  clear(svg);
  const profile = activeMigrationCluster();
  const title = document.getElementById("migrationClusterChartTitle");
  if (title) title.textContent = profile ? `Net migration per capita: ${profile.label}` : "Net migration per capita by migration trend cluster";
  const rows = profile ? (data.migrationTrend?.trendProfiles?.byProfile?.[String(profile.profile)] || []) : [];
  const dims = chartDims(svg);
  svg.setAttribute("viewBox", `0 0 ${dims.width} ${dims.height}`);
  const offsets = data.migrationTrend?.meta?.offsets || [-2, -1, 0, 1, 2];
  const xDomain = [Math.min(...offsets), Math.max(...offsets)];
  const yDomain = migrationClusterDomain();
  const x = makeScale(xDomain, [dims.left, dims.width - dims.right]);
  const y = makeScale(yDomain, [dims.height - dims.bottom, dims.top]);
  drawMigrationAxes(svg, dims, x, y, xDomain, yDomain);
  const incidentX = x(0);
  add("line", svg, { x1: incidentX, x2: incidentX, y1: dims.top, y2: dims.height - dims.bottom, class: "incident-line" });
  add("text", svg, { x: incidentX + 7, y: dims.top + 14, class: "incident-label" }).textContent = "Incident";
  if (!profile || !rows.length) {
    add("text", svg, { x: dims.left, y: dims.top + 28, class: "axis-label" }).textContent = "No measured counties for this cluster.";
    return;
  }
  const color = profileColors[Number(profile.profile) % profileColors.length] || "#5e6872";
  const drawRows = cappedSeriesRows(rows, yDomain);
  const linePath = pathFor(drawRows, x, y);
  const band = add("path", svg, { d: bandPath(drawRows, x, y), class: "risk-band active", fill: color });
  add("path", svg, { d: linePath, class: "risk-line focused", stroke: color });
  const hitLine = add("path", svg, { d: linePath, class: "risk-hit-line" });
  hitLine.addEventListener("mouseenter", event => {
    band.classList.add("active");
    showMigrationClusterTooltip(event, profile);
  });
  hitLine.addEventListener("mousemove", event => showMigrationClusterTooltip(event, profile));
  hitLine.addEventListener("mouseleave", hideTooltip);
}

function drawRelationshipAxes(svg, dims, x, y, xDomain, yDomain, xLabel = "Change in net migration per 1,000 residents", yLabel = "Change in housing market YOY index") {
  const g = add("g", svg);
  const ticks = 5;
  for (let i = 0; i <= ticks; i++) {
    const yValue = yDomain[0] + ((yDomain[1] - yDomain[0]) * i / ticks);
    const yy = y(yValue);
    add("line", g, { x1: dims.left, x2: dims.width - dims.right, y1: yy, y2: yy, stroke: "#e5e0d7" });
    add("text", g, { x: dims.left - 8, y: yy + 4, "text-anchor": "end" }).textContent = yValue.toFixed(2);
    const xValue = xDomain[0] + ((xDomain[1] - xDomain[0]) * i / ticks);
    const xx = x(xValue);
    add("line", g, { x1: xx, x2: xx, y1: dims.top, y2: dims.height - dims.bottom, stroke: "#eee9df" });
    add("text", g, { x: xx, y: dims.height - dims.bottom + 22, "text-anchor": "middle" }).textContent = xValue.toFixed(1);
  }
  add("line", g, { x1: dims.left, x2: dims.width - dims.right, y1: y(0), y2: y(0), class: "zero-line" });
  add("line", g, { x1: x(0), x2: x(0), y1: dims.top, y2: dims.height - dims.bottom, class: "zero-line" });
  add("text", g, { x: dims.left, y: dims.height - 6 }).textContent = xLabel;
  add("text", g, { x: 4, y: Math.max(12, dims.top - 8) }).textContent = yLabel;
}

function drawMigrationHousingRiskLegend() {
  const legend = document.getElementById("migrationRelationshipRiskLegend");
  if (!legend) return;
  legend.innerHTML = "";
  data.meta.riskRatings.forEach(rating => {
    const item = document.createElement("button");
    item.type = "button";
    item.className = `risk-toggle-button${rating === activeMigrationHousingRiskRating ? " active" : ""}`;
    item.setAttribute("aria-pressed", rating === activeMigrationHousingRiskRating ? "true" : "false");
    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.background = colors[rating];
    item.appendChild(swatch);
    item.appendChild(document.createTextNode(rating));
    item.addEventListener("click", () => {
      activeMigrationHousingRiskRating = rating;
      drawMigrationHousingScatter();
    });
    legend.appendChild(item);
  });
}

function drawMigrationHousingScatter() {
  const svg = document.getElementById("migrationHousingScatter");
  if (!svg) return;
  drawMigrationHousingRiskLegend();
  clear(svg);
  const allRows = (data.migrationTrend?.housingRelationship || [])
    .filter(row => row.migrationChange != null && row.housingChange != null);
  const rows = allRows.filter(row => row.riskRating === activeMigrationHousingRiskRating);
  const dims = chartDims(svg);
  svg.setAttribute("viewBox", `0 0 ${dims.width} ${dims.height}`);
  if (!allRows.length) {
    add("text", svg, { x: dims.left, y: dims.top + 28, class: "axis-label" }).textContent = "No county-level migration and housing change pairs could be measured.";
    return;
  }
  const xDomain = fullDomainFromValues(allRows.map(row => row.migrationChange), { symmetric: true, includeZero: true, padRatio: 0.08 });
  const yDomain = fullDomainFromValues(allRows.map(row => row.housingChange), { symmetric: true, includeZero: true, padRatio: 0.08 });
  const x = makeScale(xDomain, [dims.left, dims.width - dims.right]);
  const y = makeScale(yDomain, [dims.height - dims.bottom, dims.top]);
  drawRelationshipAxes(svg, dims, x, y, xDomain, yDomain, "Change in net migration per 1,000 residents", "Change in housing market YOY index");
  add("text", svg, { x: dims.left, y: dims.top + 14, class: "axis-label" }).textContent = `${activeMigrationHousingRiskRating} risk counties`;
  if (!rows.length) {
    add("text", svg, { x: dims.left, y: dims.top + 34, class: "axis-label" }).textContent = "No counties in this risk group have both migration and housing changes.";
    return;
  }
  rows.forEach(row => {
    const circle = add("circle", svg, {
      cx: x(row.migrationChange),
      cy: y(row.housingChange),
      r: 3.2,
      fill: colors[row.riskRating] || "#5e6872",
      opacity: 0.55,
    });
    circle.addEventListener("mouseenter", event => {
      const tooltip = document.getElementById("riskTooltip");
      tooltip.innerHTML = `<strong>${escapeHtml(row.riskRating || "Unknown")} risk county</strong>` +
        `<div>Migration change: ${formatMigration(row.migrationChange)}</div>` +
        `<div>Housing index change: ${formatChange(row.housingChange)}</div>` +
        `<div>${escapeHtml(row.profileLabel || "No economic profile")}</div>`;
      tooltip.style.display = "block";
      moveTooltip(event);
    });
    circle.addEventListener("mousemove", moveTooltip);
    circle.addEventListener("mouseleave", hideTooltip);
  });
}

function drawMigrationTrendClusters() {
  const summary = document.getElementById("migrationClusterSummary");
  const container = document.getElementById("migrationClusterCards");
  const payload = data.migrationTrend?.trendProfiles || {};
  const profiles = payload.profiles || [];
  if (!container || !summary) return;
  if (!profiles.length) {
    summary.innerHTML = `<p>No migration trend clusters could be generated.</p>`;
    container.innerHTML = "";
    return;
  }
  drawMigrationClusterLegend();
  drawMigrationClusterChart();
  const score = (payload.modelScores || [])[0] || {};
  summary.innerHTML = `<p>Counties are grouped with feature-based time-series clustering on net migration level, before-to-after change, first-year movement, second-year reversal, and volatility. The selected model uses ${payload.bestK || profiles.length} clusters with mean assignment confidence ${score.meanAssignmentConfidence ?? "n/a"}.</p>`;
  container.innerHTML = profiles.map(profile => {
    const mix = migrationTrendClusterRiskMix(profile.profile).sort((a, b) => b.counties - a.counties);
    const topRisk = mix[0];
    const color = profileColors[Number(profile.profile) % profileColors.length] || "#5e6872";
    return `<article class="migration-cluster-card">` +
      `<h4><span class="swatch" style="background:${color}"></span>${escapeHtml(profile.label)}</h4>` +
      `<div class="meta">${profile.countyCount ?? 0} counties; confidence ${profile.meanAssignmentConfidence ?? "n/a"}</div>` +
      `<p>${escapeHtml(profile.interpretation || "This cluster has a distinct migration path across the incident window.")}</p>` +
      `<div class="stats">` +
        `<span><strong>Before:</strong> ${formatMigration(profile.preAvg)}</span>` +
        `<span><strong>After:</strong> ${formatMigration(profile.postAvg)}</span>` +
        `<span><strong>Overall:</strong> ${formatMigration(profile.overallChange)}</span>` +
        `<span><strong>Largest NRI mix:</strong> ${topRisk ? `${escapeHtml(topRisk.rating)} (${fmtPct(topRisk.share)})` : "n/a"}</span>` +
      `</div>` +
    `</article>`;
  }).join("");
}

function showMigrationTooltip(event, groupOrRating) {
  const rating = typeof groupOrRating === "string" ? groupOrRating : groupOrRating.rating;
  const tooltip = document.getElementById("riskTooltip");
  const summary = data.migrationTrend?.profileRiskSummaries?.[String(activeMigrationEconomicProfile)]?.[rating] || data.migrationTrend?.summaries?.[rating] || {};
  const profileLabel = activeMigrationEconomicGroup()?.label;
  tooltip.innerHTML = `<strong>${escapeHtml(rating)} risk${profileLabel ? ` in ${escapeHtml(profileLabel)}` : ""}</strong>` +
    `<div>Counties: ${summary.countyCount ?? "n/a"}</div>` +
    `<div>2 years before: ${formatMigration(summary.minus2)}</div>` +
    `<div>1 year before: ${formatMigration(summary.minus1)}</div>` +
    `<div>Incident year: ${formatMigration(summary.incidentYear)}</div>` +
    `<div>2 years after: ${formatMigration(summary.plus2)}</div>`;
  tooltip.style.display = "block";
  moveTooltip(event);
}

function showMigrationEconomicTooltip(event, group) {
  const tooltip = document.getElementById("riskTooltip");
  const summary = data.migrationTrend?.profileSummaries?.[String(group.profile)] || {};
  tooltip.innerHTML = `<strong>${escapeHtml(group.label)}</strong>` +
    `<div>Counties: ${summary.countyCount ?? "n/a"}</div>` +
    `<div>2 years before: ${formatMigration(summary.minus2)}</div>` +
    `<div>1 year before: ${formatMigration(summary.minus1)}</div>` +
    `<div>Incident year: ${formatMigration(summary.incidentYear)}</div>` +
    `<div>2 years after: ${formatMigration(summary.plus2)}</div>`;
  tooltip.style.display = "block";
  moveTooltip(event);
}

function insuranceMetrics() {
  return data.insuranceTrends?.metrics || [];
}

function insuranceMetricMeta(metricKey) {
  return insuranceMetrics().find(metric => metric.key === metricKey) || insuranceMetrics()[0] || null;
}

function activeInsuranceMetricMeta() {
  const current = insuranceMetricMeta(activeInsuranceMetric);
  if (current) return current;
  return null;
}

function activeInsuranceRiskMetricMeta() {
  const current = insuranceMetricMeta(activeInsuranceRiskMetric);
  if (current) return current;
  return null;
}

function activeInsuranceEconomicGroup() {
  const groups = migrationEconomicGroups();
  if (!groups.length) return null;
  const current = groups.find(group => Number(group.profile) === Number(activeInsuranceEconomicProfile));
  if (current) return current;
  activeInsuranceEconomicProfile = groups[0].profile;
  return groups[0];
}

function insuranceMetricRows(profileId = activeInsuranceEconomicProfile, metricKey = activeInsuranceMetric) {
  return data.insuranceTrends?.byEconomicProfileMetric?.[String(profileId)]?.[metricKey] || [];
}

function insuranceRiskMetricRows(rating = activeInsuranceRiskRating, metricKey = activeInsuranceRiskMetric) {
  return data.insuranceTrends?.byRiskRatingMetric?.[rating]?.[metricKey] || [];
}

function drawInsuranceEconomicLegend() {
  const legend = document.getElementById("insuranceEconomicLegend");
  if (!legend) return;
  legend.innerHTML = "";
  migrationEconomicGroups().forEach(group => {
    const item = document.createElement("button");
    item.type = "button";
    item.className = `risk-toggle-button${Number(group.profile) === Number(activeInsuranceEconomicProfile) ? " active" : ""}`;
    item.setAttribute("data-insurance-profile", group.profile);
    item.setAttribute("aria-pressed", Number(group.profile) === Number(activeInsuranceEconomicProfile) ? "true" : "false");
    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.background = economicLineColor(group.profile);
    item.appendChild(swatch);
    item.appendChild(document.createTextNode(group.label));
    item.addEventListener("click", () => {
      activeInsuranceEconomicProfile = Number(group.profile);
      insuranceTrendPaused = true;
      stopInsuranceTrendTimer();
      drawInsuranceTrendModule();
      updateInsuranceTrendActions();
    });
    legend.appendChild(item);
  });
}

function drawInsuranceMetricLegend() {
  const legend = document.getElementById("insuranceMetricLegend");
  if (!legend) return;
  legend.innerHTML = "";
  insuranceMetrics().forEach(metric => {
    const item = document.createElement("button");
    item.type = "button";
    item.className = `risk-toggle-button${metric.key === activeInsuranceMetric ? " active" : ""}`;
    item.setAttribute("data-insurance-metric", metric.key);
    item.setAttribute("aria-pressed", metric.key === activeInsuranceMetric ? "true" : "false");
    item.appendChild(document.createTextNode(metric.label));
    item.addEventListener("click", () => {
      activeInsuranceMetric = metric.key;
      insuranceTrendPaused = true;
      stopInsuranceTrendTimer();
      drawInsuranceTrendModule();
      updateInsuranceTrendActions();
    });
    legend.appendChild(item);
  });
}

function filterInsuranceCountySeries(metricKey = activeInsuranceMetric) {
  return (data.insuranceTrends?.countySeries || []).filter(row => {
    if (row.metric !== metricKey || row.value == null) return false;
    if (insuranceTrendRiskFilterEnabled && row.riskRating !== activeInsuranceTrendRiskRating) return false;
    if (insuranceTrendEconomicFilterEnabled && Number(row.economicProfile) !== Number(activeInsuranceTrendEconomicProfile)) return false;
    if (insuranceTrendMigrationFilterEnabled && Number(row.migrationProfile) !== Number(activeInsuranceTrendMigrationProfile)) return false;
    return true;
  });
}

function insuranceTrendSelectionLabel() {
  const parts = [];
  if (insuranceTrendRiskFilterEnabled) parts.push(`${activeInsuranceTrendRiskRating} risk`);
  if (insuranceTrendEconomicFilterEnabled) parts.push(migrationProfileLabel(activeInsuranceTrendEconomicProfile));
  if (insuranceTrendMigrationFilterEnabled) {
    const profile = migrationClusterProfiles().find(item => Number(item.profile) === Number(activeInsuranceTrendMigrationProfile));
    parts.push(profile?.label || `Migration profile ${activeInsuranceTrendMigrationProfile}`);
  }
  return parts.length ? parts.join(", ") : "All counties";
}

function drawInsuranceTrendFilterControls() {
  document.querySelectorAll("[data-insurance-trend-filter-toggle]").forEach(button => {
    const filter = button.getAttribute("data-insurance-trend-filter-toggle");
    const active = filter === "risk" ? insuranceTrendRiskFilterEnabled
      : filter === "economic" ? insuranceTrendEconomicFilterEnabled
      : insuranceTrendMigrationFilterEnabled;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", active ? "true" : "false");
  });
  [
    ["insuranceTrendRiskLegend", insuranceTrendRiskFilterEnabled],
    ["insuranceTrendEconomicLegend", insuranceTrendEconomicFilterEnabled],
    ["insuranceTrendMigrationLegend", insuranceTrendMigrationFilterEnabled],
  ].forEach(([id, active]) => {
    const legend = document.getElementById(id);
    if (legend) {
      legend.hidden = !active;
      if (!active) legend.innerHTML = "";
    }
  });
  if (insuranceTrendRiskFilterEnabled) {
    drawRiskButtonLegend("insuranceTrendRiskLegend", activeInsuranceTrendRiskRating, rating => {
      activeInsuranceTrendRiskRating = rating;
      drawInsuranceTrendModule();
    }, rating => (data.insuranceTrends?.countySeries || []).some(row =>
      row.metric === activeInsuranceMetric &&
      row.riskRating === rating &&
      (!insuranceTrendEconomicFilterEnabled || Number(row.economicProfile) === Number(activeInsuranceTrendEconomicProfile)) &&
      (!insuranceTrendMigrationFilterEnabled || Number(row.migrationProfile) === Number(activeInsuranceTrendMigrationProfile))
    ));
  }
  if (insuranceTrendEconomicFilterEnabled) {
    drawProfileButtonLegend("insuranceTrendEconomicLegend", migrationEconomicProfiles(), activeInsuranceTrendEconomicProfile, profile => {
      activeInsuranceTrendEconomicProfile = Number(profile);
      drawInsuranceTrendModule();
    }, profile => economicLineColor(profile.profile), profile => (data.insuranceTrends?.countySeries || []).some(row =>
      row.metric === activeInsuranceMetric &&
      Number(row.economicProfile) === Number(profile.profile) &&
      (!insuranceTrendRiskFilterEnabled || row.riskRating === activeInsuranceTrendRiskRating) &&
      (!insuranceTrendMigrationFilterEnabled || Number(row.migrationProfile) === Number(activeInsuranceTrendMigrationProfile))
    ));
  }
  if (insuranceTrendMigrationFilterEnabled) {
    drawProfileButtonLegend("insuranceTrendMigrationLegend", migrationClusterProfiles(), activeInsuranceTrendMigrationProfile, profile => {
      activeInsuranceTrendMigrationProfile = Number(profile);
      drawInsuranceTrendModule();
    }, profile => profileColors[Number(profile.profile) % profileColors.length] || "#5e6872", profile => (data.insuranceTrends?.countySeries || []).some(row =>
      row.metric === activeInsuranceMetric &&
      Number(row.migrationProfile) === Number(profile.profile) &&
      (!insuranceTrendRiskFilterEnabled || row.riskRating === activeInsuranceTrendRiskRating) &&
      (!insuranceTrendEconomicFilterEnabled || Number(row.economicProfile) === Number(activeInsuranceTrendEconomicProfile))
    ));
  }
}

function insuranceTrendLineColor() {
  if (insuranceTrendMigrationFilterEnabled) return profileColors[Number(activeInsuranceTrendMigrationProfile) % profileColors.length] || "#5e6872";
  if (insuranceTrendEconomicFilterEnabled) return economicLineColor(activeInsuranceTrendEconomicProfile);
  if (insuranceTrendRiskFilterEnabled) return colors[activeInsuranceTrendRiskRating] || "#5e6872";
  return insuranceMetricColors[activeInsuranceMetric] || "#2f3941";
}

function drawInsuranceRiskLegend() {
  const legend = document.getElementById("insuranceRiskLegend");
  if (!legend) return;
  legend.innerHTML = "";
  data.meta.riskRatings.forEach(rating => {
    const hasRows = insuranceRiskMetricRows(rating, activeInsuranceRiskMetric).length > 0;
    if (!hasRows) return;
    const item = document.createElement("button");
    item.type = "button";
    item.className = `risk-toggle-button${rating === activeInsuranceRiskRating ? " active" : ""}`;
    item.setAttribute("aria-pressed", rating === activeInsuranceRiskRating ? "true" : "false");
    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.background = colors[rating] || "#5e6872";
    item.appendChild(swatch);
    item.appendChild(document.createTextNode(rating));
    item.addEventListener("click", () => {
      activeInsuranceRiskRating = rating;
      insuranceRiskTrendPaused = true;
      stopInsuranceRiskTrendTimer();
      drawInsuranceRiskTrendModule();
      updateInsuranceRiskTrendActions();
    });
    legend.appendChild(item);
  });
}

function drawInsuranceRiskMetricLegend() {
  const legend = document.getElementById("insuranceRiskMetricLegend");
  if (!legend) return;
  legend.innerHTML = "";
  insuranceMetrics().forEach(metric => {
    const item = document.createElement("button");
    item.type = "button";
    item.className = `risk-toggle-button${metric.key === activeInsuranceRiskMetric ? " active" : ""}`;
    item.setAttribute("aria-pressed", metric.key === activeInsuranceRiskMetric ? "true" : "false");
    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.background = insuranceMetricColors[metric.key] || "#2f3941";
    item.appendChild(swatch);
    item.appendChild(document.createTextNode(metric.label));
    item.addEventListener("click", () => {
      activeInsuranceRiskMetric = metric.key;
      const available = data.meta.riskRatings.find(rating => insuranceRiskMetricRows(rating, activeInsuranceRiskMetric).length > 0);
      if (available && !insuranceRiskMetricRows(activeInsuranceRiskRating, activeInsuranceRiskMetric).length) activeInsuranceRiskRating = available;
      insuranceRiskTrendPaused = true;
      stopInsuranceRiskTrendTimer();
      drawInsuranceRiskTrendModule();
      updateInsuranceRiskTrendActions();
    });
    legend.appendChild(item);
  });
}

function drawInsuranceEconomicMigrationChart() {
  const activeGroup = activeInsuranceEconomicGroup();
  const title = document.getElementById("insuranceEconomicTitle");
  if (title) title.textContent = activeGroup ? `Migration by economic profile: ${activeGroup.label}` : "Migration by economic profile";
  drawMigrationSeriesChart(
    "insuranceEconomicMigrationChart",
    activeGroup ? [activeGroup] : [],
    activeGroup?.profile,
    "profile",
    group => economicLineColor(group.profile),
    showMigrationEconomicTooltip
  );
}

function polarPoint(cx, cy, radius, angle) {
  return [cx + radius * Math.cos(angle), cy + radius * Math.sin(angle)];
}

function arcPath(cx, cy, radius, startAngle, endAngle) {
  const [sx, sy] = polarPoint(cx, cy, radius, startAngle);
  const [ex, ey] = polarPoint(cx, cy, radius, endAngle);
  const largeArc = endAngle - startAngle > Math.PI ? 1 : 0;
  return `M${cx},${cy} L${sx.toFixed(2)},${sy.toFixed(2)} A${radius},${radius} 0 ${largeArc} 1 ${ex.toFixed(2)},${ey.toFixed(2)} Z`;
}

function drawInsuranceRiskPieChart() {
  const svg = document.getElementById("insuranceRiskPieChart");
  if (!svg) return;
  clear(svg);
  svg.setAttribute("viewBox", "0 0 132 180");
  const profile = activeInsuranceEconomicGroup();
  if (!profile) return;
  const summaries = data.migrationTrend?.profileRiskSummaries?.[String(profile.profile)] || {};
  const rows = data.meta.riskRatings
    .map(rating => ({ rating, count: Number(summaries[rating]?.countyCount) || 0 }))
    .filter(row => row.count > 0);
  const total = rows.reduce((sum, row) => sum + row.count, 0);
  add("text", svg, { x: 66, y: 14, "text-anchor": "middle", class: "pie-label" }).textContent = "NRI mix";
  if (!total) {
    add("text", svg, { x: 66, y: 86, "text-anchor": "middle", class: "pie-note" }).textContent = "No data";
    return;
  }
  let angle = -Math.PI / 2;
  rows.forEach(row => {
    const slice = (row.count / total) * Math.PI * 2;
    const path = add("path", svg, { d: arcPath(66, 74, 48, angle, angle + slice), fill: colors[row.rating] || "#5e6872", stroke: "#fffaf0", "stroke-width": 1 });
    const title = add("title", path);
    title.textContent = `${row.rating} risk: ${fmtPct(row.count / total)} (${row.count.toLocaleString()} of ${total.toLocaleString()} counties)`;
    path.addEventListener("mouseenter", event => showInsurancePieTooltip(event, row, total, profile));
    path.addEventListener("mousemove", event => showInsurancePieTooltip(event, row, total, profile));
    path.addEventListener("mouseleave", hideTooltip);
    angle += slice;
  });
}

function showInsurancePieTooltip(event, row, total, profile) {
  const tooltip = document.getElementById("riskTooltip");
  tooltip.innerHTML = `<strong>${escapeHtml(row.rating)} risk</strong>` +
    `<div>${escapeHtml(profile.label)}</div>` +
    `<div>Share: ${fmtPct(row.count / total)}</div>` +
    `<div>Counties: ${row.count.toLocaleString()} of ${total.toLocaleString()}</div>`;
  tooltip.style.display = "block";
  moveTooltip(event);
}

function drawInsuranceMetricAxes(svg, dims, x, y, xDomain, yDomain, metric) {
  const g = add("g", svg);
  const yTicks = 5;
  for (let i = 0; i <= yTicks; i++) {
    const value = yDomain[0] + ((yDomain[1] - yDomain[0]) * i / yTicks);
    const yy = y(value);
    add("line", g, { x1: dims.left, x2: dims.width - dims.right, y1: yy, y2: yy, stroke: "#e5e0d7" });
    add("text", g, { x: dims.left - 8, y: yy + 4, "text-anchor": "end" }).textContent = metricFormat(metric, value);
  }
  for (let value = Math.ceil(xDomain[0]); value <= Math.floor(xDomain[1]); value += 1) {
    const xx = x(value);
    add("line", g, { x1: xx, x2: xx, y1: dims.top, y2: dims.height - dims.bottom, stroke: "#eee9df" });
    add("text", g, { x: xx, y: dims.height - dims.bottom + 22, "text-anchor": "middle" }).textContent = value;
  }
  if (yDomain[0] <= 0 && yDomain[1] >= 0) {
    add("line", g, { x1: dims.left, x2: dims.width - dims.right, y1: y(0), y2: y(0), class: "zero-line" });
  }
  add("text", g, { x: dims.left, y: dims.height - 8 }).textContent = "Year offset";
  add("text", g, { x: 4, y: dims.top - 8 }).textContent = metric?.label || "Insurance trend";
}

function drawInsuranceTrendChart() {
  const svg = document.getElementById("insuranceTrendChart");
  if (!svg) return;
  clear(svg);
  const metric = activeInsuranceMetricMeta();
  const rawRows = metric ? filterInsuranceCountySeries(metric.key) : [];
  const rows = summarizeSeriesRows(rawRows);
  const title = document.getElementById("insuranceMetricTitle");
  const label = insuranceTrendSelectionLabel();
  if (title) title.textContent = metric ? `${metric.label}: ${label}` : "Insurance trend around incidents";
  const dims = chartDims(svg);
  svg.setAttribute("viewBox", `0 0 ${dims.width} ${dims.height}`);
  const offsets = data.migrationTrend?.meta?.offsets || [-2, -1, 0, 1, 2];
  const xDomain = [Math.min(...offsets), Math.max(...offsets)];
  const yDomain = insuranceMetricDomain(metric?.key);
  const x = makeScale(xDomain, [dims.left, dims.width - dims.right]);
  const y = makeScale(yDomain, [dims.height - dims.bottom, dims.top]);
  drawInsuranceMetricAxes(svg, dims, x, y, xDomain, yDomain, metric);
  const incidentX = x(0);
  add("line", svg, { x1: incidentX, x2: incidentX, y1: dims.top, y2: dims.height - dims.bottom, class: "incident-line" });
  add("text", svg, { x: incidentX + 7, y: dims.top + 14, class: "incident-label" }).textContent = "Incident";
  if (rows.length) {
    const color = insuranceTrendLineColor();
    const drawRows = cappedSeriesRows(rows, yDomain);
    add("path", svg, { d: bandPath(drawRows, x, y), class: "risk-band active", fill: color });
    const linePath = pathFor(drawRows, x, y);
    add("path", svg, { d: linePath, class: "risk-line focused", stroke: color });
    const hitLine = add("path", svg, { d: linePath, class: "risk-hit-line" });
    const count = uniqueCountyCount(rawRows);
    hitLine.addEventListener("mouseenter", event => showLineSeriesTooltip(event, `${metric.label}: ${label}`, rows, value => metricFormat(metric, value), count));
    hitLine.addEventListener("mousemove", event => showLineSeriesTooltip(event, `${metric.label}: ${label}`, rows, value => metricFormat(metric, value), count));
    hitLine.addEventListener("mouseleave", hideTooltip);
  }
}

function drawInsuranceRiskTrendChart() {
  const svg = document.getElementById("insuranceRiskTrendChart");
  if (!svg) return;
  clear(svg);
  const metric = activeInsuranceRiskMetricMeta();
  if (!metric) return;
  let rows = insuranceRiskMetricRows(activeInsuranceRiskRating, metric.key);
  if (!rows.length) {
    const available = data.meta.riskRatings.find(rating => insuranceRiskMetricRows(rating, metric.key).length > 0);
    if (available) {
      activeInsuranceRiskRating = available;
      rows = insuranceRiskMetricRows(activeInsuranceRiskRating, metric.key);
    }
  }
  const dims = chartDims(svg);
  svg.setAttribute("viewBox", `0 0 ${dims.width} ${dims.height}`);
  const offsets = data.migrationTrend?.meta?.offsets || [-2, -1, 0, 1, 2];
  const xDomain = [Math.min(...offsets), Math.max(...offsets)];
  const yDomain = insuranceMetricDomain(metric.key);
  const x = makeScale(xDomain, [dims.left, dims.width - dims.right]);
  const y = makeScale(yDomain, [dims.height - dims.bottom, dims.top]);
  drawInsuranceMetricAxes(svg, dims, x, y, xDomain, yDomain, metric);
  const incidentX = x(0);
  add("line", svg, { x1: incidentX, x2: incidentX, y1: dims.top, y2: dims.height - dims.bottom, class: "incident-line" });
  add("text", svg, { x: incidentX + 7, y: dims.top + 14, class: "incident-label" }).textContent = "Incident";
  if (rows.length) {
    const color = colors[activeInsuranceRiskRating] || "#5e6872";
    const drawRows = cappedSeriesRows(rows, yDomain);
    add("path", svg, { d: bandPath(drawRows, x, y), class: "risk-band active", fill: color });
    add("path", svg, { d: pathFor(drawRows, x, y), class: "risk-line focused", stroke: color });
  }
}

function updateInsuranceRiskTrendTakeaway() {
  const container = document.getElementById("insuranceRiskTrendTakeaway");
  if (!container) return;
  const metric = activeInsuranceRiskMetricMeta();
  const summary = data.insuranceTrends?.riskSummaries?.[activeInsuranceRiskRating]?.[metric?.key] || {};
  container.innerHTML = metric
    ? `<p><strong>${escapeHtml(activeInsuranceRiskRating)} risk, ${escapeHtml(metric.label)}:</strong> ${insuranceTrendPhrase(summary, metric)}</p>`
    : `<p>No insurance trend could be measured for the selected risk group.</p>`;
}

function drawInsuranceRiskTrendModule() {
  drawInsuranceRiskLegend();
  drawInsuranceRiskMetricLegend();
  drawInsuranceRiskTrendChart();
  updateInsuranceRiskTrendTakeaway();
}

function insuranceTrendPhrase(summary, metric) {
  const start = summary?.minus2;
  const incident = summary?.incidentYear;
  const end = summary?.plus2;
  if (start == null || incident == null || end == null) return "The available data are too sparse to describe a clear trend.";
  const overallDiff = end - start;
  const earlyDiff = incident - start;
  const lateDiff = end - incident;
  const material = metric?.key === "premiumLevel" ? 50 : 0.02;
  const direction = Math.abs(overallDiff) < material ? "stays broadly level" : overallDiff > 0 ? "moves higher" : "moves lower";
  const early = Math.abs(earlyDiff) < material ? "little change before the incident" : earlyDiff > 0 ? "a pre-incident rise" : "a pre-incident decline";
  const late = Math.abs(lateDiff) < material ? "little change after the incident" : lateDiff > 0 ? "upward movement after the incident" : "softening after the incident";
  if (metric?.key === "premiumLevel") {
    return `Premium levels ${direction} across the window, with ${early} and ${late}.`;
  }
  if (metric?.key === "premiumYoy") {
    return `Premium growth ${direction} across the window, with ${early} and ${late}.`;
  }
  if (metric?.key === "nonrenewalRate") {
    return `Non-renewal pressure ${direction} across the window, with ${early} and ${late}.`;
  }
  return `The selected insurance metric ${direction} across the window, with ${early} and ${late}.`;
}

function updateInsuranceTrendTakeaways() {
  const takeaway = document.getElementById("insuranceTrendTakeaway");
  const metric = activeInsuranceMetricMeta();
  const summary = data.insuranceTrends?.overallSummaries?.[metric?.key] || {};
  if (takeaway) {
    const ranked = Object.entries(data.insuranceTrends?.summaries || {}).map(([profile, byMetric]) => {
      const item = byMetric?.[metric?.key] || {};
      return { label: migrationProfileLabel(profile), change: item.plus2 != null && item.minus2 != null ? item.plus2 - item.minus2 : null };
    }).filter(item => item.change != null).sort((a, b) => Math.abs(b.change) - Math.abs(a.change)).slice(0, 3);
    const detail = ranked.map(item => `${escapeHtml(item.label)} (${item.change > 0 ? "upward pressure" : "softening pressure"})`).join(", ");
    takeaway.innerHTML = metric
      ? `<p><strong>${escapeHtml(metric.label)}:</strong> ${insuranceTrendPhrase(summary, metric)} The most visible profile-level shifts are in ${detail || "groups with enough measured data"}.</p>`
      : `<p>No insurance trend could be measured for the selected frame.</p>`;
  }
}

function drawInsuranceTrendModule() {
  drawInsuranceTrendFilterControls();
  drawInsuranceMetricLegend();
  drawInsuranceTrendChart();
  updateInsuranceTrendTakeaways();
  drawInsuranceClusterSection();
  drawInsuranceHousingResponseModule();
  drawInsuranceRelationshipScatter();
}

function drawInsuranceClusterSection() {
  const summary = document.getElementById("insuranceClusterSummary");
  const cards = document.getElementById("insuranceClusterCards");
  if (!summary || !cards) return;
  const payload = data.insuranceProfiles || {};
  const profiles = payload.profiles || [];
  const score = (payload.modelScores || [])[0] || {};
  summary.innerHTML = `<p>Counties are assigned to insurance profile clusters using premium level, premium growth, non-renewal pressure, volatility, and trend features. The selected clustering model uses ${payload.bestK || profiles.length} clusters with mean assignment confidence ${score.meanAssignmentConfidence ?? "n/a"}.</p>`;
  cards.innerHTML = profiles.map(profile => {
    const high = featureList(profile.topHighFeatures).slice(0, 2).map(escapeHtml).join(", ");
    const low = featureList(profile.topLowFeatures).slice(0, 2).map(escapeHtml).join(", ");
    const color = profileColors[Number(profile.profile) % profileColors.length] || "#5e6872";
    return `<article class="migration-cluster-card">` +
      `<h4><span class="swatch" style="background:${color}"></span>${escapeHtml(profile.label)}</h4>` +
      `<div class="meta">${profile.countyCount ?? 0} counties; confidence ${profile.meanAssignmentConfidence ?? "n/a"}</div>` +
      `<p>Compared with other counties, this profile is higher on ${high || "selected insurance features"} and lower on ${low || "selected insurance features"}.</p>` +
    `</article>`;
  }).join("");
}

function drawInsuranceHousingControls() {
  document.querySelectorAll("[data-insurance-housing-filter-toggle]").forEach(button => {
    const filter = button.getAttribute("data-insurance-housing-filter-toggle");
    const active = filter === "risk" ? insuranceHousingRiskFilterEnabled
      : filter === "economic" ? insuranceHousingEconomicFilterEnabled
      : filter === "migration" ? insuranceHousingMigrationFilterEnabled
      : insuranceHousingProfileFilterEnabled;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", active ? "true" : "false");
  });
  [
    ["insuranceHousingRiskLegend", insuranceHousingRiskFilterEnabled],
    ["insuranceHousingEconomicLegend", insuranceHousingEconomicFilterEnabled],
    ["insuranceHousingMigrationLegend", insuranceHousingMigrationFilterEnabled],
    ["insuranceHousingProfileLegend", insuranceHousingProfileFilterEnabled],
  ].forEach(([id, active]) => {
    const legend = document.getElementById(id);
    if (legend) {
      legend.hidden = !active;
      if (!active) legend.innerHTML = "";
    }
  });
  const baseConfig = {
    riskEnabled: insuranceHousingRiskFilterEnabled,
    economicEnabled: insuranceHousingEconomicFilterEnabled,
    migrationEnabled: insuranceHousingMigrationFilterEnabled,
    insuranceEnabled: insuranceHousingProfileFilterEnabled,
    riskRating: activeInsuranceHousingRiskRating,
    economicProfile: activeInsuranceHousingEconomicProfile,
    migrationProfile: activeInsuranceHousingMigrationProfile,
    insuranceProfile: activeInsuranceHousingProfile,
  };
  if (insuranceHousingRiskFilterEnabled) {
    drawRiskButtonLegend("insuranceHousingRiskLegend", activeInsuranceHousingRiskRating, rating => {
      activeInsuranceHousingRiskRating = rating;
      drawInsuranceHousingResponseModule();
    }, rating => filterHousingSeries({ ...baseConfig, riskRating: rating }).length > 0);
  }
  if (insuranceHousingEconomicFilterEnabled) {
    drawProfileButtonLegend("insuranceHousingEconomicLegend", migrationEconomicProfiles(), activeInsuranceHousingEconomicProfile, profile => {
      activeInsuranceHousingEconomicProfile = Number(profile);
      drawInsuranceHousingResponseModule();
    }, profile => economicLineColor(profile.profile), profile => filterHousingSeries({ ...baseConfig, economicProfile: profile.profile }).length > 0);
  }
  if (insuranceHousingMigrationFilterEnabled) {
    drawProfileButtonLegend("insuranceHousingMigrationLegend", migrationClusterProfiles(), activeInsuranceHousingMigrationProfile, profile => {
      activeInsuranceHousingMigrationProfile = Number(profile);
      drawInsuranceHousingResponseModule();
    }, profile => profileColors[Number(profile.profile) % profileColors.length] || "#5e6872", profile => filterHousingSeries({ ...baseConfig, migrationProfile: profile.profile }).length > 0);
  }
  if (insuranceHousingProfileFilterEnabled) {
    drawProfileButtonLegend("insuranceHousingProfileLegend", insuranceProfiles(), activeInsuranceHousingProfile, profile => {
      activeInsuranceHousingProfile = Number(profile);
      drawInsuranceHousingResponseModule();
    }, profile => profileColors[(Number(profile.profile) + 2) % profileColors.length] || "#5e6872", profile => filterHousingSeries({ ...baseConfig, insuranceProfile: profile.profile }).length > 0);
  }
}

function drawInsuranceHousingResponseModule() {
  drawInsuranceHousingControls();
  const config = {
    riskEnabled: insuranceHousingRiskFilterEnabled,
    economicEnabled: insuranceHousingEconomicFilterEnabled,
    migrationEnabled: insuranceHousingMigrationFilterEnabled,
    insuranceEnabled: insuranceHousingProfileFilterEnabled,
    riskRating: activeInsuranceHousingRiskRating,
    economicProfile: activeInsuranceHousingEconomicProfile,
    migrationProfile: activeInsuranceHousingMigrationProfile,
    insuranceProfile: activeInsuranceHousingProfile,
  };
  const rows = filterHousingSeries(config);
  drawHousingTrendChart("insuranceHousingTrendChart", "insuranceHousingTitle", rows, housingSelectionLabel(config), housingLineColor(config));
}

function relationshipRowsWithInsurance(metricKey = activeInsuranceMetric) {
  const base = new Map((data.migrationTrend?.housingRelationship || []).map(row => [row.fips, row]));
  return (data.insuranceTrends?.countyMetricChanges || [])
    .filter(row => row.metric === metricKey && base.has(row.fips) && row.change != null)
    .map(row => ({ ...base.get(row.fips), insuranceChange: row.change, metric: metricKey }));
}

function drawInsuranceRelationshipLegend() {
  const legend = document.getElementById("insuranceRelationshipMetricLegend");
  if (!legend) return;
  const pairs = [
    ["insurance-housing", "Insurance vs housing"],
    ["insurance-migration", "Insurance vs migration"],
    ["migration-housing", "Migration vs housing"],
  ];
  legend.innerHTML = "";
  pairs.forEach(([key, label], index) => {
    const item = document.createElement("button");
    item.type = "button";
    item.className = `risk-toggle-button${key === activeInsuranceRelationshipPair ? " active" : ""}`;
    item.setAttribute("aria-pressed", key === activeInsuranceRelationshipPair ? "true" : "false");
    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.background = profileColors[index] || "#5e6872";
    item.appendChild(swatch);
    item.appendChild(document.createTextNode(label));
    item.addEventListener("click", () => {
      activeInsuranceRelationshipPair = key;
      drawInsuranceRelationshipScatter();
    });
    legend.appendChild(item);
  });
}

function drawInsuranceRelationshipRiskLegend() {
  const legend = document.getElementById("insuranceRelationshipRiskLegend");
  if (!legend) return;
  legend.innerHTML = "";
  data.meta.riskRatings.forEach(rating => {
    const item = document.createElement("span");
    item.className = "legend-item";
    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.background = colors[rating] || "#5e6872";
    item.appendChild(swatch);
    item.appendChild(document.createTextNode(rating));
    legend.appendChild(item);
  });
}

function drawInsuranceRelationshipScatter() {
  const svg = document.getElementById("insuranceRelationshipScatter");
  if (!svg) return;
  drawInsuranceRelationshipLegend();
  drawInsuranceRelationshipRiskLegend();
  clear(svg);
  const metric = activeInsuranceMetricMeta();
  const rows = relationshipRowsWithInsurance(metric?.key);
  const pair = activeInsuranceRelationshipPair;
  const xKey = pair === "migration-housing" ? "migrationChange" : "insuranceChange";
  const yKey = pair === "insurance-migration" ? "migrationChange" : "housingChange";
  const xLabel = xKey === "insuranceChange" ? `Change in ${metric?.label || "insurance metric"}` : "Change in net migration per 1,000 residents";
  const yLabel = yKey === "migrationChange" ? "Change in net migration per 1,000 residents" : "Change in housing market YOY index";
  const dims = chartDims(svg);
  svg.setAttribute("viewBox", `0 0 ${dims.width} ${dims.height}`);
  const valid = rows.filter(row => row[xKey] != null && row[yKey] != null);
  if (!valid.length) {
    add("text", svg, { x: dims.left, y: dims.top + 28, class: "axis-label" }).textContent = "No county-level records match the selected relationship.";
    return;
  }
  const xSymmetric = xKey !== "insuranceChange" || metric?.centerZero;
  const ySymmetric = yKey !== "insuranceChange" || metric?.centerZero;
  const xDomain = fullDomainFromValues(valid.map(row => Number(row[xKey])), { symmetric: xSymmetric, includeZero: true, padRatio: 0.08 });
  const yDomain = fullDomainFromValues(valid.map(row => Number(row[yKey])), { symmetric: ySymmetric, includeZero: true, padRatio: 0.08 });
  const x = makeScale(xDomain, [dims.left, dims.width - dims.right]);
  const y = makeScale(yDomain, [dims.height - dims.bottom, dims.top]);
  drawRelationshipAxes(svg, dims, x, y, xDomain, yDomain, xLabel, yLabel);
  valid.forEach(row => {
    const cx = x(Number(row[xKey]));
    const cy = y(Number(row[yKey]));
    add("circle", svg, { cx, cy, r: 3.4, fill: colors[row.riskRating] || "#5e6872", opacity: 0.65 });
  });
}

function advanceInsuranceTrendFrame() {
  const metrics = insuranceMetrics();
  const groups = migrationEconomicGroups();
  if (!metrics.length || !groups.length) return;
  const index = metrics.findIndex(metric => metric.key === activeInsuranceMetric);
  const nextIndex = (index + 1 + metrics.length) % metrics.length;
  activeInsuranceMetric = metrics[nextIndex].key;
  if (nextIndex === 0) {
    const groupIndex = groups.findIndex(group => Number(group.profile) === Number(activeInsuranceEconomicProfile));
    activeInsuranceEconomicProfile = groups[(groupIndex + 1 + groups.length) % groups.length].profile;
  }
  drawInsuranceTrendModule();
}

function startInsuranceTrendTimer() {
  stopInsuranceTrendTimer();
  if (insuranceTrendPaused) return;
  insuranceTrendTimer = window.setInterval(advanceInsuranceTrendFrame, insuranceTrendFrameMs);
}

function stopInsuranceTrendTimer() {
  if (insuranceTrendTimer) window.clearInterval(insuranceTrendTimer);
  insuranceTrendTimer = null;
}

function updateInsuranceTrendActions() {
  const toggle = document.getElementById("insuranceTrendPlayToggle");
  if (!toggle) return;
  toggle.textContent = insuranceTrendPaused ? "Resume" : "Pause";
  toggle.classList.toggle("active", insuranceTrendPaused);
  toggle.setAttribute("aria-label", insuranceTrendPaused ? "Resume insurance trend animation" : "Pause insurance trend animation");
}

function advanceInsuranceRiskTrendFrame() {
  const ratings = data.meta.riskRatings.filter(rating => insuranceRiskMetricRows(rating, activeInsuranceRiskMetric).length > 0);
  if (!ratings.length) return;
  const index = ratings.indexOf(activeInsuranceRiskRating);
  activeInsuranceRiskRating = ratings[(index + 1 + ratings.length) % ratings.length];
  drawInsuranceRiskTrendModule();
}

function startInsuranceRiskTrendTimer() {
  stopInsuranceRiskTrendTimer();
  if (insuranceRiskTrendPaused) return;
  insuranceRiskTrendTimer = window.setInterval(advanceInsuranceRiskTrendFrame, insuranceRiskTrendFrameMs);
}

function stopInsuranceRiskTrendTimer() {
  if (insuranceRiskTrendTimer) window.clearInterval(insuranceRiskTrendTimer);
  insuranceRiskTrendTimer = null;
}

function updateInsuranceRiskTrendActions() {
  const toggle = document.getElementById("insuranceRiskTrendPlayToggle");
  if (!toggle) return;
  toggle.textContent = insuranceRiskTrendPaused ? "Resume" : "Pause";
  toggle.classList.toggle("active", insuranceRiskTrendPaused);
  toggle.setAttribute("aria-label", insuranceRiskTrendPaused ? "Resume risk insurance animation" : "Pause risk insurance animation");
}

function migrationDirection(value) {
  if (value == null || Number.isNaN(value)) return "has no clear migration-rate reading";
  if (value > 5) return "the group has strong net in-migration";
  if (value > 1) return "the group has net in-migration";
  if (value < -5) return "the group has strong net out-migration";
  if (value < -1) return "the group has net out-migration";
  return "net migration is roughly balanced";
}

function migrationShiftPhrase(start, end) {
  if (start == null || end == null || Number.isNaN(start) || Number.isNaN(end)) return "the shift is unclear";
  const diff = end - start;
  if (Math.abs(diff) < 1) return "stays about level";
  return diff > 0 ? "moves higher" : "moves lower";
}

function migrationLevelPhrase(value) {
  if (value == null || Number.isNaN(value)) return "an unclear";
  if (value >= 8) return "a very high";
  if (value >= 3) return "a high";
  if (value >= -1) return "a modest";
  if (value >= -5) return "a low";
  return "a very low";
}

function migrationWindowChange(summary) {
  if (!summary || summary.minus2 == null || summary.plus2 == null) return null;
  return summary.plus2 - summary.minus2;
}

function migrationChangeText(change) {
  if (change == null || Number.isNaN(change)) return "unclear movement";
  if (Math.abs(change) < 1) return "little overall movement";
  return change > 0 ? "a clear increase" : "a clear decrease";
}

function rankedMigrationMovements() {
  const entries = [];
  Object.entries(data.migrationTrend?.summaries || {}).forEach(([rating, summary]) => {
    const change = migrationWindowChange(summary);
    if (change != null) entries.push({ label: `${rating} risk`, change, scope: "risk" });
  });
  const profileLookup = new Map(migrationEconomicProfiles().map(profile => [String(profile.profile), profile.label]));
  Object.entries(data.migrationTrend?.profileSummaries || {}).forEach(([profile, summary]) => {
    const change = migrationWindowChange(summary);
    if (change != null) entries.push({ label: profileLookup.get(profile) || `Profile ${profile}`, change, scope: "economic profile" });
  });
  Object.entries(data.migrationTrend?.profileRiskSummaries || {}).forEach(([profile, byRisk]) => {
    Object.entries(byRisk || {}).forEach(([rating, summary]) => {
      const change = migrationWindowChange(summary);
      const countyCount = Number(summary.countyCount) || 0;
      if (change != null && countyCount >= 5) {
        entries.push({ label: `${rating} risk in ${profileLookup.get(profile) || `Profile ${profile}`}`, change, scope: "combined group" });
      }
    });
  });
  return entries.sort((a, b) => Math.abs(b.change) - Math.abs(a.change));
}

function updateMigrationOverallTakeaway() {
  const container = document.getElementById("migrationOverallTakeaway");
  if (!container) return;
  const active = migrationSelectedSeries();
  const summary = active.summary || {};
  const activeChange = migrationWindowChange(summary);
  const activeSentence = `<strong>${escapeHtml(active.label)}:</strong> ${migrationDirection(summary.plus2)} two years after the incident, and the path from two years before to two years after shows ${migrationChangeText(activeChange)}.`;
  const top = rankedMigrationMovements().slice(0, 3);
  const topText = top.length
    ? ` The clearest migration shifts across the available drilldowns are in ${top.map(item => `${escapeHtml(item.label)} with ${migrationChangeText(item.change)}`).join(", ")}.`
    : "";
  container.innerHTML = `<p>${activeSentence}${topText}</p>`;
}

function updateMigrationTakeaway() {
  const container = document.getElementById("migrationTakeaway");
  if (!container) return;
  const profile = activeMigrationEconomicGroup();
  const activeRating = ensureActiveMigrationRiskRating();
  if (!profile || !activeRating) {
    container.innerHTML = `<p>No migration trend could be measured for the selected economic profile.</p>`;
    return;
  }
  const summary = data.migrationTrend?.profileRiskSummaries?.[String(activeMigrationEconomicProfile)]?.[activeMigrationRating] || {};
  const pre2 = summary.minus2;
  const pre1 = summary.minus1;
  const incident = summary.incidentYear;
  const post1 = summary.plus1;
  const post2 = summary.plus2;
  const profileLabel = profile?.label || "selected economic profile";
  container.innerHTML = `<p><strong>${escapeHtml(profileLabel)}, ${escapeHtml(activeMigrationRating)} risk counties:</strong> the selected group starts with ${migrationLevelPhrase(pre2)} net-migration rate two years before the incident (${formatMigration(pre2)}). It is ${formatMigration(incident)} in the incident year and ${formatMigration(post2)} two years after. In plain terms, ${migrationDirection(incident)} around the incident year, and the trend from one year before to one year after ${migrationShiftPhrase(pre1, post1)}.</p>`;
}

function drawMigrationShareTile() {
  const tile = document.getElementById("migrationRiskShareTile");
  if (!tile) return;
  const profile = activeMigrationEconomicGroup();
  const activeRating = ensureActiveMigrationRiskRating();
  if (!profile || !activeRating) {
    tile.innerHTML = `<span class="label">No measured risk group</span><strong>n/a</strong><span>No counties with migration data.</span>`;
    return;
  }
  const profileSummary = data.migrationTrend?.profileSummaries?.[String(activeMigrationEconomicProfile)] || {};
  const riskSummary = data.migrationTrend?.profileRiskSummaries?.[String(activeMigrationEconomicProfile)]?.[activeRating] || {};
  const total = Number(profileSummary.countyCount) || 0;
  const count = Number(riskSummary.countyCount) || 0;
  const share = total ? count / total : null;
  tile.innerHTML = `<span class="label">${escapeHtml(activeRating)} risk share</span>` +
    `<strong>${fmtPct(share)}</strong>` +
    `<span>${count.toLocaleString()} of ${total.toLocaleString()} counties in ${escapeHtml(profile.label)}.</span>`;
}

function drawRiskLegend(legendId, { pauseOnSelect = true } = {}) {
  const legend = document.getElementById(legendId);
  if (!legend) return;
  legend.innerHTML = "";
  data.meta.riskRatings.forEach(rating => {
    const item = document.createElement("button");
    item.type = "button";
    item.className = `risk-toggle-button${rating === activeRiskRating ? " active" : ""}`;
    item.setAttribute("data-risk-rating", rating);
    item.setAttribute("aria-pressed", rating === activeRiskRating ? "true" : "false");
    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.background = colors[rating];
    item.appendChild(swatch);
    item.appendChild(document.createTextNode(rating));
    item.addEventListener("click", () => {
      setActiveRiskRating(rating, { pause: pauseOnSelect, scroll: false });
    });
    legend.appendChild(item);
  });
}

function drawMigrationLegend() {
  const legend = document.getElementById("migrationLegend");
  if (!legend) return;
  legend.innerHTML = "";
  const availableRatings = availableMigrationRiskRatings(activeMigrationEconomicProfile);
  ensureActiveMigrationRiskRating();
  availableRatings.forEach(rating => {
    const item = document.createElement("button");
    item.type = "button";
    item.className = `risk-toggle-button${rating === activeMigrationRating ? " active" : ""}`;
    item.setAttribute("data-migration-rating", rating);
    item.setAttribute("aria-pressed", rating === activeMigrationRating ? "true" : "false");
    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.background = colors[rating];
    item.appendChild(swatch);
    item.appendChild(document.createTextNode(rating));
    item.addEventListener("click", () => {
      setActiveMigrationRating(rating, { pause: true });
    });
    legend.appendChild(item);
  });
}

function drawMigrationEconomicLegend() {
  const legend = document.getElementById("migrationEconomicLegend");
  if (!legend) return;
  legend.innerHTML = "";
  migrationEconomicGroups().forEach(group => {
    const item = document.createElement("button");
    item.type = "button";
    item.className = `risk-toggle-button${group.profile === activeMigrationEconomicProfile ? " active" : ""}`;
    item.setAttribute("data-migration-profile", group.profile);
    item.setAttribute("aria-pressed", group.profile === activeMigrationEconomicProfile ? "true" : "false");
    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.background = economicLineColor(group.profile);
    item.appendChild(swatch);
    item.appendChild(document.createTextNode(group.label));
    item.addEventListener("click", () => {
      setActiveMigrationEconomicProfile(group.profile, { pause: true, resetRisk: true });
    });
    legend.appendChild(item);
  });
}

function drawMigrationFilterControls() {
  document.querySelectorAll("[data-migration-filter-toggle]").forEach(button => {
    const filter = button.getAttribute("data-migration-filter-toggle");
    const active = filter === "risk" ? migrationRiskFilterEnabled : migrationProfileFilterEnabled;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", active ? "true" : "false");
  });
  const availableRatings = data.meta.riskRatings.filter(rating => migrationRowsForFilter({
    riskEnabled: true,
    profileEnabled: migrationProfileFilterEnabled,
    rating,
    profileId: activeMigrationEconomicProfile,
  }).length > 0);
  if (migrationRiskFilterEnabled && availableRatings.length && !availableRatings.includes(activeMigrationRating)) {
    activeMigrationRating = availableRatings[0];
  }
  const availableProfiles = migrationEconomicProfiles().filter(profile => migrationRowsForFilter({
    riskEnabled: migrationRiskFilterEnabled,
    profileEnabled: true,
    rating: activeMigrationRating,
    profileId: profile.profile,
  }).length > 0);
  if (migrationProfileFilterEnabled && availableProfiles.length && !availableProfiles.some(profile => Number(profile.profile) === Number(activeMigrationEconomicProfile))) {
    activeMigrationEconomicProfile = Number(availableProfiles[0].profile);
  }
  const riskLegend = document.getElementById("migrationRiskFilterLegend");
  if (riskLegend) {
    riskLegend.hidden = !migrationRiskFilterEnabled;
    riskLegend.innerHTML = "";
    if (migrationRiskFilterEnabled) {
      data.meta.riskRatings.filter(rating => migrationRowsForFilter({
        riskEnabled: true,
        profileEnabled: migrationProfileFilterEnabled,
        rating,
        profileId: activeMigrationEconomicProfile,
      }).length > 0).forEach(rating => {
        const item = document.createElement("button");
        item.type = "button";
        item.className = `risk-toggle-button${rating === activeMigrationRating ? " active" : ""}`;
        item.setAttribute("data-migration-rating-filter", rating);
        item.setAttribute("aria-pressed", rating === activeMigrationRating ? "true" : "false");
        const swatch = document.createElement("span");
        swatch.className = "swatch";
        swatch.style.background = colors[rating];
        item.appendChild(swatch);
        item.appendChild(document.createTextNode(rating));
        item.addEventListener("click", () => setActiveMigrationRating(rating));
        riskLegend.appendChild(item);
      });
    }
  }
  const profileLegend = document.getElementById("migrationProfileFilterLegend");
  if (profileLegend) {
    profileLegend.hidden = !migrationProfileFilterEnabled;
    profileLegend.innerHTML = "";
    if (migrationProfileFilterEnabled) {
      migrationEconomicProfiles().filter(profile => migrationRowsForFilter({
        riskEnabled: migrationRiskFilterEnabled,
        profileEnabled: true,
        rating: activeMigrationRating,
        profileId: profile.profile,
      }).length > 0).forEach(profile => {
        const item = document.createElement("button");
        item.type = "button";
        item.className = `risk-toggle-button${Number(profile.profile) === Number(activeMigrationEconomicProfile) ? " active" : ""}`;
        item.setAttribute("data-migration-profile-filter", profile.profile);
        item.setAttribute("aria-pressed", Number(profile.profile) === Number(activeMigrationEconomicProfile) ? "true" : "false");
        const swatch = document.createElement("span");
        swatch.className = "swatch";
        swatch.style.background = economicLineColor(profile.profile);
        item.appendChild(swatch);
        item.appendChild(document.createTextNode(profile.label));
        item.addEventListener("click", () => setActiveMigrationEconomicProfile(profile.profile));
        profileLegend.appendChild(item);
      });
    }
  }
}

function drawLegend() {
  drawRiskLegend("riskLegend", { pauseOnSelect: true });
  drawRiskLegend("economicRiskLegend", { pauseOnSelect: true });
  drawMigrationFilterControls();
}

function updateRiskLegendButtons() {
  document.querySelectorAll("[data-risk-rating]").forEach(button => {
    const isMainMapLegend = button.closest("#riskLegend") && riskViewMode === "map";
    const active = !isMainMapLegend && button.getAttribute("data-risk-rating") === activeRiskRating;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", active ? "true" : "false");
  });
}

function updateMigrationLegendButtons() {
  document.querySelectorAll("[data-migration-rating]").forEach(button => {
    const active = button.getAttribute("data-migration-rating") === activeMigrationRating;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", active ? "true" : "false");
  });
  document.querySelectorAll("[data-migration-profile]").forEach(button => {
    const active = Number(button.getAttribute("data-migration-profile")) === Number(activeMigrationEconomicProfile);
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", active ? "true" : "false");
  });
  document.querySelectorAll("[data-migration-filter-toggle]").forEach(button => {
    const filter = button.getAttribute("data-migration-filter-toggle");
    const active = filter === "risk" ? migrationRiskFilterEnabled : migrationProfileFilterEnabled;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", active ? "true" : "false");
  });
  document.querySelectorAll("[data-migration-rating-filter]").forEach(button => {
    const active = button.getAttribute("data-migration-rating-filter") === activeMigrationRating;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", active ? "true" : "false");
  });
  document.querySelectorAll("[data-migration-profile-filter]").forEach(button => {
    const active = Number(button.getAttribute("data-migration-profile-filter")) === Number(activeMigrationEconomicProfile);
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", active ? "true" : "false");
  });
}

function updatePausedActions() {
  const actions = document.getElementById("riskPausedActions");
  if (actions) actions.classList.toggle("visible", riskChartPaused);
}

function updateRiskView() {
  const chartPanel = document.getElementById("riskChartPanel");
  const mapPanel = document.getElementById("riskMapPanel");
  const commentary = document.getElementById("riskCommentary");
  const indexNote = document.getElementById("riskIndexNote");
  const weightNote = document.getElementById("riskWeightNote");
  const subhead = document.getElementById("riskViewSubhead");
  if (chartPanel) chartPanel.hidden = riskViewMode !== "chart";
  if (mapPanel) mapPanel.hidden = riskViewMode !== "map";
  const mapMode = riskViewMode === "map";
  if (mapMode) stopRiskChartTimer();
  if (commentary) commentary.hidden = mapMode;
  if (indexNote) indexNote.hidden = mapMode;
  if (weightNote) weightNote.hidden = mapMode;
  if (subhead) {
    subhead.textContent = mapMode
      ? "This map shows each county's FEMA National Risk Index rating. Green marks lower-risk counties, while orange and red mark higher-risk counties."
      : "This chart compares counties by FEMA risk level. Each line shows the typical housing market path for counties in that risk group, from one year before an incident to two years after it ends.";
  }
  document.querySelectorAll("[data-risk-view]").forEach(button => {
    button.classList.toggle("active", button.getAttribute("data-risk-view") === riskViewMode);
  });
  updateRiskLegendButtons();
}

function updateRiskCommentary() {
  const container = document.getElementById("riskCommentary");
  if (!container) return;
  container.innerHTML = riskTakeawayHtml(activeRiskRating);
}

function riskTakeawayHtml(rating) {
  const summary = data.groupSummaries[rating] || {};
  const first = summary.avgPreToMonths1To12;
  const second = summary.avgMonths1To12To13To24;
  const firstPhrase = first == null ? "has no clear first-year shift" : first >= 0 ? "strengthens in the first year" : "softens in the first year";
  const secondPhrase = second == null ? "has no clear year-two shift" : second >= 0 ? "continues strengthening in year two" : "weakens in year two";
  return `<p><strong>${escapeHtml(rating)} risk counties:</strong> The highlighted line ${firstPhrase} after an incident and ${secondPhrase}.</p>`;
}

function setActiveRiskRating(rating, { pause = false, scroll = false } = {}) {
  if (riskViewMode === "map") return;
  if (!data.meta.riskRatings.includes(rating)) return;
  activeRiskRating = rating;
  riskFrameIndex = data.meta.riskRatings.indexOf(rating);
  ensureActiveEconomicProfile();
  if (pause) {
    riskChartPaused = true;
    stopRiskChartTimer();
  }
  drawRiskChart("riskChart", activeRiskRating);
  drawEconomicRiskDrilldown();
  updateRiskLegendButtons();
  updatePausedActions();
  updateRiskCommentary();
  if (scroll) {
    document.getElementById("economicProfileSection")?.scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

function advanceEconomicProfileFrame() {
  const series = economicProfileSeriesForRisk();
  if (!series.length) return;
  const index = series.findIndex(item => item.profile === activeEconomicProfile);
  activeEconomicProfile = series[(index + 1 + series.length) % series.length].profile;
  drawEconomicRiskDrilldown();
}

function startEconomicProfileTimer() {
  stopEconomicProfileTimer();
  if (economicProfilePaused) return;
  economicProfileTimer = window.setInterval(advanceEconomicProfileFrame, economicProfileFrameMs);
}

function stopEconomicProfileTimer() {
  if (economicProfileTimer) window.clearInterval(economicProfileTimer);
  economicProfileTimer = null;
}

function updateEconomicProfileActions() {
  const toggle = document.getElementById("economicProfilePlayToggle");
  if (!toggle) return;
  toggle.textContent = economicProfilePaused ? "Resume" : "Pause";
  toggle.classList.toggle("active", economicProfilePaused);
  toggle.setAttribute("aria-label", economicProfilePaused ? "Resume economic profile animation" : "Pause economic profile animation");
}

function setActiveMigrationEconomicProfile(profileId, { pause = false, resetRisk = false } = {}) {
  const profiles = migrationEconomicProfiles();
  if (!profiles.some(profile => Number(profile.profile) === Number(profileId))) return;
  activeMigrationEconomicProfile = Number(profileId);
  if (resetRisk && migrationRiskFilterEnabled && migrationProfileFilterEnabled) {
    activeMigrationRating = data.meta.riskRatings[0] || null;
  }
  if (pause) {
    migrationPaused = true;
    stopMigrationTimer();
  }
  drawMigrationChart();
  updateMigrationLegendButtons();
  updateMigrationActions();
}

function setActiveMigrationRating(rating, { pause = false } = {}) {
  if (!data.meta.riskRatings.includes(rating)) return;
  activeMigrationRating = rating;
  if (pause) {
    migrationPaused = true;
    stopMigrationTimer();
  }
  drawMigrationChart();
  updateMigrationLegendButtons();
  updateMigrationActions();
}

function advanceMigrationFrame() {
  const ratings = availableMigrationRiskRatings(activeMigrationEconomicProfile);
  const profiles = migrationEconomicGroups();
  if (!profiles.length || !ratings.length) return;
  const ratingIndex = ratings.indexOf(activeMigrationRating);
  if (ratingIndex < ratings.length - 1) {
    setActiveMigrationRating(ratings[ratingIndex + 1]);
    return;
  }
  const profileIndex = profiles.findIndex(group => Number(group.profile) === Number(activeMigrationEconomicProfile));
  activeMigrationEconomicProfile = profiles[(profileIndex + 1 + profiles.length) % profiles.length].profile;
  activeMigrationRating = availableMigrationRiskRatings(activeMigrationEconomicProfile)[0] || null;
  drawMigrationChart();
  updateMigrationLegendButtons();
  updateMigrationActions();
}

function startMigrationTimer() {
  stopMigrationTimer();
  if (migrationPaused) return;
  migrationTimer = window.setInterval(advanceMigrationFrame, migrationFrameMs);
}

function stopMigrationTimer() {
  if (migrationTimer) window.clearInterval(migrationTimer);
  migrationTimer = null;
}

function updateMigrationActions() {
  const toggle = document.getElementById("migrationPlayToggle");
  if (!toggle) return;
  toggle.textContent = migrationPaused ? "Resume" : "Pause";
  toggle.classList.toggle("active", migrationPaused);
  toggle.setAttribute("aria-label", migrationPaused ? "Resume migration trend animation" : "Pause migration trend animation");
}

function advanceRiskFrame() {
  const ratings = data.meta.riskRatings;
  riskFrameIndex = (riskFrameIndex + 1 + ratings.length) % ratings.length;
  setActiveRiskRating(ratings[riskFrameIndex]);
}

function startRiskChartTimer() {
  stopRiskChartTimer();
  if (riskChartPaused) return;
  riskChartTimer = window.setInterval(advanceRiskFrame, riskChartFrameMs);
}

function stopRiskChartTimer() {
  if (riskChartTimer) window.clearInterval(riskChartTimer);
  riskChartTimer = null;
}

function transitionToStoryPanel(panelId) {
  const next = document.getElementById(panelId);
  if (!next || panelId === activeStoryPanel) return;
  const current = document.getElementById(activeStoryPanel);
  document.querySelectorAll(".edge-jump").forEach(jump => jump.classList.remove("visible"));
  if (current) current.classList.add("fading-out");
  window.setTimeout(() => {
    document.querySelectorAll(".story-panel").forEach(panel => {
      panel.hidden = panel.id !== panelId;
      panel.classList.remove("fading-out", "fading-in");
    });
    activeStoryPanel = panelId;
    next.hidden = false;
    next.classList.add("fading-in");
    window.setTimeout(() => next.classList.remove("fading-in"), 320);
    window.scrollTo({ top: 0, behavior: "smooth" });
    if (panelId !== "riskResponseSection") {
      stopRiskChartTimer();
    }
    if (panelId !== "economicProfileSection") {
      stopEconomicProfileTimer();
    }
    if (panelId !== "migrationTrendSection") {
      stopMigrationTimer();
    }
    if (panelId !== "insuranceProfileSection") {
      stopInsuranceTrendTimer();
      stopInsuranceRiskTrendTimer();
    }
    if (panelId === "riskResponseSection") {
      window.requestAnimationFrame(() => drawRiskChart("riskChart", activeRiskRating));
      if (!riskChartPaused) startRiskChartTimer();
    }
    if (panelId === "economicProfileSection") {
      drawEconomicRiskDrilldown();
      updateEconomicProfileActions();
      if (!economicProfilePaused) startEconomicProfileTimer();
      window.requestAnimationFrame(drawEconomicRiskDrilldown);
    }
    if (panelId === "migrationTrendSection") {
      drawMigrationChart();
      updateMigrationLegendButtons();
      updateMigrationActions();
      window.requestAnimationFrame(drawMigrationChart);
    }
    if (panelId === "insuranceProfileSection") {
      drawInsuranceTrendModule();
      updateInsuranceTrendActions();
    }
  }, 220);
}

function updateEdgeJumps(event) {
  const y = event?.clientY ?? -1;
  const topActive = y >= 0 && y <= 72;
  const bottomActive = y >= window.innerHeight - 72;
  document.querySelectorAll(".edge-jump").forEach(jump => jump.classList.remove("visible"));
  if (activeStoryPanel === "riskResponseSection" && bottomActive) {
    document.getElementById("riskBottomJump")?.classList.add("visible");
  }
  if (activeStoryPanel === "economicProfileSection") {
    if (topActive) document.getElementById("riskUpJump")?.classList.add("visible");
    if (bottomActive) document.getElementById("economicBottomJump")?.classList.add("visible");
  }
  if (activeStoryPanel === "migrationTrendSection") {
    if (topActive) document.getElementById("migrationUpJump")?.classList.add("visible");
    if (bottomActive) document.getElementById("migrationBottomJump")?.classList.add("visible");
  }
  if (activeStoryPanel === "insuranceProfileSection" && topActive) {
    document.getElementById("insuranceUpJump")?.classList.add("visible");
  }
}

function hideEdgeJumps() {
  document.querySelectorAll(".edge-jump").forEach(jump => jump.classList.remove("visible"));
}

function initRiskControls() {
  document.querySelectorAll("[data-risk-view]").forEach(button => {
    button.addEventListener("click", () => {
      riskViewMode = button.getAttribute("data-risk-view") || "chart";
      updateRiskView();
      if (riskViewMode === "map") drawCountyRiskMap();
    });
  });
  document.getElementById("riskChartResume")?.addEventListener("click", () => {
    riskChartPaused = false;
    updatePausedActions();
    startRiskChartTimer();
  });
  document.getElementById("economicProfilePlayToggle")?.addEventListener("click", () => {
    economicProfilePaused = !economicProfilePaused;
    if (economicProfilePaused) {
      stopEconomicProfileTimer();
    }
    updateEconomicProfileActions();
    if (activeStoryPanel === "economicProfileSection") startEconomicProfileTimer();
  });
  document.querySelectorAll("[data-economic-matrix-metric]").forEach(button => {
    button.addEventListener("click", () => {
      activeEconomicMatrixMetric = button.getAttribute("data-economic-matrix-metric") || "later";
      drawEconomicResponseMatrixModule();
    });
  });
  document.querySelectorAll("[data-migration-filter-toggle]").forEach(button => {
    button.addEventListener("click", () => {
      const filter = button.getAttribute("data-migration-filter-toggle");
      if (filter === "risk") migrationRiskFilterEnabled = !migrationRiskFilterEnabled;
      if (filter === "profile") migrationProfileFilterEnabled = !migrationProfileFilterEnabled;
      stopMigrationTimer();
      drawMigrationChart();
    });
  });
  document.querySelectorAll("[data-housing-filter-toggle]").forEach(button => {
    button.addEventListener("click", () => {
      const filter = button.getAttribute("data-housing-filter-toggle");
      if (filter === "risk") housingRiskFilterEnabled = !housingRiskFilterEnabled;
      if (filter === "economic") housingEconomicFilterEnabled = !housingEconomicFilterEnabled;
      if (filter === "migration") housingMigrationFilterEnabled = !housingMigrationFilterEnabled;
      drawMigrationHousingResponseModule();
    });
  });
  document.querySelectorAll("[data-insurance-housing-filter-toggle]").forEach(button => {
    button.addEventListener("click", () => {
      const filter = button.getAttribute("data-insurance-housing-filter-toggle");
      if (filter === "risk") insuranceHousingRiskFilterEnabled = !insuranceHousingRiskFilterEnabled;
      if (filter === "economic") insuranceHousingEconomicFilterEnabled = !insuranceHousingEconomicFilterEnabled;
      if (filter === "migration") insuranceHousingMigrationFilterEnabled = !insuranceHousingMigrationFilterEnabled;
      if (filter === "insurance") insuranceHousingProfileFilterEnabled = !insuranceHousingProfileFilterEnabled;
      drawInsuranceHousingResponseModule();
    });
  });
  document.querySelectorAll("[data-insurance-trend-filter-toggle]").forEach(button => {
    button.addEventListener("click", () => {
      const filter = button.getAttribute("data-insurance-trend-filter-toggle");
      if (filter === "risk") insuranceTrendRiskFilterEnabled = !insuranceTrendRiskFilterEnabled;
      if (filter === "economic") insuranceTrendEconomicFilterEnabled = !insuranceTrendEconomicFilterEnabled;
      if (filter === "migration") insuranceTrendMigrationFilterEnabled = !insuranceTrendMigrationFilterEnabled;
      drawInsuranceTrendModule();
    });
  });
  document.getElementById("migrationPlayToggle")?.addEventListener("click", () => {
    migrationPaused = !migrationPaused;
    if (migrationPaused) {
      stopMigrationTimer();
    }
    updateMigrationActions();
    if (activeStoryPanel === "migrationTrendSection") startMigrationTimer();
  });
  document.getElementById("insuranceTrendPlayToggle")?.addEventListener("click", () => {
    insuranceTrendPaused = !insuranceTrendPaused;
    if (insuranceTrendPaused) {
      stopInsuranceTrendTimer();
    }
    updateInsuranceTrendActions();
    if (activeStoryPanel === "insuranceProfileSection") startInsuranceTrendTimer();
  });
  document.getElementById("insuranceRiskTrendPlayToggle")?.addEventListener("click", () => {
    insuranceRiskTrendPaused = !insuranceRiskTrendPaused;
    if (insuranceRiskTrendPaused) {
      stopInsuranceRiskTrendTimer();
    }
    updateInsuranceRiskTrendActions();
    if (activeStoryPanel === "insuranceProfileSection") startInsuranceRiskTrendTimer();
  });
  document.getElementById("riskDrillDown")?.addEventListener("click", () => {
    transitionToStoryPanel("economicProfileSection");
  });
  document.getElementById("riskDrillUp")?.addEventListener("click", () => {
    transitionToStoryPanel("riskResponseSection");
  });
  document.getElementById("economicDrillDown")?.addEventListener("click", () => {
    transitionToStoryPanel("migrationTrendSection");
  });
  document.getElementById("migrationDrillUp")?.addEventListener("click", () => {
    transitionToStoryPanel("economicProfileSection");
  });
  document.getElementById("migrationDrillDown")?.addEventListener("click", () => {
    transitionToStoryPanel("insuranceProfileSection");
  });
  document.getElementById("insuranceDrillUp")?.addEventListener("click", () => {
    transitionToStoryPanel("migrationTrendSection");
  });
  window.addEventListener("mousemove", updateEdgeJumps, { passive: true });
  document.addEventListener("mouseleave", hideEdgeJumps);
}

function coordWalker(coords, callback) {
  if (!Array.isArray(coords)) return;
  if (typeof coords[0] === "number" && typeof coords[1] === "number") {
    callback(coords);
    return;
  }
  coords.forEach(item => coordWalker(item, callback));
}

function mapFeatureGroup(feature) {
  const stateFips = feature.properties.stateFips;
  if (stateFips === "02") return "alaska";
  if (stateFips === "15") return "hawaii";
  return "conus";
}

function normalizedLon(lon, group) {
  if (group === "alaska" && lon > 0) return lon - 360;
  return lon;
}

function albersFactory({ parallels, lat0, lon0 }) {
  const toRad = Math.PI / 180;
  const phi1 = parallels[0] * toRad;
  const phi2 = parallels[1] * toRad;
  const phi0 = lat0 * toRad;
  const lambda0 = lon0 * toRad;
  const n = 0.5 * (Math.sin(phi1) + Math.sin(phi2));
  const c = Math.cos(phi1) ** 2 + 2 * n * Math.sin(phi1);
  const rho0 = Math.sqrt(c - 2 * n * Math.sin(phi0)) / n;
  return (lon, lat) => {
    const lambda = lon * toRad;
    const phi = lat * toRad;
    const rho = Math.sqrt(Math.max(0, c - 2 * n * Math.sin(phi))) / n;
    const theta = n * (lambda - lambda0);
    return [rho * Math.sin(theta), rho0 - rho * Math.cos(theta)];
  };
}

const projectors = {
  conus: albersFactory({ parallels: [29.5, 45.5], lat0: 37.5, lon0: -96 }),
  alaska: albersFactory({ parallels: [55, 65], lat0: 50, lon0: -154 }),
  hawaii: albersFactory({ parallels: [8, 18], lat0: 20, lon0: -157 })
};

function projectedCoord(coord, group) {
  const lon = normalizedLon(coord[0], group);
  return projectors[group](lon, coord[1]);
}

function mapBounds(features) {
  const bounds = {
    conus: [Infinity, Infinity, -Infinity, -Infinity],
    alaska: [Infinity, Infinity, -Infinity, -Infinity],
    hawaii: [Infinity, Infinity, -Infinity, -Infinity]
  };
  features.forEach(feature => {
    const group = mapFeatureGroup(feature);
    coordWalker(feature.geometry.coordinates, coord => {
      const [x, y] = projectedCoord(coord, group);
      bounds[group][0] = Math.min(bounds[group][0], x);
      bounds[group][1] = Math.min(bounds[group][1], y);
      bounds[group][2] = Math.max(bounds[group][2], x);
      bounds[group][3] = Math.max(bounds[group][3], y);
    });
  });
  return bounds;
}

function makeMapProjector(features, dims) {
  const bounds = mapBounds(features);
  const panels = {
    conus: { x: 34, y: 24, width: dims.width - 68, height: dims.height - 145 },
    alaska: { x: 58, y: dims.height - 124, width: 260, height: 92 },
    hawaii: { x: 360, y: dims.height - 105, width: 178, height: 62 }
  };
  const fitted = {};
  Object.entries(panels).forEach(([group, panel]) => {
    const [minX, minY, maxX, maxY] = bounds[group];
    const xSpan = Math.max(maxX - minX, 0.001);
    const ySpan = Math.max(maxY - minY, 0.001);
    const scale = Math.min(panel.width / xSpan, panel.height / ySpan);
    const drawnWidth = xSpan * scale;
    const drawnHeight = ySpan * scale;
    fitted[group] = {
      x: panel.x + (panel.width - drawnWidth) / 2,
      y: panel.y + (panel.height - drawnHeight) / 2,
      scale,
      minX,
      maxY
    };
  });
  return (coord, group) => {
    const [px, py] = projectedCoord(coord, group);
    const panel = fitted[group];
    const x = panel.x + (px - panel.minX) * panel.scale;
    const y = panel.y + (panel.maxY - py) * panel.scale;
    return [x, y];
  };
}

function pathFromRing(ring, project, group) {
  return ring.map((coord, index) => {
    const [x, y] = project(coord, group);
    return `${index === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
  }).join(" ") + " Z";
}

function pathFromGeometry(feature, project) {
  const group = mapFeatureGroup(feature);
  const geometry = feature.geometry;
  if (geometry.type === "Polygon") {
    return geometry.coordinates.map(ring => pathFromRing(ring, project, group)).join(" ");
  }
  if (geometry.type === "MultiPolygon") {
    return geometry.coordinates.flatMap(poly => poly.map(ring => pathFromRing(ring, project, group))).join(" ");
  }
  return "";
}

function showMapTooltip(event, feature) {
  const tooltip = document.getElementById("mapTooltip");
  const props = feature.properties;
  tooltip.innerHTML = `<strong>${props.name}</strong>` +
    `<div>NRI risk rating: ${props.riskRating}</div>`;
  tooltip.style.display = "block";
  moveMapTooltip(event);
}

function moveMapTooltip(event) {
  const tooltip = document.getElementById("mapTooltip");
  const offset = 14;
  tooltip.style.left = `${Math.min(event.clientX + offset, window.innerWidth - tooltip.offsetWidth - offset)}px`;
  tooltip.style.top = `${Math.min(event.clientY + offset, window.innerHeight - tooltip.offsetHeight - offset)}px`;
}

function hideMapTooltip() {
  document.getElementById("mapTooltip").style.display = "none";
}

function showProfileMapTooltip(event, feature, assignment, title) {
  const tooltip = document.getElementById("mapTooltip");
  const props = feature.properties;
  tooltip.innerHTML = `<strong>${props.name}</strong>` +
    `<div>${escapeHtml(title)}: ${escapeHtml(assignment?.label || "Not assigned")}</div>` +
    `<div>Assignment confidence: ${fmtConfidence(assignment?.assignmentConfidence)}</div>` +
    `<div>NRI risk rating: ${escapeHtml(props.riskRating || "n/a")}</div>`;
  tooltip.style.display = "block";
  moveMapTooltip(event);
}

function padFips(value) {
  return String(value ?? "").padStart(5, "0");
}

async function loadUsCountyFeatures() {
  if (usCountyFeatures) return usCountyFeatures;
  if (!usAtlasPromise) {
    usAtlasPromise = d3.json("https://cdn.jsdelivr.net/npm/us-atlas@3/counties-10m.json");
  }
  const usTopo = await usAtlasPromise;
  const riskByFips = new Map(data.countyRiskMap.features.map(feature => [feature.properties.fips, feature.properties]));
  usCountyFeatures = topojson
    .feature(usTopo, usTopo.objects.counties)
    .features
    .filter(feature => riskByFips.has(padFips(feature.id)))
    .map(feature => ({
      ...feature,
      properties: {
        ...feature.properties,
        ...riskByFips.get(padFips(feature.id)),
      },
    }));
  return usCountyFeatures;
}

async function loadAssignedCountyFeatures(payload) {
  if (!usAtlasPromise) {
    usAtlasPromise = d3.json("https://cdn.jsdelivr.net/npm/us-atlas@3/counties-10m.json");
  }
  const usTopo = await usAtlasPromise;
  const assignmentByFips = new Map((payload?.assignments || []).map(row => [padFips(row.fips), row]));
  const riskByFips = new Map(data.countyRiskMap.features.map(feature => [feature.properties.fips, feature.properties]));
  return topojson
    .feature(usTopo, usTopo.objects.counties)
    .features
    .filter(feature => assignmentByFips.has(padFips(feature.id)))
    .map(feature => {
      const fips = padFips(feature.id);
      const assignment = assignmentByFips.get(fips);
      const risk = riskByFips.get(fips) || {};
      return {
        ...feature,
        properties: {
          ...feature.properties,
          ...risk,
          fips,
          stateFips: fips.slice(0, 2),
          name: risk.name || assignment.countyName || fips,
        },
      };
    });
}

async function drawCountyRiskMap() {
  const svg = document.getElementById("riskMap");
  clear(svg);
  const box = svg.getBoundingClientRect();
  const dims = { width: Math.max(360, box.width), height: Math.max(360, box.height) };
  svg.setAttribute("viewBox", `0 0 ${dims.width} ${dims.height}`);
  const features = await loadUsCountyFeatures();
  const projection = d3.geoAlbersUsa();
  projection.fitExtent(
    [[34, 24], [dims.width - 34, dims.height - 34]],
    { type: "FeatureCollection", features }
  );
  const geoPath = d3.geoPath(projection);
  const layer = add("g", svg);
  features.forEach(feature => {
    const rating = feature.properties.riskRating;
    const countyPath = add("path", layer, {
      d: geoPath(feature),
      class: "county",
      fill: colors[rating] || "transparent",
      "data-rating": rating
    });
    countyPath.addEventListener("mouseenter", event => showMapTooltip(event, feature));
    countyPath.addEventListener("mousemove", moveMapTooltip);
    countyPath.addEventListener("mouseleave", hideMapTooltip);
  });
}

async function drawAssignedProfileMap(payload, svgId, legendId, title) {
  const svg = document.getElementById(svgId);
  clear(svg);
  const box = svg.getBoundingClientRect();
  const dims = { width: Math.max(360, box.width), height: Math.max(360, box.height) };
  svg.setAttribute("viewBox", `0 0 ${dims.width} ${dims.height}`);
  const features = await loadAssignedCountyFeatures(payload);
  const assignments = new Map((payload?.assignments || []).map(row => [padFips(row.fips), row]));
  const projection = d3.geoAlbersUsa();
  projection.fitExtent(
    [[34, 24], [dims.width - 34, dims.height - 34]],
    { type: "FeatureCollection", features }
  );
  const geoPath = d3.geoPath(projection);
  const layer = add("g", svg);
  features.forEach(feature => {
    const assignment = assignments.get(padFips(feature.properties.fips));
    const countyPath = add("path", layer, {
      d: geoPath(feature),
      class: "county",
      fill: assignment ? profileColors[assignment.profile % profileColors.length] : "#e6e0d6",
      "data-profile": assignment?.profile ?? "missing"
    });
    countyPath.addEventListener("mouseenter", event => showProfileMapTooltip(event, feature, assignment, title));
    countyPath.addEventListener("mousemove", moveMapTooltip);
    countyPath.addEventListener("mouseleave", hideMapTooltip);
  });
  drawAssignedProfileMapLegend(payload, legendId);
}

function drawAssignedProfileMapLegend(payload, legendId) {
  const legend = document.getElementById(legendId);
  const profiles = payload?.profiles || [];
  const counts = new Map((payload?.assignments || []).map(row => [row.profile, 0]));
  (payload?.assignments || []).forEach(row => counts.set(row.profile, (counts.get(row.profile) || 0) + 1));
  legend.innerHTML = "";
  profiles.forEach(profile => {
    const item = document.createElement("span");
    item.className = "legend-item";
    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.background = profileColors[profile.profile % profileColors.length];
    item.appendChild(swatch);
    item.appendChild(document.createTextNode(`${profile.label} (${counts.get(profile.profile) || 0})`));
    legend.appendChild(item);
  });
}

function drawEconomicRiskFeatureCards() {
  const container = document.getElementById("economicRiskFeatureCards");
  if (!container) return;
  const cards = data.economicProfiles?.cards || [];
  const byRisk = data.economicProfiles?.byRiskRating || {};
  container.innerHTML = cards.map(card => {
    const profileRows = [...(byRisk[card.riskRating] || [])].sort((a, b) => (b.share || 0) - (a.share || 0));
    const top = profileRows[0] || card.topProfiles?.[0];
    const higher = (card.higherTraits || []).map(traitPhrase).join("; ");
    const lower = (card.lowerTraits || []).map(traitPhrase).join("; ");
    const color = top ? profileColors[top.profile % profileColors.length] : "#e6e0d6";
    return `<article class="profile-card">` +
      `<h3>${escapeHtml(card.riskRating)}</h3>` +
      `<div class="count">${card.countyCount ?? 0} counties</div>` +
      `<div class="trait"><strong>Largest economic profile:</strong> ${escapeHtml(top?.label || "No assigned profile")} (${fmtPct(top?.share)})</div>` +
      `<div class="trait"><strong>Higher than typical:</strong> ${higher || "No clear high traits"}</div>` +
      `<div class="trait"><strong>Lower than typical:</strong> ${lower || "No clear low traits"}</div>` +
      `<div class="swatch" aria-hidden="true" style="background:${color}; margin-top:8px;"></div>` +
      `</article>`;
  }).join("");
}

function drawEconomicProfileCommentary() {
  const container = document.getElementById("economicProfileCommentary");
  const notes = data.economicProfiles?.commentary || [];
  container.innerHTML = notes.map(text => `<p>${escapeHtml(text)}</p>`).join("");
}

function drawEconomicProfileIntro() {
  const container = document.getElementById("economicProfileIntro");
  if (!container) return;
  container.innerHTML = `<p>The objective is to separate the broad effect of NRI risk rating from the added effect of local economic structure. Counties are assigned to economic profiles, then each profile is compared within the same risk rating to show whether housing market response differs after accounting for risk level.</p>` +
    `<p>The economic profiles below summarize the county types used for the drilldowns.</p>`;
  drawEconomicProfileSummaryCards();
}

function drawEconomicProfileSummaryCards() {
  const container = document.getElementById("economicProfileSummaryCards");
  if (!container) return;
  const profiles = data.economicProfiles?.profiles || [];
  container.innerHTML = profiles.map(profile => {
    const standoutItems = standoutEconomicFeatures(profile.topHighFeatures, 4);
    const standout = standoutItems.map(escapeHtml).join(", ");
    const color = economicLineColor(profile.profile);
    const description = standout
      ? `These counties stand out for ${standout}.`
      : escapeHtml(profile.demographicDescription || "These counties have a mixed local economy without one dominant feature.");
    return `<article class="profile-card">` +
      `<h3><span class="swatch" style="background:${color};"></span>${escapeHtml(profile.label)}</h3>` +
      `<div class="trait">${description}</div>` +
    `</article>`;
  }).join("");
}

function economicProfileSeriesForRisk(rating = activeRiskRating) {
  const byRisk = data.economicProfiles?.responseByRiskRating || {};
  return (byRisk[rating] || [])
    .map(item => ({ ...item, rows: (item.rows || []).filter(d => allOffsets.includes(d.offset)) }))
    .filter(item => item.rows.length);
}

function ensureActiveEconomicProfile() {
  const series = economicProfileSeriesForRisk();
  if (!series.length) {
    activeEconomicProfile = null;
    return null;
  }
  const current = series.find(item => item.profile === activeEconomicProfile);
  if (current) return current;
  activeEconomicProfile = series[0].profile;
  return series[0];
}

function drawEconomicProfileResponseLegend() {
  const legend = document.getElementById("economicProfileResponseLegend");
  if (!legend) return;
  const series = economicProfileSeriesForRisk();
  legend.innerHTML = "";
  series.forEach(item => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `risk-toggle-button${item.profile === activeEconomicProfile ? " active" : ""}`;
    button.setAttribute("data-economic-profile", item.profile);
    button.setAttribute("aria-pressed", item.profile === activeEconomicProfile ? "true" : "false");
    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.background = economicLineColor(item.profile);
    button.appendChild(swatch);
    button.appendChild(document.createTextNode(item.label));
    button.addEventListener("click", () => {
      activeEconomicProfile = item.profile;
      economicProfilePaused = true;
      stopEconomicProfileTimer();
      updateEconomicProfileActions();
      drawEconomicRiskDrilldown();
    });
    legend.appendChild(button);
  });
}

function drawEconomicProfileResponseChart() {
  const svg = document.getElementById("economicProfileResponseChart");
  if (!svg) return;
  clear(svg);
  const riskRows = (data.byRiskRating[activeRiskRating] || []).filter(d => allOffsets.includes(d.offset));
  const profileSeries = economicProfileSeriesForRisk();
  const selectedProfile = ensureActiveEconomicProfile();
  const dims = chartDims(svg);
  svg.setAttribute("viewBox", `0 0 ${dims.width} ${dims.height}`);
  const xDomain = [Math.min(...allOffsets), Math.max(...allOffsets)];
  const yDomain = economicResponseDomain();
  const x = makeScale(xDomain, [dims.left, dims.width - dims.right]);
  const y = makeScale(yDomain, [dims.height - dims.bottom, dims.top]);
  drawPeriodShading(svg, dims, x);
  drawAxes(svg, dims, x, y, xDomain, yDomain);
  const incidentX = x(0);
  add("line", svg, { x1: incidentX, x2: incidentX, y1: dims.top, y2: dims.height - dims.bottom, class: "incident-line" });
  add("text", svg, { x: incidentX + 7, y: dims.top + 14, class: "incident-label" }).textContent = "Incident";
  if (riskRows.length) {
    const riskColor = colors[activeRiskRating] || "#5e6872";
    const drawRiskRows = cappedSeriesRows(riskRows, yDomain);
    const riskPath = pathFor(drawRiskRows, x, y);
    add("path", svg, { d: bandPath(drawRiskRows, x, y), class: "risk-band active", fill: riskColor });
    add("path", svg, { d: riskPath, class: "risk-line focused", stroke: riskColor });
  }
  if (selectedProfile) {
    const profileColor = economicLineColor(selectedProfile.profile);
    const drawProfileRows = cappedSeriesRows(selectedProfile.rows, yDomain);
    const profilePath = pathFor(drawProfileRows, x, y);
    const band = add("path", svg, { d: bandPath(drawProfileRows, x, y), class: "profile-response-band economic-profile-band active", fill: profileColor, stroke: profileColor });
    add("path", svg, { d: profilePath, class: "profile-response-line economic-profile-line", stroke: profileColor });
    const hitLine = add("path", svg, { d: profilePath, class: "profile-response-hit-line" });
    hitLine.addEventListener("mouseenter", event => {
      band.classList.add("active");
      showProfileResponseTooltip(event, activeRiskRating, selectedProfile, nearestResponseRow(event, svg, x, selectedProfile));
    });
    hitLine.addEventListener("mousemove", event => showProfileResponseTooltip(event, activeRiskRating, selectedProfile, nearestResponseRow(event, svg, x, selectedProfile)));
    hitLine.addEventListener("mouseleave", () => {
      band.classList.add("active");
      hideTooltip();
    });
  }
}

function meanMedian(rows, minOffset, maxOffset) {
  const values = (rows || [])
    .filter(row => row.offset >= minOffset && row.offset <= maxOffset && row.median != null && !Number.isNaN(row.median))
    .map(row => Number(row.median));
  if (!values.length) return null;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function periodPairs(riskRows, profileRows, minOffset, maxOffset) {
  const riskByOffset = new Map((riskRows || []).map(row => [row.offset, row]));
  return (profileRows || [])
    .filter(row => row.offset >= minOffset && row.offset <= maxOffset)
    .map(profile => ({ profile, risk: riskByOffset.get(profile.offset) }))
    .filter(pair => pair.risk && pair.profile.median != null && pair.risk.q1 != null && pair.risk.q3 != null);
}

function mostCommonLabel(counts) {
  return Object.entries(counts).sort((a, b) => b[1] - a[1])[0]?.[0] || "within";
}

function periodIqrPosition(riskRows, profileRows, minOffset, maxOffset) {
  const counts = { above: 0, upper: 0, middle: 0, lower: 0, below: 0 };
  periodPairs(riskRows, profileRows, minOffset, maxOffset).forEach(({ profile, risk }) => {
    const midpoint = (Number(risk.q1) + Number(risk.q3)) / 2;
    if (profile.median > risk.q3) counts.above += 1;
    else if (profile.median < risk.q1) counts.below += 1;
    else if (profile.median >= midpoint) counts.upper += 1;
    else counts.lower += 1;
  });
  const total = Object.values(counts).reduce((sum, value) => sum + value, 0);
  if (!total) return "not enough data";
  const label = mostCommonLabel(counts);
  const share = counts[label] / total;
  const prefix = share >= 0.67 ? "mostly" : "often";
  if (label === "above") return `${prefix} above the risk group's IQR`;
  if (label === "below") return `${prefix} below the risk group's IQR`;
  if (label === "upper") return `${prefix} in the upper half of the risk group's IQR`;
  if (label === "lower") return `${prefix} in the lower half of the risk group's IQR`;
  return `${prefix} near the middle of the risk group's IQR`;
}

function changeComparisonPhrase(profileChange, riskChange) {
  if (profileChange == null || riskChange == null) return "cannot be compared reliably";
  const diff = profileChange - riskChange;
  if (Math.abs(diff) < 0.01) return "changes at about the same pace as the risk group";
  const profileDirection = profileChange >= 0 ? "improves" : "softens";
  const riskDirection = riskChange >= 0 ? "improves" : "softens";
  if (profileChange >= 0 && riskChange >= 0) {
    return diff > 0 ? "improves faster than the risk group" : "improves more slowly than the risk group";
  }
  if (profileChange < 0 && riskChange < 0) {
    return diff < 0 ? "softens faster than the risk group" : "softens less than the risk group";
  }
  return `${profileDirection} while the risk group ${riskDirection}`;
}

function describeEconomicProfileDifference(riskRows, profileSeries) {
  const container = document.getElementById("economicProfileResponseTakeaway");
  if (!container) return;
  if (!profileSeries || !riskRows.length) {
    container.innerHTML = `<p>No profile-specific housing market movement could be measured for the selected risk group.</p>`;
    return;
  }
  const prePosition = periodIqrPosition(riskRows, profileSeries.rows, -12, -1);
  const earlyPosition = periodIqrPosition(riskRows, profileSeries.rows, 1, 12);
  const latePosition = periodIqrPosition(riskRows, profileSeries.rows, 13, 24);
  const profilePre = meanMedian(profileSeries.rows, -12, -1);
  const profileEarly = meanMedian(profileSeries.rows, 1, 12);
  const profileLate = meanMedian(profileSeries.rows, 13, 24);
  const riskPre = meanMedian(riskRows, -12, -1);
  const riskEarly = meanMedian(riskRows, 1, 12);
  const riskLate = meanMedian(riskRows, 13, 24);
  const profileFirstChange = profilePre == null || profileEarly == null ? null : profileEarly - profilePre;
  const profileSecondChange = profileEarly == null || profileLate == null ? null : profileLate - profileEarly;
  const riskFirstChange = riskPre == null || riskEarly == null ? null : riskEarly - riskPre;
  const riskSecondChange = riskEarly == null || riskLate == null ? null : riskLate - riskEarly;
  const firstChange = changeComparisonPhrase(profileFirstChange, riskFirstChange);
  const secondChange = changeComparisonPhrase(profileSecondChange, riskSecondChange);
  container.innerHTML = `<p><strong>${escapeHtml(activeRiskRating)} risk counties:</strong> The <strong>${escapeHtml(profileSeries.label)}</strong> profile is ${prePosition} before the incident, ${earlyPosition} in months 1-12, and ${latePosition} in months 13-24. From pre-incident to months 1-12, it ${firstChange}; from months 1-12 to months 13-24, it ${secondChange}.</p>`;
}

function drawEconomicRiskDrilldown() {
  const profileTitle = document.getElementById("economicProfileResponseTitle");
  const selectedProfile = ensureActiveEconomicProfile();
  if (profileTitle) {
    const profileLabel = selectedProfile?.label ? `: ${selectedProfile.label}` : "";
    profileTitle.textContent = `Economic profiles within ${activeRiskRating} risk counties${profileLabel}`;
  }
  drawEconomicProfileResponseChart();
  drawEconomicProfileResponseLegend();
  const riskRows = (data.byRiskRating[activeRiskRating] || []).filter(d => allOffsets.includes(d.offset));
  describeEconomicProfileDifference(riskRows, selectedProfile);
}

const economicMatrixMetrics = {
  early: { label: "Early response", description: "months 1-12 minus pre-incident months", start: "pre", end: "early", timing: "in the first post-incident year" },
  later: { label: "Later response", description: "months 13-24 minus pre-incident months", start: "pre", end: "late", timing: "by the second post-incident year" },
  momentum: { label: "Momentum", description: "months 13-24 minus months 1-12", start: "early", end: "late", timing: "from the first post-incident year into the second" }
};

const economicMatrixMetricOrder = ["later", "early", "momentum"];

function economicSeriesPeriodMeans(series) {
  const rows = series?.rows || [];
  return {
    pre: meanMedian(rows, -12, -1),
    early: meanMedian(rows, 1, 12),
    late: meanMedian(rows, 13, 24)
  };
}

function economicMatrixRows() {
  const profiles = data.economicProfiles?.profiles || [];
  const byRisk = data.economicProfiles?.responseByRiskRating || {};
  const out = [];
  data.meta.riskRatings.forEach(rating => {
    const seriesByProfile = new Map((byRisk[rating] || []).map(series => [Number(series.profile), series]));
    profiles.forEach(profile => {
      const series = seriesByProfile.get(Number(profile.profile));
      const means = economicSeriesPeriodMeans(series);
      const metric = economicMatrixMetrics[activeEconomicMatrixMetric] || economicMatrixMetrics.later;
      const start = means[metric.start];
      const end = means[metric.end];
      out.push({
        rating,
        profile: Number(profile.profile),
        label: profile.label,
        countyCount: series?.countyCount || 0,
        pre: means.pre,
        early: means.early,
        late: means.late,
        start,
        end,
        value: start == null || end == null ? null : end - start
      });
    });
  });
  return out;
}

function economicMatrixValueFor(row, metricKey) {
  const metric = economicMatrixMetrics[metricKey] || economicMatrixMetrics.later;
  const start = row?.[metric.start];
  const end = row?.[metric.end];
  return start == null || end == null ? null : end - start;
}

function responseMatrixColor(value, bound) {
  if (value == null || Number.isNaN(value) || !bound) return "#f4efe6";
  const t = Math.min(1, Math.abs(value) / bound);
  const alpha = 0.16 + t * 0.72;
  return value >= 0 ? `rgba(15, 118, 110, ${alpha})` : `rgba(185, 28, 28, ${alpha})`;
}

function drawEconomicMatrixMetricToggle() {
  document.querySelectorAll("[data-economic-matrix-metric]").forEach(button => {
    const metric = button.getAttribute("data-economic-matrix-metric") || "later";
    button.classList.toggle("active", metric === activeEconomicMatrixMetric);
    button.setAttribute("aria-pressed", metric === activeEconomicMatrixMetric ? "true" : "false");
  });
}

function drawEconomicResponseMatrix() {
  const container = document.getElementById("economicResponseMatrix");
  if (!container) return;
  const profiles = data.economicProfiles?.profiles || [];
  const rows = economicMatrixRows();
  const values = rows.map(row => row.value).filter(value => value != null && !Number.isNaN(value));
  const bound = Math.max(...values.map(value => Math.abs(value)), 0.01);
  container.style.gridTemplateColumns = `120px repeat(${profiles.length}, minmax(116px, 1fr))`;
  const header = [`<div></div>`, ...profiles.map(profile => `<div class="heatmap-col-label">${escapeHtml(profile.label)}</div>`)].join("");
  const body = data.meta.riskRatings.map(rating => {
    const cells = profiles.map(profile => {
      const row = rows.find(item => item.rating === rating && Number(item.profile) === Number(profile.profile));
      const value = row?.value;
      const color = responseMatrixColor(value, bound);
      const textColor = value != null && Math.abs(value) / bound > 0.72 ? "#fffaf0" : "#111827";
      const title = `${rating}: ${profile.label}; counties: ${row?.countyCount || 0}; pre-incident median: ${formatChange(row?.pre)}; months 1-12 median: ${formatChange(row?.early)}; months 13-24 median: ${formatChange(row?.late)}; ${economicMatrixMetrics[activeEconomicMatrixMetric].label}: ${formatChange(value)}`;
      return `<div class="heatmap-cell" style="background:${color}; color:${textColor}" title="${escapeHtml(title)}">` +
        `<strong>${formatChange(value)}</strong>` +
        `</div>`;
    }).join("");
    return `<div class="heatmap-label">${escapeHtml(rating)}</div>${cells}`;
  }).join("");
  container.innerHTML = header + body;
}

function updateEconomicResponseMatrixTakeaway() {
  const container = document.getElementById("economicResponseMatrixTakeaway");
  if (!container) return;
  const rows = economicMatrixRows().filter(row => row.value != null && !Number.isNaN(row.value) && row.countyCount >= 5);
  if (!rows.length) {
    container.innerHTML = `<p>No risk x economic profile comparison could be measured.</p>`;
    return;
  }
  const notes = economicMatrixMetricOrder.map(metricKey => {
    const metric = economicMatrixMetrics[metricKey];
    const ranked = rows
      .map(row => ({ ...row, metricValue: economicMatrixValueFor(row, metricKey) }))
      .filter(row => row.metricValue != null && !Number.isNaN(row.metricValue))
      .sort((a, b) => Math.abs(b.metricValue) - Math.abs(a.metricValue));
    const top = ranked[0];
    if (!top) return "";
    const movement = top.metricValue < -0.01
      ? "softening"
      : top.metricValue > 0.01
        ? (metricKey === "momentum" ? "recovery or strengthening" : "recovery")
        : "limited movement";
    return `<p><strong>${escapeHtml(metric.label)}:</strong> ${escapeHtml(top.rating)} risk ${escapeHtml(top.label)} show the largest movement (${formatChange(top.metricValue)}), indicating ${movement} ${escapeHtml(metric.timing)}.</p>`;
  }).filter(Boolean).join("");
  container.innerHTML = notes || `<p>Economic profile differences are limited where enough counties are available for comparison.</p>`;
}

function drawEconomicResponseMatrixModule() {
  drawEconomicMatrixMetricToggle();
  drawEconomicResponseMatrix();
  updateEconomicResponseMatrixTakeaway();
}

function drawEconomicProfileChart() {
  drawProfileHeatmap(data.economicProfiles, "economicProfileHeatmap");
  drawEconomicRiskDrilldown();
  drawEconomicResponseMatrixModule();
}

function drawEconomicProfileMap() {
  const svg = document.getElementById("economicProfileMap");
  if (svg) drawAssignedProfileMap(data.economicProfiles, "economicProfileMap", "economicProfileMapLegend", "Economic profile");
}

function drawProfileCards(payload, containerId) {
  const container = document.getElementById(containerId);
  const cards = payload?.cards || [];
  container.innerHTML = cards.map(card => {
    const profileItems = (card.topProfiles || []).map(profile =>
      `<li>${fmtPct(profile.share)} are <strong>${escapeHtml(profile.label)}</strong></li>`
    ).join("");
    const higher = (card.higherTraits || []).map(traitPhrase).join("; ");
    const lower = (card.lowerTraits || []).map(traitPhrase).join("; ");
    return `<article class="profile-card">` +
      `<h3>${escapeHtml(card.riskRating)}</h3>` +
      `<div class="count">${card.countyCount ?? 0} counties</div>` +
      `<ul>${profileItems}</ul>` +
      `<div class="trait"><strong>Stands out for:</strong> ${higher || "No clear high traits"}</div>` +
      `<div class="trait"><strong>Lower on:</strong> ${lower || "No clear low traits"}</div>` +
      `</article>`;
  }).join("");
}

function drawProfileCommentary(payload, containerId) {
  const container = document.getElementById(containerId);
  const notes = payload?.commentary || [];
  container.innerHTML = notes.map(text => `<p>${escapeHtml(text)}</p>`).join("");
}

function drawProfileDominanceTiles(payload, containerId) {
  const container = document.getElementById(containerId);
  const ratings = data.meta.riskRatings;
  const byRisk = payload?.byRiskRating || {};
  container.innerHTML = ratings.map(rating => {
    const rows = [...(byRisk[rating] || [])].sort((a, b) => (b.share || 0) - (a.share || 0));
    const top = rows[0];
    const next = rows[1];
    const gap = top && next ? (top.share || 0) - (next.share || 0) : null;
    const color = top ? profileColors[top.profile % profileColors.length] : "#e6e0d6";
    return `<article class="dominance-tile" style="border-top: 5px solid ${color}">` +
      `<h3>${escapeHtml(rating)}</h3>` +
      `<div class="dominant-label">${escapeHtml(top?.label || "No assigned profile")}</div>` +
      `<div class="dominant-share">${fmtPct(top?.share)}</div>` +
      `<div class="dominant-next">${next ? `Next: ${escapeHtml(next.label)} (${fmtPct(next.share)})<br>Lead: ${fmtPct(gap)}` : "No second profile"}</div>` +
      `</article>`;
  }).join("");
}

function heatmapColor(share) {
  const value = Math.max(0, Math.min(1, Number(share) || 0));
  const alpha = 0.10 + value * 0.82;
  return `rgba(0, 90, 181, ${alpha.toFixed(3)})`;
}

function drawProfileHeatmap(payload, containerId) {
  const container = document.getElementById(containerId);
  const ratings = data.meta.riskRatings;
  const profiles = payload?.profiles || [];
  const byRisk = payload?.byRiskRating || {};
  const profileLookup = new Map(profiles.map(profile => [profile.profile, profile]));
  const shareLookup = new Map();
  ratings.forEach(rating => {
    (byRisk[rating] || []).forEach(row => shareLookup.set(`${rating}|${row.profile}`, row));
  });
  container.style.gridTemplateColumns = `120px repeat(${profiles.length}, minmax(110px, 1fr))`;
  const header = [`<div></div>`, ...profiles.map(profile => `<div class="heatmap-col-label">${escapeHtml(profile.label)}</div>`)].join("");
  const rows = ratings.map(rating => {
    const cells = profiles.map(profile => {
      const row = shareLookup.get(`${rating}|${profile.profile}`);
      const share = row?.share || 0;
      const textColor = share >= 0.45 ? "#fffaf0" : "#111827";
      return `<div class="heatmap-cell" style="background:${heatmapColor(share)}; color:${textColor}" title="${escapeHtml(rating)}: ${escapeHtml(profile.label)} ${fmtPct(share)}; counties: ${row?.counties ?? 0}">` +
        `<strong>${fmtPct(share)}</strong>` +
        `</div>`;
    }).join("");
    return `<div class="heatmap-label">${escapeHtml(rating)}</div>${cells}`;
  }).join("");
  container.innerHTML = header + rows;
}

function profileResponseDims(svg) {
  const box = svg.getBoundingClientRect();
  return { width: Math.max(300, box.width), height: Math.max(220, box.height), top: 16, right: 12, bottom: 34, left: 46 };
}

function nearestResponseRow(event, svg, x, series) {
  const rect = svg.getBoundingClientRect();
  const viewWidth = Number(svg.getAttribute("viewBox").split(" ")[2]) || rect.width;
  const svgX = ((event.clientX - rect.left) / Math.max(rect.width, 1)) * viewWidth;
  return (series.rows || []).reduce((best, row) => {
    const distance = Math.abs(x(row.offset) - svgX);
    return !best || distance < best.distance ? { row, distance } : best;
  }, null)?.row || series.rows?.[0];
}

function showProfileResponseTooltip(event, rating, series, row) {
  const tooltip = document.getElementById("riskTooltip");
  tooltip.innerHTML = `<strong>${escapeHtml(series.label)}</strong>` +
    `<div>NRI risk rating: ${escapeHtml(rating)}</div>` +
    `<div>Counties: ${series.countyCount ?? "n/a"}</div>` +
    `<div>Month offset: ${row?.offset ?? "n/a"}</div>` +
    `<div>Median: ${formatChange(row?.median)}</div>`;
  tooltip.style.display = "block";
  moveTooltip(event);
}

function drawProfileResponseSvg(svg, rating, seriesRows) {
  clear(svg);
  const dims = profileResponseDims(svg);
  svg.setAttribute("viewBox", `0 0 ${dims.width} ${dims.height}`);
  const offsets = allOffsets;
  const series = (seriesRows || [])
    .map(item => ({ ...item, rows: (item.rows || []).filter(d => offsets.includes(d.offset)) }))
    .filter(item => item.rows.length);
  const yValues = series.flatMap(item => item.rows.flatMap(d => [d.q1, d.q3, d.median]));
  const xDomain = [Math.min(...offsets), Math.max(...offsets)];
  const yDomain = focusedDomainFromValues(yValues, { symmetric: true, includeZero: true, quantile: 0.9 });
  const x = makeScale(xDomain, [dims.left, dims.width - dims.right]);
  const y = makeScale(yDomain, [dims.height - dims.bottom, dims.top]);
  drawCompactPeriodShading(svg, dims, x);
  drawCompactAxes(svg, dims, x, y, xDomain, yDomain);
  const incidentX = x(0);
  add("line", svg, { x1: incidentX, x2: incidentX, y1: dims.top, y2: dims.height - dims.bottom, class: "incident-line" });
  series.forEach(item => {
    const drawRows = cappedSeriesRows(item.rows, yDomain);
    const linePath = pathFor(drawRows, x, y);
    const color = profileColors[item.profile % profileColors.length] || "#5e6872";
    const band = add("path", svg, { d: bandPath(drawRows, x, y), class: "profile-response-band", fill: color });
    add("path", svg, { d: linePath, class: "profile-response-line", stroke: color });
    const hitLine = add("path", svg, { d: linePath, class: "profile-response-hit-line" });
    hitLine.addEventListener("mouseenter", event => {
      band.classList.add("active");
      showProfileResponseTooltip(event, rating, item, nearestResponseRow(event, svg, x, item));
    });
    hitLine.addEventListener("mousemove", event => showProfileResponseTooltip(event, rating, item, nearestResponseRow(event, svg, x, item)));
    hitLine.addEventListener("mouseleave", () => {
      band.classList.remove("active");
      hideTooltip();
    });
  });
}

function drawProfileResponseCharts(payload, containerId) {
  const container = document.getElementById(containerId);
  const ratings = data.meta.riskRatings;
  const byRisk = payload?.responseByRiskRating || {};
  container.innerHTML = ratings.map((rating, index) =>
    `<article class="profile-response-panel">` +
      `<h3>${escapeHtml(rating)}</h3>` +
      `<svg id="${containerId}-${index}" class="profile-response-chart" role="img" aria-label="Housing response lines for ${escapeHtml(rating)} risk counties"></svg>` +
    `</article>`
  ).join("");
  ratings.forEach((rating, index) => {
    const svg = document.getElementById(`${containerId}-${index}`);
    drawProfileResponseSvg(svg, rating, byRisk[rating] || []);
  });
}

function drawProfileLineLegend(payload, legendId) {
  const legend = document.getElementById(legendId);
  const profiles = payload?.profiles || [];
  legend.innerHTML = "";
  profiles.forEach(profile => {
    const item = document.createElement("span");
    item.className = "legend-item";
    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.background = profileColors[profile.profile % profileColors.length];
    item.appendChild(swatch);
    item.appendChild(document.createTextNode(profile.label));
    legend.appendChild(item);
  });
}

function drawProfileResponseTakeaway(payload, containerId) {
  const container = document.getElementById(containerId);
  container.innerHTML = `<p>${escapeHtml(payload?.responseTakeaway || "No profile-specific housing market movement could be measured.")}</p>`;
}

function drawProfileContrastTable(payload, bodyId) {
  const body = document.getElementById(bodyId);
  const rows = payload?.featureContrasts || [];
  body.innerHTML = rows.map(row => {
    const higher = (row.higher || []).map(item => `<li>${traitPhrase(item)}</li>`).join("");
    const lower = (row.lower || []).map(item => `<li>${traitPhrase(item)}</li>`).join("");
    return `<tr>` +
      `<td><strong>${escapeHtml(row.riskRating)}</strong></td>` +
      `<td><ul class="trait-list">${higher}</ul></td>` +
      `<td><ul class="trait-list">${lower}</ul></td>` +
      `</tr>`;
  }).join("");
}

function redraw() {
  drawRiskChart();
  updateRiskView();
  updateRiskCommentary();
  if (riskViewMode === "map") drawCountyRiskMap();
  drawEconomicProfileChart();
  drawEconomicProfileMap();
  drawMigrationChart();
  drawInsuranceTrendModule();
}

window.addEventListener("resize", redraw);
drawLegend();
initRiskControls();
updatePausedActions();
updateEconomicProfileActions();
updateMigrationActions();
updateInsuranceTrendActions();
updateInsuranceRiskTrendActions();
updateRiskView();
drawEconomicProfileIntro();
drawEconomicProfileCommentary();
redraw();
startRiskChartTimer();
</script>
</body>
</html>
"""


def build_stormhouse_page() -> dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    windows = build_complete_windows()
    payload = build_payload(windows)
    DATA_JS.write_text("window.STORMHOUSE_DATA = " + json.dumps(payload, separators=(",", ":")) + ";\n", encoding="utf-8")
    HTML_PATH.write_text(HTML, encoding="utf-8")
    return {
        "page": "stormhouse",
        "html_file": HTML_PATH.name,
        "data_file": DATA_JS.name,
        "written_paths": [HTML_PATH, DATA_JS],
        "meta": payload["meta"],
    }
