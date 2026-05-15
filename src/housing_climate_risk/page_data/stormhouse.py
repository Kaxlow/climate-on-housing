"""Build the stormhouse county housing response visualization."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from housing_climate_risk.data_sources.raw import load_profile_inputs
from housing_climate_risk.data_sources.processed import prepare_housing_df, prepare_natural_disasters_df
from housing_climate_risk.modeling.economic_risk_profiles import build_economic_profile_outputs
from housing_climate_risk.modeling.insurance_risk_profiles import build_insurance_profile_outputs
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
    insurance_profiles = build_insurance_profile_payload()
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
        "economicProfiles": economic_profiles,
        "insuranceProfiles": insurance_profiles,
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
    .incident-line { stroke: #2f3941; stroke-width: 1.5; stroke-dasharray: 5 5; }
    .incident-label { fill: #2f3941; font-size: 12px; font-weight: 700; }
    .legend { display: flex; flex-wrap: wrap; gap: 10px 16px; color: var(--muted); font-size: 12px; margin-top: 8px; }
    .legend-item { display: inline-flex; align-items: center; gap: 6px; }
    .swatch { width: 18px; height: 3px; border-radius: 2px; display: inline-block; }
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
    .profile-response-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; margin-top: 14px; }
    .profile-response-panel { border-top: 1px solid #e6e0d6; padding-top: 10px; min-width: 0; }
    .profile-response-panel h3 { font-size: 13px; margin-bottom: 6px; }
    .profile-response-chart { width: 100%; height: 250px; display: block; }
    .profile-response-chart text { font-size: 10px; }
    .profile-response-band { stroke: none; opacity: 0; pointer-events: none; transition: opacity 120ms ease; }
    .profile-response-band.active { opacity: 0.18; }
    .profile-response-line { fill: none; stroke-width: 2.3; stroke-linejoin: round; stroke-linecap: round; }
    .profile-response-hit-line { fill: none; stroke: transparent; stroke-width: 12; stroke-linejoin: round; stroke-linecap: round; cursor: pointer; }
    .profile-map { width: 100%; height: 520px; margin-top: 14px; display: block; }
    .profile-table { width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 12px; }
    .profile-table th, .profile-table td { border-top: 1px solid #e6e0d6; padding: 8px 6px; text-align: left; vertical-align: top; }
    .profile-table th { color: #2f3941; font-size: 11px; text-transform: uppercase; letter-spacing: 0; }
    .trait-list { margin: 0; padding-left: 16px; color: var(--muted); }
    .trait-list li + li { margin-top: 3px; }
    .bar-label { fill: #2f3941; font-size: 11px; font-weight: 700; }
    .bar-axis-label { fill: var(--muted); font-size: 11px; }
    @media (max-width: 820px) {
      main { width: min(100vw - 20px, 1180px); padding-top: 18px; }
      .chart { height: 390px; }
      .map { height: 430px; }
      .profile-grid { grid-template-columns: 1fr; }
      .dominance-grid { grid-template-columns: 1fr; }
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

  <section>
    <div class="section-head">
      <div>
        <h2>Housing Market Response to Extreme Climate Incidents</h2>
        <div class="sub">This chart compares counties by FEMA risk level. Each line shows the typical housing market path for counties in that risk group, from one year before an incident to two years after it ends. Hover over a line to see the middle range for that group and a short before-and-after summary.</div>
      </div>
    </div>
    <svg id="riskChart" class="chart" role="img" aria-label="Housing market index around FEMA incidents grouped by NRI risk rating"></svg>
    <div id="riskLegend" class="legend"></div>
    <p class="note">Housing market index: this score combines prices, sale-to-list ratios, homes sold, and inventory into one number. The inputs use year-over-year changes so normal seasonal swings, like busier spring markets, have less influence.</p>
    <p class="note">Incident weighting: when a county was hit by multiple incidents, its values are averaged by month, with more recent incidents counted more heavily.</p>
    <div id="riskCommentary" class="takeaway-banner" aria-label="Takeaway on NRI risk group responses"></div>
    <div class="map-wrap">
      <h3>County Risk Map</h3>
      <div class="sub">The same FEMA risk levels are mapped county by county across the US, including Alaska and Hawaii. Green marks the lowest-risk counties and red marks the highest-risk counties.</div>
      <svg id="riskMap" class="map" role="img" aria-label="US county map colored by FEMA National Risk Index rating"></svg>
      <div id="mapLegend" class="legend"></div>
    </div>
  </section>
  <section>
    <div class="section-head">
      <div>
        <h2>Economic Profile of Risk Groups</h2>
        <div class="sub">Counties are grouped by income, wages, employment, income sources, population size, and migration patterns, then compared across FEMA National Risk Index ratings. Demographics are described after grouping, but they are not used to assign or label the economic profiles.</div>
      </div>
    </div>
    <div id="economicProfileCards" class="profile-grid"></div>
    <div id="economicProfileCommentary" class="econ-commentary key-takeaway" aria-label="Commentary on economic profile differences"></div>
    <div id="economicDominanceTiles" class="dominance-grid" aria-label="Dominant economic profile by NRI risk rating"></div>
    <div class="heatmap-wrap">
      <div id="economicProfileHeatmap" class="profile-heatmap" aria-label="Heatmap of economic profile shares by NRI risk rating"></div>
    </div>
    <div id="economicProfileResponseCharts" class="profile-response-grid" aria-label="Housing market response by economic profile and NRI risk rating"></div>
    <div id="economicProfileResponseLegend" class="legend"></div>
    <div id="economicProfileResponseTakeaway" class="takeaway-banner" aria-label="Takeaway on economic profile housing response"></div>
    <svg id="economicProfileMap" class="profile-map" role="img" aria-label="US county map colored by assigned economic profile"></svg>
    <div id="economicProfileMapLegend" class="legend"></div>
    <table class="profile-table" aria-label="Demographic descriptions of economic profiles">
      <thead>
        <tr>
          <th>Economic profile</th>
          <th>Demographic description</th>
        </tr>
      </thead>
      <tbody id="economicProfileDemographicsBody"></tbody>
    </table>
    <table class="profile-table" aria-label="Economic feature contrasts by NRI risk rating">
      <thead>
        <tr>
          <th>Risk group</th>
          <th>What is unusually high</th>
          <th>What is unusually low</th>
        </tr>
      </thead>
      <tbody id="economicContrastBody"></tbody>
    </table>
    <p class="note">The table compares the median county in each risk group with the median county nationally. "Higher" and "lower" describe the size of that gap after putting all features on the same scale.</p>
    <p class="note">The demographic descriptions are added after clustering. Race, ethnicity, age, and sex shares are not part of the economic profile model.</p>
    <p class="note">Counties are assigned probabilistically to the economic profile they most closely match. Hover over a profile in the chart or legend to see its average assignment confidence.</p>
  </section>
  <section>
    <div class="section-head">
      <div>
        <h2>Insurance Profile of Risk Groups</h2>
        <div class="sub">Counties are grouped by current home-insurance premium levels, premium growth, nonrenewal rates, and nonrenewal volatility or trend. The cards summarize the dominant insurance profile in each NRI risk group, the chart shows the full profile mix, and the table shows which insurance traits are unusually high or low.</div>
      </div>
    </div>
    <div id="insuranceProfileCards" class="profile-grid"></div>
    <div id="insuranceProfileCommentary" class="econ-commentary key-takeaway" aria-label="Commentary on insurance profile differences"></div>
    <div id="insuranceDominanceTiles" class="dominance-grid" aria-label="Dominant insurance profile by NRI risk rating"></div>
    <div class="heatmap-wrap">
      <div id="insuranceProfileHeatmap" class="profile-heatmap" aria-label="Heatmap of insurance profile shares by NRI risk rating"></div>
    </div>
    <div id="insuranceProfileResponseCharts" class="profile-response-grid" aria-label="Housing market response by insurance profile and NRI risk rating"></div>
    <div id="insuranceProfileResponseLegend" class="legend"></div>
    <div id="insuranceProfileResponseTakeaway" class="takeaway-banner" aria-label="Takeaway on insurance profile housing response"></div>
    <svg id="insuranceProfileMap" class="profile-map" role="img" aria-label="US county map colored by assigned insurance profile"></svg>
    <div id="insuranceProfileMapLegend" class="legend"></div>
    <table class="profile-table" aria-label="Insurance feature contrasts by NRI risk rating">
      <thead>
        <tr>
          <th>Risk group</th>
          <th>What is unusually high</th>
          <th>What is unusually low</th>
        </tr>
      </thead>
      <tbody id="insuranceContrastBody"></tbody>
    </table>
    <p class="note">The table compares the median county in each risk group with the median county nationally, using home-insurance premium and nonrenewal rate levels, percentiles, growth, volatility, and trend measures.</p>
    <p class="note">Counties are assigned probabilistically to the insurance profile they most closely match. Hover over a profile in the chart or legend to see its average assignment confidence.</p>
  </section>
  <div id="riskTooltip" class="tooltip" role="status" aria-live="polite"></div>
  <div id="mapTooltip" class="tooltip" role="status" aria-live="polite"></div>
</main>

<script>
const data = window.STORMHOUSE_DATA;
let usAtlasPromise = null;
let usCountyFeatures = null;
const colors = {
  "Very Low": "#16a34a",
  "Low": "#84cc16",
  "Moderate": "#facc15",
  "High": "#f97316",
  "Very High": "#dc2626"
};
const profileColors = ["#005AB5", "#DC3220", "#009E73", "#F0E442", "#7A3E9D", "#8C564B", "#00A1C9", "#111111"];
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

function pathFor(rows, x, y, key = "median") {
  return rows.map((d, i) => `${i === 0 ? "M" : "L"}${x(d.offset).toFixed(2)},${y(d[key]).toFixed(2)}`).join(" ");
}

function bandPath(rows, x, y) {
  const top = rows.map((d, i) => `${i === 0 ? "M" : "L"}${x(d.offset).toFixed(2)},${y(d.q3).toFixed(2)}`);
  const bottom = rows.slice().reverse().map(d => `L${x(d.offset).toFixed(2)},${y(d.q1).toFixed(2)}`);
  return `${top.join(" ")} ${bottom.join(" ")} Z`;
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

function drawRiskChart() {
  const svg = document.getElementById("riskChart");
  clear(svg);
  const dims = chartDims(svg);
  svg.setAttribute("viewBox", `0 0 ${dims.width} ${dims.height}`);
  const offsets = allOffsets;
  const groups = data.meta.riskRatings.map(rating => ({
    rating,
    rows: (data.byRiskRating[rating] || []).filter(d => offsets.includes(d.offset))
  })).filter(group => group.rows.length);
  const yValues = groups.flatMap(group => group.rows.flatMap(d => [d.q1, d.q3, d.median]));
  const xDomain = [Math.min(...offsets), Math.max(...offsets)];
  const yDomain = symmetricExtent(yValues);
  const x = makeScale(xDomain, [dims.left, dims.width - dims.right]);
  const y = makeScale(yDomain, [dims.height - dims.bottom, dims.top]);
  drawPeriodShading(svg, dims, x);
  drawAxes(svg, dims, x, y, xDomain, yDomain);
  const incidentX = x(0);
  add("line", svg, { x1: incidentX, x2: incidentX, y1: dims.top, y2: dims.height - dims.bottom, class: "incident-line" });
  add("text", svg, { x: incidentX + 7, y: dims.top + 14, class: "incident-label" }).textContent = "Incident";
  groups.forEach(group => {
    const linePath = pathFor(group.rows, x, y);
    const band = add("path", svg, { d: bandPath(group.rows, x, y), class: "risk-band", fill: colors[group.rating] || "#5e6872", "data-rating": group.rating });
    add("path", svg, { d: linePath, class: "risk-line", stroke: colors[group.rating] || "#5e6872", "data-rating": group.rating });
    const hitLine = add("path", svg, { d: linePath, class: "risk-hit-line", "data-rating": group.rating });
    hitLine.addEventListener("mouseenter", event => {
      band.classList.add("active");
      showTooltip(event, group.rating);
    });
    hitLine.addEventListener("mousemove", moveTooltip);
    hitLine.addEventListener("mouseleave", () => {
      band.classList.remove("active");
      hideTooltip();
    });
  });
}

function drawLegend() {
  const legend = document.getElementById("riskLegend");
  legend.innerHTML = "";
  data.meta.riskRatings.forEach(rating => {
    const item = document.createElement("span");
    item.className = "legend-item";
    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.background = colors[rating];
    item.appendChild(swatch);
    item.appendChild(document.createTextNode(rating));
    legend.appendChild(item);
  });
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

function drawMapLegend() {
  const legend = document.getElementById("mapLegend");
  const counts = data.countyRiskMap.counts || {};
  legend.innerHTML = "";
  data.meta.riskRatings.forEach(rating => {
    const item = document.createElement("span");
    item.className = "legend-item";
    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.background = colors[rating] || colors.Unrated;
    item.appendChild(swatch);
    item.appendChild(document.createTextNode(`${rating} (${counts[rating] ?? 0})`));
    legend.appendChild(item);
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

function drawEconomicProfileCards() {
  const container = document.getElementById("economicProfileCards");
  const cards = data.economicProfiles?.cards || [];
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

function drawEconomicProfileCommentary() {
  const container = document.getElementById("economicProfileCommentary");
  const notes = data.economicProfiles?.commentary || [];
  container.innerHTML = notes.map(text => `<p>${escapeHtml(text)}</p>`).join("");
}

function drawEconomicProfileChart() {
  drawProfileDominanceTiles(data.economicProfiles, "economicDominanceTiles");
  drawProfileHeatmap(data.economicProfiles, "economicProfileHeatmap");
  drawProfileResponseCharts(data.economicProfiles, "economicProfileResponseCharts");
  drawProfileLineLegend(data.economicProfiles, "economicProfileResponseLegend");
  drawProfileResponseTakeaway(data.economicProfiles, "economicProfileResponseTakeaway");
}

function drawEconomicProfileMap() {
  drawAssignedProfileMap(data.economicProfiles, "economicProfileMap", "economicProfileMapLegend", "Economic profile");
}

function drawEconomicContrastTable() {
  const body = document.getElementById("economicContrastBody");
  const rows = data.economicProfiles?.featureContrasts || [];
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

function drawEconomicProfileDemographics() {
  const body = document.getElementById("economicProfileDemographicsBody");
  const profiles = data.economicProfiles?.profiles || [];
  body.innerHTML = profiles.map(profile => {
    return `<tr>` +
      `<td><strong>${escapeHtml(profile.label)}</strong><br><span class="count">${profile.countyCount ?? 0} counties</span></td>` +
      `<td>${escapeHtml(profile.demographicDescription || "Demographic mix is close to the national county pattern.")}</td>` +
      `</tr>`;
  }).join("");
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
      return `<div class="heatmap-cell" style="background:${heatmapColor(share)}; color:${textColor}" title="${escapeHtml(rating)}: ${escapeHtml(profile.label)} ${fmtPct(share)}">` +
        `<strong>${fmtPct(share)}</strong>` +
        `${row?.counties ?? 0} counties` +
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
  const yDomain = symmetricExtent(yValues);
  const x = makeScale(xDomain, [dims.left, dims.width - dims.right]);
  const y = makeScale(yDomain, [dims.height - dims.bottom, dims.top]);
  drawCompactPeriodShading(svg, dims, x);
  drawCompactAxes(svg, dims, x, y, xDomain, yDomain);
  const incidentX = x(0);
  add("line", svg, { x1: incidentX, x2: incidentX, y1: dims.top, y2: dims.height - dims.bottom, class: "incident-line" });
  series.forEach(item => {
    const linePath = pathFor(item.rows, x, y);
    const color = profileColors[item.profile % profileColors.length] || "#5e6872";
    const band = add("path", svg, { d: bandPath(item.rows, x, y), class: "profile-response-band", fill: color });
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

function drawInsuranceProfileCards() {
  drawProfileCards(data.insuranceProfiles, "insuranceProfileCards");
}

function drawInsuranceProfileCommentary() {
  drawProfileCommentary(data.insuranceProfiles, "insuranceProfileCommentary");
}

function drawInsuranceProfileChart() {
  drawProfileDominanceTiles(data.insuranceProfiles, "insuranceDominanceTiles");
  drawProfileHeatmap(data.insuranceProfiles, "insuranceProfileHeatmap");
  drawProfileResponseCharts(data.insuranceProfiles, "insuranceProfileResponseCharts");
  drawProfileLineLegend(data.insuranceProfiles, "insuranceProfileResponseLegend");
  drawProfileResponseTakeaway(data.insuranceProfiles, "insuranceProfileResponseTakeaway");
}

function drawInsuranceProfileMap() {
  drawAssignedProfileMap(data.insuranceProfiles, "insuranceProfileMap", "insuranceProfileMapLegend", "Insurance profile");
}

function drawInsuranceContrastTable() {
  drawProfileContrastTable(data.insuranceProfiles, "insuranceContrastBody");
}

function redraw() {
  drawRiskChart();
  drawCountyRiskMap();
  drawEconomicProfileChart();
  drawEconomicProfileMap();
  drawInsuranceProfileChart();
  drawInsuranceProfileMap();
}

window.addEventListener("resize", redraw);
drawLegend();
drawMapLegend();
drawEconomicProfileCards();
drawEconomicProfileCommentary();
drawEconomicContrastTable();
drawEconomicProfileDemographics();
drawInsuranceProfileCards();
drawInsuranceProfileCommentary();
drawInsuranceContrastTable();
document.getElementById("riskCommentary").innerHTML = data.commentary.map(text => `<p>${escapeHtml(text)}</p>`).join("");
redraw();
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
