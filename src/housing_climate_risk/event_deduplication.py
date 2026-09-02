from __future__ import annotations

import re
from collections.abc import Iterable

import pandas as pd


FEMA_INCIDENT_MATCH_TOLERANCE_DAYS = 7
CROSS_SOURCE_MATCH_TOLERANCE_DAYS = 3


def normalize_event_name(value: object) -> str:
    """Return a stable comparison label for provider event names."""

    if value is None or pd.isna(value):
        return ""
    return re.sub(r"[^A-Z0-9]+", " ", str(value).upper()).strip()


def event_family(*values: object) -> str:
    """Map provider-specific incident labels to broad cross-source families."""

    text = " ".join(normalize_event_name(value) for value in values if value is not None)
    if re.search(r"WILDFIRE|WILD FIRE|FOREST FIRE|BRUSH FIRE", text):
        return "fire"
    if re.search(r"EARTHQUAKE|VOLCAN|LANDSLIDE|MUDSLIDE|TSUNAMI", text):
        return "geologic"
    if re.search(r"DROUGHT|EXTREME HEAT|EXCESSIVE HEAT|HEAT WAVE", text):
        return "heat_drought"
    if re.search(
        r"HURRICANE|TROPICAL|TYPHOON|CYCLONE|STORM|TORNADO|HAIL|WIND|"
        r"LIGHTNING|FLOOD|SURGE|RAIN|SNOW|ICE|BLIZZARD|FREEZE|COLD|"
        r"AVALANCHE|SLEET|WINTER|COASTAL",
        text,
    ):
        return "weather"
    return normalize_event_name(values[0] if values else "") or "other"


def _joined_unique(values: Iterable[object], *, numeric: bool = False) -> str:
    cleaned = {str(value).strip() for value in values if value is not None and not pd.isna(value) and str(value).strip()}
    if numeric:
        ordered = sorted(cleaned, key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value))
    else:
        ordered = sorted(cleaned)
    return "|".join(ordered)


def _cluster_intervals(
    frame: pd.DataFrame,
    *,
    group_columns: list[str],
    start_column: str,
    end_column: str,
    tolerance_days: int,
) -> pd.Series:
    """Assign transitive interval clusters within semantic comparison groups."""

    clusters = pd.Series(index=frame.index, dtype="Int64")
    next_cluster = 0
    tolerance = pd.Timedelta(days=tolerance_days)
    for _, group in frame.groupby(group_columns, dropna=False, sort=False):
        ordered = group.sort_values([start_column, end_column, "_source_order"], na_position="last")
        cluster_end: pd.Timestamp | None = None
        for index, row in ordered.iterrows():
            start = row[start_column]
            end = row[end_column]
            if pd.isna(start):
                next_cluster += 1
                clusters.at[index] = next_cluster
                cluster_end = None
                continue
            end = start if pd.isna(end) else end
            if cluster_end is None or start > cluster_end + tolerance:
                next_cluster += 1
                cluster_end = end
            else:
                cluster_end = max(cluster_end, end)
            clusters.at[index] = next_cluster
    return clusters


def canonicalize_fema_declarations(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse county-level FEMA records that describe the same incident.

    Records are compared within county and normalized declaration title, then
    clustered when their incident intervals overlap (with a short tolerance).
    A major-disaster declaration is preferred over an emergency declaration.
    """

    if frame.empty:
        return frame.copy()
    required = {
        "fips",
        "disasterNumber",
        "declarationType",
        "incidentType",
        "declarationTitle",
        "incident_begin_date",
        "incident_end_date",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"FEMA canonicalization is missing columns: {sorted(missing)}")

    work = frame.copy().reset_index(drop=True)
    work["_source_order"] = range(len(work))
    work["incident_begin_date"] = pd.to_datetime(work["incident_begin_date"], errors="coerce")
    work["incident_end_date"] = pd.to_datetime(work["incident_end_date"], errors="coerce").fillna(
        work["incident_begin_date"]
    )
    work["normalized_event_name"] = work["declarationTitle"].map(normalize_event_name)
    empty_name = work["normalized_event_name"].eq("")
    work.loc[empty_name, "normalized_event_name"] = (
        "DISASTER " + work.loc[empty_name, "disasterNumber"].astype(str)
    )
    work["event_family"] = [
        event_family(incident_type, title)
        for incident_type, title in zip(work["incidentType"], work["declarationTitle"])
    ]
    work["_incident_cluster"] = _cluster_intervals(
        work,
        group_columns=["fips", "normalized_event_name"],
        start_column="incident_begin_date",
        end_column="incident_end_date",
        tolerance_days=FEMA_INCIDENT_MATCH_TOLERANCE_DAYS,
    )
    declaration_priority = {"DR": 0, "EM": 1, "FM": 2}
    work["_declaration_priority"] = work["declarationType"].map(declaration_priority).fillna(9)
    refresh_column = "lastRefresh" if "lastRefresh" in work else None
    if refresh_column:
        work["_last_refresh"] = pd.to_datetime(work[refresh_column], errors="coerce")
    else:
        work["_last_refresh"] = pd.NaT
    work["_disaster_number_sort"] = pd.to_numeric(work["disasterNumber"], errors="coerce")
    work = work.sort_values(
        ["_incident_cluster", "_declaration_priority", "_last_refresh", "_disaster_number_sort", "_source_order"],
        ascending=[True, True, False, True, True],
        na_position="last",
    )

    rows: list[pd.Series] = []
    for _, group in work.groupby("_incident_cluster", sort=True):
        representative = group.iloc[0].copy()
        representative["incident_begin_date"] = group["incident_begin_date"].min()
        representative["incident_end_date"] = group["incident_end_date"].max()
        representative["source_record_count"] = int(len(group))
        representative["declaration_count"] = int(group["disasterNumber"].dropna().astype(str).nunique())
        representative["associated_disaster_numbers"] = _joined_unique(group["disasterNumber"], numeric=True)
        representative["associated_declaration_types"] = _joined_unique(group["declarationType"])
        representative["associated_source_record_ids"] = _joined_unique(
            group["id"] if "id" in group else group["_source_order"]
        )
        representative["canonicalization_reason"] = (
            "fema_semantic_duplicate" if len(group) > 1 else "single_source_record"
        )
        rows.append(representative)

    result = pd.DataFrame(rows).sort_values(["fips", "incident_begin_date", "disasterNumber"]).reset_index(drop=True)
    return result.drop(
        columns=[
            "_source_order",
            "_incident_cluster",
            "_declaration_priority",
            "_last_refresh",
            "_disaster_number_sort",
        ],
        errors="ignore",
    )


def canonicalize_climate_events(fema: pd.DataFrame, noaa: pd.DataFrame) -> pd.DataFrame:
    """Return one county-event record after within-NOAA and cross-source matching."""

    standardized_columns = [
        "event_source",
        "source_event_id",
        "fips",
        "event_type",
        "event_name",
        "event_start",
        "event_end",
        "total_damage_amount",
    ]
    frames = []
    for source, source_frame in (("fema", fema), ("noaa", noaa)):
        if source_frame.empty:
            continue
        missing = set(standardized_columns).difference(source_frame.columns)
        if missing:
            raise ValueError(f"{source.upper()} event canonicalization is missing columns: {sorted(missing)}")
        copy = source_frame.copy()
        copy["event_source"] = source
        copy["event_start"] = pd.to_datetime(copy["event_start"], errors="coerce")
        copy["event_end"] = pd.to_datetime(copy["event_end"], errors="coerce").fillna(copy["event_start"])
        copy["normalized_event_name"] = copy["event_name"].map(normalize_event_name)
        copy["event_family"] = [
            event_family(event_type, event_name)
            for event_type, event_name in zip(copy["event_type"], copy["event_name"])
        ]
        copy["_source_order"] = range(len(copy))
        frames.append(copy)
    if not frames:
        return pd.DataFrame(columns=standardized_columns)

    fema_work = next((frame for frame in frames if frame["event_source"].iat[0] == "fema"), pd.DataFrame())
    noaa_work = next((frame for frame in frames if frame["event_source"].iat[0] == "noaa"), pd.DataFrame())

    # NOAA event IDs can describe several records in one episode. Collapse only
    # within a county/episode; without an episode ID, require the normalized name
    # and interval to match exactly.
    noaa_rows: list[pd.Series] = []
    if not noaa_work.empty:
        episode = noaa_work.get("episode_id", pd.Series(index=noaa_work.index, dtype=object)).astype("string")
        fallback = (
            noaa_work["normalized_event_name"]
            + "|"
            + noaa_work["event_start"].astype(str)
            + "|"
            + noaa_work["event_end"].astype(str)
        )
        noaa_work["_episode_key"] = episode.where(episode.notna() & episode.ne(""), fallback)
        for _, group in noaa_work.groupby(["fips", "_episode_key"], dropna=False, sort=False):
            damage = pd.to_numeric(group["total_damage_amount"], errors="coerce")
            representative = group.loc[damage.fillna(-1).idxmax()].copy()
            representative["total_damage_amount"] = damage.max()
            representative["source_record_count"] = int(
                pd.to_numeric(group.get("source_record_count", 1), errors="coerce").fillna(1).sum()
            )
            representative["associated_source_event_ids"] = _joined_unique(group["source_event_id"], numeric=True)
            noaa_rows.append(representative)
        noaa_work = pd.DataFrame(noaa_rows).reset_index(drop=True)

    canonical_rows: list[pd.Series] = []
    fema_matches: dict[int, list[pd.Series]] = {index: [] for index in fema_work.index}
    unmatched_noaa: list[pd.Series] = []
    tolerance = pd.Timedelta(days=CROSS_SOURCE_MATCH_TOLERANCE_DAYS)
    for _, noaa_row in noaa_work.iterrows():
        if fema_work.empty:
            unmatched_noaa.append(noaa_row)
            continue
        candidates = fema_work.loc[
            fema_work["fips"].eq(noaa_row["fips"])
            & fema_work["event_family"].eq(noaa_row["event_family"])
            & fema_work["event_start"].le(noaa_row["event_end"] + tolerance)
            & fema_work["event_end"].ge(noaa_row["event_start"] - tolerance)
        ].copy()
        if candidates.empty:
            unmatched_noaa.append(noaa_row)
            continue
        candidates["_distance"] = (
            (candidates["event_start"] - noaa_row["event_start"]).abs()
            + (candidates["event_end"] - noaa_row["event_end"]).abs()
        )
        best_index = candidates.sort_values(["_distance", "event_start"]).index[0]
        fema_matches[best_index].append(noaa_row)

    for index, fema_row in fema_work.iterrows():
        representative = fema_row.copy()
        matches = fema_matches.get(index, [])
        fema_ids = str(representative.get("associated_source_event_ids") or representative["source_event_id"])
        source_keys = [f"fema:{value}" for value in fema_ids.split("|") if value]
        source_keys.extend(f"noaa:{row['source_event_id']}" for row in matches)
        representative["has_fema"] = True
        representative["has_noaa"] = bool(matches)
        representative["event_sources"] = "fema|noaa" if matches else "fema"
        representative["associated_source_event_keys"] = "|".join(source_keys)
        representative["source_event_count"] = len(source_keys)
        representative["source_record_count"] = int(representative.get("source_record_count", 1)) + sum(
            int(row.get("source_record_count", 1)) for row in matches
        )
        if matches:
            damages = pd.to_numeric(pd.Series([row["total_damage_amount"] for row in matches]), errors="coerce")
            representative["total_damage_amount"] = damages.max()
            representative["canonicalization_reason"] = "cross_source_match"
        else:
            representative["canonicalization_reason"] = (
                "fema_semantic_duplicate" if len(source_keys) > 1 else "single_source_record"
            )
        canonical_rows.append(representative)

    for noaa_row in unmatched_noaa:
        representative = noaa_row.copy()
        ids = str(representative.get("associated_source_event_ids") or representative["source_event_id"])
        source_keys = [f"noaa:{value}" for value in ids.split("|") if value]
        representative["has_fema"] = False
        representative["has_noaa"] = True
        representative["event_sources"] = "noaa"
        representative["associated_source_event_keys"] = "|".join(source_keys)
        representative["source_event_count"] = len(source_keys)
        representative["canonicalization_reason"] = (
            "noaa_episode_duplicate" if len(source_keys) > 1 else "single_source_record"
        )
        canonical_rows.append(representative)

    result = pd.DataFrame(canonical_rows)
    if result.empty:
        return pd.DataFrame(columns=standardized_columns)
    result["event_start_month"] = result["event_start"].dt.to_period("M").dt.to_timestamp()
    result["event_end_month"] = result["event_end"].dt.to_period("M").dt.to_timestamp()
    result["event_key"] = (
        result["event_source"].astype(str)
        + ":"
        + result["source_event_id"].astype(str)
        + ":"
        + result["fips"].astype(str).str.zfill(5)
        + ":"
        + result["event_start_month"].astype(str)
    )
    result["semantic_duplicate_count"] = result["source_event_count"].fillna(1).astype(int) - 1
    keep = [
        *standardized_columns,
        "event_start_month",
        "event_end_month",
        "event_key",
        "normalized_event_name",
        "event_family",
        "has_fema",
        "has_noaa",
        "event_sources",
        "associated_source_event_keys",
        "source_event_count",
        "source_record_count",
        "semantic_duplicate_count",
        "canonicalization_reason",
    ]
    return result[keep].sort_values(["fips", "event_start", "event_key"]).reset_index(drop=True)
