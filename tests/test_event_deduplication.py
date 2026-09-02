from __future__ import annotations

import pandas as pd

from housing_climate_risk.event_deduplication import (
    canonicalize_climate_events,
    canonicalize_fema_declarations,
)


def _fema_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": "em-row",
                "fips": "13165",
                "disasterNumber": "3616",
                "declarationType": "EM",
                "incidentType": "Tropical Storm",
                "declarationTitle": "HURRICANE HELENE",
                "incident_begin_date": "2024-09-24",
                "incident_end_date": "2024-10-07",
                "lastRefresh": "2024-10-01",
            },
            {
                "id": "dr-row",
                "fips": "13165",
                "disasterNumber": "4830",
                "declarationType": "DR",
                "incidentType": "Hurricane",
                "declarationTitle": "Hurricane Helene",
                "incident_begin_date": "2024-09-24",
                "incident_end_date": "2024-10-07",
                "lastRefresh": "2024-11-01",
            },
        ]
    )


def test_fema_em_and_dr_records_become_one_dr_incident() -> None:
    result = canonicalize_fema_declarations(_fema_rows())

    assert len(result) == 1
    assert result.iloc[0]["disasterNumber"] == "4830"
    assert result.iloc[0]["declarationType"] == "DR"
    assert result.iloc[0]["associated_disaster_numbers"] == "3616|4830"
    assert result.iloc[0]["associated_declaration_types"] == "DR|EM"
    assert result.iloc[0]["declaration_count"] == 2


def test_repeated_title_with_nonoverlapping_dates_remains_distinct() -> None:
    rows = _fema_rows().iloc[[1]].copy()
    later = rows.iloc[0].copy()
    later["id"] = "later-row"
    later["disasterNumber"] = "4999"
    later["incident_begin_date"] = "2025-08-01"
    later["incident_end_date"] = "2025-08-05"
    result = canonicalize_fema_declarations(pd.concat([rows, later.to_frame().T], ignore_index=True))

    assert len(result) == 2


def test_cross_source_overlap_becomes_one_canonical_incident() -> None:
    canonical_fema = canonicalize_fema_declarations(_fema_rows())
    fema = pd.DataFrame(
        {
            "event_source": "fema",
            "source_event_id": canonical_fema["disasterNumber"].astype(str),
            "fips": canonical_fema["fips"],
            "event_type": canonical_fema["incidentType"],
            "event_name": canonical_fema["declarationTitle"],
            "event_start": canonical_fema["incident_begin_date"],
            "event_end": canonical_fema["incident_end_date"],
            "total_damage_amount": None,
            "source_record_count": canonical_fema["source_record_count"],
            "associated_source_event_ids": canonical_fema["associated_disaster_numbers"],
        }
    )
    noaa = pd.DataFrame(
        [
            {
                "event_source": "noaa",
                "source_event_id": "99001",
                "episode_id": "88001",
                "fips": "13165",
                "event_type": "Hurricane",
                "event_name": "Hurricane",
                "event_start": "2024-09-26",
                "event_end": "2024-09-27",
                "total_damage_amount": 1_500_000_000,
                "source_record_count": 1,
                "associated_source_event_ids": "99001",
            }
        ]
    )

    result = canonicalize_climate_events(fema, noaa)

    assert len(result) == 1
    event = result.iloc[0]
    assert event["source_event_id"] == "4830"
    assert bool(event["has_fema"])
    assert bool(event["has_noaa"])
    assert event["event_sources"] == "fema|noaa"
    assert event["associated_source_event_keys"] == "fema:3616|fema:4830|noaa:99001"
    assert event["semantic_duplicate_count"] == 2
    assert event["canonicalization_reason"] == "cross_source_match"


def test_distinct_noaa_episodes_remain_distinct_without_fema_match() -> None:
    noaa = pd.DataFrame(
        [
            {
                "event_source": "noaa",
                "source_event_id": event_id,
                "episode_id": episode_id,
                "fips": "13165",
                "event_type": "Thunderstorm Wind",
                "event_name": "Thunderstorm Wind",
                "event_start": "2024-06-01",
                "event_end": "2024-06-01",
                "total_damage_amount": 1_100_000_000,
                "source_record_count": 1,
                "associated_source_event_ids": event_id,
            }
            for event_id, episode_id in (("1", "10"), ("2", "20"))
        ]
    )

    result = canonicalize_climate_events(pd.DataFrame(), noaa)

    assert len(result) == 2
    assert set(result["source_event_id"]) == {"1", "2"}
