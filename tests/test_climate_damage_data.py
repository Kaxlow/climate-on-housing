from __future__ import annotations

import csv

import pandas as pd

from housing_climate_risk.cli.climate_damage_data import (
    _normalize_county_name,
    _zone_mapping_rows,
    write_existing_manifest,
)


def test_normalize_county_name_removes_county_equivalent_suffixes() -> None:
    assert _normalize_county_name("Lauderdale County") == "LAUDERDALE"
    assert _normalize_county_name("Orleans Parish") == "ORLEANS"
    assert _normalize_county_name("Juneau City and Borough") == "JUNEAU"


def test_zone_mapping_rows_resolves_exact_directional_and_unmapped_names() -> None:
    noaa = pd.DataFrame(
        [
            {
                "state": "ALABAMA",
                "state_fips": "01",
                "cz_type": "Z",
                "cz_fips": 1,
                "cz_name": "LAUDERDALE",
                "total_damage": 10,
            },
            {
                "state": "ALABAMA",
                "state_fips": "01",
                "cz_type": "Z",
                "cz_fips": 2,
                "cz_name": "NORTHERN BALDWIN",
                "total_damage": 20,
            },
            {
                "state": "ALABAMA",
                "state_fips": "01",
                "cz_type": "Z",
                "cz_fips": 3,
                "cz_name": "OPEN WATERS",
                "total_damage": 30,
            },
        ]
    )
    counties = pd.DataFrame(
        [
            {"fips": "01077", "county_name": "Lauderdale County"},
            {"fips": "01003", "county_name": "Baldwin County"},
        ]
    )

    rows = _zone_mapping_rows(noaa, counties)

    assert [row["mapping_method"] for row in rows] == [
        "exact_county_name",
        "directional_or_coastal_prefix_stripped",
        "unmapped",
    ]
    assert rows[0]["mapped_fips"] == "01077"
    assert rows[1]["mapped_fips"] == "01003"
    assert rows[2]["mapped_fips"] == ""


def test_write_existing_manifest_records_urls_and_content_hashes(tmp_path) -> None:
    noaa_path = tmp_path / "noaa_storm_events_county_damage.csv"
    noaa_path.write_text("event_id,total_damage\n1,10\n", encoding="utf-8")
    estimate_path = tmp_path / "climate_damage_download_estimate.csv"
    estimate_path.write_text(
        "source,year,url,estimated_bytes\n"
        "noaa_storm_events,2025,https://example.test/noaa-2025.csv.gz,100\n",
        encoding="utf-8",
    )

    manifest_path = write_existing_manifest(tmp_path)

    with manifest_path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 1
    assert rows[0]["path"] == "noaa_storm_events_county_damage.csv"
    assert rows[0]["url"] == "https://example.test/noaa-2025.csv.gz"
    assert len(rows[0]["sha256"]) == 64
