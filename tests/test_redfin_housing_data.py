from __future__ import annotations

import io

import duckdb
import pandas as pd

from housing_climate_risk.cli import redfin_housing_data
from housing_climate_risk.cli.build_database import (
    _normalize_place_name,
    _redfin_fips_expr,
    _redfin_normalized_select,
)


class _Response(io.BytesIO):
    def __init__(self, body: bytes) -> None:
        super().__init__(body)
        self.headers = {
            "ETag": '"example-etag"',
            "Last-Modified": "Mon, 17 Aug 2026 11:49:59 GMT",
        }

    def geturl(self) -> str:
        return "https://example.test/resolved.csv"

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        self.close()


def test_download_csv_streams_validated_provider_file(tmp_path, monkeypatch) -> None:
    destination = tmp_path / "redfin.csv"
    monkeypatch.setattr(
        redfin_housing_data,
        "_open_url",
        lambda url: _Response(b'"PERIOD BEGIN","REGION NAME"\n2025-12-01,"Example County, CO"\n'),
    )

    result = redfin_housing_data._download_csv(
        "https://example.test/redfin.csv",
        destination,
        {"PERIOD BEGIN", "REGION NAME"},
        force=True,
    )

    assert result[0] == destination
    assert result[1:] == (
        "https://example.test/resolved.csv",
        '"example-etag"',
        "Mon, 17 Aug 2026 11:49:59 GMT",
        True,
    )
    assert destination.read_text(encoding="utf-8").startswith('"PERIOD BEGIN"')
    assert not destination.with_suffix(".csv.part").exists()


def test_redfin_normalization_converts_percent_units_to_proportions() -> None:
    row = {
        "LAST UPDATED": "2026-08-03",
        "FREQUENCY": "Monthly",
        "PERIOD BEGIN": "2025-12-01",
        "PERIOD END": "2025-12-31",
        "REGION ID": "1",
        "REGION TYPE": "County",
        "REGION NAME": "Example County, CO",
        "METRO": "Example, CO metro area",
        "HOMES SOLD": "100",
        "HOMES SOLD MOM (%)": "10",
        "HOMES SOLD YOY (%)": "20",
        "MEDIAN SALE PRICE NSA ($)": "300000",
        "MEDIAN SALE PRICE NSA MOM (%)": "2",
        "MEDIAN SALE PRICE NSA YOY (%)": "4",
        "MEDIAN DAYS ON MARKET (DAYS)": "30",
        "MEDIAN DAYS ON MARKET MOM (%)": "5",
        "MEDIAN DAYS ON MARKET YOY (%)": "10",
        "AVERAGE SALE TO LIST RATIO (%)": "99",
        "AVERAGE SALE TO LIST RATIO MOM (PPTS)": "1",
        "AVERAGE SALE TO LIST RATIO YOY (PPTS)": "2",
        "SHARE SOLD ABOVE ORIGINAL LIST (%)": "20",
        "SHARE SOLD ABOVE ORIGINAL LIST MOM (PPTS)": "1",
        "SHARE SOLD ABOVE ORIGINAL LIST YOY (PPTS)": "2",
        "NEW LISTINGS": "110",
        "NEW LISTINGS MOM (%)": "1",
        "NEW LISTINGS YOY (%)": "2",
        "ACTIVE LISTINGS": "200",
        "ACTIVE LISTINGS MOM (%)": "3",
        "ACTIVE LISTINGS YOY (%)": "4",
        "INVENTORY": "150",
        "INVENTORY MOM (%)": "5",
        "INVENTORY YOY (%)": "6",
        "PENDING SALES": "90",
        "PENDING SALES MOM (%)": "7",
        "PENDING SALES YOY (%)": "8",
        "MEDIAN NEW LISTING PRICE ($)": "310000",
        "MEDIAN NEW LISTING PRICE MOM (%)": "1",
        "MEDIAN NEW LISTING PRICE YOY (%)": "2",
        "MEDIAN NEW LISTING PRICE PER SQ.FT. ($)": "210",
        "MEDIAN NEW LISTING PRICE PER SQ.FT. MOM (%)": "3",
        "MEDIAN NEW LISTING PRICE PER SQ.FT. YOY (%)": "4",
        "MEDIAN SALE PRICE PER SQ.FT. ($)": "200",
        "MEDIAN SALE PRICE PER SQ.FT. MOM (%)": "5",
        "MEDIAN SALE PRICE PER SQ.FT. YOY (%)": "6",
        "MONTHS OF SUPPLY": "2.5",
        "MONTHS OF SUPPLY MOM (%)": "7",
        "MONTHS OF SUPPLY YOY (%)": "8",
        "PERCENT OFF MARKET IN TWO WEEKS (%)": "30",
        "PERCENT OFF MARKET IN TWO WEEKS MOM (PPTS)": "1",
        "PERCENT OFF MARKET IN TWO WEEKS YOY (PPTS)": "2",
    }
    drops = {
        "PERIOD BEGIN": "2025-12-01",
        "REGION NAME": "Example County, CO",
        "PERCENT ACTIVE WITH PRICE DROPS (%)": "15",
        "PERCENT ACTIVE WITH PRICE DROPS MOM (PPTS)": "1",
        "PERCENT ACTIVE WITH PRICE DROPS YOY (PPTS)": "2",
    }
    con = duckdb.connect()
    try:
        con.register("housing_frame", pd.DataFrame([row]))
        con.register("drops_frame", pd.DataFrame([drops]))
        con.execute("CREATE TABLE housing AS SELECT * FROM housing_frame")
        con.execute("CREATE TABLE drops AS SELECT * FROM drops_frame")
        select_sql = _redfin_normalized_select(
            "h", property_type="'All Residential'", price_drop_alias="d"
        )
        result = con.execute(
            f"""
            SELECT {select_sql}
            FROM housing h
            LEFT JOIN drops d
              ON h."PERIOD BEGIN" = d."PERIOD BEGIN"
             AND h."REGION NAME" = d."REGION NAME"
            """
        ).fetchone()
        columns = [item[0] for item in con.description]
        values = dict(zip(columns, result))
    finally:
        con.close()

    assert values["STATE_CODE"] == "CO"
    assert values["MEDIAN_PPSF_YOY"] == 0.06
    assert values["AVG_SALE_TO_LIST"] == 0.99
    assert values["AVG_SALE_TO_LIST_YOY"] == 0.02
    assert values["MEDIAN_DOM_YOY"] == 0.10
    assert values["PRICE_DROPS"] == 0.15
    assert values["PRICE_DROPS_YOY"] == 0.02


def test_redfin_fips_resolution_prefers_independent_city_name() -> None:
    con = duckdb.connect()
    try:
        con.execute("CREATE SCHEMA ref")
        con.execute(
            """
            CREATE TABLE ref.counties (
                fips VARCHAR, state VARCHAR, county_name VARCHAR
            )
            """
        )
        con.execute(
            """
            INSERT INTO ref.counties VALUES
                ('24005', 'MD', 'Baltimore County'),
                ('24510', 'MD', 'Baltimore City')
            """
        )
        con.create_function(
            "_normalize_place_name", _normalize_place_name, ["VARCHAR"], "VARCHAR"
        )
        expression = _redfin_fips_expr()
        county_fips = con.execute(
            f"SELECT {expression} FROM (SELECT 'Baltimore County, MD' REGION, 'MD' STATE_CODE) raw_redfin"
        ).fetchone()[0]
        city_fips = con.execute(
            f"SELECT {expression} FROM (SELECT 'Baltimore City County, MD' REGION, 'MD' STATE_CODE) raw_redfin"
        ).fetchone()[0]
    finally:
        con.close()

    assert county_fips == "24005"
    assert city_fips == "24510"
