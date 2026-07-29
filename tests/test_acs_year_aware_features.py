from __future__ import annotations

import unittest

import duckdb

from housing_climate_risk.cli.build_database import (
    _acs_feature_name,
    _acs_feature_selects,
)


class AcsYearAwareFeatureTests(unittest.TestCase):
    def test_education_labels_normalize_to_stable_concepts(self) -> None:
        old_high_school = _acs_feature_name(
            "DP02_0066PE",
            "Percent!!EDUCATIONAL ATTAINMENT!!Percent high school graduate or higher",
            "DP02",
        )
        new_high_school = _acs_feature_name(
            "DP02_0067PE",
            "Percent!!EDUCATIONAL ATTAINMENT!!Population 25 years and over!!High school graduate or higher",
            "DP02",
        )
        self.assertEqual(old_high_school, new_high_school)

    def test_selects_route_reused_codes_by_year(self) -> None:
        con = duckdb.connect()
        try:
            con.execute("CREATE SCHEMA raw")
            con.execute(
                """
                CREATE TABLE raw.census_acs5_county_dp02_test (
                    year VARCHAR,
                    state VARCHAR,
                    county VARCHAR,
                    dp02_0066_pct VARCHAR,
                    dp02_0067_pct VARCHAR,
                    domestic_in_migration_rate VARCHAR
                )
                """
            )
            con.execute(
                """
                INSERT INTO raw.census_acs5_county_dp02_test VALUES
                    ('2018', '08', '001', '87.0', '20.0', '3.1'),
                    ('2019', '08', '001', '7.0', '89.0', '3.2')
                """
            )
            lookup = {
                "DP02_0066PE": {
                    "by_year": {
                        2018: {"feature_name": "high_school_pct"},
                        2019: {"feature_name": "graduate_degree_pct"},
                    }
                },
                "DP02_0067PE": {
                    "by_year": {
                        2018: {"feature_name": "bachelors_pct"},
                        2019: {"feature_name": "high_school_pct"},
                    }
                },
            }

            selects = dict(
                _acs_feature_selects(
                    con,
                    "census_acs5_county_dp02_test",
                    lookup,
                )
            )
            result = con.execute(
                f"""
                SELECT
                    year,
                    {selects["high_school_pct"]} AS high_school_pct,
                    {selects["bachelors_pct"]} AS bachelors_pct,
                    {selects["graduate_degree_pct"]} AS graduate_degree_pct,
                    {selects["domestic_in_migration_rate"]} AS domestic_in_migration_rate
                FROM raw.census_acs5_county_dp02_test AS acs_source
                ORDER BY year
                """
            ).fetchall()

            self.assertEqual(
                result,
                [
                    ("2018", "87.0", "20.0", None, "3.1"),
                    ("2019", "89.0", None, "7.0", "3.2"),
                ],
            )
        finally:
            con.close()


if __name__ == "__main__":
    unittest.main()
