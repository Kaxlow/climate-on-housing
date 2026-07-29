from __future__ import annotations

import unittest

import duckdb

from housing_climate_risk.cli.build_database import _create_statsamerica_bea_cew_marts


class StatsAmericaMartTests(unittest.TestCase):
    def test_marts_use_canonical_ibrc_geo_id_for_county_fips(self) -> None:
        con = duckdb.connect()
        try:
            con.execute("CREATE SCHEMA raw")
            con.execute("CREATE SCHEMA mart")
            con.execute(
                """
                CREATE TABLE raw.statsamerica_bea_personal_income (
                    IBRC_GEO_ID VARCHAR,
                    Statefips VARCHAR,
                    Countyfips VARCHAR,
                    Description VARCHAR,
                    Year VARCHAR,
                    Linecode VARCHAR,
                    Data VARCHAR,
                    Disclosure VARCHAR
                )
                """
            )
            con.execute(
                """
                INSERT INTO raw.statsamerica_bea_personal_income
                VALUES ('12077', '20', '077', 'Liberty County, FL', '2023', '0020', '8000', '0')
                """
            )
            con.execute(
                """
                CREATE TABLE raw.statsamerica_cew_total_ownership (
                    IBRC_GEO_ID VARCHAR,
                    Statefips VARCHAR,
                    Countyfips VARCHAR,
                    Description VARCHAR,
                    Year VARCHAR,
                    "Ownership Code" VARCHAR,
                    "NAICS Code" VARCHAR,
                    "NAICS Description" VARCHAR,
                    Units VARCHAR,
                    Employment VARCHAR,
                    Wages VARCHAR,
                    "Average Wage" VARCHAR,
                    "Average Weekly Wage" VARCHAR
                )
                """
            )
            con.execute(
                """
                INSERT INTO raw.statsamerica_cew_total_ownership
                VALUES
                    ('12077', '20', '077', 'Liberty County, FL', '2024',
                     '0', '00', 'Total', '100', '1800', '80000000', '44000', '846'),
                    ('12077', '20', '077', 'Liberty County, FL', '2024',
                     '0', '72', 'Accommodation and Food Services',
                     '20', '300', '9000000', '30000', '577')
                """
            )

            _create_statsamerica_bea_cew_marts(con)

            bea_key = con.execute(
                """
                SELECT fips, state_fips
                FROM mart.statsamerica_bea_personal_income_annual
                """
            ).fetchone()
            cew_keys = con.execute(
                """
                SELECT DISTINCT fips, state_fips
                FROM mart.statsamerica_cew_county_sector_annual
                """
            ).fetchall()

            self.assertEqual(bea_key, ("12077", "12"))
            self.assertEqual(cew_keys, [("12077", "12")])
        finally:
            con.close()


if __name__ == "__main__":
    unittest.main()
