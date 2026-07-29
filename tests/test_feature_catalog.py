from __future__ import annotations

import unittest

from housing_climate_risk.cli.feature_marts import _catalog_rows


class FeatureCatalogTests(unittest.TestCase):
    def test_requested_county_model_features_are_retained(self) -> None:
        rows = {row["feature_name"]: row for row in _catalog_rows()}
        expected = {
            "extreme_event_count",
            "homeowners_insurance_pct_income",
            "property_taxes_pct_income",
            "utilities_pct_income",
            "earnings_by_place_of_work_per_capita_usd",
            "dividends_interest_rent_per_capita_usd",
            "transfer_receipts_per_capita_usd",
        }
        self.assertTrue(all(rows[feature]["retained"] for feature in expected))

    def test_replaced_and_removed_features_are_not_retained(self) -> None:
        rows = {row["feature_name"]: row for row in _catalog_rows()}
        removed = {
            "fema_event_count",
            "noaa_extreme_event_count",
            "risk_rating",
            "risk_score",
            "community_resilience_score",
            "expected_annual_loss_score",
            "social_vulnerability_score",
            "median_home_value_usd",
            "median_gross_rent_usd_month",
            "median_owner_costs_mortgage_usd_month",
            "price_drops_yoy",
            "housing_cost_pct_income",
            "per_capita_personal_income_usd",
            "net_earnings_per_capita_usd",
        }
        self.assertTrue(all(not rows[feature]["retained"] for feature in removed))


if __name__ == "__main__":
    unittest.main()
