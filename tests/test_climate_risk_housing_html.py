from __future__ import annotations

import re
import unittest

import duckdb
import pandas as pd

from housing_climate_risk.page_data.climate_risk_housing import (
    FEATURE_FOCUS_EVENTS,
    HTML_TEMPLATE,
    RISK_ORDER,
    STATES_PATH,
    _select_story_peer_candidates,
    latest_complete_calendar_window,
)


class ClimateRiskHousingHtmlTests(unittest.TestCase):
    def test_analysis_window_uses_latest_complete_calendar_year(self) -> None:
        con = duckdb.connect()
        try:
            con.execute("CREATE SCHEMA mart")
            con.execute(
                """
                CREATE TABLE mart.redfin_county_monthly (
                    property_type VARCHAR,
                    period_begin DATE
                )
                """
            )
            con.execute(
                """
                INSERT INTO mart.redfin_county_monthly
                SELECT 'All Residential', month
                FROM generate_series(
                    DATE '2025-01-01',
                    DATE '2026-07-01',
                    INTERVAL 1 MONTH
                ) AS dates(month)
                """
            )

            start, end = latest_complete_calendar_window(con)

            self.assertEqual(start, pd.Timestamp("2016-01-01"))
            self.assertEqual(end, pd.Timestamp("2026-01-01"))
        finally:
            con.close()

    def test_nri_link_does_not_break_javascript_string(self) -> None:
        self.assertIn(
            "href='https://www.fema.gov/flood-maps/products-tools/national-risk-index'",
            HTML_TEMPLATE,
        )
        self.assertNotIn(
            'pricingNriPlaceholder: "The <a href="https://',
            HTML_TEMPLATE,
        )

    def test_double_quoted_text_values_do_not_contain_unescaped_href_quotes(self) -> None:
        unsafe_links = re.findall(
            r'^\s+\w+:\s+".*(?<!\\)href="',
            HTML_TEMPLATE,
            flags=re.MULTILINE,
        )
        self.assertEqual(unsafe_links, [])

    def test_story_feature_focus_has_two_counties_per_risk_group(self) -> None:
        self.assertEqual(list(FEATURE_FOCUS_EVENTS), RISK_ORDER)
        for specifications in FEATURE_FOCUS_EVENTS.values():
            self.assertEqual(len(specifications), 2)
            self.assertEqual(len({item["fips"] for item in specifications}), 2)

    def test_state_boundary_shapefile_uses_downloaded_data_workspace(self) -> None:
        expected_parts = ("data", "fipsgeo")
        self.assertTrue(
            all(part in STATES_PATH.parts for part in expected_parts),
            STATES_PATH,
        )
        self.assertEqual(STATES_PATH.suffix, ".shp")

    def test_event_window_arrow_controls_are_removed(self) -> None:
        self.assertNotIn('id="event-arrow-left"', HTML_TEMPLATE)
        self.assertNotIn('id="event-arrow-right"', HTML_TEMPLATE)
        self.assertNotIn("switchEventWindow(", HTML_TEMPLATE)

    def test_rating_sequence_uses_two_second_frames(self) -> None:
        self.assertIn("}, 2000);", HTML_TEMPLATE)

    def test_rating_sequence_separates_visual_and_callout_frames(self) -> None:
        self.assertIn("const RATING_SEQUENCE_FRAMES = [", HTML_TEMPLATE)
        self.assertIn("{risk, callout: false}", HTML_TEMPLATE)
        self.assertIn("{risk, callout: true}", HTML_TEMPLATE)
        self.assertIn('classed("visible", Boolean(callout) && frame.callout)', HTML_TEMPLATE)
        self.assertIn("riskIndex < activeRiskIndex", HTML_TEMPLATE)
        self.assertNotIn("sequence-callout-frame", HTML_TEMPLATE)

    def test_pricing_takeaway_pauses_rating_sequence(self) -> None:
        self.assertIn("function pauseRatingSequence()", HTML_TEMPLATE)
        self.assertIn('section.id === "pricing-grouping"', HTML_TEMPLATE)
        self.assertIn("pauseRatingSequence();", HTML_TEMPLATE)

    def test_event_future_prompt_follows_short_window_takeaway(self) -> None:
        short_step = '{state: "takeaway-short", takeaway: "#event-window-takeaway", eventWindow: "A"}'
        prompt_step = '{state: "takeaway-future", takeaway: "#event-future-prompt", eventWindow: "A"}'
        self.assertIn(short_step, HTML_TEMPLATE)
        self.assertIn(prompt_step, HTML_TEMPLATE)
        self.assertLess(HTML_TEMPLATE.index(short_step), HTML_TEMPLATE.index(prompt_step))

    def test_feature_and_playbook_persistent_layouts_are_present(self) -> None:
        self.assertIn('class="feature-line-pane"', HTML_TEMPLATE)
        self.assertIn('class="feature-detail-stack"', HTML_TEMPLATE)
        self.assertIn('class="playbook-search-shell"', HTML_TEMPLATE)
        self.assertIn('class="playbook-scroll-body"', HTML_TEMPLATE)
        self.assertIn('class="playbook-view-stack"', HTML_TEMPLATE)

    def test_playbook_search_sits_above_its_scroll_body(self) -> None:
        search = HTML_TEMPLATE.index('class="playbook-search-shell"')
        scroll_body = HTML_TEMPLATE.index('class="playbook-scroll-body"')
        self.assertLess(search, scroll_body)
        self.assertIn(
            ".playbook-search-shell { position: relative;",
            HTML_TEMPLATE,
        )
        self.assertIn(
            'document.querySelector("#playbook .playbook-scroll-body")',
            HTML_TEMPLATE,
        )

    def test_story_navigation_and_inner_scroll_locks_are_present(self) -> None:
        self.assertIn('id="story-prev"', HTML_TEMPLATE)
        self.assertIn('id="story-next"', HTML_TEMPLATE)
        self.assertIn("function initStoryEdgeNavigation()", HTML_TEMPLATE)
        self.assertIn('"edge-visible"', HTML_TEMPLATE)
        self.assertIn("function initPanelScrollRouting()", HTML_TEMPLATE)
        self.assertIn('effectiveStep.state.startsWith("takeaway")', HTML_TEMPLATE)
        self.assertIn('classList.add("has-county-selection")', HTML_TEMPLATE)

    def test_feature_importance_hides_percent_labels(self) -> None:
        self.assertNotIn(
            'd3.format(".1%")(feature.relativeImportance || 0)',
            HTML_TEMPLATE,
        )

    def test_both_focus_county_lines_are_solid(self) -> None:
        self.assertIn('.attr("stroke-dasharray", null)', HTML_TEMPLATE)
        self.assertNotIn(
            'd.isFocus && d.focusPosition === "Below" ? "7 4" : null',
            HTML_TEMPLATE,
        )

    def test_playbook_feature_summary_uses_group_standing_labels(self) -> None:
        self.assertIn("'s standing among the ${county.riskRating", HTML_TEMPLATE)
        self.assertIn('class="playbook-scale-labels"><span>Low</span><span>High</span>', HTML_TEMPLATE)

    def test_playbook_reports_insufficient_feature_data(self) -> None:
        self.assertIn("playbookInsufficientFeatureData:", HTML_TEMPLATE)
        self.assertIn("if (!hasFeatureData)", HTML_TEMPLATE)
        self.assertIn('class="playbook-feature-insufficient"', HTML_TEMPLATE)
        self.assertIn("playbookInsufficientFeatureValue", HTML_TEMPLATE)

    def test_story_peer_selection_prefers_iqr_eligible_lines(self) -> None:
        background = pd.DataFrame(
            [
                {
                    "line_id": f"line-{index}",
                    "fips": f"{index:05d}",
                    "pct_rank": percentile,
                    "max_normalized_band_deviation": deviation,
                    "mean_normalized_band_deviation": deviation / 2,
                }
                for index, (percentile, deviation) in enumerate(
                    [(8, 0), (34, 0), (51, 0.4), (72, 0.1), (96, 0.2)]
                )
            ]
        )

        selected = _select_story_peer_candidates(
            background,
            {"line-0", "line-1"},
            count=4,
        )

        self.assertEqual(
            {selected[0]["line_id"], selected[1]["line_id"]},
            {"line-0", "line-1"},
        )
        self.assertEqual(selected[2]["line_id"], "line-3")


if __name__ == "__main__":
    unittest.main()
