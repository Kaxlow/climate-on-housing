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
    _county_average_event_window_target,
    _select_story_peer_candidates,
    build_state_geojson,
    latest_complete_calendar_window,
)


class ClimateRiskHousingHtmlTests(unittest.TestCase):
    def test_feature_target_is_average_event_window_level_by_county(self) -> None:
        rows = pd.DataFrame(
            [
                {"fips": "01001", "median_ppsf_yoy": 1.0},
                {"fips": "01001", "median_ppsf_yoy": 3.0},
                {"fips": "01001", "median_ppsf_yoy": 8.0},
                {"fips": "01003", "median_ppsf_yoy": -2.0},
                {"fips": "01003", "median_ppsf_yoy": 4.0},
            ]
        )

        target = _county_average_event_window_target(rows).set_index("fips")

        self.assertEqual(target.loc["01001", "event_window_avg_ppsf_yoy"], 4.0)
        self.assertEqual(target.loc["01003", "event_window_avg_ppsf_yoy"], 1.0)
        self.assertEqual(len(target), 2)

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
        self.assertIn('class="playbook-frame-stack"', HTML_TEMPLATE)
        self.assertIn('class="playbook-frame playbook-selected-frame"', HTML_TEMPLATE)
        self.assertIn('class="playbook-selected-layout"', HTML_TEMPLATE)

    def test_playbook_search_precedes_the_selected_county_frame(self) -> None:
        search = HTML_TEMPLATE.index('class="playbook-search-shell"')
        selected = HTML_TEMPLATE.index('class="playbook-frame playbook-selected-frame"')
        self.assertLess(search, selected)
        self.assertIn(
            ".playbook-search-shell { position: relative;",
            HTML_TEMPLATE,
        )
        self.assertNotIn('playbook-scroll-body', HTML_TEMPLATE)

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

    def test_feature_story_uses_directional_bars_and_filtered_scatter(self) -> None:
        self.assertIn('importance-bar ${(metric.rho || 0) < 0 ? "negative" : "positive"}', HTML_TEMPLATE)
        self.assertIn('const [xLow, xHigh] = iqrBounds', HTML_TEMPLATE)
        self.assertIn('.attr("stroke-dasharray", "5 5")', HTML_TEMPLATE)
        self.assertIn('.text(featureLabel(feature))', HTML_TEMPLATE)

    def test_feature_story_has_shared_title_and_selected_cards(self) -> None:
        self.assertIn('id="feature-detail-title"', HTML_TEMPLATE)
        self.assertNotIn('featureStoryIntro:', HTML_TEMPLATE)
        self.assertNotIn('featureStoryDirection:', HTML_TEMPLATE)
        self.assertNotIn('featureSubgroupIntro:', HTML_TEMPLATE)
        self.assertIn('const group = payload.groups.find(d => d.index === selectedFeatureSubgroup)', HTML_TEMPLATE)
        self.assertIn('class", "subgroup-iqr-row"', HTML_TEMPLATE)

    def test_income_feature_labels_identify_measure_and_population(self) -> None:
        expected_labels = [
            "Net Earnings per Resident (Place of Residence)",
            "Dividends, Interest & Rent per Resident",
            "Transfer Receipts per Resident",
            "Home Insurance as % of Median Household Income",
            "Property Tax as % of Median Household Income",
            "Utilities Cost as % of Median Household Income",
        ]
        for label in expected_labels:
            self.assertIn(label, HTML_TEMPLATE)
        self.assertIn("annual BEA amounts per county resident", HTML_TEMPLATE)

    def test_story_titles_lock_after_the_intro_transition(self) -> None:
        self.assertIn('stage.dataset.storyDirection = direction < 0 ? "backward" : "forward"', HTML_TEMPLATE)
        self.assertIn('.story-stage > h2 { transition: top 520ms ease', HTML_TEMPLATE)
        self.assertNotIn('.story-stage.story-step-forward > h2', HTML_TEMPLATE)
        self.assertIn('const previousContent = previousSegment || previousTakeaway', HTML_TEMPLATE)
        self.assertIn('translate: 0 70px', HTML_TEMPLATE)

    def test_text_cards_have_consistent_spacing_and_overlay_rules(self) -> None:
        self.assertIn('.takeaway-section { display: block; padding: 24px 28px; }', HTML_TEMPLATE)
        self.assertNotIn('id="event-window-takeaway" style=', HTML_TEMPLATE)
        self.assertIn('.slide:not(#pricing-grouping)', HTML_TEMPLATE)
        self.assertIn('opacity: .5; filter: none;', HTML_TEMPLATE)

    def test_story_cards_reserve_a_sources_footer(self) -> None:
        self.assertIn('.story-stage > .panel:has(> .sources) { padding-bottom: 78px; }', HTML_TEMPLATE)
        self.assertIn('bottom: 14px; margin: 0; padding: 10px 0 0;', HTML_TEMPLATE)

    def test_feature_subgroup_labels_are_short_and_ordered(self) -> None:
        self.assertIn('subgroupNames: ["Lower", "Lower-Middle", "Upper-Middle", "Upper"]', HTML_TEMPLATE)
        self.assertIn('const orderedGroups = [...payload.groups].sort((a, b) => b.index - a.index)', HTML_TEMPLATE)
        self.assertIn('attr("aria-pressed"', HTML_TEMPLATE)
        self.assertIn('height: min(47svh, 435px)', HTML_TEMPLATE)

    def test_feature_subgroup_legend_has_reliable_toggles_and_no_line_end_label(self) -> None:
        self.assertIn('hideEndLabel: true', HTML_TEMPLATE)
        self.assertIn('id="feature-subgroup-toggles"', HTML_TEMPLATE)
        self.assertIn('button.feature-subgroup-control', HTML_TEMPLATE)
        self.assertIn('selectFeatureSubgroup(Number(d.index))', HTML_TEMPLATE)
        self.assertIn('.feature-subgroup-controls.visible { display: grid; }', HTML_TEMPLATE)
        self.assertIn('risk === "Very High" && label === "Upper-Middle" ? "Middle" : label', HTML_TEMPLATE)

    def test_both_focus_county_lines_are_solid(self) -> None:
        self.assertNotIn(
            'd.isFocus && d.focusPosition === "Below" ? "7 4" : null',
            HTML_TEMPLATE,
        )

    def test_playbook_uses_feature_analysis_and_subgroup_data(self) -> None:
        self.assertIn("function playbookFeatureProfile(county)", HTML_TEMPLATE)
        self.assertIn("subgroupByFips", HTML_TEMPLATE)
        self.assertIn('class="playbook-subgroup-badge"', HTML_TEMPLATE)
        self.assertNotIn("modelCountyProfiles", HTML_TEMPLATE)
        self.assertNotIn("modelTopFeaturesByRisk", HTML_TEMPLATE)

    def test_playbook_reports_insufficient_feature_data(self) -> None:
        self.assertIn("playbookInsufficientFeatureData:", HTML_TEMPLATE)
        self.assertIn("const hasFeatureData = Boolean(", HTML_TEMPLATE)
        self.assertIn('style("display", "none").text("")', HTML_TEMPLATE)
        self.assertIn("Insufficient feature data available for {county}.", HTML_TEMPLATE)
        self.assertIn('class="playbook-feature-insufficient"', HTML_TEMPLATE)

    def test_playbook_subgroup_copy_is_a_county_sentence(self) -> None:
        self.assertIn(
            "With the above features, {county} falls within the {subgroup} range of {risk} Risk counties.",
            HTML_TEMPLATE,
        )

    def test_playbook_event_comparison_is_qualitative_and_handles_volatility(self) -> None:
        self.assertIn("function eventWindowTrendStats(points)", HTML_TEMPLATE)
        self.assertIn("function qualitativeRelation(difference, threshold)", HTML_TEMPLATE)
        self.assertIn("function alignmentExtent(observed, expected, threshold)", HTML_TEMPLATE)
        self.assertIn("eventAlignmentSummary:", HTML_TEMPLATE)
        self.assertIn("eventAlignmentWithoutSubgroup:", HTML_TEMPLATE)
        self.assertIn("riskAlignment", HTML_TEMPLATE)
        self.assertIn("subgroupAlignment", HTML_TEMPLATE)
        self.assertIn("volatileEventSummary:", HTML_TEMPLATE)
        self.assertIn("const tooVolatile = (", HTML_TEMPLATE)
        self.assertNotIn("relativeDirection:", HTML_TEMPLATE)

    def test_map_boundaries_share_a_cohesive_visual_treatment(self) -> None:
        self.assertIn('.county { stroke: #d8e1dd; stroke-width: .34;', HTML_TEMPLATE)
        self.assertIn('.state-boundary { fill: none; stroke: #60756e; stroke-width: .9;', HTML_TEMPLATE)

    def test_state_boundaries_are_dissolved_from_displayed_counties(self) -> None:
        county_geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"fips": "01001"},
                    "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
                },
                {
                    "type": "Feature",
                    "properties": {"fips": "01003"},
                    "geometry": {"type": "Polygon", "coordinates": [[[1, 0], [2, 0], [2, 1], [1, 1], [1, 0]]]},
                },
            ],
        }
        result = build_state_geojson(county_geojson, {"01": ("AL", None)})
        self.assertEqual(len(result["features"]), 1)
        self.assertEqual(result["features"][0]["properties"]["state"], "AL")
        self.assertEqual(result["features"][0]["geometry"]["type"], "Polygon")

    def test_playbook_has_four_frames_and_risk_group_comparison(self) -> None:
        for state in ('"search"', '"profile"', '"history-events"', '"history-compare"'):
            self.assertIn(f"state: {state}", HTML_TEMPLATE)
        self.assertIn("function buildRiskGroupSeries(county)", HTML_TEMPLATE)
        self.assertIn("function drawPlaybookHistory(county, compareRisk = false)", HTML_TEMPLATE)
        self.assertIn("d.q1 - .5 * (d.q3 - d.q1)", HTML_TEMPLATE)
        self.assertIn("d.q3 + .5 * (d.q3 - d.q1)", HTML_TEMPLATE)
        self.assertIn('.duration(900)', HTML_TEMPLATE)
        self.assertIn('id="playbook-back-to-search"', HTML_TEMPLATE)
        self.assertIn('id="playbook-profile-map"', HTML_TEMPLATE)

    def test_pricing_takeaway_keeps_main_card_faded_behind_it(self) -> None:
        self.assertIn(
            '#pricing-grouping .story-stage[data-story-state^="takeaway"] > .panel > *:not(.takeaway) { opacity: .5; filter: none; }',
            HTML_TEMPLATE,
        )

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
