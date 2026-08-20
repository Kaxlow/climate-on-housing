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
        self.assertIn('.sequence-callout { position: relative;', HTML_TEMPLATE)
        self.assertIn('margin: 4px 24px 0 58px;', HTML_TEMPLATE)

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

    def test_feature_story_uses_uniform_bars_direction_markers_and_filtered_scatter(self) -> None:
        self.assertIn('correlation-marker ${(metric.rho || 0) < 0 ? "negative" : "positive"}', HTML_TEMPLATE)
        self.assertIn('class", "importance-bar"', HTML_TEMPLATE)
        self.assertIn('const [xLow, xHigh] = iqrBounds', HTML_TEMPLATE)
        self.assertIn('.attr("stroke-dasharray", "5 5")', HTML_TEMPLATE)
        self.assertIn('.text(featureLabel(feature))', HTML_TEMPLATE)

    def test_feature_story_has_shared_title_and_selected_cards(self) -> None:
        self.assertIn('id="feature-detail-title"', HTML_TEMPLATE)
        self.assertNotIn('featureStoryIntro:', HTML_TEMPLATE)
        self.assertNotIn('featureStoryDirection:', HTML_TEMPLATE)
        self.assertNotIn('featureSubgroupIntro:', HTML_TEMPLATE)
        self.assertIn('const group = payload.groups.find(d => d.index === selectedFeatureSubgroup)', HTML_TEMPLATE)
        self.assertIn('id="feature-distribution-chart"', HTML_TEMPLATE)
        self.assertIn('drawFeatureSubgroupPanel()', HTML_TEMPLATE)

    def test_feature_story_adds_shared_subgroup_summary_frame(self) -> None:
        self.assertIn('class="feature-frame" data-frame="3"', HTML_TEMPLATE)
        self.assertIn('{state: "feature-frame-3"}', HTML_TEMPLATE)
        self.assertIn('function subgroupFeatureRelations(risk, subgroup)', HTML_TEMPLATE)
        self.assertIn('function drawFeatureSubgroupSummary()', HTML_TEMPLATE)
        self.assertIn('higher: "above average"', HTML_TEMPLATE)
        self.assertIn('lower: "below average"', HTML_TEMPLATE)
        self.assertIn('close: "average"', HTML_TEMPLATE)
        self.assertIn('state === "feature-frame-2" || state === "feature-frame-3"', HTML_TEMPLATE)

    def test_feature_subgroup_summary_scrolls_only_its_rows(self) -> None:
        self.assertIn('.feature-subgroup-summary { display: flex; flex-direction: column;', HTML_TEMPLATE)
        self.assertIn('height: 100%; min-height: 0; overflow: hidden;', HTML_TEMPLATE)
        self.assertIn('.feature-subgroup-summary-rows { display: grid;', HTML_TEMPLATE)
        self.assertIn('const rowScroller = summary.append("div").attr("class", "feature-subgroup-summary-rows");', HTML_TEMPLATE)

    def test_subgroup_toggles_activate_on_pointer_down_without_selecting_text(self) -> None:
        self.assertIn('.feature-subgroup-control-label { display: block; pointer-events: none; user-select: none; }', HTML_TEMPLATE)
        self.assertIn('.on("pointerdown", (event, d) => {', HTML_TEMPLATE)
        self.assertIn('if (event.detail !== 0) return;', HTML_TEMPLATE)

    def test_feature_subgroup_lines_have_additional_upper_domain_margin(self) -> None:
        self.assertIn('opts.upperDomainPadding || 0', HTML_TEMPLATE)
        self.assertIn('upperDomainPadding: 0.16', HTML_TEMPLATE)

    def test_intro_and_feature_context_use_inline_tooltips(self) -> None:
        self.assertNotIn('id="t-scatter-fn1"', HTML_TEMPLATE)
        self.assertNotIn('id="t-scatter-fn2"', HTML_TEMPLATE)
        self.assertIn('scatterFootnotesTooltip:', HTML_TEMPLATE)
        self.assertIn('id=\\"feature-performance-term\\"', HTML_TEMPLATE)
        self.assertIn('featurePerformanceTooltip:', HTML_TEMPLATE)

    def test_feature_threshold_negative_state_and_distribution_filter(self) -> None:
        self.assertIn('(metric.absRho || 0) >= 0.3', HTML_TEMPLATE)
        self.assertIn('negative-active', HTML_TEMPLATE)
        self.assertIn('scatter-negative', HTML_TEMPLATE)
        self.assertIn('value >= lowerBound && value <= upperBound', HTML_TEMPLATE)
        self.assertIn('featureDistributionOutlierTooltip:', HTML_TEMPLATE)
        self.assertIn('const removeOutliers = selectedFeatureRisk !== "Very High";', HTML_TEMPLATE)
        self.assertIn('featureDistributionVeryHighTooltip:', HTML_TEMPLATE)

    def test_feature_importance_and_distribution_controls_match_latest_design(self) -> None:
        self.assertIn('class", "importance-strong-group"', HTML_TEMPLATE)
        self.assertNotIn('featureOutcomeTopic:', HTML_TEMPLATE)
        self.assertIn('featureDistributionPrevious:', HTML_TEMPLATE)
        self.assertIn('featureDistributionNext:', HTML_TEMPLATE)
        self.assertNotIn('id="feature-distribution-group-label"', HTML_TEMPLATE)
        self.assertIn('rotateFeature = direction =>', HTML_TEMPLATE)

    def test_information_tooltips_are_persistent_and_clickable(self) -> None:
        self.assertIn('.tooltip.persistent { pointer-events: auto; }', HTML_TEMPLATE)
        self.assertIn('activeInfoTooltipTrigger = element', HTML_TEMPLATE)
        self.assertIn('document.addEventListener("pointerdown"', HTML_TEMPLATE)
        self.assertNotIn('element.addEventListener("pointerleave"', HTML_TEMPLATE)

    def test_latest_section_layout_and_distribution_interactions(self) -> None:
        self.assertIn('class="rating-line-pane"', HTML_TEMPLATE)
        self.assertIn('event-horizon-number', HTML_TEMPLATE)
        self.assertIn('eventHorizonYears:', HTML_TEMPLATE)
        self.assertIn('.feature-frame[data-frame="1"] { display: flex; flex-direction: column; overflow: hidden; }', HTML_TEMPLATE)
        self.assertIn('featureDistributionTitle: "County Distribution"', HTML_TEMPLATE)
        self.assertIn('if (options.length > 1) controls.append("button")', HTML_TEMPLATE)
        self.assertIn('countyDisplayName(county) || d.fips', HTML_TEMPLATE)
        self.assertIn('--takeaway-bottom: 52px', HTML_TEMPLATE)

    def test_tooltip_position_is_clamped_to_the_viewport(self) -> None:
        self.assertIn('function showTooltip(event, content', HTML_TEMPLATE)
        self.assertIn('window.innerWidth - width - edge', HTML_TEMPLATE)
        self.assertIn('window.innerHeight - height - edge', HTML_TEMPLATE)

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
        self.assertIn("Economic and demographic features use ten-year county averages.", HTML_TEMPLATE)

    def test_story_titles_lock_after_the_intro_transition(self) -> None:
        self.assertIn('stage.dataset.storyDirection = direction < 0 ? "backward" : "forward"', HTML_TEMPLATE)
        self.assertIn('.story-stage > h2 { transition: top 520ms ease', HTML_TEMPLATE)
        self.assertNotIn('.story-stage.story-step-forward > h2', HTML_TEMPLATE)
        self.assertIn('const previousContent = previousSegment || previousTakeaway', HTML_TEMPLATE)
        self.assertIn('translate: 0 70px', HTML_TEMPLATE)

    def test_text_cards_have_consistent_spacing_and_overlay_rules(self) -> None:
        self.assertIn('.takeaway-section { display: block; padding: 24px 28px; }', HTML_TEMPLATE)
        self.assertNotIn('id="event-window-takeaway" style=', HTML_TEMPLATE)
        self.assertIn('--takeaway-space: min(17svh, 132px)', HTML_TEMPLATE)
        self.assertIn('> .panel > *:not(.takeaway) { opacity: 1; filter: none; }', HTML_TEMPLATE)

    def test_story_cards_reserve_a_sources_footer(self) -> None:
        self.assertIn('.story-stage > .panel:has(> .sources) { padding-bottom: 78px; }', HTML_TEMPLATE)
        self.assertIn('bottom: 14px; margin: 0; padding: 10px 0 0;', HTML_TEMPLATE)

    def test_feature_subgroup_labels_are_performance_based_and_ordered(self) -> None:
        self.assertIn('subgroupNamesFour: ["Strong Overperformers", "Mild Overperformers", "Mild Underperformers", "Strong Underperformers"]', HTML_TEMPLATE)
        self.assertIn('subgroupNamesThree: ["Overperformers", "Average Performers", "Underperformers"]', HTML_TEMPLATE)
        self.assertIn('const orderedGroups = [...payload.groups].sort((a, b) => a.index - b.index)', HTML_TEMPLATE)
        self.assertIn('attr("aria-pressed"', HTML_TEMPLATE)
        self.assertIn('startFeatureSubgroupSequence()', HTML_TEMPLATE)

    def test_feature_subgroup_legend_has_reliable_toggles_and_no_line_end_label(self) -> None:
        self.assertIn('hideEndLabel: true', HTML_TEMPLATE)
        self.assertIn('id="feature-subgroup-toggles"', HTML_TEMPLATE)
        self.assertIn('button.feature-subgroup-control', HTML_TEMPLATE)
        self.assertIn('selectFeatureSubgroup(Number(d.index), true)', HTML_TEMPLATE)
        self.assertIn('.feature-subgroup-controls.visible { display: grid; }', HTML_TEMPLATE)
        self.assertIn('class="feature-plot-shell"', HTML_TEMPLATE)
        self.assertIn('grid-template-columns: minmax(0, 1fr) 146px;', HTML_TEMPLATE)
        self.assertIn('cursor: pointer !important;', HTML_TEMPLATE)
        self.assertIn('pointer-events: auto; touch-action: manipulation;', HTML_TEMPLATE)
        self.assertIn('cursor: pointer; pointer-events: none;', HTML_TEMPLATE)
        self.assertIn('marginRight: 24', HTML_TEMPLATE)
        self.assertIn('const names = count === 3 ? TEXT.subgroupNamesThree : TEXT.subgroupNamesFour', HTML_TEMPLATE)

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
        self.assertIn("playbookInsufficientEventWindowData:", HTML_TEMPLATE)
        self.assertIn("const hasFeatureData = summarizeSubgroup", HTML_TEMPLATE)
        self.assertIn("const insufficientCopy = !profile.row", HTML_TEMPLATE)
        self.assertIn('style("display", "none").text("")', HTML_TEMPLATE)
        self.assertIn("Insufficient feature data available for {county}.", HTML_TEMPLATE)
        self.assertIn(
            "could not be determined because there were insufficient housing data to form a complete event window for analysis.",
            HTML_TEMPLATE,
        )
        self.assertIn('class="playbook-feature-insufficient"', HTML_TEMPLATE)

    def test_playbook_subgroup_copy_is_a_county_sentence(self) -> None:
        self.assertIn(
            "{county}'s house price growth rate around extreme climate events makes it a {subgroup} among {risk} Risk counties.",
            HTML_TEMPLATE,
        )
        self.assertIn("function playbookPerformanceName(label)", HTML_TEMPLATE)
        self.assertIn("subgroup: playbookPerformanceName(profile.subgroupName)", HTML_TEMPLATE)

    def test_playbook_uses_analysis_significance_rule_and_scrolls_only_feature_values(self) -> None:
        self.assertIn("function mostImportantFeatureMetrics(risk)", HTML_TEMPLATE)
        self.assertIn("const metrics = mostImportantFeatureMetrics(county.riskRating);", HTML_TEMPLATE)
        self.assertIn(".playbook-feature-summary { display: grid; gap: 6px; max-height:", HTML_TEMPLATE)
        self.assertIn("overflow-y: auto;", HTML_TEMPLATE)

    def test_later_playbook_frames_use_the_subgroup_feature_summary(self) -> None:
        self.assertIn("function renderPlaybookFeatureSummary(county, summarizeSubgroup = false)", HTML_TEMPLATE)
        self.assertIn('state === "history-events" || state === "history-compare"', HTML_TEMPLATE)
        self.assertIn('playbookSubgroupFeatureTitle:', HTML_TEMPLATE)
        self.assertIn('const subgroupRelations = subgroupFeatureRelations(county.riskRating, profile.subgroup);', HTML_TEMPLATE)

    def test_playbook_history_profile_card_has_fixed_chrome_and_scrollable_traits(self) -> None:
        self.assertIn('playbookSubgroupFeatureTitle: "County Traits"', HTML_TEMPLATE)
        self.assertIn('.playbook-back-button { align-self: flex-start;', HTML_TEMPLATE)
        self.assertIn('padding: 5px 8px; font-size: 10px;', HTML_TEMPLATE)
        self.assertIn('[data-story-state^="history-"] #playbook-selected-county-name { display: none !important; }', HTML_TEMPLATE)
        self.assertIn('[data-story-state^="history-"] .playbook-feature-summary { flex: 1 1 auto; min-height: 0; max-height: none; }', HTML_TEMPLATE)
        self.assertIn('[data-story-state^="history-"] .playbook-subgroup-badge { flex: 0 0 auto; margin: auto 0 0; }', HTML_TEMPLATE)

    def test_playbook_county_history_line_is_distinct_from_risk_colors(self) -> None:
        self.assertIn('const COUNTY_LINE_COLOR = "#2456a6";', HTML_TEMPLATE)
        self.assertNotIn('playbook-county-line-halo', HTML_TEMPLATE)
        self.assertIn('attr("stroke", COUNTY_LINE_COLOR).attr("stroke-width", 2.4)', HTML_TEMPLATE)
        self.assertIn('{label: TEXT.playbookSeriesLegend, color: COUNTY_LINE_COLOR, opacity: 1, line: true}', HTML_TEMPLATE)

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
        self.assertIn('.county { stroke: #ffffff; stroke-width: .45;', HTML_TEMPLATE)
        self.assertIn('.state-boundary { fill: none; stroke: #173f37; stroke-width: 1.4;', HTML_TEMPLATE)

    def test_playbook_event_list_scrolls_within_frame_three(self) -> None:
        self.assertIn('.playbook-events-pane { display: flex; flex-direction: column; }', HTML_TEMPLATE)
        self.assertIn('overflow-y: auto; overscroll-behavior: contain; scrollbar-gutter: stable;', HTML_TEMPLATE)

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
        result = build_state_geojson(
            {"01001", "01003"},
            {"01": ("AL", None)},
            county_geojson,
        )
        self.assertEqual(len(result["features"]), 1)
        self.assertEqual(result["features"][0]["properties"]["state"], "AL")
        self.assertEqual(result["features"][0]["geometry"]["type"], "Polygon")
        self.assertEqual(len(result["features"][0]["geometry"]["coordinates"]), 1)

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

    def test_takeaway_keeps_main_card_fully_visible_behind_it(self) -> None:
        self.assertNotIn('opacity: .5; filter: none;', HTML_TEMPLATE)
        self.assertIn('padding-bottom: calc(var(--takeaway-space) + var(--takeaway-footnote-space) + 12px) !important;', HTML_TEMPLATE)
        self.assertIn('function syncTakeawaySpace(section, takeaway)', HTML_TEMPLATE)
        self.assertIn('if (previousTakeaway === nextTakeaway && previousContent !== nextContent)', HTML_TEMPLATE)
        self.assertIn('max-height: none;', HTML_TEMPLATE)

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
