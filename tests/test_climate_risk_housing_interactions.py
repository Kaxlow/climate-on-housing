import json
import shutil
import subprocess
import unittest

from housing_climate_risk.page_data import climate_risk_housing as page

SCRIPT = (page.WEB_ASSET_DIR / "climate-risk-housing.js").read_text(encoding="utf-8")
CSS = (page.WEB_ASSET_DIR / "climate-risk-housing.css").read_text(encoding="utf-8")


class ChartInteractionTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Node.js required")
    def test_motion_preference_explicit_override(self):
        helper = "function storyMotionReduced()" + SCRIPT.split("function storyMotionReduced()", 1)[1].split("function updateStoryMotionPreference", 1)[0]
        code = "let storyMotionOverride=null;const storyMotionPreference={matches:true};" + helper + """
const results=[storyMotionReduced()];
storyMotionOverride='full';results.push(storyMotionReduced());
storyMotionOverride='reduce';storyMotionPreference.matches=false;results.push(storyMotionReduced());
storyMotionOverride=null;results.push(storyMotionReduced());
console.log(JSON.stringify(results));
"""
        result = subprocess.run(["node", "-e", code], capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout), [True, False, True, False])

    @unittest.skipUnless(shutil.which("node"), "Node.js required")
    def test_nearest_month_and_median_only_or_quartiles(self):
        helpers = "function nearestSeriesMonth" + SCRIPT.split("function nearestSeriesMonth", 1)[1].split("function attachSeriesTooltip", 1)[0]
        script = '''
const fmtPct=v=>(100*v).toFixed(1)+'%';
const d3={utcFormat:()=>d=>d.toISOString().slice(0,7)};
''' + helpers + '''
const rows=[{month:-1,median:.1,q1:.05,q3:.2},{month:0,median:null},{month:1,median:.3,q1:.2,q3:.4}];
const opts={label:'Medium Risk',month:d=>d.month};
console.log(JSON.stringify([
 nearestSeriesMonth(rows,.9,opts.month).month,
 seriesTooltipText(rows[0],opts),
 seriesTooltipText(rows[2],{...opts,quartiles:true}),
 seriesTooltipText(nearestSeriesMonth(rows,0,opts.month),opts),
 seriesTooltipText({date:new Date('2020-03-01'),value:0},{label:'Example County',month:d=>d.date,value:d=>d.value}),
]));
'''
        result = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True)
        month, median, quartiles, gap, county = json.loads(result.stdout)
        self.assertEqual(month, 1)
        self.assertIn("1 month before event start", median)
        self.assertNotIn("quartile", median)
        self.assertIn("Median: 10.0%", median)
        self.assertNotIn("Median PPSF YoY:", median)
        self.assertIn("Upper quartile: 40.0%", quartiles)
        self.assertIn("Lower quartile: 20.0%", quartiles)
        self.assertIsNone(gap)
        self.assertIn("2020-03", county)
        self.assertIn("0.0%", county)

    def test_active_series_and_playbook_county_only(self):
        self.assertIn("if (key === activeRisk && !opts.disableTooltip)", SCRIPT)
        self.assertIn("quartiles: !opts.hideBands", SCRIPT)
        subgroup = SCRIPT.split("function drawFeatureSubgroupLines()", 1)[1].split("function drawFeatureHeatmaps", 1)[0]
        self.assertIn("disableTooltip: true", subgroup)
        self.assertIn("value: d => d.value", subgroup)
        history = SCRIPT.split("function drawPlaybookHistory", 1)[1].split("function fillTextTemplate", 1)[0]
        self.assertEqual(history.count("attachSeriesTooltip("), 1)
        self.assertIn("attachSeriesTooltip(countyLine", history)
        self.assertIn("countyHit.transition(transition)", history)

    def test_mobile_layout_and_native_scrolling(self):
        self.assertIn("@media (max-width: 760px)", CSS)
        self.assertIn("scroll-snap-type: none", CSS)
        self.assertIn("function configureMobileStory", SCRIPT)
        self.assertIn("if (isMobileView()) return;", SCRIPT)
        routing = SCRIPT.split("function initPanelScrollRouting", 1)[1].split("function updateStoryFromScroll", 1)[0]
        self.assertNotIn("preventDefault", routing)
        self.assertIn('lastChartWidth', SCRIPT)

    def test_shared_wheel_and_button_motion(self):
        navigation = SCRIPT.split("function navigateStory(direction)", 1)[1].split("function handleStoryWheel", 1)[0]
        wheel = SCRIPT.split("function handleStoryWheel", 1)[1].split("function initStoryEdgeNavigation", 1)[0]
        self.assertIn("animateStoryScroll(target)", navigation)
        self.assertIn("animateStoryScroll((storyScrollTarget", wheel)
        self.assertIn("isMobileView()", wheel)
        self.assertIn("event.ctrlKey", wheel)
        self.assertIn("node.scrollHeight > node.clientHeight", wheel)
        self.assertIn('storyMotionReduced()', wheel)
        self.assertIn("background: var(--forest); color: white;", CSS)

    def test_count_tooltip_tracks_period(self):
        code = SCRIPT.split("function drawAffectedMap", 1)[1].split("function startRiskTimer", 1)[0]
        self.assertIn('activeEventWindow === "before"', code)
        self.assertIn('activeEventWindow === "B" ? "five" : "three"', code)
        self.assertIn("complete monthly Median PPSF YoY data", code)

    def test_hidden_copy_does_not_intercept_charts(self):
        rule = CSS.split(".story-stage > .section-copy {", 1)[1].split("}", 1)[0]
        self.assertIn("pointer-events: none", rule)
        self.assertIn("visibility: hidden", rule)
        self.assertIn(".axis, .grid { pointer-events: none; }", CSS)

    def test_transition_and_final_frame_footer(self):
        html = (page.WEB_ASSET_DIR / "page.html").read_text(encoding="utf-8")
        self.assertNotIn('[data-story-state$="-intro"]', CSS)
        self.assertIn('[data-story-state="card-before-intro"]', CSS)
        self.assertIn('title.getAnimations().forEach', SCRIPT)
        self.assertIn('duration: 900, easing: "cubic-bezier(.45, 0, .2, 1)"', SCRIPT)
        self.assertIn('scale(${matrix.a})', SCRIPT)
        self.assertIn('function fitStoryTitle(title)', SCRIPT)
        self.assertIn('while (lines() > 2', SCRIPT)
        self.assertIn('width: `${titleBefore.width / scale}px`', SCRIPT)
        self.assertIn('storyNavigationAnimation = requestAnimationFrame(tick)', SCRIPT)
        self.assertNotIn('panel.animate(', SCRIPT)
        self.assertIn('position: absolute !important; top: 110px; bottom: 22px;', CSS)
        self.assertIn('main { padding-bottom: 0; }', CSS)
        self.assertIn('#pricing .story-stage #score-scatter { flex: 1;', CSS)
        self.assertIn('const chartLayoutObserver = new ResizeObserver', SCRIPT)
        self.assertIn('preserveSeries: subgroupFrames.includes(previousState)', SCRIPT)
        self.assertNotIn('width: 34px; height: 3px; margin-bottom: 6px;', CSS)
        self.assertIn('prefers-reduced-motion: reduce', SCRIPT)
        self.assertIn('d3.select(".analytics-privacy").property("hidden"', SCRIPT)
        footer = html.split('<div class="sources playbook-footer">', 1)[1].split("</section>", 1)[0]
        self.assertIn('class="analytics-privacy" hidden', footer)
        self.assertEqual(html.count('class="analytics-privacy"'), 1)

    def test_takeaway_reservation_and_targeted_transitions(self):
        sync = SCRIPT.split("function syncTakeawaySpace", 1)[1].split("function activateStoryTakeaway", 1)[0]
        self.assertNotIn("requestAnimationFrame", sync)
        self.assertIn("takeaway.offsetHeight", sync)
        self.assertIn('renderEventSection({animate: false})', SCRIPT)
        self.assertIn("playbookEntering.filter", SCRIPT)
        self.assertNotIn("panel.animate(", SCRIPT)
        self.assertIn("function syncFeaturePanelHeights()", SCRIPT)
        self.assertIn("#features .feature-story-grid { flex: 1;", CSS)
        observer = SCRIPT.split("const chartLayoutObserver = new ResizeObserver", 1)[1].split("let lastChartWidth", 1)[0]
        self.assertNotIn("requestAnimationFrame", observer)

    def test_event_morph_survives_redundant_resize(self):
        self.assertIn('const displayedLines = new Map()', SCRIPT)
        self.assertIn('svg.selectAll("*").interrupt()', SCRIPT)
        self.assertIn('displayedLines.get(key)', SCRIPT)
        self.assertIn('xAxis.transition(transition).call(axis.scale(x))', SCRIPT)
        self.assertIn('entry.target instanceof SVGSVGElement', SCRIPT)
        self.assertIn('entry.target.viewBox.baseVal', SCRIPT)

    def test_feature_title_inclusive_layout_and_aligned_significance_rows(self):
        self.assertIn('--feature-heading-height', SCRIPT)
        self.assertIn('grid-template-rows: subgrid', CSS)
        self.assertIn('grid-row: span var(--strong-count)', CSS)
        self.assertIn('.feature-frame[data-frame="3"] { inset: 0 !important; }', CSS)
        self.assertIn('top: -88px', CSS)
        self.assertIn('top: -48px', CSS)
        self.assertIn('grid-auto-rows: minmax(22px, max-content)', CSS)
        self.assertIn('position: static !important; width: auto !important; flex-shrink: 0; margin: 3mm 0 0 !important', CSS)
        self.assertIn('.importance-group-label:first-child { margin-top: 0; }', CSS)
        self.assertIn('font-size: 13px; line-height: 1.35', CSS)
        self.assertIn('font-size: clamp(26px, 2.3vw, 34px)', CSS)

    def test_event_caption_lane_stays_reserved_between_steps(self):
        self.assertIn('--event-caption-height', CSS)
        self.assertIn('TEXT.eventWindowATakeaway, TEXT.eventWindowBTakeaway', SCRIPT)
        self.assertIn('panel.style.setProperty("--event-caption-height"', SCRIPT)
        self.assertIn('#events #t-events-card-title { min-height: 30px; }', CSS)

    def test_feature_columns_do_not_measure_animated_rows_for_height(self):
        sync = SCRIPT.split('function syncFeaturePanelHeights()', 1)[1].split('function drawFeatureHeatmaps', 1)[0]
        self.assertNotIn('lastRow', sync)
        self.assertNotIn('grid.style.maxHeight =', sync)
        self.assertIn('grid.style.removeProperty("max-height")', sync)

    def test_mobile_swipe_story_and_persistent_county_resize(self):
        self.assertIn("window.addEventListener('touchend'", SCRIPT)
        self.assertIn("Math.abs(dy) > 45", SCRIPT)
        self.assertIn('const stride = pages ? extra / pages : 0', SCRIPT)
        self.assertNotIn('Step ${position + 1}', SCRIPT)
        self.assertIn('const batchSize = 120;', SCRIPT)
        self.assertIn('while (seriesIndex < series.length) drawCountyBatch()', SCRIPT)
        self.assertIn('Swipe up for more', CSS)

    def test_mobile_map_gestures_and_animation_defaults(self):
        self.assertIn('function enableMapGestures', SCRIPT)
        self.assertIn(".style('touch-action', 'none')", SCRIPT)
        self.assertIn('.info-tooltip, [data-map-gestures]', SCRIPT)
        self.assertNotIn('if (isMobileView() && reset)', SCRIPT)
        self.assertNotIn('if (isMobileView() && automatic)', SCRIPT)
        self.assertNotIn('if (automatic && isMobileView())', SCRIPT)
        self.assertIn("const retainMobileBottom = isMobileView() && effectiveStep.state.startsWith('takeaway')", SCRIPT)
        self.assertIn('if (lockInnerScroll && panel && !retainMobileBottom)', SCRIPT)
        self.assertIn('stage.dataset.storyIndex === String(index) && isMobileView()', SCRIPT)
        self.assertIn('padding: 3px 4px 3px 17px !important', CSS)
        self.assertIn('.mobile-story .playbook-frame-stack { flex: 0 0 auto !important;', CSS)
        self.assertIn('.feature-footnote-tabs { position: static !important; order: 99;', CSS)
        self.assertIn('.mobile-story .risk-legend-button { padding-left: 18px; }', CSS)

    def test_cursors_feedback_and_axis_labels(self):
        self.assertNotIn("cursor: crosshair", CSS)
        self.assertIn('options.onClick ? "pointer" : "default"', SCRIPT)
        self.assertIn("if (options.onClick) options.onClick(event)", SCRIPT)
        self.assertIn('state !== storyConfigForSection("playbook").at(-1).state', SCRIPT)
        self.assertNotIn('.text("Monthly Median PPSF YoY")', SCRIPT)
        self.assertNotIn('if (opts.yAxisLabel)', SCRIPT)
        self.assertIn(".analytics-privacy { font-size: 11px; }", CSS)


if __name__ == "__main__":
    unittest.main()
