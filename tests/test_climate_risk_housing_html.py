from __future__ import annotations

import re
import inspect
import unittest
import subprocess
import json
import shutil
from unittest.mock import patch

import duckdb
import pandas as pd

from housing_climate_risk.page_data import climate_risk_housing as page_builder
from housing_climate_risk.page_data.climate_risk_housing import (
    FEATURE_FOCUS_EVENTS,
    HTML_TEMPLATE,
    RISK_ORDER,
    STATE_AND_DC_FIPS,
    STATES_PATH,
    _county_average_event_window_target,
    _select_story_peer_candidates,
    assign_performer_subgroup_from_ranges,
    build_state_geojson,
    filter_current_state_county_events,
    latest_complete_calendar_window,
)


# Assemble the source assets for existing markup/behavior unit tests. Production
# serves them separately, and bundle tests below verify that separation.
HTML_TEMPLATE = HTML_TEMPLATE.replace(
    '<link rel="stylesheet" href="climate-risk-housing.css">',
    '<style>' + (page_builder.WEB_ASSET_DIR / 'climate-risk-housing.css').read_text(encoding='utf-8') + '</style>',
).replace(
    '<script src="climate-risk-housing.js"></script>',
    '<script>' + (page_builder.WEB_ASSET_DIR / 'climate-risk-housing.js').read_text(encoding='utf-8') + '</script>',
)


class ClimateRiskHousingHtmlTests(unittest.TestCase):
    def test_feedback_link_is_public_and_at_end_of_page(self):
        self.assertIn('https://docs.google.com/forms/d/e/1FAIpQLSeLPeN1D-bAfJLxSRhsbeAQwMIf3DBNm-nuqyVe30ZasESyxA/viewform', HTML_TEMPLATE)
        self.assertIn('pageFeedback: "t-page-feedback"', HTML_TEMPLATE)
        self.assertIn('<div class="sources playbook-footer"><div id="t-playbook-sources"></div><div class="page-feedback" id="t-page-feedback"></div></div>', HTML_TEMPLATE)
        self.assertIn('.sources:not(.playbook-footer), #t-playbook-sources', HTML_TEMPLATE)
        self.assertIn('background: #e5e5e5; color: #000;', HTML_TEMPLATE)
        self.assertNotIn('Takes about 2 minutes.', HTML_TEMPLATE)

    @unittest.skipUnless(shutil.which("node"), "Node.js is needed for heading layout tests")
    def test_playbook_conclusion_fits_one_line_and_has_two_plus_icons(self):
        function = "function fitPlaybookConclusionTitle" + HTML_TEMPLATE.split("function fitPlaybookConclusionTitle", 1)[1].split("function renderPlaybookOutlook", 1)[0]
        script = '''
const title={clientWidth:700,scrollWidth:600,style:{}};
const value={clientWidth:100,scrollWidth:200,style:{}};
const document={querySelector:()=>title,querySelectorAll:()=>[value]};
''' + function + '''
fitPlaybookConclusionTitle();const wide=title.style.fontSize;
title.clientWidth=300;fitPlaybookConclusionTitle();
console.log(JSON.stringify([wide,title.style.fontSize,value.style.fontSize]));
'''
        result = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout), ["20px", "10px", "8px"])
        self.assertEqual(HTML_TEMPLATE.count('class="playbook-scorecard-plus"'), 2)

    def test_feature_payload_reuses_correlation_target_for_subgroups(self):
        from types import SimpleNamespace

        fips = ["01001", "01003", "01005", "01007"]
        features = [item[2] for item in page_builder.WITHIN_GROUP_FEATURES]
        frames = {
            "feature.county_economic_annual": pd.DataFrame({"fips": fips, **{key: [1., 2., 3., 4.] for key in features[:8]}}),
            "feature.county_demographic_annual": pd.DataFrame({"fips": fips, **{key: [1., 2., 3., 4.] for key in features[8:]}}),
            "feature.county_risk": pd.DataFrame({"fips": fips, "risk_rating": ["Relatively Low"] * 4}),
            "mart.redfin_county_monthly": pd.DataFrame({"fips": fips, "county": fips, "state": ["AL"] * 4, "historical_average_ppsf_yoy": [1.] * 4}),
        }

        class Connection:
            def execute(self, sql, *_):
                return SimpleNamespace(df=lambda: next(frame.copy() for table, frame in frames.items() if f"FROM {table}" in sql))

        months = list(range(-12, 37))
        trajectories = [
            (fips[0], "a", [0.] + [20.] * 48),
            (fips[0], "b", [2.] * 49),
            (fips[1], "c", [5.] * 49),
            (fips[2], "d", [7.] * 49),
            (fips[3], "e", [1.] * 49),
        ]
        rows = [{"fips": county, "line_id": event, "event_key": event, "event_window_month": month, "median_ppsf_yoy": value}
                for county, event, values in trajectories for month, value in zip(months, values)]
        context = SimpleNamespace(analysis_start=pd.Timestamp("2016-01-01"), analysis_end=pd.Timestamp("2026-01-01"), affected=pd.DataFrame(rows))
        with patch.object(page_builder, "current_county_fips", return_value=frozenset(fips)), patch.object(page_builder, "_bootstrap_spearman_ci", return_value=(float("nan"), float("nan"))):
            payload = page_builder.build_feature_payload(Connection(), event_context=context)
        correlation_targets = {row["fips"]: row["target"] for row in payload["countyRowsByRisk"]["Low"]}
        subgroup_targets = {row["fips"]: row["target"] for row in payload["countyRowsByRisk"]["Low"]}
        self.assertEqual(correlation_targets, subgroup_targets)
        self.assertEqual(subgroup_targets[fips[0]], 11.)
        self.assertEqual(payload["subgroupByFips"], {fips[0]: 0, fips[2]: 1, fips[1]: 2, fips[3]: 3})
        self.assertEqual(payload["playbookSubgroupByFips"], payload["subgroupByFips"])
        strongest = payload["subgroupsByRisk"]["Low"]["groups"][0]
        self.assertEqual((strongest["targetMin"], strongest["targetMedian"], strongest["targetMax"]), (11., 11., 11.))
        self.assertEqual(strongest["values"][0]["value"], 1.)
        self.assertEqual(strongest["values"][1]["value"], 11.)

    @unittest.skipUnless(shutil.which("node"), "Node.js is needed for rendering tests")
    def test_playbook_insufficient_history_replaces_performance_text(self):
        function = "function renderPlaybookPerformanceTakeaway" + HTML_TEMPLATE.split("function renderPlaybookPerformanceTakeaway", 1)[1].split("function renderPlaybookFeatureSummary", 1)[0]
        script = '''
let output='';
const TEXT={playbookInsufficientHistory:'Insufficient housing data available for selected county to comment on local housing market performance.'};
const d3={select:()=>({text:s=>{output=s}})};
const playbookHasSufficientHistory=()=>false;
const playbookFeatureProfile=()=>{throw Error('Must not display a subgroup with insufficient history')};
''' + function + "renderPlaybookPerformanceTakeaway({fips:'test'});console.log(JSON.stringify(output));"
        result = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout), "Insufficient housing data available for selected county to comment on local housing market performance.")

    @unittest.skipUnless(shutil.which("node"), "Node.js is needed for coverage tests")
    def test_playbook_half_history_threshold_and_last_frame(self):
        functions = "function playbookHasSufficientHistory" + HTML_TEMPLATE.split("function playbookHasSufficientHistory", 1)[1].split("function syncPlaybookStoryLength", 1)[0]
        script = '''
const DATA={playbook:{monthlyHistoryMonths:Array(120).fill('month'),monthlyHistoryValuesByFips:{}}};
const STORY_CONFIG={playbook:[{state:'history-compare'},{state:'history-outlook'}]};
let selectedCountyFips='test';
const playbookCountyByFips=new Map([['test',{fips:'test'}]]);
const playbookFeatureProfile=()=>({subgroupName:'Strong Overperformers'});
''' + functions + '''
console.log(JSON.stringify([0,59,60,61,120].map(count=>{
DATA.playbook.monthlyHistoryValuesByFips.test=Array(120).fill(null).map((v,i)=>i<count?0:v);
return [playbookHasSufficientHistory({fips:'test'}),storyConfigForSection('playbook').at(-1).state];
})));
'''
        result = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout), [
            [False, "history-compare"], [False, "history-compare"],
            [True, "history-outlook"], [True, "history-outlook"], [True, "history-outlook"],
        ])

    @unittest.skipUnless(shutil.which("node"), "Node.js is needed for arrow tests")
    def test_playbook_factor_arrows_combine_county_position_and_correlation(self):
        function = "function playbookFactorArrows" + HTML_TEMPLATE.split("function playbookFactorArrows", 1)[1].split("function renderPlaybookOutlook", 1)[0]
        script = "const playbookCountyFeaturePosition=c=>c.position;" + function + "console.log(JSON.stringify(['Higher','Lower','At peer median','Data unavailable'].map(position=>[.5,-.5,0].map(rho=>playbookFactorArrows({position},{rho,feature:'x'})))));"
        result = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True)
        pairs = [[(r["factorUp"], r["growthUp"]) for r in row] for row in json.loads(result.stdout)]
        self.assertEqual(pairs, [
            [(True, True), (True, False), (True, False)],
            [(False, False), (False, True), (False, False)],
            [(False, False), (False, True), (False, False)],
            [(None, None), (None, None), (None, None)],
        ])

    @unittest.skipUnless(shutil.which("node"), "Node.js is needed for sizing tests")
    def test_scatter_height_fills_available_space_without_repeat_redraw(self):
        function = "function fitFeatureScatterToTakeaway" + HTML_TEMPLATE.split("function fitFeatureScatterToTakeaway", 1)[1].split("function drawFeatureScatter", 1)[0]
        script = '''
let height=220,draws=0,active=true;
const selectedFeatureKey='income';
const svg={getBoundingClientRect:()=>({top:150,height})};
const relationship={getBoundingClientRect:()=>({top:570})};
const stage={querySelector:s=>s.includes('scatter-active')?svg:relationship,style:{setProperty:(key,value)=>{height=parseInt(value)}}};
const document={querySelector:()=>active?stage:null};
const drawFeatureScatter=()=>{draws++};
''' + function + '''
fitFeatureScatterToTakeaway();fitFeatureScatterToTakeaway();active=false;fitFeatureScatterToTakeaway();
console.log(JSON.stringify([height,draws]));
'''
        result = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout), [414, 1])

    @unittest.skipUnless(shutil.which("node"), "Node.js is needed for score tests")
    def test_scorecard_range_matching_and_arrow_scores(self):
        functions = "function playbookFactorAssessment" + HTML_TEMPLATE.split("function playbookFactorAssessment", 1)[1].split("function renderPlaybookOutlook", 1)[0]
        script = '''
const rows=[{fips:'o1',values:{x:2}},{fips:'o2',values:{x:8}},{fips:'u1',values:{x:6}},{fips:'u2',values:{x:12}}];
const RISK_COLORS={Low:'#11794f',High:'#be302e'};
const playbookCountyFeaturePosition=(county,feature)=>feature==='missing'?'Data unavailable':DATA.features.allCountyRowsByRisk.Low.at(-1).values.x<7?'Lower':'Higher';
const DATA={features:{countyRowsByRisk:{Low:rows},allCountyRowsByRisk:{Low:[...rows,{fips:'test',values:{x:7}}]},subgroupsByRisk:{Low:{groups:[{index:0},{index:3}]}},subgroupByFips:{o1:0,o2:0,u1:3,u2:3}}};
const d3={ascending:(a,b)=>a-b,median:a=>{const v=[...a].sort((a,b)=>a-b);return (v[Math.floor((v.length-1)/2)]+v[Math.floor(v.length/2)])/2}};
const subgroupName=g=>g.index===0?'Strong Overperformers':'Strong Underperformers';
const playbookPerformanceDisplayName=n=>n;
const playbookEvents=c=>c.events||[];
const countyDisplayName=()=> 'Test County';
const featureLabel=f=>f;
''' + functions + '''
const county={fips:'test',riskRating:'Low'};
const testRow=DATA.features.allCountyRowsByRisk.Low.at(-1);
const matches=[3,6,7,8,14,-2,null].map(x=>{testRow.values.x=x;return playbookFeatureSubgroupMatch(county,'x')});
testRow.values.x=3;
const labels=[];
for(const risk of ['Low','High']) for(const subgroup of ['Strong Overperformers','Strong Underperformers']) {
const html=playbookScorecard(county,{subgroupName:subgroup,assignmentSource:'event-window'},[{feature:'x',rho:-.5}],risk);
labels.push(html.match(/score-risk-[^"]+">([^<]+)/)[1]);
}
testRow.values.x=7;
labels.push(playbookScorecard(county,{subgroupName:'Strong Underperformers'},[{feature:'x',rho:-.5}],'High').match(/score-risk-[^"]+">([^<]+)/)[1]);
labels.push(playbookScorecard(county,{subgroupName:'Strong Overperformers'},[{feature:'missing',rho:.5}],'Low').match(/score-risk-[^"]+">([^<]+)/)[1]);
console.log(JSON.stringify({matches,labels}));
'''
        result = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True)
        result = json.loads(result.stdout)
        self.assertEqual(result["matches"], [True, True, False, False, False, True, None])
        self.assertEqual(result["labels"], ["Low Risk", "Moderate Risk", "Moderate Risk", "High Risk", "High Risk", "Insufficient data"])

    @unittest.skipUnless(shutil.which("node"), "Node.js is needed for story tests")
    def test_rapid_scroll_keeps_only_one_takeaway_visible(self):
        function = "function activateStoryTakeaway" + HTML_TEMPLATE.split("function activateStoryTakeaway", 1)[1].split("function applyStoryStep", 1)[0]
        script = '''
const element=(classes=[])=>({classList:{values:new Set(classes),add(...v){v.forEach(x=>this.values.add(x))},remove(...v){v.forEach(x=>this.values.delete(x))},contains(x){return this.values.has(x)}},offsetWidth:100});
const a=element(),b=element(['segmented']),segments=[element(),element()];
b.querySelectorAll=()=>segments;
const section={querySelectorAll:()=>[a,b,...segments],querySelector:s=>s==='#a'?a:b};
const takeawayTransitionTimers=new WeakMap();
const syncTakeawaySpace=()=>{};
''' + function + '''
for(let i=0;i<1000;i++) {
 const step=[{takeaway:'#a'},{takeaway:'#b',segment:0},{takeaway:'#b',segment:1},{}][i%4];
 activateStoryTakeaway(section,step,i%2?1:-1);
 if([a,b].filter(e=>e.classList.contains('story-active-takeaway')).length!==(step.takeaway?1:0)) throw Error('overlap');
 if(segments.filter(e=>e.classList.contains('story-active-segment')).length!==(step.takeaway==='#b'?1:0)) throw Error('segment overlap');
 if([a,b,...segments].some(e=>e.classList.contains('story-outgoing-takeaway'))) throw Error('stale outgoing card');
}
'''
        subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True)

    @unittest.skipUnless(shutil.which("node"), "Node.js is needed for score tests")
    def test_factor_directions_and_balanced_score(self):
        functions = "function playbookFactorAssessment" + HTML_TEMPLATE.split("function playbookFactorAssessment", 1)[1].split("function playbookFeatureSubgroupMatch", 1)[0]
        script = '''
const playbookCountyFeaturePosition=(c,f)=>f==='tie'?'At peer median':f==='missing'?'Data unavailable':'Higher';
const featureLabel=f=>f,countyDisplayName=()=> 'Test County',playbookEvents=()=>[];
const playbookPerformanceDisplayName=n=>n,RISK_COLORS={Low:'green'};
''' + functions + '''
const county={riskRating:'Low'};
const factors=[{feature:'positive',rho:.5},{feature:'negative',rho:-.5},{feature:'tie',rho:.5},{feature:'missing',rho:.5}];
const result=factors.map(f=>playbookFactorAssessment(county,f));
const html=playbookScorecard(county,{subgroupName:'Mild Overperformer'},factors.slice(0,2),'Low');
console.log(JSON.stringify([result.map(x=>x.growth),result[1].description.includes('associated with lower Median PPSF YoY'),html.includes('aria-label="Balanced"'),html.includes('Moderate Risk'),html.includes('https://getquoll.com/')]));
'''
        result = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout), [[1, -1, 0, None], True, True, True, True])

    @unittest.skipUnless(shutil.which("node"), "Node.js is needed for feature tests")
    def test_factor_context_matches_county_direction(self):
        context = HTML_TEMPLATE.split("  playbookFactorContext:", 1)[1].split("  // ---- Events section ----", 1)[0]
        function = "function playbookFactorAssessment" + HTML_TEMPLATE.split("function playbookFactorAssessment", 1)[1].split("function playbookScorecard", 1)[0]
        script = "const TEXT={playbookFactorContext:" + context + "};" + '''
const playbookCountyFeaturePosition=c=>c.position;
const countyDisplayName=()=> 'Example County';
const featureLabel=f=>f;
''' + function + '''
const metric={feature:'net_earnings_per_capita_usd',rho:-.5};
const high=playbookFactorAssessment({position:'Higher',riskRating:'Low'},metric).description;
const low=playbookFactorAssessment({position:'Lower',riskRating:'Low'},metric).description;
const reverse=playbookFactorAssessment({position:'Higher',riskRating:'Low'},{...metric,rho:.5}).description;
console.log(JSON.stringify([high.includes('less affordable'),low.includes('more affordable'),reverse.includes('opposite to the usual explanation')]));
'''
        result = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout), [True, True, True])

    @unittest.skipUnless(shutil.which("node"), "Node.js is needed for feature tests")
    def test_factor_percentages_use_selected_subgroup_and_complete_cohort(self):
        function = "function subgroupFeatureRelations" + HTML_TEMPLATE.split("function subgroupFeatureRelations", 1)[1].split("function drawFeatureSubgroupSummary", 1)[0]
        script = '''
const DATA={features:{countyRowsByRisk:{Low:[{fips:'a',values:{x:1}},{fips:'b',values:{x:3}},{fips:'c',values:{x:5}},{fips:'d',values:{x:7}},{fips:'e',values:{x:null}}]},allCountyRowsByRisk:{Low:[{values:{x:1000}}]},subgroupByFips:{a:3,b:3,c:0,d:0,e:0}}};
DATA.features.subgroupsByRisk={Low:{groups:[{index:0},{index:3}]}};
const d3={ascending:(a,b)=>a-b,median:a=>(a[1]+a[2])/2};
const subgroupName=g=>g.index===0?'Strong Overperformers':'Strong Underperformers';
let rho=.5;
const mostImportantFeatureMetrics=()=>[{feature:'x',rho}];
''' + function + '''
const result=[0,3].map(index=>subgroupFeatureRelations('Low',{index,traits:[{feature:'x',median:4}]})[0]);
rho=-.5;result.push(subgroupFeatureRelations('Low',{index:0,traits:[{feature:'x',median:4}]})[0]);
console.log(JSON.stringify(result.map(r=>[r.peerMedian,r.matching,r.denominator,r.percentage])));
'''
        result = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout), [[4, 2, 2, 100], [4, 2, 2, 100], [4, 0, 2, 0]])

    @unittest.skipUnless(shutil.which("node"), "Node.js is needed for story tests")
    def test_question_cards_and_event_context_precede_their_charts(self):
        config = "const STORY_CONFIG =" + HTML_TEMPLATE.split("const STORY_CONFIG =", 1)[1].split("function playbookHasPerformanceGroup", 1)[0]
        result = subprocess.run(["node", "-e", config + "console.log(JSON.stringify(STORY_CONFIG));"], capture_output=True, text=True, check=True)
        story = json.loads(result.stdout)
        pricing = story["pricing-grouping"]
        self.assertEqual(pricing[-1]["takeaway"], "#pricing-question")
        events = story["events"]
        states = [step["state"] for step in events]
        self.assertEqual(states.index("card-before"), states.index("card-before-intro") + 1)
        self.assertEqual(states.index("card-short"), states.index("card-after-intro") + 1)
        self.assertEqual(states.index("takeaway-after-question"), states.index("takeaway-before") + 1)
        self.assertEqual(states.index("card-after-intro"), states.index("takeaway-after-question") + 1)
        self.assertEqual(events[-1]["takeaway"], "#event-variation-question")
        self.assertNotIn("segment", events[-1])
        for target in ["pricing-question", "event-overview-question", "event-future-prompt", "event-variation-question"]:
            self.assertIn(f'class="takeaway question-card" id="{target}"', HTML_TEMPLATE)

    @unittest.skipUnless(shutil.which("node"), "Node.js is needed for scorecard tests")
    def test_scorecard_uses_all_county_peers_and_distinguishes_fallback(self):
        functions = "function playbookCountyFeaturePosition" + HTML_TEMPLATE.split("function playbookCountyFeaturePosition", 1)[1].split("function renderPlaybookOutlook", 1)[0]
        script = '''
const DATA={features:{allCountyRowsByRisk:{Low:[{fips:'a',values:{x:1}},{fips:'b',values:{x:3}},{fips:'c',values:{x:5}}]},countyRowsByRisk:{Low:[]},subgroupsByRisk:{Low:{groups:[]}}}};
const d3={median:a=>a.slice().sort((a,b)=>a-b)[Math.floor(a.length/2)]};
const TEXT={playbookScorecardNote:'Not a forecast'};
const RISK_COLORS={Low:'#11794f'};
const playbookEvents=c=>c.events;
const featureLabel=f=>f;
const countyDisplayName=()=> 'Example County';
const playbookPerformanceDisplayName=n=>n;
''' + functions + '''
const a={fips:'a',riskRating:'Low',events:[]};
const observed=playbookScorecard({...a,events:[{}]},{subgroupName:'Strong',assignmentSource:'event-window'},[{feature:'x'}],'Low');
const fallback=playbookScorecard(a,{subgroupName:'Weak',assignmentSource:'ten-year-median-quartiles'},[{feature:'x'}],'Low');
console.log(JSON.stringify([playbookCountyFeaturePosition(a,'x'),playbookCountyFeaturePosition({...a,fips:'c'},'x'),playbookCountyFeaturePosition(a,'missing'),observed.includes('performance when'),fallback.includes('performance if'),fallback.includes('Historical-median fallback'),observed.includes('Observed complete-event-window subgroup')]));
'''
        result = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout), ["Lower", "Higher", "Data unavailable", True, True, False, False])

    @unittest.skipUnless(shutil.which("node"), "Node.js is needed for feature summary tests")
    def test_feature_summary_uses_only_two_peer_categories(self):
        function = "function subgroupFeatureRelations" + HTML_TEMPLATE.split("function subgroupFeatureRelations", 1)[1].split("function drawFeatureSubgroupSummary", 1)[0]
        script = '''
const DATA={features:{subgroupByFips:{},countyRowsByRisk:{Low:[{values:{x:1}},{values:{x:3}},{values:{x:5}}]},subgroupsByRisk:{Low:{groups:[{index:0},{index:3}]}}}};
const d3={ascending:(a,b)=>a-b,quantileSorted:values=>values[1],median:values=>values[1]};
const subgroupName=g=>g.index===0?'Strong Overperformers':'Strong Underperformers';
let rho = 0;
const mostImportantFeatureMetrics=()=>[{feature:'x',rho}];
''' + function + "console.log(JSON.stringify([0,3].map(index=>[-.5,0,.5].map(r=>{rho=r;return subgroupFeatureRelations('Low',{index,traits:[{feature:'x',median:3}]} )[0].relation;}))));"
        result = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout), [["lower", "lower", "higher"], ["higher", "higher", "lower"]])

    def test_significant_factor_selection_retains_threshold_matches_and_tops_up(self):
        cases = [
            ([.2, -.1, .05, .01], [0, 1, 2]),
            ([.5, .2, -.1, .01], [0, 1, 2]),
            ([.5, -.3, .2, .01], [0, 1, 2]),
            ([.5, -.4, .3, .2], [0, 1, 2]),
            ([.5, -.4, .3, -.35, .2], [0, 1, 3, 2]),
            ([None, .2, -.2], [1, 2]),
            ([None, None], []),
        ]
        for values, expected in cases:
            metrics = [{"feature": str(i), "rho": v, "absRho": abs(v) if v is not None else None} for i, v in enumerate(values)]
            selected = page_builder._select_significant_feature_metrics(metrics)
            self.assertEqual([int(m["feature"]) for m in selected], expected)
            if shutil.which("node"):
                function = "function mostImportantFeatureMetrics" + HTML_TEMPLATE.split("function mostImportantFeatureMetrics", 1)[1].split("function subgroupFeatureRelations", 1)[0]
                script = "const DATA={features:{importanceByRisk:{Low:" + json.dumps(metrics) + "}}};" + function + "console.log(JSON.stringify(mostImportantFeatureMetrics('Low').map(m=>Number(m.feature))));"
                result = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True)
                self.assertEqual(json.loads(result.stdout), expected)
        self.assertEqual(page_builder._select_significant_feature_metrics([{"rho": float("nan")}]), [])

    def test_correlation_target_collapses_events_before_summarizing_months(self):
        rows = pd.DataFrame({
            "fips": ["01001"] * 6 + ["01003"] * 3,
            "event_window_month": [-12, 0, 36] * 3,
            "median_ppsf_yoy": [0., 100., 100., 10., 10., 10., 2., 4., 6.],
        })
        result = page_builder._county_median_trajectory_target(rows).set_index("fips")
        column = page_builder.FEATURE_PERFORMANCE_TARGET_COLUMN
        # Monthly medians [5,55,55] -> 55, unlike pooled median 10 or mean 38.33.
        self.assertEqual(result.loc["01001", column], 55.)
        self.assertEqual(result.loc["01003", column], 4.)
        self.assertTrue(page_builder._county_median_trajectory_target(rows.iloc[:0]).empty)

    @unittest.skipUnless(shutil.which("node"), "Node.js is needed for rendering tests")
    def test_warning_intro_limits_event_claim_to_observed_assignments(self):
        function = "function renderPlaybookPerformanceTakeaway" + HTML_TEMPLATE.split("function renderPlaybookPerformanceTakeaway", 1)[1].split("function renderPlaybookFeatureSummary", 1)[0]
        script = '''
let output='',tooltip='';
const d3={select:()=>({html:s=>{output=s},text:s=>{output=s},node:()=>({querySelector:()=>({append:t=>{tooltip=t}})})})};
const makeInfoButton=s=>s;
const TEXT={playbookHistoryComparisonUnavailable:'Unavailable'};
const playbookFeatureProfile=c=>c.profile;
const playbookHasSufficientHistory=()=>true;
const countyDisplayName=()=> 'Example County';
const playbookPerformanceDisplayName=n=>n;
''' + function + '''
console.log(JSON.stringify(['event-window','ten-year-median-quartiles',null].map(source=>{
tooltip='';renderPlaybookPerformanceTakeaway({riskRating:'Low',profile:{assignmentSource:source,subgroupName:source?'Mild Overperformers':null}}); return [output,tooltip];
})));
'''
        result = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True)
        observed, fallback, unavailable = json.loads(result.stdout)
        self.assertIn("event window", observed[1])
        self.assertIn("past ten years", fallback[1])
        self.assertNotIn("Based on", observed[0])
        self.assertIn("<strong>Mild Overperformers</strong>", observed[0])
        self.assertEqual(unavailable, ["Unavailable", ""])

    @unittest.skipUnless(shutil.which("node"), "Node.js is needed for rendering tests")
    def test_relative_position_ranges_and_category_icons(self):
        position = "function playbookRelativePosition" + HTML_TEMPLATE.split("function playbookRelativePosition", 1)[1].split("function renderPlaybookPerformanceTakeaway", 1)[0]
        icon = "function playbookFeatureCategoryIcon" + HTML_TEMPLATE.split("function playbookFeatureCategoryIcon", 1)[1].split("function renderPlaybookOutlook", 1)[0]
        script = '''
const DATA = {features:{featureMeta:{income:{category:'Economic'}, age:{category:'Demographic'}}}};
''' + position + icon + '''
console.log(JSON.stringify([
 [10,11,12,9,8].map(a => playbookRelativePosition(a,10,14,6)),
 playbookRelativePosition(null,10,14,6),
 playbookRelativePosition(10,10,10,10),
 playbookFeatureCategoryIcon('income').includes('aria-label="Economic feature"'),
 playbookFeatureCategoryIcon('age').includes('aria-label="Demographic feature"'),
 playbookFeatureCategoryIcon('missing')
]));
'''
        result = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout), [
            ["mid range", "mid range", "upper range", "mid range", "lower range"],
            None, "mid range", True, True, "",
        ])

    @unittest.skipUnless(shutil.which("node"), "Node.js is needed for rendering tests")
    def test_outlook_separates_copy_from_dashboard_and_clears_stale_copy(self):
        function = "function renderPlaybookOutlook" + HTML_TEMPLATE.split("function renderPlaybookOutlook", 1)[1].split("function selectPlaybookCounty", 1)[0]
        script = '''
const nodes = {};
const d3 = {select: id => nodes[id] ||= {content:'', hidden:false,
 attr(){return this;}, property(key,value){this[key]=value; return this;},
 html(value){this.content=value; return this;}, text(value){this.content=value; return this;},
 selectAll(){return {each(){}};}}};
const RISK_ORDER = ['Low'];
const TEXT = {playbookTopFactorsTitle:'Top factors',playbookFactorAssociation:'Associated with growth',playbookFactorContext:{},playbookWarningIntroWithEvents:'With events',playbookWarningIntroNoEvents:'No events',playbookWarningTakeaway:'Takeaway',playbookOutlookInsufficientRisk:'Missing risk',playbookOutlookInsufficientFeatures:'Missing features'};
const playbookScorecard = () => 'Scorecard';
const playbookFeatureProfile = () => ({subgroupName:'Strong Overperformers'});
const playbookHasSufficientHistory=()=>true;
const playbookFactorArrows = () => ({factorUp:true,growthUp:false});
const playbookFactorAssessment = (county,metric) => ({...metric,factorUp:metric.feature==='AtMedian'?null:true,growth:metric.feature==='AtMedian'?0:metric.rho>0?1:-1,positionLabel:metric.feature==='AtMedian'?'At median':'Above median'});
const requestAnimationFrame=()=>{};
const fitPlaybookConclusionTitle=()=>{};
const mostImportantFeatureMetrics = () => [{feature:'Income',rho:0.4},{feature:'Insurance',rho:-0.4},{feature:'AtMedian',rho:0.4}];
const featureLabel = value => value;
const playbookFeatureCategoryIcon = () => '<svg aria-label="Economic feature"></svg>';
const playbookEvents = county => county.events;
const fillTextTemplate = value => value;
const countyDisplayName = () => 'Example County';
const playbookPerformanceName = value => value;
''' + function + '''
renderPlaybookOutlook({riskRating:'Low',events:[]});
const result = [nodes['#playbook-warning-intro'].content, nodes['#playbook-warning-takeaway'].content,
 nodes['#playbook-event-commentary'].content.includes('Income'),
 nodes['#playbook-event-commentary'].content.includes('Takeaway'), nodes['#playbook-warning-intro'].hidden,
 nodes['#playbook-event-commentary'].content.includes('AtMedian')];
renderPlaybookOutlook({riskRating:'Low',events:[{}]});
result.push(nodes['#playbook-warning-intro'].content);
renderPlaybookOutlook({riskRating:null,events:[]});
result.push(nodes['#playbook-warning-intro'].hidden, nodes['#playbook-warning-takeaway'].hidden, nodes['#playbook-warning-intro'].content);
console.log(JSON.stringify(result));
'''
        result = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout), ["Top factors", "Scorecard", True, False, False, False, "Top factors", True, True, ""])

    def test_playbook_frame_heading_and_fixed_conclusion_layout(self):
        self.assertIn('min-height: 0; gap: 8px; overflow-y: auto; overscroll-behavior: contain;', HTML_TEMPLATE)
        self.assertIn('.playbook-factor-column h4 { flex-shrink: 0;', HTML_TEMPLATE)
        self.assertIn('font-size: 16px; flex: 1; margin: 0; display: flex; align-items: center; justify-content: center; text-align: center;', HTML_TEMPLATE)
        self.assertEqual(HTML_TEMPLATE.count('id="playbook-warning-intro"'), 1)
        self.assertLess(HTML_TEMPLATE.index('id="playbook-warning-intro"'), HTML_TEMPLATE.index('<div class="playbook-selected-layout">'))
        self.assertIn('.playbook-score-result a { font-weight: 800; }', HTML_TEMPLATE)
        self.assertIn('grid-template-rows: minmax(0, 1fr) auto; padding: 0;', HTML_TEMPLATE)
        self.assertNotIn('id="playbook-frame-title"', HTML_TEMPLATE)
        self.assertIn('playbookH2: "Climate Playbook: {frameTitle}"', HTML_TEMPLATE)
        self.assertIn("d3.select('#t-playbook-h2').text(fillTextTemplate", HTML_TEMPLATE)
        self.assertIn('grid-template-rows: auto minmax(0, 1fr) auto; gap: clamp(8px, 1.3vh, 16px); overflow: hidden;', HTML_TEMPLATE)
        self.assertIn('background: #f9e8e7; border-color: #dfb1ae;', HTML_TEMPLATE)
        self.assertIn('grid-template-columns: minmax(0, 1fr) auto auto;', HTML_TEMPLATE)
        self.assertIn('class="feature-percentage-value"', HTML_TEMPLATE)
        self.assertIn('have values <strong>${d.relation', HTML_TEMPLATE)
        self.assertIn('.html(d => playbookFeatureCategoryIcon(d.feature))', HTML_TEMPLATE)
        self.assertIn('grid-auto-rows: minmax(min-content, 1fr)', HTML_TEMPLATE)
        self.assertIn('position: static; flex: 0 0 auto; margin-top: 5px;', HTML_TEMPLATE)

    @unittest.skipUnless(shutil.which("node"), "Node.js is needed for browser calculation tests")
    def test_playbook_statistics_omit_null_months_and_unknown_risk(self):
        function = "function playbookHistoricalStatistics" + HTML_TEMPLATE.split("function playbookHistoricalStatistics", 1)[1].split("function playbookFeatureProfile", 1)[0]
        script = '''
const RISK_ORDER = ['Low'];
const d3 = {median: values => {const v = values.slice().sort((a,b) => a-b); return v.length ? (v[Math.floor((v.length-1)/2)] + v[Math.floor(v.length/2)])/2 : undefined;}};
const playbookHistoryRows = () => [{value: 0}, {value: null}, {value: 8}];
const buildRiskGroupSeries = () => [{median: 1,q3: 10,q1: -10}, {median: 100,q3: 200,q1: 0}, {median: 3,q3: 20,q1: -2}, {median: null,q3: null,q1: null}];
''' + function + "console.log(JSON.stringify([playbookHistoricalStatistics({riskRating:'Low'}), playbookHistoricalStatistics({riskRating:null})]));"
        result = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout), [
            {"a": 4, "b": 3, "c": 20, "d": -2},
            {"a": None, "b": None, "c": None, "d": None},
        ])

    @unittest.skipUnless(shutil.which("node"), "Node.js is needed for browser calculation tests")
    def test_playbook_uses_observed_groups_only_for_event_counties(self):
        function = "function playbookFeatureProfile" + HTML_TEMPLATE.split("function playbookFeatureProfile", 1)[1].split("function renderPlaybookPerformanceStatus", 1)[0]
        assignment = "function historicalPerformanceAssignment" + HTML_TEMPLATE.split("function historicalPerformanceAssignment", 1)[1].split("function playbookHistoricalStatistics", 1)[0]
        script = '''
const DATA = {features: {countyRowsByRisk: {Low: []}, subgroupByFips: {observed: 0, noevents: 0}, subgroupsByRisk: {Low: {groups: [{index: 0}]}}}};
const mostImportantFeatureMetrics = () => [];
const playbookEvents = county => county.fips === 'noevents' ? [] : [{}];
const playbookHistoricalStatistics = () => ({a: 8, b: 10, c: 14, d: 6});
const subgroupName = () => 'Strong Overperformers';
''' + assignment + function + '''
console.log(JSON.stringify(['observed','incomplete','noevents'].map(fips => {
const profile = playbookFeatureProfile({fips, riskRating: 'Low'});
return [profile.subgroupName, profile.assignmentSource];
})));
'''
        result = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout), [
            ["Strong Overperformers", "event-window"],
            ["Strong Underperformers", "ten-year-median-quartiles"],
            ["Strong Underperformers", "ten-year-median-quartiles"],
        ])

    @unittest.skipUnless(shutil.which("node"), "Node.js is needed for browser calculation tests")
    def test_historical_quartile_fallback_boundaries(self):
        function = HTML_TEMPLATE.split("function historicalPerformanceAssignment", 1)[1].split("function playbookHistoricalStatistics", 1)[0]
        script = "function historicalPerformanceAssignment" + function + "\nconsole.log(JSON.stringify([" + ",".join(
            "historicalPerformanceAssignment(" + arguments + ")" for arguments in [
                "11,10,14,6", "12,10,14,6", "9,10,14,6", "8,10,14,6",
                "10,10,14,6", "11,10,10,10", "9,10,10,10", "null,10,14,6",
            ]) + "]));"
        result = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout), [
            "Mild Overperformers", "Strong Overperformers", "Mild Underperformers",
            "Strong Underperformers", "Mild Overperformers", "Strong Overperformers",
            "Strong Underperformers", None,
        ])

    def test_shared_performance_target_uses_monthly_medians_not_pooled_median(self):
        rows = pd.DataFrame({
            "fips": ["01001"] * 6 + ["01003"] * 3,
            "event_key": ["a"] * 3 + ["b"] * 3 + ["c"] * 3,
            "event_window_month": [-12, 0, 36] * 3,
            "median_ppsf_yoy": [0., 20., 20., 2., 2., 2., 5., 5., 5.],
        })
        column = page_builder.FEATURE_PERFORMANCE_TARGET_COLUMN
        median = page_builder._county_median_trajectory_target(rows).set_index("fips")
        pooled = rows.groupby("fips")["median_ppsf_yoy"].median()
        self.assertEqual(median.loc["01001", column], 11.)
        self.assertEqual(median.loc["01003", column], 5.)
        self.assertEqual(median[column].idxmax(), "01001")
        self.assertEqual(pooled.idxmax(), "01003")
        empty = page_builder._county_median_trajectory_target(rows.iloc[:0])
        self.assertEqual(list(empty.columns), ["fips", column])
        self.assertTrue(empty.empty)

    @patch(
        "housing_climate_risk.page_data.climate_risk_housing.current_county_fips",
        return_value=frozenset({"06037", "11001", "72001", "09003"}),
    )
    def test_event_analysis_keeps_only_current_state_and_dc_counties(
        self,
        _mock_current_county_fips,
    ) -> None:
        events = pd.DataFrame(
            {
                "fips": ["06037", "11001", "06000", "72001", "09001", "99137"],
                "event_key": ["ca", "dc", "aggregate", "pr", "legacy", "special"],
            }
        )
        current_nri_fips = {"06037", "11001", "72001", "09003"}

        result = filter_current_state_county_events(events, current_nri_fips)

        self.assertEqual(result["fips"].tolist(), ["06037", "11001"])
        self.assertIn("11", STATE_AND_DC_FIPS)
        self.assertNotIn("72", STATE_AND_DC_FIPS)

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

    def test_analysis_window_advances_when_a_new_complete_year_is_available(
        self,
    ) -> None:
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
                    DATE '2026-12-01',
                    INTERVAL 1 MONTH
                ) AS dates(month)
                """
            )

            start, end = latest_complete_calendar_window(con)

            self.assertEqual(start, pd.Timestamp("2017-01-01"))
            self.assertEqual(end, pd.Timestamp("2027-01-01"))
        finally:
            con.close()

    def test_page_main_shares_one_maximum_event_context(self) -> None:
        source = inspect.getsource(page_builder.main)
        self.assertEqual(source.count("build_max_affected_event_context(con)"), 1)
        self.assertIn("build_feature_payload(con, event_context=event_context)", source)
        self.assertIn("build_event_windows(con, event_context=event_context)", source)
        context_source = inspect.getsource(
            page_builder.build_max_affected_event_context
        )
        self.assertIn("post_event_months=60", context_source)
        self.assertNotIn(
            "build_affected_event_windows",
            inspect.getsource(page_builder.build_feature_payload),
        )
        self.assertNotIn(
            "build_affected_event_windows",
            inspect.getsource(page_builder.build_event_windows),
        )

    def test_nri_link_does_not_break_javascript_string(self) -> None:
        self.assertIn(
            "href='https://www.fema.gov/flood-maps/products-tools/national-risk-index'",
            HTML_TEMPLATE,
        )
        self.assertNotIn(
            'pricingNriPlaceholder: "The <a href="https://',
            HTML_TEMPLATE,
        )

    def test_double_quoted_text_values_do_not_contain_unescaped_href_quotes(
        self,
    ) -> None:
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

    def test_rating_and_event_sequences_share_faster_frames(self) -> None:
        self.assertIn("const RISK_SEQUENCE_INTERVAL = 1200;", HTML_TEMPLATE)
        self.assertGreaterEqual(HTML_TEMPLATE.count("RISK_SEQUENCE_INTERVAL"), 3)

    def test_rating_sequence_uses_clickable_legend_without_callouts(self) -> None:
        self.assertIn(
            "const RATING_SEQUENCE_FRAMES = RISK_ORDER.map(risk => ({risk}));",
            HTML_TEMPLATE,
        )
        self.assertIn('id="rating-risk-legend"', HTML_TEMPLATE)
        self.assertIn('id="rating-play-button"', HTML_TEMPLATE)
        self.assertNotIn("ratingSequenceCallouts", HTML_TEMPLATE)
        self.assertNotIn('id="rating-sequence-callout"', HTML_TEMPLATE)
        self.assertIn("riskIndex < activeRiskIndex", HTML_TEMPLATE)

    def test_pricing_takeaway_pauses_rating_sequence(self) -> None:
        self.assertIn("function pauseRatingSequence(manual = false)", HTML_TEMPLATE)
        self.assertIn('section.id === "pricing-grouping"', HTML_TEMPLATE)
        self.assertIn("pauseRatingSequence();", HTML_TEMPLATE)

    def test_event_future_prompt_follows_short_window_takeaway(self) -> None:
        short_step = '{state: "takeaway-short", takeaway: "#event-window-takeaway", eventWindow: "A"}'
        prompt_step = '{state: "takeaway-future", takeaway: "#event-future-prompt", eventWindow: "A"}'
        self.assertIn(short_step, HTML_TEMPLATE)
        self.assertIn(prompt_step, HTML_TEMPLATE)
        self.assertLess(
            HTML_TEMPLATE.index(short_step), HTML_TEMPLATE.index(prompt_step)
        )

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
        self.assertNotIn("playbook-scroll-body", HTML_TEMPLATE)

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

    def test_feature_story_uses_uniform_bars_direction_markers_and_filtered_scatter(
        self,
    ) -> None:
        self.assertIn(
            'correlation-marker ${(metric.rho || 0) < 0 ? "negative" : "positive"}',
            HTML_TEMPLATE,
        )
        self.assertIn('class", "importance-bar"', HTML_TEMPLATE)
        self.assertIn("const [xLow, xHigh] = iqrBounds", HTML_TEMPLATE)
        self.assertIn('.attr("stroke-dasharray", "5 5")', HTML_TEMPLATE)
        self.assertIn(".text(featureLabel(feature))", HTML_TEMPLATE)

    def test_feature_story_has_shared_title_and_selected_cards(self) -> None:
        self.assertIn('id="feature-detail-title"', HTML_TEMPLATE)
        self.assertNotIn("featureStoryIntro:", HTML_TEMPLATE)
        self.assertNotIn("featureStoryDirection:", HTML_TEMPLATE)
        self.assertNotIn("featureSubgroupIntro:", HTML_TEMPLATE)
        self.assertIn(
            "const group = payload.groups.find(d => d.index === selectedFeatureSubgroup)",
            HTML_TEMPLATE,
        )
        self.assertIn('id="feature-distribution-chart"', HTML_TEMPLATE)
        self.assertIn("drawFeatureSubgroupPanel()", HTML_TEMPLATE)

    def test_feature_story_adds_shared_subgroup_summary_frame(self) -> None:
        self.assertIn('class="feature-frame" data-frame="3"', HTML_TEMPLATE)
        self.assertIn('{state: "feature-frame-3"}', HTML_TEMPLATE)
        self.assertIn(
            "function subgroupFeatureRelations(risk, subgroup)", HTML_TEMPLATE
        )
        self.assertIn("function drawFeatureSubgroupSummary()", HTML_TEMPLATE)
        self.assertIn('higher: "higher"', HTML_TEMPLATE)
        self.assertIn('lower: "lower"', HTML_TEMPLATE)
        self.assertIn('close: "average"', HTML_TEMPLATE)
        self.assertIn(
            'state === "feature-frame-2" || state === "feature-frame-3"', HTML_TEMPLATE
        )

    def test_feature_subgroup_summary_scrolls_only_its_rows(self) -> None:
        self.assertIn(
            ".feature-subgroup-summary { display: flex; flex-direction: column;",
            HTML_TEMPLATE,
        )
        self.assertIn("height: 100%; min-height: 0; overflow: hidden;", HTML_TEMPLATE)
        self.assertIn(".feature-subgroup-summary-rows { display: grid;", HTML_TEMPLATE)
        self.assertIn(
            'const rowScroller = summary.append("div").attr("class", "feature-subgroup-summary-rows");',
            HTML_TEMPLATE,
        )

    def test_subgroup_toggles_activate_on_pointer_down_without_selecting_text(
        self,
    ) -> None:
        self.assertIn(
            ".feature-subgroup-control-label { display: block; pointer-events: none; user-select: none; }",
            HTML_TEMPLATE,
        )
        self.assertIn('.on("pointerdown", (event, d) => {', HTML_TEMPLATE)
        self.assertIn("if (event.detail !== 0) return;", HTML_TEMPLATE)

    def test_feature_subgroup_lines_have_additional_upper_domain_margin(self) -> None:
        self.assertIn("opts.upperDomainPadding || 0", HTML_TEMPLATE)
        self.assertIn("upperDomainPadding: 0.16", HTML_TEMPLATE)

    def test_intro_and_feature_context_use_inline_tooltips(self) -> None:
        self.assertNotIn('id="t-scatter-fn1"', HTML_TEMPLATE)
        self.assertNotIn('id="t-scatter-fn2"', HTML_TEMPLATE)
        self.assertIn("scatterFootnotesTooltip:", HTML_TEMPLATE)
        self.assertIn('id=\\"feature-performance-term\\"', HTML_TEMPLATE)
        self.assertIn("featurePerformanceTooltip:", HTML_TEMPLATE)

    def test_feature_threshold_negative_state_and_distribution_filter(self) -> None:
        self.assertIn("(metric.absRho || 0) >= 0.3", HTML_TEMPLATE)
        self.assertIn("negative-active", HTML_TEMPLATE)
        self.assertIn("scatter-negative", HTML_TEMPLATE)
        self.assertIn("value >= lowerBound && value <= upperBound", HTML_TEMPLATE)
        self.assertIn("featureDistributionOutlierTooltip:", HTML_TEMPLATE)
        self.assertIn(
            'const removeOutliers = selectedFeatureRisk !== "Very High";', HTML_TEMPLATE
        )
        self.assertIn("featureDistributionVeryHighTooltip:", HTML_TEMPLATE)

    def test_feature_importance_and_distribution_controls_match_latest_design(
        self,
    ) -> None:
        self.assertIn('class", "importance-strong-group"', HTML_TEMPLATE)
        self.assertNotIn("featureOutcomeTopic:", HTML_TEMPLATE)
        self.assertIn("featureDistributionPrevious:", HTML_TEMPLATE)
        self.assertIn("featureDistributionNext:", HTML_TEMPLATE)
        self.assertNotIn('id="feature-distribution-group-label"', HTML_TEMPLATE)
        self.assertIn("rotateFeature = direction =>", HTML_TEMPLATE)

    def test_information_tooltips_are_persistent_and_clickable(self) -> None:
        self.assertIn(".tooltip.persistent { pointer-events: auto; }", HTML_TEMPLATE)
        self.assertIn("activeInfoTooltipTrigger = element", HTML_TEMPLATE)
        self.assertIn('document.addEventListener("pointerdown"', HTML_TEMPLATE)
        self.assertNotIn('element.addEventListener("pointerleave"', HTML_TEMPLATE)

    def test_latest_section_layout_and_distribution_interactions(self) -> None:
        self.assertIn('class="rating-line-pane"', HTML_TEMPLATE)
        self.assertIn("event-horizon-number", HTML_TEMPLATE)
        self.assertIn("eventHorizonYears:", HTML_TEMPLATE)
        self.assertIn(
            '.feature-frame[data-frame="1"] { display: flex; flex-direction: column; overflow: hidden; }',
            HTML_TEMPLATE,
        )
        self.assertIn('featureDistributionTitle: "County Distribution"', HTML_TEMPLATE)
        self.assertIn(
            'if (options.length > 1) controls.append("button")', HTML_TEMPLATE
        )
        self.assertIn("countyDisplayName(county) || d.fips", HTML_TEMPLATE)
        self.assertIn("--takeaway-bottom: 52px", HTML_TEMPLATE)

    def test_tooltip_position_is_clamped_to_the_viewport(self) -> None:
        self.assertIn("function showTooltip(event, content", HTML_TEMPLATE)
        self.assertIn("window.innerWidth - width - edge", HTML_TEMPLATE)
        self.assertIn("window.innerHeight - height - edge", HTML_TEMPLATE)

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
        self.assertIn(
            "Economic and demographic features use ten-year county averages.",
            HTML_TEMPLATE,
        )

    def test_story_titles_lock_after_the_intro_transition(self) -> None:
        self.assertIn(
            'stage.dataset.storyDirection = direction < 0 ? "backward" : "forward"',
            HTML_TEMPLATE,
        )
        self.assertIn(".story-stage > h2 { transition: top 520ms ease", HTML_TEMPLATE)
        self.assertNotIn(".story-stage.story-step-forward > h2", HTML_TEMPLATE)
        self.assertIn(
            "takeawayTransitionTimers.delete(section)", HTML_TEMPLATE
        )
        self.assertIn("translate: 0 70px", HTML_TEMPLATE)

    def test_text_cards_have_consistent_spacing_and_overlay_rules(self) -> None:
        self.assertIn(
            ".takeaway-section { display: block; padding: 24px 28px; }", HTML_TEMPLATE
        )
        self.assertNotIn('id="event-window-takeaway" style=', HTML_TEMPLATE)
        self.assertIn("--takeaway-space: min(17svh, 132px)", HTML_TEMPLATE)
        self.assertIn(
            "> .panel > *:not(.takeaway) { opacity: 1; filter: none; }", HTML_TEMPLATE
        )

    def test_story_cards_reserve_a_sources_footer(self) -> None:
        self.assertIn(
            ".story-stage > .panel:has(> .sources) { padding-bottom: 78px; }",
            HTML_TEMPLATE,
        )
        self.assertIn("bottom: 14px; margin: 0; padding: 10px 0 0;", HTML_TEMPLATE)

    def test_named_group_line_plots_require_complete_monthly_histories(self) -> None:
        price_source = inspect.getsource(page_builder.build_price_risk)
        event_source = inspect.getsource(page_builder._build_window_data)
        feature_source = inspect.getsource(page_builder.build_feature_payload)
        self.assertIn("filter_complete_event_window_lines", price_source)
        self.assertIn("required_history_months", price_source)
        self.assertIn("filter_complete_event_window_lines", event_source)
        self.assertIn("required_x_values=required", event_source)
        self.assertIn("FEATURE_PERFORMANCE_TARGET_COLUMN", feature_source)
        self.assertIn("analysis_group = risk_counties.dropna", feature_source)
        self.assertIn("performance_group = analysis_group.copy()", feature_source)
        self.assertIn("target_column=FEATURE_PERFORMANCE_TARGET_COLUMN", feature_source)

    def test_complete_history_tooltip_is_attached_to_named_plot_headers(self) -> None:
        self.assertIn("completeMonthlyPlotTooltip:", HTML_TEMPLATE)
        self.assertIn("function setCompleteMonthlyPlotTitle", HTML_TEMPLATE)
        self.assertIn(
            'setCompleteMonthlyPlotTitle("#t-pricing-card-title", TEXT.pricingCardTitle)',
            HTML_TEMPLATE,
        )
        self.assertIn(
            'setCompleteMonthlyPlotTitle("#feature-chart-title", TEXT.featureLineTitle)',
            HTML_TEMPLATE,
        )
        self.assertIn("setCompleteMonthlyPlotTitle(\n      eventTitleElement", HTML_TEMPLATE)

    def test_source_tooltips_list_linked_raw_datasets_without_mart_details(self) -> None:
        self.assertIn("html: labelText === TEXT.featureSourcesTopic", HTML_TEMPLATE)
        for key in [
            "pricingSources",
            "eventsSources",
            "featureSourcesNote",
            "playbookSources",
        ]:
            match = re.search(rf"^  {key}: (.+)$", HTML_TEMPLATE, flags=re.MULTILINE)
            self.assertIsNotNone(match)
            source_text = match.group(1)
            self.assertIn('href="https://', source_text)
            self.assertNotIn("<code>", source_text)
            self.assertNotIn("mart.", source_text)

    def test_feature_subgroup_labels_are_performance_based_and_ordered(self) -> None:
        self.assertIn(
            'subgroupNamesFour: ["Strong Overperformers", "Mild Overperformers", "Mild Underperformers", "Strong Underperformers"]',
            HTML_TEMPLATE,
        )
        self.assertIn(
            'subgroupNamesThree: ["Overperformers", "Average Performers", "Underperformers"]',
            HTML_TEMPLATE,
        )
        self.assertIn(
            "const orderedGroups = [...payload.groups].sort((a, b) => b.index - a.index)",
            HTML_TEMPLATE,
        )
        self.assertIn('attr("aria-pressed"', HTML_TEMPLATE)
        self.assertIn("startFeatureSubgroupSequence()", HTML_TEMPLATE)

    def test_feature_subgroup_legend_has_reliable_toggles_and_no_line_end_label(
        self,
    ) -> None:
        self.assertIn("hideEndLabel: true", HTML_TEMPLATE)
        self.assertIn('id="feature-subgroup-toggles"', HTML_TEMPLATE)
        self.assertIn("button.feature-subgroup-control", HTML_TEMPLATE)
        self.assertIn("selectFeatureSubgroup(Number(d.index), true)", HTML_TEMPLATE)
        self.assertIn(
            ".feature-subgroup-controls.visible { display: flex; }", HTML_TEMPLATE
        )
        self.assertIn('class="feature-plot-shell"', HTML_TEMPLATE)
        self.assertIn('class="feature-subgroup-control-stack"', HTML_TEMPLATE)
        self.assertIn("cursor: pointer !important;", HTML_TEMPLATE)
        self.assertIn(
            "pointer-events: auto; touch-action: manipulation;", HTML_TEMPLATE
        )
        self.assertIn("cursor: pointer; pointer-events: none;", HTML_TEMPLATE)
        self.assertIn("marginRight: 24", HTML_TEMPLATE)
        self.assertIn(
            "const names = count === 3 ? TEXT.subgroupNamesThree : TEXT.subgroupNamesFour",
            HTML_TEMPLATE,
        )

    def test_both_focus_county_lines_are_solid(self) -> None:
        self.assertNotIn(
            'd.isFocus && d.focusPosition === "Below" ? "7 4" : null',
            HTML_TEMPLATE,
        )

    def test_playbook_uses_feature_analysis_and_subgroup_data(self) -> None:
        self.assertIn("function playbookFeatureProfile(county)", HTML_TEMPLATE)
        self.assertIn("historicalPerformanceAssignment", HTML_TEMPLATE)
        self.assertIn('class="playbook-subgroup-badge"', HTML_TEMPLATE)
        self.assertNotIn("modelCountyProfiles", HTML_TEMPLATE)
        self.assertNotIn("modelTopFeaturesByRisk", HTML_TEMPLATE)

    def test_playbook_reports_insufficient_feature_data(self) -> None:
        self.assertIn("playbookInsufficientFeatureData:", HTML_TEMPLATE)
        self.assertIn("playbookInsufficientEventWindowData:", HTML_TEMPLATE)
        self.assertIn("const hasFeatureData = summarizeSubgroup", HTML_TEMPLATE)
        self.assertIn("const insufficientCopy = !profile.row", HTML_TEMPLATE)
        self.assertIn('style("display", "none").text("")', HTML_TEMPLATE)
        self.assertIn(
            "Insufficient feature data available for {county}.", HTML_TEMPLATE
        )
        self.assertIn(
            "could not be determined because no housing observations were available in the event-window analysis period.",
            HTML_TEMPLATE,
        )
        self.assertIn('class="playbook-feature-insufficient"', HTML_TEMPLATE)

    def test_playbook_subgroup_copy_is_a_county_sentence(self) -> None:
        self.assertIn(
            "{county}'s house price growth rate around extreme climate events makes it a {subgroup} among {risk} Risk counties.",
            HTML_TEMPLATE,
        )
        self.assertIn("function playbookPerformanceName(label)", HTML_TEMPLATE)
        self.assertIn(
            "subgroup: playbookPerformanceName(profile.subgroupName)", HTML_TEMPLATE
        )

    def test_playbook_uses_analysis_significance_rule_and_scrolls_only_feature_values(
        self,
    ) -> None:
        self.assertIn("function mostImportantFeatureMetrics(risk)", HTML_TEMPLATE)
        self.assertIn(
            "const metrics = mostImportantFeatureMetrics(county.riskRating);",
            HTML_TEMPLATE,
        )
        self.assertIn(
            ".playbook-feature-summary { display: grid; gap: 6px; max-height:",
            HTML_TEMPLATE,
        )
        self.assertIn("overflow-y: auto;", HTML_TEMPLATE)

    def test_later_playbook_frames_use_performance_status_and_warning_dashboard(
        self,
    ) -> None:
        self.assertIn(
            "function renderPlaybookPerformanceStatus(county, compact = false)",
            HTML_TEMPLATE,
        )
        self.assertIn('if (state === "history-compare")', HTML_TEMPLATE)
        self.assertNotIn("renderPlaybookPerformanceStatus(county, false);", HTML_TEMPLATE)
        self.assertIn('id="playbook-history-comparison"', HTML_TEMPLATE)
        self.assertIn("renderPlaybookPerformanceTakeaway(county);", HTML_TEMPLATE)
        self.assertIn('class="playbook-factor-columns"', HTML_TEMPLATE)
        self.assertIn(
            "const metrics = mostImportantFeatureMetrics(risk);", HTML_TEMPLATE
        )
        self.assertIn("flex: 1 1 calc((100% - 21px) / 4)", HTML_TEMPLATE)
        self.assertIn(".playbook-warning-grid { display: flex; flex-wrap: wrap; justify-content: space-between;", HTML_TEMPLATE)
        self.assertIn('grid-template-rows: auto minmax(0, 1fr) auto;', HTML_TEMPLATE)
        self.assertIn('.playbook-warning-grid { min-height: 100%; align-content: stretch; align-items: stretch; }', HTML_TEMPLATE)
        self.assertIn('padding: 0; border: 0; background: transparent; box-shadow: none; overflow-y: auto;', HTML_TEMPLATE)
        self.assertIn('#playbook-warning-takeaway { grid-row: 3; margin-top: 0; align-self: end; }', HTML_TEMPLATE)
        self.assertNotIn('.attr("title", strong ? TEXT.featureStrongTooltip', HTML_TEMPLATE)
        self.assertIn('.attr("title", `${featureLabel(feature)} · |ρ| ${d3.format(".2f")(metric.absRho || 0)}`)', HTML_TEMPLATE)
        self.assertIn('class="playbook-warning-takeaway"', HTML_TEMPLATE)
        self.assertNotIn('"\\u2191 rising"', HTML_TEMPLATE)
        self.assertNotIn('"\\u2193 falling"', HTML_TEMPLATE)

    def test_playbook_drops_outlook_for_counties_without_performer_assignment(
        self,
    ) -> None:
        self.assertIn("function playbookHasPerformanceGroup(county)", HTML_TEMPLATE)
        self.assertIn(
            'config.filter(step => step.state !== "history-outlook")', HTML_TEMPLATE
        )
        self.assertIn("syncPlaybookStoryLength(county);", HTML_TEMPLATE)
        self.assertIn("playbookInsufficientPerformance:", HTML_TEMPLATE)
        self.assertIn(
            "had insufficient data so its housing market performance could not be reliably determined",
            HTML_TEMPLATE,
        )

    def test_playbook_outlook_is_full_width_without_history_plot(self) -> None:
        self.assertIn(
            '[data-story-state="history-outlook"] .playbook-history-pane { display: none; }',
            HTML_TEMPLATE,
        )
        self.assertIn(
            '[data-story-state="history-outlook"] .playbook-selected-layout { grid-template-columns: minmax(0, 1fr);',
            HTML_TEMPLATE,
        )
        branch_start = HTML_TEMPLATE.index('if (state === "history-outlook")')
        branch_end = HTML_TEMPLATE.index("function initPlaybook()", branch_start)
        outlook_branch = HTML_TEMPLATE[branch_start:branch_end]
        self.assertIn("renderPlaybookOutlook(county);", outlook_branch)
        self.assertNotIn("drawPlaybookHistory", outlook_branch)

    def test_playbook_history_profile_card_has_fixed_chrome_and_scrollable_traits(
        self,
    ) -> None:
        self.assertIn('playbookSubgroupFeatureTitle: "County Traits"', HTML_TEMPLATE)
        self.assertIn(".playbook-back-button { position: absolute;", HTML_TEMPLATE)
        self.assertIn("padding: 5px 8px; font-size: 10px;", HTML_TEMPLATE)
        self.assertIn(
            '[data-story-state^="history-"] #playbook-selected-county-name { display: none !important; }',
            HTML_TEMPLATE,
        )
        self.assertIn(
            '[data-story-state="history-compare"] .playbook-performance-pane { display: none;',
            HTML_TEMPLATE,
        )
        self.assertIn('d3.select("#playbook-history-comparison")', HTML_TEMPLATE)

    def test_monthly_lines_keep_gaps_and_controls_are_icon_only(self) -> None:
        self.assertNotIn("interpolateInternalHistory", HTML_TEMPLATE)
        self.assertIn("d3.line().defined(d => d.value != null)", HTML_TEMPLATE)
        self.assertIn("function setSequenceButton(selector, paused)", HTML_TEMPLATE)
        self.assertIn('.text(paused ? "\\u25B6" : "\\u275A\\u275A")', HTML_TEMPLATE)
        self.assertNotIn('"▶ Resume"', HTML_TEMPLATE)
        self.assertNotIn('"❚❚ Pause"', HTML_TEMPLATE)

    def test_event_overview_is_a_full_width_bar_chart(self) -> None:
        self.assertIn("function drawEventRiskBars()", HTML_TEMPLATE)
        self.assertIn(".event-overview-chart { width: 100%;", HTML_TEMPLATE)
        self.assertIn("rect.event-risk-bar", HTML_TEMPLATE)
        self.assertNotIn("function drawEventRiskPie()", HTML_TEMPLATE)

    def test_event_overview_takeaways_are_separate_scroll_steps(self) -> None:
        self.assertIn(
            "Extreme climate events have affected counties across the board, even the low risk ones.",
            HTML_TEMPLATE,
        )
        first = '{state: "takeaway-overview-0", takeaway: "#event-overview-takeaway",'
        second = '{state: "takeaway-overview-1", takeaway: "#event-overview-question",'
        self.assertIn(first, HTML_TEMPLATE)
        self.assertIn(second, HTML_TEMPLATE)
        self.assertLess(HTML_TEMPLATE.index(first), HTML_TEMPLATE.index(second))

    def test_feature_scatter_uses_named_rows_and_short_performer_controls(self) -> None:
        self.assertIn(
            "DATA.features.countyRowsByRisk[selectedFeatureRisk]", HTML_TEMPLATE
        )
        self.assertIn("countyDisplayName(d)", HTML_TEMPLATE)
        self.assertIn(
            'subgroupShortNamesFour: ["Strong", "Mildly Strong", "Mildly Weak", "Weak"]',
            HTML_TEMPLATE,
        )
        self.assertNotIn('id="feature-very-high-info-slot"', HTML_TEMPLATE)
        self.assertIn("veryHighButton.after(info)", HTML_TEMPLATE)
        self.assertIn('info.classList.add("risk-legend-info")', HTML_TEMPLATE)
        self.assertNotIn("TEXT.featureVeryHighGroupTooltip", HTML_TEMPLATE)

    def test_playbook_fallback_assignment_uses_subgroup_ranges_and_extremes(
        self,
    ) -> None:
        groups = [
            {"index": 0, "targetMin": 0.08, "targetMax": 0.12},
            {"index": 1, "targetMin": 0.04, "targetMax": 0.07},
            {"index": 2, "targetMin": 0.00, "targetMax": 0.03},
            {"index": 3, "targetMin": -0.05, "targetMax": -0.01},
        ]
        self.assertEqual(assign_performer_subgroup_from_ranges(0.09, groups), 0)
        self.assertEqual(assign_performer_subgroup_from_ranges(0.15, groups), 0)
        self.assertEqual(assign_performer_subgroup_from_ranges(-0.08, groups), 3)
        self.assertEqual(assign_performer_subgroup_from_ranges(0.035, groups), 1)

    def test_playbook_county_history_line_is_distinct_from_risk_colors(self) -> None:
        self.assertIn('const COUNTY_LINE_COLOR = "#2456a6";', HTML_TEMPLATE)
        self.assertNotIn("playbook-county-line-halo", HTML_TEMPLATE)
        self.assertIn(
            'attr("stroke", COUNTY_LINE_COLOR).attr("stroke-width", 2.4)', HTML_TEMPLATE
        )
        self.assertIn(
            "{label: TEXT.playbookSeriesLegend, color: COUNTY_LINE_COLOR, opacity: 1, line: true}",
            HTML_TEMPLATE,
        )

    def test_playbook_event_comparison_is_qualitative_and_handles_volatility(
        self,
    ) -> None:
        self.assertIn("function eventWindowTrendStats(points)", HTML_TEMPLATE)
        self.assertIn(
            "function qualitativeRelation(difference, threshold)", HTML_TEMPLATE
        )
        self.assertIn(
            "function alignmentExtent(observed, expected, threshold)", HTML_TEMPLATE
        )
        self.assertIn("eventAlignmentSummary:", HTML_TEMPLATE)
        self.assertIn("eventAlignmentWithoutSubgroup:", HTML_TEMPLATE)
        self.assertIn("riskAlignment", HTML_TEMPLATE)
        self.assertIn("subgroupAlignment", HTML_TEMPLATE)
        self.assertIn("volatileEventSummary:", HTML_TEMPLATE)
        self.assertIn("const tooVolatile = (", HTML_TEMPLATE)
        self.assertNotIn("relativeDirection:", HTML_TEMPLATE)

    def test_map_boundaries_share_a_cohesive_visual_treatment(self) -> None:
        self.assertIn(".county { stroke: #ffffff; stroke-width: .45;", HTML_TEMPLATE)
        self.assertIn(
            ".state-boundary { fill: none; stroke: #173f37; stroke-width: 1.4;",
            HTML_TEMPLATE,
        )

    def test_playbook_event_list_scrolls_within_frame_three(self) -> None:
        self.assertIn(
            '[data-story-state="history-events"] .playbook-events-pane { display: flex;',
            HTML_TEMPLATE,
        )
        self.assertIn(
            "overflow-y: auto; overscroll-behavior: contain; scrollbar-gutter: stable;",
            HTML_TEMPLATE,
        )

    def test_state_boundaries_are_dissolved_from_displayed_counties(self) -> None:
        county_geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"fips": "01001"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
                    },
                },
                {
                    "type": "Feature",
                    "properties": {"fips": "01003"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[1, 0], [2, 0], [2, 1], [1, 1], [1, 0]]],
                    },
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

    def test_playbook_has_five_requested_frames_and_risk_group_comparison(self) -> None:
        for state in (
            '"search"',
            '"history-map"',
            '"history-events"',
            '"history-compare"',
            '"history-outlook"',
        ):
            self.assertIn(f"state: {state}", HTML_TEMPLATE)
        self.assertIn("function buildRiskGroupSeries(county)", HTML_TEMPLATE)
        self.assertIn(
            "function drawPlaybookHistory(county, compareRisk = false, showEvents = true)",
            HTML_TEMPLATE,
        )
        self.assertIn("d.q1 - .5 * (d.q3 - d.q1)", HTML_TEMPLATE)
        self.assertIn("d.q3 + .5 * (d.q3 - d.q1)", HTML_TEMPLATE)
        self.assertIn(".duration(900)", HTML_TEMPLATE)
        self.assertIn('id="playbook-back-to-search"', HTML_TEMPLATE)
        self.assertIn('id="playbook-profile-map"', HTML_TEMPLATE)

    def test_takeaway_keeps_main_card_fully_visible_behind_it(self) -> None:
        self.assertNotIn("opacity: .5; filter: none;", HTML_TEMPLATE)
        self.assertIn(
            "padding-bottom: calc(var(--takeaway-space) + var(--takeaway-footnote-space) + 12px) !important;",
            HTML_TEMPLATE,
        )
        self.assertIn("function syncTakeawaySpace(section, takeaway)", HTML_TEMPLATE)
        self.assertIn(
            "if (takeaway !== section.querySelector('.takeaway.story-active-takeaway')) return;",
            HTML_TEMPLATE,
        )
        self.assertIn("max-height: none;", HTML_TEMPLATE)

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
