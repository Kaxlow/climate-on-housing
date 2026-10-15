"""Behavioral coverage for county detection and the optional questions gate."""
import json
from pathlib import Path
import subprocess
import unittest

SCRIPT = (Path(__file__).parents[1] / 'src/housing_climate_risk/page_data/web/climate-risk-housing.js').read_text(encoding='utf-8')

class PlaybookFlowTests(unittest.TestCase):
    def run_js(self, code):
        result = subprocess.run(['node', '-e', code], capture_output=True, text=True, check=True)
        return json.loads(result.stdout)

    def test_detection_success_international_failure_and_missing_county(self):
        function = 'async function detectPlaybookCounty()' + SCRIPT.split('async function detectPlaybookCounty()', 1)[1].split('function renderPlaybookQuestions', 1)[0]
        code = r'''
let countyDetectionStarted=false;
const elements={};
const document={getElementById:id=>elements[id] ||= {hidden:true}};
const county={fips:'08031',county:'Denver County'};
const playbookCountyByFips=new Map([['08031',county]]);
const countyDisplayName=c=>c.county;
let selected=null, advanced=false, calls=[];
const selectPlaybookCounty=c=>selected=c.fips,goToPlaybookProfile=()=>advanced=true;
let location={country_code:'US',longitude:-104.99,latitude:39.74};
let geoid='0803199999',fail=false;
const fetch=async()=>({ok:!fail,json:async()=>location});
const censusCountyAt=async(x,y)=>{calls.push([x,y]);return {result:{geographies:{'County Subdivisions':[{GEOID:geoid}]}}}};
''' + function + r'''
(async()=>{
await detectPlaybookCounty();
const before=selected;
elements['playbook-location-continue'].onclick();
await detectPlaybookCounty();
const success=[before,selected,advanced,calls.length,calls[0],elements['playbook-location-message'].textContent];
countyDetectionStarted=false;location={country_code:'CA'};await detectPlaybookCounty();
const international=elements['playbook-location-message'].textContent;
countyDetectionStarted=false;fail=true;await detectPlaybookCounty();
const failure=elements['playbook-location-message'].textContent;
countyDetectionStarted=false;fail=false;location={country_code:'US',longitude:0,latitude:0};geoid='99999999';await detectPlaybookCounty();
console.log(JSON.stringify([success,international,failure,elements['playbook-location-message'].textContent]));
})();
'''
        success, international, failure, missing = self.run_js(code)
        self.assertEqual(success, [None, '08031', True, 1, [-104.99,39.74], 'You appear to be in Denver County.'])
        self.assertIn("don't appear to be in the US", international)
        self.assertIn('could not be detected', failure)
        self.assertEqual(failure, missing)

    def test_questions_remain_locked_until_explicit_unlock(self):
        function = 'function storyConfigForSection(id)' + SCRIPT.split('function storyConfigForSection(id)',1)[1].split('function syncPlaybookStoryLength',1)[0]
        code = "const STORY_CONFIG={playbook:[{state:'conclusion'},{state:'questions'}]};let playbookQuestionsUnlocked=false;" + function + "const before=storyConfigForSection('playbook').map(s=>s.state);playbookQuestionsUnlocked=true;console.log(JSON.stringify([before,storyConfigForSection('playbook').map(s=>s.state)]));"
        self.assertEqual(self.run_js(code), [['conclusion'],['conclusion','questions']])
