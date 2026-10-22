"""Execute the production controller with deterministic input and animation clocks."""
import shutil
import subprocess
import unittest

from housing_climate_risk.page_data import climate_risk_housing as page

SCRIPT = (page.WEB_ASSET_DIR / "climate-risk-housing.js").read_text(encoding="utf-8")
CONTROLLER = SCRIPT.split("let storyViewportHeight", 1)[1].split("function initStoryEdgeNavigation", 1)[0]

HARNESS = r"""
const assert = require('node:assert/strict');
let clock = 1000, sequence = 0, reduced = false;
const callbacks = new Map(), listeners = {};
const element = {disabled:false};
class Element {}
const document = {
 documentElement:{scrollHeight:12000,clientWidth:1440,style:{setProperty(){}}},
 body:{}, activeElement:null, fonts:{ready:{then(){}}},
 querySelector:()=>null, querySelectorAll:()=>[], getElementById:()=>element,
 addEventListener:(type,fn)=>{listeners[type]=fn;},
};
const window = {innerHeight:1000,innerWidth:1440,scrollY:0,
 scrollTo({top}){this.scrollY=top;positions.push(top);},
 addEventListener:(type,fn)=>{listeners[type]=fn;},
};
const positions = [];
const performance = {now:()=>clock};
const requestAnimationFrame=fn=>{callbacks.set(++sequence,fn);return sequence;};
const cancelAnimationFrame=id=>callbacks.delete(id);
const storyMotionReduced=()=>reduced;
const updateStoryFromScroll=()=>{};
const scheduleStoryChartRedraw=()=>{};
const fitStoryTitle=()=>{};
const STORY_CONFIG={roadmap:[{},{},{},{},{}],pricing:[{},{},{},{}],playbook:[{},{},{},{}]};
const storyConfigForSection=id=>STORY_CONFIG[id];
let selectedCountyFips=null;
const getComputedStyle=element=>element.computed||{overflowY:'visible',opacity:'1'};
function deliver(timestamp) {
 clock=timestamp;
 const batch=[...callbacks.values()];callbacks.clear();
 batch.forEach(callback=>callback(timestamp));
}
function finishMotion() {for(let i=0;i<20&&callbacks.size;i++)deliver(clock+16);}
function wheel(timestamp,deltaY=120,target=new Element(),extra={}) {
 const event={timeStamp:timestamp,target,deltaY,deltaX:0,deltaMode:0,cancelable:true,
  preventDefault(){this.prevented=true;},...extra};
 target.closest??=()=>null;
 clock=timestamp;handleStoryWheel(event);return event;
}
"""


@unittest.skipUnless(shutil.which("node"), "Node.js required")
class StoryNavigationTests(unittest.TestCase):
    def test_loading_waits_for_a_replacement_and_finishes_with_empty_data(self):
        callback = SCRIPT.split('renderWhenNear("#score-scatter", ', 1)[1].split('\n}, "350px', 1)[0] + '\n}'
        scenario = """
const assert=require('node:assert/strict');
let scoreHistoryData, calls=0;
const loadDeferredData=async()=>({series:[],months:[]});
const drawScoreScatter=async()=>++calls===1?false:undefined;
const render=""" + callback + """;
(async()=>{await render();assert.equal(calls,2);assert.equal(scoreHistoryData.series.length,0);})();
"""
        result = subprocess.run(["node", "-e", scenario], capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)

    def run_controller(self, scenario):
        setup = """
storyFrameCache=Array.from({length:8},(_,index)=>({id:'roadmap',index,top:index*1000}));
"""
        result = subprocess.run(
            ["node", "-e", HARNESS + "let storyViewportHeight" + CONTROLLER + setup + scenario],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_delayed_first_timestamp_never_moves_backwards(self):
        self.run_controller("""
window.scrollY=5000;animateStoryScroll(6000);
deliver(984);assert.equal(window.scrollY,5000);
deliver(968);assert.equal(window.scrollY,5000);
deliver(1094);assert.equal(window.scrollY,5875);
deliver(1204);assert.equal(window.scrollY,6000);
assert.equal(storyNavigationAnimation,null);
assert.ok(positions.every((y,i)=>!i||y>=positions[i-1]));
""")

    def test_cancelled_callback_cannot_overwrite_new_destination(self):
        self.run_controller("""
animateStoryScroll(1000);const stale=[...callbacks.values()][0];
animateStoryScroll(2000);stale(9000);assert.equal(window.scrollY,0);
finishMotion();assert.equal(window.scrollY,2000);
""")

    def test_navigation_updates_a_pending_resize_anchor(self):
        self.run_controller("""
storyResizeAnchor=storyFrameCache[0];
animateStoryScroll(2000);
assert.equal(storyResizeAnchor.index,2);
finishMotion();assert.equal(window.scrollY,2000);
""")

    def test_wheel_burst_and_trailing_momentum_advance_once(self):
        self.run_controller("""
wheel(1000,4000);deliver(1010);
for(const t of [1040,1080,1120]){wheel(t);deliver(t);}
deliver(1230);assert.equal(window.scrollY,1000);
wheel(1260);wheel(1380);assert.equal(storyNavigationAnimation,null);
wheel(1540);finishMotion();assert.equal(window.scrollY,2000);
wheel(clock+160,-120);finishMotion();assert.equal(window.scrollY,1000);
""")

    def test_line_page_and_pixel_wheels_share_the_frame_limit(self):
        self.run_controller("""
reduced=true;
for(const deltaMode of [0,1,2]){
 wheelGesture=null;window.scrollY=0;
 wheel(clock+200,deltaMode===0?8000:1,new Element(),{deltaMode});
 assert.equal(window.scrollY,1000);
}
""")

    def test_nested_scroll_owner_keeps_momentum_at_its_boundary(self):
        self.run_controller("""
const pane=Object.assign(new Element(),{scrollHeight:1000,clientHeight:200,scrollTop:300,
 style:{},parentElement:document.body,computed:{overflowY:'auto'}});
assert.equal(wheel(1000,120,pane).prevented,undefined);
pane.scrollTop=800;
assert.equal(wheel(1080,120,pane).prevented,true);
assert.equal(storyNavigationAnimation,null);
wheel(1240,120,pane);finishMotion();assert.equal(window.scrollY,1000);
""")

    def test_maps_zoom_and_horizontal_input_retain_native_ownership(self):
        self.run_controller("""
const map=Object.assign(new Element(),{closest:()=>({})});
for(const event of [wheel(1000,120,map),wheel(1200,120,new Element(),{ctrlKey:true}),
 wheel(1400,120,new Element(),{deltaX:300})])assert.equal(event.prevented,undefined);
assert.equal(storyNavigationAnimation,null);
""")

    def test_reduced_motion_preserves_gestures_and_playbook_gate(self):
        self.run_controller("""
reduced=true;wheel(1000);wheel(1080);assert.equal(window.scrollY,1000);
storyFrameCache=[{id:'playbook',index:2,top:1000},{id:'playbook',index:3,top:2000}];
navigateStory(1);assert.equal(window.scrollY,1000);
selectedCountyFips='08031';navigateStory(1);assert.equal(window.scrollY,2000);
""")


if __name__ == "__main__":
    unittest.main()
