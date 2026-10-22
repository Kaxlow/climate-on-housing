// NODE_PATH must resolve Playwright. Results and screenshots go to ignored test-results/.
const {chromium} = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const http = require('node:http');
const root = path.resolve(__dirname, '../output');
const resultsDir = path.resolve(__dirname, '../test-results/navigation');
const profileOnly = process.argv.includes('--profile');
const renderingOnly = process.argv.includes('--rendering');
const baseline = process.argv.includes('--baseline');
const inputSessions = new WeakMap();
const server = http.createServer((req, res) => {
  const name = decodeURIComponent(new URL(req.url, 'http://localhost').pathname);
  let file = path.resolve(root, '.' + (name === '/' ? '/index.html' : name));
  if (!file.startsWith(root + path.sep)) { res.writeHead(403).end(); return; }
  if (baseline && /climate-risk-housing\.(js|css)$/.test(file)) file = path.join(resultsDir, '../navigation-baseline', path.basename(file));
  fs.readFile(file, (error, data) => {
    if (error) { res.writeHead(404).end(); return; }
    res.setHeader('Content-Type', file.endsWith('.js') ? 'text/javascript' : file.endsWith('.css') ? 'text/css' : 'text/html');
    res.end(data);
  });
});

async function ready(page) {
  page.setDefaultTimeout(30000);
  page.setDefaultNavigationTimeout(30000);
  await page.route('**/analytics/**', route => route.abort());
  await page.route('https://ipapi.co/**', route => route.fulfill({contentType: 'application/json', body: '{}'}));
  await page.goto(`http://127.0.0.1:${server.address().port}`, {waitUntil: 'domcontentloaded'});
  await page.waitForLoadState('load');
  await page.waitForFunction(() => document.querySelector('#roadmap .story-stage'));
  await page.evaluate(() => document.fonts.ready);
}
async function settle(page) {
  await page.waitForFunction(() => storyNavigationAnimation == null && !storyScrollTarget);
  await page.waitForTimeout(200); // The next action is a fresh gesture.
}
async function jump(page, id, index) {
  await page.evaluate(({id, index}) => {
    cancelStoryNavigation();
    const section = document.getElementById(id);
    const height = typeof storyViewportHeight === 'undefined' ? innerHeight : storyViewportHeight;
    window.scrollTo({top: section.offsetTop + index * height, behavior: 'instant'});
    if (typeof wheelGesture !== 'undefined') wheelGesture = null;
    updateStoryFromScroll();
  }, {id, index});
  await page.waitForTimeout(200);
}
async function wheel(page, direction = 1, count = 10, point = {x: 4, y: 200}) {
  await page.mouse.move(point.x, point.y);
  const events = [], session = inputSessions.get(page);
  for (let i = 0; i < count; i++) {
    // Preserve hardware-like timestamps while the page is busy rendering.
    events.push(session.send('Input.dispatchMouseEvent', {type: 'mouseWheel', x: point.x, y: point.y,
      deltaX: 0, deltaY: direction * 300, timestamp: Date.now() / 1000}));
    await new Promise(resolve => setTimeout(resolve, 10));
  }
  await Promise.all(events);
  await settle(page);
}
async function swipe(page, cdp, direction = 1, point = {x: 5, y: 250}) {
  const touch = (y) => [{x: point.x, y, id: 1}];
  await cdp.send('Input.dispatchTouchEvent', {type: 'touchStart', touchPoints: touch(point.y)});
  for (let i = 1; i <= 5; i++) {
    await cdp.send('Input.dispatchTouchEvent', {type: 'touchMove', touchPoints: touch(point.y - direction * i * 22)});
    await page.waitForTimeout(16);
  }
  await cdp.send('Input.dispatchTouchEvent', {type: 'touchEnd', touchPoints: []});
  await settle(page);
}
async function location(page) {
  return page.evaluate(() => {
    const frame = activeStoryFrame();
    return {id: frame.id, index: frame.index};
  });
}
async function record(page, action) {
  await page.evaluate(() => {
    window.navigationSamples = [];
    window.recordingNavigation = true;
    const tick = t => {
      navigationSamples.push({t, y: scrollY});
      if (recordingNavigation) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  });
  await action();
  return page.evaluate(() => { recordingNavigation = false; return navigationSamples; });
}
function monotonic(samples) {
  assert.ok(samples.every((sample, i) => !i || sample.y >= samples[i - 1].y - 2), 'forward scrolling never rebounds');
}
async function checkTakeaways(page, gesture) {
  await jump(page, 'playbook', 2);
  await page.waitForFunction(() => playbookSectionRendered);
  await page.evaluate(() => selectPlaybookCounty(playbookCountyByFips.get('08031')));
  const steps = await page.evaluate(() => Object.entries(STORY_CONFIG).flatMap(([id, config]) =>
    config.flatMap((step, index) => step.takeaway ? [{id, index, state: step.state}] : [])));
  for (const step of steps) {
    await jump(page, step.id, step.index - 1);
    await page.waitForFunction(id => id === 'pricing' ? scoreScatterRendered : id === 'pricing-grouping' ? ratingSectionRendered
      : id === 'events' ? eventSectionRendered : id === 'features' ? featureSectionRendered : playbookSectionRendered, step.id);
    await page.evaluate(id => {
      const panel = document.querySelector(`#${id} .story-stage > .panel`);
      panel.scrollTop = panel.scrollHeight;
    }, step.id);
    await gesture();
    assert.deepEqual(await location(page), {id: step.id, index: step.index}, `reach ${step.id}/${step.state}`);
    await page.waitForTimeout(350);
    const visibility = await page.evaluate(({id, state}) => {
      const stage = document.querySelector(`#${id} .story-stage`);
      const cards = [...stage.querySelectorAll('.takeaway.story-active-takeaway')];
      const card = cards[0], rect = card?.getBoundingClientRect(), panel = stage.querySelector(':scope > .panel').getBoundingClientRect();
      return {state: stage.dataset.storyState, count: cards.length, opacity: card && getComputedStyle(card).opacity,
        visible: rect && rect.height > 0 && rect.top >= panel.top - 2 && rect.bottom <= Math.min(panel.bottom, innerHeight) + 2};
    }, step);
    assert.deepEqual(visibility, {state: step.state, count: 1, opacity: '1', visible: true}, `readable ${step.id}/${step.state}`);
  }
  return steps.length;
}
async function checkViewport(browser, viewport, touch, exhaustive) {
  const context = await browser.newContext({viewport, isMobile: touch, hasTouch: touch});
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await ready(page);
  const cdp = await context.newCDPSession(page);
  inputSessions.set(page, cdp);
  const gesture = direction => touch ? swipe(page, cdp, direction) : wheel(page, direction);
  for (const cache of ['cold', 'warm']) {
    await jump(page, 'roadmap', 4);
    monotonic(await record(page, () => gesture(1)));
    assert.deepEqual(await location(page), {id: 'pricing', index: 0}, `${cache} roadmap boundary`);
  }
  await gesture(1);
  assert.deepEqual(await location(page), {id: 'pricing', index: 1});
  await gesture(-1);
  assert.deepEqual(await location(page), {id: 'pricing', index: 0});
  await page.keyboard.press('ArrowDown');await settle(page);
  assert.deepEqual(await location(page), {id: 'pricing', index: 1});
  // Repeated keydown events from a held key retain one-frame progression.
  await page.keyboard.down('ArrowUp');await page.keyboard.down('ArrowUp');await page.keyboard.up('ArrowUp');await settle(page);
  assert.deepEqual(await location(page), {id: 'pricing', index: 0});
  await page.emulateMedia({reducedMotion: 'reduce'});
  await gesture(1);
  assert.deepEqual(await location(page), {id: 'pricing', index: 1});
  await page.emulateMedia({reducedMotion: 'no-preference'});
  // Height changes rebase the same logical frame after the gesture settles.
  await page.setViewportSize({width: viewport.width, height: viewport.height - 70});
  await page.waitForTimeout(450);
  assert.deepEqual(await location(page), {id: 'pricing', index: 1});
  assert.ok(await page.evaluate(() => Math.abs(scrollY - (document.getElementById('pricing').offsetTop + storyViewportHeight)) < 2));
  await page.setViewportSize(viewport);await page.waitForTimeout(450);
  await page.setViewportSize({width: viewport.width, height: viewport.height - 70});
  await page.waitForFunction(() => storyResizeAnchor != null);
  await page.locator('#pricing .story-stage > .panel').evaluate(node => {node.scrollTop = node.scrollHeight;});
  await page.keyboard.press('ArrowDown');await settle(page);
  await page.waitForTimeout(400);
  assert.deepEqual(await location(page), {id: 'pricing', index: 2}, 'navigation supersedes a pending viewport anchor');
  await page.setViewportSize(viewport);await page.waitForTimeout(450);
  await jump(page, 'pricing', 1);
  if (touch && viewport.width < 760) {
    await page.setViewportSize({width: 844, height: 390});await page.waitForTimeout(450);
    assert.deepEqual(await location(page), {id: 'pricing', index: 1}, 'orientation preserves the active frame');
    await page.setViewportSize(viewport);await page.waitForTimeout(450);
    assert.deepEqual(await location(page), {id: 'pricing', index: 1});
  }
  // A real short roadmap scrolls internally before its next story step.
  await jump(page, 'roadmap', 2);
  const roadmapOverflow = await page.locator('#roadmap .story-stage').evaluate(node => node.scrollHeight > node.clientHeight + 2);
  if (touch && roadmapOverflow) {
    await swipe(page, cdp, 1, {x: 40, y: Math.min(250, viewport.height - 20)});
    assert.deepEqual(await location(page), {id: 'roadmap', index: 2});
    await page.locator('#roadmap .story-stage').evaluate(node => {node.scrollTop = node.scrollHeight;});
    await swipe(page, cdp, 1, {x: 40, y: Math.min(250, viewport.height - 20)});
    assert.deepEqual(await location(page), {id: 'roadmap', index: 3});
  }
  // Overflow is owned by its pane for the whole gesture, including its tail.
  await jump(page, 'roadmap', 2);
  await page.evaluate(() => {
    const pane = document.createElement('div');pane.id = 'navigation-overflow-probe';pane.style.cssText = 'position:fixed;z-index:30;left:20px;top:140px;width:180px;height:120px;overflow:auto;overscroll-behavior:contain;background:white';
    pane.innerHTML = '<div style="height:180px">Scrollable content</div>';document.body.append(pane);
  });
  const point = {x: 70, y: 200};
  if (touch) await swipe(page, cdp, 1, point);else await wheel(page, 1, 10, point);
  assert.deepEqual(await location(page), {id: 'roadmap', index: 2});
  assert.ok(await page.locator('#navigation-overflow-probe').evaluate(node => node.scrollTop > 0));
  await page.locator('#navigation-overflow-probe').evaluate(node => {node.scrollTop = node.scrollHeight;});
  if (touch) await swipe(page, cdp, 1, point);else await wheel(page, 1, 1, point);
  assert.deepEqual(await location(page), {id: 'roadmap', index: 3});
  await page.locator('#navigation-overflow-probe').evaluate(node => node.remove());
  const takeawayCount = exhaustive ? await checkTakeaways(page, () => gesture(1)) : 0;
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1));
  assert.deepEqual(errors, []);
  await page.screenshot({path: path.join(resultsDir, `${viewport.width}x${viewport.height}-${touch ? 'touch' : 'wheel'}.png`)});
  console.log(JSON.stringify({viewport, touch, takeawayCount, roadmapOverflow, errors}));
  await context.close();
}
async function profile(browser, mobile) {
  const context = await browser.newContext({viewport: {width: mobile ? 390 : 1440, height: mobile ? 700 : 1000}, isMobile: mobile, hasTouch: mobile});
  const page = await context.newPage();await ready(page);
  await jump(page, 'pricing', 1);
  try {
    await page.waitForFunction(() => document.querySelector('#score-scatter-canvas').dataset.rendered === 'true');
  } catch (error) {
    console.log(JSON.stringify(await page.evaluate(() => ({profileFailure: true, y: scrollY,
      state: document.querySelector('#pricing .story-stage').dataset.storyState,
      loadStatus: document.querySelector('#pricing .section-load-status')?.textContent,
      series: scoreHistoryData?.series?.length, workerFailed: typeof scoreRasterWorkerFailed === 'undefined' ? null : scoreRasterWorkerFailed,
      progress: document.querySelector('#score-scatter-canvas').dataset.progress,
      task: typeof scoreRenderTask === 'undefined' ? null : scoreRenderTask && {size: scoreRenderTask.size, frame: scoreRenderTask.frame}}))));
    throw error;
  }
  await page.evaluate(() => {
    window.profileTasks = [];
    new PerformanceObserver(list => profileTasks.push(...list.getEntries().map(entry => entry.duration))).observe({type: 'longtask'});
  });
  const samples = await record(page, async () => {
    for (const index of [2, 3, 1, 2, 3, 1]) {
      await jump(page, 'pricing', index);await page.waitForTimeout(300);
    }
  });
  const tasks = await page.evaluate(() => profileTasks);
  const gaps = samples.slice(1).map((sample, index) => sample.t - samples[index].t).sort((a, b) => a - b);
  const result = {baseline, mobile, longTasks: tasks.length, longestTaskMs: Math.max(0, ...tasks),
    p95FrameGapMs: gaps[Math.floor(gaps.length * .95)], maxFrameGapMs: Math.max(...gaps)};
  console.log(JSON.stringify(result));await context.close();return result;
}
async function checkCanvasFallback(browser) {
  const context = await browser.newContext({viewport: {width: 1440, height: 1000}});
  const page = await context.newPage();
  await page.addInitScript(() => {window.Worker = undefined;});
  await ready(page);await jump(page, 'pricing', 1);
  await page.waitForFunction(() => document.querySelector('#score-scatter-canvas').dataset.rendered === 'true');
  const result = await page.evaluate(async () => {
    const canvas = document.querySelector('#score-scatter-canvas');
    const container = document.querySelector('#score-scatter');
    const previous = canvas.toDataURL();
    scoreRenderTask = null;
    const cancelled = drawScoreScatter();
    container.style.width = `${container.clientWidth - 20}px`;
    const replacement = drawScoreScatter();
    const retainedBitmap = canvas.toDataURL() === previous;
    let frames = 0;
    const tick = () => {frames++;if (scoreRenderTask.frame != null) requestAnimationFrame(tick);};
    requestAnimationFrame(tick);
    return {cancelled: await cancelled, rendered: await replacement, retainedBitmap, frames, worker: scoreRasterWorker};
  });
  assert.equal(result.cancelled, false);assert.equal(result.rendered, true);
  assert.equal(result.retainedBitmap, true);assert.ok(result.frames >= 1);assert.equal(result.worker, null);
  console.log(JSON.stringify({fallback: result}));await context.close();
}
async function checkLoadingRecovery(browser) {
  const context = await browser.newContext({viewport: {width: 390, height: 700}, isMobile: true, hasTouch: true});
  const page = await context.newPage();
  let fail = true;
  await page.route('**/climate-risk-housing-county-history.js', async route => {
    if (fail) await route.abort();else await route.continue();
  });
  await ready(page);await jump(page, 'pricing', 1);
  await page.locator('#pricing .section-load-status button').waitFor();
  const before = await location(page);
  fail = false;
  await page.locator('#pricing .section-load-status button').click();
  await page.waitForFunction(() => document.querySelector('#score-scatter-canvas').dataset.rendered === 'true');
  assert.deepEqual(await location(page), before);
  assert.equal(await page.locator('#pricing .section-load-status').count(), 0);
  console.log(JSON.stringify({loadingRecovery: true}));await context.close();
}
(async () => {
  fs.mkdirSync(resultsDir, {recursive: true});
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const browser = await chromium.launch({headless: true, ...(process.env.CHROME_PATH ? {executablePath: process.env.CHROME_PATH} : {})});
  try {
    if (!profileOnly && !renderingOnly) {
      for (const [viewport, touch, exhaustive] of [
        [{width: 1440, height: 1000}, false, true], [{width: 1024, height: 650}, false, false],
        [{width: 390, height: 844}, true, true], [{width: 390, height: 700}, true, false],
        [{width: 844, height: 390}, true, false], [{width: 390, height: 400}, true, false],
      ]) await checkViewport(browser, viewport, touch, exhaustive);
    }
    if (!profileOnly) {await checkCanvasFallback(browser);await checkLoadingRecovery(browser);}
    const results = [];
    for (const mobile of [false, true]) results.push(await profile(browser, mobile));
    fs.writeFileSync(path.join(resultsDir, baseline ? 'profile-before.json' : 'profile-after.json'), JSON.stringify(results, null, 2));
  } finally {await browser.close();server.close();}
})().catch(error => {console.error(error);server.close();process.exitCode = 1;});
