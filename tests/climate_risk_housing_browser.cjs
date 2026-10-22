// Run with NODE_PATH pointing to a Playwright installation.
const {chromium} = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const http = require('node:http');
const root = path.resolve(__dirname, '../output');
const shots = process.argv[2];
const server = http.createServer((req, res) => {
  const name = decodeURIComponent(new URL(req.url, 'http://localhost').pathname);
  const file = path.resolve(root, '.' + (name === '/' ? '/index.html' : name));
  if (!file.startsWith(root + path.sep)) { res.writeHead(403).end(); return; }
  fs.readFile(file, (error, data) => {
    if (error) { res.writeHead(404).end(); return; }
    res.setHeader('Content-Type', file.endsWith('.js') ? 'text/javascript' : file.endsWith('.css') ? 'text/css' : 'text/html');
    res.end(data);
  });
});
async function frame(page, id, state) {
  await page.evaluate(({id, state}) => {
    cancelStoryNavigation();
    const section = document.getElementById(id);
    const index = storyConfigForSection(id).findIndex(step => step.state === state);
    window.scrollTo({top: section.offsetTop + index * storyViewportHeight, behavior: 'instant'});
    updateStoryFromScroll();
  }, {id, state});
  await page.waitForFunction(({id, state}) => document.querySelector(`#${id} .story-stage`).dataset.storyState === state, {id, state});
  await page.waitForTimeout(350);
}
async function shot(page, name) {
  if (shots) { fs.mkdirSync(shots, {recursive: true}); await page.screenshot({path: path.join(shots, name + '.png')}); }
}
async function checkDirectCountyReturn(page, button) {
  await page.evaluate(() => {
    window.returnStates = [];
    window.returnObserver = new MutationObserver(records => {
      window.returnStates.push(...records.map(record => record.oldValue));
    });
    window.returnObserver.observe(document.querySelector('#playbook .story-stage'), {attributes: true, attributeFilter: ['data-story-state'], attributeOldValue: true});
  });
  const before = await page.locator('#playbook .story-stage').getAttribute('data-story-state');
  await page.locator(button).click();
  assert.equal(await page.locator('#playbook .story-stage').getAttribute('data-story-state'), 'search');
  await page.waitForTimeout(350);
  const states = await page.evaluate(() => {
    window.returnObserver.disconnect();
    return [...window.returnStates, document.querySelector('#playbook .story-stage').dataset.storyState];
  });
  assert.deepEqual([...new Set(states)], [before, 'search'], 'return visits only the starting frame and county selection');
}
async function checkMapToolbar(page, selector) {
  const map = await page.locator(selector).boundingBox();
  const controls = page.locator(selector).locator('..').locator('.map-view-controls, .playbook-map-controls');
  const toolbar = await controls.boundingBox();
  assert.ok(toolbar.y + toolbar.height <= map.y, `${selector} controls sit above the map`);
  assert.ok(toolbar.x >= map.x && toolbar.x + toolbar.width <= map.x + map.width + 1, `${selector} controls fit the map width`);
}
async function mapPoint(page, selector, fips) {
  return page.evaluate(({selector, fips}) => {
    const svg = document.querySelector(selector);
    const feature = DATA.geojson.features.find(d => d.properties.fips === fips);
    const {width, height} = svg.viewBox.baseVal;
    const projection = d3.geoAlbersUsa().fitExtent([[8, 8], [width - 8, height - 8]], DATA.geojson);
    const [x, y] = d3.geoPath(projection).centroid(feature);
    const point = new DOMPoint(x, y).matrixTransform(svg.querySelector('g').getScreenCTM());
    return {x: point.x, y: point.y};
  }, {selector, fips});
}
async function check(page, mobile, browser) {
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.route('**/analytics/**', route => route.abort());
  await page.goto(`http://127.0.0.1:${server.address().port}`);
  await page.waitForFunction(() => typeof d3 !== 'undefined' && document.querySelector('.story-stage'));
  assert.equal(await page.locator('#t-hero-h1').innerText(), 'Are Climate Risks Priced Into Housing Markets?');
  assert.equal(await page.locator('#t-hero-dek').innerText(), 'Climate change leads to more severe, damaging weather events. What does this mean to housing in your area?');
  await shot(page, `intro-${mobile ? 'mobile' : 'desktop'}`);
  assert.equal(await page.locator('.roadmap-hint, .playbook-risk-equation-result').count(), 0);
  await frame(page, 'playbook', 'search');
  await page.waitForFunction(() => playbookSectionRendered);
  await frame(page, 'playbook', 'search');
  const point = await mapPoint(page, '#county-selection-map', '08031');
  await page.mouse.click(point.x, point.y);
  assert.equal(await page.evaluate(() => selectedCountyFips), '08031');
  assert.equal(await page.locator('#playbook .story-stage').getAttribute('data-story-state'), 'search');
  assert.equal(await page.locator('#playbook-map-hint').innerText(), 'Denver County, CO selected.');
  const selectionText = await page.locator('#playbook-map-hint').boundingBox();
  const continueButton = await page.locator('#playbook-selection-continue').boundingBox();
  assert.ok(continueButton.x >= selectionText.x + selectionText.width, 'Continue sits to the right of the selection');
  assert.ok(Math.abs(continueButton.y + continueButton.height / 2 - selectionText.y - selectionText.height / 2) < 2, 'selection and Continue share a horizontal centerline');
  assert.equal(await page.locator('#playbook-selection-continue').isVisible(), true);
  await page.waitForTimeout(500);
  await page.locator('#playbook-map-zoom-toggle').click();
  await page.waitForTimeout(280);
  assert.ok(await page.evaluate(() => d3.zoomTransform(document.querySelector('#county-selection-map')).k > 1));
  await page.locator('#playbook-map-recenter').click();
  await page.waitForTimeout(280);
  await checkMapToolbar(page, '#county-selection-map');
  assert.equal(await page.evaluate(() => d3.zoomTransform(document.querySelector('#county-selection-map')).k), 1);
  const map = await page.locator('#county-selection-map').boundingBox();
  const cursor = {x: map.x + map.width * .72, y: map.y + map.height * .55};
  if (mobile) {
    // Native Chromium touch input exercises D3's two-finger pinch handler.
    const cdp = await page.context().newCDPSession(page);
    const touches = distance => [
      {x: cursor.x - distance, y: cursor.y, id: 1},
      {x: cursor.x + distance, y: cursor.y, id: 2},
    ];
    await cdp.send('Input.dispatchTouchEvent', {type: 'touchStart', touchPoints: touches(18)});
    await cdp.send('Input.dispatchTouchEvent', {type: 'touchMove', touchPoints: touches(36)});
    await cdp.send('Input.dispatchTouchEvent', {type: 'touchEnd', touchPoints: []});
  } else {
    await page.mouse.move(cursor.x, cursor.y);
    await page.mouse.wheel(0, -300);
  }
  await page.waitForTimeout(280);
  const transform = await page.evaluate(() => ({...d3.zoomTransform(document.querySelector('#county-selection-map'))}));
  assert.ok(transform.k > 1, 'wheel or pinch zooms away from the selected county');
  assert.ok(Math.abs(-transform.x / (transform.k - 1) - (cursor.x - map.x)) < 4, 'zoom centers on input position');
  assert.equal(await page.evaluate(() => selectedCountyFips), '08031');
  await page.locator('#playbook-map-recenter').click();
  await page.waitForTimeout(280);
  await shot(page, `county-search-${mobile ? 'mobile' : 'desktop'}`);
  await page.locator('#county-search').fill('08031');
  await page.locator('#county-results > div').first().click();
  assert.equal(await page.locator('#playbook .story-stage').getAttribute('data-story-state'), 'search');
  assert.equal(await page.locator('#playbook-map-hint').innerText(), 'Denver County, CO selected.');
  await page.locator('#playbook-selection-continue').click();
  await page.waitForFunction(() => document.querySelector('#playbook .story-stage').dataset.storyState === 'history-map');
  await page.waitForTimeout(500);
  assert.match(await page.locator('#playbook-back-to-search').innerText(), /Back to county selection/);
  assert.match(await page.locator('#t-playbook-history-title').innerText(), /Denver County's Median PPSF YoY over Time/);
  assert.deepEqual(await page.locator('#playbook-ppsf-history .axis-label').allTextContents(), ['Median PPSF YoY', 'Year']);
  assert.equal(await page.locator('#playbook-history-legend').isVisible(), false);
  assert.equal(await page.locator('#playbook-profile-map').getAttribute('data-map-gestures'), null);
  assert.equal(await page.locator('#playbook-profile-map').getAttribute('data-map-locked'), 'true');
  await shot(page, `performance-${mobile ? 'mobile' : 'desktop'}`);
  await frame(page, 'playbook', 'history-events');
  assert.equal(await page.locator('#playbook-history-legend').isVisible(), false);
  await frame(page, 'playbook', 'history-compare');
  assert.equal(await page.locator('#playbook-history-legend').isVisible(), true);
  await frame(page, 'playbook', 'history-outlook');
  const fontSizes = await page.evaluate(() => ({heading: parseFloat(getComputedStyle(document.querySelector('.playbook-factor-column h4')).fontSize), card: parseFloat(getComputedStyle(document.querySelector('.playbook-warning-row')).fontSize)}));
  assert.ok(fontSizes.heading >= fontSizes.card);
  await shot(page, `makeup-${mobile ? 'mobile' : 'desktop'}`);
  const missing = await page.evaluate(() => {
    const county = playbookCountyByFips.get(selectedCountyFips);
    const saved = DATA.playbook.monthlyHistoryValuesByFips[county.fips];
    DATA.playbook.monthlyHistoryValuesByFips[county.fips] = [];
    renderPlaybookOutlook(county);
    const metrics = mostImportantFeatureMetrics(county.riskRating);
    const score = playbookScorecard(county, {subgroupName: 'Strong Overperformer'}, metrics, 'Low');
    return {fips:county.fips, saved, score};
  });
  assert.match(missing.score, /No information/);
  assert.match(missing.score, /Moderate Risk/);
  assert.equal(await page.locator('#playbook-event-commentary').innerText(), 'Insufficient housing data available for selected county to comment on local housing market performance.');
  const empty = await page.locator('#playbook-event-commentary').evaluate(el => ({font:parseFloat(getComputedStyle(el).fontSize),align:getComputedStyle(el).alignItems,justify:getComputedStyle(el).justifyContent}));
  assert.ok(empty.font >= 18);
  assert.equal(empty.align, 'center');
  assert.equal(empty.justify, 'center');
  const cardBox = await page.locator('#playbook .story-stage > .panel').boundingBox();
  const messageBox = await page.locator('#playbook-event-commentary').boundingBox();
  assert.ok(Math.abs(messageBox.y + messageBox.height / 2 - cardBox.y - cardBox.height / 2) < 80, 'missing makeup is centered in the full card');
  const emptyScroll = await page.locator('#playbook-event-commentary').evaluate(el => {
    const nodes = [];
    for (let node = el; node && !node.classList.contains('story-stage'); node = node.parentElement) {
      const before = node.getBoundingClientRect().y;
      node.scrollTop = 100;
      nodes.push({scroll: node.scrollTop, shift: node.getBoundingClientRect().y - before, overflow: node.scrollHeight - node.clientHeight});
    }
    return nodes;
  });
  assert.ok(emptyScroll.every(node => node.scroll === 0 && Math.abs(node.shift) < 1 && node.overflow <= 1), `empty makeup stays fixed: ${JSON.stringify(emptyScroll)}`);
  await page.evaluate(() => {
    window.emptyMessagePositions = [];
    window.sampleEmptyMessage = true;
    const sample = () => {
      const range = document.createRange();
      range.selectNodeContents(document.querySelector('#playbook-event-commentary'));
      window.emptyMessagePositions.push(range.getBoundingClientRect().y);
      if (window.sampleEmptyMessage) requestAnimationFrame(sample);
    };
    sample();
  });
  const messageCenter = {x: messageBox.x + messageBox.width / 2, y: messageBox.y + messageBox.height / 2};
  const scrollMakeup = async direction => {
    if (mobile) {
      const cdp = await page.context().newCDPSession(page);
      await cdp.send('Input.dispatchTouchEvent', {type: 'touchStart', touchPoints: [{...messageCenter, id: 1}]});
      await cdp.send('Input.dispatchTouchEvent', {type: 'touchMove', touchPoints: [{x: messageCenter.x, y: messageCenter.y - direction * 70, id: 1}]});
      await cdp.send('Input.dispatchTouchEvent', {type: 'touchEnd', touchPoints: []});
      await cdp.detach();
    } else {
      await page.mouse.move(messageCenter.x, messageCenter.y);
      await page.mouse.wheel(0, direction * 150);
    }
    await page.waitForTimeout(500);
  };
  await scrollMakeup(1);
  assert.equal(await page.locator('#playbook .story-stage').getAttribute('data-story-state'), 'takeaway-makeup');
  await scrollMakeup(-1);
  assert.equal(await page.locator('#playbook .story-stage').getAttribute('data-story-state'), 'history-outlook');
  const messageMovement = await page.evaluate(() => {
    window.sampleEmptyMessage = false;
    return Math.max(...window.emptyMessagePositions) - Math.min(...window.emptyMessagePositions);
  });
  assert.ok(messageMovement < 1, `missing message stays fixed during forward/backward gestures (${messageMovement}px)`);
  await shot(page, `missing-makeup-${mobile ? 'mobile' : 'desktop'}`);
  await checkDirectCountyReturn(page, '#playbook-back-to-search');
  await frame(page, 'playbook', 'conclusion');
  assert.match(await page.locator('#playbook-conclusion').innerText(), /County Makeup\s+No information/);
  await shot(page, `conclusion-${mobile ? 'mobile' : 'desktop'}`);
  await checkDirectCountyReturn(page, '#playbook-conclusion .playbook-back-button');
  await frame(page, 'playbook', 'conclusion');
  assert.equal(await page.locator('#playbook-conclusion .playbook-back-button').count(), 1, 'return button survives scorecard rerender');
  await page.evaluate(({fips,saved}) => { DATA.playbook.monthlyHistoryValuesByFips[fips] = saved; }, missing);
  await frame(page, 'events', 'card-overview');
  await page.waitForFunction(() => eventSectionRendered);
  assert.ok((await page.locator('#event-risk-pie').textContent()).includes('Number of counties affected by events'));
  await frame(page, 'events', 'card-short');
  await page.locator('#affected-county-count .info-tooltip-trigger').click();
  const tooltip = await page.locator('#tooltip').innerText();
  const risk = await page.evaluate(() => selectedRisk);
  await page.waitForFunction(risk => selectedRisk !== risk, risk);
  assert.equal(await page.locator('#tooltip').innerText(), tooltip);
  assert.equal(await page.locator('#tooltip').isVisible(), true);
  await page.locator('#t-events-h2').click();
  assert.equal(await page.locator('#tooltip').isVisible(), false);
  const firstPath = await page.locator('#affected-map path.county').first().elementHandle();
  await page.waitForTimeout(1300);
  assert.ok(await firstPath.evaluate(node => node.isConnected), 'risk animation reuses county paths');
  const wrapper = page.locator('#affected-map').locator('..');
  const countyPoint = await mapPoint(page, '#affected-map', '08031');
  await page.mouse.click(countyPoint.x, countyPoint.y);
  await wrapper.locator('.map-zoom-county').click();
  await page.waitForTimeout(260);
  assert.ok(await page.evaluate(() => d3.zoomTransform(document.querySelector('#affected-map')).k > 1));
  await wrapper.locator('.map-recenter').click();
  await page.waitForTimeout(260);
  await checkMapToolbar(page, '#affected-map');
  assert.equal(await page.evaluate(() => d3.zoomTransform(document.querySelector('#affected-map')).k), 1);
  await shot(page, `events-${mobile ? 'mobile' : 'desktop'}`);
  await frame(page, 'pricing-grouping', 'card-main');
  await page.waitForFunction(() => ratingSectionRendered);
  await checkMapToolbar(page, '#rating-map');
  await shot(page, `risk-groups-${mobile ? 'mobile' : 'desktop'}`);
  await frame(page, 'pricing-grouping', 'copy');
  const hiddenMapPoint = await mapPoint(page, '#rating-map', '08031');
  await page.mouse.move(hiddenMapPoint.x, hiddenMapPoint.y);
  assert.equal(await page.locator('#tooltip').isVisible(), false);
  assert.equal(await page.locator('#rating-map').evaluate(el => getComputedStyle(el).visibility), 'hidden');
  await shot(page, `risk-copy-${mobile ? 'mobile' : 'desktop'}`);

  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1));
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({mobile, fontSizes, inputZoom: transform.k, errors}));
}
(async () => {
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const browser = await chromium.launch({headless: true, ...(process.env.CHROME_PATH ? {executablePath: process.env.CHROME_PATH} : {})});
  try {
    for (const mobile of [false, true]) {
      const context = await browser.newContext({viewport: {width: mobile ? 390 : 1440, height: mobile ? 844 : 1000}, isMobile: mobile, hasTouch: mobile});
      await check(await context.newPage(), mobile, browser);
      await context.close();
    }
  } finally { await browser.close(); server.close(); }
})().catch(error => { console.error(error); server.close(); process.exitCode = 1; });
