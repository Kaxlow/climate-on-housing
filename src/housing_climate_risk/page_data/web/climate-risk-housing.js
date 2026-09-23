/* ============================================================
   TEXT — Every visible string on the page lives here.
   Edit any value below and rebuild to update the page.
   ============================================================ */
const TEXT = {
  pageFeedback: '<a href="https://docs.google.com/forms/d/e/1FAIpQLSeLPeN1D-bAfJLxSRhsbeAQwMIf3DBNm-nuqyVe30ZasESyxA/viewform" target="_blank" rel="noopener noreferrer">Share feedback on this page</a>',
  // ---- Hero ----
  heroH1: "Are Climate Risks Priced Into Housing Markets?",
  heroDek: "Climate change results in more severe weather events and natural disasters that cause substantial damage to properties and in extreme cases, devastate local communities. What does all this mean to you as a homeowner?",
  storyPrevLabel: "Prev",
  storyNextLabel: "Next",
  sourcesLabel: "Sources",
  informationTooltipLabel: "More information",
  completeMonthlyPlotTooltip: "Only counties with complete monthly Median PPSF YoY data throughout the displayed period are included in this plot.",

  // ---- Pricing section ----
  pricingH2: "To Begin: What Does Growth in Housing Markets Look Like Across the United States?",
  scatterTitle: "Median Price-Per-Square-Foot (PPSF) Year-Over-Year (YoY) by County",
  scatterSub: "Each line represents a county's monthly Median PPSF YoY over the latest 10 complete calendar years.",
  scatterFootnotesTooltip: "Values beyond the 10th–90th percentile are capped to keep extreme observations from compressing the visible pattern.",
  pricingScoreScatterTakeaway: "<span class=\"takeaway-section\">From county-level median house price growth over the last 10 years, there is significant variation and there doesn't seem to be a clear pattern.</span><span class=\"takeaway-section\">However, the impact of climate change is uneven across the country, so looking from a climate angle might reveal a more meaningful pattern.</span>",
  pricingGroupingSubtitle: "A Climate Perspective: What Happens When Counties are Grouped by Climate Risk?",
  pricingNriPlaceholder: "The <a href='https://www.fema.gov/flood-maps/products-tools/national-risk-index' target='_blank' rel='noopener'>FEMA National Risk Index (NRI)</a> serves as a measure of climate-related risk exposure. It summarizes a county's expected annual loss, social vulnerability, and community resilience across natural hazards. Counties are assigned a risk rating along a scale from \"Very Low\" to \"Very High\".",
  pricingCardTitle: "Median PPSF YoY by Climate Risk",
  pricingCardText: "House price growth of counties grouped by their FEMA National Risk Index (NRI) risk rating.",
  pricingTakeaway: "<span class=\"takeaway-section\">A pattern now emerges: The greater the climate risk, the weaker the housing price growth.</span><span class=\"takeaway-section\">Additionally, higher risk groups show a narrower band of housing price growth rates.</span>",
  pricingQuestion: "There is clearly a relationship between climate risk and housing market growth. How do markets respond when an extreme climate event occurs?",
  pricingSources: 'Sources: <a href="https://www.redfin.com/news/data-center/downloads/" target="_blank" rel="noopener">Redfin monthly county Housing Market Tracker</a>; <a href="https://hazards.fema.gov/nri/data-resources" target="_blank" rel="noopener">FEMA National Risk Index county data</a>.',

  playbookTopFactorsTitle: "Top Factors for {county}",
  playbookFrameTitles: {search: "Select a County", "history-map": "County's Past Housing Market Performance", "history-events": "County Performance in the Context of Past Events", "history-compare": "County Performance Relative to Risk Group", "history-outlook": "County Market Resilience Assessment"},
  playbookInsufficientHistory: "Insufficient housing data available for selected county to comment on local housing market performance.",
  playbookFactorContext: {
    net_earnings_per_capita_usd: {
      higher: "Areas with higher earnings income may have less affordable housing markets that deter buyers, slowing price growth.",
      lower: "Areas with lower earnings income may have more affordable housing markets that attract buyers, supporting faster price growth.",
      expectedRhoSign: -1,
    },
    age_65_plus_pct: {
      higher: "A larger older population may move less frequently, reducing housing market activity and slowing price growth.",
      lower: "A smaller older population may mean more residential mobility and housing demand, supporting faster price growth.",
      expectedRhoSign: -1,
    },
    dividends_interest_rent_per_capita_usd: {
      higher: "Higher investment income may signal wealthier markets with higher starting prices and slower percentage price growth.",
      lower: "Lower investment income may signal lower starting prices and more room for percentage price growth.",
      expectedRhoSign: -1,
    },
    transfer_receipts_per_capita_usd: {
      higher: "Greater reliance on benefits may signal economic vulnerability that weakens housing demand and price growth.",
      lower: "Less reliance on benefits may signal less economic vulnerability, supporting housing demand and price growth.",
      expectedRhoSign: -1,
    },
    property_taxes_pct_income: {
      higher: "A higher property-tax burden may constrain buyers' budgets and slow housing price growth.",
      lower: "A lower property-tax burden may improve affordability and support housing demand and price growth.",
      expectedRhoSign: -1,
    },
    utilities_pct_income: {
      higher: "A higher utility-cost share may reflect lower incomes and home prices; affordable starting prices can attract buyers and support faster growth.",
      lower: "A lower utility-cost share may reflect higher incomes and home prices, leaving less room for percentage price growth.",
      expectedRhoSign: 1,
    },
    owner_cost_burden_30pct_plus_pct: {
      higher: "More cost-burdened households may indicate financial vulnerability that constrains housing demand and price growth.",
      lower: "Fewer cost-burdened households may indicate greater financial capacity, supporting housing demand and price growth.",
      expectedRhoSign: -1,
    },
    net_migration_rate_pct: {
      higher: "More net arrivals can increase housing demand and support faster price growth.",
      lower: "Fewer net arrivals or more departures can weaken housing demand and slow price growth.",
      expectedRhoSign: 1,
    },
    disability_pct: {
      higher: "A larger population share with disabilities may coincide with lower incomes and home prices; affordable starting prices can attract buyers and support faster growth.",
      lower: "A smaller population share with disabilities may coincide with higher incomes and home prices, leaving less room for percentage price growth.",
      expectedRhoSign: 1,
    },
  },
  // ---- Events section ----
  eventsH2: "What Happened to Housing Markets in Counties where Extreme Climate Events Occurred?",
  eventsCopy: "Here's a look at housing markets that had past events happen, taken from <a href='https://www.fema.gov/disaster/declarations' target='_blank' rel='noopener'>FEMA disaster declarations</a> and <span id=\"events-noaa-term\"><a href='https://www.ncei.noaa.gov/stormevents/' target='_blank' rel='noopener'>NOAA billion-dollar storm events</a></span> over the last 10 years. The next charts will show how these markets performed in the year before the event and in the years after.",
  eventsNoaaTooltip: "Events whose total recorded damage is greater than or equal to a billion dollars",
  eventsCardTitle: "Median PPSF YoY Around Extreme Climate Events",
  eventsOverviewTitle: "Over the last 10 years:",
  eventsOverviewTakeaway: "Extreme climate events have affected counties across the board, even the low risk ones.",
  eventsOverviewQuestion: "How do the risk groups differ in response to events?",
  eventsBeforeContext: "First, look at housing price growth before an event:",
  eventsAfterContext: "Now, look at housing price growth after an event:",
  eventsBeforeTitle: "Median PPSF YoY 1 year before event",
  eventsBeforeTakeaway: "Before an event, housing price growth seems stable.",
  eventsAfterQuestion: "What does it look like after an event?",
  eventsShortTitle: "Median PPSF YoY in 3 years after event",
  eventsLongTitle: "Median PPSF YoY in 5 years after event",
  eventHorizonYears: {A: "3", B: "5"},
  riskSidebarLabel: "Risk rating",
  eventWindowATakeaway: "Past the 2-year mark post-event, house price growth momentum diverges across the different risk bands. Growth weakening is more pronounced in higher risk groups.",
  eventFuturePrompt: "What does it look like further into the future?",
  eventWindowBTakeaway: "Around the 4-year mark post-event, house price growth across the different risk bands begin to converge to the same level. It appears that the event’s impact fades from view eventually.",
  eventsTakeaway: "In higher-risk counties, there is a time lag after an event before house price growth declines significantly. Homeowners who made it through the period of weakness then experienced some subsequent recovery.",
  eventsVariationQuestion: "Looking at the bands representing the different risk groups, there is a wide range of growth rates. What is behind this variation?",
  eventsSources: 'Sources: <a href="https://www.redfin.com/news/data-center/downloads/" target="_blank" rel="noopener">Redfin monthly county Housing Market Tracker</a>; <a href="https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries" target="_blank" rel="noopener">OpenFEMA Disaster Declarations Summaries</a>; <a href="https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/" target="_blank" rel="noopener">NOAA Storm Events details</a>; <a href="https://hazards.fema.gov/nri/data-resources" target="_blank" rel="noopener">FEMA National Risk Index county data</a>.',

  // ---- Features section ----
  featuresH2: "What Factors Influence a County's Housing Market Performance within Risk Groups?",
  featuresCopy: "From the vast data on counties, a model to understand counties' housing market <span id=\"feature-performance-term\">performance</span> can be built upon the most significant data types.",
  featurePerformanceTooltip: "Performance is defined in terms of different Median PPSF YoY over the same time period.",
  featureSidebarLabel: "Risk rating",
  featureLineTitle: "Median PPSF YoY around events",
  featureScatterTitle: "Median PPSF YoY around events vs. {feature}",
  featureOutcomeTerm: "Median PPSF YoY around events",
  featureOutcomeTooltip: "For each county, take the median across months -12 through event start and months 1–36 after event end. Spearman correlation compares this county-level median with the county feature value.\n\nIf a county had multiple complete event trajectories, it is represented by a single trajectory built from the medians across the individual trajectories at each relative month.",
  featureFrame1Title: "Which data types matter most to {risk} Risk counties?",
  featureFrame2Title: "What types of counties exist within the {risk} Risk group?",
  featureFrame3Title: "What factors define {subgroup} in the {risk} Risk group?",
  featureGroupByCategory: "Group by Category",
  featureOrderBySignificance: "Order by Significance",
  featureClickHint: "Select a type of data to reveal its relationship with Median PPSF YoY around events.",
  featureStrongTooltip: "Selected factor: |ρ| ≥ 0.30, or next-highest |ρ| to reach at least three factors.",
  featureSourcesTopic: "Sources",
  featureRankingTopic: "Ranking",
  featureSourcesNote: '<a href="https://www.redfin.com/news/data-center/downloads/" target="_blank" rel="noopener">Redfin monthly county Housing Market Tracker</a>; <a href="https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries" target="_blank" rel="noopener">OpenFEMA Disaster Declarations Summaries</a>; <a href="https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/" target="_blank" rel="noopener">NOAA Storm Events details</a>; <a href="https://hazards.fema.gov/nri/data-resources" target="_blank" rel="noopener">FEMA National Risk Index county data</a>; <a href="https://api.census.gov/data.html" target="_blank" rel="noopener">Census ACS 5-year county tables</a>; <a href="https://www.statsamerica.org/downloads/default.aspx" target="_blank" rel="noopener">StatsAmerica BEA Personal Income and Components of Population Change</a>.',
  featureRankingNote: "Data types are ranked by descending absolute Spearman correlation (|ρ|). Select all factors with |ρ| ≥ 0.30; if fewer than three qualify, add the next-highest correlations to reach three. Economic and demographic features use ten-year county averages.",
  featureCategories: {
    Economic: "Economic Data",
    Demographic: "Demographic Data",
  },
  featureLabels: {
    net_earnings_per_capita_usd: "Net Earnings per Resident (Place of Residence)",
    dividends_interest_rent_per_capita_usd: "Dividends, Interest & Rent per Resident",
    transfer_receipts_per_capita_usd: "Transfer Receipts per Resident",
    homeowners_insurance_pct_income: "Home Insurance as % of Median Household Income",
    property_taxes_pct_income: "Property Tax as % of Median Household Income",
    utilities_pct_income: "Utilities Cost as % of Median Household Income",
    owner_cost_burden_30pct_plus_pct: "Share of Cost-Burdened Households",
    unemployment_rate_pct: "Unemployment Rate",
    net_migration_rate_pct: "Net Migration Rate",
    age_65_plus_pct: "Aged Population Share",
    communication_barrier_pct: "Share of Population with Language Barrier",
    disability_pct: "Share of Population with Disabled Status",
  },
  featureRelationship: "{feature} has a {strength} {direction} relationship with Median PPSF YoY around events.",
  featureRelationshipStrength: {weak: "weak", moderate: "moderate", strong: "strong"},
  featureRelationshipDirection: {positive: "positive", negative: "negative"},
  featureTakeaway: "Housing price growth is jointly influenced by NRI Risk Rating and other attributes.",
  subgroupNamesFour: ["Strong Overperformers", "Mild Overperformers", "Mild Underperformers", "Strong Underperformers"],
  subgroupNamesThree: ["Overperformers", "Average Performers", "Underperformers"],
  subgroupShortNamesFour: ["Strong", "Mildly Strong", "Mildly Weak", "Weak"],
  subgroupShortNamesThree: ["Strong", "Average", "Weak"],
  subgroupFallback: "Subgroup {number}",
  subgroupCount: "{count} counties",
  featureDistributionFallback: "No valid feature correlations are available for this risk group.",
  featureDistributionTitle: "County Distribution",
  featureDistributionOutlierTooltip: "Outlier values beyond 1.5 times the interquartile range are not shown in this plot.",
  featureDistributionVeryHighTooltip: "All available values are shown for the Very High Risk group because this group has relatively few counties.",
  featureDistributionPrevious: "Previous feature",
  featureDistributionNext: "Next feature",
  featureDistributionSelected: "Selected subgroup",
  featureDistributionLevel: {higher: "higher", lower: "lower"},
  featureSubgroupTakeaway: "Counties with <b>{level} {feature}</b> tend to have <b>poorer housing price growth</b> in the <b>{risk} Risk</b> group.",
  featureSubgroupSummaryIntro: "Compared with other {risk} Risk counties, counties in {subgroup} tend to have:",
  featurePeerRelation: {
    higher: "higher",
    lower: "lower",
    close: "average",
  },
  featureSubgroupSummaryUnavailable: "A feature summary is not available for this subgroup.",
  featureGroupMedianLegend: "{risk} group median",
  featureEventMarker: "Event",
  featureXAxis: "Months from event",
  featureYAxis: "Median PPSF YoY",
  featureScatterYAxis: "Median PPSF YoY",

  // ---- Playbook section ----
  playbookH2: "Climate Playbook: {frameTitle}",
  playbookSearchPlaceholder: "Search for a county by name, state, or FIPS…",
  playbookInsufficientFeatureData: "Insufficient feature data available for {county}.",
  playbookInsufficientEventWindowData: "The most important data types for {county} could not be determined because no housing observations were available in the event-window analysis period.",
  playbookInsufficientFeatureValue: "Insufficient data",
  playbookHistoryTitle: "Monthly Median PPSF YoY Over the Past 10 Years",
  playbookFeatureTitle: "Most important data types for {risk} Risk counties",
  playbookSubgroupFeatureTitle: "County Traits",
  playbookSubgroup: "{county}'s house price growth rate around extreme climate events makes it a {subgroup} among {risk} Risk counties.",
  playbookPastEventsTitle: "Past extreme weather events",
  playbookNoPastEvents: "No <span class=\"playbook-event-definition\">extreme weather events</span> occurred during this 10-year period.",
  playbookEventDefinition: "FEMA disaster declarations and NOAA billion-dollar storm events.",
  playbookZoomOut: "Zoom out",
  playbookZoomIn: "Zoom to county",
  playbookEventLegend: "Extreme event period",
  playbookMissingDataLegend: "Missing county data",
  playbookSeriesLegend: "County Median PPSF YoY",
  playbookRiskSeriesLegend: "{risk} Risk median and IQR",
  playbookRiskUnavailable: "NRI risk rating unavailable",
  playbookHistoryComparison: "Over the past 10 years, {county}'s Median PPSF YoY was in the <b>{relation}</b> within the {risk} Risk group. The county's ten-year median was {countyMedian}, compared with the risk group's typical monthly median of {groupMedian}.",
  playbookHistoryComparisonUnavailable: "There is insufficient housing or climate-risk data to compare this county with its risk group over the past 10 years.",
  playbookInsufficientPerformance: "{county} had insufficient data so its housing market performance could not be reliably determined.",
  playbookWarningIntroNoEvents: "From {county}'s housing price growth performance relative to its peers in the {risk} Risk group, it is a {subgroup} within the group. &rarr; If an event were to happen, watch these factors:",
  playbookWarningIntroWithEvents: "From {county}'s housing price growth performance{eventContext} relative to its peers in the {risk} Risk group, it is a {subgroup} within the group. &rarr; When an event happens, watch these factors:",
  playbookWarningTakeaway: "When these factors start trending in the directions shown above, it's a sign that the county's housing price growth would begin to decline.",
  playbookOutlookInsufficientRisk: "There is insufficient data about this county to identify warning signs of its future housing market performance.",
  playbookOutlookInsufficientFeatures: "There is insufficient feature data to identify reliable indicators of this county's future housing market performance.",
  playbookTakeaways: {
    noEvents: "<strong>{county} had no extreme climate events in the last 10 years.</strong><span class=\"playbook-summary-detail\">Given its profile as {countyContext}, its Median PPSF YoY would be expected to {expectation} after an extreme climate event.</span>",
    eventAlignmentSummary: "<strong>{county}'s post-event change {riskAlignment} with the {risk} Risk expectation.</strong><span class=\"playbook-summary-detail\">The county {countyChange}; its risk group typically {groupChange}.<br>Its Median PPSF YoY level relative to its risk group {subgroupAlignment} with the {subgroup} within the {risk} Risk category.<br>The percentage-point change covers one year before each event began through three years after it ended.</span>",
    eventAlignmentWithoutSubgroup: "<strong>{county}'s post-event change {riskAlignment} with the {risk} Risk expectation.</strong><span class=\"playbook-summary-detail\">The county {countyChange}; its risk group typically {groupChange}.<br>Its relative performance within its risk group could not be assessed because sufficient feature data were unavailable.<br>The percentage-point change covers one year before each event began through three years after it ended.</span>",
    volatileEventSummary: "<strong>{county}'s Median PPSF YoY was too volatile to assess whether its post-event change aligned with the {risk} Risk expectation.</strong><span class=\"playbook-summary-detail\">Its level relative to its risk group also could not be compared meaningfully with the {subgroupContext}.</span>",
    insufficientHistory: "{county} had insufficient housing data from one year before its events through three years after they ended to compare with its risk group.",
    groupExpectation: "By year 3, counties with {risk} climate risk typically {groupBehavior} versus their pre-event-year level.",
    groupBehaviorChange: "{direction} by about {magnitude} percentage points",
    groupBehaviorFlat: "remain broadly steady ({magnitude} percentage-point change)",
    unavailableExpectation: "A group-level post-event expectation is unavailable for this NRI rating.",
  },
  playbookTakeawayTerms: {
    event: "event",
    events: "events",
    declined: "declined",
    decline: "decline",
    increased: "increased",
    increase: "increase",
    unchanged: "was broadly unchanged",
    aligned: "broadly in line",
    notAligned: "not clearly in line",
    down: "down",
    up: "up",
    littleChanged: "little changed",
    insufficient: "insufficient pre/post data",
  },
  playbookSources: 'Sources: <a href="https://www.redfin.com/news/data-center/downloads/" target="_blank" rel="noopener">Redfin monthly county Housing Market Tracker</a>; <a href="https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries" target="_blank" rel="noopener">OpenFEMA Disaster Declarations Summaries</a>; <a href="https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/" target="_blank" rel="noopener">NOAA Storm Events details</a>; <a href="https://hazards.fema.gov/nri/data-resources" target="_blank" rel="noopener">FEMA National Risk Index county data</a>; <a href="https://api.census.gov/data.html" target="_blank" rel="noopener">Census ACS 5-year county tables</a>; <a href="https://www.statsamerica.org/downloads/default.aspx" target="_blank" rel="noopener">StatsAmerica BEA Personal Income and Components of Population Change</a>.',
  riskImpacts: {
    "Very Low": "Counties with Very Low climate risk tend to maintain steady house price growth around climate events, with minimal disruption to market momentum.",
    "Low": "Counties with Low climate risk typically see modest softening of house price growth about two years after the event, but generally recover within three years.",
    "Medium": "Counties with Medium climate risk experience noticeable softening of house price growth around the two-year mark after the event.",
    "High": "Counties with High climate risk see significant deceleration in house price growth following an event, with the decline primarily occurring 18-24 months after event end.",
    "Very High": "Counties with Very High climate risk face substantial impacts on house price growth, with softening trends that can persist for several years after events.",
  },
};

/* Populate all text elements from the TEXT object above.
   Each entry maps a TEXT key to an element id. Uses innerHTML
   so sources with <a>/<code> tags render correctly. */
function hydrateText() {
  const map = {
    heroH1: "t-hero-h1",
    heroDek: "t-hero-dek",
    pricingH2: "t-pricing-h2",
    scatterTitle: "t-scatter-title",
    scatterSub: "t-scatter-sub",
    pricingScoreScatterTakeaway: "score-scatter-takeaway",
    pricingTakeaway: "pricing-takeaway",
    pricingQuestion: "pricing-question",
    eventsOverviewQuestion: "event-overview-question",
    eventsVariationQuestion: "event-variation-question",
    pricingGroupingSubtitle: "t-pricing-grouping-subtitle",
    pricingNriPlaceholder: "t-pricing-nri-placeholder",
    pricingCardTitle: "t-pricing-card-title",
    pricingCardText: "t-pricing-card-text",
    pricingSources: "t-pricing-sources",
    eventsH2: "t-events-h2",
    eventsCopy: "t-events-copy",
    eventsCardTitle: "t-events-card-title",
    eventsOverviewTakeaway: "event-overview-takeaway",
    eventsBeforeTakeaway: "event-before-takeaway",
    eventsAfterQuestion: "event-after-question",
    eventFuturePrompt: "event-future-prompt",
    eventsTakeaway: "event-takeaway",
    eventsSources: "t-events-sources",
    featuresH2: "t-features-h2",
    featuresCopy: "t-features-copy",
    featureTakeaway: "feature-takeaway",
    playbookH2: "t-playbook-h2",
    playbookHistoryTitle: "t-playbook-history-title",
    playbookSources: "t-playbook-sources",
    pageFeedback: "t-page-feedback",
  };
  for (const [key, id] of Object.entries(map)) {
    const el = document.getElementById(id);
    if (el && TEXT[key] != null) {
      el.innerHTML = TEXT[key];
      el.classList.toggle("segmented", Boolean(el.querySelector(".takeaway-section")));
    }
  }
  document.getElementById("county-search").placeholder = TEXT.playbookSearchPlaceholder;
  document.getElementById("story-prev").setAttribute("aria-label", TEXT.storyPrevLabel);
  document.getElementById("story-prev").title = TEXT.storyPrevLabel;
  document.getElementById("story-next").setAttribute("aria-label", TEXT.storyNextLabel);
  document.getElementById("story-next").title = TEXT.storyNextLabel;
  const scatterSub = document.getElementById("t-scatter-sub");
  scatterSub?.append(" ", makeInfoButton(TEXT.scatterFootnotesTooltip, {label: TEXT.informationTooltipLabel}));
  setCompleteMonthlyPlotTitle("#t-pricing-card-title", TEXT.pricingCardTitle);
  const performanceTerm = document.getElementById("feature-performance-term");
  performanceTerm?.after(makeInfoButton(TEXT.featurePerformanceTooltip, {label: TEXT.informationTooltipLabel}));
  const noaaTerm = document.getElementById("events-noaa-term");
  noaaTerm?.after(" ", makeInfoButton(TEXT.eventsNoaaTooltip, {label: TEXT.informationTooltipLabel}));
  condenseSourceDisclosures();
}

const HAZARD_ICONS = {
  overall: "\u{1F30E}",
};

const DATA = JSON.parse(document.getElementById("page-data").textContent);
const RISK_ORDER = ["Very Low", "Low", "Medium", "High", "Very High"];
const RATING_SEQUENCE_FRAMES = RISK_ORDER.map(risk => ({risk}));
const RISK_SEQUENCE_INTERVAL = 1200;
const RISK_COLORS = {"Very Low":"#16803c","Low":"#79b851","Medium":"#e0b33b","High":"#df7d2f","Very High":"#b42318"};
const COUNTY_LINE_COLOR = "#2456a6";
const FEATURE_HIGHER_LINE_COLOR = "#175d8f";
const FEATURE_LOWER_LINE_COLOR = "#8a4f7d";
const fmtPct = d3.format("+.1%");
const fmtShare = d3.format(".0%");
const fmtAxisPct = value => Math.abs(value) >= 10 ? `${value > 0 ? "+" : ""}${d3.format(".2s")(value * 100)}%` : fmtPct(value);
const fmtNum = d3.format(",.1f");
const fmtMoney = d3.format("$,.0f");
const parsePriceMonth = d3.utcParse("%Y-%m-%d");
const tooltip = d3.select("#tooltip");
let activeInfoTooltipTrigger = null;

function hideTooltip(force = false) {
  if (activeInfoTooltipTrigger && !force) return;
  tooltip.style("display", "none").style("visibility", "hidden").classed("persistent", false);
}

function closeInfoTooltip() {
  if (activeInfoTooltipTrigger) activeInfoTooltipTrigger.setAttribute("aria-expanded", "false");
  activeInfoTooltipTrigger = null;
  hideTooltip(true);
}

function showTooltip(event, content, {html = true, anchor = null} = {}) {
  if (activeInfoTooltipTrigger && anchor !== activeInfoTooltipTrigger) return;
  if (html) tooltip.html(content); else tooltip.text(content);
  tooltip.style("display", "block").style("visibility", "hidden");
  const anchorRect = anchor?.getBoundingClientRect();
  const originX = Number.isFinite(event?.clientX) ? event.clientX : (anchorRect?.right || 8);
  const originY = Number.isFinite(event?.clientY) ? event.clientY : (anchorRect?.top || 8);
  const node = tooltip.node();
  const width = node.offsetWidth, height = node.offsetHeight;
  const gap = 12, edge = 8;
  let left = originX + gap;
  let top = originY + gap;
  if (left + width > window.innerWidth - edge) left = originX - width - gap;
  if (top + height > window.innerHeight - edge) top = originY - height - gap;
  left = Math.max(edge, Math.min(left, window.innerWidth - width - edge));
  top = Math.max(edge, Math.min(top, window.innerHeight - height - edge));
  tooltip.style("left", `${left}px`).style("top", `${top}px`).style("visibility", "visible");
}

function attachInfoTooltip(element, content, {html = false} = {}) {
  element.setAttribute("aria-expanded", "false");
  element.addEventListener("click", event => {
    event.preventDefault();
    event.stopPropagation();
    if (activeInfoTooltipTrigger === element) {
      closeInfoTooltip();
      return;
    }
    closeInfoTooltip();
    activeInfoTooltipTrigger = element;
    element.setAttribute("aria-expanded", "true");
    tooltip.classed("persistent", true);
    showTooltip(event, content, {html, anchor: element});
  });
}

document.addEventListener("pointerdown", event => {
  if (!activeInfoTooltipTrigger) return;
  if (event.target === activeInfoTooltipTrigger || tooltip.node().contains(event.target)) return;
  closeInfoTooltip();
});
document.addEventListener("keydown", event => {
  if (event.key === "Escape") closeInfoTooltip();
});

function makeInfoButton(content, options = {}) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "info-tooltip-trigger";
  button.textContent = "?";
  button.setAttribute("aria-label", options.label || TEXT.informationTooltipLabel);
  attachInfoTooltip(button, content, options);
  return button;
}

function setCompleteMonthlyPlotTitle(target, content, {html = false} = {}) {
  const element = typeof target === "string" ? document.querySelector(target) : target;
  if (!element) return;
  element.innerHTML = "";
  const label = document.createElement("span");
  if (html) label.innerHTML = content;
  else label.textContent = content;
  const info = makeInfoButton(TEXT.completeMonthlyPlotTooltip, {
    label: TEXT.informationTooltipLabel,
  });
  info.classList.add("plot-completeness-info");
  element.append(label, info);
}

function condenseSourceDisclosures() {
  document.querySelectorAll(".panel > .sources:not(.playbook-footer), #t-playbook-sources").forEach(source => {
    const content = source.innerHTML.replace(/^\s*Sources:\s*/i, "");
    source.innerHTML = "";
    const label = document.createElement("span");
    label.textContent = TEXT.sourcesLabel;
    source.append(label, makeInfoButton(content, {html: true, label: `${TEXT.sourcesLabel}: ${TEXT.informationTooltipLabel}`}));
  });
}
const countyByFips = new Map(DATA.priceRisk.counties.map(d => [d.fips, d]));
let playbookCountyByFips = new Map();
let scoreHistoryData = null;
const deferredDataPromises = new Map();
let ratingSequenceIndex = 0;
let ratingSequenceTimer = null;
let ratingSequencePaused = false;
let scoreScatterState = null;
let scoreScatterRendered = false;
let scoreTooltipFrame = null;
let scoreRenderGeneration = 0;
let ratingSectionRendered = false;
let eventSectionRendered = false;
let featureSectionRendered = false;
let playbookSectionRendered = false;
// Event section state
let selectedRisk = "Very Low";
let riskTimer = null;
let riskAutoPaused = false;
let activeEventWindow = "overview"; // "overview", "before", "A", or "B"
let lastEventDomain = null;
let lastEventSource = null;
// Features section state
let selectedFeatureRisk = "Medium";
let selectedFeatureKey = null;
let selectedFeatureSubgroup = null;
let featureOrderMode = "category";
let selectedDistributionFeature = null;
let featureSubgroupSequenceTimer = null;
let featureSubgroupSequencePaused = false;
// Playbook state
let selectedCountyFips = null;
let playbookMapZoomed = false;
let playbookZoomBehavior = null;
let playbookMapTransform = d3.zoomIdentity;
let playbookSelectedTransform = d3.zoomIdentity;

function loadDeferredData(filename, globalName) {
  if (window[globalName]) return Promise.resolve(window[globalName]);
  if (deferredDataPromises.has(globalName)) return deferredDataPromises.get(globalName);
  const promise = new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = new URL(filename, document.baseURI).href;
    script.async = true;
    script.onload = () => {
      script.remove();
      if (window[globalName]) resolve(window[globalName]);
      else reject(new Error(`Deferred data file did not define ${globalName}`));
    };
    script.onerror = () => {
      script.remove();
      reject(new Error(`Unable to load deferred data file: ${filename}`));
    };
    document.head.appendChild(script);
  });
  const retryable = promise.catch(error => {
    deferredDataPromises.delete(globalName);
    throw error;
  });
  deferredDataPromises.set(globalName, retryable);
  return retryable;
}

async function ensureGeography() {
  if (DATA.geojson) return;
  const geography = await loadDeferredData("climate-risk-housing-geography.js", "CLIMATE_RISK_HOUSING_GEOGRAPHY");
  Object.assign(DATA, geography);
}

async function ensureEvents() {
  if (!DATA.eventWindows) DATA.eventWindows = await loadDeferredData("climate-risk-housing-events.js", "CLIMATE_RISK_HOUSING_EVENTS");
}

async function ensureFeatures() {
  if (!DATA.features) DATA.features = await loadDeferredData("climate-risk-housing-features.js", "CLIMATE_RISK_HOUSING_FEATURES");
}

function hazardLabel(key) { return DATA.priceRisk.hazards.find(h => h.key === key)?.label || key; }
function hazardCounty(county, key) { return county?.hazards?.[key] || {}; }
function capValue(value, domain) {
  if (value == null || !Number.isFinite(value)) return null;
  return Math.max(domain[0], Math.min(domain[1], value));
}
function robustDomain(values) {
  const valid = values.filter(v => v != null && Number.isFinite(v)).sort(d3.ascending);
  if (!valid.length) return [0, 1];
  const lo = d3.quantileSorted(valid, .01);
  const hi = d3.quantileSorted(valid, .99);
  return lo === hi ? [lo - 1, hi + 1] : [lo, hi];
}
function outlierFreeValues(values) {
  const sorted = values.filter(v => v != null && Number.isFinite(v)).sort(d3.ascending);
  if (!sorted.length) return sorted;
  const lo = d3.quantileSorted(sorted, .01);
  const hi = d3.quantileSorted(sorted, .99);
  return sorted.filter(v => v >= lo && v <= hi);
}
function colorScale(values, color, robust = false) {
  const valid = values.filter(v => v != null).sort(d3.ascending);
  const domain = robust ? robustDomain(valid) : d3.extent(valid);
  return d3.scaleSequentialSymlog(domain[0] === domain[1] ? [domain[0] - 1, domain[1] + 1] : domain, color).constant(.05);
}
function scaleLegendHtml(domain, colorA, colorB, label, formatter = fmtNum) {
  return `<div class="scale-legend"><span>${formatter(domain[0])}</span><div class="scale-bar" style="background:linear-gradient(90deg,${colorA},${colorB})"></div><span>${formatter(domain[1])}</span></div><span>${label}</span>`;
}
function pctText(value) { return value == null ? "n/a" : fmtPct(value); }
function ratingValue(rating) { return RISK_ORDER.indexOf(rating) + 1; }
function sparsePctTicks(domain) {
  const [lo, hi] = domain;
  return [lo, 0, 0.1, 1, hi]
    .filter(v => v != null && Number.isFinite(v) && v >= lo && v <= hi)
    .filter((v, i, arr) => arr.findIndex(x => Math.abs(x - v) < 1e-9) === i);
}

function drawStateBoundaries(target, path) {
  if (!DATA.stateGeojson?.features?.length) return;
  target.append("g").attr("class", "state-boundaries")
    .selectAll("path")
    .data(DATA.stateGeojson.features)
    .join("path")
    .attr("class", "state-boundary")
    .attr("d", path);
}

function drawMap(svgId, fillFn, tooltipFn, legendId, legendHtml, clickFn) {
  const svg = d3.select(svgId);
  const width = svg.node().clientWidth || 520;
  const height = svg.node().clientHeight || 430;
  svg.attr("viewBox", [0,0,width,height]).selectAll("*").remove();
  const projection = d3.geoAlbersUsa().fitSize([width, height], DATA.geojson);
  const path = d3.geoPath(projection);
  svg.on("mouseleave", () => hideTooltip());
  svg.append("g").selectAll("path")
    .data(DATA.geojson.features)
    .join("path")
    .attr("class","county")
    .attr("d", path)
    .attr("fill", d => fillFn(countyByFips.get(d.properties.fips), d.properties.fips))
    .style("cursor", clickFn ? "pointer" : null)
    .on("mousemove", (event, d) => {
      const html = tooltipFn(countyByFips.get(d.properties.fips), d.properties.fips);
      if (!html) {
        hideTooltip();
        return;
      }
      showTooltip(event, html);
    })
    .on("click", clickFn ? (event, d) => clickFn(d.properties.fips) : null);
  drawStateBoundaries(svg, path);
  if (legendId) d3.select(legendId).html(legendHtml || "");
}

function drawScoreScatter() {
  const container = document.querySelector("#score-scatter");
  const canvas = document.querySelector("#score-scatter-canvas");
  const series = scoreHistoryData?.series || [];
  const months = (scoreHistoryData?.months || []).map(parsePriceMonth);
  if (!container || !canvas || !series.length || !months.length) return;
  const generation = ++scoreRenderGeneration;
  delete canvas.dataset.rendered;
  const width = container.clientWidth || 960, height = container.clientHeight || 340;
  const margin = {top: 18, right: 18, bottom: 46, left: 68};
  const p10 = DATA.priceRisk.summary.historyPpsfCapLower;
  const p90 = DATA.priceRisk.summary.historyPpsfCapUpper;
  const clipDomain = [p10, p90];
  const cappedSpan = Math.max(p90 - p10, 0.01);
  const yDomain = [p10 - cappedSpan * 0.08, p90 + cappedSpan * 0.08];
  const x = d3.scaleUtc().domain(d3.extent(months)).range([margin.left, width - margin.right]);
  const y = d3.scaleLinear().domain(yDomain).nice().range([height - margin.bottom, margin.top]);

  const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(width * pixelRatio);
  canvas.height = Math.round(height * pixelRatio);
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  const context = canvas.getContext("2d");
  context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
  context.clearRect(0, 0, width, height);

  const svg = d3.select("#score-scatter-axes");
  svg.attr("viewBox", [0,0,width,height]).selectAll("*").remove();
  const yTicks = sparsePctTicks(y.domain());
  svg.append("g").attr("class","grid").attr("transform",`translate(${margin.left},0)`).call(d3.axisLeft(y).tickValues(yTicks).tickSize(-(width-margin.left-margin.right)).tickFormat(""));
  svg.append("g").attr("class","axis").attr("transform",`translate(0,${height-margin.bottom})`).call(d3.axisBottom(x).ticks(d3.utcYear.every(2)).tickFormat(d3.utcFormat("%Y")));
  svg.append("g").attr("class","axis").attr("transform",`translate(${margin.left},0)`).call(d3.axisLeft(y).tickValues(yTicks).tickFormat(fmtAxisPct));
  svg.append("text").attr("x",width/2).attr("y",height-8).attr("text-anchor","middle").attr("fill","#66717b").attr("font-size",12).text("Month");
  svg.append("text").attr("transform","rotate(-90)").attr("x",-height/2).attr("y",20).attr("text-anchor","middle").attr("fill","#66717b").attr("font-size",12).text("Monthly Median PPSF YoY");
  scoreScatterState = {container, series, months, x, y, clipDomain, margin, width, height, renderedCount: 0};
  scoreScatterRendered = true;
  const batchSize = 120;
  let seriesIndex = 0;
  function drawCountyBatch() {
    if (generation !== scoreRenderGeneration) return;
    const batchEnd = Math.min(series.length, seriesIndex + batchSize);
    context.save();
    context.beginPath();
    context.rect(margin.left, margin.top, width - margin.left - margin.right, height - margin.top - margin.bottom);
    context.clip();
    context.beginPath();
    for (; seriesIndex < batchEnd; seriesIndex += 1) {
      const county = series[seriesIndex];
      let started = false;
      county.values.forEach((value, index) => {
        if (value == null || !Number.isFinite(value) || !months[index]) {
          started = false;
          return;
        }
        const px = x(months[index]);
        const py = y(capValue(value, clipDomain));
        if (started) context.lineTo(px, py);
        else {
          context.moveTo(px, py);
          started = true;
        }
      });
    }
    context.strokeStyle = "rgba(91, 122, 138, 0.12)";
    context.lineWidth = 1.1;
    context.stroke();
    context.restore();
    scoreScatterState.renderedCount = batchEnd;
    canvas.dataset.progress = String(batchEnd);
    if (seriesIndex < series.length) requestAnimationFrame(drawCountyBatch);
    else canvas.dataset.rendered = "true";
  }
  requestAnimationFrame(drawCountyBatch);
}

function initScoreScatter() {
  const container = document.querySelector("#score-scatter");
  if (!container) return;
  container.addEventListener("mousemove", event => {
    if (!scoreScatterState) return;
    const clientX = event.clientX, clientY = event.clientY;
    if (scoreTooltipFrame != null) cancelAnimationFrame(scoreTooltipFrame);
    scoreTooltipFrame = requestAnimationFrame(() => {
      scoreTooltipFrame = null;
      const {series, months, x, y, clipDomain, margin, width, height} = scoreScatterState;
      const rect = container.getBoundingClientRect();
      const px = clientX - rect.left, py = clientY - rect.top;
      if (px < margin.left || px > width - margin.right || py < margin.top || py > height - margin.bottom) {
        hideTooltip();
        return;
      }
      const monthIndex = d3.bisector(d => d).center(months, x.invert(px));
      let nearest = null, nearestDistance = Infinity;
      for (const county of series.slice(0, scoreScatterState.renderedCount)) {
        const value = county.values[monthIndex];
        if (value == null || !Number.isFinite(value)) continue;
        const distance = Math.abs(y(capValue(value, clipDomain)) - py);
        if (distance < nearestDistance) {
          nearest = county;
          nearestDistance = distance;
        }
      }
      if (!nearest || nearestDistance > 10) {
        hideTooltip();
        return;
      }
      showTooltip({clientX, clientY}, `<strong>${countyDisplayName(nearest)}</strong>`);
    });
  });
  container.addEventListener("mouseleave", () => hideTooltip());
}

function currentRatingSequenceFrame() {
  return RATING_SEQUENCE_FRAMES[
    Math.max(0, Math.min(ratingSequenceIndex, RATING_SEQUENCE_FRAMES.length - 1))
  ];
}

function updateRiskLegend(selector, activeRisk, onSelect, {tooltipText = null} = {}) {
  const legend = d3.select(selector);
  legend.selectAll(".risk-legend-info").remove();
  const activeIndex = RISK_ORDER.indexOf(activeRisk);
  const buttons = legend.selectAll("button.risk-legend-button")
    .data(RISK_ORDER).join("button")
    .attr("type", "button")
    .attr("class", "risk-legend-button")
    .style("--risk-color", risk => RISK_COLORS[risk])
    .classed("revealed", risk => RISK_ORDER.indexOf(risk) <= activeIndex)
    .classed("active", risk => risk === activeRisk)
    .attr("aria-pressed", risk => risk === activeRisk ? "true" : "false")
    .attr("aria-label", risk => `${risk} Risk`)
    .text(risk => risk)
    .on("click", (event, risk) => onSelect(risk));
  if (tooltipText) {
    const veryHighButton = buttons.filter(risk => risk === "Very High").node();
    if (veryHighButton) {
      const info = makeInfoButton(tooltipText, {label: TEXT.informationTooltipLabel});
      info.classList.add("risk-legend-info");
      veryHighButton.after(info);
    }
  }
}

function setSequenceButton(selector, paused) {
  d3.select(selector)
    .text(paused ? "\u25B6" : "\u275A\u275A")
    .attr("aria-label", paused ? "Resume animation" : "Pause animation")
    .attr("title", paused ? "Resume animation" : "Pause animation")
    .attr("aria-pressed", paused ? "true" : "false");
}

function drawRatingScatter() {
  const ratingHistory = DATA.priceRisk.ratingHistory || [];
  const data = ratingHistory.filter(d => d.median != null);
  data.forEach(d => { d.monthDate = parsePriceMonth(d.month); });
  const svg = d3.select("#rating-scatter");
  const width = svg.node().clientWidth || 520, height = svg.node().clientHeight || 620;
  const margin = {top: 18, right: 24, bottom: 46, left: 68};
  svg.attr("viewBox", [0,0,width,height]).selectAll("*").remove();
  const bandValues = data.flatMap(d => [d.q1, d.q3]).filter(v => v != null && Number.isFinite(v));
  const bandExtent = bandValues.length ? d3.extent(bandValues) : [0, 1];
  const bandSpan = bandExtent[1] - bandExtent[0];
  const bandPadding = bandSpan > 0 ? bandSpan * 0.05 : 0.01;
  const yDomain = [bandExtent[0] - bandPadding, bandExtent[1] + bandPadding];
  const x = d3.scaleUtc().domain(d3.extent(data, d => d.monthDate)).range([margin.left, width - margin.right]);
  const y = d3.scaleLinear().domain(yDomain).nice().range([height - margin.bottom, margin.top]);
  const yTicks = sparsePctTicks(y.domain());
  svg.append("g").attr("class","grid").attr("transform",`translate(${margin.left},0)`).call(d3.axisLeft(y).tickValues(yTicks).tickSize(-(width-margin.left-margin.right)).tickFormat(""));
  svg.append("g").attr("class","axis").attr("transform",`translate(0,${height-margin.bottom})`).call(d3.axisBottom(x).ticks(d3.utcYear.every(2)).tickFormat(d3.utcFormat("%Y")));
  svg.append("g").attr("class","axis").attr("transform",`translate(${margin.left},0)`).call(d3.axisLeft(y).tickValues(yTicks).tickFormat(fmtAxisPct));
  svg.append("text").attr("transform","rotate(-90)").attr("x",-height/2).attr("y",20).attr("text-anchor","middle").attr("fill","#66717b").attr("font-size",12).text("Monthly Median PPSF YoY");
  const area = d3.area().x(d => x(d.monthDate)).y0(d => y(capValue(d.q1, yDomain))).y1(d => y(capValue(d.q3, yDomain)));
  const line = d3.line().x(d => x(d.monthDate)).y(d => y(capValue(d.median, yDomain)));
  const activeSequenceRisk = currentRatingSequenceFrame().risk;
  const activeRiskIndex = activeSequenceRisk == null
    ? RISK_ORDER.length
    : RISK_ORDER.indexOf(activeSequenceRisk);
  for (const risk of RISK_ORDER) {
    const riskIndex = RISK_ORDER.indexOf(risk);
    if (activeSequenceRisk && riskIndex > activeRiskIndex) continue;
    const rows = data.filter(d => d.riskRating === risk).sort((a,b)=>d3.ascending(a.monthDate,b.monthDate));
    if (!rows.length) continue;
    const background = activeSequenceRisk && riskIndex < activeRiskIndex;
    svg.append("path").datum(rows).attr("class",`band ${background ? "background" : ""}`).attr("fill",RISK_COLORS[risk]).attr("d",area);
    svg.append("path").datum(rows).attr("class",`line ${background ? "background" : ""}`).attr("stroke",RISK_COLORS[risk]).attr("d",line);
  }
}

function drawRatingMap() {
  const activeRisk = currentRatingSequenceFrame().risk;
  drawMap("#rating-map",
    (county, fips) => {
      if (!county) return "#ece7df";
      const rating = hazardCounty(county, "overall").rating;
      if (activeRisk && rating !== activeRisk) return "#e6e8e5";
      return RISK_COLORS[rating] || "#ece7df";
    },
    (county, fips) => {
      if (!county) return "";
      return `<strong>${countyDisplayName(county)}</strong><br>NRI risk rating: ${hazardCounty(county, "overall").rating ?? "n/a"}`;
    },
    null,
    null
  );
}

function renderRatingSequence() {
  drawRatingScatter();
  drawRatingMap();
  updateRiskLegend("#rating-risk-legend", currentRatingSequenceFrame().risk, risk => {
    pauseRatingSequence(true);
    ratingSequenceIndex = RISK_ORDER.indexOf(risk);
    renderRatingSequence();
  });
  setSequenceButton("#rating-play-button", ratingSequencePaused);
}

function pauseRatingSequence(manual = false) {
  clearInterval(ratingSequenceTimer);
  ratingSequenceTimer = null;
  if (manual) ratingSequencePaused = true;
  setSequenceButton("#rating-play-button", true);
}

function startRatingSequence(reset = true) {
  pauseRatingSequence();
  ratingSequencePaused = false;
  if (reset) ratingSequenceIndex = 0;
  renderRatingSequence();
  ratingSequenceTimer = setInterval(() => {
    ratingSequenceIndex = (ratingSequenceIndex + 1) % RATING_SEQUENCE_FRAMES.length;
    renderRatingSequence();
  }, RISK_SEQUENCE_INTERVAL);
}

function drawLineChart(svgId, source, groupKey, horizonLimit, activeRisk = null, minMonth = -12, opts = {}) {
  const data = source.filter(d => d.month >= minMonth && d.month <= horizonLimit);
  const svg = d3.select(svgId);
  const width = svg.node().clientWidth || 700, height = svg.node().clientHeight || 430;
  const margin = {top: 22, right: opts.marginRight ? Math.min(opts.marginRight, width * 0.3) : 96, bottom: 42, left: 58};
  svg.attr("viewBox", [0,0,width,height]).selectAll("*").remove();
  const x = d3.scaleLinear().domain([minMonth, horizonLimit]).range([margin.left, width - margin.right]);
  const hideOthers = opts.hideOtherGroups;
  const domainData = hideOthers && activeRisk ? data.filter(d => d[groupKey] === activeRisk) : data;
  const values = domainData.flatMap(d => [d.q1, d.median, d.q3]).filter(v => v != null);
  if (opts.extraDomainValues) values.push(...opts.extraDomainValues);
  const valueExtent = d3.extent(values);
  const valueSpan = Math.max((valueExtent[1] || 0) - (valueExtent[0] || 0), 0.01);
  const yDomain = [
    valueExtent[0] - valueSpan * (opts.lowerDomainPadding || 0),
    valueExtent[1] + valueSpan * (opts.upperDomainPadding || 0),
  ];
  const y = d3.scaleLinear().domain(yDomain).nice().range([height - margin.bottom, margin.top]);
  svg.append("g").attr("class","grid").attr("transform",`translate(${margin.left},0)`).call(d3.axisLeft(y).ticks(6).tickSize(-(width-margin.left-margin.right)).tickFormat(""));
  const useYears = horizonLimit > 12;
  const yearTicks = [minMonth, 0, ...d3.range(12, horizonLimit + 1, 12)];
  const axis = useYears
    ? d3.axisBottom(x).tickValues(yearTicks).tickFormat(d => d < 0 ? `${Math.abs(d / 12)}y pre` : d === 0 ? (opts.eventLabel || "event") : `${d / 12}y`)
    : d3.axisBottom(x).ticks(8).tickFormat(d => d === 0 ? "event" : d);
  svg.append("g").attr("class","axis").attr("transform",`translate(0,${height-margin.bottom})`).call(axis);
  svg.append("g").attr("class","axis").attr("transform",`translate(${margin.left},0)`).call(d3.axisLeft(y).ticks(6).tickFormat(fmtPct));
  if (opts.yAxisLabel) svg.append("text").attr("transform","rotate(-90)").attr("x",-height/2).attr("y",14).attr("text-anchor","middle").attr("fill","#66717b").attr("font-size",11).text(opts.yAxisLabel);
  const boltPoints = d3.range(margin.top, height - margin.bottom + 1, 9)
    .map((py, index) => [x(0) + (index % 2 ? 3 : -3), py]);
  svg.append("path").attr("class", "event-line").attr("d", d3.line()(boltPoints));
  const grouped = d3.group(data, d => groupKey ? d[groupKey] : "All affected counties");
  for (const [key, rows] of grouped) {
    rows.sort((a,b)=>d3.ascending(a.month,b.month));
    if (hideOthers && key !== activeRisk) continue;
    const isBackground = !hideOthers && activeRisk && RISK_ORDER.indexOf(key) < RISK_ORDER.indexOf(activeRisk);
    const isHidden = !hideOthers && activeRisk && RISK_ORDER.indexOf(key) > RISK_ORDER.indexOf(activeRisk);
    if (isHidden) continue;
    const color = groupKey === "riskRating" ? RISK_COLORS[key] : "#0f766e";
    const areaFn = d3.area().x(d=>x(d.month)).y0(d=>y(d.q1)).y1(d=>y(d.q3));
    const lineFn = d3.line().x(d=>x(d.month)).y(d=>y(d.median));
    const transition = opts.animate ? svg.transition().duration(620).ease(d3.easeCubicInOut) : null;
    const previousX = opts.previousDomain
      ? d3.scaleLinear().domain(opts.previousDomain).range([margin.left, width - margin.right])
      : x;
    const initialArea = d3.area().x(d=>previousX(d.month)).y0(d=>y(d.q1)).y1(d=>y(d.q3));
    const initialLine = d3.line().x(d=>previousX(d.month)).y(d=>y(d.median));
    const previousRows = (opts.previousSource || [])
      .filter(d => d.month >= (opts.previousDomain?.[0] ?? minMonth) && d.month <= (opts.previousDomain?.[1] ?? horizonLimit) && (!groupKey || d[groupKey] === key))
      .sort((a, b) => d3.ascending(a.month, b.month));
    const initialRows = previousRows.length ? previousRows : rows;
    if (!opts.hideBands) {
      const band = svg.append("path").attr("class",`band ${isBackground ? "background" : ""}`).attr("fill",color).attr("d", initialArea(initialRows)).datum(rows);
      if (transition) band.transition(transition).attr("d", areaFn);
    }
    const medianLine = svg.append("path").attr("class",`line ${isBackground ? "background" : ""}`).attr("stroke",color).attr("d", initialLine(initialRows)).datum(rows);
    if (transition) medianLine.transition(transition).attr("d", lineFn);
    const last = rows.at(-1);
    if (last && !isBackground && !opts.hideEndLabel) svg.append("text").attr("x",x(last.month)+5).attr("y",y(last.median)+4).attr("fill",color).attr("font-size",12).attr("font-weight",800).text(key);
  }
  if (!opts.hideXAxisLabel) svg.append("text").attr("x",width/2).attr("y",height-8).attr("text-anchor","middle").attr("fill","#66717b").attr("font-size",12).text(opts.xAxisLabel || (useYears ? "Years from event start / after event end" : "Months from event start / after event end"));
  return {x, y, margin, width, height};
}

function initButtons() {
  // Hazard sidebar (right) for pricing section — vertical, uniform size, with icons
  d3.select("#rating-play-button").on("click", () => {
    if (!ratingSectionRendered) return;
    if (ratingSequencePaused) startRatingSequence(false); else pauseRatingSequence(true);
  });

  // Event section: arrow nav for windows

  // Event section: risk rating sidebar (right, color-coded)
  d3.select("#event-play-button").on("click", () => {
    if (!eventSectionRendered) return;
    if (riskAutoPaused) startRiskTimer(); else pauseRiskTimer(true);
  });

  d3.select("#feature-sequence-resume").on("click", () => {
    if (!featureSectionRendered) return;
    featureSubgroupSequencePaused = !featureSubgroupSequencePaused;
    if (featureSubgroupSequencePaused) clearInterval(featureSubgroupSequenceTimer);
    else startFeatureSubgroupSequence();
    drawFeatureSubgroupDetail();
    drawFeatureSubgroupLines();
  });

  // Features risk group toggle (centered above the card)
  updateRiskLegend("#feature-risk-legend", selectedFeatureRisk, d => {
      if (!featureSectionRendered) return;
      selectedFeatureRisk=d;
      selectedFeatureKey=null;
      selectedFeatureSubgroup=null;
      selectedDistributionFeature=null;
      featureSubgroupSequencePaused=false;
      clearInterval(featureSubgroupSequenceTimer);
      drawFeatureHeatmaps();
    });
}

function median(values) {
  const valid = values.filter(v => v != null && Number.isFinite(v)).sort(d3.ascending);
  return valid.length ? d3.median(valid) : null;
}

function countyDisplayName(county) {
  if (!county) return "";
  const name = county.county || "";
  const suffix = county.state ? `, ${county.state}` : "";
  return suffix && name.endsWith(suffix) ? name : `${name}${suffix}`;
}

function featureExampleIsHigher(county) {
  return county.samplePosition.startsWith("Above") || county.samplePosition.startsWith("Higher");
}

function featureExampleRole(county) {
  return featureExampleIsHigher(county)
    ? (county.samplePosition.startsWith("Above") ? "above group median" : "higher trajectory")
    : (county.samplePosition.startsWith("Below") ? "below group median" : "lower trajectory");
}

function featureExampleColor(county) {
  return featureExampleIsHigher(county) ? FEATURE_HIGHER_LINE_COLOR : FEATURE_LOWER_LINE_COLOR;
}

function featureScaleHtml(contribution) {
  const score = contribution == null || !Number.isFinite(+contribution)
    ? null
    : Math.max(-1, Math.min(1, +contribution));
  const p = score == null ? null : 50 + score * 50;
  const markerPosition = p == null ? null : Math.max(3, Math.min(97, p));
  const marker = markerPosition == null ? "" : `<span class="feature-scale-arrow" style="left:${markerPosition}%;"></span>`;
  return `<div class="feature-scale"><span>Lower PPSF YoY</span><span class="feature-scale-bar">${marker}</span><span style="text-align:right;">Higher PPSF YoY</span></div>`;
}

function featurePercentileScaleHtml(percentile) {
  const p = percentile == null || !Number.isFinite(+percentile)
    ? null
    : Math.max(0, Math.min(100, +percentile));
  const markerPosition = p == null ? null : Math.max(3, Math.min(97, p));
  const marker = markerPosition == null ? "" : `<span class="feature-scale-arrow" style="left:${markerPosition}%;"></span>`;
  return `<div class="feature-scale"><span>Low value</span><span class="feature-scale-bar percentile">${marker}</span><span style="text-align:right;">High value</span></div>`;
}

// ---- render event section with two windows ----
function activeWindowData() {
  if (activeEventWindow === "before") return DATA.eventWindows.windowBefore;
  return activeEventWindow === "B" ? DATA.eventWindows.windowB : DATA.eventWindows.windowA;
}

function renderEventSection() {
  const overview = activeEventWindow === "overview";
  d3.select("#event-time-context").style("display", overview ? "none" : null)
    .text(activeEventWindow === "before" ? TEXT.eventsBeforeContext : TEXT.eventsAfterContext);
  d3.select("#event-overview-stat").style("display", overview ? null : "none");
  d3.select("#event-overview").classed("visible", overview);
  d3.select("#event-detail-grid").style("display", overview ? "none" : null);
  d3.select("#event-sequence-row").style("display", overview ? "none" : null);
  if (overview) {
    const count = DATA.eventWindows.summary?.affectedCounties || 0;
    const total = DATA.eventWindows.summary?.totalCounties || 0;
    d3.select("#event-overview-stat").html(`<strong>${d3.format(",d")(count)}</strong> counties / <strong>${total ? d3.format(".1%")(count / total) : "N/A"}</strong> of counties had experienced climate events`);
    d3.select("#t-events-card-title").text(replaceFeatureText(TEXT.eventsOverviewTitle, {count: d3.format(",d")(count)}));
    d3.select("#events-window-subtitle").text("");
    document.getElementById("t-events-card-title").dataset.window = "overview";
    drawEventRiskBars();
    return;
  }
  const wd = activeWindowData();
  const before = activeEventWindow === "before";
  const eventTitle = before ? TEXT.eventsBeforeTitle : activeEventWindow === "A" ? TEXT.eventsShortTitle : TEXT.eventsLongTitle;
  const horizonYears = before ? null : TEXT.eventHorizonYears[activeEventWindow];
  const highlightedYears = horizonYears ? `<span class="event-horizon-number${activeEventWindow === "B" ? " changed" : ""}">${horizonYears}</span>` : null;
  const eventTitleElement = document.getElementById("t-events-card-title");
  if (eventTitleElement.dataset.window !== activeEventWindow) {
    eventTitleElement.dataset.window = activeEventWindow;
    setCompleteMonthlyPlotTitle(
      eventTitleElement,
      horizonYears ? eventTitle.replace(horizonYears, highlightedYears) : eventTitle,
      {html: Boolean(horizonYears)},
    );
  }
  d3.select("#events-window-subtitle").text("");
  updateRiskLegend("#event-risk-legend", selectedRisk, risk => {
    selectedRisk = risk;
    pauseRiskTimer(true);
    renderEventSection();
  });
  setSequenceButton("#event-play-button", riskAutoPaused);
  const ta = activeEventWindow === "A" ? TEXT.eventWindowATakeaway : TEXT.eventWindowBTakeaway;
  d3.select("#event-window-takeaway").text(ta);
  const nextDomain = [-12, before ? 0 : activeEventWindow === "A" ? 36 : 60];
  drawLineChart("#event-window", wd.byRating, "riskRating",
    nextDomain[1],
    selectedRisk,
    -12,
    {hideBands: before, hideEndLabel: true, hideXAxisLabel: true, marginRight: 24, animate: true, previousDomain: lastEventDomain || nextDomain, previousSource: lastEventSource}
  );
  lastEventDomain = nextDomain;
  lastEventSource = wd.byRating;
  drawAffectedMap(wd);
}

function drawEventRiskBars() {
  const counts = DATA.eventWindows.summary?.riskCounts || {};
  const data = RISK_ORDER.map(risk => ({risk, count: counts[risk] || 0})).filter(d => d.count > 0);
  const svg = d3.select("#event-risk-pie");
  const width = svg.node().clientWidth || 1000, height = svg.node().clientHeight || 430;
  svg.attr("viewBox", [0, 0, width, height]).selectAll("*").remove();
  const margin = {top: 26, right: 70, bottom: 38, left: 120};
  svg.append("text").attr("transform", "rotate(-90)").attr("x", -height / 2).attr("y", 18)
    .attr("text-anchor", "middle").attr("font-size", 12).attr("fill", "#17332d").text("NRI risk rating");
  const x = d3.scaleLinear().domain([0, d3.max(data, d => d.count) || 1]).nice().range([margin.left, width - margin.right]);
  const y = d3.scaleBand().domain(data.map(d => d.risk)).range([margin.top, height - margin.bottom]).padding(.28);
  svg.append("g").attr("class", "grid").attr("transform", `translate(0,${height - margin.bottom})`)
    .call(d3.axisBottom(x).ticks(Math.max(3, Math.floor(width / 180))).tickSize(-(height - margin.top - margin.bottom)).tickFormat(""));
  svg.selectAll("rect.event-risk-bar").data(data).join("rect")
    .attr("class", "event-risk-bar").attr("x", x(0)).attr("y", d => y(d.risk))
    .attr("width", d => Math.max(1, x(d.count) - x(0))).attr("height", y.bandwidth())
    .attr("fill", d => RISK_COLORS[d.risk])
    .on("mousemove", (event, d) => showTooltip(event, `<strong>${d.risk} Risk</strong><br>${d3.format(",d")(d.count)} counties`))
    .on("mouseleave", () => hideTooltip());
  svg.append("g").attr("class", "axis").attr("transform", `translate(${margin.left},0)`).call(d3.axisLeft(y));
  svg.append("g").attr("class", "axis").attr("transform", `translate(0,${height - margin.bottom})`).call(d3.axisBottom(x).ticks(Math.max(3, Math.floor(width / 180))).tickFormat(d3.format(",d")));
  svg.selectAll("text.event-risk-count").data(data).join("text")
    .attr("class", "event-risk-count").attr("x", d => x(d.count) + 7).attr("y", d => y(d.risk) + y.bandwidth() / 2 + 4)
    .attr("font-size", 11).attr("font-weight", 800).text(d => d3.format(",d")(d.count));
}

function drawAffectedMap(wd) {
  const affected = new Map((wd.affectedCounties || []).map(d => [d.fips, d.riskRating]));
  const selectedCount = new Set((wd.affectedCounties || []).filter(d => d.riskRating === selectedRisk).map(d => d.fips)).size;
  d3.select("#affected-county-count").style("--active-risk-color", RISK_COLORS[selectedRisk]).text(`${d3.format(",d")(selectedCount)} counties`);
  drawMap("#affected-map",
    (county, fips) => affected.get(fips) === selectedRisk ? RISK_COLORS[selectedRisk] : "#e6dfd5",
    (county, fips) => affected.get(fips) === selectedRisk ? `<strong>${countyDisplayName(county) || fips}</strong><br>NRI rating: ${selectedRisk}` : "",
    null, null
  );
}

function startRiskTimer() {
  clearInterval(riskTimer);
  riskAutoPaused = false;
  renderEventSection();
  riskTimer = setInterval(() => {
    selectedRisk = RISK_ORDER[(RISK_ORDER.indexOf(selectedRisk) + 1) % RISK_ORDER.length];
    renderEventSection();
  }, RISK_SEQUENCE_INTERVAL);
}

function pauseRiskTimer(manual = false) {
  clearInterval(riskTimer);
  riskTimer = null;
  if (manual) riskAutoPaused = true;
  setSequenceButton("#event-play-button", true);
}

// ---- features section ----
function featureLabel(feature) {
  return TEXT.featureLabels[feature] || feature;
}

function replaceFeatureText(template, values) {
  return Object.entries(values).reduce(
    (result, [key, value]) => result.replaceAll(`{${key}}`, String(value)),
    template,
  );
}

function activeFeatureMetric(feature) {
  return (DATA.features.importanceByRisk[selectedFeatureRisk] || []).find(d => d.feature === feature);
}

function drawFeatureImportanceV2() {
  const metrics = new Map((DATA.features.importanceByRisk[selectedFeatureRisk] || []).map(d => [d.feature, d]));
  d3.select("#feature-order-controls").selectAll("button")
    .text(function() { return this.dataset.order === "category" ? TEXT.featureGroupByCategory : TEXT.featureOrderBySignificance; })
    .classed("active", function() { return this.dataset.order === featureOrderMode; })
    .attr("aria-pressed", function() { return this.dataset.order === featureOrderMode ? "true" : "false"; })
    .on("click", function() {
      featureOrderMode = this.dataset.order;
      drawFeatureImportanceV2();
    });
  d3.select("#feature-click-hint").text(TEXT.featureClickHint);
  const chart = d3.select("#feature-importance-chart");
  chart.selectAll("*").remove();
  let lastCategory = null;
  const features = featureOrderMode === "significance"
    ? [...DATA.features.featureOrder].sort((a, b) => (metrics.get(b)?.absRho || 0) - (metrics.get(a)?.absRho || 0))
    : DATA.features.featureOrder;
  const significantFeatures = new Set(mostImportantFeatureMetrics(selectedFeatureRisk).map(metric => metric.feature));
  const strongContainer = featureOrderMode === "significance" && significantFeatures.size > 0
    ? chart.append("div").attr("class", "importance-strong-group")
    : null;
  features.forEach(feature => {
    const meta = DATA.features.featureMeta[feature];
    if (featureOrderMode === "category" && meta.category !== lastCategory) {
      chart.append("div").attr("class", "importance-group-label").text(TEXT.featureCategories[meta.category] || meta.category);
      lastCategory = meta.category;
    }
    const metric = metrics.get(feature) || {};
    const width = Math.min(100, Math.max(0, (metric.absRho || 0) * 200));
    const strong = featureOrderMode === "significance" && significantFeatures.has(feature);
    const negativeActive = selectedFeatureKey === feature && (metric.rho || 0) < 0;
    const button = (strong ? strongContainer : chart).append("button")
      .attr("type", "button")
      .attr("class", `importance-row${selectedFeatureKey === feature ? " active" : ""}${negativeActive ? " negative-active" : ""}`)
      .attr("aria-pressed", selectedFeatureKey === feature ? "true" : "false")
      .attr("title", `${featureLabel(feature)} · |ρ| ${d3.format(".2f")(metric.absRho || 0)}`)
      .on("click", () => {
        selectedFeatureKey = selectedFeatureKey === feature ? null : feature;
        drawFeatureHeatmaps();
      });
    button.append("span").attr("class", `correlation-marker ${(metric.rho || 0) < 0 ? "negative" : "positive"}`);
    button.append("span").attr("class", "importance-label").text(featureLabel(feature));
    button.append("span").attr("class", "importance-bar-track")
      .append("span").attr("class", "importance-bar").style("display", "block").style("width", `${width}%`);
  });
  const tabs = document.getElementById("feature-footnote-tabs");
  tabs.innerHTML = "";
  [
    [TEXT.featureSourcesTopic, TEXT.featureSourcesNote],
    [TEXT.featureRankingTopic, replaceFeatureText(TEXT.featureRankingNote, {threshold: d3.format(".2f")(DATA.features.minimumEffect)})],
  ].forEach(([labelText, content]) => {
    const topic = document.createElement("span");
    topic.className = "feature-footnote-topic";
    topic.append(
      document.createTextNode(labelText),
      makeInfoButton(content, {
        html: labelText === TEXT.featureSourcesTopic,
        label: `${labelText}: ${TEXT.informationTooltipLabel}`,
      }),
    );
    tabs.appendChild(topic);
  });
}

function drawFeatureMedianLine() {
  d3.select(".feature-line-pane").classed("scatter-active", false).classed("scatter-negative", false);
  setCompleteMonthlyPlotTitle("#feature-chart-title", TEXT.featureLineTitle);
  d3.select("#feature-relationship").attr("hidden", true);
  drawLineChart("#feature-event-window", DATA.eventWindows.windowA.byRating, "riskRating", 36, selectedFeatureRisk, -12, {hideOtherGroups: true, hideEndLabel: true, marginRight: 24, xAxisLabel: TEXT.featureXAxis, yAxisLabel: TEXT.featureYAxis, eventLabel: TEXT.featureEventMarker});
}

function featureTickFormatter(feature) {
  const format = DATA.features.featureMeta[feature]?.format;
  if (format === "currency") return d3.format("$,.2s");
  if (format === "pct") return fmtPct;
  if (format === "percent") return value => `${d3.format(".1f")(value)}%`;
  return d3.format(".2s");
}

function fitFeatureScatterToTakeaway() {
  const stage = document.querySelector('#features .story-stage[data-story-state="takeaway-feature"]');
  const svg = stage?.querySelector('.scatter-active #feature-event-window');
  const relationship = stage?.querySelector('#feature-relationship:not([hidden])');
  if (!svg || !relationship || !selectedFeatureKey) return;
  const available = Math.max(120, Math.floor(relationship.getBoundingClientRect().top - svg.getBoundingClientRect().top - 6));
  if (Math.abs(svg.getBoundingClientRect().height - available) > 1) {
    stage.style.setProperty('--feature-scatter-height', `${available}px`);
    drawFeatureScatter(selectedFeatureKey);
  }
}

function drawFeatureScatter(feature) {
  const allRows = (DATA.features.countyRowsByRisk[selectedFeatureRisk] || [])
    .map(d => ({fips: d.fips, county: d.county, state: d.state, x: d.values[feature], y: d.target}))
    .filter(d => d.x != null && d.y != null);
  const iqrBounds = values => {
    const sorted = values.filter(Number.isFinite).sort(d3.ascending);
    const q1 = d3.quantileSorted(sorted, .25), q3 = d3.quantileSorted(sorted, .75);
    const spread = q3 - q1;
    return [q1 - 1.5 * spread, q3 + 1.5 * spread];
  };
  const [xLow, xHigh] = iqrBounds(allRows.map(d => d.x));
  const [yLow, yHigh] = iqrBounds(allRows.map(d => d.y));
  const rows = allRows.filter(d => d.x >= xLow && d.x <= xHigh && d.y >= yLow && d.y <= yHigh);
  const metric = activeFeatureMetric(feature) || {};
  const rho = metric.rho || 0;
  d3.select(".feature-line-pane").classed("scatter-active", true).classed("scatter-negative", rho < 0);
  const svg = d3.select("#feature-event-window");
  const width = svg.node().clientWidth || 700, height = svg.node().clientHeight || 500;
  const margin = {top: 24, right: 24, bottom: 58, left: 68};
  svg.attr("viewBox", [0, 0, width, height]).selectAll("*").remove();
  const x = d3.scaleLinear().domain(d3.extent(rows, d => d.x)).nice().range([margin.left, width - margin.right]);
  const y = d3.scaleLinear().domain(d3.extent(rows, d => d.y)).nice().range([height - margin.bottom, margin.top]);
  svg.append("g").attr("class", "grid").attr("transform", `translate(${margin.left},0)`).call(d3.axisLeft(y).ticks(6).tickSize(-(width - margin.left - margin.right)).tickFormat(""));
  svg.append("g").attr("class", "axis").attr("transform", `translate(0,${height - margin.bottom})`).call(d3.axisBottom(x).ticks(6).tickFormat(featureTickFormatter(feature)));
  svg.append("g").attr("class", "axis").attr("transform", `translate(${margin.left},0)`).call(d3.axisLeft(y).ticks(6).tickFormat(fmtPct));
  svg.append("g").selectAll("circle").data(rows).join("circle")
    .attr("cx", d => x(d.x)).attr("cy", d => y(d.y)).attr("r", 3.2)
    .attr("fill", RISK_COLORS[selectedFeatureRisk]).attr("opacity", .55)
    .on("mousemove", (event, d) => showTooltip(event, `<strong>${countyDisplayName(d)}</strong><br>${featureLabel(feature)}: ${featureTickFormatter(feature)(d.x)}<br>${TEXT.featureScatterYAxis}: ${fmtPct(d.y)}`))
    .on("mouseleave", () => hideTooltip());
  if (rows.length >= 2) {
    const meanX = d3.mean(rows, d => d.x), meanY = d3.mean(rows, d => d.y);
    const denominator = d3.sum(rows, d => (d.x - meanX) ** 2);
    const slope = denominator ? d3.sum(rows, d => (d.x - meanX) * (d.y - meanY)) / denominator : 0;
    const intercept = meanY - slope * meanX;
    const [trendStart, trendEnd] = x.domain();
    svg.append("line")
      .attr("x1", x(trendStart)).attr("y1", y(intercept + slope * trendStart))
      .attr("x2", x(trendEnd)).attr("y2", y(intercept + slope * trendEnd))
      .attr("stroke", "#17332d").attr("stroke-width", 2).attr("stroke-dasharray", "5 5")
      .attr("pointer-events", "none");
  }
  svg.append("text").attr("x", width / 2).attr("y", height - 9).attr("text-anchor", "middle").attr("fill", "#66717b").attr("font-size", 12).text(featureLabel(feature));
  svg.append("text").attr("transform", "rotate(-90)").attr("x", -height / 2).attr("y", 18).attr("text-anchor", "middle").attr("fill", "#66717b").attr("font-size", 12).text(TEXT.featureScatterYAxis);
  const chartTitle = document.getElementById("feature-chart-title");
  chartTitle.innerHTML = "";
  const outcomeLabel = document.createElement("span");
  outcomeLabel.textContent = TEXT.featureOutcomeTerm;
  const comparisonLabel = document.createElement("span");
  comparisonLabel.textContent = ` vs. ${featureLabel(feature)}`;
  chartTitle.append(outcomeLabel, makeInfoButton(TEXT.featureOutcomeTooltip), comparisonLabel);
  const strengthKey = Math.abs(rho) > 0.5 ? "strong" : Math.abs(rho) >= 0.3 ? "moderate" : "weak";
  const directionKey = rho >= 0 ? "positive" : "negative";
  d3.select("#feature-relationship").attr("hidden", null).text(replaceFeatureText(TEXT.featureRelationship, {
    feature: featureLabel(feature),
    strength: TEXT.featureRelationshipStrength[strengthKey],
    direction: TEXT.featureRelationshipDirection[directionKey],
  }));
  requestAnimationFrame(fitFeatureScatterToTakeaway);
}

const FEATURE_SUBGROUP_COLORS = ["#54278f", "#807dba", "#6baed6", "#2171b5"];

function subgroupName(group, count) {
  const names = count === 3 ? TEXT.subgroupNamesThree : TEXT.subgroupNamesFour;
  return names[group.index] || replaceFeatureText(TEXT.subgroupFallback, {number: group.index + 1});
}

function subgroupDisplayName(group, count) {
  const names = count === 3 ? TEXT.subgroupShortNamesThree : TEXT.subgroupShortNamesFour;
  return names[group.index] || subgroupName(group, count);
}

function subgroupProseName(label) {
  return String(label || "").toLowerCase();
}

function subgroupPerformanceName(label) {
  return String(label || "")
    .replace(/^(Strong|Mild)\s+/i, "")
    .toLowerCase();
}

function playbookPerformanceName(label) {
  const names = {
    "Strong Overperformers": "strong overperformer",
    "Mild Overperformers": "mild overperformer",
    "Mild Underperformers": "mild underperformer",
    "Strong Underperformers": "strong underperformer",
    "Overperformers": "overperformer",
    "Average Performers": "average performer",
    "Underperformers": "underperformer",
  };
  return names[label] || subgroupProseName(label).replace(/s$/, "");
}

function playbookPerformanceDisplayName(label) {
  return playbookPerformanceName(label).replace(/\b\w/g, character => character.toUpperCase());
}

function mostImportantFeatureMetrics(risk) {
  const metrics = [...(DATA.features.importanceByRisk[risk] || [])]
    .filter(metric => Number.isFinite(metric.rho))
    .sort((a, b) => (b.absRho || 0) - (a.absRho || 0));
  const strongest = metrics.filter(metric => (metric.absRho || 0) >= 0.3);
  return metrics.slice(0, Math.max(3, strongest.length));
}

function subgroupFeatureRelations(risk, subgroup) {
  if (!subgroup) return [];
  const peerRows = DATA.features.countyRowsByRisk[risk] || [];
  return (subgroup.traits || []).map(trait => {
    const peerValues = peerRows
      .map(row => row.values?.[trait.feature])
      .filter(Number.isFinite)
      .sort(d3.ascending);
    if (!Number.isFinite(trait.median) || !peerValues.length) return null;
    const metric = mostImportantFeatureMetrics(risk).find(item => item.feature === trait.feature);
    if (!Number.isFinite(metric?.rho)) return null;
    const groups = DATA.features.subgroupsByRisk[risk]?.groups || [];
    const overperformer = /overperformer/i.test(subgroupName(subgroup, groups.length, risk));
    const relation = (metric.rho > 0) === overperformer ? "higher" : "lower";
    const peerMedian = d3.median(peerValues);
    const members = peerRows.filter(row => DATA.features.subgroupByFips[row.fips] === subgroup.index)
      .map(row => row.values?.[trait.feature]).filter(Number.isFinite);
    const matching = members.filter(value => relation === "higher" ? value > peerMedian : value < peerMedian).length;
    return {...trait, relation, peerMedian, matching, denominator: members.length,
      percentage: members.length ? 100 * matching / members.length : null};
  }).filter(Boolean);
}

function drawFeatureSubgroupSummary() {
  const payload = DATA.features.subgroupsByRisk[selectedFeatureRisk] || {groups: []};
  const group = payload.groups.find(d => d.index === selectedFeatureSubgroup);
  const groupLabel = group ? subgroupName(group, payload.groups.length) : "";
  const relations = subgroupFeatureRelations(selectedFeatureRisk, group);
  const summary = d3.select("#feature-subgroup-summary").html("");
  summary.append("h3").text(replaceFeatureText(TEXT.featureFrame3Title, {
    subgroup: groupLabel,
    risk: selectedFeatureRisk,
  }));
  if (!relations.length) {
    summary.append("p").attr("class", "feature-subgroup-summary-intro").text(TEXT.featureSubgroupSummaryUnavailable);
    return;
  }
  summary.append("p").attr("class", "feature-subgroup-summary-intro").text(replaceFeatureText(TEXT.featureSubgroupSummaryIntro, {
    subgroup: groupLabel,
    risk: selectedFeatureRisk,
  }));
  const rowScroller = summary.append("div").attr("class", "feature-subgroup-summary-rows");
  const rows = rowScroller.selectAll("div.feature-subgroup-summary-row").data(relations, d => d.feature).join("div")
    .attr("class", "feature-subgroup-summary-row");
  rows.append("div").attr("class", "feature-subgroup-factor-name")
    .html(d => playbookFeatureCategoryIcon(d.feature)).append("strong").text(d => featureLabel(d.feature));
  rows.append("span").attr("class", "feature-peer-percentage").html(d => d.percentage == null
    ? "No county feature values available in this subgroup."
    : `<span class="feature-percentage-value">${d3.format('.1f')(d.percentage)}%</span><span>have values <strong>${d.relation === 'higher' ? 'above' : 'below'}</strong> its risk-group median.</span>`);
}

function featureDistributionOptions(payload) {
  return mostImportantFeatureMetrics(selectedFeatureRisk);
}

function drawFeatureSubgroupPanel() {
  const payload = DATA.features.subgroupsByRisk[selectedFeatureRisk] || {groups: [], excludedOutliers: 0};
  if (selectedFeatureSubgroup == null || !payload.groups.some(d => d.index === selectedFeatureSubgroup)) {
    selectedFeatureSubgroup = [...payload.groups].sort((a, b) => b.index - a.index)[0]?.index ?? null;
  }
  const options = featureDistributionOptions(payload);
  if (!options.some(metric => metric.feature === selectedDistributionFeature)) selectedDistributionFeature = options[0]?.feature || null;
  const controls = d3.select("#feature-distribution-controls");
  controls.classed("single-feature", options.length <= 1).selectAll("*").remove();
  if (!payload.hasStrongFeatures) controls.append("span").attr("class", "feature-click-hint").style("grid-column", "1 / -1").text(TEXT.featureDistributionFallback);
  const rotateFeature = direction => {
    const current = Math.max(0, options.findIndex(metric => metric.feature === selectedDistributionFeature));
    selectedDistributionFeature = options[(current + direction + options.length) % options.length]?.feature || null;
    drawFeatureSubgroupPanel();
  };
  if (options.length > 1) controls.append("button").attr("type", "button").attr("aria-label", TEXT.featureDistributionPrevious)
    .attr("title", TEXT.featureDistributionPrevious).text("←").on("click", () => rotateFeature(-1));
  controls.append("span").attr("class", "feature-distribution-current").text(featureLabel(selectedDistributionFeature));
  if (options.length > 1) controls.append("button").attr("type", "button").attr("aria-label", TEXT.featureDistributionNext)
    .attr("title", TEXT.featureDistributionNext).text("→").on("click", () => rotateFeature(1));
  const group = payload.groups.find(d => d.index === selectedFeatureSubgroup);
  if (!group || !selectedDistributionFeature) return;
  const rows = DATA.features.countyRowsByRisk[selectedFeatureRisk] || [];
  const allValues = rows.map(row => row.values[selectedDistributionFeature]).filter(Number.isFinite).sort(d3.ascending);
  const rawSubgroupPoints = rows
    .filter(row => DATA.features.subgroupByFips[row.fips] === selectedFeatureSubgroup)
    .map(row => ({fips: row.fips, value: row.values[selectedDistributionFeature]}))
    .filter(point => Number.isFinite(point.value)).sort((a, b) => d3.ascending(a.value, b.value));
  const rawSubgroupValues = rawSubgroupPoints.map(point => point.value);
  const q1All = d3.quantileSorted(allValues, .25), q3All = d3.quantileSorted(allValues, .75);
  const spread = q3All - q1All;
  const lowerBound = spread > 0 ? q1All - 1.5 * spread : d3.min(allValues);
  const upperBound = spread > 0 ? q3All + 1.5 * spread : d3.max(allValues);
  const removeOutliers = selectedFeatureRisk !== "Very High";
  const displayedValues = removeOutliers
    ? allValues.filter(value => value >= lowerBound && value <= upperBound)
    : allValues;
  const subgroupPoints = removeOutliers
    ? rawSubgroupPoints.filter(point => point.value >= lowerBound && point.value <= upperBound)
    : rawSubgroupPoints;
  const subgroupValues = subgroupPoints.map(point => point.value);
  const svg = d3.select("#feature-distribution-chart");
  const width = svg.node().clientWidth || 320, height = svg.node().clientHeight || 210;
  const margin = {top: 8, right: 18, bottom: 34, left: 10};
  svg.attr("viewBox", [0, 0, width, height]).selectAll("*").remove();
  if (!displayedValues.length || !subgroupValues.length) return;
  const x = d3.scaleLinear().domain(d3.extent(displayedValues)).nice().range([margin.left, width - margin.right]);
  const selectedColor = FEATURE_SUBGROUP_COLORS[group.index];
  const drawDistribution = (points, y, color, opacity) => {
    const values = points.map(point => point.value).sort(d3.ascending);
    svg.append("g").selectAll("circle").data(points).join("circle")
      .attr("cx", d => x(d.value)).attr("cy", (d, i) => y + ((i % 7) - 3) * 1.6)
      .attr("r", 2.8).attr("fill", color).attr("opacity", opacity)
      .style("cursor", "pointer")
      .on("mousemove", (event, d) => {
        const county = countyByFips.get(d.fips);
        showTooltip(event, `<strong>${countyDisplayName(county) || d.fips}</strong><br>${featureLabel(selectedDistributionFeature)}: ${featureTickFormatter(selectedDistributionFeature)(d.value)}`);
      })
      .on("mouseleave", () => hideTooltip());
    const q1 = d3.quantileSorted(values, .25), medianValue = d3.quantileSorted(values, .5), q3 = d3.quantileSorted(values, .75);
    svg.append("rect").attr("x", x(q1)).attr("y", y - 11).attr("width", Math.max(1, x(q3) - x(q1))).attr("height", 22).attr("fill", color).attr("opacity", .2).attr("stroke", color);
    svg.append("line").attr("x1", x(medianValue)).attr("x2", x(medianValue)).attr("y1", y - 13).attr("y2", y + 13).attr("stroke", color).attr("stroke-width", 3);
  };
  drawDistribution(subgroupPoints, (margin.top + height - margin.bottom) / 2, selectedColor, .48);
  svg.append("g").attr("class", "axis").attr("transform", `translate(0,${height - margin.bottom})`).call(d3.axisBottom(x).ticks(5).tickFormat(featureTickFormatter(selectedDistributionFeature)));
  const distributionTitle = document.getElementById("feature-distribution-title");
  const distributionTitleKey = `${selectedFeatureRisk}:${selectedDistributionFeature}`;
  if (distributionTitle.dataset.feature !== distributionTitleKey) {
    if (activeInfoTooltipTrigger && distributionTitle.contains(activeInfoTooltipTrigger)) closeInfoTooltip();
    distributionTitle.innerHTML = "";
    distributionTitle.dataset.feature = distributionTitleKey;
    distributionTitle.append(
      document.createTextNode(replaceFeatureText(TEXT.featureDistributionTitle, {feature: featureLabel(selectedDistributionFeature)})),
      makeInfoButton(removeOutliers ? TEXT.featureDistributionOutlierTooltip : TEXT.featureDistributionVeryHighTooltip, {label: TEXT.informationTooltipLabel}),
    );
  }
  const correlation = options.find(metric => metric.feature === selectedDistributionFeature)?.rho;
  const levelKey = correlation < 0 ? "higher" : "lower";
  const level = TEXT.featureDistributionLevel[levelKey];
  d3.select("#feature-subgroup-takeaway").html(replaceFeatureText(TEXT.featureSubgroupTakeaway, {
    level,
    feature: featureLabel(selectedDistributionFeature),
    risk: selectedFeatureRisk,
  }));
  setSequenceButton("#feature-sequence-resume", featureSubgroupSequencePaused);
}

function drawFeatureSubgroupDetail() {
  const state = document.querySelector("#features .story-stage")?.dataset.storyState;
  if (state === "feature-frame-3") drawFeatureSubgroupSummary();
  else drawFeatureSubgroupPanel();
  setSequenceButton("#feature-sequence-resume", featureSubgroupSequencePaused);
}

function selectFeatureSubgroup(index, userInitiated = true) {
  selectedFeatureSubgroup = index;
  if (userInitiated) {
    featureSubgroupSequencePaused = true;
    clearInterval(featureSubgroupSequenceTimer);
  }
  drawFeatureSubgroupDetail();
  drawFeatureSubgroupLines();
}

function startFeatureSubgroupSequence() {
  clearInterval(featureSubgroupSequenceTimer);
  const payload = DATA.features.subgroupsByRisk[selectedFeatureRisk] || {groups: []};
  const sequence = [...payload.groups].sort((a, b) => b.index - a.index);
  if (!sequence.length) return;
  if (selectedFeatureSubgroup == null) selectedFeatureSubgroup = sequence[0].index;
  if (featureSubgroupSequencePaused) return;
  featureSubgroupSequenceTimer = setInterval(() => {
    const current = sequence.findIndex(group => group.index === selectedFeatureSubgroup);
    selectedFeatureSubgroup = sequence[(current + 1) % sequence.length].index;
    drawFeatureSubgroupDetail();
    drawFeatureSubgroupLines();
  }, RISK_SEQUENCE_INTERVAL);
}

function drawFeatureSubgroupLines() {
  const payload = DATA.features.subgroupsByRisk[selectedFeatureRisk] || {groups: []};
  const domainValues = payload.groups.flatMap(group => group.values.map(d => d.value).filter(Number.isFinite));
  d3.select(".feature-line-pane").classed("scatter-active", false).classed("scatter-negative", false);
  const chart = drawLineChart("#feature-event-window", DATA.eventWindows.windowA.byRating, "riskRating", 36, selectedFeatureRisk, -12, {hideOtherGroups: true, hideEndLabel: true, extraDomainValues: domainValues, upperDomainPadding: 0.16, marginRight: 24, xAxisLabel: TEXT.featureXAxis, yAxisLabel: TEXT.featureYAxis, eventLabel: TEXT.featureEventMarker});
  const svg = d3.select("#feature-event-window");
  const line = d3.line().defined(d => d.value != null).x(d => chart.x(d.month)).y(d => chart.y(d.value));
  svg.selectAll("path.feature-subgroup-line").data(payload.groups).join("path")
    .attr("class", "line feature-subgroup-line").attr("stroke", d => FEATURE_SUBGROUP_COLORS[d.index])
    .attr("stroke-width", d => d.index === selectedFeatureSubgroup ? 4.5 : 2.5)
    .attr("opacity", d => selectedFeatureSubgroup == null || d.index === selectedFeatureSubgroup ? 1 : .16)
    .attr("d", d => line(d.values)).style("cursor", "pointer")
    .on("click", (event, d) => selectFeatureSubgroup(d.index, true));
  const orderedGroups = [...payload.groups].sort((a, b) => b.index - a.index);
  const subgroupButtons = d3.select("#feature-subgroup-toggles")
    .classed("visible", true)
    .selectAll("button.feature-subgroup-control")
    .data(orderedGroups, d => d.index)
    .join("button")
    .attr("type", "button")
    .attr("class", d => `feature-subgroup-control${d.index === selectedFeatureSubgroup ? " active" : ""}`)
    .attr("aria-pressed", d => d.index === selectedFeatureSubgroup ? "true" : "false")
    .attr("aria-label", d => `${subgroupName(d, payload.groups.length, selectedFeatureRisk)} subgroup${d.index === selectedFeatureSubgroup ? ", selected" : ""}`)
    .style("--subgroup-color", d => FEATURE_SUBGROUP_COLORS[d.index])
    .on("pointerdown", (event, d) => {
      if (event.button !== 0) return;
      event.preventDefault();
      event.stopPropagation();
      selectFeatureSubgroup(Number(d.index), true);
    })
    .on("click", (event, d) => {
      if (event.detail !== 0) return;
      event.preventDefault();
      event.stopPropagation();
      selectFeatureSubgroup(Number(d.index), true);
  });
  subgroupButtons.selectAll("span.feature-subgroup-control-label")
    .data(d => [d], d => d.index)
    .join("span")
    .attr("class", "feature-subgroup-control-label")
    .text(d => subgroupName(d, payload.groups.length));
  setCompleteMonthlyPlotTitle("#feature-chart-title", TEXT.featureLineTitle);
  d3.select("#feature-relationship").attr("hidden", true);
}

function drawFeatureHeatmaps() {
  const state = document.querySelector("#features .story-stage")?.dataset.storyState || "feature-frame-1";
  updateRiskLegend("#feature-risk-legend", selectedFeatureRisk, risk => {
    selectedFeatureRisk = risk;
    selectedFeatureKey = null;
    selectedFeatureSubgroup = null;
    selectedDistributionFeature = null;
    featureSubgroupSequencePaused = false;
    clearInterval(featureSubgroupSequenceTimer);
    drawFeatureHeatmaps();
  });
  drawFeatureImportanceV2();
  d3.select("#feature-detail-title").text(replaceFeatureText(
    state === "feature-frame-2" ? TEXT.featureFrame2Title : state === "feature-frame-3" ? TEXT.featureFrame3Title : TEXT.featureFrame1Title,
    {risk: selectedFeatureRisk, subgroup: ""},
  ));
  const detailStack = document.querySelector(".feature-detail-stack");
  const detailTitle = document.getElementById("feature-detail-title");
  detailStack?.style.setProperty("--feature-frame-top", `${(detailTitle?.offsetHeight || 25) + 8}px`);
  if (state === "feature-frame-2" || state === "feature-frame-3") {
    if (selectedFeatureSubgroup == null) {
      const groups = DATA.features.subgroupsByRisk[selectedFeatureRisk]?.groups || [];
      selectedFeatureSubgroup = [...groups].sort((a, b) => b.index - a.index)[0]?.index ?? null;
    }
    drawFeatureSubgroupDetail();
    drawFeatureSubgroupLines();
    startFeatureSubgroupSequence();
  }
  else {
    clearInterval(featureSubgroupSequenceTimer);
    d3.select("#feature-subgroup-toggles").classed("visible", false).selectAll("*").remove();
    if ((state === "feature-frame-1" || state === "takeaway-feature") && selectedFeatureKey) drawFeatureScatter(selectedFeatureKey);
    else drawFeatureMedianLine();
  }
}

function formatFeatureVal(v, fmt) {
  if (v == null) return "n/a";
  if (fmt === "currency") return fmtMoney(v);
  if (fmt === "percent") return `${d3.format(",.2f")(v)}%`;
  if (fmt === "pct" || fmt === "signed_pct") return d3.format("+.2%")(v);
  if (fmt === "temperature_f") return `${d3.format(",.1f")(v)} °F`;
  if (fmt === "inches") return `${d3.format(",.2f")(v)} in`;
  return fmtNum(v);
}

// ---- playbook section ----
function updatePlaybookZoomControl() {
  const county = playbookCountyByFips.get(selectedCountyFips);
  const zoomed = playbookMapTransform.k > 1.01;
  d3.select("#playbook-map-zoom-toggle")
    .style("display", county || zoomed ? "inline-flex" : "none")
    .text(zoomed ? TEXT.playbookZoomOut : TEXT.playbookZoomIn);
}

function drawPlaybookMap(svgSelector = "#county-selection-map", county = null, autoZoom = false, interactive = true) {
  const svg = d3.select(svgSelector);
  const width = svg.node().clientWidth || 1100;
  const height = svg.node().clientHeight || 380;
  svg.attr("viewBox", [0, 0, width, height]).selectAll("*").remove();
  const projection = d3.geoAlbersUsa().fitSize([width, height], DATA.geojson);
  const path = d3.geoPath(projection);
  const group = svg.append("g");

  group.selectAll("path")
    .data(DATA.geojson.features)
    .join("path")
    .attr("class", "county")
    .attr("d", path)
    .attr("fill", d => d.properties.fips === county?.fips ? "#172026" : "#d8d0c4")
    .style("cursor", interactive ? "pointer" : "default")
    .on("mousemove", (event, d) => {
      const profile = playbookCountyByFips.get(d.properties.fips);
      if (!profile) return;
      showTooltip(event, `<strong>${countyDisplayName(profile)}</strong>`);
    })
    .on("mouseleave", () => hideTooltip())
    .on("click", (event, d) => {
      if (!interactive) return;
      const profile = playbookCountyByFips.get(d.properties.fips);
      if (profile) {
        selectPlaybookCounty(profile);
        goToPlaybookProfile();
      }
    });
  drawStateBoundaries(group, path);

  if (!interactive) {
    svg.on(".zoom", null);
    if (county) {
      const selectedFeature = DATA.geojson.features.find(d => d.properties.fips === county.fips);
      if (selectedFeature) {
        const [cx, cy] = path.centroid(selectedFeature);
        const scale = county.state === "AK" ? 3.2 : county.state === "HI" ? 4.2 : 5.2;
        group.attr("transform", d3.zoomIdentity.translate(width / 2, height / 2).scale(scale).translate(-cx, -cy));
      }
    }
    return;
  }

  playbookZoomBehavior = d3.zoom()
    .scaleExtent([1, 12])
    .translateExtent([[-width * .35, -height * .35], [width * 1.35, height * 1.35]])
    .on("zoom", event => {
      playbookMapTransform = event.transform;
      playbookMapZoomed = event.transform.k > 1.01;
      group.attr("transform", event.transform);
      updatePlaybookZoomControl();
    });
  svg.call(playbookZoomBehavior).on("dblclick.zoom", null);
  svg.call(playbookZoomBehavior.transform, d3.zoomIdentity);

  if (county) {
    const selectedFeature = DATA.geojson.features.find(d => d.properties.fips === county.fips);
    if (selectedFeature) {
      const [cx, cy] = path.centroid(selectedFeature);
      const scale = county.state === "AK" ? 3.2 : county.state === "HI" ? 4.2 : 5.2;
      playbookSelectedTransform = d3.zoomIdentity.translate(width / 2, height / 2).scale(scale).translate(-cx, -cy);
      if (autoZoom) svg.transition().duration(520).call(playbookZoomBehavior.transform, playbookSelectedTransform);
    }
  } else {
    playbookSelectedTransform = d3.zoomIdentity;
    playbookMapTransform = d3.zoomIdentity;
    playbookMapZoomed = false;
    updatePlaybookZoomControl();
  }
}

function historicalPerformanceAssignment(a, b, c, d) {
  if (![a, b, c, d].every(Number.isFinite)) return null;
  if (a === b) return "Mild Overperformers";
  if (a > b) return a - b < .5 * (c - b) ? "Mild Overperformers" : "Strong Overperformers";
  return b - a < .5 * (b - d) ? "Mild Underperformers" : "Strong Underperformers";
}

function playbookHistoricalStatistics(county) {
  if (!RISK_ORDER.includes(county.riskRating)) return {a: null, b: null, c: null, d: null};
  const values = playbookHistoryRows(county).map(d => d.value).filter(Number.isFinite);
  const series = buildRiskGroupSeries(county);
  return {
    a: d3.median(values) ?? null,
    b: d3.median(series.map(d => d.median).filter(Number.isFinite)) ?? null,
    c: d3.median(series.map(d => d.q3).filter(Number.isFinite)) ?? null,
    d: d3.median(series.map(d => d.q1).filter(Number.isFinite)) ?? null,
  };
}

function playbookFeatureProfile(county) {
  const metrics = mostImportantFeatureMetrics(county.riskRating);
  const row = (DATA.features.countyRowsByRisk[county.riskRating] || []).find(d => d.fips === county.fips);
  const subgroupIndex = DATA.features.subgroupByFips?.[county.fips];
  const payload = DATA.features.subgroupsByRisk[county.riskRating] || {groups: []};
  const subgroup = payload.groups.find(d => d.index === subgroupIndex);
  const observed = playbookEvents(county).length > 0 && subgroup;
  const stats = observed ? null : playbookHistoricalStatistics(county);
  const fallbackName = stats ? historicalPerformanceAssignment(stats.a, stats.b, stats.c, stats.d) : null;
  return {
    metrics,
    row,
    subgroup,
    assignmentSource: observed ? "event-window" : fallbackName ? "ten-year-median-quartiles" : null,
    subgroupName: observed ? subgroupName(subgroup, payload.groups.length, county.riskRating) : fallbackName,
  };
}

function renderPlaybookPerformanceStatus(county, compact = false) {
  const risk = county.hazards?.overall?.rating || county.riskRating;
  const profile = playbookFeatureProfile(county);
  const container = d3.select("#playbook-performance-status")
    .attr("class", `playbook-performance-status${compact ? " compact" : ""}`);
  if (!risk || !RISK_ORDER.includes(risk)) {
    container.html("");
    return profile;
  }
  if (!profile.subgroupName) {
    container.html(`<span>${fillTextTemplate(TEXT.playbookInsufficientPerformance, {county: countyDisplayName(county)})}</span>`);
    return profile;
  }
  container.html(
    `<strong>${playbookPerformanceDisplayName(profile.subgroupName)}</strong>`
    + `<span>among</span>`
    + `<strong>${risk} Risk Group</strong>`
  );
  return profile;
}

function playbookRelativePosition(a, b, c, d) {
  if (![a, b, c, d].every(Number.isFinite)) return null;
  if (a === b) return "mid range";
  if (a > b && a - b >= .5 * (c - b)) return "upper range";
  if (a < b && b - a >= .5 * (b - d)) return "lower range";
  return "mid range";
}

function renderPlaybookPerformanceTakeaway(county) {
  const target = d3.select("#playbook-history-comparison");
  if (!playbookHasSufficientHistory(county)) {
    target.text(TEXT.playbookInsufficientHistory);
    return;
  }
  const profile = playbookFeatureProfile(county);
  if (!profile.subgroupName) {
    target.text(TEXT.playbookHistoryComparisonUnavailable);
    return;
  }
  const basis = profile.assignmentSource === "event-window"
    ? "Within each risk group, counties are ranked by the median of their Median PPSf YoY values across the event window, which is defined as the period of one year up till the event start and three years after the event end.\n\n If a county had multiple events, it is first represented by a single event trajectory built from the medians across the individual events at each relative month."
    : "The county's median value of Median PPSF YoY over the past ten years is compared with the median values of its risk group's monthly median and quartile values of Median PPSF YoY over the same period. A gap at least halfway from the group median toward the relevant quartile is Strong; a smaller gap is Mild.";
  target.html(`<span>${countyDisplayName(county)} is a <strong>${playbookPerformanceDisplayName(profile.subgroupName)}</strong> within the <strong>${county.riskRating} risk group</strong>.</span>`);
  target.node().querySelector("span").append(makeInfoButton(basis, {label: TEXT.informationTooltipLabel}));
}

function renderPlaybookFeatureSummary(county, summarizeSubgroup = false) {
  const profile = playbookFeatureProfile(county);
  const subgroupRelations = subgroupFeatureRelations(county.riskRating, profile.subgroup);
  const featureTitle = summarizeSubgroup && profile.subgroupName
    ? replaceFeatureText(TEXT.playbookSubgroupFeatureTitle, {
      subgroup: profile.subgroupName,
      risk: county.riskRating || "Unknown",
    })
    : replaceFeatureText(TEXT.playbookFeatureTitle, {risk: county.riskRating || "Unknown"});
  d3.select("#playbook-feature-title").text(featureTitle);
  const hasFeatureData = summarizeSubgroup
    ? Boolean(profile.subgroupName && subgroupRelations.length)
    : Boolean(
      profile.row
      && profile.metrics.length
      && profile.subgroupName
      && profile.metrics.every(metric => Number.isFinite(profile.row.values?.[metric.feature]))
    );
  if (!hasFeatureData) {
    const insufficientCopy = !profile.row
      ? TEXT.playbookInsufficientEventWindowData
      : TEXT.playbookInsufficientFeatureData;
    d3.select("#playbook-feature-summary").html(`<div class="playbook-feature-insufficient">${replaceFeatureText(insufficientCopy, {county: countyDisplayName(county)})}</div>`);
    d3.select("#playbook-subgroup-summary").style("display", "none").text("");
  } else if (summarizeSubgroup) {
    const summary = d3.select("#playbook-feature-summary").html("");
    const rows = summary.selectAll("div.playbook-feature-row").data(subgroupRelations, d => d.feature).join("div")
      .attr("class", "playbook-feature-row");
    rows.append("strong").text(d => featureLabel(d.feature));
    rows.append("span").attr("class", d => `feature-peer-relation ${d.relation}`).text(d => TEXT.featurePeerRelation[d.relation]);
    d3.select("#playbook-subgroup-summary")
      .style("display", null)
      .text(replaceFeatureText(TEXT.playbookSubgroup, {
        county: countyDisplayName(county),
        subgroup: playbookPerformanceName(profile.subgroupName),
        risk: county.riskRating || "Unknown",
      }));
  } else {
    d3.select("#playbook-feature-summary").html(profile.metrics.map(metric => {
      const value = profile.row.values?.[metric.feature];
      const format = DATA.features.featureMeta[metric.feature]?.format;
      return `<div class="playbook-feature-row"><strong>${featureLabel(metric.feature)}</strong><span>${value == null ? TEXT.playbookInsufficientFeatureValue : formatFeatureVal(value, format)}</span></div>`;
    }).join(""));
    d3.select("#playbook-subgroup-summary")
      .style("display", null)
      .text(replaceFeatureText(TEXT.playbookSubgroup, {
        county: countyDisplayName(county),
        subgroup: playbookPerformanceName(profile.subgroupName),
        risk: county.riskRating || "Unknown",
      }));
  }
}

function buildRiskGroupSeries(county) {
  const months = DATA.playbook.monthlyHistoryMonths || [];
  const histories = DATA.playbook.monthlyHistoryValuesByFips || {};
  const selected = (DATA.playbook.counties || []).filter(peer => peer.riskRating === county.riskRating && histories[peer.fips]);
  return months.map((month, index) => {
    const valid = selected.map(peer => histories[peer.fips]?.[index]).filter(Number.isFinite).sort(d3.ascending);
    return valid.length
      ? {month, q1: d3.quantileSorted(valid, .25), median: d3.quantileSorted(valid, .5), q3: d3.quantileSorted(valid, .75)}
      : {month, q1: null, median: null, q3: null};
  });
}

function playbookEvents(county) {
  const parseMonth = d3.utcParse("%Y-%m");
  return ((DATA.playbook.eventsByFips || {})[county.fips] || []).map(d => ({
    eventKey: d[0], source: d[1], type: d[2], name: d[3], start: d[4], end: d[5],
    startDate: parseMonth(d[4]), endDate: parseMonth(d[5]),
  }));
}

function renderPlaybookEventList(county) {
  const events = playbookEvents(county);
  d3.select("#playbook-events-title").text(TEXT.playbookPastEventsTitle);
  const list = d3.select("#playbook-event-column");
  if (!events.length) {
    list.html(`<div class="playbook-event-card">${TEXT.playbookNoPastEvents}</div>`);
    const definition = list.select(".playbook-event-definition").node();
    if (definition) definition.appendChild(makeInfoButton(TEXT.playbookEventDefinition, {label: "Definition of extreme weather events"}));
    return;
  }
  list.html(events.map(event => `<div class="playbook-event-card" data-event-key="${event.eventKey}"><strong>${eventIcon(event)} ${normalCase(event.name || event.type)}</strong><br>${d3.utcFormat("%b %Y")(event.startDate)} to ${d3.utcFormat("%b %Y")(event.endDate)}</div>`).join(""));
  list.selectAll("[data-event-key]")
    .on("mouseenter", function() {
      const eventKey = this.dataset.eventKey;
      d3.select("#playbook-ppsf-history").selectAll(".event-period")
        .classed("event-focused", d => d.eventKey === eventKey)
        .classed("event-muted", d => d.eventKey !== eventKey);
    })
    .on("mouseleave", () => d3.select("#playbook-ppsf-history").selectAll(".event-period").classed("event-focused", false).classed("event-muted", false));
}

function drawPlaybookHistory(county, compareRisk = false, showEvents = true) {
  d3.select("#t-playbook-history-title").text(countyDisplayName(county));
  const parseMonth = d3.utcParse("%Y-%m");
  const historyMonths = DATA.playbook.monthlyHistoryMonths || [];
  const historyValues = (DATA.playbook.monthlyHistoryValuesByFips || {})[county.fips] || [];
  const observedHistory = historyMonths.map((month, index) => ({
    month,
    value: historyValues[index] ?? null,
    date: parseMonth(month),
  })).filter(d => d.value != null);
  const historyStart = parseMonth(DATA.playbook.historyStart);
  const historyEnd = parseMonth(DATA.playbook.historyEnd);
  const historyDomainEnd = d3.utcDay.offset(d3.utcMonth.offset(historyEnd, 1), -1);
  const observedByMonth = new Map(observedHistory.map(d => [d.month, d.value]));
  const history = d3.utcMonth.range(historyStart, d3.utcMonth.offset(historyEnd, 1)).map(date => {
    const month = d3.utcFormat("%Y-%m")(date);
    return {date, month, value: observedByMonth.has(month) ? observedByMonth.get(month) : null, interpolated: false};
  });
  const observed = history.filter(d => d.value != null);
  const riskSeries = buildRiskGroupSeries(county);
  const events = playbookEvents(county);
  const svg = d3.select("#playbook-ppsf-history");
  const width = svg.node().clientWidth || 1050;
  const height = svg.node().clientHeight || 430;
  const margin = {top: 22, right: 24, bottom: 42, left: 62};
  svg.attr("viewBox", [0, 0, width, height]).selectAll("*").remove();
  const clipId = `playbook-history-clip-${county.fips}`;
  svg.append("defs").append("clipPath").attr("id", clipId).append("rect")
    .attr("x", margin.left).attr("y", margin.top)
    .attr("width", width - margin.left - margin.right).attr("height", height - margin.top - margin.bottom);
  const plot = svg.append("g").attr("clip-path", `url(#${clipId})`);

  const x = d3.scaleUtc().domain([historyStart, historyDomainEnd]).range([margin.left, width - margin.right]);
  const riskValues = compareRisk ? riskSeries.flatMap(d => [d.q1, d.q3]).filter(Number.isFinite) : [];
  const extentValues = observed.map(d => d.value).concat(riskValues);
  const valueExtent = extentValues.length ? d3.extent(extentValues) : [-0.1, 0.1];
  const padding = Math.max((valueExtent[1] - valueExtent[0]) * 0.12, 0.01);
  const y = d3.scaleLinear().domain([valueExtent[0] - padding, valueExtent[1] + padding]).nice()
    .range([height - margin.bottom, margin.top]);

  const grid = svg.append("g").attr("class", "grid").attr("transform", `translate(${margin.left},0)`)
    .call(d3.axisLeft(y).ticks(6).tickSize(-(width - margin.left - margin.right)).tickFormat(""));

  const chartStart = x.domain()[0], chartEnd = x.domain()[1];
  plot.selectAll("rect.event-period")
    .data(showEvents ? events.filter(d => d.endDate >= chartStart && d.startDate <= chartEnd) : [])
    .join("rect")
    .attr("class", "event-period")
    .attr("data-event-key", d => d.eventKey)
    .attr("x", d => x(d3.max([d.startDate, chartStart])))
    .attr("y", margin.top)
    .attr("width", d => Math.max(3, x(d3.min([d.endDate, chartEnd])) - x(d3.max([d.startDate, chartStart]))))
    .attr("height", height - margin.top - margin.bottom)
    .attr("fill", "#df7d2f").attr("opacity", .16)
    .on("mousemove", (event, d) => showTooltip(event, `<strong>${normalCase(d.name || d.type)}</strong><br>${d3.utcFormat("%b %Y")(d.startDate)} to ${d3.utcFormat("%b %Y")(d.endDate)}<br>${normalCase(d.source)}`))
    .on("mouseleave", () => hideTooltip());

  const riskArea = d3.area().defined(d => d.q1 != null && d.q3 != null)
    .x(d => x(parseMonth(d.month))).y0(d => y(d.q1)).y1(d => y(d.q3));
  const riskLine = d3.line().defined(d => d.median != null)
    .x(d => x(parseMonth(d.month))).y(d => y(d.median));
  let riskBand = null, riskMedian = null;
  if (compareRisk) {
    riskBand = plot.append("path").datum(riskSeries).attr("class", "band playbook-risk-band")
      .attr("fill", RISK_COLORS[county.riskRating] || "#66717b").attr("opacity", .14).attr("d", riskArea);
    riskMedian = plot.append("path").datum(riskSeries).attr("class", "line playbook-risk-median")
      .attr("stroke", RISK_COLORS[county.riskRating] || "#66717b").attr("stroke-width", 2)
      .attr("stroke-dasharray", "6 4").attr("opacity", .9).attr("d", riskLine);
  }
  const countyLinePath = d3.line().defined(d => d.value != null).x(d => x(d.date)).y(d => y(d.value));
  const countyLine = plot.append("path").datum(history).attr("class", "line playbook-county-line")
    .attr("stroke", COUNTY_LINE_COLOR).attr("stroke-width", 2.4)
    .attr("d", countyLinePath);
  svg.append("g").attr("class", "axis").attr("transform", `translate(0,${height - margin.bottom})`)
    .call(d3.axisBottom(x).ticks(d3.utcYear.every(1)).tickFormat(d3.utcFormat("%Y")));
  const yAxis = svg.append("g").attr("class", "axis").attr("transform", `translate(${margin.left},0)`)
    .call(d3.axisLeft(y).ticks(6).tickFormat(fmtPct));
  svg.append("text").attr("x", margin.left).attr("y", 13).attr("fill", "#66717b").attr("font-size", 11)
    .text("Median PPSF YoY");
  const legendItems = [
    {label: TEXT.playbookSeriesLegend, color: COUNTY_LINE_COLOR, opacity: 1, line: true},
  ];
  if (compareRisk) legendItems.push({label: replaceFeatureText(TEXT.playbookRiskSeriesLegend, {risk: county.riskRating}), color: RISK_COLORS[county.riskRating] || "#66717b", opacity: .85, line: true});
  if (showEvents && events.length) legendItems.push({label: TEXT.playbookEventLegend, color: "#df7d2f", opacity: .22});
  d3.select("#playbook-history-legend").html(legendItems.map(item =>
    `<span class="playbook-history-legend-item"><span class="playbook-history-legend-swatch" style="background:${item.color};opacity:${item.opacity};${item.line ? "height:3px;border:none;" : ""}"></span>${item.label}</span>`
  ).join(""));
  if (!observed.length) {
    svg.append("text").attr("x", width / 2).attr("y", height / 2)
      .attr("text-anchor", "middle").attr("fill", "#66717b")
      .text(TEXT.playbookMissingDataLegend);
  }
  if (compareRisk) {
    const validRisk = riskSeries.filter(d => Number.isFinite(d.q1) && Number.isFinite(d.q3));
    const limitsByMonth = new Map(validRisk.map(d => [d.month, {low: d.q1 - .5 * (d.q3 - d.q1), high: d.q3 + .5 * (d.q3 - d.q1)}]));
    const outside = observed.some(d => {
      const limits = limitsByMonth.get(d.month);
      return limits && (d.value < limits.low || d.value > limits.high);
    });
    const zoomLow = d3.min(validRisk, d => d.q1 - .5 * (d.q3 - d.q1));
    const zoomHigh = d3.max(validRisk, d => d.q3 + .5 * (d.q3 - d.q1));
    if (outside && Number.isFinite(zoomLow) && Number.isFinite(zoomHigh) && zoomHigh > zoomLow) {
      const zoomY = y.copy().domain([zoomLow, zoomHigh]).nice();
      const transition = svg.transition().delay(180).duration(900).ease(d3.easeCubicInOut);
      yAxis.transition(transition).call(d3.axisLeft(zoomY).ticks(6).tickFormat(fmtPct));
      grid.transition(transition).call(d3.axisLeft(zoomY).ticks(6).tickSize(-(width - margin.left - margin.right)).tickFormat(""));
      countyLine.transition(transition).attr("d", d3.line().defined(d => d.value != null).x(d => x(d.date)).y(d => zoomY(d.value)));
      riskBand?.transition(transition).attr("d", d3.area().defined(d => d.q1 != null && d.q3 != null).x(d => x(parseMonth(d.month))).y0(d => zoomY(d.q1)).y1(d => zoomY(d.q3)));
      riskMedian?.transition(transition).attr("d", d3.line().defined(d => d.median != null).x(d => x(parseMonth(d.month))).y(d => zoomY(d.median)));
    }
  }
}

function fillTextTemplate(template, values) {
  return Object.entries(values).reduce(
    (text, [key, value]) => text.replaceAll(`{${key}}`, value == null ? "" : String(value)),
    template || ""
  );
}

function normalCase(value) {
  const text = String(value || "").trim();
  if (!text) return "Unnamed event";
  if (text !== text.toUpperCase()) return text.charAt(0).toUpperCase() + text.slice(1);
  return text.toLowerCase().replace(/\b\w/g, letter => letter.toUpperCase())
    .replace(/\bFema\b/g, "FEMA").replace(/\bNoaa\b/g, "NOAA");
}

function eventIcon(event) {
  const label = `${event.type || ""} ${event.name || ""}`.toLowerCase();
  if (/earthquake|seismic/.test(label)) return "💥";
  if (/tornado|funnel/.test(label)) return "🌪️";
  if (/wildfire|forest fire|fire/.test(label)) return "🔥";
  if (/flood|surge|coastal/.test(label)) return "🌊";
  if (/hurricane|typhoon|tropical/.test(label)) return "🌀";
  if (/hail|ice/.test(label)) return "🧊";
  if (/winter|snow|blizzard|freeze/.test(label)) return "❄️";
  if (/drought|heat/.test(label)) return "☀️";
  if (/wind|storm|thunder|lightning/.test(label)) return "⛈️";
  return "⚠️";
}

function eventChangeHtml(delta, copy, terms) {
  if (delta == null) return `<span class="event-change flat">${terms.insufficient}</span>`;
  const magnitude = Math.abs(delta * 100);
  const intensity = Math.min(1, magnitude / 8);
  const opacity = 0.45 + intensity * 0.55;
  const size = 14 + Math.min(12, magnitude * 0.75);
  if (delta === 0) return `<span class="event-change flat"><span class="event-change-arrow">→</span>${copy.eventNoChange}</span>`;
  const isUp = delta > 0;
  const changeText = fillTextTemplate(copy.eventChange, {
    direction: isUp ? terms.up : terms.down,
    magnitude: magnitude.toFixed(1),
  });
  return `<span class="event-change ${isUp ? "up" : "down"}"><span class="event-change-arrow" style="font-size:${size}px;opacity:${opacity};">${isUp ? "↑" : "↓"}</span>${changeText}</span>`;
}

function eventChangeDirection(delta) {
  if (delta == null || !Number.isFinite(+delta)) return null;
  if (delta < -0.005) return "down";
  if (delta > 0.005) return "up";
  return "flat";
}

function riskGroupEventExpectation(risk, copy = TEXT.playbookTakeaways, terms = TEXT.playbookTakeawayTerms) {
  const rows = (DATA.eventWindows?.windowA?.byRating || [])
    .filter(row => row.riskRating === risk && row.median != null);
  const before = rows.filter(row => row.month >= -12 && row.month < 0).map(row => row.median);
  const after = rows.filter(row => row.month >= 25 && row.month <= 36).map(row => row.median);
  if (!before.length || !after.length) {
    return {delta: null, direction: null, description: copy.unavailableExpectation};
  }
  const delta = d3.median(after) - d3.median(before);
  const direction = eventChangeDirection(delta);
  const groupBehavior = fillTextTemplate(
    direction === "flat" ? copy.groupBehaviorFlat : copy.groupBehaviorChange,
    {
      direction: direction === "down" ? terms.decline : terms.increase,
      magnitude: Math.abs(delta * 100).toFixed(1),
    },
  );
  return {
    delta,
    direction,
    description: fillTextTemplate(copy.groupExpectation, {
      risk,
      groupBehavior,
    }),
    groupBehavior,
  };
}

function eventWindowTrendStats(points) {
  const valid = points.filter(d => Number.isFinite(d.month) && Number.isFinite(d.value));
  if (valid.length < 3) return {slope: null, r2: null};
  const meanMonth = d3.mean(valid, d => d.month);
  const meanValue = d3.mean(valid, d => d.value);
  const monthVariance = d3.sum(valid, d => (d.month - meanMonth) ** 2);
  const valueVariance = d3.sum(valid, d => (d.value - meanValue) ** 2);
  if (!monthVariance || !valueVariance) return {slope: 0, r2: 0};
  const covariance = d3.sum(valid, d => (d.month - meanMonth) * (d.value - meanValue));
  return {
    slope: covariance / monthVariance,
    r2: (covariance * covariance) / (monthVariance * valueVariance),
  };
}

function qualitativeRelation(difference, threshold) {
  if (Math.abs(difference) <= threshold) return "similar to";
  return difference > 0 ? "higher than" : "lower than";
}

function alignmentExtent(observed, expected, threshold) {
  if (!Number.isFinite(observed) || !Number.isFinite(expected)) return "could not be compared";
  const distance = Math.abs(observed - expected);
  if (distance <= threshold) return "closely aligns";
  if (distance <= threshold * 2) return "partially aligns";
  return "does not align";
}

function describePpsfChange(delta, terms) {
  if (Math.abs(delta) <= .005) return "remained broadly steady";
  const direction = delta > 0 ? terms.increased : terms.declined;
  return `${direction} by about ${Math.abs(delta * 100).toFixed(1)} percentage points`;
}

function renderPlaybookCommentary(county, history, events) {
  const container = d3.select("#playbook-event-commentary");
  const risk = county.hazards?.overall?.rating || county.riskRating || "Unknown";
  const copy = TEXT.playbookTakeaways;
  const terms = TEXT.playbookTakeawayTerms;
  const countyLabel = countyDisplayName(county);
  const groupExpectation = riskGroupEventExpectation(risk, copy, terms);
  const profile = playbookFeatureProfile(county);
  if (!events.length) {
    container.attr("class", "playbook-commentary neutral");
    container.html(fillTextTemplate(copy.noEvents, {
      county: countyLabel,
      countyContext: profile.subgroupName
        ? `a ${risk} Risk county in the ${subgroupProseName(profile.subgroupName)} range`
        : `a ${risk} Risk county`,
      expectation: groupExpectation.groupBehavior || "follow the broader risk-group pattern",
    }));
    return;
  }

  const eventWindowPoints = events.flatMap(event => {
    const preStart = d3.utcMonth.offset(event.startDate, -12);
    const postEnd = d3.utcMonth.offset(event.endDate, 36);
    return history
      .filter(d => d.value != null && !d.interpolated && d.date >= preStart && d.date <= postEnd)
      .map(d => ({month: d3.utcMonth.count(event.startDate, d.date), value: d.value}));
  });
  const groupRows = (DATA.eventWindows?.windowA?.byRating || [])
    .filter(row => row.riskRating === risk && row.month >= -12 && row.month <= 36 && row.median != null);
  const eventWindowValues = eventWindowPoints.map(d => d.value);
  const groupValues = groupRows.map(row => row.median);
  const countyBefore = eventWindowPoints.filter(d => d.month >= -12 && d.month < 0).map(d => d.value);
  const countyAfter = eventWindowPoints.filter(d => d.month >= 25 && d.month <= 36).map(d => d.value);
  if (!eventWindowValues.length || !groupValues.length || !countyBefore.length || !countyAfter.length || groupExpectation.delta == null) {
    container.attr("class", "playbook-commentary neutral");
    container.html(fillTextTemplate(copy.insufficientHistory, {county: countyLabel}));
    return;
  }
  const countyMedian = d3.median(eventWindowValues);
  const groupMedian = d3.median(groupValues);
  const typicalGroupIqr = d3.median(groupRows.map(row => Math.max(0, (row.q3 ?? row.median) - (row.q1 ?? row.median)))) || 0;
  const countyTrend = eventWindowTrendStats(eventWindowPoints);
  const countyDeviation = d3.deviation(eventWindowValues) || 0;
  const groupDeviation = d3.deviation(groupValues) || 0;
  const tooVolatile = (
    eventWindowValues.length >= 12
    && (countyTrend.r2 ?? 0) < .08
    && countyDeviation > Math.max(.04, groupDeviation * 1.75, typicalGroupIqr * 1.25)
  );
  container.attr("class", "playbook-commentary");
  if (tooVolatile) {
    container.html(fillTextTemplate(copy.volatileEventSummary, {
      county: countyLabel,
      risk,
      subgroupContext: profile.subgroupName
        ? `${subgroupProseName(profile.subgroupName)} subgroup shown at left`
        : "feature subgroup because sufficient feature data were unavailable",
    }));
    return;
  }
  const levelThreshold = Math.max(.01, typicalGroupIqr * .5);
  const countyChange = d3.median(countyAfter) - d3.median(countyBefore);
  const riskAlignment = alignmentExtent(countyChange, groupExpectation.delta, levelThreshold);
  const riskTargets = (DATA.features.countyRowsByRisk[risk] || []).map(d => d.target).filter(Number.isFinite);
  const riskTargetMedian = riskTargets.length ? d3.median(riskTargets) : null;
  const subgroupExpectedGap = profile.subgroup && Number.isFinite(profile.subgroup.targetMedian) && Number.isFinite(riskTargetMedian)
    ? profile.subgroup.targetMedian - riskTargetMedian
    : null;
  const observedGap = countyMedian - groupMedian;
  const template = Number.isFinite(subgroupExpectedGap)
    ? copy.eventAlignmentSummary
    : copy.eventAlignmentWithoutSubgroup;
  container.html(fillTextTemplate(template, {
    county: countyLabel,
    risk,
    riskAlignment,
    countyChange: describePpsfChange(countyChange, terms),
    groupChange: groupExpectation.groupBehavior,
    subgroup: subgroupProseName(profile.subgroupName),
    subgroupAlignment: alignmentExtent(observedGap, subgroupExpectedGap, levelThreshold),
    countyLevel: qualitativeRelation(observedGap, levelThreshold),
    subgroupLevel: qualitativeRelation(subgroupExpectedGap, levelThreshold),
  }));
}

function renderPlaybookHazards(county) {
  const rating = county.hazards?.overall?.rating || county.riskRating;
  const color = RISK_COLORS[rating] || "#66717b";
  d3.select("#playbook-hazard-ratings").html(
    `<div class="hazard-rating-overall"><div class="hazard-rating-item"><span>NRI risk rating</span><strong style="color:${color};">${rating || TEXT.playbookRiskUnavailable}</strong></div></div>`
  );
}

function playbookFeatureCategoryIcon(feature) {
  const category = DATA.features.featureMeta?.[feature]?.category;
  if (!["Economic", "Demographic"].includes(category)) return "";
  const drawing = category === "Economic"
    ? '<path d="M3 10h18L12 3zM5 12v7m7-7v7m7-7v7M3 21h18"/>'
    : '<circle cx="9" cy="7" r="3"/><path d="M3 21v-4a6 6 0 0 1 12 0v4M16 4a3 3 0 0 1 0 6m2 3a5 5 0 0 1 3 4v4"/>';
  return `<svg class="playbook-feature-category-icon" role="img" aria-label="${category} feature" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><title>${category} feature</title>${drawing}</svg>`;
}

function playbookCountyFeaturePosition(county, feature) {
  const peers = DATA.features.allCountyRowsByRisk?.[county.riskRating] || DATA.features.countyRowsByRisk[county.riskRating] || [];
  const value = peers.find(row => row.fips === county.fips)?.values?.[feature];
  const values = peers.map(row => row.values?.[feature]).filter(Number.isFinite);
  if (!Number.isFinite(value) || !values.length) return "Data unavailable";
  const median = d3.median(values);
  return value > median ? "Higher" : value < median ? "Lower" : "At peer median";
}

function playbookFactorAssessment(county, metric) {
  const position = playbookCountyFeaturePosition(county, metric.feature);
  const factorUp = position === "Higher" ? true : position === "Lower" ? false : null;
  const growth = position === "Data unavailable" || !Number.isFinite(metric.rho) ? null
    : position === "At peer median" || metric.rho === 0 ? 0
    : factorUp === (metric.rho > 0) ? 1 : -1;
  const positionLabel = position === "Higher" ? "Above median" : position === "Lower" ? "Below median"
    : position === "At peer median" ? "At median" : "Data unavailable";
  const contextOptions = typeof TEXT !== "undefined" ? TEXT.playbookFactorContext?.[metric.feature] : null;
  const context = typeof contextOptions === "string" ? contextOptions
    : contextOptions && Math.sign(metric.rho) === contextOptions.expectedRhoSign
      ? contextOptions[factorUp ? "higher" : "lower"]
      : contextOptions ? "This risk group's observed association runs opposite to the usual explanation for this factor; the correlation alone does not establish why." : null;
  const description = growth == null ? "The county feature value or its correlation is unavailable."
    : growth === 0 ? `${featureLabel(metric.feature)} is at the peer median or has zero correlation, so no directional growth association is assigned.`
    : `${countyDisplayName(county)} has ${factorUp ? "higher" : "lower"} ${featureLabel(metric.feature)} values than its peers within the ${county.riskRating} risk group. ${factorUp ? "Higher" : "Lower"} values are associated with ${growth > 0 ? "higher" : "lower"} Median PPSF YoY ${context ? " - " + context : ""}`;
  return {...metric, factorUp, growth, positionLabel, description};
}

function playbookScorecard(county, profile, metrics, risk) {
  const condition = playbookEvents(county).length ? "when" : "if";
  const factors = metrics.map(metric => playbookFactorAssessment(county, metric));
  const complete = factors.length > 0 && factors.every(item => item.growth != null);
  const positive = factors.filter(item => item.growth === 1).length;
  const negative = factors.filter(item => item.growth === -1).length;
  const balance = complete ? Math.sign(positive - negative) : null;
  const riskUp = ["Very Low", "Low"].includes(risk);
  const performerUp = /overperformer/i.test(profile.subgroupName || "");
  const arrow = signal => signal == null ? '<span class="playbook-score-arrow" aria-label="Unscored">?</span>'
    : signal === 0 ? '<span class="playbook-score-arrow" aria-label="Balanced">—</span>'
    : `<span class="playbook-score-arrow playbook-arrow-${signal > 0 ? "up" : "down"}" aria-label="${signal > 0 ? "Favorable" : "Unfavorable"}">${signal > 0 ? "↑" : "↓"}</span>`;
  const score = Number(riskUp) + Number(performerUp) + Number(balance === 1);
  const label = !complete ? "Insufficient data" : score === 3 ? "Low Risk" : score === 2 ? "Moderate Risk" : "High Risk";
  const level = !complete ? "unknown" : score === 3 ? "low" : score === 2 ? "moderate" : "high";
  const takeaway = !complete ? "Some significant features could not be scored; no overall risk label is assigned."
    : score === 3 ? "Safer environment that can benefit from steps to reduce potential climate damage"
    : score === 2 ? "Climate damage is a real possibility, know the steps to reduce it"
    : "Take steps to reduce climate damage";
  return `<div class="playbook-scorecard"><strong class="playbook-scorecard-title">Conclusion: ${countyDisplayName(county)}'s performance ${condition} an extreme climate event happens</strong>`
    + `<div class="playbook-scorecard-part"><strong>NRI Risk Rating</strong><div class="playbook-scorecard-value"><span style="color:${RISK_COLORS[risk] || "var(--ink)"}">${risk}</span>${arrow(riskUp ? 1 : -1)}</div></div>`
    + '<span class="playbook-scorecard-plus" aria-hidden="true">+</span>'
    + `<div class="playbook-scorecard-part"><strong>Performance vs. Peers</strong><div class="playbook-scorecard-value ${performerUp ? "favorable" : "unfavorable"}"><span>${playbookPerformanceDisplayName(profile.subgroupName)}</span>${arrow(performerUp ? 1 : -1)}</div></div>`
    + '<span class="playbook-scorecard-plus" aria-hidden="true">+</span>'
    + `<div class="playbook-scorecard-part"><strong>County Makeup</strong><div class="playbook-scorecard-value ${balance > 0 ? "favorable" : balance < 0 ? "unfavorable" : ""}"><span>${balance == null ? "Data unavailable" : balance > 0 ? "More positive" : balance < 0 ? "More negative" : "Balanced"}</span>${arrow(balance)}</div></div>`
    + `<div class="playbook-score-result"><div><strong>Overall score:</strong> <b class="score-risk-${level}">${label}</b></div><p class="playbook-score-takeaway">${takeaway}${complete ? ' — <a href="https://getquoll.com/" target="_blank" rel="noopener">Here’s how</a>' : ""}</p></div></div>`;
}

function playbookFeatureSubgroupMatch(county, feature) {
  const risk = county.riskRating;
  const peers = DATA.features.allCountyRowsByRisk?.[risk] || DATA.features.countyRowsByRisk[risk] || [];
  const value = peers.find(row => row.fips === county.fips)?.values?.[feature];
  if (!Number.isFinite(value)) return null;
  const rows = DATA.features.countyRowsByRisk[risk] || [];
  const groups = DATA.features.subgroupsByRisk[risk]?.groups || [];
  const candidates = groups.map(group => {
    const values = rows.filter(row => DATA.features.subgroupByFips[row.fips] === group.index)
      .map(row => row.values?.[feature]).filter(Number.isFinite).sort(d3.ascending);
    if (!values.length) return null;
    return {over: /overperformer/i.test(subgroupName(group, groups.length, risk)),
      inside: value >= values[0] && value <= values[values.length - 1],
      rangeDistance: Math.max(values[0] - value, value - values[values.length - 1], 0),
      medianDistance: Math.abs(value - d3.median(values))};
  }).filter(Boolean);
  if (!candidates.some(item => item.over) || !candidates.some(item => !item.over)) return null;
  // Prefer a containing range. Overlaps use nearest subgroup median; out-of-range
  // values use nearest range then median. Exact ties favor underperformers.
  candidates.sort((a, b) => Number(b.inside) - Number(a.inside)
    || a.rangeDistance - b.rangeDistance || a.medianDistance - b.medianDistance
    || Number(a.over) - Number(b.over));
  return candidates[0].over;
}

function playbookFactorArrows(county, metric) {
  const position = playbookCountyFeaturePosition(county, metric.feature);
  if (position === "Data unavailable" || !Number.isFinite(metric.rho)) return {factorUp: null, growthUp: null};
  const factorUp = position === "Higher";
  return {factorUp, growthUp: metric.rho !== 0 && (factorUp === (metric.rho > 0))};
}

function fitPlaybookConclusionTitle() {
  document.querySelectorAll('#playbook-warning-takeaway .playbook-scorecard-value > span:first-child').forEach(value => {
    value.style.fontSize = '16px';
    if (value.clientWidth > 0 && value.scrollWidth > value.clientWidth) {
      value.style.fontSize = `${Math.floor(16 * value.clientWidth / value.scrollWidth * 10) / 10}px`;
    }
  });
  const title = document.querySelector('#playbook-warning-takeaway .playbook-scorecard-title');
  if (!title || !title.clientWidth) return;
  title.style.fontSize = '20px';
  if (title.scrollWidth > title.clientWidth) {
    title.style.fontSize = `${Math.floor(20 * title.clientWidth / title.scrollWidth * 10) / 10}px`;
  }
}

function renderPlaybookOutlook(county) {
  const container = d3.select("#playbook-event-commentary").attr("class", "playbook-commentary");
  const introElement = d3.select("#playbook-warning-intro").property("hidden", true).html("");
  const takeawayElement = d3.select("#playbook-warning-takeaway").property("hidden", true).html("");
  if (!playbookHasSufficientHistory(county)) {
    container.attr("class", "playbook-commentary neutral").text(TEXT.playbookInsufficientHistory);
    return;
  }
  const risk = county.hazards?.overall?.rating || county.riskRating;
  if (!risk || !RISK_ORDER.includes(risk)) {
    container.attr("class", "playbook-commentary neutral").text(TEXT.playbookOutlookInsufficientRisk);
    return;
  }
  const profile = playbookFeatureProfile(county);
  const metrics = mostImportantFeatureMetrics(risk);
  if (!profile.subgroupName || !metrics.length) {
    container.attr("class", "playbook-commentary neutral").text(TEXT.playbookOutlookInsufficientFeatures);
    return;
  }
  const factors = metrics.map(metric => playbookFactorAssessment(county, metric));
  introElement.property("hidden", false).html(fillTextTemplate(TEXT.playbookTopFactorsTitle, {county: countyDisplayName(county), risk}));
  takeawayElement.property("hidden", false).html(playbookScorecard(county, profile, metrics, risk));
  const card = factor => `<div class="playbook-warning-row" data-factor="${factor.feature}"><span class="playbook-factor-name">${playbookFeatureCategoryIcon(factor.feature)}<strong>${featureLabel(factor.feature)}</strong></span><span class="playbook-factor-position" aria-label="${factor.positionLabel}">${factor.factorUp === true ? "▲" : factor.factorUp === false ? "▼" : "—"} ${factor.positionLabel}</span></div>`;
  const column = (direction, title) => {
    const rows = factors.filter(factor => factor.growth === direction);
    return `<section class="playbook-factor-column ${direction > 0 ? "positive" : "negative"}"><h4>${title}</h4><div class="playbook-factor-list">${rows.length ? rows.map(card).join("") : '<p class="playbook-factor-empty">None.</p>'}</div></section>`;
  };
  const other = factors.filter(factor => factor.positionLabel !== "At median" && (factor.growth === 0 || factor.growth == null));
  container.html('<div class="playbook-factor-columns">'
    + column(1, "Associated with higher Median PPSF YoY")
    + column(-1, "Associated with lower Median PPSF YoY") + '</div>'
    + (other.length ? '<div class="playbook-factor-other"><h4>Neutral or unavailable</h4>' + other.map(card).join("") + '</div>' : ""));
  container.selectAll(".playbook-warning-row").each(function() {
    const factor = factors.find(item => item.feature === this.dataset.factor);
    this.append(makeInfoButton(factor.description, {label: TEXT.informationTooltipLabel}));
  });
  requestAnimationFrame(fitPlaybookConclusionTitle);
}

function selectPlaybookCounty(county) {
  selectedCountyFips = county.fips;
  const playbookPanel = document.querySelector("#playbook .panel");
  playbookPanel?.classList.add("has-county-selection");
  syncPlaybookStoryLength(county);
  d3.select("#playbook-selected-county-name").style("display", "block").text(countyDisplayName(county));
  renderPlaybookHazards(county);
  d3.select("#playbook-performance-status").html("");
  drawPlaybookMap("#playbook-profile-map", county, true, false);
}

function goToPlaybookProfile() {
  const section = document.querySelector("#playbook");
  if (!section) return;
  const viewport = window.innerHeight || 1;
  window.scrollTo({
    top: section.offsetTop + viewport * 2,
    behavior: "smooth",
  });
}

function goToPlaybookSearch() {
  const section = document.querySelector("#playbook");
  if (!section) return;
  section.style.setProperty("--story-steps", STORY_CONFIG.playbook.length);
  window.scrollTo({top: section.offsetTop + (window.innerHeight || 1), behavior: "smooth"});
}

function playbookHistoryRows(county) {
  const parseMonth = d3.utcParse("%Y-%m");
  const months = DATA.playbook.monthlyHistoryMonths || [];
  const values = (DATA.playbook.monthlyHistoryValuesByFips || {})[county.fips] || [];
  return months.map((month, index) => ({date: parseMonth(month), month, value: values[index] ?? null, interpolated: false}));
}

function renderPlaybookFrame() {
  const state = document.querySelector("#playbook .story-stage")?.dataset.storyState || "search";
  const county = playbookCountyByFips.get(selectedCountyFips);
  d3.select('#t-playbook-h2').text(fillTextTemplate(TEXT.playbookH2, {frameTitle: TEXT.playbookFrameTitles[county ? state : 'search'] || TEXT.playbookFrameTitles.search}));
  if (state === "search" || !county) {
    drawPlaybookMap("#county-selection-map", county || null, false, true);
    return;
  }
  d3.select("#playbook-selected-county-name").style("display", "block").text(countyDisplayName(county));
  renderPlaybookHazards(county);
  d3.select("#playbook-performance-status").html("");
  if (state === "history-map") {
    drawPlaybookHistory(county, false, false);
    drawPlaybookMap("#playbook-profile-map", county, true, false);
    return;
  }
  if (state === "history-events") {
    drawPlaybookHistory(county, false, true);
    renderPlaybookEventList(county);
    return;
  }
  if (state === "history-compare") {
    drawPlaybookHistory(county, true, true);
    renderPlaybookPerformanceTakeaway(county);
    return;
  }
  if (state === "history-outlook") {
    d3.select("#playbook-performance-status").html("");
    renderPlaybookOutlook(county);
  }
}

function initPlaybook() {
  if (!DATA.playbook?.available) {
    d3.select("#county-search").property("disabled", true).property("placeholder", DATA.playbook?.message || "County data unavailable");
    return;
  }
  drawPlaybookMap("#county-selection-map", null, false, true);
  const input = d3.select("#county-search");
  const results = d3.select("#county-results");
  input.on("input", function() {
    const query = this.value.toLowerCase().trim();
    if (query.length < 2) { results.style("display", "none").html(""); return; }
    const matches = DATA.playbook.counties.filter(c =>
      c.county.toLowerCase().includes(query) || c.state.toLowerCase().includes(query) || c.fips.includes(query)
    ).slice(0, 20);
    results.style("display", matches.length ? "block" : "none").html("")
      .selectAll("div").data(matches).join("div")
      .style("padding", "8px 12px").style("cursor", "pointer")
      .style("border-bottom", "1px solid var(--line)").style("font-size", "13px")
      .html(d => `<strong>${countyDisplayName(d)}</strong> <span style="color:var(--muted);">(${d.hazards?.overall?.rating || "Unknown risk"})</span>`)
      .on("click", (event, d) => {
        selectPlaybookCounty(d);
        goToPlaybookProfile();
        input.property("value", "");
        results.style("display", "none").html("");
      });
  });
  d3.select("#playbook-map-zoom-in").on("click", () => {
    if (playbookZoomBehavior) d3.select("#county-selection-map").transition().duration(220).call(playbookZoomBehavior.scaleBy, 1.5);
  });
  d3.select("#playbook-map-zoom-minus").on("click", () => {
    if (playbookZoomBehavior) d3.select("#county-selection-map").transition().duration(220).call(playbookZoomBehavior.scaleBy, 1 / 1.5);
  });
  d3.select("#playbook-map-zoom-toggle").on("click", () => {
    if (!playbookZoomBehavior) return;
    const target = playbookMapTransform.k > 1.01 ? d3.zoomIdentity : playbookSelectedTransform;
    d3.select("#county-selection-map").transition().duration(420).call(playbookZoomBehavior.transform, target);
  });
  d3.select("#playbook-back-to-search").on("click", goToPlaybookSearch);
}

const STORY_CONFIG = {
  pricing: [
    {state: "title"},
    {state: "card-main"},
    {state: "takeaway-0", takeaway: "#score-scatter-takeaway", segment: 0},
    {state: "takeaway-1", takeaway: "#score-scatter-takeaway", segment: 1},
  ],
  "pricing-grouping": [
    {state: "title"},
    {state: "copy"},
    {state: "card-main", ratingSequence: true},
    {state: "takeaway-0", takeaway: "#pricing-takeaway", segment: 0},
    {state: "takeaway-1", takeaway: "#pricing-takeaway", segment: 1},
    {state: "takeaway-2", takeaway: "#pricing-question"},
  ],
  events: [
    {state: "title"},
    {state: "copy"},
    {state: "card-overview", eventWindow: "overview"},
    {state: "takeaway-overview-0", takeaway: "#event-overview-takeaway", eventWindow: "overview"},
    {state: "takeaway-overview-1", takeaway: "#event-overview-question", eventWindow: "overview"},
    {state: "card-before-intro", eventWindow: "before"},
    {state: "card-before", eventWindow: "before"},
    {state: "takeaway-before", takeaway: "#event-before-takeaway", eventWindow: "before"},
    {state: "takeaway-after-question", takeaway: "#event-after-question", eventWindow: "before"},
    {state: "card-after-intro", eventWindow: "A"},
    {state: "card-short", eventWindow: "A"},
    {state: "takeaway-short", takeaway: "#event-window-takeaway", eventWindow: "A"},
    {state: "takeaway-future", takeaway: "#event-future-prompt", eventWindow: "A"},
    {state: "card-long", eventWindow: "B"},
    {state: "takeaway-long", takeaway: "#event-window-takeaway", eventWindow: "B"},
    {state: "takeaway-0", takeaway: "#event-takeaway", eventWindow: "B"},
    {state: "takeaway-1", takeaway: "#event-variation-question", eventWindow: "B"},
  ],
  features: [
    {state: "title"},
    {state: "copy"},
    {state: "feature-frame-1"},
    {state: "takeaway-feature", takeaway: "#feature-takeaway"},
    {state: "feature-frame-2"},
    {state: "feature-frame-3"},
  ],
  playbook: [
    {state: "title"},
    {state: "search"},
    {state: "history-map"},
    {state: "history-events"},
    {state: "history-compare"},
    {state: "history-outlook"},
  ],
};

function playbookHasSufficientHistory(county) {
  const months = DATA.playbook?.monthlyHistoryMonths || [];
  const values = DATA.playbook?.monthlyHistoryValuesByFips?.[county?.fips] || [];
  const available = months.reduce((count, _, index) => count + Number(Number.isFinite(values[index])), 0);
  return county != null && months.length > 0 && available / months.length >= .5;
}

function playbookHasPerformanceGroup(county) {
  return playbookHasSufficientHistory(county) && playbookFeatureProfile(county).subgroupName != null;
}

function storyConfigForSection(id) {
  const config = STORY_CONFIG[id] || [];
  if (id !== "playbook" || !selectedCountyFips) return config;
  const county = playbookCountyByFips.get(selectedCountyFips);
  return playbookHasPerformanceGroup(county)
    ? config
    : config.filter(step => step.state !== "history-outlook");
}

function syncPlaybookStoryLength(county) {
  const section = document.getElementById("playbook");
  if (!section) return;
  const stepCount = playbookHasPerformanceGroup(county)
    ? STORY_CONFIG.playbook.length
    : STORY_CONFIG.playbook.length - 1;
  section.style.setProperty("--story-steps", stepCount);
}

const takeawayTransitionTimers = new WeakMap();

function syncTakeawaySpace(section, takeaway) {
  const panel = section.querySelector(":scope > .story-stage > .panel");
  if (!panel) return;
  if (!takeaway) {
    panel.style.removeProperty("--takeaway-space");
    return;
  }
  requestAnimationFrame(() => {
    const height = Math.ceil(takeaway.getBoundingClientRect().height);
    if (takeaway !== section.querySelector('.takeaway.story-active-takeaway')) return;
    if (height > 0) panel.style.setProperty("--takeaway-space", `${height}px`);
    if (section.id === "features") fitFeatureScatterToTakeaway();
  });
}

function activateStoryTakeaway(section, step, direction) {
  clearTimeout(takeawayTransitionTimers.get(section));
  takeawayTransitionTimers.delete(section);
  // Commit one visible card synchronously. Interrupted exits must not leave
  // active classes behind or let an obsolete timeout hide the new selection.
  section.querySelectorAll(".takeaway, .takeaway-section").forEach(element => {
    element.classList.remove(
      "story-active-takeaway", "story-active-segment", "story-outgoing-takeaway",
      "story-slide-out-up", "story-slide-out-down", "story-slide-in-up", "story-slide-in-down",
    );
  });
  const next = step.takeaway ? section.querySelector(step.takeaway) : null;
  if (!next) { syncTakeawaySpace(section, null); return; }
  next.classList.add("story-active-takeaway");
  const segment = next.classList.contains("segmented")
    ? next.querySelectorAll(".takeaway-section")[step.segment ?? 0] : null;
  if (segment) segment.classList.add("story-active-segment");
  const content = segment || next;
  void content.offsetWidth;
  content.classList.add(direction >= 0 ? "story-slide-in-up" : "story-slide-in-down");
  syncTakeawaySpace(section, next);
}

function applyStoryStep(section, step, index) {
  const stage = section.querySelector(".story-stage");
  if (!stage) return;
  const effectiveStep = (
    section.id === "playbook"
    && step.state !== "title"
    && step.state !== "search"
    && !selectedCountyFips
  )
    ? {...step, state: "search"}
    : step;
  if (
    stage.dataset.storyIndex === String(index)
    && stage.dataset.storyState === effectiveStep.state
  ) return;
  const previousState = stage.dataset.storyState;
  const previousIndex = Number(stage.dataset.storyIndex);
  const direction = Number.isFinite(previousIndex) ? Math.sign(index - previousIndex) : 1;
  stage.dataset.storyIndex = String(index);
  stage.dataset.storyState = effectiveStep.state;
  stage.dataset.storyDirection = direction < 0 ? "backward" : "forward";
  stage.classList.remove("story-step-forward", "story-step-backward");
  void stage.offsetWidth;
  stage.classList.add(direction < 0 ? "story-step-backward" : "story-step-forward");
  activateStoryTakeaway(section, effectiveStep, direction);
  const panel = stage.querySelector(".panel");
  const lockInnerScroll = (
    effectiveStep.state.startsWith("takeaway")
    || section.id === "features"
    || section.id === "playbook"
  );
  panel?.classList.toggle("inner-scroll-locked", lockInnerScroll);
  if (lockInnerScroll && panel) panel.scrollTop = 0;

  if (effectiveStep.eventWindow && activeEventWindow !== effectiveStep.eventWindow) {
    activeEventWindow = effectiveStep.eventWindow;
    if (eventSectionRendered) renderEventSection();
  }
  if (effectiveStep.ratingSequence && previousState !== effectiveStep.state && ratingSectionRendered) {
    startRatingSequence();
  }
  if (
    section.id === "pricing-grouping"
    && effectiveStep.state.startsWith("takeaway")
  ) {
    pauseRatingSequence();
  }
  if (section.id === "features" && featureSectionRendered) {
    drawFeatureHeatmaps();
  }
  if (section.id === "playbook" && playbookSectionRendered) {
    renderPlaybookFrame();
  }
}

function storyNavigationStops() {
  const viewport = window.innerHeight || 1;
  const stops = [document.querySelector(".hero")?.offsetTop || 0];
  for (const id of Object.keys(STORY_CONFIG)) {
    const section = document.getElementById(id);
    if (!section) continue;
    const config = storyConfigForSection(id);
    config.forEach((step, index) => {
      stops.push(section.offsetTop + index * viewport);
    });
  }
  return [...new Set(stops.map(stop => Math.round(stop)))].sort((a, b) => a - b);
}

function updateStoryNavigation() {
  const stops = storyNavigationStops();
  const current = window.scrollY;
  const tolerance = (window.innerHeight || 1) * 0.2;
  document.getElementById("story-prev").disabled = !stops.some(stop => stop < current - tolerance);
  document.getElementById("story-next").disabled = !stops.some(stop => stop > current + tolerance);
}

function navigateStory(direction) {
  const stops = storyNavigationStops();
  const current = window.scrollY;
  const tolerance = (window.innerHeight || 1) * 0.2;
  const target = direction > 0
    ? stops.find(stop => stop > current + tolerance)
    : [...stops].reverse().find(stop => stop < current - tolerance);
  if (target == null) return;
  window.scrollTo({top: target, behavior: "smooth"});
}

function initStoryEdgeNavigation() {
  const previous = document.getElementById("story-prev");
  const next = document.getElementById("story-next");
  const hide = () => {
    previous.classList.remove("edge-visible");
    next.classList.remove("edge-visible");
  };
  document.addEventListener("pointermove", event => {
    const edgeSize = 64;
    previous.classList.toggle("edge-visible", event.clientY <= edgeSize);
    next.classList.toggle(
      "edge-visible",
      event.clientY >= (window.innerHeight || 1) - edgeSize,
    );
  }, {passive: true});
  document.documentElement.addEventListener("mouseleave", hide);
  window.addEventListener("blur", hide);
}

function initPanelScrollRouting() {
  const standardPanels = [...document.querySelectorAll(".story-stage > .panel")]
    .filter(panel => !panel.closest("#playbook"));
  standardPanels.forEach(scrollContainer => {
    scrollContainer.addEventListener("wheel", event => {
      const ownerPanel = scrollContainer.closest(".panel");
      if (ownerPanel?.classList.contains("inner-scroll-locked")) return;
      const canScroll = scrollContainer.scrollHeight > scrollContainer.clientHeight + 1;
      if (!canScroll) return;
      const atTop = scrollContainer.scrollTop <= 0;
      const atBottom = (
        scrollContainer.scrollTop + scrollContainer.clientHeight
        >= scrollContainer.scrollHeight - 1
      );
      const leavingCard = (event.deltaY < 0 && atTop) || (event.deltaY > 0 && atBottom);
      if (leavingCard) {
        event.preventDefault();
        window.scrollBy({top: event.deltaY, behavior: "auto"});
      } else {
        event.stopPropagation();
      }
    }, {passive: false});
  });
}

function updateStoryFromScroll() {
  const viewport = window.innerHeight || 1;
  document.querySelectorAll(".slide[data-story-ready='true']").forEach(section => {
    const config = storyConfigForSection(section.id);
    const relative = (window.scrollY - section.offsetTop + viewport * .42) / viewport;
    const index = Math.max(0, Math.min(config.length - 1, Math.floor(relative)));
    applyStoryStep(section, config[index], index);
  });
  updateStoryNavigation();
}

function initScrollStory() {
  Object.entries(STORY_CONFIG).forEach(([id, config]) => {
    const section = document.getElementById(id);
    if (!section || section.dataset.storyReady) return;
    const stage = document.createElement("div");
    stage.className = "story-stage";
    while (section.firstChild) stage.appendChild(section.firstChild);
    section.appendChild(stage);
    section.style.setProperty("--story-steps", config.length);
    section.dataset.storyReady = "true";
  });
  let scheduled = false;
  const requestUpdate = () => {
    if (scheduled) return;
    scheduled = true;
    requestAnimationFrame(() => {
      scheduled = false;
      updateStoryFromScroll();
    });
  };
  window.addEventListener("scroll", requestUpdate, {passive: true});
  window.addEventListener("resize", requestUpdate);
  document.getElementById("story-prev").addEventListener("click", () => navigateStory(-1));
  document.getElementById("story-next").addEventListener("click", () => navigateStory(1));
  initStoryEdgeNavigation();
  initPanelScrollRouting();
  updateStoryFromScroll();
}

// ---- bootstrap ----
hydrateText();
initScrollStory();
initButtons();
initScoreScatter();
function renderWhenNear(selector, render, rootMargin = "350px 0px") {
  const target = document.querySelector(selector);
  if (!target) return;
  let pending = false;
  let finished = false;
  const status = document.createElement("div");
  status.className = "section-load-status";
  status.setAttribute("role", "status");
  const host = target.querySelector(".panel") || target.parentElement;
  const attempt = async () => {
    if (pending || finished) return;
    pending = true;
    status.textContent = "Loading chart data…";
    host.appendChild(status);
    target.removeAttribute("data-load-error");
    try {
      await render();
      finished = true;
      status.remove();
      sectionObserver.disconnect();
    } catch (error) {
      console.error(error);
      target.setAttribute("data-load-error", "true");
      status.textContent = "Chart data could not be loaded. ";
      const retry = document.createElement("button");
      retry.type = "button";
      retry.textContent = "Retry";
      retry.addEventListener("click", attempt);
      status.appendChild(retry);
    } finally {
      pending = false;
    }
  };
  const sectionObserver = new IntersectionObserver(entries => {
    if (!entries.some(entry => entry.isIntersecting)) return;
    attempt();
  }, {rootMargin});
  sectionObserver.observe(target);
}
renderWhenNear("#score-scatter", async () => {
  scoreHistoryData = await loadDeferredData(
    "climate-risk-housing-county-history.js",
    "CLIMATE_RISK_HOUSING_COUNTY_HISTORY"
  );
  drawScoreScatter();
});
renderWhenNear("#pricing-grouping", async () => {
  await ensureGeography();
  startRatingSequence();
  ratingSectionRendered = true;
});
renderWhenNear("#events", async () => {
  await Promise.all([ensureGeography(), ensureEvents()]);
  renderEventSection();
  startRiskTimer();
  eventSectionRendered = true;
});
renderWhenNear("#features", async () => {
  await Promise.all([ensureEvents(), ensureFeatures()]);
  drawFeatureHeatmaps();
  featureSectionRendered = true;
});
renderWhenNear("#playbook", async () => {
  const [playbook] = await Promise.all([
    loadDeferredData("climate-risk-housing-playbook.js", "CLIMATE_RISK_HOUSING_PLAYBOOK"),
    ensureGeography(), ensureEvents(), ensureFeatures(),
  ]);
  DATA.playbook = playbook;
  playbookCountyByFips = new Map(DATA.playbook.counties.map(d => [d.fips, d]));
  initPlaybook();
  playbookSectionRendered = true;
  renderPlaybookFrame();
}, "1200px 0px");
let resizeTimer = null;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    if (scoreScatterRendered) drawScoreScatter();
    if (ratingSectionRendered) {
      drawRatingScatter();
      drawRatingMap();
    }
    if (eventSectionRendered) renderEventSection();
    if (featureSectionRendered) drawFeatureHeatmaps();
    if (playbookSectionRendered) {
      renderPlaybookFrame();
    }
    document.querySelectorAll(".slide[data-story-ready='true']").forEach(section => {
      syncTakeawaySpace(section, section.querySelector(".takeaway.story-active-takeaway"));
    });
  }, 150);
});
