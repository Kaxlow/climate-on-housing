# Climate Risk and Housing Methodology

## Purpose and scope

This document describes the data and analytical methods used to build **Which
Way the Wind Blows: Climate Risk and U.S. Housing Markets**
(`output/climate-risk-housing.html`). The production implementation is in:

- `src/housing_climate_risk/cli/download_data.py`
- `src/housing_climate_risk/cli/build_database.py`
- `src/housing_climate_risk/cli/feature_marts.py`
- `src/housing_climate_risk/cli/analysis_marts.py`
- `src/housing_climate_risk/event_deduplication.py`
- `src/housing_climate_risk/page_data/climate_risk_housing.py`
- `src/housing_climate_risk/page_data/event_windows.py`
- `config/data_sources.yaml`

The page explores the relationship between county housing-market performance, climate risk, and the occurrence of actual extreme climate events. It culminates in a climate playbook that informs the reader of what to watch out for when extreme climate events happen in future.

## Unit of analysis

The primary geographic unit is the U.S. county or county equivalent. Sources
are joined with five-character county FIPS codes, padded with leading zeroes as
needed. The pipeline uses monthly county housing and weather observations,
annual county characteristics, county-event-month event windows, and
county-level map summaries.

## Data sources

| Source | Use |
| --- | --- |
| [Redfin Data Center](https://www.redfin.com/news/data-center/downloads/) | Monthly county Housing Market Tracker, property-type, and price-drop measures; see Redfin's [methodology](https://www.redfin.com/news/data-center/methodology/) |
| FEMA National Risk Index (NRI) | Overall county risk scores and ratings |
| FEMA disaster declarations | Event locations, types, and dates |
| NOAA Storm Events | County storm events, dates, and estimated damage |
| NCEI Climate at a Glance | County monthly weather measures |
| U.S. Census Bureau American Community Survey | Economic, demographic, housing-cost, and affordability characteristics |
| StatsAmerica and underlying BEA/CEW series | Income, employment, earnings, and population change |
| U.S. Census Bureau cartographic boundary files | County and state geometry used to derive the local interactive-map GeoJSON |

Insurance premium and non-renewal fields are also materialized when their local
inputs are available. The database records source-file metadata in `meta.files`
and ACS variable mappings in `meta.acs_variable_features`. File metadata
includes the upstream source URL and a SHA-256 content hash. Climate-damage
lineage is also retained in
`data/climate_damage/climate_damage_source_manifest.csv`.

The repository excludes provider data. Populate the ignored local `data/`
workspace from the latest available provider releases with:

```powershell
download-data all
```

The bootstrap downloads Redfin, Census, FEMA, NOAA, and StatsAmerica inputs; selects the
latest annual vintage when a provider exposes versioned files; derives the NOAA
forecast-zone-to-county crosswalk; and validates required filenames and schemas.
It writes resolved URLs, provider versions, and UTC retrieval timestamps to the
ignored local `data/download_receipt.yaml`. Metadata and expected schemas are
committed in `config/data_sources.yaml`. Mutable APIs and unversioned downloads
can change after retrieval, affecting future reproducibility.

The county FIPS master at `data/fipsgeo/fips_master_v2.csv` is committed with
the repository. The county processed Feather snapshot is optional;
when absent, its private insurance premium and non-renewal features are not
available. `download-data all` reports missing manual inputs together at the
end.

## Database construction

`build-database` creates `data/quoll.duckdb` with five data layers, plus
metadata:

1. **Raw:** CSV and Feather extracts are loaded into `raw`, and source metadata
   is recorded. Census special-value codes and invalid negative values are
   converted to null where a field should be nonnegative. Legitimate negatives,
   such as changes, anomalies, temperatures, and year-over-year rates, remain.
2. **Reference:** `ref` contains normalized county and state identifiers.
3. **Mart:** `mart` contains analysis-ready Redfin, NRI, FEMA, NOAA, NCEI, ACS,
   insurance, population, income, and employment tables. Common county/date
   keys are indexed.
4. **Feature:** `feature` blends normalized variables into five domain marts:
   county economic, demographic, climate, housing, and risk data.
   `feature.catalog` is the authoritative inventory of feature definitions,
   units, sources, and temporal grains.
5. **Analysis:** `analysis` persists the canonical extreme-event cohort,
   county-event-month housing windows, event-window configuration, and aggregate
   summaries used by publication notebooks and downstream diagnostics.

`meta` records source-file and ACS-variable lineage. `build-database
--marts-only` rebuilds the reference, mart, feature, and analysis layers from
existing raw tables; it is not a clean-clone bootstrap. The page builder opens
DuckDB read-only.

## Climate-risk definitions

The page presents FEMA NRI risk. Source labels are normalized for display:

| FEMA label | Display group | Ordinal value |
| --- | --- | ---: |
| Very Low | Very Low | 1 |
| Relatively Low | Low | 2 |
| Relatively Moderate or Moderate | Medium | 3 |
| Relatively High | High | 4 |
| Very High | Very High | 5 |

Ordinal values support rank correlations and display logic. They do not imply
equal distance between adjacent FEMA categories.

## Housing-market outcome

The principal outcome is Redfin's **median sale price per square foot,
year-over-year change** (`MEDIAN_PPSF_YOY`) for `All Residential` properties.
The ingestion layer converts Redfin's percentages and percentage-point changes
to proportions. It combines the all-residential Housing Market Tracker with the
property-type file and joins the separate Price Drops file for all-residential
price-drop shares.

Historical charts show monthly observations at the county level across the latest ten
complete calendar years. The builder identifies the most recent year for which the
all-residential mart contains all 12 calendar months, then selects January 1 of the
ninth preceding year through January 1 following that latest complete year. Months
from a newer incomplete year remain in the mart but do not enter the ten-year
calendar-history charts.
Here, a complete year means that all 12 months occur somewhere in the
all-residential mart, not that every county has observations in those months.
The incomplete-year exclusion applies to calendar-history charts; event-relative
windows can use any available mart months needed for their selected horizon.
All present-day county and county-equivalent FIPS in the 50
states and District of Columbia are eligible; Puerto Rico, other U.S. territories,
state/aggregate codes ending in `000`, and legacy or special codes that do not match a
present-day county are excluded. An individual county line retains null months as
visible gaps. The NRI-group line plot uses only counties with a valid observation in
all 120 displayed months. For that fixed cohort, the line is the monthly median and
its surrounding band is the monthly 25th–75th percentile interval.

The all-county history chart caps displayed values at the pooled 10th and 90th
percentiles of the eligible county-month observations. This is a display cap,
not deletion or imputation of observations, and does not change the underlying
values used for NRI-group summaries or Playbook comparisons. Nulls still break
the county lines.

## Disaster event selection

The event analysis combines:

- FEMA declarations with a valid county FIPS and incident start date; and
- NOAA storm events with a valid county FIPS and start date and at least **$1
  billion** in recorded total damage.

The page's phrase "NOAA billion-dollar storm events" refers to this filter on
NOAA Storm Events records, not to a separate national billion-dollar-disasters
catalog. The threshold is applied before canonical cross-source merging.
Duplicate or related NOAA records are not summed to create a qualifying event;
canonicalization retains the maximum contributing damage amount.

FEMA types Biological, Dam/Levee Break, Chemical, Terrorist, Other, and Toxic
Substances are excluded from the intended climate and destructive-weather
scope. Raw FEMA and NOAA tables preserve duplicate provider entries; semantic
event deduplication is not applied during raw loading.
Deduplication begins at the mart boundary. `mart.fema_disaster_declarations`
normalizes declaration titles and clusters records for the same county when
their like-named incident periods overlap or are separated by no more than seven
days. When emergency (`EM`) and major-disaster (`DR`) declarations describe that
incident, the `DR` declaration is retained as its canonical record. The
canonical interval spans the contributing records, and the mart retains source
row counts plus lists of associated disaster numbers, declaration types, and
source record IDs.

`mart.noaa_storm_events` removes repeated source records with the same event ID,
county, type, and interval. Qualifying NOAA billion-dollar records are then
compared with canonical FEMA incidents in `mart.climate_events`. Records in the
same county and broad event family are treated as one physical incident when
their periods overlap within three days; FEMA is the canonical display record,
while NOAA damage and both sources' identifiers are retained as lineage. NOAA
records from the same county and NOAA episode are also treated as one incident.
All feature, analysis, page, and Climate Playbook event processing reads this
unified canonical mart.
A missing event end is set to its start; an end before the start causes the
record to be removed. Dates are reduced to calendar months. Each canonical
county-incident receives a unique key, and its contributing provider keys remain
available in `associated_source_event_keys`. The page retains events starting
within the same latest-ten-complete-calendar-years period used for the housing
histories.

That dynamically derived period defines the page view. The reusable `analysis`
layer also persists window summaries for post-event horizons of 12,
24, 36, 48, and 60 months when the available housing coverage permits them.

## Event-window analysis

Events are matched to Redfin observations for the same county once, using the
maximum range required by the page: 12 months before event start through 60
months after event end. Three analytical views are derived from that shared
county-event-month table:

- **Pre-event view:** 12 months before the event start through its start month.
- **Window A:** 12 months before the event start through its start month, then
  months 1–36 after the event end.
- **Window B:** the same start-relative 12-month pre-event observations, then
  months 1–60 measured after the event end.

All views use a split-anchored month index: nonpositive months are measured
from the event start, while positive months are measured from the event end.
The shared maximum affected-event intermediate retains the 12 pre-start months
required by all display views and up to 60 months after the event end. That table
is passed to both the event-frame and feature-analysis payload builders. Month 0
is the event-start month. For multi-month events, subsequent months through the
event-end month are omitted; positive month 1 is the month after the event end.

The charts grouped by NRI risk rating include only county-event trajectories with
a non-null median PPSF year-over-year observation in every displayed month: months
-12 through 0 for the pre-event frame, months -12 through 0 and 1 through 36 for
Window A, or months -12 through 0 and 1 through 60 for Window B. Each trajectory
corresponds to a unique county-event observation. For the resulting fixed cohort,
the median and interquartile range are calculated at each relative month across
county-event trajectories. A county with several qualifying events therefore
contributes several observations at a relative month. Map/legend county counts
count unique counties, not trajectories.

Completeness is evaluated independently for each view: 13 observations for the
pre-event view, 49 for Window A, and 73 for Window B. Window A does **not** require
five years of post-event data. Its cohort can be larger than Window B's, so its
aggregates are **not generally equal** to a subset of Window B's aggregates.
Only the raw maximum intermediate is shared. Affected-county counts, county-event
averages, percentile ranks, and example selections are recomputed for each view.
The pre-event frame displays median lines without IQR bands; A and B display both.

The overview counts distinct counties with at least one qualifying canonical
event starting in the ten-year period, regardless of housing-window completeness.
The page's event-context filter also requires a matching county record in the
NRI mart, although that record need not have a usable risk-rating label.
Its percentage denominator is the current county/county-equivalent set in the
50 states and DC. The bar chart counts those affected counties by NRI rating.
Consequently, overview counts need not match any line-plot cohort.

## Within-risk-group feature analysis

This section explores which economic and demographic county characteristics
accompany stronger or weaker housing growth among counties in the same overall
risk group. Production features are supplied through the `feature` domain marts
and the definitions in `feature.catalog`. Their underlying sources include ACS
and StatsAmerica's BEA series. The cataloged homeowners-insurance affordability
measure is derived from ACS. Features are primarily county averages over the
latest ten years available in each applicable mart. Some measures are
constructed from related fields, such as weighted midpoints of ACS cost buckets.

Displayed income components are annual BEA amounts per county resident: net
earnings by place of residence; dividends, interest, and rent; and transfer
receipts. The homeowners-insurance, property-tax, and utility cost shares use county median annual household income as their denominator. Each is averaged
over the latest ten years available in its mart before the county comparison.

The feature-analysis cohort is the same cohort used by the displayed three-year
event-window line plot: NRI-rated counties with at least one complete trajectory
covering months -12 through 0 and 1 through 36. Incomplete trajectories are excluded,
even when another trajectory from the same county qualifies. Feature-specific
correlations additionally require a non-null value for that feature.

If a county had multiple unique events, the target Median PPSF YoY used to compute feature correlations is calculated by taking the median across a county's complete
events **at each relative month**, then take the median of those 49 monthly
medians. Each county contributes one pair to each feature correlation. Note that overlapping event windows can still reuse the same calendar observation.

For each NRI risk group, the page ranks retained features by the absolute
Spearman correlation between the county feature value and the county's
median-of-monthly-medians housing target defined above. The sign indicates whether the
descriptive relationship is positive or negative. Scatterplots exclude points
outside the 1.5-IQR fences on **either** the feature or the housing target and add
a linear trend fitted to the displayed points. This display trimming does not
change the Spearman coefficient computed from the original non-null pairs.
Spearman requires at least three pairs and variation in both variables.

A 95% percentile interval is stored when there are at least 12 pairs. The current
implementation draws 160 bootstrap samples of the already-computed paired ranks
and correlates those sampled ranks; it does not re-rank within each replicate.
Seeds are fixed by risk and feature. The stored `passesThreshold` flag is true
when the interval lies entirely above +0.10 or below -0.10, but this flag does
**not** filter the displayed feature ranking or significant-factor selection.
Features with absolute point correlation greater than or equal to 0.30 are selected as significant
factors for the page. If fewer than three meet that threshold within a risk group,
add factors in descending absolute-correlation order until three are selected.
If fewer than three finite correlations exist, use all that are available.
"Significant" is therefore a page-selection label, not a hypothesis-test result:
top-up features can have an absolute correlation below 0.30.

To obtain the county performance view, the analysis uses the same complete
county-event trajectories selected for the one-year-before through three-years-after
line plot. If a county had multiple unique events, those trajectories are collapsed into a single, median value at each relative
month, then the county's target Median PPSF YoY is computed by taking the median of those 49 monthly medians. The target is computed
once and reused, rather than separately pooling all event-month observations.
Overlapping windows can still repeat a calendar observation. Counties are sorted
in descending order by this target within their NRI group and divided
deterministically into four approximately equal-sized
groups: Strong Overperformers, Mild Overperformers, Mild Underperformers, and
Strong Underperformers. Ties are
resolved by FIPS for stable assignment. Counties without qualifying events or
without at least one complete event-window trajectory are not assigned a subgroup in this feature analysis,
their subgroup assignment comes later in [County Climate Playbook](#county-climate-playbook).

The performer-group trend chart first takes each county's median across complete
events at each relative month, then the median across member counties. This gives
counties equal weight at each month.

The companion distribution plot shows selected significant features for the
active subgroup. Values outside the risk-group 1.5-IQR fences are hidden except
for Very High, where they remain visible. The takeaway is feature-wide: negative
correlation means higher feature values are associated with poorer growth;
positive correlation means lower values are associated with poorer growth.

The list of defining factors for a performer subgroup within a risk group assigns a "higher" or "lower" label
to each significant factor. The label is determined by the feature's correlation sign and subgroup:

| Correlation sign | Overperformer subgroup | Underperformer subgroup |
| --- | --- | --- |
| Positive | Higher | Lower |
| Negative | Lower | Higher |

## County Climate Playbook

### Frame sequence and coverage gate

1. **County search/map:** select a county from the U.S. map or search results.
2. **History/local map:** display the county's ten-year Median PPSF YoY history
   without event overlays, alongside a zoomed local map.
3. **Past events:** retain the history plot, add canonical event overlays, and
   show the overall NRI rating and scrollable event list on the right.
4. **Peer comparison:** overlay the risk group's monthly median and IQR on the
   county history and zoom in on the plot to compare if needed. The expanded NRI card
   displays the county's performer subgroup, with a tooltip explaining the
   assignment method.
5. **Top factors and scorecard:** a series of cards highlighting the top factors associated
   with the county's performance and a scorecard that concludes the county's likely performance
   when an extreme climate event occurs.

A county is not assigned a performer subgroup if it does not have non-null Median PPSF YoY values
for at least 50% of the months across the past 10 calendar years. In this case, peer comparison in
terms of performance is not available, and neither can a conclusion be drawn about its top factors
and likely performance when an extreme climate event occurs, thus for such counties frame 5 is not
applicable.

### Performer assignment in Frame 4

The Playbook first reuses the feature-analysis subgroup for a county with a
qualifying event and an observed complete-window assignment. Otherwise, it uses
a historical fallback for the Playbook only; it does not add that county to the
feature-analysis training/comparison cohort.

Define these summaries over the selected ten-year history:

| Symbol | Summary |
| --- | --- |
| `a` | Median of the selected county's non-null monthly Median PPSF YoY values |
| `b` | Median across months of its risk group's monthly county median |
| `c` | Median across months of its risk group's monthly county 75th percentile |
| `d` | Median across months of its risk group's monthly county 25th percentile |

At each month, risk-group quartiles use all available finite observations in
the Playbook's risk group, including the selected county. Peer counties need not
have complete histories or pass the selected county's 50% gate. Months without
observations are omitted from the relevant across-month median. These are
**medians of monthly summaries**, not pooled quantiles over county-month records.
They differ from the earlier climate-risk history plot, whose cohort must have
all 120 observations.

The fallback rules are:

| Condition | Assignment |
| --- | --- |
| `a > b` and `a - b < 0.5 * (c - b)` | Mild Overperformer |
| `a > b` and `a - b >= 0.5 * (c - b)` | Strong Overperformer |
| `a < b` and `b - a < 0.5 * (b - d)` | Mild Underperformer |
| `a < b` and `b - a >= 0.5 * (b - d)` | Strong Underperformer |
| `a == b` | Mild Overperformer (implementation tie convention) |
| Any of `a`, `b`, `c`, `d` unavailable | No fallback assignment |

### Top factor cards in Frame 5

For each significant factor, the county's value is compared against the median of all counties within the same risk group. 
The first arrow is up for a value strictly above the
peer median and down otherwise. A second arrow next to "Median PPSF YoY" combines
that position with the sign of the complete-event-cohort correlation:

| County feature position | Positive correlation | Negative correlation |
| --- | --- | --- |
| Above peer median | Growth arrow up | Growth arrow down |
| At or below peer median | Growth arrow down | Growth arrow up |

An exact median tie takes the below-median display branch. A zero correlation
produces a down growth arrow. Missing
feature values or non-finite correlations display "Data unavailable" instead
of arrows. Tooltips provide qualitative context and the correlation direction.

### County performance scorecard

Three component cards each contribute one up/down signal:

| Card | Up signal | Down signal |
| --- | --- | --- |
| NRI risk rating | Very Low or Low | Medium, High, or Very High |
| Performance subgroup | Mild or Strong Overperformer | Mild or Strong Underperformer |
| Significant county data | Strict majority of selected features match overperformer subgroups | Half or fewer match overperformer subgroups |

For the third signal, the implementation compares each county feature with the
observed feature ranges of the event-cohort performer subgroups within its NRI
risk group:

1. Prefer subgroups whose inclusive minimum–maximum range contains the value.
2. If ranges overlap, choose the subgroup whose median is closest to the value.
3. If no range contains it, choose the nearest range boundary, then the nearest
   subgroup median to break a distance tie.
4. An exact overperformer/underperformer tie favors underperformers.

The three equally weighted signals produce the displayed overall score:

| Up signals | Display label | Display takeaway |
| ---: | --- | --- |
| 3 | Low Risk | Safer environment that can benefit from steps to reduce potential climate damage |
| 2 | Moderate Risk | Climate damage is a real possibility, know the steps to reduce it |
| 0 or 1 | High Risk | Take steps to reduce climate damage |

If required feature comparisons are unavailable, the overall card instead shows
"Insufficient data".

This score is a **rule-based presentation heuristic**, rather than a
trained prediction model, an official FEMA rating, or a calibrated estimate of
climate damage or housing returns. Its Low/Moderate/High labels are distinct
from the five-category NRI scale.

## Geography and generated page

Only geometries represented in the playbook data enter the page payload.
Geometry is simplified while preserving topology (with a larger tolerance for
Alaska), and polygon orientation is normalized for browser rendering. State
outlines are dissolved from unsimplified county geometries, stripped of interior
rings, and only then simplified. This preserves the exterior state perimeter
without turning county-level simplification gaps into internal state lines.

`build-climate-risk-housing` queries the marts, constructs the analytical
payloads, and embeds filtered GeoJSON. It writes a three-file publication
bundle:

- `output/climate-risk-housing.html`
- `output/climate-risk-housing-county-history.js`
- `output/climate-risk-housing-playbook.js`

The JavaScript files hold deferred county-history and Climate Playbook payloads
and must remain beside the HTML file when it is opened or published. The output
does not query DuckDB at runtime. D3 and Google Fonts are its external browser
resources.

## Limitations

- **Association, not causation:** interest rates, migration, housing supply,
  income, insurance, policy, and other factors are not isolated.
- **Uneven source coverage:** missing Redfin, NRI, event, or feature data changes
  the sample in each view.
- **Performer cohort and fallback:** the feature-analysis subgroups represent only
  event-affected counties with complete event-window housing data. A Playbook-only
  fallback compares a county's ten-year median with its risk group's typical
  monthly quartiles. Direct assignments and fallback labels use different periods
  and definitions and should not be interpreted as interchangeable event responses.
  The 50% Playbook coverage threshold is a display eligibility rule, not validation
  that the observed months represent the missing months.
- **Event duplication and overlap:** Despite deduplication efforts, some FEMA and
  NOAA records may represent the same event. Event windows can overlap, and
  observations are not necessarily independent.
- **Damage threshold:** NOAA selection depends on a $1 billion cutoff and the
  accuracy and completeness of recorded damage.
- **County aggregation:** county summaries conceal neighborhood-level exposure
  and market variation.
- **Risk measurement:** NRI is FEMA's modeled expected-risk summary, not a direct
  measure of a particular event's local severity. The current NRI snapshot is
  also applied to historical housing and event periods, so it should not be read
  as a contemporaneous historical risk rating.
- **Year-over-year outcome:** adjacent monthly observations share information
  because each compares with the previous year.
- **Correlation-based feature ranking:** rank and direction depend on the
  available features, aggregation window, and counties represented in each risk
  group. Correlation does not establish a causal contribution. Feature averages
  may cover different calendar years from the housing/event data and are not
  restricted to pre-event information. Top-up selection can include weak
  correlations; the stored bootstrap interval does not control the selection.
- **Playbook arrows and overall score:** group-level associations need not match
  an individual county's historical performance. Median-based factor arrows and
  performer-range matching use different reference rules. The equal-weight score
  is not a validated classifier or a calibrated risk probability, and cannot
  establish the damage claims implied by its short display takeaways.
- **Mutable Redfin source:** Redfin's public county files can be revised. The
  download receipt records retrieval time and provider response metadata, but a
  later clean rebuild may not reproduce byte-identical source data.
- **Optional insurance input:** insurance premium and non-renewal features are
  unavailable when the private county Feather snapshot is not supplied.
- **Dynamic Redfin analysis period:** Redfin's mutable files can add or revise
  observations. A rebuild retains all available mart history and moves the page's
  ten-year window forward when a newer complete calendar year is present, so
  results and the displayed period can change over time.
- **Static extracts:** results can change when inputs are revised and rebuilt.

## Reproduction

From the repository root:

```powershell
pip install -e .
download-data all
build-database
build-climate-risk-housing
```

Set `CENSUS_API_KEY` before running the bootstrap. The committed FIPS master
requires no manual retrieval. The private
`county_processed_data.feather` input is optional and only adds its insurance
features.

If an existing `data/quoll.duckdb` already contains current raw tables:

```powershell
build-database --marts-only
build-climate-risk-housing
```

`--marts-only` cannot initialize a fresh clone because it depends on those
existing raw tables.

Viewing or publishing the committed page does not require DuckDB or any source
data. It requires the HTML file and both deferred JavaScript payloads listed
above. Full reproduction selects the latest available public data,
so provider revisions can change future results. Version-specific URLs and
retrieval metadata are recorded where available; exact historical reproduction
is not guaranteed for mutable APIs or unversioned downloads.

## Maintenance

This document describes the current implementation. Changes to date ranges,
event filters, feature definitions, completeness rules, database-layer
contracts, model inputs, source-manifest metadata, or aggregations should update
this document in the same change. Publication changes must also keep the HTML
and its two deferred JavaScript artifacts synchronized.
