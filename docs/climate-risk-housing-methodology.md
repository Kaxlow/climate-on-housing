# Climate Risk and Housing Methodology

## Purpose and scope

This document describes the data and analytical methods used to build **Which
Way the Wind Blows: Climate Risk and U.S. Housing Markets**
(`output/climate-risk-housing.html`). The production implementation is in:

- `src/housing_climate_risk/cli/download_data.py`
- `src/housing_climate_risk/cli/build_database.py`
- `src/housing_climate_risk/cli/feature_marts.py`
- `src/housing_climate_risk/cli/analysis_marts.py`
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
can change after retrieval, affecting future reproducability.

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
from a newer incomplete year remain in the mart but do not enter the page analysis.
All present-day county and county-equivalent FIPS in the 50
states and District of Columbia are eligible; Puerto Rico, other U.S. territories,
state/aggregate codes ending in `000`, and legacy or special codes that do not match a
present-day county are excluded. An individual county line retains null months as
visible gaps. The NRI-group line plot uses only counties with a valid observation in
all 120 displayed months. For that fixed cohort, the line is the monthly median and
its surrounding band is the monthly 25th–75th percentile interval.

## Disaster event selection

The event analysis combines:

- FEMA declarations with a valid county FIPS and incident start date; and
- NOAA storm events with a valid county FIPS and start date and at least **$1
  billion** in recorded total damage.

FEMA types Biological, Dam/Levee Break, Chemical, Terrorist, Other, and Toxic
Substances are excluded from the intended climate and destructive-weather
scope. The raw FEMA declaration table preserves every provider row.
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
is passed to both the event-frame and feature-analysis payload builders. Months between the event's start and end are not part of any display window.

The charts grouped by NRI risk rating include only county-event trajectories with
a non-null median PPSF year-over-year observation in every displayed month: months
-12 through 0 for the pre-event frame, months -12 through 0 and 1 through 36 for
Window A, or months -12 through 0 and 1 through 60 for Window B. Each trajectory
corresponds to a unique county-event observation. For the resulting fixed cohort,
the median and interquartile range are calculated at each relative month. Although
the monthly Window A aggregates are equivalent to the matching subset of Window B,
window-dependent affected-county counts, county averages, percentile ranks, and
example selections are recalculated for the applicable view.

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
event-window line plot: NRI-rated counties that had an event and a valid Median PPSF YoY observation in every month from month -12 through
the event start (month 0) and months 1 through 36 after the event end. If a county had multiple events and complete event-window data for those events, it is represented by a single event window whose Median PPSF YoY values are aggregated across the multiple events by taking the median of all monthly observations. It should be noted that the county's multiple event windows may overlap and lead to repeated Median PPSF YoY observations.

For each NRI risk group, the page ranks retained features by the absolute
Spearman correlation between the county feature value and the median of the county's median PPSF YoY across the one-year-before through three-years-after event window. The sign of the correlation indicates whether the
descriptive relationship is positive or negative. Scatterplots trim feature
outliers using the interquartile range rule and add an overall linear trend
line. A 95% percentile confidence interval is calculated from 160 bootstrap
samples drawn with replacement. A feature clears the minimum-effect filter only
when that interval lies entirely above +0.10 or below -0.10. Features with
absolute point correlation greater than or equal to 0.30 receive the strongest-correlation
visual treatment.

To obtain the county performance view, the analysis uses the same complete
county-event trajectories selected for the one-year-before through three-years-after
line plot. Each included county is represented by the median of all its monthly
Median PPSF YoY observations pooled across those complete trajectories. Overlapping windows
can repeat a calendar observation, as before. Counties are sorted in descending
order by that median within their NRI group and divided deterministically into four approximately equal-sized
groups: Strong Overperformers, Mild Overperformers, Mild Underperformers, and
Strong Underperformers. These are labelled as "Weak", "Mildly Weak", "Mildly Strong", and "Strong" respectively in `climate-risk-housing.html`. Ties are resolved by FIPS for stable assignment. The smaller Very High-risk sample is divided into three groups: Overperformers,
Average Performers, and Underperformers. Counties without qualifying events or
without at least one complete event-window trajectory are not assigned a subgroup in this feature analysis.

The performer-group trend chart describes housing performance around
events for each performer group within each risk group. Each performer group is represented by a trend line which is the month-level median across all group members. The companion
distribution plot shows strongly correlated feature values for the selected
performance group after excluding values beyond 1.5 times the risk group's
interquartile range.

## County Climate Playbook

Through a county search interface in the first frame, the user selects a target county.

The second frame shows the selected county's historic Median PPSF YoY line plot, with missing monthly data appearing as breaks in the line.

The third frame displays the county's NRI risk and past extreme weather events over the last ten years.

The fourth frame overlays the selected risk group's monthly median and IQR, and describes the relative position of the county's Median PPSF YoY within its risk group over the last ten years. The relative position is determined by the following rules:
- Upper Range: Closer to 75th percentile than median, or above 75th percentile
- Lower Range: Closer to 25th percentile than median, or below 25th percentile
- Mid Range: Everything else

The fifth frame first assigns performer groups to counties that were not given one in the [Within-risk-group feature analysis](#within-risk-group-feature-analysis) section because they lacked an event-window with complete monthly Median PPSF YoY across that period. The performer group is determined through this method:
1. Compute the following:
- County's median of its non-null median PPSF YoY values over the last ten years
- County's risk group's medians of:
    - its monthly median values of median PPSF YoY over the last 10 years
    - its monthly 75th percentile values of median PPSF YoY over the last 10 years
    - its monthly 25th percentile values of median PPSF YoY over the last 10 years
2. Assign the performer group based on these rules:
- If county's median is above risk group's median and closer to the group's 75th percentile than the group's median, or greater than the group's 75th percentile, it is a Strong Overperformer
- If county's median is above the risk group's median and is closer to the group's median than the group's 75th percentile, it is a Mild Overperformer
- If county's median is below the risk group's median and is closer to the group's median than the group's 25th percentile, it is a Mild Underperformer
- If county's median is below risk group's median and closer to the group's 25th percentile than the group's median, or less than the group's 25th percentile, it is a Strong Underperformer
Next, it indicates the selected county's performer group within its risk group.
Finally, a dashboard highlights the factors significantly correlated to Median PPSF YoY for the county's risk group, and indicates the movement direction for each factor that is associated with decreasing Median PPSF YoY.

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
  event-affected counties with observed event-window housing data. A Playbook-only
  fallback compares a county's ten-year median with its risk group's typical
  monthly quartiles. Direct assignments and fallback labels use different periods
  and definitions and should not be interpreted as interchangeable event responses.
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
  group. Correlation does not establish a causal contribution.
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
