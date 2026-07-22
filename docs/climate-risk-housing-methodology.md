# Climate Risk and Housing Methodology

## Purpose and scope

This document describes the data and analytical methods used to build **Which
Way the Wind Blows: Climate Risk and U.S. Housing Markets**
(`output/climate-risk-housing.html`). The production implementation is in:

- `src/housing_climate_risk/cli/build_database.py`
- `src/housing_climate_risk/page_data/climate_risk_housing.py`
- `src/housing_climate_risk/page_data/event_windows.py`

The page explores whether county housing-market performance varies with
measured climate risk and around major disaster events. It is descriptive and
exploratory. Its associations are not causal estimates of the effect of climate
risk or disasters on housing prices.

## Unit of analysis

The primary geographic unit is the U.S. county or county equivalent. Sources
are joined with five-character county FIPS codes, padded with leading zeroes as
needed. The pipeline uses monthly county housing and weather observations,
annual county characteristics, county-event-month event windows, and
county-level map summaries.

Coverage varies by source and measure. A county enters a view only when it has
the fields that view requires, so sample sizes can differ between charts.

## Data sources

| Source | Use |
| --- | --- |
| Redfin | Monthly county housing-market measures |
| FEMA National Risk Index (NRI) | Overall and hazard-specific county risk scores and ratings |
| FEMA disaster declarations | Event locations, types, and dates |
| NOAA Storm Events | County storm events, dates, and estimated damage |
| NCEI Climate at a Glance | County monthly weather measures |
| U.S. Census Bureau American Community Survey | Economic, demographic, housing-cost, and affordability characteristics |
| StatsAmerica and underlying BEA/CEW series | Income, employment, earnings, and population change |
| Local county boundary GeoJSON | Interactive map geometry |

Insurance premium and non-renewal fields are also materialized when their local
inputs are available. The database records source-file metadata in `meta.files`
and ACS variable mappings in `meta.acs_variable_features`.

The page uses retained extracts under `data/` rather than fetching live data.
It therefore reflects the source versions present when the database was last
rebuilt.

## Database construction

`build-database` creates `data/quoll.duckdb` in three analytical layers:

1. **Raw:** CSV and Feather extracts are loaded into `raw`, and source metadata
   is recorded. Census special-value codes and invalid negative values are
   converted to null where a field should be nonnegative. Legitimate negatives,
   such as changes, anomalies, temperatures, and year-over-year rates, remain.
2. **Reference:** `ref` contains normalized county and state identifiers.
3. **Mart:** `mart` contains analysis-ready Redfin, NRI, FEMA, NOAA, NCEI, ACS,
   insurance, population, income, and employment tables. Common county/date
   keys are indexed.

`build-database --marts-only` rebuilds reference and mart tables from existing
raw tables. The page builder opens DuckDB read-only.

## Climate-risk definitions

The page presents overall FEMA NRI risk and five hazard-specific measures:
riverine flooding, tornado, wildfire, hail, and earthquake. Source labels are
normalized for display:

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

The principal outcome is Redfin's **median price per square foot,
year-over-year change** (`MEDIAN_PPSF_YOY`) for `All Residential` properties.
Values are parsed as numeric, and sentinel values at or below `-888888000` are
treated as missing.

Historical pricing charts show monthly observations from January 2016 through
December 2025. They include only counties with a valid `MEDIAN_PPSF_YOY`
observation in all 120 months. The county-history chart draws each eligible
county's monthly series. The risk-group chart displays the median and
25th–75th percentile interval across eligible counties for every risk group and
month.

## Disaster event selection

The event analysis combines:

- FEMA declarations with a valid county FIPS and incident start date; and
- NOAA storm events with a valid county FIPS and start date and at least **$1
  billion** in recorded total damage.

FEMA types Biological, Dam/Levee Break, Chemical, Terrorist, Other, and Toxic
Substances are excluded from the intended climate and destructive-weather
scope. A missing event end is set to its start; an end before the start causes
the record to be removed. Dates are reduced to calendar months. Each
county-source-event-start combination receives a unique key. The page retains
events starting from January 2016 through December 2025.

## Event-window analysis

Events are matched to Redfin observations for the same county. Two windows are
built around median PPSF year-over-year change:

- **Window A:** 12 months before the event start through its start month, then
  months 1–36 after the event end.
- **Window B:** the same 12-month pre-event period, then months 1–60 after the
  event end.

Month zero is the event start month. Positive months begin after the event end,
negative months are counted from the event start.

For each window, only county-event trajectories with a non-null value at every required month
are retained. This makes full lines comparable but favors counties and events
with continuous Redfin coverage.

At each relative month, trajectories are grouped by overall NRI rating. The
page reports their median and interquartile range. A county associated with
multiple qualifying events can contribute multiple trajectories.

To examine differences between counties in the same risk group, for each risk group, two example county lines that are algorithmically determined to be significantly distinct from one another are selected and their features are compared to illustrate the relationship between county features and house price growth.

## Within-risk-group feature analysis

This section explores which county characteristics accompany stronger or
weaker housing growth among counties in the same overall risk group. Candidate
features cover economics, migration and demographics, owner costs and
affordability, and weather. They are primarily county averages over the latest
ten years available in each mart. Some measures are constructed from related
fields, such as weighted midpoints of ACS cost buckets.

For complete county-event lines in the 12-month pre-event through 36-month
post-event window:

1. Housing growth is averaged over the full line.
2. The median line average is calculated within the risk group.
3. A line's **relative position** equals its average minus that group median.
4. Within each risk group, each feature's Spearman correlation with relative
   position is calculated when at least ten observations are available.
5. The ten features with the largest absolute correlations are shown.

Feature values are divided into up to five within-group quantile buckets, and
county feature percentiles are calculated within the group. The displayed
feature contribution combines centered percentile with the correlation's
direction and magnitude. It is a descriptive display score—not a regression
coefficient, causal contribution, or model feature importance.

County-event lines, rather than necessarily unique counties, form the
correlation sample, so counties with several events can receive more weight.
The feature search is exploratory; it has no multiple-testing adjustment or
uncertainty intervals.

## County Climate Playbook

The lookup combines overall and hazard-specific NRI measures, monthly county
median PPSF year-over-year history, and qualifying FEMA/NOAA event periods from
2016–2025. Rule-based narrative text compares observed movement with the broad
pattern expected for a risk group.

## Geography and generated page

Only geometries represented in the playbook data enter the page payload.
Geometry is simplified while preserving topology (with a larger tolerance for
Alaska), and polygon orientation is normalized for browser rendering.

`build-climate-risk-housing` queries the marts, constructs the analytical
payloads, embeds filtered GeoJSON, and writes one self-contained HTML file. The
output does not query DuckDB at runtime. D3 and Google Fonts are its external
browser resources.

## Limitations

- **Association, not causation:** interest rates, migration, housing supply,
  income, insurance, policy, and other factors are not isolated.
- **Uneven source coverage:** missing Redfin, NRI, event, or feature data changes
  the sample in each view.
- **Complete-case selection:** requiring every event-window month can materially
  reduce and bias longer-horizon samples.
- **Event duplication and overlap:** FEMA and NOAA can represent the same event,
  event windows can overlap, and observations are not necessarily independent.
- **Damage threshold:** NOAA selection depends on a $1 billion cutoff and the
  accuracy and completeness of recorded damage.
- **County aggregation:** county summaries conceal neighborhood-level exposure
  and market variation.
- **Risk measurement:** NRI is FEMA's modeled expected-risk summary, not a direct
  measure of a particular event's local severity.
- **Year-over-year outcome:** adjacent monthly observations share information
  because each compares with the previous year.
- **Exploratory feature ranking:** correlations are unadjusted, potentially
  confounded, and selected from many candidates.
- **Static extracts:** results can change when inputs are revised and rebuilt.

## Reproduction

From the repository root:

```powershell
pip install -e .
build-database
build-climate-risk-housing
```

If raw database tables are already current:

```powershell
build-database --marts-only
build-climate-risk-housing
```

The deliverable is written to `output/climate-risk-housing.html`.
Reproduction requires the retained files under `data/`; the repository alone
may not suffice if large or licensed inputs are distributed separately.

## Maintenance

This document describes the current implementation. Changes to date ranges,
event filters, feature definitions, completeness rules, or aggregations should
update this document in the same change.
