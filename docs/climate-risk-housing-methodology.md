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
| [Redfin Data Center](https://www.redfin.com/news/data-center/downloads/) | Monthly county Housing Market Tracker, property-type, and price-drop measures; see Redfin's [methodology](https://www.redfin.com/news/data-center/methodology/) |
| FEMA National Risk Index (NRI) | Overall and hazard-specific county risk scores and ratings |
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
can change after retrieval, so later runs may not exactly reproduce earlier
results.

The county FIPS master at `data/fipsgeo/fips_master_v2.csv` is committed with
the repository. Redfin's mutable public county CSVs are downloaded into the
ignored local workspace. The county processed Feather snapshot is optional;
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

The principal outcome is Redfin's **median sale price per square foot,
year-over-year change** (`MEDIAN_PPSF_YOY`) for `All Residential` properties.
The ingestion layer converts Redfin's percentages and percentage-point changes
to proportions. It combines the all-residential Housing Market Tracker with the
property-type file and joins the separate Price Drops file for all-residential
price-drop shares. The mart is restricted to January 2012 through December 2025,
matching the former extract's period.

Historical charts show monthly observations at the county level across the latest ten
complete calendar years. In charts where counties are grouped by NRI risk rating, only counties with complete monthly observations across the corresponding period are included. For each group, counties are aggregated and the median and 25th–75th percentile interval are calculated for each month. The median, 25th, and 75th percentile values over time are used to create a line plot with a surrounding band that represents the group's house price growth trajectory over time.

## Disaster event selection

The event analysis combines:

- FEMA declarations with a valid county FIPS and incident start date; and
- NOAA storm events with a valid county FIPS and start date and at least **$1
  billion** in recorded total damage.

FEMA types Biological, Dam/Levee Break, Chemical, Terrorist, Other, and Toxic
Substances are excluded from the intended climate and destructive-weather
scope. The raw FEMA declaration table preserves the source rows, while
`mart.fema_disaster_declarations` is incident-level: declarations with the same
county, incident type, title, start date, and end date are treated as one
incident. When both emergency (`EM`) and major-disaster (`DR`) declarations
describe that incident, the `DR` declaration is retained as its canonical
record; the shared incident dates are unchanged. The mart also retains the
declaration count and lists of associated disaster numbers and declaration
types.
A missing event end is set to its start; an end before the start causes the
record to be removed. Dates are reduced to calendar months. Each
county-source-event-start combination receives a unique key. The page retains
events starting within the same latest-ten-complete-calendar-years period used
for the housing histories.

That dynamically derived period defines the page view. The reusable `analysis`
layer also persists complete-window summaries for post-event horizons of 12,
24, 36, 48, and 60 months when the available housing coverage permits them.

## Event-window analysis

Events are matched to Redfin observations for the same county. Two windows are
built around median PPSF year-over-year change:

- **Window A:** 12 months before the event start through its start month, then
  months 1–36 after the event end.
- **Window B:** the same start-relative 12-month pre-event observations, then
  months 1–60 measured after the event end.

Both windows use a split-anchored month index: nonpositive months are measured
from the event start, while positive months are measured from the event end.
The raw page-data build retains the 12 pre-start months required by both display
windows and up to 60 months after the event end. Months between the event's
start and end are not part of either display window.

The charts grouped by NRI risk rating include only county-event trajectories
with a non-null median PPSF year-over-year value at every required month in the
corresponding window: months -12 through 0 and 1 through 36 for Window A, or
months -12 through 0 and 1 through 60 for Window B. Completeness is evaluated
separately for each window, so a county-event trajectory can qualify for Window
A but not Window B.

At each relative month, trajectories are grouped by overall NRI rating. The
page reports their median and interquartile range. A county associated with
multiple qualifying events can contribute multiple trajectories.

To illustrate within-group differences, the page compares two configured focus
county-event lines for each risk group when those lines meet coverage and
feature-eligibility requirements. Selection logic supplies eligible fallbacks
when a configured example is unavailable. These examples are descriptive rather than being statistically significant in terms of difference.

## Within-risk-group feature analysis

This section explores which county characteristics accompany stronger or
weaker housing growth among counties in the same overall risk group. Production
features are supplied through the `feature` domain marts and the definitions in
`feature.catalog`. Their underlying sources include ACS, StatsAmerica and its
BEA/CEW series, NCEI weather, NRI risk, and Redfin housing. Optional private
insurance extracts are materialized separately when present; the cataloged
homeowners-insurance affordability measure is derived from ACS. Features are
primarily county averages over the latest ten years available in each
applicable mart. Some measures are constructed from related fields, such as
weighted midpoints of ACS cost buckets.

Displayed income components are annual BEA amounts per county resident: net
earnings by place of residence; dividends, interest, and rent; and transfer
receipts. The homeowners-insurance, property-tax, and utility cost shares instead
use county median annual household income as their denominator. Each is averaged
over the latest ten years available in its mart before the county comparison.

The feature-analysis outcome is each county's arithmetic mean of monthly median
PPSF YoY across all of its complete event-window observations: month -12 through
the event start (month 0) and months 1 through 36 after the event end. Every
county contributes one observation to this analysis. Because all qualifying
events have the same complete window, multiple events receive equal weight
within a county; overlapping event windows can repeat the same calendar housing
observation.

For each NRI risk group, the page ranks retained features by the absolute
Spearman correlation between the county feature value and this county-level
average PPSF YoY around events. The sign of the correlation indicates whether the
descriptive relationship is positive or negative. Scatterplots trim feature
outliers using the interquartile range rule and add an overall linear trend
line.

Counties within each risk group are divided into four ordered subgroups using
the most prominent feature patterns. The page reports the middle 50% (IQR) of
the leading feature values for the selected subgroup. These associations and
subgroups are descriptive, not predictions or causal estimates.

## County Climate Playbook

The lookup combines overall and hazard-specific NRI measures, the leading
within-risk-group county features and subgroup assignment, monthly county median
PPSF year-over-year history, and qualifying FEMA/NOAA event periods from
2016–2025. Its comparison view overlays the selected risk group's monthly median
and IQR. Event narratives assess how closely the county's post-event change
matches its risk-group expectation and how closely its relative PPSF YoY level
matches the feature-subgroup expectation shown in the profile. Highly volatile
histories are reported as inconclusive. Counties without a qualifying event
receive a risk-group and subgroup-based expectation instead.

## Geography and generated page

Only geometries represented in the playbook data enter the page payload.
Geometry is simplified while preserving topology (with a larger tolerance for
Alaska), and polygon orientation is normalized for browser rendering. State
outlines are dissolved from the displayed county geometries so both boundary
layers use exactly the same edges.

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
- **Complete-case selection:** requiring every event-window month can materially
  reduce and bias longer-horizon samples.
- **Event duplication and overlap:** FEMA and NOAA can represent the same event,
  event windows can overlap, and observations are not necessarily independent.
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
- **Illustrative examples:** configured county-event examples and fallback
  selection are descriptive; their differences are not formal significance
  findings.
- **Mutable Redfin source:** Redfin's public county files can be revised. The
  download receipt records retrieval time and provider response metadata, but a
  later clean rebuild may not reproduce byte-identical source data.
- **Optional insurance input:** insurance premium and non-renewal features are
  unavailable when the private county Feather snapshot is not supplied.
- **Fixed Redfin analysis period:** the downloaded Redfin files may contain newer
  observations, but the housing mart remains fixed at January 2012 through
  December 2025 to preserve the analysis period requested for this publication.
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
requires no manual retrieval. Redfin county housing data is fetched by
`download-data all` from the public Download Hub. The private
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
above. Full analytical reproduction selects the latest available public data,
so provider revisions can change future results. Version-specific URLs and
retrieval metadata are recorded where available; exact historical reproduction
is not guaranteed for mutable APIs or unversioned downloads.

## Maintenance

This document describes the current implementation. Changes to date ranges,
event filters, feature definitions, completeness rules, database-layer
contracts, model inputs, source-manifest metadata, or aggregations should update
this document in the same change. Publication changes must also keep the HTML
and its two deferred JavaScript artifacts synchronized.
