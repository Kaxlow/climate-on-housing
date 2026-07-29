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
- `src/housing_climate_risk/modeling/county_relative_ppsf/`
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
| Redfin | Monthly county housing-market measures |
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

The bootstrap downloads Census, FEMA, NOAA, and StatsAmerica inputs; selects the
latest annual vintage when a provider exposes versioned files; derives the NOAA
forecast-zone-to-county crosswalk; and validates required filenames and schemas.
It writes resolved URLs, provider versions, and UTC retrieval timestamps to the
ignored local `data/download_receipt.yaml`. Metadata and expected schemas are
committed in `config/data_sources.yaml`. Mutable APIs and unversioned downloads
can change after retrieval, so later runs may not exactly reproduce earlier
results.

The county FIPS master at `data/fipsgeo/fips_master_v2.csv` is committed with
the repository. The Redfin county extract is a required user-supplied input for
the housing marts and final page, but it is intentionally not distributed or
downloaded by this project. The county processed Feather snapshot is optional;
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

January 2016 through December 2025 is a fixed publication window, not a
dynamically advancing endpoint. Although the acquisition command selects the
latest provider data, downloading a newer release does not automatically extend
these charts.

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
events starting from January 2016 through December 2025.

The fixed dates above define the page view. The reusable `analysis` layer also
persists complete-window summaries for post-event horizons of 12, 24, 36, 48,
and 60 months when the available housing coverage permits them.

## Event-window analysis

Events are matched to Redfin observations for the same county. Two windows are
built around median PPSF year-over-year change:

- **Window A:** 12 months before the event start through its start month, then
  months 1–36 after the event end.
- **Window B:** the same start-relative 12-month pre-event observations, then
  months 1–60 measured after the event end.

For both displayed windows, nonpositive months are counted relative to the event
start and positive months begin after the event end. Window B is therefore
described in the implementation as end-anchored for its post-event segment; it
is not a single continuous month index around the event end. The raw join keeps
up to 24 pre-start months so events spanning multiple months still supply the
observations required by both display windows.

For each window, only county-event trajectories with a non-null value at every required month
are retained. This makes full lines comparable but favors counties and events
with continuous Redfin coverage.

At each relative month, trajectories are grouped by overall NRI rating. The
page reports their median and interquartile range. A county associated with
multiple qualifying events can contribute multiple trajectories.

To illustrate within-group differences, the page compares two configured focus
county-event lines for each risk group when those lines meet coverage and
feature-eligibility requirements. Selection logic supplies eligible fallbacks
when a configured example is unavailable. These examples are descriptive; no
statistical test establishes that the paired lines are significantly different.

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

Exploratory data analysis is performed in two stages. First, ACS and
StatsAmerica features are profiled separately to identify source-specific
missingness, duplicate county-year records, outliers, geographic and temporal
coverage, and changes in variable definitions. Units, denominators, dollar-year
bases, county FIPS codes, and aggregation periods are checked before features
are combined. This source-level review prevents join or definition problems
from being mistaken for substantive relationships.

Second, the domain marts are combined into the county feature matrix used for
within-risk-group analysis and county-relative PPSF modeling. This review
checks join coverage and unmatched counties, row loss or duplication,
missingness introduced by the merge, overlapping or contradictory measures
across sources, feature distributions within each risk group, and pairwise
correlations that may reveal redundancy. Descriptive relationships with housing
growth are evaluated only after these checks. Thus, source datasets are not
analyzed exclusively in isolation: separate profiling supports data validation,
while the blended dataset is the basis for the final analytical EDA.

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

The separate county-relative PPSF modeling workflow consumes the same cataloged
feature contract and writes model artifacts under
`output/models/county_relative_ppsf/`. Its feature importance is distinct from
the descriptive Spearman ranking and contribution display described above.

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
- **Exploratory feature ranking:** correlations are unadjusted, potentially
  confounded, and selected from many candidates.
- **Illustrative examples:** configured county-event examples and fallback
  selection are descriptive; their differences are not formal significance
  findings.
- **Private housing input:** the repository does not distribute Redfin county
  data, so a clean clone cannot regenerate the housing marts or final page from
  source.
- **Optional insurance input:** insurance premium and non-renewal features are
  unavailable when the private county Feather snapshot is not supplied.
- **Fixed publication period:** housing histories and page events stop in
  December 2025 even when the bootstrap retrieves newer public source data.
- **Static extracts:** results can change when inputs are revised and rebuilt.

## Reproduction

From the repository root:

```powershell
pip install -e .
download-data all
build-database
train-county-relative-ppsf --n-jobs 1
build-climate-risk-housing
```

Set `CENSUS_API_KEY` before running the bootstrap. The committed FIPS master
requires no manual retrieval. A full rebuild additionally requires the user to
supply `data/housing/Redfin-Housing-Market-By-County.csv`; without it, the
housing marts, model, and final page cannot be regenerated. The private
`county_processed_data.feather` input is optional and only adds its insurance
features.

If an existing `data/quoll.duckdb` already contains current raw tables:

```powershell
build-database --marts-only
train-county-relative-ppsf --n-jobs 1
build-climate-risk-housing
```

`--marts-only` cannot initialize a fresh clone because it depends on those
existing raw tables. Retraining is necessary only when regenerating the model
artifacts consumed by the page; the committed small model artifacts can
otherwise be retained.

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
