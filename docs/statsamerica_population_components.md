# StatsAmerica Components of Population Change Dataset

## Overview

This dataset contains **TRUE net migration data** (in-migration minus out-migration) for U.S. counties, states, metros, and EDDs from 1990-2025.

**Source**: https://www.statsamerica.org/downloads/default.aspx  
**Provider**: Indiana Business Research Center (IBRC) at Indiana University  
**Original Data**: U.S. Census Bureau Population Estimates  

## Why This Data is Better Than ACS Migration Data

| Feature | ACS B07001 (Current) | StatsAmerica Components | Winner |
|---------|---------------------|------------------------|--------|
| **Net Migration** | ❌ In-migration only | ✅ TRUE net (in - out) | StatsAmerica |
| **Separate Flows** | ✅ Can see in-migration | ✅ Domestic + international separate | Tie |
| **Time Coverage** | 2015-2024 (we have) | 1990-2025 | StatsAmerica |
| **Data Freshness** | Annual (ACS 5-year) | Annual | Tie |
| **County Coverage** | All counties | All counties | Tie |
| **Calculation Method** | Survey-based (current residents) | Model-based (births/deaths/population change) | Different approaches |

### Key Advantage

**ACS B07001** asks current residents: "Where did you live 1 year ago?"  
→ Only captures people who moved IN (and are still there)  
→ Misses people who moved OUT  

**StatsAmerica** uses residual method:  
`Net Migration = Population Change - (Births - Deaths) - International Migration`  
→ Captures TRUE net effect (in - out)

## Data Files Downloaded

The download contains 4 CSV files:

1. **`Components of Population Change - U.S., States, and Counties.csv`** ← Main file for counties
2. `Components of Population Change - Metros, Micros.csv` (metro/micro areas)
3. `Components of Population Change - EDDs.csv` (Economic Development Districts)
4. `Components of Population Change - Metadata.csv` (dataset documentation)

## County-Level Data Structure

**File**: `Components of Population Change - U.S., States, and Counties.csv`  
**Rows**: ~122,000 (counties × years)  
**Columns**: 10

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `IBRC_Geo_ID` | String | IBRC geographic identifier | "1001" |
| `Statefips` | String | 2-digit state FIPS (zero-padded) | "01" |
| `Countyfips` | String | 3-digit county FIPS (zero-padded) | "001" |
| `Description` | String | County name and state | "Autauga County, AL" |
| `Year` | Integer | Calendar year | 1990-2025 |
| `Births` | Integer | Number of births | 140 |
| `Deaths` | Integer | Number of deaths | 77 |
| `Net International Migration` | Integer | International in-migration minus out-migration | 1 |
| `Net Domestic Migration` | Integer | Domestic in-migration minus out-migration | 73 |
| `Residual` | Integer | Statistical residual/adjustment | 1 |

### Example Data

```csv
IBRC_Geo_ID,Statefips,Countyfips,Description,Year,Births,Deaths,Net International Migration,Net Domestic Migration,Residual
1001,01,001,"Autauga County, AL",1990,140,77,1,73,1
1001,01,001,"Autauga County, AL",2023,658,518,3,-2105,0
```

**Autauga County 2023 interpretation**:
- Births: 658
- Deaths: 518
- Natural increase: +140
- International migration: +3 (net)
- Domestic migration: -2,105 (net) ← This is the key insight!
- Total net migration: +3 + (-2,105) = **-2,102**

## Key Metrics Available

### 1. Natural Increase
```
Natural Increase = Births - Deaths
```

### 2. Net International Migration
Already provided (in minus out for international)

### 3. Net Domestic Migration
Already provided (in minus out for domestic US)

### 4. Total Net Migration
```
Total Net Migration = Net International Migration + Net Domestic Migration
```

### 5. Total Population Change
```
Total Population Change = Natural Increase + Net International Migration + Net Domestic Migration + Residual
```

### 6. Net Migration Rate (can calculate)
```python
net_migration_rate = (net_international + net_domestic) / population_start
```

## Download Script

```bash
# Download the dataset
download-statsamerica-population-components

# Download and show detailed info
download-statsamerica-population-components --info

# Force re-download
download-statsamerica-population-components --force

# Custom output directory
download-statsamerica-population-components --output-dir /path/to/output
```

## Integration with Database

### Current Status
✅ Download script created  
⬜ Database table created (`mart.statsamerica_population_components_annual`)  
⬜ Integrated into `build-database` pipeline  
⬜ Climate risk model updated to use true net migration  

### Proposed Database Schema

```sql
CREATE TABLE mart.statsamerica_population_components_annual AS
SELECT
    lpad(Statefips, 2, '0') || lpad(Countyfips, 3, '0') AS fips,
    lpad(Statefips, 2, '0') AS state_fips,
    CAST(Year AS INTEGER) AS year,
    Description AS county_name,
    CAST(Births AS INTEGER) AS births,
    CAST(Deaths AS INTEGER) AS deaths,
    CAST("Net International Migration" AS INTEGER) AS net_international_migration,
    CAST("Net Domestic Migration" AS INTEGER) AS net_domestic_migration,
    CAST(Residual AS INTEGER) AS residual,
    -- Computed columns
    CAST(Births AS INTEGER) - CAST(Deaths AS INTEGER) AS natural_increase,
    CAST("Net International Migration" AS INTEGER) + CAST("Net Domestic Migration" AS INTEGER) AS total_net_migration
FROM raw.statsamerica_population_components
WHERE Statefips != '0'  -- Exclude U.S. total and state-only rows
  AND Countyfips != '000'
  AND Year IS NOT NULL
```

## Comparison with ACS Migration Data

To validate and understand differences:

```sql
WITH acs_migration AS (
    SELECT
        fips,
        year,
        domestic_in_migration_rate + moved_from_abroad_rate AS acs_total_in_migration_rate
    FROM mart.acs_county_demographic_annual
    WHERE year BETWEEN 2015 AND 2023
),
statsamerica AS (
    SELECT
        fips,
        year,
        total_net_migration
    FROM mart.statsamerica_population_components_annual
    WHERE year BETWEEN 2015 AND 2023
),
population AS (
    SELECT fips, year, total_population
    FROM mart.acs_county_demographic_annual
    WHERE year BETWEEN 2015 AND 2023
)
SELECT
    p.fips,
    p.year,
    a.acs_total_in_migration_rate,
    s.total_net_migration,
    (s.total_net_migration::DOUBLE / p.total_population) AS statsamerica_net_migration_rate,
    a.acs_total_in_migration_rate - (s.total_net_migration::DOUBLE / p.total_population) AS difference
FROM population p
LEFT JOIN acs_migration a ON p.fips = a.fips AND p.year = a.year
LEFT JOIN statsamerica s ON p.fips = s.fips AND p.year = s.year
WHERE p.fips = '06037'  -- Los Angeles County example
ORDER BY year DESC;
```

**Expected result**:
- ACS in-migration rate will be **higher** (only counts people moving in)
- StatsAmerica net migration rate can be **negative** (more out than in)
- Difference approximates out-migration rate

## Use Cases

### 1. True Climate Risk Migration Impact
```sql
-- Counties with net OUT-migration after major disasters
SELECT
    s.fips,
    s.year,
    s.county_name,
    s.total_net_migration,
    s.net_domestic_migration,
    d.incident_type,
    d.disaster_number
FROM mart.statsamerica_population_components_annual s
INNER JOIN mart.fema_disaster_declarations d
    ON s.fips = d.fips
    AND s.year = YEAR(d.incident_begin_date)
WHERE s.total_net_migration < -1000  -- Net loss of 1000+ people
ORDER BY s.total_net_migration ASC
LIMIT 100;
```

### 2. Improved Climate Risk Model Feature
Replace `in_migration_rate` with `net_migration_rate`:

```python
# Old feature (in-migration only)
in_migration_rate = (domestic_in + international_in) / population

# New feature (true net migration)
net_migration_rate = (domestic_net + international_net) / population
# Can be negative!
```

### 3. Migration Response to Climate Events
```sql
-- Average net migration change after wildfires
SELECT
    AVG(post.total_net_migration - pre.total_net_migration) AS avg_migration_change
FROM mart.noaa_storm_events e
INNER JOIN mart.statsamerica_population_components_annual pre
    ON e.fips = pre.fips
    AND pre.year = e.event_year - 1
INNER JOIN mart.statsamerica_population_components_annual post
    ON e.fips = post.fips
    AND post.year = e.event_year + 1
WHERE e.event_type = 'Wildfire'
  AND e.property_damage_amount > 1000000;
```

## Data Quality Notes

### Residual Column
The `Residual` column represents:
- Statistical adjustments
- Census corrections
- Rounding errors
- Unaccounted population changes

Usually small (< 50), but can be larger in Census years or for small counties.

### Missing Data
- Some counties may have missing data for certain years
- Early years (1990-2000) may have more gaps
- Use `WHERE Residual IS NOT NULL` to filter complete records

### Encoding
- Main county file: UTF-8
- Metro/Micro file: Latin-1 (has special characters in city names)
- Handle both in data processing

## Next Steps

1. **Add to `build-database` pipeline**
   - Load into `raw.statsamerica_population_components`
   - Create `mart.statsamerica_population_components_annual`

2. **Update climate risk model**
   - Replace `in_migration_rate` with `net_migration_rate`
   - Use StatsAmerica data as primary source
   - Keep ACS data for validation/comparison

3. **Validate data**
   - Compare with ACS migration for overlapping years
   - Check against IRS county-to-county migration data
   - Verify negative net migration makes sense (e.g., Detroit, New Orleans post-Katrina)

4. **Documentation**
   - Update CLAUDE.md to reference StatsAmerica data
   - Update climate risk prediction docs
   - Add data source citation to visualizations

## Citation

```
Indiana Business Research Center (IBRC). Components of Population Change. 
Indiana University Kelley School of Business. 
Source: U.S. Census Bureau Population Estimates.
Retrieved from https://www.statsamerica.org/downloads/
```
