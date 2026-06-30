# Housing Market YoY Index

## Location

The housing market YoY index is computed in `src/housing_climate_risk/page_data/climate_housing_utils.py` inside `prepare_housing_df()`.

## Inputs

The index uses four Redfin county-month year-over-year measures:

- `MEDIAN_PPSF_YOY`
- `AVG_SALE_TO_LIST_YOY`
- `HOMES_SOLD_YOY`
- `INVENTORY_YOY`

These inputs represent price growth, sale-to-list strength, transaction volume growth, and inventory growth.

## Computation

The process is:

1. Sort housing rows by `fips` and `MONTH`.
2. Convert each Redfin YoY component to numeric values.
3. Standardize each component across the full housing dataset:

   ```text
   z = (value - component_mean) / component_standard_deviation
   ```

4. Reverse the standardized `INVENTORY_YOY` component by multiplying it by `-1`.
   This makes lower inventory growth contribute positively to the index.
5. Compute `HOUSING_MARKET_INDEX` as the row-wise simple average of the four standardized components:

   ```text
   HOUSING_MARKET_INDEX =
       mean(
           z(MEDIAN_PPSF_YOY),
           z(AVG_SALE_TO_LIST_YOY),
           z(HOMES_SOLD_YOY),
           -z(INVENTORY_YOY)
       )
   ```

6. Compute `HOUSING_MARKET_INDEX_MOM` as the within-county month-to-month difference in `HOUSING_MARKET_INDEX`:

   ```text
   HOUSING_MARKET_INDEX_MOM =
       HOUSING_MARKET_INDEX_current_month
       - HOUSING_MARKET_INDEX_previous_county_month
   ```

The row-wise average uses available standardized components and skips missing values. A row can therefore receive an index value even when one or more of the four Redfin YoY components is missing.

## Interpretation

Higher `HOUSING_MARKET_INDEX` values indicate stronger YoY housing market conditions under this equal-weight definition:

- higher median price per square foot growth;
- higher average sale-to-list ratio growth;
- higher homes-sold growth;
- lower inventory growth.

The index is a designed composite measure, not a statistically fitted factor. Each component receives equal weight after standardization, except that inventory is directionally reversed so tighter inventory increases the score.
