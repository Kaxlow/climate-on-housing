# Local data workspace

This directory is intentionally excluded from Git except for this file. Populate
it from the original providers and private inputs with:

```powershell
download-data all
```

The bootstrap creates the directory tree, downloads the latest available public
data, validates required filenames and columns, builds derived NOAA county
outputs, and writes `data/download_receipt.yaml` with resolved source URLs,
provider versions, and UTC retrieval timestamps.

Three inputs are not downloaded:

- `housing/Redfin-Housing-Market-By-County.csv` — required private Redfin extract.
- `fipsgeo/fips_master_v2.csv` — required private county reference.
- `20260401_county_processed_data/county_processed_data.feather` — optional
  private insurance-feature snapshot.

The bootstrap exits with status 2 and prints a consolidated manual-retrieval
list when required private inputs or public outputs are missing. Metadata,
expected schemas, attribution, and reproducibility notes are committed in
[`../config/data_sources.yaml`](../config/data_sources.yaml).
