# Third-party data

The MIT license in this repository applies to the project code and original
documentation. It does not relicense source data obtained from third-party
providers. Provider files are kept in the ignored local `data/` workspace.

## Redfin

Housing-market measures are downloaded from the
[Redfin Data Center Download Hub](https://www.redfin.com/news/data-center/downloads/).
Definitions, coverage, seasonal-adjustment details, and revision practices are
described in Redfin's
[Data Center methodology](https://www.redfin.com/news/data-center/methodology/).

Attribution: Data provided by Redfin, a national real estate brokerage.

The pipeline downloads Redfin's mutable monthly all-county Housing Market
Tracker, Property Types, and Price Drops files. It records retrieval metadata
locally and does not commit those provider CSVs. Users are responsible for
complying with Redfin's applicable terms when using or redistributing the data.

The remaining public providers and their attribution notes are cataloged in
[`config/data_sources.yaml`](config/data_sources.yaml).
