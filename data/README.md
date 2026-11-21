# Dataset directory

Download the TSMC2014 Foursquare NYC/TKY data from Dingqi Yang's original page:
https://sites.google.com/site/yangdingqi/home/foursquare-dataset

Convert the tab-separated raw files using `scripts/prepare_foursquare.py`.
Expected generated files:

- `foursquare_nyc.csv`
- `foursquare_tky.csv`

Expected CSV columns:
`userId, venueId, venueCategory, latitude, longitude, utcTimestamp`

Raw and processed datasets are not included in this Git repository. See the
root README for provenance, conversion commands, and citation information.
