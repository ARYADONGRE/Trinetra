#!/usr/bin/env python3
"""Convert a TSMC2014 Foursquare tab-separated file to TRINETRA CSV schema.

Expected TSMC2014 column order:
userId, venueId, venueCategoryId, venueCategory, latitude, longitude,
timezoneOffset, utcTimestamp

Example:
  python scripts/prepare_foursquare.py \
      --input /path/to/dataset_TSMC2014_NYC.txt \
      --output data/foursquare_nyc.csv
"""
import argparse
from pathlib import Path
import pandas as pd

COLUMNS = [
    "userId", "venueId", "venueCategoryId", "venueCategory",
    "latitude", "longitude", "timezoneOffset", "utcTimestamp"
]
OUTPUT_COLUMNS = [
    "userId", "venueId", "venueCategory", "latitude", "longitude", "utcTimestamp"
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Original TSMC2014 tab-separated file")
    parser.add_argument("--output", required=True, help="Destination CSV path")
    args = parser.parse_args()

    source = Path(args.input)
    destination = Path(args.output)
    if not source.is_file():
        raise FileNotFoundError(f"Input file does not exist: {source}")

    # TSMC2014 files are tab-separated and are commonly distributed without a header.
    frame = pd.read_csv(source, sep="\t", header=None, names=COLUMNS, dtype={
        "userId": "string",
        "venueId": "string",
        "venueCategoryId": "string",
        "venueCategory": "string",
        "utcTimestamp": "string",
    })

    frame = frame[OUTPUT_COLUMNS].copy()
    frame["latitude"] = pd.to_numeric(frame["latitude"], errors="coerce")
    frame["longitude"] = pd.to_numeric(frame["longitude"], errors="coerce")
    frame["utcTimestamp"] = pd.to_datetime(frame["utcTimestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=[
        "userId", "venueId", "venueCategory", "latitude", "longitude", "utcTimestamp"
    ])
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    print(f"Saved {len(frame):,} check-ins to {destination}")
    print(f"Users: {frame['userId'].nunique():,}")
    print(f"Venues: {frame['venueId'].nunique():,}")
    print(f"Venue categories: {frame['venueCategory'].nunique():,}")

if __name__ == "__main__":
    main()
