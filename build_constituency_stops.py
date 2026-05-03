"""
Download Nairobi constituency boundaries from GADM (level 2),
spatial-join with nairobi_stops.csv, and save:
  - nairobi_constituencies.geojson   (boundary polygons)
  - constituency_stops.json          (constituency -> list of stops)
"""
import json
import os
import re
import requests
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── 1. Download Nairobi constituency boundaries from GADM ─────────────────────
GADM_URL = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_KEN_2.json"

print("Downloading Nairobi constituency boundaries from GADM...")
resp = requests.get(GADM_URL, timeout=90)
resp.raise_for_status()
data = resp.json()

# Filter Nairobi only and normalise names (GADM uses CamelCase e.g. "DagorettiNorth")
def split_camel(name):
    return re.sub(r"([a-z])([A-Z])", r"\1 \2", name)

features = []
for feat in data["features"]:
    props = feat.get("properties", {})
    if str(props.get("NAME_1", "")).lower() != "nairobi":
        continue
    raw_name = props.get("NAME_2", "")
    name     = split_camel(raw_name).title()
    features.append({
        "type":       "Feature",
        "properties": {"constituency": name},
        "geometry":   feat["geometry"],
    })

print(f"  Found {len(features)} Nairobi constituency polygons")

# Save boundary GeoJSON
geojson_out = {"type": "FeatureCollection", "features": features}
with open(os.path.join(BASE_DIR, "nairobi_constituencies.geojson"), "w", encoding="utf-8") as f:
    json.dump(geojson_out, f, ensure_ascii=False, indent=2)
print("  Saved nairobi_constituencies.geojson")

# ── 3. Spatial join stops → constituencies ────────────────────────────────────
stops_df = pd.read_csv(os.path.join(BASE_DIR, "nairobi_stops.csv"))
stops_df = stops_df.dropna(subset=["stop_lat", "stop_lon"])

stops_gdf = gpd.GeoDataFrame(
    stops_df,
    geometry=[Point(row.stop_lon, row.stop_lat) for _, row in stops_df.iterrows()],
    crs="EPSG:4326",
)

const_gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")

joined = gpd.sjoin(stops_gdf, const_gdf[["constituency", "geometry"]], how="left", predicate="within")

# ── 4. Build constituency → stops mapping ────────────────────────────────────
result = {}
for const, grp in joined.groupby("constituency"):
    result[const] = [
        {
            "stop_id":   row["stop_id"],
            "stop_name": row["stop_name"],
            "lat":       round(float(row["stop_lat"]), 6),
            "lon":       round(float(row["stop_lon"]), 6),
            "routes":    [r for r in str(row.get("routes", "") or "").split() if r],
        }
        for _, row in grp.iterrows()
    ]

# Stops that fell outside all constituency polygons
unmatched = joined[joined["constituency"].isna()]
if not unmatched.empty:
    result["_unmatched"] = [
        {"stop_id": row["stop_id"], "stop_name": row["stop_name"],
         "lat": round(float(row["stop_lat"]), 6), "lon": round(float(row["stop_lon"]), 6)}
        for _, row in unmatched.iterrows()
    ]

out_path = os.path.join(BASE_DIR, "constituency_stops.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(f"\nConstituencies found: {len([k for k in result if not k.startswith('_')])}")
for const in sorted(k for k in result if not k.startswith("_")):
    print(f"  {const}: {len(result[const])} stops")
if "_unmatched" in result:
    print(f"  (unmatched outside boundaries: {len(result['_unmatched'])})")
print(f"\nSaved to constituency_stops.json")
