"""
Export full GPS polylines for every route from shapes.shp → route_shapes_latlon.csv
Columns: route_name, direction, sequence, lat, lon
"""

import os
import csv
import shapefile

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
SHP_FILE  = os.path.join(BASE_DIR, "shapes.shp")
OUT_FILE  = os.path.join(BASE_DIR, "route_shapes_latlon.csv")

sf     = shapefile.Reader(SHP_FILE)
fields = [f[0] for f in sf.fields[1:]]

rows_written = 0
with open(OUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["route_name", "direction", "sequence", "lat", "lon"])
    for sr in sf.shapeRecords():
        rec       = dict(zip(fields, sr.record))
        route     = str(rec.get("route_name", "")).strip()
        direction = int(rec.get("direction", 0))
        for seq, (lon, lat) in enumerate(sr.shape.points):
            writer.writerow([route, direction, seq, round(lat, 7), round(lon, 7)])
            rows_written += 1

print(f"Saved {rows_written:,} points to {OUT_FILE}")
