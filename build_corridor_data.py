"""
Build route corridor + landmark data -> nairobi_route_corridors.txt
Loaded by app.py and injected into the system prompt as a new section.

For each route:
  - Road corridor  : from route_long (e.g. "Railways-Langata Road-Karen")
  - Named stops    : from stops.shp, top stops by trip_count
  - Key landmarks  : well-known named stops (malls, hospitals, junctions)
"""
import shapefile, csv
from collections import defaultdict

BASE = "c:/Users/HP/Downloads/GIS_DATA_2019"

# ── Well-known landmark keywords ──────────────────────────────────────────────
LANDMARK_KEYWORDS = {
    "mall", "hospital", "school", "university", "college", "market",
    "stage", "junction", "roundabout", "gate", "park", "stadium",
    "church", "mosque", "temple", "police", "bank", "petrol", "matatu",
    "plaza", "centre", "center", "road", "avenue", "way", "lane",
    "naivas", "carrefour", "quickmart", "equity", "barclays", "kcb",
    "railway", "airport", "terminus", "bypass", "estate"
}

def is_landmark(name):
    nl = name.lower()
    return any(kw in nl for kw in LANDMARK_KEYWORDS)

# ── Read stops shapefile ──────────────────────────────────────────────────────
sf = shapefile.Reader(f"{BASE}/stops.shp")
fields = [f[0] for f in sf.fields[1:]]
stops_shp = [dict(zip(fields, rec)) for rec in sf.records()]

# route_name -> list of (stop_name, trip_count)
route_stops = defaultdict(list)
for s in stops_shp:
    name       = str(s.get("stop_name", "")).strip()
    trip_count = int(s.get("trip_count", 0) or 0)
    route_nams = str(s.get("route_nams", "") or "").strip()
    if not name or not route_nams:
        continue
    for rname in route_nams.split():
        route_stops[rname].append((name, trip_count))

# ── Read shapes shapefile for route_long ──────────────────────────────────────
sf2 = shapefile.Reader(f"{BASE}/shapes.shp")
fields2 = [f[0] for f in sf2.fields[1:]]

# route_name -> route_long (prefer direction=0 / outbound)
route_long_map = {}
for rec in sf2.records():
    d = dict(zip(fields2, rec))
    rname   = str(d["route_name"]).strip()
    rlong   = str(d["route_long"]).strip()
    dirn    = int(d["direction"])
    if rname not in route_long_map or dirn == 0:
        route_long_map[rname] = rlong

# ── Read transit CSV for fare + headsign ─────────────────────────────────────
transit = {}
with open(f"{BASE}/nairobi_transit_data.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        transit[row["route_name"]] = row

# ── Build corridor text ───────────────────────────────────────────────────────
lines = []
for rname, tmeta in sorted(transit.items()):
    route_long = route_long_map.get(rname, tmeta.get("route_long", ""))
    fare       = int(float(tmeta["fare_KSh"]))
    headsign   = tmeta.get("headsign", "")

    # Corridor waypoints (clean dashes to arrows for readability)
    corridor = route_long.replace("-", " → ")

    # Top stops: sort by trip_count desc, deduplicate, cap at 8
    stops = route_stops.get(rname, [])
    seen  = set()
    top   = []
    # Priority 1: high trip_count stops
    for sname, tc in sorted(stops, key=lambda x: -x[1]):
        if sname.lower() not in seen:
            seen.add(sname.lower())
            top.append(sname)
        if len(top) >= 8:
            break
    # Priority 2: fill with landmark-named stops if under 5
    if len(top) < 5:
        for sname, tc in stops:
            if sname.lower() not in seen and is_landmark(sname):
                seen.add(sname.lower())
                top.append(sname)
            if len(top) >= 8:
                break

    stops_str = ", ".join(top) if top else "—"
    lines.append(
        f"{rname} | {corridor} | {fare} KSh | stops: {stops_str}"
    )

output = "\n".join(lines)

with open(f"{BASE}/nairobi_route_corridors.txt", "w", encoding="utf-8") as f:
    f.write(output)

print(f"Written {len(lines)} routes to nairobi_route_corridors.txt")
print(f"Total size: {len(output):,} chars")
print()
print("Sample (first 5 routes):")
for l in lines[:5]:
    print(" ", l)
