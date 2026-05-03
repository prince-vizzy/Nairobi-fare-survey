"""
Extract every road waypoint used by each route, assign a unique color per route,
and save road_routes.json:
  {
    "route_colors":  { route_id → "#rrggbb" },
    "road_routes":   { road_name → { "display": "...", "routes": [...] } },
    "route_roads":   { route_id  → ["Road A", "Road B", ...] }
  }

Run:  python build_route_roads.py
Output: road_routes.json  (loaded by app.py for snap resolution + map colors)
"""

import json
import colorsys
import pandas as pd
from collections import defaultdict

import re as _re

# Whole-word match only — avoids "railways" matching "way", "drive in" matching "drive"
_ROAD_RE = _re.compile(
    r'\b(road|avenue|ave|street|bypass|highway|lane|crescent|boulevard)\b',
    _re.IGNORECASE,
)

# Canonical road aliases: normalise typos / merged strings
ROAD_ALIASES: dict[str, str] = {
    "ngongroad":                  "ngong road",   # no-space typo in route 102
    "ngorng road":                "ngong road",   # transposed-letter typo
    "westlands bypass":           "bypass",
    "bypass ruiru, city cabanas": "bypass",       # messy compound waypoint
    "thika road bypass":          "bypass",
}

def is_road(wp: str) -> bool:
    # Normalise aliases first so "NgongRoad" → "ngong road" before the regex check
    return bool(_ROAD_RE.search(normalise_road(wp)))

def normalise_road(wp: str) -> str:
    wl = wp.strip().lower()
    return ROAD_ALIASES.get(wl, wl)

def parse_waypoints(route_long: str) -> list[str]:
    """Split on dashes but handle the 'Bypass Ruiru, City Cabanas' compound."""
    raw = str(route_long).replace(" - ", "-")
    return [w.strip() for w in raw.split("-") if w.strip()]

# ── Load transit data ─────────────────────────────────────────────────────────
df = pd.read_csv("nairobi_transit_data.csv")

# Deduplicate: one row per route_name
seen: set[str] = set()
routes: list[dict] = []
for _, row in df.iterrows():
    rid = str(row["route_name"])
    if rid in seen:
        continue
    seen.add(rid)
    routes.append({
        "route_name": rid,
        "route_long": str(row["route_long"]),
        "headsign":   str(row["headsign"]),
        "fare":       int(row["fare_KSh"]),
    })

routes.sort(key=lambda r: r["route_name"])
n = len(routes)

# ── Assign unique colors via HSV spread ───────────────────────────────────────
route_colors: dict[str, str] = {}
for i, r in enumerate(routes):
    hue = i / n
    rf, gf, bf = colorsys.hsv_to_rgb(hue, 0.78, 0.88)
    route_colors[r["route_name"]] = "#{:02x}{:02x}{:02x}".format(
        int(rf * 255), int(gf * 255), int(bf * 255)
    )

# ── Cross-reference roads ↔ routes ───────────────────────────────────────────
road_routes: dict[str, list[str]] = defaultdict(list)   # road_norm → [route_ids]
route_roads: dict[str, list[str]] = defaultdict(list)   # route_id  → [road_display]

for r in routes:
    rid  = r["route_name"]
    wps  = parse_waypoints(r["route_long"])
    seen_roads: set[str] = set()

    for wp in wps:
        if not is_road(wp):
            continue
        norm    = normalise_road(wp)
        display = norm.title()
        if norm in seen_roads:
            continue
        seen_roads.add(norm)
        if rid not in road_routes[norm]:
            road_routes[norm].append(rid)
        route_roads[rid].append(display)

# ── Build final output ────────────────────────────────────────────────────────
output = {
    "route_colors": route_colors,
    "road_routes": {
        road: {
            "display": road.title(),
            "routes":  sorted(rids),
            "color_sample": route_colors.get(sorted(rids)[0], "#888"),
        }
        for road, rids in sorted(road_routes.items())
    },
    "route_roads": {
        rid: roads
        for rid, roads in sorted(route_roads.items())
    },
}

with open("road_routes.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

# ── Print summary ─────────────────────────────────────────────────────────────
print(f"Routes processed : {n}")
print(f"Unique road names: {len(road_routes)}")
print()
print(f"{'Road':<28}  Routes")
print("-" * 70)
for road, data in output["road_routes"].items():
    routes_str = ", ".join(data["routes"])
    print(f"  {data['display']:<26}  {routes_str}")

print()
print("Per-route road coverage:")
print("-" * 70)
for rid, roads in output["route_roads"].items():
    roads_str = " | ".join(roads) if roads else "(no named roads)"
    color = route_colors[rid]
    print(f"  {rid:<8}  {color}  {roads_str}")

print()
print("Saved -> road_routes.json")
