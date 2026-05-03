"""
Nairobi Transit Snap-to-Network Router
=======================================
Senior Geospatial Engineer implementation.

Given a user GPS position and destination, this script:
  1. Loads bus routes (LineStrings) and stops (Points)
  2. Generates informal roadside pick-up points by sampling route polylines
  3. Snaps the user to the network via two methods:
       a. Nearest Point  — KDTree on all stops (formal + roadside)
       b. Nearest Road   — shapely.ops.nearest_points on route LineStrings
  4. Filters routes that pass near the destination
  5. Scores each candidate: walk_to_board + bus_travel + walk_to_dest
  6. Outputs the winner and renders an interactive Folium map

Usage
-----
  python transit_snap.py
  # Opens  transit_result.html  in the working directory
"""

from __future__ import annotations

import sys, os, math, time

# Force UTF-8 on Windows consoles so Unicode print() calls don't crash
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Line-buffered log file so progress is visible even when stdout is swallowed
_LOG = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "transit_snap.log"), "w", buffering=1, encoding="utf-8")
_T0  = time.time()
_LOG.write("log open\n"); _LOG.flush()

def log(msg: str) -> None:
    t = time.time() - _T0
    line = f"[{t:6.1f}s] {msg}"
    _LOG.write(line + "\n"); _LOG.flush()
    print(line)

_LOG.write("importing json\n"); _LOG.flush()
import json
_LOG.write("importing warnings\n"); _LOG.flush()
import warnings
_LOG.write("importing webbrowser\n"); _LOG.flush()
import webbrowser
_LOG.write("importing dataclasses\n"); _LOG.flush()
from dataclasses import dataclass, field
_LOG.write("importing typing\n"); _LOG.flush()
from typing import Optional
_LOG.write("importing numpy\n"); _LOG.flush()
import numpy as np
_LOG.write("importing pandas\n"); _LOG.flush()
import pandas as pd
_LOG.write("importing geopandas\n"); _LOG.flush()
import geopandas as gpd
_LOG.write("importing shapefile\n"); _LOG.flush()
import shapefile as shp_reader
_LOG.write("importing shapely\n"); _LOG.flush()
from shapely.geometry import Point, LineString, MultiLineString
from shapely.ops import nearest_points
_LOG.write("importing folium\n"); _LOG.flush()
import folium
_LOG.write("all imports done\n"); _LOG.flush()

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
SHAPES_SHP      = os.path.join(BASE_DIR, "shapes.shp")
STOPS_CSV       = os.path.join(BASE_DIR, "nairobi_stops.csv")
TRANSIT_CSV     = os.path.join(BASE_DIR, "nairobi_transit_data.csv")
GEOJSON_OUT     = os.path.join(BASE_DIR, "bus_routes.geojson")
MAP_OUT         = os.path.join(BASE_DIR, "transit_result.html")

# ── Journey constants ─────────────────────────────────────────────────────────
AVG_BUS_KMH     = 25.0      # average Nairobi matatu speed
WALK_KMH        = 4.5       # pedestrian speed
DEST_SNAP_M     = 1_200     # max metres a route can be from destination and still qualify
ALIGHT_SNAP_M   = 900       # max metres to search for an alighting stop near destination
ROADSIDE_STEP_M = 350       # generate one informal stop every N metres along each route

# ── Projection ────────────────────────────────────────────────────────────────
# WGS 84 / UTM zone 37S — metres, appropriate for Nairobi
CRS_WGS84 = "EPSG:4326"
CRS_UTM   = "EPSG:32737"

# ── Nairobi informal roadside pick-up points (Ngong Road / Karen corridor) ────
# These represent well-known flagging spots that don't appear in formal stop lists.
ROADSIDE_EXTRAS = [
    # Ngong Road corridor
    {"name": "Bomas Turnoff",           "lat": -1.3108, "lon": 36.7505},
    {"name": "Galleria Mall",           "lat": -1.3168, "lon": 36.7275},
    {"name": "Karen Crossroads",        "lat": -1.3205, "lon": 36.7085},
    {"name": "Karen Hardy",             "lat": -1.3301, "lon": 36.6992},
    {"name": "Karen C Stage",           "lat": -1.3189, "lon": 36.6910},
    {"name": "Winterscho",              "lat": -1.3220, "lon": 36.7030},
    {"name": "Langata South Rd Jct",    "lat": -1.3350, "lon": 36.7320},
    {"name": "Langata Rd / Mbagathi",   "lat": -1.3080, "lon": 36.7610},
    {"name": "Wilson Airport Gate",     "lat": -1.3143, "lon": 36.8140},
    {"name": "Haile Selassie Ave",      "lat": -1.2960, "lon": 36.8200},
    {"name": "Kenyatta Ave / City Hall","lat": -1.2887, "lon": 36.8198},
    {"name": "Uhuru Park Gate",         "lat": -1.2945, "lon": 36.8097},
    {"name": "Upper Hill Hospl",        "lat": -1.2979, "lon": 36.8047},
    {"name": "Ngong Rd / Ring Rd Jct",  "lat": -1.2957, "lon": 36.7851},
    {"name": "Prestige Plaza",          "lat": -1.3002, "lon": 36.7730},
    {"name": "Adams Arcade",            "lat": -1.3046, "lon": 36.7650},
    {"name": "Junction Mall",           "lat": -1.3110, "lon": 36.7560},
    {"name": "Dagoretti Corner",        "lat": -1.3010, "lon": 36.7400},
    {"name": "Kawangware Stage",        "lat": -1.2780, "lon": 36.7305},
    {"name": "Rongai Stage (CBD)",      "lat": -1.2880, "lon": 36.8260},
]


# ══════════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════════

def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres."""
    R = 6_371_000
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def walk_time_min(dist_m: float) -> float:
    return dist_m / (WALK_KMH * 1000 / 60)


def bus_time_min(dist_m: float) -> float:
    return dist_m / (AVG_BUS_KMH * 1000 / 60)


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — Build / load bus_routes.geojson
# ══════════════════════════════════════════════════════════════════════════════

def build_geojson() -> str:
    """Convert shapes.shp to bus_routes.geojson (direction-0 preferred per route)."""
    log("Building bus_routes.geojson from shapes.shp ...")
    sf     = shp_reader.Reader(SHAPES_SHP)
    fields = [f[0] for f in sf.fields[1:]]
    best: dict[str, dict] = {}

    for sr in sf.shapeRecords():
        rec       = dict(zip(fields, sr.record))
        route     = str(rec.get("route_name", "")).strip()
        direction = int(rec.get("direction", 0))
        coords    = [(p[0], p[1]) for p in sr.shape.points]   # (lon, lat)
        if len(coords) < 2:
            continue
        if route not in best or direction == 0:
            best[route] = {"coords": coords, "direction": direction,
                           "route_name": route}

    features = []
    for rname, data in best.items():
        features.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": data["coords"]},
            "properties": {"route_name": rname, "direction": data["direction"]},
        })

    geojson = {"type": "FeatureCollection", "features": features}
    with open(GEOJSON_OUT, "w", encoding="utf-8") as f:
        json.dump(geojson, f)
    log(f"  Wrote {len(features)} routes to {GEOJSON_OUT}")
    return GEOJSON_OUT


# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — Load datasets
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Stop:
    stop_id:  str
    name:     str
    lat:      float
    lon:      float
    routes:   list[str]
    kind:     str = "formal"   # "formal" | "roadside" | "sampled"


def load_routes_gdf() -> gpd.GeoDataFrame:
    if not os.path.exists(GEOJSON_OUT):
        build_geojson()
    gdf = gpd.read_file(GEOJSON_OUT)          # WGS84
    gdf["route_name"] = gdf["route_name"].astype(str).str.strip()
    return gdf


def load_stops(routes_gdf: gpd.GeoDataFrame) -> list[Stop]:
    """Load formal stops, add roadside extras, sample route polylines."""
    stops: list[Stop] = []

    # ── Formal stops ──────────────────────────────────────────────────────────
    df = pd.read_csv(STOPS_CSV, dtype=str)
    for _, row in df.iterrows():
        try:
            lat = float(row["stop_lat"])
            lon = float(row["stop_lon"])
        except (ValueError, KeyError):
            continue
        if math.isnan(lat) or math.isnan(lon):
            continue
        raw_routes = str(row.get("routes", "") or "")
        rlist = [r.strip() for r in raw_routes.split() if r.strip()]
        stops.append(Stop(
            stop_id = str(row.get("stop_id", "")),
            name    = str(row.get("stop_name", "Unknown")),
            lat     = lat,
            lon     = lon,
            routes  = rlist,
            kind    = "formal",
        ))

    # ── Known roadside extras ─────────────────────────────────────────────────
    for i, rs in enumerate(ROADSIDE_EXTRAS):
        stops.append(Stop(
            stop_id = f"RS{i:03d}",
            name    = rs["name"],
            lat     = rs["lat"],
            lon     = rs["lon"],
            routes  = [],
            kind    = "roadside",
        ))

    # ── Sampled points along each route polyline (batch projection) ───────────
    # Collect all interpolated points in UTM first, then project the whole
    # batch to WGS84 in a single GeoSeries.to_crs() call — orders of magnitude
    # faster than projecting one point at a time.
    routes_utm   = routes_gdf.to_crs(CRS_UTM)
    utm_pts:  list[Point]  = []
    pt_meta:  list[tuple]  = []   # (route_name, k)

    for _, row in routes_utm.iterrows():
        line  = row.geometry
        rname = str(row["route_name"])
        length = line.length
        if length == 0:
            continue
        n_pts = max(1, int(length // ROADSIDE_STEP_M))
        for k in range(1, n_pts):
            utm_pts.append(line.interpolate(k / n_pts * length))
            pt_meta.append((rname, k))

    if utm_pts:
        wgs_series = gpd.GeoSeries(utm_pts, crs=CRS_UTM).to_crs(CRS_WGS84)
        for (rname, k), pt in zip(pt_meta, wgs_series):
            stops.append(Stop(
                stop_id = f"SP_{rname}_{k}",
                name    = f"Roadside ({rname})",
                lat     = pt.y,
                lon     = pt.x,
                routes  = [rname],
                kind    = "sampled",
            ))

    log(f"  Stop pool: {sum(s.kind=='formal' for s in stops)} formal "
        f"+ {sum(s.kind=='roadside' for s in stops)} roadside "
        f"+ {sum(s.kind=='sampled' for s in stops)} sampled "
        f"= {len(stops)} total")
    return stops


# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — KDTree snap-to-stop  +  shapely snap-to-road
# ══════════════════════════════════════════════════════════════════════════════

def snap_to_stop(user_lat: float, user_lon: float,
                 stops: list[Stop],
                 k: int = 10) -> list[tuple[float, Stop]]:
    """
    Return k nearest stops sorted by haversine distance.
    Uses vectorised numpy for a fast pre-filter in degree space,
    then refines with true haversine on the top candidates.
    """
    lats = np.array([s.lat for s in stops])
    lons = np.array([s.lon for s in stops])
    # Degree-space squared distance — cheap pre-filter
    sq   = (lats - user_lat) ** 2 + (lons - user_lon) ** 2
    top_k = min(k * 4, len(stops))
    idxs  = np.argpartition(sq, top_k - 1)[:top_k]
    results = []
    for idx in idxs:
        s = stops[idx]
        d = haversine_m(user_lat, user_lon, s.lat, s.lon)
        results.append((d, s))
    results.sort(key=lambda x: x[0])
    return results[:k]


def snap_to_road(user_lat: float, user_lon: float,
                 routes_gdf: gpd.GeoDataFrame,
                 only_routes: list[str] | None = None) -> dict[str, tuple[float, float, float]]:
    """
    Find the nearest point on each route LineString to the user.
    Pass only_routes to restrict work to a small candidate set.

    Returns {route_name: (dist_m, snap_lat, snap_lon)}
    """
    user_pt_wgs = Point(user_lon, user_lat)

    # Subset to candidate routes only
    if only_routes:
        subset = routes_gdf[routes_gdf["route_name"].isin(set(only_routes))]
    else:
        subset = routes_gdf

    subset_utm  = subset.to_crs(CRS_UTM)
    user_gs     = gpd.GeoSeries([user_pt_wgs], crs=CRS_WGS84).to_crs(CRS_UTM)
    user_pt_utm = user_gs.iloc[0]

    result:       dict[str, tuple] = {}
    snap_pts_utm: list[Point]      = []
    snap_rnames:  list[str]        = []

    for _, row in subset_utm.iterrows():
        rname = str(row["route_name"])
        line  = row.geometry
        if line is None or line.is_empty:
            result[rname] = (float("inf"), user_lat, user_lon)
            continue
        snap_pt_utm, _ = nearest_points(line, user_pt_utm)
        dist_m          = float(user_pt_utm.distance(snap_pt_utm))
        snap_pts_utm.append(snap_pt_utm)
        snap_rnames.append(rname)
        result[rname] = (dist_m, None, None)

    # Batch-project snap points back to WGS84 in one call
    if snap_pts_utm:
        wgs_snaps = gpd.GeoSeries(snap_pts_utm, crs=CRS_UTM).to_crs(CRS_WGS84)
        for rname, pt in zip(snap_rnames, wgs_snaps):
            d = result[rname][0]
            result[rname] = (d, pt.y, pt.x)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Step 4 — Filter routes that pass near the destination
# ══════════════════════════════════════════════════════════════════════════════

def filter_routes_near_dest(dest_lat: float, dest_lon: float,
                             routes_gdf: gpd.GeoDataFrame,
                             max_m: float = DEST_SNAP_M) -> list[str]:
    """Return route_names whose polyline passes within max_m of the destination."""
    dest_pt_wgs = Point(dest_lon, dest_lat)
    routes_utm  = routes_gdf.to_crs(CRS_UTM)
    dest_gs     = gpd.GeoSeries([dest_pt_wgs], crs=CRS_WGS84).to_crs(CRS_UTM)
    dest_pt_utm = dest_gs.iloc[0]

    candidates = []
    for _, row in routes_utm.iterrows():
        rname = str(row["route_name"])
        line  = row.geometry
        if line is None or line.is_empty:
            continue
        d = line.distance(dest_pt_utm)
        if d <= max_m:
            candidates.append((d, rname))

    candidates.sort()
    return [rname for _, rname in candidates]


# ══════════════════════════════════════════════════════════════════════════════
# Step 5 — Find alighting stop per route near the destination
# ══════════════════════════════════════════════════════════════════════════════

def find_alight_stop(dest_lat: float, dest_lon: float,
                     route_name: str,
                     stops: list[Stop],
                     max_m: float = ALIGHT_SNAP_M) -> Optional[Stop]:
    """
    Find the nearest stop that serves this route and is within max_m of the destination.
    Falls back to any stop within max_m if no route-specific stop found.
    """
    candidates = []
    for s in stops:
        d = haversine_m(dest_lat, dest_lon, s.lat, s.lon)
        if d > max_m:
            continue
        if route_name in s.routes or s.kind == "sampled" and route_name in s.routes:
            candidates.append((d, s))

    if not candidates:
        # Fallback: any stop within max_m
        for s in stops:
            d = haversine_m(dest_lat, dest_lon, s.lat, s.lon)
            if d <= max_m:
                candidates.append((d, s))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


# ══════════════════════════════════════════════════════════════════════════════
# Step 6 — Journey scoring
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class JourneyOption:
    route_name:     str
    fare_ksh:       float
    board_stop:     Stop
    board_dist_m:   float
    board_method:   str           # "stop" | "road_snap"
    alight_stop:    Stop
    alight_dist_m:  float         # to destination
    bus_dist_m:     float
    total_time_min: float
    snap_lat:       float = 0.0   # road snap coords (if method == road_snap)
    snap_lon:       float = 0.0


def score_route(route_name: str,
                user_lat: float, user_lon: float,
                dest_lat: float, dest_lon: float,
                stops: list[Stop],
                road_snaps: dict,
                transit_meta: pd.DataFrame) -> Optional[JourneyOption]:
    """
    Score a single candidate route. Returns None if un-routable.
    """
    # ── Find alighting stop ───────────────────────────────────────────────────
    alight = find_alight_stop(dest_lat, dest_lon, route_name, stops)
    if alight is None:
        return None
    alight_dist_m = haversine_m(dest_lat, dest_lon, alight.lat, alight.lon)

    # ── Boarding: compare nearest stop vs road snap ───────────────────────────
    route_stops = [s for s in stops if route_name in s.routes]
    if not route_stops:
        # Use road snap only
        if route_name not in road_snaps:
            return None
        snap_d, snap_lat, snap_lon = road_snaps[route_name]
        board_method = "road_snap"
        board_dist_m = snap_d
        board_stop   = Stop(stop_id="SNAP", name=f"Roadside flag ({route_name})",
                            lat=snap_lat, lon=snap_lon, routes=[route_name], kind="sampled")
        snap_lat_out, snap_lon_out = snap_lat, snap_lon
    else:
        # Nearest formal/sampled stop on this route
        best_d   = float("inf")
        best_s   = route_stops[0]
        for s in route_stops:
            d = haversine_m(user_lat, user_lon, s.lat, s.lon)
            if d < best_d:
                best_d, best_s = d, s
        stop_dist = best_d

        # Compare with road snap
        snap_d, snap_lat, snap_lon = road_snaps.get(route_name, (float("inf"), 0, 0))
        if snap_d < stop_dist:
            board_method = "road_snap"
            board_dist_m = snap_d
            board_stop   = Stop(stop_id="SNAP", name=f"Roadside flag ({route_name})",
                                lat=snap_lat, lon=snap_lon, routes=[route_name], kind="sampled")
            snap_lat_out, snap_lon_out = snap_lat, snap_lon
        else:
            board_method = "stop"
            board_dist_m = stop_dist
            board_stop   = best_s
            snap_lat_out = best_s.lat
            snap_lon_out = best_s.lon

    # ── Bus distance (haversine boarding → alighting, scaled by route factor) ─
    bus_dist_m = haversine_m(board_stop.lat, board_stop.lon, alight.lat, alight.lon)

    # ── Journey time ──────────────────────────────────────────────────────────
    total_min = (walk_time_min(board_dist_m)
                 + bus_time_min(bus_dist_m)
                 + walk_time_min(alight_dist_m))

    # ── Fare ──────────────────────────────────────────────────────────────────
    row = transit_meta[transit_meta["route_name"].astype(str) == route_name]
    fare = float(row["fare_KSh"].iloc[0]) if not row.empty else 50.0

    return JourneyOption(
        route_name    = route_name,
        fare_ksh      = fare,
        board_stop    = board_stop,
        board_dist_m  = board_dist_m,
        board_method  = board_method,
        alight_stop   = alight,
        alight_dist_m = alight_dist_m,
        bus_dist_m    = bus_dist_m,
        total_time_min= total_min,
        snap_lat      = snap_lat_out,
        snap_lon      = snap_lon_out,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Step 7 — Folium visualisation
# ══════════════════════════════════════════════════════════════════════════════

# Pulsing red CSS marker for user position
PULSE_CSS = """
<style>
.pulse-wrap { position: relative; width: 24px; height: 24px; }
.pulse-dot  { width: 12px; height: 12px; border-radius: 50%;
              background: #e74c3c; position: absolute; top: 6px; left: 6px; }
.pulse-ring { width: 24px; height: 24px; border-radius: 50%;
              border: 3px solid #e74c3c; position: absolute; top: 0; left: 0;
              animation: pulse 1.6s ease-out infinite; opacity: 0; }
@keyframes pulse { 0%{transform:scale(0.4);opacity:1} 100%{transform:scale(2.2);opacity:0} }
</style>
<div class="pulse-wrap"><div class="pulse-ring"></div><div class="pulse-dot"></div></div>
"""


def generate_map(user_lat: float, user_lon: float,
                 dest_lat: float, dest_lon: float,
                 dest_name: str,
                 winner: JourneyOption,
                 all_options: list[JourneyOption],
                 routes_gdf: gpd.GeoDataFrame,
                 stops: list[Stop]) -> folium.Map:

    # ── Base map ──────────────────────────────────────────────────────────────
    mid_lat = (user_lat + dest_lat) / 2
    mid_lon = (user_lon + dest_lon) / 2
    m = folium.Map(location=[mid_lat, mid_lon], zoom_start=13,
                   tiles="CartoDB positron")

    # ── All routes — thin grey background ────────────────────────────────────
    all_fg = folium.FeatureGroup(name="All routes", show=True)
    for _, row in routes_gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        if isinstance(geom, LineString):
            coords = [(y, x) for x, y in geom.coords]
        elif isinstance(geom, MultiLineString):
            coords = [(y, x) for part in geom.geoms for x, y in part.coords]
        else:
            continue
        folium.PolyLine(coords, color="#cccccc", weight=1.5, opacity=0.5,
                        tooltip=row["route_name"]).add_to(all_fg)
    all_fg.add_to(m)

    # ── Runner-up routes — thin coloured ─────────────────────────────────────
    runners_fg = folium.FeatureGroup(name="Other candidates", show=True)
    runner_colors = ["#3498db", "#9b59b6", "#1abc9c", "#f39c12"]
    for i, opt in enumerate(all_options[1:5], 0):
        rrow = routes_gdf[routes_gdf["route_name"] == opt.route_name]
        if rrow.empty:
            continue
        geom = rrow.iloc[0].geometry
        if geom is None or geom.is_empty:
            continue
        if isinstance(geom, LineString):
            coords = [(y, x) for x, y in geom.coords]
        elif isinstance(geom, MultiLineString):
            coords = [(y, x) for part in geom.geoms for x, y in part.coords]
        else:
            continue
        color = runner_colors[i % len(runner_colors)]
        folium.PolyLine(
            coords, color=color, weight=4, opacity=0.55, dash_array="8 5",
            tooltip=f"Route {opt.route_name} — {opt.total_time_min:.0f} min",
        ).add_to(runners_fg)
    runners_fg.add_to(m)

    # ── Optimal route — thick solid ───────────────────────────────────────────
    winner_fg = folium.FeatureGroup(name=f"Best route ({winner.route_name})", show=True)
    rrow = routes_gdf[routes_gdf["route_name"] == winner.route_name]
    if not rrow.empty:
        geom = rrow.iloc[0].geometry
        if isinstance(geom, LineString):
            coords = [(y, x) for x, y in geom.coords]
        elif isinstance(geom, MultiLineString):
            coords = [(y, x) for part in geom.geoms for x, y in part.coords]
        else:
            coords = []
        if coords:
            folium.PolyLine(
                coords, color="#27ae60", weight=7, opacity=0.9,
                tooltip=f"Route {winner.route_name}",
            ).add_to(winner_fg)
    winner_fg.add_to(m)

    # ── Dotted walking line: user → boarding stop ─────────────────────────────
    walk_fg = folium.FeatureGroup(name="Walking legs", show=True)
    folium.PolyLine(
        [(user_lat, user_lon), (winner.board_stop.lat, winner.board_stop.lon)],
        color="#e67e22", weight=3, dash_array="6 8", opacity=0.9,
        tooltip=f"Walk {winner.board_dist_m:.0f} m to board",
    ).add_to(walk_fg)
    # Dotted walking line: alighting stop → destination
    folium.PolyLine(
        [(winner.alight_stop.lat, winner.alight_stop.lon), (dest_lat, dest_lon)],
        color="#e67e22", weight=3, dash_array="6 8", opacity=0.9,
        tooltip=f"Walk {winner.alight_dist_m:.0f} m to destination",
    ).add_to(walk_fg)
    walk_fg.add_to(m)

    # ── User position — pulsing red icon ─────────────────────────────────────
    folium.Marker(
        [user_lat, user_lon],
        icon=folium.DivIcon(html=PULSE_CSS, icon_size=(24, 24), icon_anchor=(12, 12)),
        tooltip="Your location",
    ).add_to(m)

    # ── Boarding stop marker ──────────────────────────────────────────────────
    board_label = (
        f"<b>Board here</b><br>"
        f"{winner.board_stop.name}<br>"
        f"Walk {winner.board_dist_m:.0f} m from you<br>"
        f"Catch Route <b>{winner.route_name}</b>"
    )
    folium.CircleMarker(
        [winner.board_stop.lat, winner.board_stop.lon],
        radius=10, color="#27ae60", fill=True, fill_color="#2ecc71", fill_opacity=0.9,
        popup=folium.Popup(board_label, max_width=220),
        tooltip=f"Board: {winner.board_stop.name}",
    ).add_to(m)

    # ── Alighting stop marker ─────────────────────────────────────────────────
    alight_label = (
        f"<b>Alight here</b><br>"
        f"{winner.alight_stop.name}<br>"
        f"Walk {winner.alight_dist_m:.0f} m to {dest_name}"
    )
    folium.CircleMarker(
        [winner.alight_stop.lat, winner.alight_stop.lon],
        radius=10, color="#8e44ad", fill=True, fill_color="#9b59b6", fill_opacity=0.9,
        popup=folium.Popup(alight_label, max_width=220),
        tooltip=f"Alight: {winner.alight_stop.name}",
    ).add_to(m)

    # ── Destination marker ────────────────────────────────────────────────────
    folium.Marker(
        [dest_lat, dest_lon],
        icon=folium.Icon(color="red", icon="flag", prefix="fa"),
        tooltip=dest_name,
        popup=f"<b>{dest_name}</b>",
    ).add_to(m)

    # ── Info panel (floating HTML box) ────────────────────────────────────────
    walk_m     = winner.board_dist_m
    bus_km     = winner.bus_dist_m / 1000
    total_min  = winner.total_time_min
    method_txt = "nearest stop" if winner.board_method == "stop" else "roadside flag"

    panel_html = f"""
    <div style="position:fixed;bottom:30px;left:30px;z-index:9999;
                background:#fff;border-radius:10px;padding:14px 18px;
                box-shadow:0 4px 16px rgba(0,0,0,.25);font-family:sans-serif;
                max-width:320px;border-left:6px solid #27ae60;">
      <div style="font-size:15px;font-weight:700;color:#27ae60;margin-bottom:6px;">
        Best Route &nbsp;&#x2714;</div>
      <div style="font-size:20px;font-weight:800;margin-bottom:4px;">
        Route {winner.route_name}</div>
      <div style="color:#555;font-size:13px;line-height:1.7;">
        Walk <b>{walk_m:.0f} m</b> to <b>{winner.board_stop.name}</b><br>
        ({method_txt}) to catch Route <b>{winner.route_name}</b><br>
        Alight at <b>{winner.alight_stop.name}</b><br>
        Walk <b>{winner.alight_dist_m:.0f} m</b> to {dest_name}<br>
        Fare: <b>KSh {winner.fare_ksh:.0f}</b>
        &nbsp;|&nbsp; Est. <b>{total_min:.0f} min</b> total
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(panel_html))

    # ── Layer control ─────────────────────────────────────────────────────────
    folium.LayerControl(collapsed=False).add_to(m)

    return m


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main(user_lat: float  = -1.2921,
         user_lon: float  = 36.8219,
         dest_lat: float  = -1.3201,
         dest_lon: float  = 36.7029,
         dest_name: str   = "Karen"):

    log("Nairobi Transit Snap-to-Network Router")
    log(f"User : ({user_lat}, {user_lon})")
    log(f"Dest : {dest_name} ({dest_lat}, {dest_lon})")

    # ── Load data ─────────────────────────────────────────────────────────────
    log("Loading routes ...")
    routes_gdf   = load_routes_gdf()
    transit_meta = pd.read_csv(TRANSIT_CSV)
    log(f"  {len(routes_gdf)} route polylines loaded")

    log("Loading stops ...")
    stops = load_stops(routes_gdf)

    # ── Nearest stops (numpy vectorised) ─────────────────────────────────────
    nearest = snap_to_stop(user_lat, user_lon, stops, k=5)
    log("Nearest stops to user:")
    for d, s in nearest:
        log(f"  {d:6.0f} m  [{s.kind:8s}]  {s.name}")

    # ── Filter routes near destination FIRST (cheap vectorised op) ──────────
    log("Filtering routes near destination ...")
    candidate_routes = filter_routes_near_dest(dest_lat, dest_lon, routes_gdf)
    log(f"  {len(candidate_routes)} routes pass within {DEST_SNAP_M} m of {dest_name}")
    if not candidate_routes:
        log("  Widening search to 3 km ...")
        candidate_routes = filter_routes_near_dest(dest_lat, dest_lon, routes_gdf, max_m=3000)

    # ── Snap-to-road only for the candidate routes (fast - small subset) ─────
    log(f"Snapping user to {len(candidate_routes)} candidate route polylines ...")
    road_snaps = snap_to_road(user_lat, user_lon, routes_gdf,
                              only_routes=candidate_routes)

    # ── Score each candidate ─────────────────────────────────────────────────
    log("Scoring candidates ...")
    options: list[JourneyOption] = []
    for rname in candidate_routes:
        opt = score_route(rname, user_lat, user_lon, dest_lat, dest_lon,
                          stops, road_snaps, transit_meta)
        if opt:
            options.append(opt)

    if not options:
        log("ERROR: No routable options found.")
        return

    options.sort(key=lambda o: o.total_time_min)
    winner = options[0]

    # ── Console output ────────────────────────────────────────────────────────
    log("=" * 55)
    log(f"  WINNER: Route {winner.route_name}")
    log("=" * 55)
    log(f"  Walk {winner.board_dist_m:.0f} m to [{winner.board_stop.name}]")
    log(f"  Board method : {winner.board_method}")
    log(f"  Ride {winner.bus_dist_m/1000:.1f} km  alight at [{winner.alight_stop.name}]")
    log(f"  Walk {winner.alight_dist_m:.0f} m to {dest_name}")
    log(f"  Fare         : KSh {winner.fare_ksh:.0f}")
    log(f"  Total time   : {winner.total_time_min:.1f} min")
    log("=" * 55)

    if len(options) > 1:
        log("Other options (ranked):")
        for opt in options[1:5]:
            log(f"  Route {opt.route_name:8s}  {opt.total_time_min:.0f} min  "
                f"KSh {opt.fare_ksh:.0f}  board: {opt.board_stop.name[:30]}")

    # ── Generate map ──────────────────────────────────────────────────────────
    log(f"Generating map: {MAP_OUT}")
    m = generate_map(user_lat, user_lon, dest_lat, dest_lon,
                     dest_name, winner, options, routes_gdf, stops)
    m.save(MAP_OUT)
    log("Done. Map saved.")
    _LOG.close()
    webbrowser.open(f"file://{MAP_OUT}")


if __name__ == "__main__":
    main(
        user_lat  = -1.2921,
        user_lon  = 36.8219,
        dest_lat  = -1.3201,
        dest_lon  = 36.7029,
        dest_name = "Karen",
    )
