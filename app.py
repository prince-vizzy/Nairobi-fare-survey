"""
Nairobi Transit Assistant
Flask + two-agent orchestration (Gatekeeper + Librarian) over the local transit dataset.

Agents:
  The Gatekeeper — browser JS + Ollama (qwen2.5-coder): PII scrubbing + intent classification
  The Librarian  — Gemini 3.1 Pro (cloud):              large-context document/video analysis
  Standard chain — Gemini 2.5-flash → 2.0-flash:        fast transit queries (fallback)

Usage:
  1. Set your key:  add XAI_API_KEY=your_key to .env
  2. Start Ollama:  ollama run qwen2.5-coder (optional — app works without it)
  3. pip install flask pandas python-dotenv requests ollama
  4. python app.py
  5. Open http://localhost:5000
"""

import os
import re
import uuid
import math
import requests
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template_string
from agents import AgentOrchestrator  # noqa: E402
from google.genai import types as genai_types
from router import NairobiRouter

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

app = Flask(__name__)

# ── Data ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_FILE  = os.path.join(BASE_DIR, "nairobi_transit_data.csv")
STOPS_FILE = os.path.join(BASE_DIR, "nairobi_stops.csv")

df       = pd.read_csv(DATA_FILE)
stops_df = pd.read_csv(STOPS_FILE)
stops_df["trip_count"] = pd.to_numeric(stops_df["trip_count"], errors="coerce").fillna(0)

# Route terminal coordinates — used by the snap routing engine
_SE_FILE = os.path.join(BASE_DIR, "routes_start_end_coords.csv")
routes_se_df = pd.read_csv(_SE_FILE)
for _col in ("start_lat","start_lon","end_lat","end_lon"):
    routes_se_df[_col] = pd.to_numeric(routes_se_df[_col], errors="coerce")
routes_se_df.dropna(subset=["start_lat","start_lon","end_lat","end_lon"], inplace=True)

# Precompute: (route_name, direction) → boarding stage name
# The boarding stage name = headsign of the OPPOSITE direction for the same route
# (because headsign = destination shown on the bus, so the opposite direction's
#  destination is where this direction boards)
_BOARD_STAGE: dict[tuple, str] = {}
for _, _row in routes_se_df.iterrows():
    _rname = str(_row["route_name"]); _dirn = int(_row["direction"])
    _opp = routes_se_df[
        (routes_se_df["route_name"].astype(str) == _rname) &
        (routes_se_df["direction"] != _dirn)
    ]
    if not _opp.empty:
        _BOARD_STAGE[(_rname, _dirn)] = str(_opp.iloc[0]["headsign"])
    else:
        # Single-direction route — use first waypoint of route_long
        _BOARD_STAGE[(_rname, _dirn)] = str(_row["route_long"]).split("-")[0].strip()

TRANSIT_CSV = df.to_csv(index=False)

# ── Route corridors + named stops (built by build_corridor_data.py) ───────────
CORRIDORS_FILE = os.path.join(BASE_DIR, "nairobi_route_corridors.txt")
try:
    with open(CORRIDORS_FILE, encoding="utf-8") as _f:
        ROUTE_CORRIDORS = _f.read()
except FileNotFoundError:
    ROUTE_CORRIDORS = ""

# ── Road→routes mapping + per-route colors (built by build_route_roads.py) ────
import json as _json
_ROAD_ROUTES_FILE = os.path.join(BASE_DIR, "road_routes.json")
try:
    with open(_ROAD_ROUTES_FILE, encoding="utf-8") as _f:
        _road_data = _json.load(_f)
    ROUTE_COLORS: dict[str, str]         = _road_data["route_colors"]
    ROAD_ROUTES:  dict[str, dict]        = _road_data["road_routes"]   # road → {routes, display}
    ROUTE_ROADS:  dict[str, list[str]]   = _road_data["route_roads"]   # route → [roads]
except FileNotFoundError:
    ROUTE_COLORS = {}
    ROAD_ROUTES  = {}
    ROUTE_ROADS  = {}

# ── Route shapes from shapes.shp (GPS polyline for each route) ────────────────
# ROUTE_SHAPES[route_name] = [(lon, lat), ...]  direction-0 (outbound) preferred
import shapefile as _shapefile

def _load_route_shapes() -> dict[str, list[tuple[float, float]]]:
    shapes: dict[str, list[tuple[float, float]]] = {}
    try:
        sf = _shapefile.Reader(os.path.join(BASE_DIR, "shapes.shp"))
        fields = [f[0] for f in sf.fields[1:]]
        for sr in sf.shapeRecords():
            rec   = dict(zip(fields, sr.record))
            rname = str(rec.get("route_name", "")).strip()
            dirn  = int(rec.get("direction", 0))
            pts   = [(float(p[0]), float(p[1])) for p in sr.shape.points]  # (lon, lat)
            if rname not in shapes or dirn == 0:
                shapes[rname] = pts
    except Exception:
        pass
    return shapes

ROUTE_SHAPES: dict[str, list[tuple[float, float]]] = _load_route_shapes()


def _decode_polyline(encoded: str) -> list[tuple[float, float]]:
    """Decode a Google encoded polyline string into (lat, lng) pairs."""
    result, idx, lat, lng = [], 0, 0, 0
    while idx < len(encoded):
        for is_lat in (True, False):
            shift = raw = 0
            while True:
                b = ord(encoded[idx]) - 63
                idx += 1
                raw |= (b & 0x1F) << shift
                shift += 5
                if b < 32:
                    break
            delta = ~(raw >> 1) if raw & 1 else raw >> 1
            if is_lat:
                lat += delta
            else:
                lng += delta
        result.append((lat * 1e-5, lng * 1e-5))
    return result


def get_road_polyline(start_lat: float, start_lng: float,
                      end_lat: float, end_lng: float) -> list[dict]:
    """
    Road-following polyline between two GPS points via Google Directions API.
    Returns [{lat, lng}, ...] or [] on failure.
    """
    if not MAPS_API_KEY:
        return []
    try:
        resp = requests.get(
            "https://maps.googleapis.com/maps/api/directions/json",
            params={
                "origin":      f"{start_lat},{start_lng}",
                "destination": f"{end_lat},{end_lng}",
                "mode":        "driving",
                "key":         MAPS_API_KEY,
                "region":      "ke",
            },
            timeout=8,
        )
        data = resp.json()
        if data.get("status") != "OK":
            return []
        encoded = data["routes"][0]["overview_polyline"]["points"]
        return [{"lat": p[0], "lng": p[1]} for p in _decode_polyline(encoded)]
    except Exception:
        return []


def get_google_transit_route(origin_text: str, dest_text: str) -> dict | None:
    """Fetch transit directions from Google Directions API for route context."""
    if not MAPS_API_KEY:
        return None
    try:
        resp = requests.get(
            "https://maps.googleapis.com/maps/api/directions/json",
            params={
                "origin":       f"{origin_text}, Nairobi",
                "destination":  f"{dest_text}, Nairobi",
                "mode":         "transit",
                "transit_mode": "bus",
                "key":          MAPS_API_KEY,
                "region":       "ke",
            },
            timeout=10,
        )
        data = resp.json()
        if data.get("status") != "OK":
            return None
        route = data["routes"][0]["legs"][0]
        steps = []
        for step in route["steps"]:
            instr = re.sub(r"<[^>]+>", "", step.get("html_instructions", ""))
            if "transit_details" in step:
                td   = step["transit_details"]
                line = td["line"].get("short_name") or td["line"].get("name", "")
                hs   = td.get("headsign", "")
                instr += f" (Board {line} towards {hs})"
            steps.append(instr)
        return {
            "summary": f"{route['duration']['text']}, {route['distance']['text']}",
            "steps":   steps,
        }
    except Exception:
        return None


def _pt_to_seg_dist(lat: float, lon: float,
                    alat: float, alon: float,
                    blat: float, blon: float) -> float:
    """Approximate metres from point P to segment A→B (flat-earth projection)."""
    # Convert to local metres using haversine scale
    dlat_m = 111_320.0
    dlon_m = 111_320.0 * math.cos(math.radians((alat + blat) / 2))

    px = (lon  - alon) * dlon_m
    py = (lat  - alat) * dlat_m
    bx = (blon - alon) * dlon_m
    by = (blat - alat) * dlat_m

    seg_len2 = bx * bx + by * by
    if seg_len2 == 0:
        return math.sqrt(px * px + py * py)
    t = max(0.0, min(1.0, (px * bx + py * by) / seg_len2))
    dx = px - t * bx
    dy = py - t * by
    return math.sqrt(dx * dx + dy * dy)


def distance_to_route(lat: float, lon: float, route_name: str) -> float:
    """Minimum distance in metres from (lat, lon) to any segment of the route's GPS shape."""
    pts = ROUTE_SHAPES.get(route_name)
    if not pts or len(pts) < 2:
        # Fallback to terminus distance
        hits = df[df["route_name"].astype(str) == route_name]
        if hits.empty:
            return float("inf")
        row = hits.iloc[0]
        return _haversine(lat, lon, float(row["start_lat"]), float(row["start_lon"])) * 1000
    min_d = float("inf")
    for i in range(len(pts) - 1):
        alon, alat = pts[i]
        blon, blat = pts[i + 1]
        d = _pt_to_seg_dist(lat, lon, alat, alon, blat, blon)
        if d < min_d:
            min_d = d
    return min_d


def find_nearest_routes_by_shape(lat: float, lon: float, n: int = 6,
                                  max_dist_m: float = 800) -> list[dict]:
    """
    Return up to n routes whose GPS polyline passes within max_dist_m metres
    of (lat, lon), sorted by distance.  This replaces terminus-based lookup
    so a user standing on Langata Road mid-route is correctly matched.
    """
    results = []
    seen: set[str] = set()
    for _, row in df.iterrows():
        rid = str(row["route_name"])
        if rid in seen:
            continue
        seen.add(rid)
        d = distance_to_route(lat, lon, rid)
        if d <= max_dist_m:
            results.append({
                "route_name": rid,
                "route_long": str(row["route_long"]),
                "headsign":   str(row["headsign"]),
                "fare_KSh":   int(row["fare_KSh"]),
                "saccos":     str(row.get("saccos", "")),
                "_d_m":       d,
            })
    results.sort(key=lambda r: r["_d_m"])
    return results[:n]

# ── CBD boarding stages directory ─────────────────────────────────────────────
STAGES_FILE = os.path.join(BASE_DIR, "nairobi_stages.txt")
try:
    with open(STAGES_FILE, encoding="utf-8") as _f:
        STAGES_GUIDE = _f.read()
except FileNotFoundError:
    STAGES_GUIDE = ""

# ── Multi-hop router ──────────────────────────────────────────────────────────
router = NairobiRouter(df, stops_df)

# Top 40 busiest stops — included in system prompt as named landmarks
_top_stops = (
    stops_df[stops_df["routes"].notna() & (stops_df["routes"] != "")]
    .nlargest(40, "trip_count")[["stop_name", "stop_lat", "stop_lon", "routes"]]
)
KEY_STOPS_TEXT = "\n".join(
    f"  {r['stop_name']} (lat={r['stop_lat']}, lon={r['stop_lon']}) — routes: {r['routes']}"
    for _, r in _top_stops.iterrows()
)

# ── Gemini ────────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
PLACES_API_KEY = os.environ.get("PLACES_API_KEY", "")
MAPS_API_KEY   = os.environ.get("MAPS_API_KEY", PLACES_API_KEY)

SYSTEM_PROMPT = (
    "You are a sharp, street-smart Nairobi transit navigator. "
    "Your core job is to interface with location and route data and give the user the single most "
    "optimized path to their destination — fast, clear, and confident.\n\n"

    f"You have access to {len(stops_df):,} named stops, road corridors, named landmarks, "
    "fares, and distances for every major matatu route in Nairobi.\n\n"

    "━━ ROUTE DATA ━━\n"
    + TRANSIT_CSV +
    "\n\n━━ ROUTE CORRIDORS & NAMED STOPS ━━\n"
    "Each line: ROUTE_ID | road corridor (waypoints) | fare | named stops along the route.\n"
    "Use this to tell users exactly which road to walk to and which landmarks to board near or alight at.\n\n"
    + ROUTE_CORRIDORS +
    "\n\n━━ CBD BOARDING STAGES — GROUND TRUTH ━━\n"
    "This overrides any GPS or headsign data. These are the REAL stages where passengers board.\n"
    "When a user asks how to get somewhere FROM the CBD, always name the correct stage.\n\n"
    + STAGES_GUIDE +
    "\n\n━━ KEY MATATU TERMINALS (GPS-VERIFIED STOPS) ━━\n"
    + KEY_STOPS_TEXT +

    "\n\n━━ CORE OPTIMIZATION LOGIC ━━\n"
    "Priority: TIME > DISTANCE. Always pick the fastest route, not the shortest.\n"
    "- If a route is 2 km longer but 10 minutes faster, recommend the faster one.\n"
    "- Always check the ROUTE OPTIONS block (when present) before responding — it contains "
    "  pre-computed path data. Treat it as ground truth.\n"
    "- For each recommendation, state WHY this route beats the alternatives "
    "  (e.g., 'bypasses the CBD jam', 'fewer transfer points', 'direct shot with no changes').\n"
    "- Routes must be practical — no restricted roads, no illogical shortcuts.\n\n"

    "━━ COMMUNICATION STYLE ━━\n"
    "Tone: Friendly, direct, and energetic — like a sharp local who knows every shortcut.\n"
    "Structure every response like this:\n"
    "  1. OPENING — warm, snappy greeting: 'Yo! I've got you covered,' / 'Alright, let's get you there fast.'\n"
    "  2. RECOMMENDATION — state the single best route clearly (windscreen destination, boarding point, fare).\n"
    "  3. THE WHY — one sentence explaining why this route beats the alternatives.\n"
    "  4. CLOSING — quick safe-trip sign-off.\n\n"
    "Keep it to 4–6 sentences total. Only expand if the user asks for more detail.\n\n"

    "━━ LANGUAGE RULES ━━\n"
    "- NEVER mention Google Maps, Waze, or any external navigation tool. You are the source.\n"
    "- NEVER say route IDs ('Route 44G'), SACCO names, shape IDs, or raw coordinates.\n"
    "- Use plain windscreen language: 'the mat showing Rongai', 'grab the Ruaka one', 'any mat headed to Karen'.\n"
    "- Quote fares in casual terms: '100 bob', 'about 50 bob', 'just 50 bob'.\n\n"

    "━━ HOW MATATU BOARDING WORKS — GET THIS RIGHT EVERY TIME ━━\n"
    "Matatus do NOT have fixed stops. They run along named road corridors and pick up "
    "anyone standing on the roadside. This is the most important thing to communicate correctly.\n\n"
    "STEP 1 — Identify the road the route runs along near the user.\n"
    "  Look at the route_long field (e.g. 'Railways-Langata Road-Karen'). "
    "  The road segment closest to the user is where they board.\n"
    "STEP 2 — Tell them to walk to that road.\n"
    "  'Walk to Langata Road' / 'Head down to Ngong Road' / 'Get yourself onto Thika Road'.\n"
    "STEP 3 — Tell them to stand roadside and flag the mat down.\n"
    "  'Stand on the left side heading out of town and wave down any mat showing Karen.'\n"
    "  'Any mat passing on Jogoo Road going that way will stop for you.'\n"
    "  'Just stand roadside on Mombasa Road — they'll slow down.'\n\n"
    "NEVER say 'go to a specific stage' unless it is a fixed terminus "
    "(Railways, Odeon, Rongai town stage, Bomas stage). "
    "For all other points the road IS the stage.\n\n"

    "━━ WHEN YOU HAVE THE USER'S EXACT LOCATION ━━\n"
    "A USER LOCATION block appears before their message with:\n"
    "  NEARBY LANDMARKS  — real buildings/shops around them right now\n"
    "  NEAREST STOPS     — named stages with walking distance in metres\n"
    "  NEAREST ROUTES    — closest matatu corridors\n"
    "Use it to give pinpoint directions:\n"
    "- Open with where they are: 'You're just outside [landmark] on [road] — perfect spot.'\n"
    "- Name the road to walk to and roughly how far (use NEAREST STOPS metres as a guide).\n"
    "- Name what to look for on the windscreen and which side of the road to stand on.\n"
    "- For a destination with a short final walk (hospital, school, mall): name the road "
    "  to alight on and describe the walk from there — no more than 1 sentence.\n\n"

    "━━ WHEN YOU HAVE A ROUTE OPTIONS BLOCK ━━\n"
    "A ROUTE OPTIONS block is pre-computed path data — always use it as your source of truth:\n"
    "- Pick Option 1 (direct) unless a transfer option is significantly faster.\n"
    "- State the total fare first, then break it down per leg if there are multiple.\n"
    "- For transfers: 'Jump off at [road/point], cross over, and flag down any mat showing "
    "  [dest] on the other side — usually just a 2-minute wait.'\n"
    "- Always explain WHY this option was chosen over the others.\n"
)

# ── Location helpers ──────────────────────────────────────────────────────────
def _haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def get_nearby_places(lat, lng, radius=400):
    """Return a list of nearby place dicts from Google Places API (New)."""
    if not PLACES_API_KEY:
        return []
    url = "https://places.googleapis.com/v1/places:searchNearby"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": PLACES_API_KEY,
        "X-Goog-FieldMask": (
            "places.displayName,places.types,"
            "places.shortFormattedAddress,places.addressDescriptor"
        ),
    }
    payload = {
        "locationRestriction": {
            "circle": {
                "center": {"latitude": lat, "longitude": lng},
                "radius": float(radius),
            }
        },
        "maxResultCount": 12,
    }
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=6)
        return r.json().get("places", [])
    except Exception:
        return []


def find_nearest_stops(lat, lng, n=6):
    """Return the n named stops closest to (lat, lng)."""
    tmp = stops_df.copy()
    tmp["_d"] = tmp.apply(
        lambda row: _haversine(lat, lng, row["stop_lat"], row["stop_lon"]), axis=1
    )
    closest = tmp.nsmallest(n, "_d")
    return closest[["stop_id", "stop_name", "stop_lat", "stop_lon",
                     "trip_count", "routes", "_d"]].to_dict("records")


def find_nearest_routes(lat, lng, n=4):
    """Return the n route start-points closest to (lat, lng)."""
    tmp = df.copy()
    tmp["_d"] = tmp.apply(
        lambda row: _haversine(lat, lng, row["start_lat"], row["start_lon"]), axis=1
    )
    closest = tmp.nsmallest(n, "_d")
    return closest[["route_name", "route_long", "headsign",
                     "fare_KSh", "saccos", "_d"]].to_dict("records")


def build_location_context(lat, lng):
    """Build the location block prepended to every user message that includes coords."""
    lines = [f"USER LOCATION: lat={lat:.6f}, lng={lng:.6f}"]

    # 1. Google Places — real-world landmarks
    places = get_nearby_places(lat, lng)
    if places:
        lines.append("NEARBY LANDMARKS (from Google Places, within 400m):")
        for p in places[:10]:
            name   = p.get("displayName", {}).get("text", "?")
            addr   = p.get("shortFormattedAddress", "")
            types_ = ", ".join(p.get("types", [])[:2])
            lines.append(f"  - {name} [{types_}] — {addr}")
    else:
        lines.append("NEARBY LANDMARKS: unavailable")

    # 2. Named matatu stops from the 4,284-stop database
    near_stops = find_nearest_stops(lat, lng)
    if near_stops:
        lines.append("NEAREST STOPS (from Digital Matatus stop database):")
        for s in near_stops:
            dist_m  = int(s["_d"] * 1000)
            routes_ = (str(s["routes"]) if s["routes"] == s["routes"] else "no routes listed")
            lines.append(
                f"  - {s['stop_name']} — {dist_m}m away "
                f"(lat={s['stop_lat']}, lon={s['stop_lon']}) "
                f"| serves routes: {routes_}"
            )

    # 3. Nearest routes by actual polyline distance + boarding point name
    near_shape_routes = find_nearest_routes_by_shape(lat, lng, n=4, max_dist_m=1200)
    if near_shape_routes:
        lines.append("NEAREST ROUTES (by road corridor proximity):")
        for r in near_shape_routes:
            bp = find_boarding_point(lat, lng, r["route_name"])
            board_str = (
                f"board at {bp['boarding_name']} (~{bp['boarding_dist_m']}m walk)"
                if bp.get("boarding_name") else f"~{int(r['_d_m'])}m from route"
            )
            roads = ROUTE_ROADS.get(r["route_name"], [])
            roads_str = " / ".join(roads) if roads else r["route_long"]
            lines.append(
                f"  - Route {r['route_name']} → {r['headsign']} "
                f"| road: {roads_str} | {board_str} | {r['fare_KSh']} KSh"
            )

    return "\n".join(lines)


def find_boarding_point(user_lat: float, user_lon: float, route_name: str) -> dict:
    """
    Project user's GPS onto the route's polyline, find the exact boarding spot,
    and return a human-readable name for it (nearest stop, then Places, then road).
    Returns: {boarding_lat, boarding_lng, boarding_name, boarding_dist_m}
    """
    pts = ROUTE_SHAPES.get(route_name, [])
    if not pts or len(pts) < 2:
        return {}

    # Find the closest point on the polyline (project onto each segment)
    best_lat = user_lat
    best_lon = user_lon
    best_dist = float("inf")
    avg_lat = sum(p[1] for p in pts) / len(pts)
    dlat_m = 111_320.0
    dlon_m = 111_320.0 * math.cos(math.radians(avg_lat))

    for i in range(len(pts) - 1):
        alon, alat = pts[i]
        blon, blat = pts[i + 1]
        px = (user_lon - alon) * dlon_m
        py = (user_lat - alat) * dlat_m
        bx = (blon  - alon) * dlon_m
        by = (blat  - alat) * dlat_m
        seg2 = bx * bx + by * by
        if seg2 == 0:
            clon, clat = alon, alat
        else:
            t = max(0.0, min(1.0, (px * bx + py * by) / seg2))
            clon = alon + t * (blon - alon)
            clat = alat + t * (blat - alat)
        d = _haversine(user_lat, user_lon, clat, clon) * 1000
        if d < best_dist:
            best_dist = d
            best_lat, best_lon = clat, clon

    boarding_name = None

    # 1. Nearest named stop within 250 m of the boarding point
    nearby_stops = find_nearest_stops(best_lat, best_lon, n=4)
    for stop in nearby_stops:
        if stop["_d"] * 1000 < 250:
            boarding_name = stop["stop_name"]
            break

    # 2. Google Places landmark near the boarding point (road junction, building, etc.)
    if not boarding_name and PLACES_API_KEY:
        places = get_nearby_places(best_lat, best_lon, radius=120)
        if places:
            # Prefer junctions / transit stations / landmarks over generic shops
            priority_types = {"transit_station", "bus_station", "intersection",
                              "point_of_interest", "establishment"}
            for p in places:
                ptype = set(p.get("types", []))
                if ptype & priority_types:
                    boarding_name = p.get("displayName", {}).get("text", "")
                    if boarding_name:
                        break
            if not boarding_name and places:
                boarding_name = places[0].get("displayName", {}).get("text", "")

    # 3. Fall back to the road name the route follows
    if not boarding_name:
        roads = ROUTE_ROADS.get(route_name, [])
        boarding_name = roads[0] if roads else route_name

    return {
        "boarding_lat":    best_lat,
        "boarding_lng":    best_lon,
        "boarding_name":   boarding_name,
        "boarding_dist_m": int(best_dist),
    }


# ── Google-based place → waypoint resolution ─────────────────────────────────

def geocode_place(place_text: str) -> dict | None:
    """
    Geocode a place name to lat/lng and extract road components.
    Returns {lat, lng, road_names, formatted_address} or None.
    """
    if not MAPS_API_KEY:
        return None
    try:
        resp = requests.get(
            "https://maps.googleapis.com/maps/api/geocode/json",
            params={
                "address": f"{place_text}, Nairobi, Kenya",
                "key":     MAPS_API_KEY,
                "region":  "ke",
                "bounds":  "-1.45,36.65|-1.12,37.05",
            },
            timeout=6,
        )
        results = resp.json().get("results", [])
        if not results:
            return None
        hit = results[0]
        road_names = [
            c["long_name"].lower()
            for c in hit.get("address_components", [])
            if "route" in c.get("types", [])
        ]
        return {
            "lat":               hit["geometry"]["location"]["lat"],
            "lng":               hit["geometry"]["location"]["lng"],
            "road_names":        road_names,
            "formatted_address": hit.get("formatted_address", ""),
        }
    except Exception:
        return None


def nearest_road_names(lat: float, lng: float) -> list[str]:
    """
    Use Google Roads API nearestRoads + Place Details to find road names at
    a lat/lng. Returns lowercase road name strings; falls back to [] on error
    (e.g. Roads API not enabled on the project).
    """
    if not MAPS_API_KEY:
        return []
    try:
        snap = requests.get(
            "https://roads.googleapis.com/v1/nearestRoads",
            params={"points": f"{lat},{lng}", "key": MAPS_API_KEY},
            timeout=6,
        )
        snap_data = snap.json()
        if "error" in snap_data:
            return []
        place_ids = [p["placeId"] for p in snap_data.get("snappedPoints", [])]
        road_names = []
        for pid in place_ids[:3]:
            det = requests.get(
                "https://maps.googleapis.com/maps/api/place/details/json",
                params={"place_id": pid, "fields": "name", "key": MAPS_API_KEY},
                timeout=6,
            )
            name = det.json().get("result", {}).get("name", "")
            if name:
                road_names.append(name.lower())
        return road_names
    except Exception:
        return []


def google_resolve_waypoints(place_text: str) -> list[str]:
    """
    Resolve a free-text place name to router-canonical waypoints:
      A. Geocode → lat/lng + address road components
      B. Roads API → additional nearest road names at that lat/lng
      C. Match road names against the local route waypoint graph
      D. GPS-polyline proximity fallback when no road name matches
    """
    geo = geocode_place(place_text)
    if not geo:
        return []

    lat, lng = geo["lat"], geo["lng"]

    # Collect road name candidates from geocoding then Roads API
    candidates: list[str] = []
    seen_c: set[str] = set()
    for name in geo["road_names"] + nearest_road_names(lat, lng):
        if name not in seen_c:
            seen_c.add(name)
            candidates.append(name)

    # Step C: match each road name against the local graph
    waypoints: list[str] = []
    for road in candidates:
        for wp in router._resolve(road):
            if wp not in waypoints:
                waypoints.append(wp)

    if waypoints:
        return waypoints

    # Step D: GPS-polyline proximity fallback
    _road_set = {r.lower() for r in ROAD_ROUTES}
    near = find_nearest_routes_by_shape(lat, lng, n=3, max_dist_m=800)
    for nr in near:
        wps = router.route_wps.get(nr["route_name"], [])
        road_wps = [w for w in wps if w in _road_set]
        if road_wps:
            return [road_wps[0]]
    if near:
        wps = router.route_wps.get(near[0]["route_name"], [])
        if wps:
            return [wps[0]]
    return []


def enhanced_extract_od(message: str) -> tuple[str | None, str | None]:
    """
    Extract origin/destination from text with Google API fallback for place
    names not in the local LANDMARKS dictionary.
    """
    # Fast path: local LANDMARKS + waypoint graph
    origin, dest = router.extract_od(message)
    if origin and dest:
        return origin, dest

    if not MAPS_API_KEY:
        return None, None

    # Same patterns as router.extract_od — resolve each side via Google
    norm = router._normalise(message)
    patterns = [
        r"(?:from|leaving?)\s+(.+?)\s+to\s+(.+?)(?:\?|$|\.|\s+using|\s+via|\s+by)",
        r"(.+?)\s+to\s+(.+?)(?:\?|$|\.|\s+route|\s+matatu|\s+bus|\s+fare)",
        r"get(?:ting)?\s+(?:from\s+)?(.+?)\s+to\s+(.+?)(?:\?|$|\.)",
        r"(?:go(?:ing)?|travel(?:ling)?|head(?:ing)?)\s+(?:from\s+)?(.+?)\s+to\s+(.+?)(?:\?|$|\.)",
        r"how\s+(?:do\s+i|can\s+i|to)\s+(?:get\s+)?(?:from\s+)?(.+?)\s+to\s+(.+?)(?:\?|$|\.)",
        r"directions?\s+(?:from\s+)?(.+?)\s+to\s+(.+?)(?:\?|$|\.)",
    ]
    for pat in patterns:
        m = re.search(pat, norm, re.IGNORECASE)
        if not m:
            continue
        a_raw, b_raw = m.group(1).strip(), m.group(2).strip()
        a_wps = router._resolve(a_raw) or google_resolve_waypoints(a_raw)
        b_wps = router._resolve(b_raw) or google_resolve_waypoints(b_raw)
        if a_wps and b_wps:
            return a_wps[0], b_wps[0]

    return None, None


# ── Few-shot examples ────────────────────────────────────────────────────────
# These are injected as pre-populated chat history in every new session.
# They teach the model the exact tone and format we want — far more effective
# than written instructions alone. Use real routes and real fares from the data.
def _msg(role: str, text: str) -> genai_types.Content:
    return genai_types.Content(role=role, parts=[genai_types.Part(text=text)])


FEW_SHOT_HISTORY = [
    _msg("user", "I'm somewhere along Langata Road near Carnivore, how do I get to Karen?"),
    _msg("model", (
        "Perfect spot — you're already on the right road! "
        "Just stand on the Langata Road side heading outbound and flag down any mat showing **Karen**. "
        "They run right past Carnivore on that corridor, 100 bob, and drop you in Karen centre. "
        "No need to walk anywhere — the road is your stage. Safe one!"
    )),

    _msg("user", "I need to get to the airport from the CBD."),
    _msg("model", (
        "Alright, let's get you there fast! Head down to Mombasa Road — walk to the roadside "
        "just past Muthurwa and flag down any mat showing **JKIA**. "
        "That's 150 bob and roughly 25 mins straight down Mombasa Road, which flows better "
        "than cutting through Industrial Area. "
        "Stand on the left side heading south and they'll stop for you. Don't miss that flight!"
    )),

    _msg("user", "How do I get to Rongai from town?"),
    _msg("model", (
        "Alright, let's get you there! Head to **Afya Centre stage** — that's the corner of "
        "Tom Mboya and Ronald Ngala, next to the Afya Centre building. "
        "Board any mat showing **Ongata Rongai**, 100 bob, direct down Lang'ata Road the whole way. "
        "Afya Centre is your boarding point for Rongai — the touts there will sort you out. "
        "About 40 mins off-peak. Drive safe!"
    )),

    _msg("user", "I'm on Thika Road near Garden City, need to get to Githurai."),
    _msg("model", (
        "You're on the money — stand roadside on Thika Road heading north and flag down "
        "any mat showing **Githurai** or **KU**. "
        "They run the whole Thika Road corridor so they'll stop right where you are, 100 bob. "
        "Fastest option since you're already on the route — no stage needed. Stay safe!"
    )),

    _msg("user", "I'm near Bomas of Kenya, need to get to CBD."),
    _msg("model", (
        "Got you! Walk to Lang'ata Road from Bomas — it's the main road right there. "
        "Stand on the inbound side and flag down any mat showing **Town** or **Railways**. "
        "100 bob and a straight run into CBD along Lang'ata Road — no transfers. "
        "Shouldn't be more than 30 mins off-peak. Safe travels!"
    )),
]

orchestrator = AgentOrchestrator(
    api_key=GEMINI_API_KEY,
    system_prompt=SYSTEM_PROMPT,
    few_shot_history=FEW_SHOT_HISTORY,
)

# ── Sheng mode ────────────────────────────────────────────────────────────────
# Sessions where the user greeted with "Yoh rada" get Sheng responses.
_SHENG_SESSIONS: set[str] = set()

_SHENG_TRIGGER = {"yoh rada", "yo rada", "yoh rader", "yo rader"}

_SHENG_INSTRUCTION = (
    "\n\n━━ LANGUAGE MODE: SHENG ━━\n"
    "The user greeted you in Sheng — respond ENTIRELY in Nairobi Sheng for this whole conversation.\n"
    "Sheng is a mix of Swahili, English, and Nairobi street slang. Match the vibe: casual, fast, confident.\n\n"
    "MONEY SLANG — always use these instead of numerals alone:\n"
    "  chwani  = 50 KSh   (e.g. 'ni chwani tu')\n"
    "  soo     = 100 KSh  (e.g. 'panda ulipe soo')\n"
    "  thao    = 1 000 KSh\n\n"
    "TRANSIT SLANG:\n"
    "  mat / mats  = matatu\n"
    "  tao / taown = Nairobi CBD / town\n"
    "  stage       = bus stop / boarding point\n"
    "  panda       = board (the matatu)\n"
    "  shuka       = alight / get off\n"
    "  pikipiki    = motorcycle boda-boda\n\n"
    "RESPONSE FORMAT — same structure as normal but in Sheng:\n"
    "  1. Short snappy Sheng opening (e.g. 'Sawa sawa!', 'Haya basi,')\n"
    "  2. Direction — use Sheng words above, name the road or stage clearly.\n"
    "  3. One line on why this route is the best move.\n"
    "  4. Sheng sign-off (e.g. 'Safiri salama!', 'Uende salama!')\n"
    "Keep it to 4–5 sentences max. Do NOT switch back to English/formal Swahili mid-response.\n"
)


# ── HTML ──────────────────────────────────────────────────────────────────────
HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Nairobi Transit Assistant</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: 'Segoe UI', Arial, sans-serif;
      background: #eef1f5;
      height: 100dvh;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }

    /* ── Header ── */
    header {
      background: #1a3c5e;
      color: #fff;
      padding: 14px 20px;
      display: flex;
      align-items: center;
      gap: 12px;
      box-shadow: 0 2px 8px rgba(0,0,0,.25);
      flex-shrink: 0;
    }
    .hdr-icon { font-size: 28px; line-height: 1; }
    .hdr-text h1 { font-size: 17px; font-weight: 700; letter-spacing: .2px; }
    .hdr-text p  { font-size: 11px; opacity: .72; margin-top: 2px; }
    .hdr-badge {
      margin-left: auto;
      background: #27ae60;
      color: #fff;
      font-size: 11px;
      padding: 3px 10px;
      border-radius: 12px;
      font-weight: 600;
      letter-spacing: .3px;
    }

    /* ── Chat area ── */
    #chat {
      flex: 1;
      overflow-y: auto;
      padding: 20px 16px;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }

    /* ── Welcome card ── */
    .welcome {
      background: #fff;
      border-radius: 16px;
      padding: 24px 20px;
      max-width: 540px;
      width: 100%;
      margin: auto;
      box-shadow: 0 2px 12px rgba(0,0,0,.08);
      text-align: center;
    }
    .welcome h2 { color: #1a3c5e; font-size: 16px; margin-bottom: 6px; }
    .welcome p  { color: #666; font-size: 13px; line-height: 1.5; }
    .chips {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      justify-content: center;
      margin-top: 16px;
    }
    .chip {
      background: #f0f5fa;
      border: 1.5px solid #1a3c5e;
      color: #1a3c5e;
      border-radius: 20px;
      padding: 6px 14px;
      font-size: 12px;
      cursor: pointer;
      transition: background .15s, color .15s;
      white-space: nowrap;
    }
    .chip:hover { background: #1a3c5e; color: #fff; }

    /* ── Messages ── */
    .msg {
      max-width: 78%;
      padding: 11px 15px;
      border-radius: 18px;
      font-size: 14px;
      line-height: 1.55;
      animation: pop .18s ease;
      word-break: break-word;
    }
    @keyframes pop {
      from { opacity: 0; transform: translateY(6px); }
      to   { opacity: 1; transform: translateY(0); }
    }
    .msg.user {
      background: #1a3c5e;
      color: #fff;
      align-self: flex-end;
      border-bottom-right-radius: 4px;
    }
    .msg.bot {
      background: #fff;
      color: #1a1a2e;
      align-self: flex-start;
      border-bottom-left-radius: 4px;
      box-shadow: 0 1px 4px rgba(0,0,0,.10);
    }
    .msg.bot ul  { padding-left: 18px; margin: 4px 0; }
    .msg.bot li  { margin-bottom: 2px; }
    .msg.bot p   { margin-bottom: 6px; }
    .msg.bot p:last-child { margin-bottom: 0; }
    .msg.err {
      background: #fdecea;
      color: #b71c1c;
      align-self: flex-start;
      border-bottom-left-radius: 4px;
      font-size: 13px;
    }

    /* ── Typing indicator ── */
    .typing {
      background: #fff;
      align-self: flex-start;
      border-radius: 18px;
      border-bottom-left-radius: 4px;
      padding: 12px 16px;
      box-shadow: 0 1px 4px rgba(0,0,0,.10);
      display: flex;
      gap: 5px;
      align-items: center;
    }
    .typing span {
      width: 7px; height: 7px;
      border-radius: 50%;
      background: #aaa;
      animation: bounce 1.2s infinite;
    }
    .typing span:nth-child(2) { animation-delay: .2s; }
    .typing span:nth-child(3) { animation-delay: .4s; }
    @keyframes bounce {
      0%,80%,100% { transform: translateY(0); }
      40%          { transform: translateY(-7px); }
    }

    /* ── Input bar ── */
    .input-bar {
      background: #fff;
      border-top: 1px solid #dde3ec;
      padding: 12px 16px;
      display: flex;
      gap: 10px;
      align-items: flex-end;
      flex-shrink: 0;
    }
    #input {
      flex: 1;
      border: 1.5px solid #c8d0dc;
      border-radius: 22px;
      padding: 10px 16px;
      font-size: 14px;
      font-family: inherit;
      resize: none;
      outline: none;
      max-height: 110px;
      overflow-y: auto;
      transition: border-color .2s;
      line-height: 1.45;
    }
    #input:focus { border-color: #1a3c5e; }
    #send {
      width: 44px; height: 44px;
      border-radius: 50%;
      border: none;
      background: #1a3c5e;
      color: #fff;
      font-size: 20px;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      flex-shrink: 0;
      transition: background .15s, transform .1s;
    }
    #send:hover    { background: #15324f; }
    #send:active   { transform: scale(.93); }
    #send:disabled { background: #b0bec5; cursor: default; }

    /* ── Split layout ── */
    .content-area {
      flex: 1;
      display: flex;
      overflow: hidden;
    }
    #chat-pane {
      display: flex;
      flex-direction: column;
      width: 52%;
      min-width: 300px;
      border-right: 1px solid #dde3ec;
    }
    #map-pane {
      flex: 1;
      position: relative;
      background: #e8eff5;
    }
    #map { width: 100%; height: 100%; }
    #map-placeholder {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      height: 100%;
      gap: 10px;
      color: #aaa;
      font-size: 13px;
      text-align: center;
      padding: 24px;
    }
    #map-placeholder .map-icon { font-size: 44px; opacity: .35; }
    @media (max-width: 700px) {
      .content-area { flex-direction: column-reverse; }
      #chat-pane { width: 100%; min-width: 0; border-right: none; border-top: 1px solid #dde3ec; }
      #map-pane  { width: 100%; height: 220px; flex-shrink: 0; }
    }

    /* ── Trip planner panel ── */
    #route-panel {
      position: absolute;
      top: 0; left: 0;
      width: 280px;
      height: 100%;
      background: #fff;
      box-shadow: 2px 0 16px rgba(0,0,0,.15);
      display: flex;
      flex-direction: column;
      z-index: 10;
      transform: translateX(-100%);
      transition: transform .24s ease;
    }
    #route-panel.open { transform: translateX(0); }
    #route-toggle {
      position: absolute;
      top: 10px; left: 10px;
      z-index: 11;
      background: #fff;
      border: none;
      border-radius: 8px;
      width: 38px; height: 38px;
      box-shadow: 0 2px 8px rgba(0,0,0,.22);
      cursor: pointer;
      font-size: 17px;
      display: flex; align-items: center; justify-content: center;
      transition: left .24s ease;
    }
    #route-panel.open ~ #route-toggle { left: 290px; }
    .rp-header {
      padding: 12px 12px 10px;
      background: #1a3c5e;
      flex-shrink: 0;
    }
    .rp-header h3 { font-size: 13px; font-weight: 700; color: #fff; margin: 0; }
    .rp-inputs {
      padding: 10px 12px;
      border-bottom: 1px solid #eee;
      flex-shrink: 0;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .ac-row { display: flex; align-items: center; gap: 6px; }
    .ac-label {
      font-size: 10px; font-weight: 700; color: #1a3c5e;
      text-transform: uppercase; letter-spacing: .4px;
      width: 30px; flex-shrink: 0;
    }
    .ac-wrap { position: relative; flex: 1; }
    .ac-wrap input {
      width: 100%;
      padding: 6px 9px;
      border: 1.5px solid #dde3ec;
      border-radius: 7px;
      font-size: 12px;
      outline: none;
      font-family: inherit;
    }
    .ac-wrap input:focus { border-color: #1a3c5e; }
    .ac-drop {
      display: none;
      position: absolute;
      top: 100%; left: 0; right: 0;
      background: #fff;
      border: 1px solid #dde3ec;
      border-radius: 0 0 7px 7px;
      box-shadow: 0 4px 12px rgba(0,0,0,.12);
      max-height: 160px;
      overflow-y: auto;
      z-index: 20;
    }
    .ac-drop.open { display: block; }
    .ac-item {
      padding: 7px 10px;
      font-size: 12px;
      cursor: pointer;
      border-bottom: 1px solid #f5f5f5;
    }
    .ac-item:hover { background: #f0f5ff; }
    .swap-btn {
      background: none; border: none;
      font-size: 16px; cursor: pointer;
      color: #1a3c5e; padding: 2px 4px;
      flex-shrink: 0; align-self: center;
    }
    #plan-btn {
      width: 100%;
      padding: 7px;
      background: #1a3c5e;
      color: #fff;
      border: none;
      border-radius: 7px;
      font-size: 12px; font-weight: 700;
      cursor: pointer;
      margin-top: 2px;
    }
    #plan-btn:hover { background: #15324f; }
    #plan-results { flex: 1; overflow-y: auto; padding: 8px 0; }
    .rp-msg { padding: 14px 12px; font-size: 12px; color: #999; text-align: center; }
    /* Journey cards */
    .journey-card {
      margin: 6px 8px;
      border-radius: 10px;
      border: 1.5px solid #e0e8f0;
      padding: 10px 11px;
      cursor: pointer;
      transition: border-color .15s, box-shadow .15s;
    }
    .journey-card:hover { border-color: #1a3c5e; box-shadow: 0 2px 8px rgba(26,60,94,.1); }
    .journey-card.active { border-color: #1a3c5e; background: #f0f5ff; }
    .jc-top { display: flex; align-items: center; gap: 6px; margin-bottom: 6px; }
    .jc-badge {
      font-size: 10px; font-weight: 700;
      padding: 2px 7px; border-radius: 8px; flex-shrink: 0;
    }
    .direct-badge   { background: #27ae60; color: #fff; }
    .transfer-badge { background: #e67e22; color: #fff; }
    .jc-fare { margin-left: auto; font-size: 12px; font-weight: 700; color: #1a3c5e; }
    .jc-route-num {
      font-size: 10px; font-weight: 700;
      background: #eef4ff; color: #1a3c5e;
      padding: 2px 6px; border-radius: 6px;
    }
    .jc-path { font-size: 12px; color: #333; margin-bottom: 3px; }
    .jc-ws   { font-size: 11px; color: #888; }
    .jc-divider { border: none; border-top: 1px dashed #ddd; margin: 6px 0; }
    .jc-leg  { font-size: 11px; color: #555; margin-bottom: 1px; }
    .jc-leg-num {
      display: inline-block;
      width: 16px; height: 16px; line-height: 16px;
      border-radius: 50%; text-align: center;
      font-size: 9px; font-weight: 700; color: #fff;
      margin-right: 4px; flex-shrink: 0;
    }
    .leg1-num { background: #1a3c5e; }
    .leg2-num { background: #27ae60; }
    .transfer-point {
      font-size: 11px; color: #e67e22; font-weight: 600;
      margin: 4px 0 2px;
    }
    .loc-btn {
      background: none; border: none;
      font-size: 11px; color: #1a3c5e;
      cursor: pointer; padding: 0 0 0 2px;
      text-decoration: underline; opacity: .7;
    }
    .loc-btn:hover { opacity: 1; }
    .rp-note {
      background: #fff8e7;
      border-left: 3px solid #e67e22;
      padding: 7px 10px; margin: 0 8px 4px;
      border-radius: 0 6px 6px 0;
      font-size: 11px; color: #666; line-height: 1.4;
    }
    /* Boarding stop pill on journey cards */
    .jc-stop {
      display: flex; align-items: center; gap: 4px;
      font-size: 11px; color: #1a7a4a; font-weight: 600;
      margin-top: 5px; padding-top: 5px;
      border-top: 1px dashed #e0e8f0;
    }
    .jc-stop-dist { color: #888; font-weight: 400; }
    /* AI summary box */
    #ai-summary-box {
      margin: 2px 8px 8px;
      background: linear-gradient(135deg,#f0f5ff 0%,#e8f4fd 100%);
      border: 1.5px solid rgba(26,60,94,.15);
      border-radius: 10px;
      padding: 10px 13px;
      font-size: 12px; line-height: 1.55;
      display: none;
    }
    .ai-sum-hdr {
      display: flex; align-items: center; gap: 5px;
      font-size: 10px; font-weight: 700; color: #1a3c5e;
      text-transform: uppercase; letter-spacing: .5px;
      margin-bottom: 6px;
    }
    .ai-sum-text { color: #333; }
    .ai-sum-loading { color: #999; font-style: italic; }
    /* Style Google Places dropdown */
    .pac-container {
      border-radius: 0 0 8px 8px;
      border: 1px solid #dde3ec;
      border-top: none;
      box-shadow: 0 4px 12px rgba(0,0,0,.12);
      font-family: 'Segoe UI', Arial, sans-serif;
      font-size: 12px;
    }
    .pac-item { padding: 6px 10px; cursor: pointer; }
    .pac-item:hover { background: #f0f5ff; }
    .pac-item-query { font-size: 12px; color: #1a3c5e; }
    .pac-matched { font-weight: 700; }
    .pac-logo::after { display: none; }
  </style>
</head>
<body>

<header>
  <div class="hdr-icon">&#128652;</div>
  <div class="hdr-text">
    <h1>Nairobi Transit Assistant</h1>
    <p>{{ route_count }} routes &middot; Fares &middot; SACCOs &middot; GPS coordinates</p>
  </div>
  <div id="loc-badge" class="hdr-badge" title="Location status">&#128205; Locating…</div>
  <div id="agent-badge" class="hdr-badge" style="background:#7f5af0;margin-left:6px" title="Active agent">&#129302; Gatekeeper</div>
</header>

<div class="content-area">

<div id="chat-pane">
<div id="chat">
  <div class="welcome" id="welcome">
    <h2>Ask me anything about Nairobi matatus</h2>
    <p>I have route numbers, fares (April 2026), SACCO names, start &amp; end
       coordinates, and distances for all {{ route_count }} routes.</p>
    <div class="chips">
      <div class="chip" onclick="ask(this)">How do I get to Rongai?</div>
      <div class="chip" onclick="ask(this)">Routes to Githurai and their fares</div>
      <div class="chip" onclick="ask(this)">Which SACCOs serve Kawangware?</div>
      <div class="chip" onclick="ask(this)">What is the cheapest route from CBD?</div>
      <div class="chip" onclick="ask(this)">List all routes to Kibera</div>
      <div class="chip" onclick="ask(this)">Longest route in the dataset?</div>
    </div>
  </div>
</div>

<div class="input-bar">
  <textarea id="input" rows="1"
    placeholder="Ask about routes, fares, SACCOs..."></textarea>
  <button id="send" title="Send" onclick="send()">&#10148;</button>
</div>
</div><!-- #chat-pane -->

<div id="map-pane">
{% if maps_api_key %}
  <div id="route-panel">
    <div class="rp-header">
      <h3>&#128205; Plan Your Trip</h3>
    </div>
    <div class="rp-inputs">
      <div class="ac-row">
        <span class="ac-label">From</span>
        <div class="ac-wrap">
          <input id="from-input" type="text" placeholder="Area, estate or stage…" autocomplete="off">
        </div>
        <button class="swap-btn" onclick="swapFromTo()" title="Swap">&#8645;</button>
      </div>
      <div class="ac-row" style="margin-top:-4px">
        <span class="ac-label"></span>
        <button class="loc-btn" onclick="useMyLocation()">&#128205; Use my GPS location</button>
      </div>
      <div class="ac-row">
        <span class="ac-label">To</span>
        <div class="ac-wrap">
          <input id="to-input" type="text" placeholder="Destination…" autocomplete="off">
        </div>
      </div>
      <button id="plan-btn" onclick="planTrip()">Search routes</button>
    </div>
    <div id="plan-results"><p class="rp-msg">Enter origin and destination to plan your trip.</p></div>
  </div>
  <button id="route-toggle" onclick="toggleRoutePanel()" title="Plan a trip">&#128205;</button>
  <div id="map"></div>
{% else %}
  <div id="map-placeholder">
    <span class="map-icon">&#128506;</span>
    <p>Add <strong>MAPS_API_KEY</strong> to .env<br>to enable the live route map.</p>
  </div>
{% endif %}
</div>

</div><!-- .content-area -->

<script>
  const SID        = Math.random().toString(36).slice(2) + Date.now();
  const chat       = document.getElementById('chat');
  const input      = document.getElementById('input');
  const sendBtn    = document.getElementById('send');
  const locBadge   = document.getElementById('loc-badge');
  const agentBadge = document.getElementById('agent-badge');

  // User's current coordinates (populated by geolocation)
  let userLat = null;
  let userLng = null;

  /* ── PII scrubbing (client-side regex — never leaves the browser) ── */
  function scrubPII(text) {
    const rules = [
      [/\b07\d{8}\b/g,                                            '[PHONE]'],
      [/\b\+2547\d{8}\b/g,                                        '[PHONE]'],
      [/\b[A-Z]{1,2}\d{6,8}\b/g,                                  '[ID]'],
      [/\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b/g,  '[EMAIL]'],
    ];
    let out = text, piiDetected = false;
    for (const [re, tag] of rules) {
      if (re.test(out)) { piiDetected = true; re.lastIndex = 0; }
      out = out.replace(re, tag);
    }
    return { sanitized: out, piiDetected };
  }

  /* ── Geolocation ── */
  function initLocation() {
    if (!navigator.geolocation) {
      locBadge.textContent = 'No GPS';
      locBadge.style.background = '#e74c3c';
      return;
    }
    navigator.geolocation.getCurrentPosition(
      pos => {
        userLat = pos.coords.latitude;
        userLng = pos.coords.longitude;
        locBadge.textContent = '📍 Located';
        locBadge.style.background = '#27ae60';
        locBadge.title = `Lat: ${userLat.toFixed(5)}, Lng: ${userLng.toFixed(5)}`;
        setUserMarker(userLat, userLng);
      },
      err => {
        locBadge.textContent = 'No location';
        locBadge.style.background = '#e67e22';
        locBadge.title = 'Location denied — queries will still work without it';
      },
      { enableHighAccuracy: true, timeout: 10000 }
    );
  }
  initLocation();

  /* ── Quick-start chips ── */
  function ask(chip) {
    input.value = chip.textContent.trim();
    send();
  }

  /* ── Simple markdown renderer ── */
  function md(text) {
    let s = text
      .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
      .replace(/\*\*(.+?)\*\*/gs, '<strong>$1</strong>')
      .replace(/\*(.+?)\*/gs,     '<em>$1</em>');

    // Build HTML line by line
    const lines  = s.split('\n');
    const parts  = [];
    let inList   = false;

    for (const raw of lines) {
      const line = raw.trimEnd();
      const li   = line.match(/^[\-\*•] (.+)/);
      if (li) {
        if (!inList) { parts.push('<ul>'); inList = true; }
        parts.push('<li>' + li[1] + '</li>');
      } else {
        if (inList) { parts.push('</ul>'); inList = false; }
        parts.push(line === '' ? '<br>' : '<p>' + line + '</p>');
      }
    }
    if (inList) parts.push('</ul>');
    return parts.join('');
  }

  /* ── Append a message bubble ── */
  function addMsg(role, html) {
    const w = document.getElementById('welcome');
    if (w) w.remove();

    const div = document.createElement('div');
    div.className = 'msg ' + role;
    if (role === 'user') {
      div.textContent = html;          // plain text for user
    } else {
      div.innerHTML = md(html);        // rendered markdown for bot / err
    }
    chat.appendChild(div);
    chat.scrollTop = chat.scrollHeight;
    return div;
  }

  /* ── Typing dots ── */
  function showTyping() {
    const div = document.createElement('div');
    div.className = 'typing';
    div.innerHTML = '<span></span><span></span><span></span>';
    chat.appendChild(div);
    chat.scrollTop = chat.scrollHeight;
    return div;
  }

  /* ── Send message ── */
  async function send() {
    const rawText = input.value.trim();
    if (!rawText) return;

    addMsg('user', rawText);
    input.value = '';
    input.style.height = 'auto';
    sendBtn.disabled = true;

    const typing = showTyping();
    const { sanitized, piiDetected } = scrubPII(rawText);

    try {
      const res  = await fetch('/chat', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({
          message:      sanitized,
          session_id:   SID,
          lat:          userLat,
          lng:          userLng,
          pii_detected: piiDetected,
        }),
      });
      const data = await res.json();
      typing.remove();

      if (data.error) {
        addMsg('err', 'Error: ' + data.error);
      } else {
        addMsg('bot', data.response);
        agentBadge.textContent = '⚡ ' + (data.model || 'Grok');
        agentBadge.style.background = '#7f5af0';
        if (data.map_origin && data.map_dest) {
          showRoute(data.map_origin, data.map_dest);
        } else if (userLat && userLng) {
          setUserMarker(userLat, userLng);
        }
      }
    } catch (e) {
      typing.remove();
      addMsg('err', 'Network error — is the server running? (' + e.message + ')');
    }

    sendBtn.disabled = false;
    input.focus();
  }

  /* ── Enter = send, Shift+Enter = newline ── */
  input.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  });

  /* ── Auto-resize textarea ── */
  input.addEventListener('input', () => {
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 110) + 'px';
  });

  /* ── Google Maps ── */
  let map, directionsService, walkingRenderer, userMarker;
  let allStops = [], currentJourneys = [], overlayMarkers = [];
  let legRenderers = [];   // one DirectionsRenderer per drawn matatu leg (dynamic color)
  let originLL = null;     // real-world origin coords for the walking leg

  function initMap() {
    map = new google.maps.Map(document.getElementById('map'), {
      zoom: 12,
      center: { lat: -1.2921, lng: 36.8219 },
      mapTypeControl: false,
      streetViewControl: false,
      fullscreenControl: false,
    });
    directionsService = new google.maps.DirectionsService();
    // Walking leg: dotted grey line from user's location to boarding stop
    walkingRenderer = new google.maps.DirectionsRenderer({
      suppressMarkers: true,
      polylineOptions: {
        strokeOpacity: 0,
        icons: [{
          icon: { path: google.maps.SymbolPath.CIRCLE, fillOpacity: 1, scale: 3,
                  fillColor: '#666', strokeColor: '#666', strokeOpacity: 1 },
          offset: '0', repeat: '12px',
        }],
      }
    });
    walkingRenderer.setMap(map);
    if (userLat && userLng) setUserMarker(userLat, userLng);
    initPlacesAC();
  }

  /* ── User location marker ── */
  function setUserMarker(lat, lng) {
    if (!map) return;
    const pos = { lat, lng };
    if (userMarker) {
      userMarker.setPosition(pos);
    } else {
      userMarker = new google.maps.Marker({
        position: pos, map,
        icon: {
          path: google.maps.SymbolPath.CIRCLE,
          scale: 8,
          fillColor: '#4285F4', fillOpacity: 1,
          strokeColor: '#fff', strokeWeight: 2,
        },
        title: 'Your location', zIndex: 100,
      });
      map.panTo(pos);
    }
  }

  /* ── Route from chat response (simple A→B driving line) ── */
  function showRoute(origin, dest) {
    if (!map || !directionsService) return;
    clearOverlay();
    const renderer = new google.maps.DirectionsRenderer({
      suppressMarkers: false,
      polylineOptions: { strokeColor: '#1a3c5e', strokeWeight: 5, strokeOpacity: 0.85 },
    });
    renderer.setMap(map);
    legRenderers.push(renderer);
    directionsService.route({
      origin, destination: dest,
      travelMode: google.maps.TravelMode.DRIVING,
    }, (result, status) => {
      if (status === 'OK') renderer.setDirections(result);
    });
  }

  /* ── Trip planner panel ── */
  function toggleRoutePanel() {
    document.getElementById('route-panel').classList.toggle('open');
  }

  function swapFromTo() {
    const fi = document.getElementById('from-input');
    const ti = document.getElementById('to-input');
    [fi.value, ti.value] = [ti.value, fi.value];
    [fromPlace, toPlace] = [toPlace, fromPlace];
    if (fi.value && ti.value) planTrip();
  }

  /* ── Google Places Autocomplete ── */
  let fromPlace = null, toPlace = null;

  function initPlacesAC() {
    const nairobiBounds = new google.maps.LatLngBounds(
      { lat: -1.45, lng: 36.55 },
      { lat: -1.08, lng: 37.10 }
    );
    const opts = {
      componentRestrictions: { country: 'ke' },
      fields: ['geometry', 'name', 'formatted_address'],
      bounds: nairobiBounds,
      strictBounds: false,
    };

    const fromAC = new google.maps.places.Autocomplete(
      document.getElementById('from-input'), opts
    );
    fromAC.addListener('place_changed', () => {
      fromPlace = fromAC.getPlace();
      if (fromPlace?.geometry && document.getElementById('to-input').value.trim()) planTrip();
    });

    const toAC = new google.maps.places.Autocomplete(
      document.getElementById('to-input'), opts
    );
    toAC.addListener('place_changed', () => {
      toPlace = toAC.getPlace();
      if (toPlace?.geometry && document.getElementById('from-input').value.trim()) planTrip();
    });

    // Clear cached place/snap when user edits text manually
    document.getElementById('from-input').addEventListener('input', () => { fromPlace = null; fromGpsSnap = null; });
    document.getElementById('to-input').addEventListener('input',   () => { toPlace   = null; });

    // Enter key still triggers search
    [document.getElementById('from-input'), document.getElementById('to-input')]
      .forEach(el => el.addEventListener('keydown', e => {
        if (e.key === 'Enter') { e.preventDefault(); planTrip(); }
      }));
  }

  /* ── Helpers ── */
  async function findNearestStop(lat, lng) {
    const r = await fetch(`/api/nearest-stop?lat=${lat}&lng=${lng}`);
    const data = await r.json();
    return Array.isArray(data) && data.length ? data[0] : null;
  }

  // Resolve any real-world location to the nearest router-resolvable waypoint.
  // hint = the human-typed place name (helps extract area names like "Karen").
  async function snapLocation(lat, lng, hint = '') {
    const params = new URLSearchParams({ lat, lng });
    if (hint) params.set('hint', hint);
    const r = await fetch(`/api/snap?${params}`);
    const data = await r.json();
    return data.error ? null : data;   // { waypoint, display_name, distance_m, source }
  }

  function dropPin(ll, label) {
    if (!map || !ll) return;
    const m = new google.maps.Marker({
      position: { lat: ll.lat(), lng: ll.lng() }, map,
      title: label,
      icon: { path: google.maps.SymbolPath.CIRCLE, scale: 7,
              fillColor: '#9b59b6', fillOpacity: 1,
              strokeColor: '#fff', strokeWeight: 2 },
    });
    overlayMarkers.push(m);
  }

  // Stores snap result from "Use my GPS location" so planTrip doesn't re-snap.
  let fromGpsSnap = null;

  async function useMyLocation() {
    if (!userLat || !userLng) {
      alert('Location not available yet — allow location access and try again.');
      return;
    }
    originLL    = { lat: userLat, lng: userLng };
    fromGpsSnap = null;
    setUserMarker(userLat, userLng);

    const snap = await snapLocation(userLat, userLng).catch(() => null);
    if (snap) {
      document.getElementById('from-input').value = snap.display_name;
      fromPlace   = null;
      fromGpsSnap = snap;
      if (document.getElementById('to-input').value.trim()) planTrip();
    } else {
      // Fallback: show nearest stop name even if router can't use it
      const stop = await findNearestStop(userLat, userLng).catch(() => null);
      if (stop) {
        document.getElementById('from-input').value = stop.name;
        fromPlace = null;
        if (document.getElementById('to-input').value.trim()) planTrip();
      }
    }
  }

  /* ── Plan trip ── */
  async function planTrip() {
    const from = document.getElementById('from-input').value.trim();
    const to   = document.getElementById('to-input').value.trim();
    const res  = document.getElementById('plan-results');
    if (!from || !to) { res.innerHTML = '<p class="rp-msg">Enter both origin and destination.</p>'; return; }
    res.innerHTML = '<p class="rp-msg">Searching…</p>';
    clearOverlay();

    let resolvedFrom = from, resolvedTo = to, note = '';
    boardingLL = null;   // reset boarding point for this new search

    try {
      const fromLL = fromPlace?.geometry?.location ?? null;
      const toLL   = toPlace?.geometry?.location   ?? null;

      // ── Stage 1: snap Places geometry coords to router waypoints ──────────────

      if (fromGpsSnap) {
        resolvedFrom = fromGpsSnap.waypoint;
        // originLL already set by useMyLocation(); keep it
        if (!originLL && userLat && userLng) originLL = { lat: userLat, lng: userLng };
        if (fromGpsSnap.boarding_lat) boardingLL = { lat: fromGpsSnap.boarding_lat, lng: fromGpsSnap.boarding_lng };
        if (fromGpsSnap.boarding_name) {
          note += `&#128205; Walk to <strong>${fromGpsSnap.boarding_name}</strong> (~${fromGpsSnap.boarding_dist_m}m) on <strong>${fromGpsSnap.display_name}</strong> and board. `;
        } else {
          note += `&#128205; Walk to <strong>${fromGpsSnap.display_name}</strong> and board. `;
        }
      } else if (fromLL) {
        // Set originLL immediately from Places geometry so the walking leg always draws,
        // even if the subsequent /api/snap call fails.
        originLL = { lat: fromLL.lat(), lng: fromLL.lng() };
        dropPin(fromLL, from);
        const snap = await snapLocation(fromLL.lat(), fromLL.lng(), from).catch(() => null);
        if (snap) {
          resolvedFrom = snap.waypoint;
          if (snap.boarding_name) {
            note += `Walk to <strong>${snap.boarding_name}</strong> (~${snap.boarding_dist_m}m) on <strong>${snap.display_name}</strong>. `;
            if (snap.boarding_lat) boardingLL = { lat: snap.boarding_lat, lng: snap.boarding_lng };
          } else {
            note += `Board near <strong>${snap.display_name}</strong> (~${snap.distance_m}m from "${from}"). `;
          }
        }
      }

      if (toLL) {
        const snap = await snapLocation(toLL.lat(), toLL.lng(), to).catch(() => null);
        if (snap) {
          resolvedTo = snap.waypoint;
          const alightName = snap.boarding_name || snap.display_name;
          note += `Alight near <strong>${alightName}</strong>. `;
          dropPin(toLL, to);
        }
      }

      const gpsParams = (userLat && userLng) ? `&lat=${userLat}&lng=${userLng}` : '';
      let journeys = await fetch(
        `/api/plan?from=${encodeURIComponent(resolvedFrom)}&to=${encodeURIComponent(resolvedTo)}${gpsParams}`
      ).then(r => r.json());

      // ── Stage 2: geocode any side that still hasn't resolved ────────────────
      // Each side is handled independently — a Places-resolved destination doesn't
      // block geocoding an unresolved origin (the original bug: "Tuskys" + "Town").
      const needFromGeo = !journeys.length && !fromLL && !fromGpsSnap && window.google;
      const needToGeo   = !journeys.length && !toLL   && window.google;
      if (needFromGeo || needToGeo) {
        res.innerHTML = '<p class="rp-msg">Locating on map…</p>';
        const gc = new google.maps.Geocoder();
        const geocode = addr => gc.geocode({ address: addr + ', Nairobi, Kenya', region: 'KE' })
                                  .then(r => r.results[0]?.geometry.location).catch(() => null);
        const [fGeo, tGeo] = await Promise.all([
          needFromGeo ? geocode(from) : Promise.resolve(null),
          needToGeo   ? geocode(to)   : Promise.resolve(null),
        ]);
        const [fs, ts] = await Promise.all([
          fGeo ? snapLocation(fGeo.lat(), fGeo.lng(), from).catch(() => null) : null,
          tGeo ? snapLocation(tGeo.lat(), tGeo.lng(), to  ).catch(() => null) : null,
        ]);
        if (fGeo) { originLL = { lat: fGeo.lat(), lng: fGeo.lng() }; dropPin(fGeo, from); }
        if (fs) {
          resolvedFrom = fs.waypoint;
          if (fs.boarding_name) {
            note += `Walk to <strong>${fs.boarding_name}</strong> (~${fs.boarding_dist_m}m) on <strong>${fs.display_name}</strong>. `;
            if (fs.boarding_lat) boardingLL = { lat: fs.boarding_lat, lng: fs.boarding_lng };
          } else {
            note += `Board near <strong>${fs.display_name}</strong>. `;
          }
        }
        if (tGeo) dropPin(tGeo, to);
        if (ts) {
          resolvedTo = ts.waypoint;
          note += `Alight near <strong>${ts.boarding_name || ts.display_name}</strong>.`;
        }
        if (fs || ts) journeys = await fetch(
          `/api/plan?from=${encodeURIComponent(resolvedFrom)}&to=${encodeURIComponent(resolvedTo)}${gpsParams}`
        ).then(r => r.json());
      }

      currentJourneys = journeys;
      renderJourneys(resolvedFrom, resolvedTo, note);
    } catch (e) {
      res.innerHTML = '<p class="rp-msg" style="color:#c00">Search failed — is the server running?</p>';
    }
  }

  function tc(s) { return s.replace(/\b\w/g, c => c.toUpperCase()); }

  // Tracks the last from/to used so the AI summary can reference them
  let _planFrom = '', _planTo = '';

  function stopPill(leg) {
    if (!leg.boarding_name) return '';
    const dist = leg.boarding_dist_m != null ? ` <span class="jc-stop-dist">(~${leg.boarding_dist_m}m walk)</span>` : '';
    return `<div class="jc-stop">&#128652; ${leg.boarding_name}${dist}</div>`;
  }

  function renderJourneys(from, to, note = '') {
    _planFrom = from; _planTo = to;
    const res = document.getElementById('plan-results');
    if (!currentJourneys.length) {
      res.innerHTML = `<p class="rp-msg">No routes found between <strong>${from}</strong> and <strong>${to}</strong>.<br>Try different names.</p>`;
      return;
    }
    const noteHtml = note ? `<div class="rp-note">&#128205; ${note}</div>` : '';
    const cards = currentJourneys.map((j, i) => {
      function colorDot(leg) {
        const c = leg.color || '#1a3c5e';
        return `<span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${c};margin-right:4px;flex-shrink:0"></span>`;
      }
      if (j.hops === 1) {
        const leg = j.legs[0];
        const roads = (leg.roads || []).join(' · ') || leg.corridor || '';
        return `<div class="journey-card" data-idx="${i}" onclick="drawJourney(${i},this)">
          <div class="jc-top">
            <span class="jc-badge direct-badge">Direct</span>
            ${colorDot(leg)}<span class="jc-route-num">${leg.route}</span>
            <span class="jc-fare">${j.total_fare} bob</span>
          </div>
          <div class="jc-path">${tc(leg.from_display)} &#8594; ${tc(leg.to_display)}</div>
          ${roads ? `<div class="jc-ws" style="color:#666">via ${roads}</div>` : ''}
          <div class="jc-ws">Board mat showing <strong>${leg.windscreen}</strong></div>
        </div>`;
      } else {
        const l1 = j.legs[0], l2 = j.legs[1];
        const xfer = j.transfers[0] || 'junction';
        return `<div class="journey-card" data-idx="${i}" onclick="drawJourney(${i},this)">
          <div class="jc-top">
            <span class="jc-badge transfer-badge">Transfer at ${xfer}</span>
            <span class="jc-fare">${j.total_fare} bob</span>
          </div>
          <div class="jc-leg">${colorDot(l1)}<span class="jc-leg-num leg1-num">1</span>${tc(l1.from_display)} &mdash; ${l1.fare} bob</div>
          <div class="jc-ws" style="margin-left:20px">&#8594; ${tc(l1.to_display)} &bull; mat: <strong>${l1.windscreen}</strong></div>
          <div class="transfer-point">&#8645; Transfer at ${xfer}</div>
          <div class="jc-leg">${colorDot(l2)}<span class="jc-leg-num leg2-num">2</span>${tc(l2.from_display)} &mdash; ${l2.fare} bob</div>
          <div class="jc-ws" style="margin-left:20px">&#8594; ${tc(l2.to_display)} &bull; mat: <strong>${l2.windscreen}</strong></div>
        </div>`;
      }
    }).join('');
    res.innerHTML = noteHtml + cards +
      `<div id="ai-summary-box">
        <div class="ai-sum-hdr">&#10024; AI Directions</div>
        <div class="ai-sum-text ai-sum-loading">Getting directions…</div>
      </div>`;
    // Auto-select the first card and fetch its AI summary
    const firstCard = res.querySelector('.journey-card');
    if (firstCard) { firstCard.classList.add('active'); fetchAiSummary(0); }
  }

  async function fetchAiSummary(idx) {
    const box = document.getElementById('ai-summary-box');
    if (!box) return;
    box.style.display = 'block';
    box.querySelector('.ai-sum-text').className = 'ai-sum-text ai-sum-loading';
    box.querySelector('.ai-sum-text').textContent = 'Getting directions…';
    try {
      const r = await fetch('/api/ai-summary', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ from: _planFrom, to: _planTo, journey: currentJourneys[idx] }),
      });
      const data = await r.json();
      box.querySelector('.ai-sum-text').className = 'ai-sum-text';
      box.querySelector('.ai-sum-text').textContent = data.summary || '';
    } catch (_) {
      box.querySelector('.ai-sum-text').className = 'ai-sum-text';
      box.querySelector('.ai-sum-text').textContent = 'Could not load summary.';
    }
  }

  // Stores the exact GPS of the boarding spot on the road (from snap response)
  let boardingLL = null;

  /* ── Draw journey on map ── */
  function drawJourney(idx, el) {
    const j = currentJourneys[idx];
    if (!j || !map || !directionsService) return;
    clearOverlay();
    document.querySelectorAll('.journey-card').forEach(c => c.classList.remove('active'));
    if (el) el.classList.add('active');

    // Refresh AI summary for the selected journey
    fetchAiSummary(idx);

    // Walking leg: user location → nearest bus stop on THIS specific route.
    // Prefer the per-route boarding coords returned by /api/plan (snapped from user GPS),
    // then the global boardingLL from useMyLocation, then the route's from-waypoint coords.
    const firstLeg = j.legs[0];
    const walkDest = (firstLeg.boarding_lat
        ? { lat: firstLeg.boarding_lat, lng: firstLeg.boarding_lng }
        : null)
      || boardingLL
      || (firstLeg.from_lat ? { lat: firstLeg.from_lat, lng: firstLeg.from_lon } : null)
      || (firstLeg.route_start_lat ? { lat: firstLeg.route_start_lat, lng: firstLeg.route_start_lon } : null);
    if (originLL && walkDest) {
      drawWalkingLeg(originLL, walkDest);
    }

    // Draw each matatu leg using its actual GPS shape + unique route color
    for (const leg of j.legs) {
      drawLeg(leg);
    }

    // Transfer point marker (2-hop journeys)
    if (j.hops > 1) {
      const l1 = j.legs[0];
      if (l1.to_lat) {
        const m = new google.maps.Marker({
          position: { lat: l1.to_lat, lng: l1.to_lon }, map,
          title: `Transfer: ${j.transfers[0]}`,
          zIndex: 50,
          icon: {
            path: google.maps.SymbolPath.CIRCLE,
            scale: 10,
            fillColor: '#e67e22', fillOpacity: 1,
            strokeColor: '#fff', strokeWeight: 2,
          },
        });
        overlayMarkers.push(m);
      }
    }
  }

  // Walking directions: dotted grey from real origin to boarding stop
  function drawWalkingLeg(from, to) {
    if (!from || !to || !directionsService) return;
    directionsService.route({
      origin:      from,
      destination: to,
      travelMode:  google.maps.TravelMode.WALKING,
    }, (result, status) => {
      if (status === 'OK') walkingRenderer.setDirections(result);
    });
  }

  // Draw one matatu leg by fetching its actual GPS shape from the server,
  // then rendering it as a Polyline in the route's unique color.
  // No Directions API is used — the route follows its real-world road corridor.
  async function drawLeg(leg) {
    const color  = leg.color || '#1a3c5e';
    const params = new URLSearchParams();
    if (leg.from) params.set('from_wp', leg.from);
    if (leg.to)   params.set('to_wp',   leg.to);

    let pts = [];
    try {
      const r = await fetch(`/api/route-shape/${encodeURIComponent(leg.route)}?${params}`);
      if (r.ok) pts = await r.json();
    } catch (_) {}

    if (pts.length >= 2) {
      // Draw the actual GPS corridor
      const poly = new google.maps.Polyline({
        path: pts,
        strokeColor:   color,
        strokeWeight:  5,
        strokeOpacity: 0.88,
        map: map,
        zIndex: 10,
      });
      legRenderers.push(poly);

      // Boarding pin (start of leg)
      const startPin = new google.maps.Marker({
        position: pts[0], map,
        title: `Board: ${leg.from_display || leg.from}`,
        icon: { path: google.maps.SymbolPath.CIRCLE, scale: 7,
                fillColor: color, fillOpacity: 1,
                strokeColor: '#fff', strokeWeight: 2 },
        zIndex: 20,
      });
      // Alighting pin (end of leg)
      const endPin = new google.maps.Marker({
        position: pts[pts.length - 1], map,
        title: `Alight: ${leg.to_display || leg.to}`,
        icon: { path: google.maps.SymbolPath.CIRCLE, scale: 7,
                fillColor: '#fff', fillOpacity: 1,
                strokeColor: color, strokeWeight: 2.5 },
        zIndex: 20,
      });
      overlayMarkers.push(startPin, endPin);
    } else {
      // Shape not available — fall back to a driving directions line
      const fallbackRenderer = new google.maps.DirectionsRenderer({
        suppressMarkers: false,
        polylineOptions: { strokeColor: color, strokeWeight: 5, strokeOpacity: 0.88 },
      });
      fallbackRenderer.setMap(map);
      legRenderers.push(fallbackRenderer);
      const originLat = leg.from_lat  || leg.route_start_lat;
      const originLon = leg.from_lon  || leg.route_start_lon;
      const destLat   = leg.to_lat    || leg.route_end_lat;
      const destLon   = leg.to_lon    || leg.route_end_lon;
      if (originLat && destLat && directionsService) {
        directionsService.route({
          origin:      { lat: originLat, lng: originLon },
          destination: { lat: destLat,   lng: destLon   },
          travelMode:  google.maps.TravelMode.DRIVING,
        }, (result, status) => {
          if (status === 'OK') fallbackRenderer.setDirections(result);
        });
      }
    }
  }

  function clearOverlay() {
    legRenderers.forEach(r => {
      if (typeof r.setMap === 'function') r.setMap(null);
      if (typeof r.setDirections === 'function') r.setDirections({ routes: [] });
    });
    legRenderers = [];
    if (walkingRenderer) walkingRenderer.setDirections({ routes: [] });
    overlayMarkers.forEach(m => m.setMap(null));
    overlayMarkers = [];
  }
</script>

{% if maps_api_key %}
<script src="https://maps.googleapis.com/maps/api/js?key={{ maps_api_key }}&libraries=places&callback=initMap" async defer></script>
{% endif %}
</body>
</html>
"""


# ── Flask routes ──────────────────────────────────────────────────────────────
@app.route("/api/routes")
def api_routes():
    seen, routes = set(), []
    for _, row in df.iterrows():
        rid = str(row["route_name"])
        if rid in seen:
            continue
        seen.add(rid)
        routes.append({
            "id":        rid,
            "name":      rid,
            "corridor":  str(row["route_long"]),
            "headsign":  str(row["headsign"]),
            "fare":      int(row["fare_KSh"]),
            "dist_km":   round(float(row["dist_km"]), 1),
            "start_lat": float(row["start_lat"]),
            "start_lon": float(row["start_lon"]),
            "end_lat":   float(row["end_lat"]),
            "end_lon":   float(row["end_lon"]),
            "saccos":    str(row["saccos"]) if pd.notna(row["saccos"]) else "",
        })
    return jsonify(routes)


@app.route("/api/snap")
def api_snap():
    """
    Resolve a real-world lat/lng (+ optional place-name hint) to the nearest
    router-resolvable waypoint, preferring road names over stop names.

    Resolution order:
      1. Hint text parts (e.g. "Karen" extracted from a full Places address)
      2. Routes whose GPS polyline passes within 600 m — return their road waypoint
         ordered from the end of the route closest to the user
      3. Nearby named stops that appear in the router graph (fallback)
    """
    try:
        lat  = float(request.args.get("lat", 0))
        lng  = float(request.args.get("lng", 0))
        hint = request.args.get("hint", "").strip()
    except ValueError:
        return jsonify(error="Invalid coordinates"), 400

    _road_set = {r.lower() for r in ROAD_ROUTES}

    def _best_wp(waypoints: list[str]) -> str | None:
        """Pick the best router waypoint: prefer named roads, fall back to any."""
        fallback = None
        for wp in waypoints:
            resolved = router._resolve(wp)
            if not resolved or resolved[0] == "__cbd__":
                continue
            rw = resolved[0].lower()
            if rw in _road_set:
                return resolved[0]       # named road — take it immediately
            if fallback is None:
                fallback = resolved[0]
        return fallback

    # 1. Try hint text parts — but always enrich with a boarding point from polyline
    if hint:
        for candidate in [hint] + [p.strip() for p in hint.replace("|", ",").split(",")]:
            if not candidate:
                continue
            resolved = router._resolve(candidate)
            if resolved:
                # Find the nearest passing route to get a specific boarding landmark
                near_for_bp = find_nearest_routes_by_shape(lat, lng, n=3, max_dist_m=1200)
                bp = {}
                for r in near_for_bp:
                    bp = find_boarding_point(lat, lng, r["route_name"])
                    if bp.get("boarding_name"):
                        break
                return jsonify(
                    waypoint=resolved[0],
                    display_name=router._display_wp(resolved[0]),
                    source="hint",
                    distance_m=bp.get("boarding_dist_m", 0),
                    boarding_lat=bp.get("boarding_lat"),
                    boarding_lng=bp.get("boarding_lng"),
                    boarding_name=bp.get("boarding_name"),
                    boarding_dist_m=bp.get("boarding_dist_m", 0),
                )

    # 2. Routes whose actual GPS polyline passes near the user
    near = find_nearest_routes_by_shape(lat, lng, n=8, max_dist_m=1000)
    for r in near:
        rid     = r["route_name"]
        all_wps = router.route_wps.get(rid, [])
        if not all_wps:
            continue
        # Order waypoints from the end of the route nearest to the user's position
        # by projecting the user onto the polyline and taking the closest segment index
        pts = ROUTE_SHAPES.get(rid, [])
        if pts and len(pts) >= 2:
            min_idx, min_d = 0, float("inf")
            for i, (plon, plat) in enumerate(pts):
                d = _haversine(lat, lng, plat, plon) * 1000
                if d < min_d:
                    min_d, min_idx = d, i
            # If closest point is in the second half, reverse waypoint order
            frac = min_idx / max(len(pts) - 1, 1)
            ordered_wps = all_wps if frac <= 0.5 else list(reversed(all_wps))
        else:
            hits = df[df["route_name"].astype(str) == rid]
            if hits.empty:
                continue
            row = hits.iloc[0]
            d_s = _haversine(lat, lng, float(row["start_lat"]), float(row["start_lon"]))
            d_e = _haversine(lat, lng, float(row["end_lat"]),   float(row["end_lon"]))
            ordered_wps = all_wps if d_s <= d_e else list(reversed(all_wps))

        # Prefer road waypoints from this route, then any waypoint
        road_wps = [w for w in ordered_wps if w in _road_set]
        best = _best_wp(road_wps or ordered_wps[:6])
        if best:
            is_road = best.lower() in _road_set
            bp      = find_boarding_point(lat, lng, rid)
            return jsonify(
                waypoint=best,
                display_name=router._display_wp(best),
                source="road" if is_road else "corridor",
                distance_m=int(r["_d_m"]),
                boarding_lat=bp.get("boarding_lat"),
                boarding_lng=bp.get("boarding_lng"),
                boarding_name=bp.get("boarding_name"),
                boarding_dist_m=bp.get("boarding_dist_m"),
            )

    # 3. Nearby stops — walk their routes' waypoint lists to find a proper road waypoint.
    #    Never return a stop name: stop names are in wp_routes but NOT in route_wps,
    #    so the router's _actual_wp check would reject them and produce no journeys.
    _road_set_local = {r.lower() for r in ROAD_ROUTES}
    nearby_stops = find_nearest_stops(lat, lng, n=8)
    for stop in nearby_stops:
        routes_str = str(stop.get("routes", ""))
        if not routes_str or routes_str == "nan":
            continue
        for rid in routes_str.split():
            wps = router.route_wps.get(rid, [])
            if not wps:
                continue
            # Prefer a named road waypoint; otherwise take any resolved waypoint
            road_wps = [w for w in wps if w in _road_set_local]
            best = road_wps[0] if road_wps else wps[0]
            resolved = router._resolve(best)
            if resolved:
                return jsonify(
                    waypoint=resolved[0],
                    display_name=router._display_wp(resolved[0]),
                    source="stop_road",
                    distance_m=int(stop["_d"] * 1000),
                )

    return jsonify(error="No matatu route passes near this location"), 404


@app.route("/api/nearest-stop")
def api_nearest_stop():
    try:
        lat = float(request.args["lat"])
        lng = float(request.args["lng"])
    except (KeyError, ValueError):
        return jsonify(error="Provide lat and lng"), 400
    nearby = find_nearest_stops(lat, lng, n=3)
    return jsonify([{
        "name":       s["stop_name"],
        "lat":        s["stop_lat"],
        "lon":        s["stop_lon"],
        "distance_m": int(s["_d"] * 1000),
        "routes":     str(s["routes"]).split() if str(s["routes"]) != "nan" else [],
    } for s in nearby])


@app.route("/api/route-shape/<route_name>")
def api_route_shape(route_name: str):
    """
    Return a road-following GPS polyline for a route segment.

    When board_lat/lng and alight_lat/lng are provided:
      1. Try Google Directions API (driving) for an accurate road polyline.
      2. Fall back to shapes.shp segment crop.

    Without coordinates, return the full shapes.shp polyline.
    """
    board_lat  = request.args.get("board_lat",  type=float)
    board_lng  = request.args.get("board_lng",  type=float)
    alight_lat = request.args.get("alight_lat", type=float)
    alight_lng = request.args.get("alight_lng", type=float)

    # ── Primary: Google Directions when we have exact coords ──────────────────
    if board_lat and board_lng and alight_lat and alight_lng:
        google_pts = get_road_polyline(board_lat, board_lng, alight_lat, alight_lng)
        if len(google_pts) >= 3:
            return jsonify(google_pts)

    # ── Fallback: shapes.shp segment ─────────────────────────────────────────
    pts = ROUTE_SHAPES.get(route_name)
    if not pts:
        return jsonify(error="Shape not found"), 404

    def _closest_idx_ll(lat: float, lon: float) -> int:
        best_i, best_d = 0, float("inf")
        for i, (plon, plat) in enumerate(pts):
            d = (plat - lat) ** 2 + (plon - lon) ** 2
            if d < best_d:
                best_d, best_i = d, i
        return best_i

    start_i, end_i = 0, len(pts) - 1
    if board_lat is not None and board_lng is not None:
        start_i = _closest_idx_ll(board_lat, board_lng)
    if alight_lat is not None and alight_lng is not None:
        end_i = _closest_idx_ll(alight_lat, alight_lng)

    if start_i > end_i:
        start_i, end_i = end_i, start_i

    segment = pts[start_i: end_i + 1]
    return jsonify([{"lat": p[1], "lng": p[0]} for p in segment])



@app.route("/api/plan")
def api_plan():
    origin_raw = request.args.get("from", "").strip()
    dest_raw   = request.args.get("to",   "").strip()
    user_lat   = request.args.get("lat",  type=float)
    user_lng   = request.args.get("lng",  type=float)
    if not origin_raw or not dest_raw:
        return jsonify(error="Provide 'from' and 'to' params"), 400

    # Pre-resolve via Google if the local graph doesn't recognise the place
    def _pre_resolve(raw: str) -> str:
        if router._resolve(raw):
            return raw
        wps = google_resolve_waypoints(raw)
        return wps[0] if wps else raw

    origin_resolved = _pre_resolve(origin_raw)
    dest_resolved   = _pre_resolve(dest_raw)

    journeys = router.find_journeys(origin_resolved, dest_resolved)
    if not journeys:
        return jsonify([])

    def stop_coords(wp: str):
        clean = wp.strip("_").replace("_", " ").strip()
        hits  = stops_df[stops_df["stop_name"].str.lower() == clean.lower()]
        if hits.empty:
            hits = stops_df[stops_df["stop_name"].str.lower().str.contains(clean.lower(), na=False)]
        if not hits.empty:
            r = hits.iloc[0]
            return float(r["stop_lat"]), float(r["stop_lon"])
        return None, None

    def route_meta(rid: str):
        hits = df[df["route_name"].astype(str) == rid]
        if hits.empty:
            return {}
        r = hits.iloc[0]
        return {
            "route_start_lat": float(r["start_lat"]),
            "route_start_lon": float(r["start_lon"]),
            "route_end_lat":   float(r["end_lat"]),
            "route_end_lon":   float(r["end_lon"]),
            "corridor":        str(r["route_long"]),
            "saccos":          str(r["saccos"]) if pd.notna(r["saccos"]) else "",
            "color":           ROUTE_COLORS.get(rid, "#1a3c5e"),
            "roads":           ROUTE_ROADS.get(rid, []),
        }

    result = []
    for j in journeys:
        enriched_legs = []
        for leg in j["legs"]:
            from_lat, from_lon = stop_coords(leg["from"])
            to_lat,   to_lon   = stop_coords(leg["to"])
            enriched_legs.append({
                "route":        leg["route"],
                "from":         leg["from"],
                "to":           leg["to"],
                "from_display": router._display_wp(leg["from"]),
                "to_display":   router._display_wp(leg["to"]),
                "fare":         leg["fare"],
                "windscreen":   leg["windscreen"],
                "from_lat":     from_lat,
                "from_lon":     from_lon,
                "to_lat":       to_lat,
                "to_lon":       to_lon,
                **route_meta(leg["route"]),
            })
        result.append({
            "hops":       j["hops"],
            "total_fare": j["total_fare"],
            "transfers":  [router._display_wp(t) for t in j["transfers"]],
            "legs":       enriched_legs,
        })

    # Add nearest boarding stop for the first leg of every journey
    for j_item in result:
        first_leg = j_item["legs"][0]
        bp_lat = user_lat if user_lat is not None else first_leg.get("route_start_lat")
        bp_lng = user_lng if user_lng is not None else first_leg.get("route_start_lon")
        if bp_lat is not None:
            bp = find_boarding_point(bp_lat, bp_lng, first_leg["route"])
            first_leg["boarding_name"]   = bp.get("boarding_name")
            first_leg["boarding_dist_m"] = bp.get("boarding_dist_m")
            first_leg["boarding_lat"]    = bp.get("boarding_lat")
            first_leg["boarding_lng"]    = bp.get("boarding_lng")

    return jsonify(result)


@app.route("/api/ai-summary", methods=["POST"])
def api_ai_summary():
    """
    Generate a short AI directions summary for a specific journey.
    Accepts {from, to, journey} and returns {summary}.
    """
    body      = request.get_json(silent=True) or {}
    from_text = body.get("from", "").strip()
    to_text   = body.get("to",   "").strip()
    journey   = body.get("journey", {})

    legs = journey.get("legs", [])
    if not legs:
        return jsonify(summary="No route data available."), 400

    leg1     = legs[0]
    fare     = journey.get("total_fare", 0)
    boarding = leg1.get("boarding_name") or leg1.get("from_display", "")
    dist_m   = leg1.get("boarding_dist_m")
    corridor = leg1.get("corridor") or " / ".join(leg1.get("roads") or [])
    ws       = leg1.get("windscreen", "")
    xfers    = journey.get("transfers", [])

    parts = [f"Journey: {from_text} → {to_text}."]
    if corridor:
        parts.append(f"Route corridor: {corridor}.")
    if boarding and dist_m is not None:
        parts.append(f"Nearest boarding stop: {boarding} (~{dist_m}m walk from the user's location).")
    elif boarding:
        parts.append(f"Board at or near: {boarding}.")
    parts.append(f"Board any mat showing '{ws}'. Total fare: {fare} KSh.")
    if len(legs) > 1:
        leg2 = legs[1]
        xfer = xfers[0] if xfers else "the junction"
        parts.append(
            f"Transfer at {xfer}, then board mat showing '{leg2.get('windscreen', '')}' "
            f"(+{leg2.get('fare', 0)} KSh)."
        )

    prompt = (
        " ".join(parts) +
        "\n\nWrite a 2-3 sentence casual, street-smart Nairobi directions summary. "
        "Name the bus stop or road to walk to, which mat windscreen to look for, and the fare. "
        "Speak directly to the person — no bullet points, just natural speech."
    )

    try:
        summary_sid = f"summary_{uuid.uuid4().hex[:8]}"
        result = orchestrator.process(summary_sid, prompt, intent="simple_transit")
        return jsonify(summary=result["reply"])
    except Exception:
        return jsonify(summary="Tap the chat tab and ask me for directions!"), 200


@app.route("/api/transit-snap")
def api_transit_snap():
    """
    Stop-based routing: find stops near user + near destination, intersect routes.
    Direct  = route serves a stop near user AND a stop near destination.
    Transfer = two routes share a stop in the database; leg-1 serves user, leg-2 serves dest.
    """
    user_lat  = request.args.get("user_lat",  type=float)
    user_lng  = request.args.get("user_lng",  type=float)
    dest_lat  = request.args.get("dest_lat",  type=float)
    dest_lng  = request.args.get("dest_lng",  type=float)

    if None in (user_lat, user_lng, dest_lat, dest_lng):
        return jsonify(error="Provide user_lat, user_lng, dest_lat, dest_lng"), 400

    try:
        import math as _math

        AVG_BUS_KMH = 25.0
        WALK_KMH    = 4.5
        STAGE_R     = 1200  # metres — search radius around user/dest for a stage
        XFER_R      = 800   # metres — how close transfer terminals must be to each other

        def hav(lat1, lon1, lat2, lon2):
            R = 6_371_000
            p1, p2 = _math.radians(lat1), _math.radians(lat2)
            dp, dl = _math.radians(lat2 - lat1), _math.radians(lon2 - lon1)
            a = _math.sin(dp/2)**2 + _math.cos(p1)*_math.cos(p2)*_math.sin(dl/2)**2
            return 2 * R * _math.asin(_math.sqrt(a))

        # ── Precompute distances for every route terminal row ─────────────────
        rows = routes_se_df.to_dict("records")
        for r in rows:
            r["d_start_user"] = hav(user_lat, user_lng, r["start_lat"], r["start_lon"])
            r["d_end_dest"]   = hav(dest_lat, dest_lng, r["end_lat"],   r["end_lon"])

        # ── Routes whose START terminal is near the user ─────────────────────
        user_rows = [r for r in rows if r["d_start_user"] <= STAGE_R]
        if not user_rows:
            user_rows = [r for r in rows if r["d_start_user"] <= STAGE_R * 2]
        if not user_rows:
            user_rows = sorted(rows, key=lambda r: r["d_start_user"])[:5]

        # ── Routes whose END terminal is near the destination ─────────────────
        dest_rows = [r for r in rows if r["d_end_dest"] <= STAGE_R]
        if not dest_rows:
            dest_rows = [r for r in rows if r["d_end_dest"] <= STAGE_R * 2]
        if not dest_rows:
            dest_rows = sorted(rows, key=lambda r: r["d_end_dest"])[:5]

        dest_route_set = {r["route_name"] for r in dest_rows}

        # ── Direct options: route in both sets ────────────────────────────────
        options = []
        seen_direct: set = set()
        for ur in user_rows:
            rname = ur["route_name"]
            if rname not in dest_route_set or rname in seen_direct:
                continue
            seen_direct.add(rname)
            # Matching dest row for this route (same route, direction reaching dest)
            dr = next(r for r in dest_rows if r["route_name"] == rname)
            bus_d   = hav(ur["start_lat"], ur["start_lon"], dr["end_lat"], dr["end_lon"])
            walk1   = ur["d_start_user"] / (WALK_KMH * 1000 / 60)
            walk2   = dr["d_end_dest"]   / (WALK_KMH * 1000 / 60)
            total_t = walk1 + bus_d / (AVG_BUS_KMH * 1000 / 60) + walk2
            meta    = df[df["route_name"].astype(str) == rname]
            fare    = float(meta["fare_KSh"].iloc[0]) if not meta.empty else 50.0
            board_stage  = _BOARD_STAGE.get((rname, int(ur["direction"])), ur["headsign"])
            alight_stage = str(dr["headsign"])
            options.append({
                "type":           "direct",
                "route":          rname,
                "fare_ksh":       int(fare),
                "board":  {"lat": ur["start_lat"], "lon": ur["start_lon"], "name": board_stage  + " Stage"},
                "alight": {"lat": dr["end_lat"],   "lon": dr["end_lon"],   "name": alight_stage + " Stage"},
                "bus_dist_m":     round(bus_d),
                "total_time_min": round(total_t, 1),
                "color":          ROUTE_COLORS.get(rname, "#27ae60"),
            })
        options.sort(key=lambda o: o["total_time_min"])

        # ── Transfer options ──────────────────────────────────────────────────
        # A transfer terminal is a point where leg-1 ends (close to where leg-2 starts)
        transfer_options: list[dict] = []
        seen_pairs: set = set()

        for ur in user_rows:
            rname1 = ur["route_name"]
            # leg-1 ends at this row's end terminal
            xfer_lat, xfer_lon = ur["end_lat"], ur["end_lon"]
            xfer_label = ur["headsign"] + " Stage"

            for dr in dest_rows:
                rname2 = dr["route_name"]
                if rname2 == rname1:
                    continue
                pair_key = (min(rname1, rname2), max(rname1, rname2))
                if pair_key in seen_pairs:
                    continue

                # leg-2 must START near where leg-1 ends
                d_xfer = hav(xfer_lat, xfer_lon, dr["start_lat"], dr["start_lon"])
                if d_xfer > XFER_R:
                    continue

                seen_pairs.add(pair_key)
                leg1_d  = hav(ur["start_lat"], ur["start_lon"], xfer_lat, xfer_lon)
                leg2_d  = hav(dr["start_lat"], dr["start_lon"], dr["end_lat"], dr["end_lon"])
                walk1   = ur["d_start_user"] / (WALK_KMH * 1000 / 60)
                walk2   = dr["d_end_dest"]   / (WALK_KMH * 1000 / 60)
                total_t = walk1 + leg1_d / (AVG_BUS_KMH * 1000 / 60) + 5.0 + leg2_d / (AVG_BUS_KMH * 1000 / 60) + walk2

                meta1 = df[df["route_name"].astype(str) == rname1]
                fare1 = float(meta1["fare_KSh"].iloc[0]) if not meta1.empty else 50.0
                meta2 = df[df["route_name"].astype(str) == rname2]
                fare2 = float(meta2["fare_KSh"].iloc[0]) if not meta2.empty else 50.0

                board_stage1 = _BOARD_STAGE.get((rname1, int(ur["direction"])), ur["headsign"])
                alight_stage2 = str(dr["headsign"])
                transfer_options.append({
                    "type":           "transfer",
                    "route1":         rname1,
                    "route2":         rname2,
                    "fare_ksh":       int(fare1 + fare2),
                    "board":         {"lat": ur["start_lat"], "lon": ur["start_lon"], "name": board_stage1 + " Stage"},
                    "transfer_stop": {"lat": xfer_lat,        "lon": xfer_lon,        "name": str(ur["headsign"]) + " Stage"},
                    "alight":        {"lat": dr["end_lat"],   "lon": dr["end_lon"],   "name": alight_stage2 + " Stage"},
                    "total_time_min": round(total_t, 1),
                    "color":          ROUTE_COLORS.get(rname1, "#2980b9"),
                    "color2":         ROUTE_COLORS.get(rname2, "#8e44ad"),
                })

        transfer_options.sort(key=lambda o: o["total_time_min"])
        all_opts = sorted(options[:5] + transfer_options[:4], key=lambda o: o["total_time_min"])
        winner   = all_opts[0] if all_opts else None

        # Nearest stages to user for display in sidebar
        nearest = sorted(rows, key=lambda r: r["d_start_user"])[:3]
        return jsonify({
            "winner":  winner,
            "options": all_opts[:6],
            "nearest_stops": [
                {"name": r["headsign"] + " Stage", "lat": r["start_lat"],
                 "lon": r["start_lon"], "dist_m": int(r["d_start_user"])}
                for r in nearest
            ],
        })
    except Exception as exc:
        import traceback
        return jsonify(error=str(exc), trace=traceback.format_exc()), 500


@app.route("/api/google-transit")
def api_google_transit():
    """
    Primary router: send origin + destination coords to Google Directions
    (transit/bus mode) and parse the result into structured route legs.

    Google detects intermediate transfer nodes (e.g. Bomas junction where
    Route 24 and Route 125 physically cross) that our terminal-only CSV misses.
    Fares are cross-referenced with local nairobi_transit_data.csv.

    Returns: { routes: [ { legs, total_fare, duration, transfers }, ... ] }
    """
    s_lat = request.args.get("s_lat", type=float)
    s_lng = request.args.get("s_lng", type=float)
    e_lat = request.args.get("e_lat", type=float)
    e_lng = request.args.get("e_lng", type=float)

    if None in (s_lat, s_lng, e_lat, e_lng):
        return jsonify(error="Provide s_lat, s_lng, e_lat, e_lng"), 400
    if not MAPS_API_KEY:
        return jsonify(error="MAPS_API_KEY not set", routes=[])

    try:
        resp = requests.get(
            "https://maps.googleapis.com/maps/api/directions/json",
            params={
                "origin":                      f"{s_lat},{s_lng}",
                "destination":                 f"{e_lat},{e_lng}",
                "mode":                        "transit",
                "transit_mode":                "bus",
                "alternatives":                "true",
                "key":                         MAPS_API_KEY,
                "region":                      "ke",
            },
            timeout=10,
        )
        data = resp.json()

        if data.get("status") != "OK":
            return jsonify(routes=[], status=data.get("status"))

        routes_out = []
        for route in data.get("routes", [])[:3]:
            leg = route["legs"][0]
            transit_legs = []
            steps_out    = []   # all steps in order: WALKING + TRANSIT

            for step in leg.get("steps", []):
                mode = step.get("travel_mode", "")

                if mode == "WALKING":
                    steps_out.append({
                        "mode":     "WALKING",
                        "polyline": step["polyline"]["points"],
                        "distance": step["distance"]["text"],
                        "duration": step["duration"]["text"],
                    })
                    continue

                if mode != "TRANSIT":
                    continue

                td   = step["transit_details"]
                line = td["line"]
                rnum = line.get("short_name") or line.get("name", "")

                fare = 50
                hits = df[df["route_name"].astype(str) == rnum]
                if not hits.empty:
                    fare = int(hits["fare_KSh"].iloc[0])
                else:
                    hits2 = df[df["route_name"].astype(str).str.startswith(rnum[:3])]
                    if not hits2.empty:
                        fare = int(hits2["fare_KSh"].iloc[0])

                dep   = td["departure_stop"]
                arr   = td["arrival_stop"]
                color = ROUTE_COLORS.get(rnum, "#2980b9")

                transit_step = {
                    "mode":     "TRANSIT",
                    "route":    rnum,
                    "headsign": td.get("headsign", ""),
                    "fare":     fare,
                    "color":    color,
                    "polyline": step["polyline"]["points"],
                    "board": {
                        "name": dep["name"],
                        "lat":  dep["location"]["lat"],
                        "lon":  dep["location"]["lng"],
                    },
                    "alight": {
                        "name": arr["name"],
                        "lat":  arr["location"]["lat"],
                        "lon":  arr["location"]["lng"],
                    },
                    "num_stops": td.get("num_stops", 0),
                }
                transit_legs.append(transit_step)
                steps_out.append(transit_step)

            if not transit_legs:
                continue

            total_fare   = sum(l["fare"] for l in transit_legs)
            duration_sec = leg.get("duration", {}).get("value", 0)
            routes_out.append({
                "type":         "google",
                "duration":     leg["duration"]["text"],
                "duration_sec": duration_sec,
                "distance":     leg["distance"]["text"],
                "legs":         transit_legs,
                "steps":        steps_out,
                "total_fare":   total_fare,
                "transfers":    len(transit_legs) - 1,
            })

        # Rank by fastest first
        routes_out.sort(key=lambda r: r["duration_sec"])
        return jsonify(routes=routes_out)

    except Exception as exc:
        import traceback
        return jsonify(error=str(exc), trace=traceback.format_exc()), 500


@app.route("/api/terminals")
def api_terminals():
    """
    Return every route's start and end terminal as map pins.
    Each terminal carries its stage name, coordinates, and route color.
    Only direction=0 rows are used to avoid duplicate pins.
    """
    out = []
    seen: set = set()
    for _, row in routes_se_df[routes_se_df["direction"] == 0].iterrows():
        rname  = str(row["route_name"])
        color  = ROUTE_COLORS.get(rname, "#27ae60")
        # Start terminal
        s_name = _BOARD_STAGE.get((rname, 0), str(row["route_long"]).split("-")[0].strip())
        s_key  = (round(float(row["start_lat"]), 4), round(float(row["start_lon"]), 4))
        if s_key not in seen:
            seen.add(s_key)
            out.append({"lat": float(row["start_lat"]), "lng": float(row["start_lon"]),
                        "name": s_name, "route": rname, "color": color, "kind": "start"})
        # End terminal
        e_name = str(row["headsign"])
        e_key  = (round(float(row["end_lat"]), 4), round(float(row["end_lon"]), 4))
        if e_key not in seen:
            seen.add(e_key)
            out.append({"lat": float(row["end_lat"]), "lng": float(row["end_lon"]),
                        "name": e_name, "route": rname, "color": color, "kind": "end"})
    return jsonify(out)


@app.route("/api/route-map")
def api_route_map():
    lat  = request.args.get("lat",  type=float)
    lng  = request.args.get("lng",  type=float)
    dest = request.args.get("to",   "").strip()
    if not lat or not lng or not dest:
        return "<p>Provide lat, lng, and to params</p>", 400
    try:
        from routing_engine import NairobiRoutingEngine, FoliumMapGenerator, get_walk_graph
        engine   = NairobiRoutingEngine(df, stops_df, ROUTE_SHAPES)
        options  = engine.get_route_options((lat, lng), dest)
        dest_ll  = engine.geocode(dest)
        walk_g   = get_walk_graph()
        fmap     = FoliumMapGenerator(walk_g).generate((lat, lng), dest_ll, dest, options)
        html     = fmap._repr_html_()
        return html, 200, {"Content-Type": "text/html"}
    except Exception as exc:
        return f"<p>Route map error: {exc}</p>", 500


@app.route("/api/stops")
def api_stops():
    stops = []
    for _, row in stops_df.iterrows():
        routes_val = str(row.get("routes", ""))
        if not routes_val or routes_val == "nan":
            continue
        stops.append({
            "id":     str(row["stop_id"]),
            "name":   str(row["stop_name"]),
            "lat":    float(row["stop_lat"]),
            "lon":    float(row["stop_lon"]),
            "routes": routes_val.split(),
        })
    return jsonify(stops)


@app.route("/constituency_stops.json")
def constituency_stops():
    with open(os.path.join(BASE_DIR, "constituency_stops.json"), encoding="utf-8") as f:
        return f.read(), 200, {"Content-Type": "application/json"}


@app.route("/nairobi_constituencies.geojson")
def nairobi_constituencies():
    with open(os.path.join(BASE_DIR, "nairobi_constituencies.geojson"), encoding="utf-8") as f:
        return f.read(), 200, {"Content-Type": "application/json"}


@app.route("/bus_routes.geojson")
def bus_routes_geojson():
    with open(os.path.join(BASE_DIR, "bus_routes.geojson"), encoding="utf-8") as f:
        return f.read(), 200, {"Content-Type": "application/json"}


@app.route("/api/nearby-stops")
def api_nearby_stops():
    lat = request.args.get("lat", type=float)
    lng = request.args.get("lng", type=float)
    n   = request.args.get("n", 8, type=int)
    if lat is None or lng is None:
        return jsonify(error="lat/lng required"), 400

    stops = find_nearest_stops(lat, lng, n=n)

    # Determine constituency via shapely point-in-polygon
    constituency = None
    try:
        from shapely.geometry import Point as _Pt, shape as _shape
        import json as _j
        with open(os.path.join(BASE_DIR, "nairobi_constituencies.geojson"), encoding="utf-8") as _f:
            const_data = _j.load(_f)
        pt = _Pt(lng, lat)
        for feat in const_data["features"]:
            if _shape(feat["geometry"]).contains(pt):
                constituency = feat["properties"]["constituency"]
                break
    except Exception:
        pass

    return jsonify({
        "constituency": constituency,
        "stops": [
            {
                "stop_id":   s["stop_id"],
                "stop_name": s["stop_name"],
                "lat":       float(s["stop_lat"]),
                "lng":       float(s["stop_lon"]),
                "dist_m":    int(s["_d"] * 1000),
                "routes":    [r for r in str(s.get("routes", "") or "").split() if r],
            }
            for s in stops
        ],
    })


@app.route("/api/rebuild-cache", methods=["POST"])
def rebuild_cache():
    """Re-run build_transfer_cache.py and hot-reload the cache into the router module."""
    import subprocess, router as _router_mod, json as _json
    result = subprocess.run(
        ["python", os.path.join(BASE_DIR, "build_transfer_cache.py")],
        capture_output=True, text=True, cwd=BASE_DIR,
    )
    if result.returncode != 0:
        return jsonify(error=result.stderr), 500
    with open(os.path.join(BASE_DIR, "transfer_cache.json"), encoding="utf-8") as f:
        _router_mod._TRANSFER_CACHE.update(_json.load(f))
    return jsonify(ok=True, output=result.stdout)


@app.route("/exported_style.json")
def map_style():
    with open(os.path.join(BASE_DIR, "exported_style.json"), encoding="utf-8") as f:
        return f.read(), 200, {"Content-Type": "application/json"}


@app.route("/manifest.json")
def pwa_manifest():
    import json as _json
    data = {
        "name": "Shii Ngapi",
        "short_name": "ShiiNgapi",
        "description": "Nairobi bus fare & route finder",
        "start_url": "/",
        "scope": "/",
        "display": "standalone",
        "orientation": "portrait",
        "background_color": "#0a140c",
        "theme_color": "#3ecf6e",
        "icons": [
            {"src": "/static/icon-192.png", "sizes": "192x192", "type": "image/png", "purpose": "any maskable"},
            {"src": "/static/icon-512.png", "sizes": "512x512", "type": "image/png", "purpose": "any maskable"}
        ]
    }
    return app.response_class(_json.dumps(data), mimetype="application/manifest+json")


@app.route("/sw.js")
def service_worker():
    js = """\
const CACHE = 'shii-ngapi-v2';
const SHELL = ['/', '/static/icon-192.png', '/static/icon-512.png', '/manifest.json'];

self.addEventListener('install', e => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))
    ).then(() => clients.claim())
  );
});

self.addEventListener('fetch', e => {
  if (e.request.method !== 'GET') return;
  // Network-first for API calls, cache-first for static assets
  const isApi = e.request.url.includes('/api/') || e.request.url.includes('/route') || e.request.url.includes('/snap');
  if (isApi) {
    e.respondWith(fetch(e.request).catch(() => caches.match(e.request)));
  } else {
    e.respondWith(
      caches.match(e.request).then(cached => cached || fetch(e.request).then(res => {
        const clone = res.clone();
        caches.open(CACHE).then(c => c.put(e.request, clone));
        return res;
      }))
    );
  }
});
"""
    return js, 200, {
        "Content-Type": "application/javascript",
        "Service-Worker-Allowed": "/"
    }


@app.route("/transit-map")
@app.route("/")
def transit_map():
    with open(os.path.join(BASE_DIR, "transit_map.html"), encoding="utf-8") as f:
        html = f.read()
    html = html.replace("{{maps_api_key}}", MAPS_API_KEY)
    return html, 200, {"Content-Type": "text/html"}


@app.route("/phone")
def phone():
    with open(os.path.join(BASE_DIR, "phone.html"), encoding="utf-8") as f:
        html = f.read()
    html = html.replace("{{maps_api_key}}", MAPS_API_KEY)
    return html, 200, {"Content-Type": "text/html"}


@app.route("/chat", methods=["POST"])
def chat():
    body         = request.get_json(silent=True) or {}
    message      = (body.get("message") or "").strip()
    session_id   = body.get("session_id") or str(uuid.uuid4())
    lat          = body.get("lat")
    lng          = body.get("lng")
    intent       = body.get("intent", "simple_transit")   # pre-classified by browser
    pii_detected = body.get("pii_detected", False)        # flagged by browser

    if not message:
        return jsonify(error="Empty message"), 400

    # ── Sheng mode detection ──────────────────────────────────────────────────
    if message.lower().strip() in _SHENG_TRIGGER:
        _SHENG_SESSIONS.add(session_id)
        return jsonify(
            response="Semaje!",
            session_id=session_id,
            model="sheng-mode",
            agent="sheng",
            intent=intent,
            pii_detected=pii_detected,
        )

    # ── Multi-hop routing context ─────────────────────────────────────────────
    enriched = message
    origin, dest = enhanced_extract_od(message)
    if origin and dest:
        try:
            journeys = router.find_journeys(origin, dest)
            if journeys:
                route_ctx = router.format_context(origin, dest, journeys)
                enriched  = route_ctx + "\n\nUSER MESSAGE: " + message
        except Exception:
            pass

        # ── Google real-time transit context ──────────────────────────────────
        try:
            g = get_google_transit_route(origin, dest)
            if g:
                g_ctx = (
                    f"\n━━ GOOGLE REAL-TIME ROUTE ━━\n"
                    f"Path: {origin} → {dest}\n"
                    f"Summary: {g['summary']}\n"
                    "Steps:\n" + "\n".join(f"- {s}" for s in g["steps"])
                )
                enriched = g_ctx + "\n\n" + enriched
        except Exception:
            pass

    # ── Location context (GPS) ────────────────────────────────────────────────
    if lat is not None and lng is not None:
        try:
            loc_ctx  = build_location_context(float(lat), float(lng))
            enriched = loc_ctx + "\n\n" + enriched
        except Exception:
            pass

    if session_id in _SHENG_SESSIONS:
        enriched = enriched + _SHENG_INSTRUCTION

    map_origin = f"{origin}, Nairobi, Kenya" if origin else None
    map_dest   = f"{dest}, Nairobi, Kenya"   if dest   else None

    try:
        result = orchestrator.process(session_id, enriched, intent=intent)
        return jsonify(
            response=result["reply"],
            session_id=session_id,
            model=result["model"],
            agent=result["agent"],
            intent=intent,
            pii_detected=pii_detected,
            map_origin=map_origin,
            map_dest=map_dest,
        )
    except Exception as exc:
        return jsonify(error=str(exc)), 503


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n  Shii Ngapi — Nairobi Transit Assistant")
    print(f"  Routes : {len(df)} | Stops: {len(stops_df):,}")
    print(f"  Model  : gemini-2.0-flash-lite (free — 1500/day, 30 RPM)")
    print(f"  API key: {'set' if GEMINI_API_KEY else 'NOT SET — add GEMINI_API_KEY to .env'}")
    print("  Open   : http://localhost:5000\n")
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
