"""
Nairobi Transit Routing Engine
================================
Compares Direct vs Transfer routes by price, time, and walking distance.
Generates interactive Folium maps with:
  - Solid green  = direct matatu route
  - Solid blue   = transfer leg 1
  - Solid purple = transfer leg 2
  - Dashed orange = walking segments (follows OSMnx walking network)
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import networkx as nx
import folium
from geopy.distance import geodesic
from geopy.geocoders import Nominatim

try:
    import osmnx as ox
    _HAS_OSMNX = True
except ImportError:
    _HAS_OSMNX = False
    warnings.warn("osmnx not installed — walking paths will use straight lines", stacklevel=2)

warnings.filterwarnings("ignore")

# ── Journey constants ─────────────────────────────────────────────────────────
AVG_MATATU_KMH    = 25.0   # average in-city matatu speed
WALK_KMH          = 4.5    # pedestrian walking speed
TRANSFER_PENALTY  = 10.0   # extra minutes assumed at a transfer point
MAX_WALK_M        = 800    # ignore stops further than this from origin/dest

# ── Scoring weights  (lower total score = better option) ─────────────────────
W_PRICE    = 0.40
W_TIME     = 0.40
W_TRANSFER = 0.20

# ── Map colours ───────────────────────────────────────────────────────────────
C_DIRECT   = "#27ae60"   # green  — direct route
C_LEG1     = "#2980b9"   # blue   — transfer leg 1
C_LEG2     = "#8e44ad"   # purple — transfer leg 2
C_WALK     = "#e67e22"   # orange — walking segments
C_XFER     = "#e67e22"   # orange — transfer stop marker


# ══════════════════════════════════════════════════════════════════════════════
# Data classes
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Stop:
    stop_id:   str
    name:      str
    lat:       float
    lon:       float
    stop_type: str = "roadside"          # "terminal" | "roadside"
    routes:    list = field(default_factory=list)


@dataclass
class RouteLeg:
    route_id:   str
    route_long: str
    from_stop:  Stop
    to_stop:    Stop
    fare:       int
    dist_km:    float
    time_min:   float
    shape_pts:  list   # [(lat, lon), …]  from shapes.shp


@dataclass
class RouteOption:
    option_id:       str
    option_type:     str    # "direct" | "transfer"
    legs:            list   # list[RouteLeg]
    total_fare:      int
    total_time_min:  float
    walk_origin_m:   float
    walk_dest_m:     float
    walk_xfer_m:     float
    score:           float
    summary:         str = ""


# ══════════════════════════════════════════════════════════════════════════════
# Stop Manager
# ══════════════════════════════════════════════════════════════════════════════

# Named stops that are always treated as terminal-class (large icon on map)
_TERMINAL_KEYWORDS = {
    "railways", "odeon", "ambassadeur", "kencom", "koja",
    "rongai town", "bomas", "jkia", "wilson", "thika",
    "westlands", "athi river", "syokimau",
    # walk-adjacent in-CBD stops (near but not inside a main terminal)
    "agip", "kaka", "gill house", "racecourse road", "gpo", "tom mboya",
}


class StopManager:
    """Loads stops from a DataFrame and provides fast nearest-stop lookups."""

    def __init__(self, stops_df: pd.DataFrame):
        self._stops: dict[str, Stop] = {}
        self._list: list[Stop] = []
        self._build(stops_df)

    def _build(self, df: pd.DataFrame):
        for _, row in df.iterrows():
            name  = str(row["stop_name"]).strip()
            sid   = str(row["stop_id"])
            lat   = float(row["stop_lat"])
            lon   = float(row["stop_lon"])
            rstr  = str(row.get("routes", ""))
            routes = rstr.split() if rstr not in ("", "nan") else []
            stype = "terminal" if any(k in name.lower() for k in _TERMINAL_KEYWORDS) \
                    else "roadside"
            s = Stop(sid, name, lat, lon, stype, routes)
            self._stops[sid] = s
            self._list.append(s)

    # ── Lookups ───────────────────────────────────────────────────────────────

    def nearest(self, lat: float, lon: float, n: int = 6,
                max_m: float = MAX_WALK_M) -> list[Stop]:
        """Return up to n stops within max_m metres, sorted by distance."""
        def d(s):
            return _haversine_m((lat, lon), (s.lat, s.lon))
        return [s for s in sorted(self._list, key=d) if d(s) <= max_m][:n]

    def all_stops(self) -> list[Stop]:
        return self._list


# ══════════════════════════════════════════════════════════════════════════════
# Transit Graph Builder
# ══════════════════════════════════════════════════════════════════════════════

class TransitGraphBuilder:
    """
    Builds a directed NetworkX graph from the transit CSV.
    Nodes  = lowercase waypoint names (from route_long hyphen-separated field).
    Edges  = consecutive waypoint pairs on a route, weighted by fare/time/dist.
    """

    def __init__(self, transit_df: pd.DataFrame):
        self.df = transit_df.copy()

    def build(self) -> nx.MultiDiGraph:
        G = nx.MultiDiGraph()
        seen: set[str] = set()

        for _, row in self.df.iterrows():
            rid = str(row["route_name"])
            if rid in seen:
                continue
            seen.add(rid)

            fare      = int(row.get("fare_KSh", 50))
            dist_km   = float(row.get("dist_km", 1.0))
            headsign  = str(row.get("headsign", ""))
            wps = [w.strip().lower() for w in str(row["route_long"]).split("-")
                   if w.strip()]

            n_segs = max(len(wps) - 1, 1)
            for i in range(len(wps) - 1):
                a, b = wps[i], wps[i + 1]
                seg_dist = dist_km / n_segs
                G.add_edge(
                    a, b,
                    route_id  = rid,
                    headsign  = headsign,
                    fare      = fare,
                    dist_km   = seg_dist,
                    time_min  = (seg_dist / AVG_MATATU_KMH) * 60,
                )
        return G


# ══════════════════════════════════════════════════════════════════════════════
# OSMnx walking-network helpers  (gracefully degrade without osmnx)
# ══════════════════════════════════════════════════════════════════════════════

_WALK_GRAPH = None   # module-level cache — loaded once on first use


def get_walk_graph(place: str = "Nairobi, Kenya"):
    """Download (and cache) the Nairobi pedestrian network from OSM."""
    global _WALK_GRAPH
    if _WALK_GRAPH is not None:
        return _WALK_GRAPH
    if not _HAS_OSMNX:
        return None
    try:
        _WALK_GRAPH = ox.graph_from_place(place, network_type="walk",
                                           simplify=True, retain_all=False)
        return _WALK_GRAPH
    except Exception:
        return None


def _snap_node(G, lat: float, lon: float):
    """Return OSMnx node id nearest to (lat, lon), or None."""
    if G is None:
        return None
    try:
        return ox.nearest_nodes(G, lon, lat)
    except Exception:
        return None


def _walk_coords(G, na, nb) -> list[tuple[float, float]]:
    """
    Return (lat, lon) list along the shortest walkable path.
    Falls back to an empty list so callers can use a straight line instead.
    """
    if G is None or na is None or nb is None:
        return []
    try:
        path = nx.shortest_path(G, na, nb, weight="length")
        return [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in path]
    except Exception:
        return []


def _walk_dist_m(G, na, nb) -> float:
    """Shortest walking distance in metres between two OSM nodes."""
    if G is None or na is None or nb is None:
        return 0.0
    try:
        return float(nx.shortest_path_length(G, na, nb, weight="length"))
    except Exception:
        return 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Main Routing Engine
# ══════════════════════════════════════════════════════════════════════════════

class NairobiRoutingEngine:
    """
    Entry point for route planning.

    Usage::
        engine = NairobiRoutingEngine(transit_df, stops_df, shapes)
        options = engine.get_route_options((lat, lng), "Karen")
    """

    def __init__(self,
                 transit_df: pd.DataFrame,
                 stops_df:   pd.DataFrame,
                 shapes:     dict | None = None):
        self.transit_df = transit_df.copy()
        self.stop_mgr   = StopManager(stops_df)
        self.shapes     = shapes or {}          # {route_id: [(lon, lat), …]}
        self.graph      = TransitGraphBuilder(transit_df).build()
        self._geocoder  = Nominatim(user_agent="nairobi_transit_engine_v2", timeout=8)
        self._G_walk    = None   # lazy-loaded on first use

    # ── Public API ────────────────────────────────────────────────────────────

    def geocode(self, keyword: str) -> tuple[float, float] | None:
        try:
            loc = self._geocoder.geocode(f"{keyword}, Nairobi, Kenya")
            return (loc.latitude, loc.longitude) if loc else None
        except Exception:
            return None

    def get_route_options(self,
                          user_origin: tuple[float, float],
                          destination_keyword: str) -> list[RouteOption]:
        """
        Find and rank Direct + Transfer route options.

        Parameters
        ----------
        user_origin          : (lat, lon) of the user's current position
        destination_keyword  : plain-text destination, e.g. "Karen" or "Rongai"

        Returns
        -------
        Up to 4 RouteOption objects sorted by composite score (lowest = best).
        """
        dest = self.geocode(destination_keyword)
        if not dest:
            return []

        G  = self._lazy_walk()
        o_node = _snap_node(G, *user_origin)
        d_node = _snap_node(G, *dest)

        orig_stops = self.stop_mgr.nearest(*user_origin)
        dest_stops = self.stop_mgr.nearest(*dest)

        options: list[RouteOption] = []

        # ── Direct routes ─────────────────────────────────────────────────────
        for os_ in orig_stops:
            for ds_ in dest_stops:
                common = set(os_.routes) & set(ds_.routes)
                for rid in common:
                    leg = self._make_leg(rid, os_, ds_)
                    if leg is None:
                        continue

                    os_node = _snap_node(G, os_.lat, os_.lon)
                    ds_node = _snap_node(G, ds_.lat, ds_.lon)
                    walk_o  = _walk_dist_m(G, o_node, os_node) or _haversine_m(user_origin, (os_.lat, os_.lon))
                    walk_d  = _walk_dist_m(G, ds_node, d_node) or _haversine_m(dest, (ds_.lat, ds_.lon))

                    t_total = leg.time_min + _walk_time(walk_o + walk_d)
                    score   = self._score(leg.fare, t_total, 0, walk_o + walk_d)

                    options.append(RouteOption(
                        option_id      = f"direct_{rid}",
                        option_type    = "direct",
                        legs           = [leg],
                        total_fare     = leg.fare,
                        total_time_min = t_total,
                        walk_origin_m  = walk_o,
                        walk_dest_m    = walk_d,
                        walk_xfer_m    = 0.0,
                        score          = score,
                        summary        = (
                            f"Direct | {rid} | {leg.fare} KES"
                            f" | ~{t_total:.0f} min | Walk: {walk_o+walk_d:.0f} m"
                        ),
                    ))

        # ── 1-transfer routes ─────────────────────────────────────────────────
        for os_ in orig_stops[:3]:
            for ds_ in dest_stops[:3]:
                for r1 in os_.routes:
                    for r2 in ds_.routes:
                        if r1 == r2:
                            continue
                        for t_stop in self._transfer_stops(r1, r2)[:2]:
                            leg1 = self._make_leg(r1, os_, t_stop)
                            leg2 = self._make_leg(r2, t_stop, ds_)
                            if leg1 is None or leg2 is None:
                                continue

                            os_node = _snap_node(G, os_.lat, os_.lon)
                            ds_node = _snap_node(G, ds_.lat, ds_.lon)
                            walk_o  = _walk_dist_m(G, o_node, os_node) or _haversine_m(user_origin, (os_.lat, os_.lon))
                            walk_d  = _walk_dist_m(G, ds_node, d_node) or _haversine_m(dest, (ds_.lat, ds_.lon))

                            fare    = leg1.fare + leg2.fare
                            t_total = (leg1.time_min + leg2.time_min
                                       + TRANSFER_PENALTY + _walk_time(walk_o + walk_d))
                            score   = self._score(fare, t_total, 1, walk_o + walk_d)

                            options.append(RouteOption(
                                option_id      = f"xfer_{r1}_{r2}_{t_stop.stop_id}",
                                option_type    = "transfer",
                                legs           = [leg1, leg2],
                                total_fare     = fare,
                                total_time_min = t_total,
                                walk_origin_m  = walk_o,
                                walk_dest_m    = walk_d,
                                walk_xfer_m    = 0.0,
                                score          = score,
                                summary        = (
                                    f"Transfer @ {t_stop.name} | {r1}→{r2}"
                                    f" | {fare} KES | ~{t_total:.0f} min"
                                    f" | Walk: {walk_o+walk_d:.0f} m"
                                ),
                            ))

        # Deduplicate and rank
        seen: set[str] = set()
        ranked: list[RouteOption] = []
        for opt in sorted(options, key=lambda x: x.score):
            if opt.option_id not in seen:
                seen.add(opt.option_id)
                ranked.append(opt)

        return ranked[:4]

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _lazy_walk(self):
        if self._G_walk is None:
            self._G_walk = get_walk_graph()
        return self._G_walk

    def _make_leg(self, route_id: str, from_stop: Stop, to_stop: Stop) -> RouteLeg | None:
        rows = self.transit_df[self.transit_df["route_name"].astype(str) == route_id]
        if rows.empty:
            return None
        row = rows.iloc[0]
        raw = self.shapes.get(route_id, [])
        shape_pts = [(p[1], p[0]) for p in raw]   # (lon,lat) → (lat,lon)
        is_walk = str(row.get("fare_source", "")).lower() == "walk"
        speed   = WALK_KMH if is_walk else AVG_MATATU_KMH
        return RouteLeg(
            route_id   = route_id,
            route_long = str(row.get("route_long", "")),
            from_stop  = from_stop,
            to_stop    = to_stop,
            fare       = int(row.get("fare_KSh", 0)) if is_walk else int(row.get("fare_KSh", 50)),
            dist_km    = float(row.get("dist_km", 1.0)),
            time_min   = float(row.get("dist_km", 1.0)) / speed * 60,
            shape_pts  = shape_pts,
        )

    def _transfer_stops(self, r1: str, r2: str) -> list[Stop]:
        return [s for s in self.stop_mgr.all_stops()
                if r1 in s.routes and r2 in s.routes]

    def _score(self, fare: float, time_min: float,
               n_xfer: int, walk_m: float) -> float:
        """
        Composite cost — lower is better.
        Normalised against typical Nairobi journey ranges.
        """
        price_n  = fare / 300.0
        time_n   = time_min / 90.0
        walk_n   = walk_m / 1000.0
        xfer_pen = n_xfer * TRANSFER_PENALTY / 90.0
        return (W_PRICE * price_n
                + W_TIME  * (time_n + walk_n * 0.5)
                + W_TRANSFER * xfer_pen)


# ══════════════════════════════════════════════════════════════════════════════
# Folium Map Generator
# ══════════════════════════════════════════════════════════════════════════════

class FoliumMapGenerator:
    """
    Converts a list of RouteOption objects into a layered, interactive Folium map.

    Visual language
    ---------------
    Direct route    → solid green polyline
    Transfer leg 1  → solid blue polyline
    Transfer leg 2  → solid purple polyline
    Walking segment → dashed orange polyline (follows OSMnx walking network)
    Boarding stop   → blue circle marker  (larger = terminal)
    Alighting stop  → green circle marker
    Transfer stop   → orange marker with exchange icon
    Origin          → red person icon
    Destination     → green flag icon
    """

    def __init__(self, G_walk=None):
        self.G = G_walk

    # ── Public ────────────────────────────────────────────────────────────────

    def generate(self,
                 origin:      tuple[float, float],
                 destination: tuple[float, float],
                 dest_name:   str,
                 options:     list[RouteOption]) -> folium.Map:
        """Return a folium.Map ready for .save() or ._repr_html_()."""
        centre = (
            (origin[0] + destination[0]) / 2,
            (origin[1] + destination[1]) / 2,
        )
        m = folium.Map(location=centre, zoom_start=13,
                       tiles="CartoDB positron")

        # Fixed markers
        folium.Marker(
            origin,
            popup=folium.Popup("<b>🧍 Your Location</b>", max_width=180),
            tooltip="Your location",
            icon=folium.Icon(color="red", icon="user", prefix="fa"),
        ).add_to(m)

        folium.Marker(
            destination,
            popup=folium.Popup(f"<b>🏁 {dest_name}</b>", max_width=180),
            tooltip=dest_name,
            icon=folium.Icon(color="green", icon="flag", prefix="fa"),
        ).add_to(m)

        # One FeatureGroup per option (toggleable in LayerControl)
        for i, opt in enumerate(options):
            label = f"Option {chr(65+i)} — {opt.summary}"
            fg = folium.FeatureGroup(name=label, show=(i == 0))

            if opt.option_type == "direct":
                self._draw_direct(fg, opt, origin, destination, i)
            else:
                self._draw_transfer(fg, opt, origin, destination, i)

            fg.add_to(m)

        folium.LayerControl(collapsed=False).add_to(m)
        return m

    # ── Route drawing ─────────────────────────────────────────────────────────

    def _draw_direct(self, fg, opt: RouteOption,
                     origin: tuple, dest: tuple, idx: int):
        leg   = opt.legs[0]
        color = C_DIRECT

        route_pts = self._route_pts(leg)
        folium.PolyLine(
            route_pts, color=color, weight=6, opacity=0.88,
            tooltip=f"Option {chr(65+idx)} — Direct: {leg.route_id}",
            popup=folium.Popup(
                f"<b>✅ Direct Route</b><br>"
                f"Route:&nbsp;<b>{leg.route_id}</b><br>"
                f"Corridor:&nbsp;{leg.route_long}<br>"
                f"Fare:&nbsp;<b>{leg.fare} KES</b><br>"
                f"Time:&nbsp;~{leg.time_min:.0f} min matatu<br>"
                f"Walk origin:&nbsp;{opt.walk_origin_m:.0f} m<br>"
                f"Walk dest:&nbsp;{opt.walk_dest_m:.0f} m<br>"
                f"<hr><i>{opt.summary}</i>",
                max_width=240,
            ),
        ).add_to(fg)

        # Walking: origin → boarding stop
        self._walk_line(fg, origin, (leg.from_stop.lat, leg.from_stop.lon),
                        f"Walk to {leg.from_stop.name} (~{opt.walk_origin_m:.0f} m)")
        # Walking: alighting stop → destination
        self._walk_line(fg, (leg.to_stop.lat, leg.to_stop.lon), dest,
                        f"Walk to destination (~{opt.walk_dest_m:.0f} m)")

        self._stop_pin(fg, leg.from_stop, "board")
        self._stop_pin(fg, leg.to_stop,   "alight")

    def _draw_transfer(self, fg, opt: RouteOption,
                       origin: tuple, dest: tuple, idx: int):
        leg1, leg2 = opt.legs[0], opt.legs[1]

        # Leg 1 — blue
        folium.PolyLine(
            self._route_pts(leg1), color=C_LEG1, weight=6, opacity=0.88,
            tooltip=f"Leg 1: {leg1.route_id} — {leg1.fare} KES",
            popup=folium.Popup(
                f"<b>🔵 Leg 1</b><br>"
                f"Route:&nbsp;<b>{leg1.route_id}</b><br>"
                f"Fare:&nbsp;{leg1.fare} KES<br>"
                f"Time:&nbsp;~{leg1.time_min:.0f} min",
                max_width=200,
            ),
        ).add_to(fg)

        # Leg 2 — purple
        folium.PolyLine(
            self._route_pts(leg2), color=C_LEG2, weight=6, opacity=0.88,
            tooltip=f"Leg 2: {leg2.route_id} — {leg2.fare} KES",
            popup=folium.Popup(
                f"<b>🟣 Leg 2</b><br>"
                f"Route:&nbsp;<b>{leg2.route_id}</b><br>"
                f"Fare:&nbsp;{leg2.fare} KES<br>"
                f"Time:&nbsp;~{leg2.time_min:.0f} min",
                max_width=200,
            ),
        ).add_to(fg)

        # Transfer stop
        t = leg1.to_stop
        folium.Marker(
            (t.lat, t.lon),
            popup=folium.Popup(
                f"<b>⇄ Transfer: {t.name}</b><br>"
                f"Alight {leg1.route_id} here<br>"
                f"Board {leg2.route_id} here<br>"
                f"+{TRANSFER_PENALTY:.0f} min wait penalty",
                max_width=210,
            ),
            tooltip=f"Transfer: {t.name}",
            icon=folium.Icon(color="orange", icon="exchange", prefix="fa"),
        ).add_to(fg)

        # Walking legs
        self._walk_line(fg, origin, (leg1.from_stop.lat, leg1.from_stop.lon),
                        f"Walk to {leg1.from_stop.name} (~{opt.walk_origin_m:.0f} m)")
        self._walk_line(fg, (leg2.to_stop.lat, leg2.to_stop.lon), dest,
                        f"Walk to destination (~{opt.walk_dest_m:.0f} m)")

        self._stop_pin(fg, leg1.from_stop, "board")
        self._stop_pin(fg, leg1.to_stop,   "transfer")
        self._stop_pin(fg, leg2.to_stop,   "alight")

    # ── Drawing helpers ───────────────────────────────────────────────────────

    def _route_pts(self, leg: RouteLeg) -> list[tuple[float, float]]:
        """Shape points for a leg; straight line if no GPS shape available."""
        if leg.shape_pts and len(leg.shape_pts) >= 2:
            return leg.shape_pts
        return [(leg.from_stop.lat, leg.from_stop.lon),
                (leg.to_stop.lat,   leg.to_stop.lon)]

    def _walk_line(self, fg, a: tuple, b: tuple, tooltip: str):
        """Draw a dashed walking segment, following OSMnx network if available."""
        pts = self._osm_walk(a, b)
        folium.PolyLine(
            pts, color=C_WALK, weight=3, opacity=0.85,
            dash_array="8 5", tooltip=tooltip,
        ).add_to(fg)

    def _osm_walk(self, a: tuple, b: tuple) -> list[tuple[float, float]]:
        if self.G is None:
            return [a, b]
        try:
            na = ox.nearest_nodes(self.G, a[1], a[0])
            nb = ox.nearest_nodes(self.G, b[1], b[0])
            pts = _walk_coords(self.G, na, nb)
            return pts if len(pts) >= 2 else [a, b]
        except Exception:
            return [a, b]

    def _stop_pin(self, fg, stop: Stop, role: str):
        colours = {"board": "blue", "alight": "green", "transfer": "orange"}
        colour  = colours.get(role, "gray")
        radius  = 12 if stop.stop_type == "terminal" else 8
        folium.CircleMarker(
            (stop.lat, stop.lon),
            radius=radius,
            color=colour, fill=True,
            fill_color=colour, fill_opacity=0.75,
            popup=folium.Popup(
                f"<b>{stop.name}</b><br>"
                f"Type: {stop.stop_type}<br>"
                f"Role: {role}<br>"
                f"Routes: {', '.join(stop.routes[:6])}",
                max_width=210,
            ),
            tooltip=f"{role.title()}: {stop.name}",
        ).add_to(fg)


# ══════════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════════

def _haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Straight-line distance in metres between two (lat, lon) tuples."""
    return geodesic(a, b).meters


def _walk_time(dist_m: float) -> float:
    """Walking time in minutes for a given distance in metres."""
    return (dist_m / 1000.0) / WALK_KMH * 60.0
