"""
Nairobi matatu multi-hop router.

Builds a waypoint graph from route_long fields and the stops database.
For any origin→destination query it finds:
  • Direct routes (single leg)
  • 1-transfer routes (two legs via a common interchange)

Transfer detection is pure set-intersection: if route A passes through
waypoint X and route B also passes through X, then X is a valid transfer.
"""

import re
import math
import json
import os
from collections import defaultdict
import pandas as pd

# ── Pre-computed transfer cache ───────────────────────────────────────────────
_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "transfer_cache.json")
try:
    with open(_CACHE_PATH, encoding="utf-8") as _f:
        _TRANSFER_CACHE: dict = json.load(_f)
except FileNotFoundError:
    _TRANSFER_CACHE = {"direct": {}, "transfers": {}}

# ── Road corridor waypoints ───────────────────────────────────────────────────
# These are physical roads in the dataset. When an origin/destination resolves
# to one of these, the instruction is "get to the road and flag down" rather
# than "walk to a specific stage".
ROAD_WAYPOINTS = {
    "langata road", "ngong road", "ngorng road", "ngongroad",
    "mombasa road", "jogoo road", "juja road", "kiambu road",
    "valley road", "munyu road", "thika road", "kangundo road",
    "othaya road", "peponi road", "racecourse road", "mbagathi road",
    "naivasha road", "landhies road", "accra road",
}

# ── Landmark aliases ──────────────────────────────────────────────────────────
# Maps common user inputs → canonical waypoint names in the route dataset.
# Road-based entries let people say "I'm at Langata Barracks" and the router
# correctly resolves them to a corridor (Langata Road) rather than a fixed stop.
LANDMARKS = {
    # CBD / city centre synonyms
    "cbd": "railways", "town": "railways", "nairobi cbd": "railways",
    "kencom": "railways", "posta": "railways", "ambassadeur": "railways",
    "gposta": "railways", "archives": "railways", "khoja": "koja",

    # Malls — mapped to nearest corridor waypoint
    "westgate": "westlands", "sarit": "westlands", "sarit centre": "westlands",
    "village market": "westlands", "delta": "westlands",
    "abc place": "westlands", "mega mall": "westlands",
    "junction": "ngong road", "junction mall": "ngong road",
    "t-mall": "ngong road",
    "the hub": "karen", "hub mall": "karen",
    "galleria": "bomas", "galleria mall": "bomas",
    "garden city": "roysambu", "thika road mall": "roysambu", "trm": "roysambu",
    "two rivers": "regen", "two rivers mall": "regen",
    "yaya": "yaya centre", "yaya mall": "yaya centre",
    "prestige plaza": "ngong road", "nairobi hospital": "ngong road",

    # Hospitals
    "knh": "knh", "kenyatta hospital": "knh",
    "kenyatta national hospital": "knh",
    "aga khan": "pangani", "aga khan hospital": "pangani",
    "mp shah": "mp shah", "mp shah hospital": "mp shah",
    "parklands hospital": "mp shah",
    "mater": "mater hospital", "mater hospital": "mater hospital",
    "nairobi west hospital": "langata road",
    "mbagathi hospital": "mbagathi road",

    # Universities / schools
    "ku": "ku", "kenyatta university": "ku",
    "strathmore": "strathmore school",
    "usiu": "roysambu",
    "uon": "railways", "university of nairobi": "railways",

    # Airports
    "jkia": "jkia", "airport": "jkia",
    "nairobi airport": "jkia", "wilson": "wilson airport",
    "wilson airport": "wilson airport",

    # ── Langata Road corridor ────────────────────────────────────────────────
    "langata barracks": "langata road",
    "carnivore": "langata road",
    "langata barracks junction": "langata road",
    "nairobi west": "langata road",
    "impala club": "langata road",
    "langata link": "langata road",
    "south c": "langata road",
    "hardy": "hardy",

    # ── Ngong Road corridor ──────────────────────────────────────────────────
    "upper hill": "ngong road",
    "times tower": "ngong road",
    "shankardass": "ngong road",
    "milimani": "ngong road",
    "hurlingham": "ngong road",
    "uchumi ngong road": "ngong road",
    "daystar": "ngong road",

    # ── Mombasa Road corridor ────────────────────────────────────────────────
    "embakasi": "mombasa road",
    "cabanas": "cabanas",
    "mlolongo": "mlolongo",
    "athi river": "athi river",
    "industrial area": "mombasa road",
    "bellevue": "mombasa road",

    # ── Jogoo Road corridor ──────────────────────────────────────────────────
    "buruburu": "buruburu",
    "donholm": "donholm",
    "shauri moyo": "jogoo road",
    "jerusalem": "jogoo road",

    # ── Juja Road (Pangani/Huruma) corridor ──────────────────────────────────
    "pangani": "pangani",
    "huruma": "huruma",
    "mathare": "mathare",

    # ── Kiambu Road corridor ─────────────────────────────────────────────────
    "muthaiga": "kiambu road",
    "runda": "kiambu road",
    "ridgeways": "kiambu road",

    # ── Thika Road corridor (through Roysambu/Githurai) ──────────────────────
    "kasarani": "kasarani",
    "safari park": "roysambu",
    "roasters": "roysambu",
    "allsops": "allsops",

    # ── Valley Road ──────────────────────────────────────────────────────────
    "yaya centre": "yaya centre",

    # General shortcuts
    "rongai": "ongata rongai", "bomas of kenya": "bomas",
    "dagoretti": "dagoretti market", "dagoretti corner": "dagoretti market",
    "proggie": "proggie", "progressive": "proggie",
    "kahawa": "kahawa west",

    # ── Kahawa / Bypass / KU Referral corridor ───────────────────────────────
    "kahawa west": "kahawa west",
    "bypass": "bypass",
    "thika bypass": "bypass",
    "membley": "membley",
    "ku referral": "ku referral",
    "kutrrh": "ku referral",
    "kenyatta university referral": "ku referral",
    "ku teaching": "ku referral",

    # ── Syokimau / Mombasa Road south ────────────────────────────────────────
    "syokimau": "syokimau",

    # ── Thika ────────────────────────────────────────────────────────────────
    "thika": "thika",
    "thika town": "thika",

    # ── Parklands ────────────────────────────────────────────────────────────
    "parklands": "parklands",
    "highridge": "parklands",

    # ── CBD streets & sub-stages ─────────────────────────────────────────────
    "moi avenue": "odeon",
    "moi ave": "odeon",
    "odeon cinema": "odeon",
    "odeon stage": "odeon",
    "otc roundabout": "otc",
    "o.t.c": "otc",
    "otc stage": "otc",
    "ronald ngala": "railways",
    "ronald ngala street": "railways",
    "kenyatta avenue": "railways",
    "haile selassie": "railways",
    "haile selassie avenue": "railways",
    "city hall": "railways",
    "afya centre": "railways",
    "afya house": "railways",
    "gill house": "railways",
    "harambee": "railways",
    "harambee avenue": "railways",
    "nation centre": "railways",
    "globe cinema": "railways",
    "central police": "railways",
    "serena": "railways",
    "gpo": "railways",
    "hilton": "railways",
    "bus station nairobi": "bus station",
    "country bus station": "bus station",
    "muthurwa": "muthurwa",
    "gikomba": "gikomba",
    "gikomba market": "gikomba",
    "ngara": "ngara",
    "ngara stage": "ngara",
    "equity ngara": "ngara",

    # ── Eastleigh ─────────────────────────────────────────────────────────────
    "eastleigh": "eastleigh",
    "garissa lodge": "eastleigh",
    "pumwani": "eastleigh",

    # ── Rongai / Kiserian ─────────────────────────────────────────────────────
    "tumaini rongai": "ongata rongai",
    "rongai tumaini": "ongata rongai",
    "tumaini stage": "ongata rongai",
    "nibs": "ongata rongai",
    "nibs rongai": "ongata rongai",
    "uchumi rongai": "ongata rongai",
    "maasai mall": "ongata rongai",
    "rongai town": "ongata rongai",
    "rimpa": "ongata rongai",
    "kiserian": "kiserian",

    # ── Pipeline / Tassia / Donholm ──────────────────────────────────────────
    "tassia": "pipeline",
    "tumaini tassia": "pipeline",
    "pipeline estate": "pipeline",
    "fedha": "fedha",
    "imara daima": "imara daima",
    "imara": "imara daima",
    "utawala": "utawala",

    # ── Umoja / Kayole / Komarock ─────────────────────────────────────────────
    "umoja": "umoja",
    "umoja estate": "umoja",
    "umoja 2": "umoja 2",
    "kayole": "kayole",
    "kayole junction": "kayole junction",
    "komarocks": "komarocks",
    "komarock": "komarocks",
    "kariobangi": "kariobangi",
    "kariobangi north": "kariobangi north",
    "kariobangi south": "kariobangi south",
    "lucky summer": "lucky summer",
    "kariokor": "kariokor",

    # ── Buruburu / Donholm / Jogoo Road ──────────────────────────────────────
    "buru buru": "buruburu",
    "buru": "buruburu",
    "donholm": "donholm",
    "donholm stage": "donholm",
    "maringo": "maringo",
    "jerusalem": "jerusalem",
    "makadara": "makadara",
    "shauri moyo": "shauri moyo",

    # ── Zimmerman / Githurai / Thika Road north ──────────────────────────────
    "zimmerman": "zimmerman",
    "zimmerman estate": "zimmerman",
    "githurai": "githurai",
    "githurai 44": "githurai",
    "githurai 45": "githurai",
    "baba dogo": "baba dogo",
    "mwiki": "mwiki",
    "dandora": "dandora",
    "ruai": "ruai",
    "njiru": "njiru",
    "outering": "outering",
    "lucky summer": "lucky summer",

    # ── Westlands / Chiromo / UN / Parklands ────────────────────────────────
    "chiromo": "chiromo",
    "un complex": "un", "unep": "un",
    "united nations": "un", "gigiri": "un",
    "lavington": "lavington",
    "kileleshwa": "kileleshwa",
    "spring valley": "spring valley",
    "ukay": "ukay centre", "ukay centre": "ukay centre",
    "runda": "runda", "runda estate": "runda",
    "githogoro": "githogoro",
    "karura": "karura", "karura forest": "karura",

    # ── Kawangware / Kangemi / Kabete / Uthiru ───────────────────────────────
    "kawangware": "kawangware",
    "kangemi": "kangemi",
    "kabete": "kabete",
    "uthiru": "uthiru",
    "kinoo": "kinoo",
    "wangige": "wangige",
    "kikuyu": "kikuyu",

    # ── Kibera / Langata extended ────────────────────────────────────────────
    "kibera": "kibera",
    "olympic": "kibera",
    "jamhuri": "jamhuri",
    "ngumo": "ngumo",
    "south b": "south b",
    "nkoroi": "nkoroi",

    # ── Karen / Hardy extended ───────────────────────────────────────────────
    "karen hospital": "karen hospital",
    "ngong town": "ngong",
    "ngong": "ngong",

    # ── Ruiru / Juja / Thika road corridors ──────────────────────────────────
    "ruiru": "ruiru",
    "juja": "juja",
    "juja town": "juja",
    "ndenderu": "ndenderu",
    "ruaka": "ruaka",
    "banana": "banana",

    # ── South / Mombasa Road extended ────────────────────────────────────────
    "mlolongo": "mlolongo",
    "athi river": "athi river",
    "kitengela": "kitengela",
    "cabanas": "cabanas",
    "bellevue": "belleview",
}

# Nairobi fare bands — round model estimates to nearest valid step
FARE_BANDS = [20, 30, 40, 50, 60, 70, 80, 100, 120, 150, 200]

# Transit interchange hubs: all names are treated as the same boarding point.
# This is how Nairobi works — Odeon, Railways, Ambassadeur are all "town".
INTERCHANGE_HUBS: list[tuple[str, list[str]]] = [
    ("__cbd__", [
        "railways", "ambassadeur", "kencom", "odeon", "koja",
        "town", "cbd", "khoja", "archives", "posta",
        "commercial", "bus station", "gill house",
        "agip", "gpo", "tom mboya", "racecourse road",
    ]),
    ("__otc__",       ["otc", "o.t.c", "otc roundabout", "kaka"]),
    ("__ngara__",     ["ngara", "ngara stage", "equity ngara"]),
    ("__muthurwa__",  ["muthurwa"]),
    ("__westlands__", ["westlands", "westlands bypass", "ukay centre"]),
    ("__regen__",     ["regen", "two rivers"]),
    ("__ngong_road__", ["ngong road", "ngorng road", "ngongroad"]),  # typos in data
    ("__pangani__",   ["pangani", "pangani flyover", "pangani terminus"]),
    ("__roysambu__",  ["roysambu", "roysabu"]),                       # typo in data
    ("__bypass__",    ["bypass", "thika road bypass"]),
    ("__kahawa_west__", ["kahawa west", "kahawa"]),
    ("__ku_referral__", ["ku referral", "ku referral (kutrrh)", "kutrrh"]),
    ("__donholm__",   ["donholm", "caltex donholm"]),
    ("__thika__",     ["thika", "thika town"]),
    ("__rongai__",    ["ongata rongai", "rongai"]),
    ("__pipeline__",  ["pipeline", "tassia"]),
]

# Build lookup: canonical_name → member set, and member → canonical_name
_HUB_OF: dict[str, str] = {}
_HUB_MEMBERS: dict[str, set[str]] = {}
for _canon, _members in INTERCHANGE_HUBS:
    _HUB_MEMBERS[_canon] = set(_members)
    for _m in _members:
        _HUB_OF[_m] = _canon


def _snap(fare: float) -> int:
    return min(FARE_BANDS, key=lambda b: abs(b - fare))


class NairobiRouter:

    def __init__(self, transit_df: pd.DataFrame, stops_df: pd.DataFrame):
        self.df      = transit_df.copy()
        self.stops   = stops_df.copy()
        self._build_index()

    # ── Index building ────────────────────────────────────────────────────────

    @staticmethod
    def _parse_wps(route_long: str) -> list[str]:
        return [w.strip().lower() for w in str(route_long).split("-") if w.strip()]

    def _build_index(self):
        # route_id → ordered waypoints list
        self.route_wps: dict[str, list[str]] = {}
        # route_id → meta (fare, dist, headsign, route_long)
        self.route_meta: dict[str, dict] = {}
        # waypoint → set of route_ids that include it
        self.wp_routes: dict[str, set[str]] = defaultdict(set)

        seen = set()
        for _, row in self.df.iterrows():
            rid = str(row["route_name"])
            if rid in seen:
                continue
            seen.add(rid)
            wps = self._parse_wps(row["route_long"])
            self.route_wps[rid]  = wps
            self.route_meta[rid] = {
                "fare":       int(row["fare_KSh"]),
                "dist_km":    float(row["dist_km"]),
                "headsign":   str(row["headsign"]),
                "route_long": str(row["route_long"]),
            }
            for wp in wps:
                self.wp_routes[wp].add(rid)

        # Add stop names from the stops database as extra waypoints
        for _, row in self.stops.iterrows():
            sname = str(row["stop_name"]).strip().lower()
            routes_str = str(row.get("routes", ""))
            if not routes_str or routes_str == "nan":
                continue
            for rid in routes_str.split():
                if rid in self.route_wps:
                    self.wp_routes[sname].add(rid)

        # Merge interchange hubs: index every route under the hub canonical name
        # so that "odeon" routes and "railways" routes all appear at "__cbd__"
        for canon, members in _HUB_MEMBERS.items():
            hub_routes: set[str] = set()
            for m in members:
                hub_routes |= self.wp_routes.get(m, set())
            if hub_routes:
                self.wp_routes[canon] = hub_routes

        self._all_wps = set(self.wp_routes.keys())

    # ── Name resolution ───────────────────────────────────────────────────────

    @staticmethod
    def _normalise(text: str) -> str:
        """Strip punctuation variants that users type (backtick, curly quotes, dashes, slashes)."""
        return (
            text.strip().lower()
            .replace("`", "")
            .replace("’", "").replace("’", "")
            .replace("’", "")
            .replace("-", " ")
            .replace("/", " ")
        ).strip()

    def _resolve(self, place: str) -> list[str]:
        """Map user input to one or more canonical waypoint names."""
        p = self._normalise(place)
        p = LANDMARKS.get(p, p)

        # If the name belongs to a hub, use the hub canonical name
        if p in _HUB_OF:
            return [_HUB_OF[p]]

        # Exact match
        if p in self._all_wps:
            return [p]

        # Substring matches (user said "karen" → matches "karen hospital", "karen")
        matches = sorted(
            [wp for wp in self._all_wps
             if not wp.startswith("__") and (p in wp or wp in p)],
            key=lambda wp: abs(len(wp) - len(p))
        )
        return matches[:3] if matches else []

    # ── Fare estimation for a partial leg ─────────────────────────────────────

    def _actual_wp(self, rid: str, hub_or_wp: str) -> str | None:
        """
        If hub_or_wp is a hub canonical name, find which member waypoint
        actually appears in this route's waypoint list.
        """
        wps = self.route_wps.get(rid, [])
        if hub_or_wp in wps:
            return hub_or_wp
        if hub_or_wp in _HUB_MEMBERS:
            for member in _HUB_MEMBERS[hub_or_wp]:
                if member in wps:
                    return member
        return None

    def _leg_fare(self, rid: str, from_wp: str, to_wp: str) -> int:
        wps       = self.route_wps[rid]
        full_fare = self.route_meta[rid]["fare"]
        n         = len(wps) - 1

        fw = self._actual_wp(rid, from_wp)
        tw = self._actual_wp(rid, to_wp)
        if fw is None or tw is None:
            return full_fare

        try:
            i = wps.index(fw)
            j = wps.index(tw)
        except ValueError:
            return full_fare

        steps = abs(j - i)
        if steps == 0:
            return 0
        if steps >= n:
            return full_fare

        fraction = steps / n
        raw = full_fare * (0.4 + 0.6 * fraction)
        return _snap(raw)

    def _windscreen(self, rid: str, from_wp: str, to_wp: str) -> str:
        """What the destination on the windscreen will show."""
        wps = self.route_wps[rid]
        fw  = self._actual_wp(rid, from_wp)
        tw  = self._actual_wp(rid, to_wp)
        if fw is None or tw is None:
            return self.route_meta[rid]["headsign"].title()
        try:
            i = wps.index(fw)
            j = wps.index(tw)
        except ValueError:
            return self.route_meta[rid]["headsign"].title()
        terminus = wps[-1] if j >= i else wps[0]
        return terminus.title()

    # ── Journey finder ────────────────────────────────────────────────────────

    def find_journeys(self, origin_raw: str, dest_raw: str) -> list[dict]:
        origins = self._resolve(origin_raw)
        dests   = self._resolve(dest_raw)

        if not origins or not dests:
            return []

        journeys: list[dict] = []
        short_leg_filtered = False  # set when a route is skipped for being < 10 km

        for origin in origins:
            for dest in dests:
                if origin == dest:
                    continue

                # Check pre-computed cache first (both key orderings)
                for key in (f"{origin}|{dest}", f"{dest}|{origin}"):
                    rev = key.startswith(f"{dest}|")
                    hit = _TRANSFER_CACHE["direct"].get(key) or _TRANSFER_CACHE["transfers"].get(key)
                    if hit:
                        from_wp = hit["to"] if rev else hit["from"]
                        to_wp   = hit["from"] if rev else hit["to"]
                        if "leg1_route" in hit:
                            r1, r2 = (hit["leg2_route"], hit["leg1_route"]) if rev else (hit["leg1_route"], hit["leg2_route"])
                            xfr    = hit["transfer_at"]
                            journeys.append({
                                "hops": 2,
                                "legs": [
                                    {"route": r1, "from": from_wp, "to": xfr,
                                     "fare": hit["fare"] // 2,
                                     "windscreen": self._windscreen(r1, from_wp, xfr)},
                                    {"route": r2, "from": xfr, "to": to_wp,
                                     "fare": hit["fare"] - hit["fare"] // 2,
                                     "windscreen": self._windscreen(r2, xfr, to_wp)},
                                ],
                                "total_fare": hit["fare"],
                                "transfers":  [xfr],
                            })
                        else:
                            rid = hit["route"]
                            journeys.append({
                                "hops": 1,
                                "legs": [{"route": rid, "from": from_wp, "to": to_wp,
                                          "fare": hit["fare"],
                                          "windscreen": self._windscreen(rid, from_wp, to_wp)}],
                                "total_fare": hit["fare"],
                                "transfers":  [],
                            })
                        break

                r_from = self.wp_routes.get(origin, set())
                r_to   = self.wp_routes.get(dest,   set())

                # ── Direct ──────────────────────────────────────────────────
                for rid in r_from & r_to:
                    if rid.startswith("WALK-"):
                        continue
                    if not self._actual_wp(rid, origin) or not self._actual_wp(rid, dest):
                        continue
                    meta = self.route_meta[rid]
                    if meta["dist_km"] < 10.0:
                        short_leg_filtered = True
                        continue  # too short for a matatu — suggest walk or boda
                    fare = self._leg_fare(rid, origin, dest)
                    ws   = self._windscreen(rid, origin, dest)
                    journeys.append({
                        "hops":       1,
                        "legs":       [{"route": rid, "from": origin, "to": dest,
                                        "fare": fare, "windscreen": ws}],
                        "total_fare": fare,
                        "transfers":  [],
                    })

                # ── 1-transfer ───────────────────────────────────────────────
                # Find all waypoints reachable from origin that can also reach dest.
                # Normalise raw waypoints to their hub canonical name so that
                # "odeon" and "ambassadeur" collapse to the same transfer point.
                reachable_from_origin: dict[str, set[str]] = defaultdict(set)
                for rid in r_from:
                    if rid.startswith("WALK-"):
                        continue
                    for wp in self.route_wps.get(rid, []):
                        canonical = _HUB_OF.get(wp, wp)
                        reachable_from_origin[canonical].add(rid)

                for transfer_wp, r1_set in reachable_from_origin.items():
                    if transfer_wp in (origin, dest):
                        continue
                    r2_set = (self.wp_routes.get(transfer_wp, set()) & r_to) - {
                        rid for rid in self.wp_routes.get(transfer_wp, set()) if rid.startswith("WALK-")
                    }
                    if not r2_set:
                        continue

                    # Filter r1: route must actually contain both origin and transfer
                    valid_r1 = set()
                    for rid in r1_set:
                        if rid.startswith("WALK-"):
                            continue
                        if self._actual_wp(rid, origin) and self._actual_wp(rid, transfer_wp):
                            valid_r1.add(rid)

                    # Filter r2: route must actually contain both transfer and dest
                    valid_r2 = set()
                    for rid in r2_set:
                        if rid.startswith("WALK-"):
                            continue
                        if self._actual_wp(rid, transfer_wp) and self._actual_wp(rid, dest):
                            valid_r2.add(rid)

                    if not valid_r1 or not valid_r2:
                        continue

                    r1 = min(valid_r1, key=lambda r: self._leg_fare(r, origin, transfer_wp))
                    r2 = min(valid_r2, key=lambda r: self._leg_fare(r, transfer_wp, dest))

                    if r1 == r2:
                        continue

                    f1 = self._leg_fare(r1, origin, transfer_wp)
                    f2 = self._leg_fare(r2, transfer_wp, dest)
                    if f1 == 0 or f2 == 0:
                        continue

                    f1 = self._leg_fare(r1, origin, transfer_wp)
                    f2 = self._leg_fare(r2, transfer_wp, dest)

                    journeys.append({
                        "hops":       2,
                        "legs": [
                            {"route": r1, "from": origin,      "to": transfer_wp,
                             "fare": f1, "windscreen": self._windscreen(r1, origin, transfer_wp)},
                            {"route": r2, "from": transfer_wp, "to": dest,
                             "fare": f2, "windscreen": self._windscreen(r2, transfer_wp, dest)},
                        ],
                        "total_fare": f1 + f2,
                        "transfers":  [transfer_wp],
                    })

        # Deduplicate by route+leg signature
        seen: set[tuple] = set()
        unique: list[dict] = []
        for j in journeys:
            key = tuple((l["route"], l["from"], l["to"]) for l in j["legs"])
            if key not in seen:
                seen.add(key)
                unique.append(j)

        unique.sort(key=lambda j: (j["hops"], j["total_fare"]))

        # Drop 2-hop options when a direct route already covers the same pair
        has_direct = any(j["hops"] == 1 for j in unique)
        if has_direct:
            unique = [j for j in unique if j["hops"] == 1] + \
                     [j for j in unique if j["hops"] > 1][:2]

        # If no matatu journeys found, check why and inject a walk/boda suggestion.
        if not unique:
            all_origins = self._resolve(origin_raw)
            all_dests   = self._resolve(dest_raw)
            same_hub = (all_origins and all_dests and all_origins[0] == all_dests[0])
            if same_hub or short_leg_filtered:
                unique = [{"hops": 0, "legs": [], "total_fare": 0,
                           "transfers": [], "suggestion": "walk_or_boda"}]

        return unique[:4]

    # ── Format for AI ─────────────────────────────────────────────────────────

    @staticmethod
    def _display_wp(wp: str) -> str:
        """Convert internal waypoint name (including hub names) to readable label."""
        hub_labels = {
            "__cbd__":          "Town/CBD",
            "__westlands__":    "Westlands",
            "__ngong_road__":   "Ngong Road",
            "__pangani__":      "Pangani",
            "__roysambu__":     "Roysambu",
            "__bypass__":       "Bypass",
            "__kahawa_west__":  "Kahawa West",
            "__ku_referral__":  "KU Referral (KUTRRH)",
            "__donholm__":      "Donholm",
            "__thika__":        "Thika",
            "__rongai__":       "Ongata Rongai",
            "__pipeline__":     "Pipeline/Tassia",
            "__otc__":          "OTC",
            "__ngara__":        "Ngara",
            "__muthurwa__":     "Muthurwa",
            "__regen__":        "Two Rivers/Regen",
        }
        return hub_labels.get(wp, wp.title())

    @staticmethod
    def _is_road(wp: str) -> bool:
        return wp.lower() in ROAD_WAYPOINTS or wp.startswith("__")

    def _board_instruction(self, wp: str, windscreen: str) -> str:
        """
        Return the boarding instruction phrased correctly:
        - Road waypoint  → "get to [road] and flag down any mat showing '[ws]'"
        - Named hub      → "head to [hub area] and board any mat showing '[ws]'"
        - Named stop     → "board at [stop]"
        """
        label = self._display_wp(wp)
        wp_lc = wp.lower()
        if wp_lc in ROAD_WAYPOINTS:
            return f"get to {label} and flag down any mat showing '{windscreen}'"
        if wp.startswith("__"):
            return f"head to {label} — board any mat showing '{windscreen}'"
        return f"board at {label}"

    def _alight_instruction(self, wp: str) -> str:
        label = self._display_wp(wp)
        wp_lc = wp.lower()
        if wp_lc in ROAD_WAYPOINTS:
            return f"alight anywhere along {label}"
        return f"alight at {label}"

    def format_context(self, origin_raw: str, dest_raw: str, journeys: list[dict]) -> str:
        if not journeys:
            return ""

        # Walk/boda suggestion for short-distance journeys (under 10 km)
        if len(journeys) == 1 and journeys[0].get("suggestion") == "walk_or_boda":
            return (
                f"ROUTE OPTIONS: {origin_raw.title()} → {dest_raw.title()}\n"
                "This journey is under 10 km — no matatu needed.\n"
                "SUGGESTION: A bodaboda (motorcycle taxi) is the best option here. "
                "Typical fare 50–200 KSh depending on exact distance; negotiate before boarding. "
                "If it is genuinely walkable (under ~2 km, ~20 min), mention walking as an alternative.\n"
                "Do NOT recommend a matatu for this journey."
            )

        lines = [f"ROUTE OPTIONS: {origin_raw.title()} → {dest_raw.title()}"]

        for i, j in enumerate(journeys, 1):
            total  = j["total_fare"]
            n_legs = j["hops"]
            if n_legs == 1:
                tag = "Direct"
            else:
                transfers = " + ".join(self._display_wp(wp) for wp in j["transfers"])
                tag = f"Transfer at {transfers}"

            lines.append(f"\nOption {i} [{tag}] — ~{total} KSh:")
            for k, leg in enumerate(j["legs"], 1):
                arrow  = "●" if k == 1 else "↳"
                # Use the specific waypoint the route actually passes through
                # (e.g. "odeon" instead of vague "__cbd__") so the AI names the
                # right boarding stage and doesn't confuse nearby CBD sub-stops.
                actual_from = self._actual_wp(leg["route"], leg["from"]) or leg["from"]
                actual_to   = self._actual_wp(leg["route"], leg["to"])   or leg["to"]
                board  = self._board_instruction(actual_from, leg["windscreen"])
                alight = self._alight_instruction(actual_to)
                lines.append(
                    f"  {arrow} Route {leg['route']} — {board}, {alight} | ~{leg['fare']} KSh"
                )

        lines.append("\nUse this routing data to give natural, conversational directions.")
        return "\n".join(lines)

    # ── Origin/destination extractor ─────────────────────────────────────────

    def extract_od(self, message: str):
        """
        Parse origin and destination from a natural language transit query.
        Returns (origin_str, dest_str) or (None, None).
        """
        # Normalise punctuation before matching (handles backticks, curly quotes, etc.)
        msg = self._normalise(message)
        patterns = [
            r"(?:from|leaving?)\s+(.+?)\s+to\s+(.+?)(?:\?|$|\.|\s+using|\s+via|\s+by)",
            r"(.+?)\s+to\s+(.+?)(?:\?|$|\.|\s+route|\s+matatu|\s+bus|\s+fare)",
            r"get(?:ting)?\s+(?:from\s+)?(.+?)\s+to\s+(.+?)(?:\?|$|\.)",
            r"(?:go(?:ing)?|travel(?:ling)?|head(?:ing)?)\s+(?:from\s+)?(.+?)\s+to\s+(.+?)(?:\?|$|\.)",
            r"how\s+(?:do\s+i|can\s+i|to)\s+(?:get\s+)?(?:from\s+)?(.+?)\s+to\s+(.+?)(?:\?|$|\.)",
            r"directions?\s+(?:from\s+)?(.+?)\s+to\s+(.+?)(?:\?|$|\.)",
            r"(?:reach|arrive\s+at)\s+(.+?)\s+from\s+(.+?)(?:\?|$|\.)",
        ]

        for pat in patterns:
            m = re.search(pat, msg, re.IGNORECASE)
            if not m:
                continue
            a, b = m.group(1).strip(), m.group(2).strip()

            # Swap for "reach X from Y" pattern
            if "reach" in pat or "arrive" in pat:
                a, b = b, a

            # Validate both resolve to something in the graph
            if self._resolve(a) and self._resolve(b):
                return a, b

        return None, None
