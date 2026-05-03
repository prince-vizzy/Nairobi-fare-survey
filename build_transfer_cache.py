"""
Pre-compute all direct and transfer routes between every terminal waypoint pair
and save to transfer_cache.json.
"""
import pandas as pd
import json
from router import NairobiRouter, _HUB_OF, ROAD_WAYPOINTS

df       = pd.read_csv("nairobi_transit_data.csv")
stops_df = pd.read_csv("nairobi_stops.csv")
stops_df["trip_count"] = pd.to_numeric(stops_df["trip_count"], errors="coerce").fillna(0)
r = NairobiRouter(df, stops_df)

# Collect all terminal waypoints normalised to hub canonical names
terminals = set()
for wps in r.route_wps.values():
    if wps:
        terminals.add(_HUB_OF.get(wps[0],  wps[0]))
        terminals.add(_HUB_OF.get(wps[-1], wps[-1]))

# Remove road corridors and malformed entries
skip = ROAD_WAYPOINTS | {
    "bypass ruiru, city cabanas",
    "ambassadeur,kabete,kawangware",
    "carbanas",
}
terminals = sorted(t for t in terminals if t not in skip and "road" not in t)

total_pairs = len(terminals) * (len(terminals) - 1) // 2
print(f"Nodes: {len(terminals)}  Pairs: {total_pairs}")

direct_cache = {}
xfer_cache   = {}

for i, o in enumerate(terminals):
    for d in terminals[i + 1:]:
        journeys = r.find_journeys(o, d)
        if not journeys:
            continue
        key     = f"{o}|{d}"
        directs = [j for j in journeys if j["hops"] == 1]
        xfers   = [j for j in journeys if j["hops"] == 2]

        if directs:
            best = min(directs, key=lambda j: j["total_fare"])
            direct_cache[key] = {
                "from":  o,
                "to":    d,
                "route": best["legs"][0]["route"],
                "fare":  best["total_fare"],
            }

        if xfers:
            best = min(xfers, key=lambda j: j["total_fare"])
            xfer_cache[key] = {
                "from":        o,
                "to":          d,
                "leg1_route":  best["legs"][0]["route"],
                "leg2_route":  best["legs"][1]["route"],
                "transfer_at": best["transfers"][0] if best["transfers"] else "",
                "fare":        best["total_fare"],
            }

    if i % 20 == 0:
        print(f"  {i}/{len(terminals)} nodes processed ...")

out = {"direct": direct_cache, "transfers": xfer_cache}
with open("transfer_cache.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)

print(f"Direct cached:   {len(direct_cache)}")
print(f"Transfer cached: {len(xfer_cache)}")
print("Saved to transfer_cache.json")
