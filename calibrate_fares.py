"""
calibrate_fares.py
Compares old fares vs survey, fits a corridor-aware fare model,
calculates adjustments for every route, handles direction asymmetry,
and writes updated CSVs.

Run:  python calibrate_fares.py
"""

import pandas as pd
import numpy as np
import os, sys

sys.stdout.reconfigure(encoding="utf-8")

BASE        = os.path.dirname(os.path.abspath(__file__))
SURVEY_FILE = os.path.join(BASE, "Fare - Responses.csv")
COORDS_FILE = os.path.join(BASE, "routes_fares_coords.csv")
TRANSIT     = os.path.join(BASE, "nairobi_transit_data.csv")
CLEAN_FILE  = os.path.join(BASE, "routes_clean.csv")

NAIROBI_BANDS = [40, 50, 60, 70, 80, 100, 120, 150, 200]   # valid fare steps

# ── 1. Load data ──────────────────────────────────────────────────────────────
survey = pd.read_csv(SURVEY_FILE)
survey["Fare Paid (KSh)"] = pd.to_numeric(survey["Fare Paid (KSh)"], errors="coerce")
survey["Direction"] = survey["Direction"].str.replace("→", "->", regex=False)

coords = pd.read_csv(COORDS_FILE, dtype={"route_name": str})
clean  = pd.read_csv(CLEAN_FILE,  dtype={"id": str})
transit = pd.read_csv(TRANSIT,    dtype={"route_name": str})

display_to_id = dict(zip(clean["display"], clean["id"]))
survey["route_id"] = survey["Route"].map(display_to_id)

# ── 2. Survey medians per route per direction ─────────────────────────────────
# direction flag: 0 = away from CBD (forward), 1 = toward CBD (reverse)
def infer_direction(route_str, direction_str):
    """0 = forward (CBD/origin → dest), 1 = reverse."""
    d = str(direction_str).lower()
    r = str(route_str).lower()
    parts = r.split(" - ")
    if len(parts) == 2 and any(cbd in parts[1] for cbd in ["cbd", "town", "odeon", "koja"]):
        return 1
    if any(cbd in d.split("->")[1] if "->" in d else "" for cbd in ["cbd", "town"]):
        return 1
    return 0

survey["dir_flag"] = survey.apply(
    lambda r: infer_direction(r["Route"], r["Direction"]), axis=1
)

survey_agg = (
    survey.groupby(["route_id", "dir_flag"])["Fare Paid (KSh)"]
    .agg(survey_fare="median", n="count")
    .reset_index()
    .dropna(subset=["route_id"])
)

# Build lookup: route_id -> {0: fare, 1: fare}
survey_fares: dict[str, dict] = {}
for _, row in survey_agg.iterrows():
    rid = str(row["route_id"])
    if rid not in survey_fares:
        survey_fares[rid] = {}
    survey_fares[rid][int(row["dir_flag"])] = (int(row["survey_fare"]), int(row["n"]))

# ── 3. Delta table: old vs survey ────────────────────────────────────────────
print("=" * 75)
print("OLD vs SURVEY FARES")
print("=" * 75)
print(f"{'ID':<8} {'Route':<32} {'km':>5} {'Old':>5} {'Fwd':>5} {'Rev':>5} {'Δ Fwd':>6}  Notes")
print("-" * 75)

deltas = []
for rid, dir_fares in sorted(survey_fares.items()):
    row0 = coords[(coords.route_name == rid) & (coords.direction == 0)]
    if row0.empty:
        continue
    old_fare = int(row0.iloc[0]["fare_KSh"])
    dist     = row0.iloc[0]["dist_km"]
    disp     = clean[clean.id == rid]["display"].values
    disp_str = disp[0] if len(disp) else rid

    fwd_fare, fwd_n = dir_fares.get(0, (None, 0))
    rev_fare, rev_n = dir_fares.get(1, (None, 0))

    delta_fwd = (fwd_fare - old_fare) if fwd_fare else None
    asym = " ASYM" if (fwd_fare and rev_fare and fwd_fare != rev_fare) else ""
    flag = " <<<" if delta_fwd and abs(delta_fwd) >= 50 else (" <<" if delta_fwd and abs(delta_fwd) >= 20 else "")

    fwd_str = str(fwd_fare) if fwd_fare else "  -"
    rev_str = str(rev_fare) if rev_fare else "  -"
    dlt_str = f"{delta_fwd:+d}" if delta_fwd is not None else "  -"

    print(f"{rid:<8} {disp_str:<32} {dist:>5.1f} {old_fare:>5} {fwd_str:>5} {rev_str:>5} {dlt_str:>6}  {flag}{asym}")
    deltas.append(dict(route_id=rid, old=old_fare, fwd=fwd_fare, rev=rev_fare, delta=delta_fwd, dist=dist))

# ── 4. Corridor-aware fare model ──────────────────────────────────────────────
# Calibrated from survey + previously verified fares (viraltea, nairobilifestyle, kenyan-post)
#
# The survey revealed: most "80 KSh area estimate" routes are actually 50 KSh.
# Key corridors override distance-only logic.

CORRIDOR_RULES = [
    # (keywords_in_route_long,          fare,  label)
    (["kiserian"],                        200, "long suburb"),
    (["limuru"],                          200, "long suburb"),
    (["thika"],                           150, "long corridor"),
    (["ruiru", "juja"],                   150, "long corridor"),
    (["kiambu", "githunguri", "ndumberi"],150, "far north suburb"),
    (["ongata rongai", "rongai"],         100, "rongai corridor (survey)"),
    (["kitengela", "mlolongo", "athi"],   150, "mombasa rd long"),
    (["jkia", "airport"],                 150, "airport corridor (survey)"),
    (["karen", "hardy"],                  100, "karen corridor (survey)"),
    (["ngong road", "ngong"],             150, "ngong road premium"),
    (["langata", "bomas"],                 50, "langata short (survey)"),
    (["westlands", "lavington", "kileleshwa", "highridge", "hurlingham", "yaya"], 50, "westlands belt"),
    (["githurai", "roysambu"],             70, "githurai corridor (verified)"),
    (["ruaka", "ndenderu", "wangige", "kabete", "kinoo"], 100, "ruaka belt"),
    (["kikuyu", "dagoretti", "uthiru"],   100, "dagoretti belt"),
    (["kawangware"],                       100, "kawangware"),
    (["kibera"],                           100, "kibera"),
    (["kangemi"],                           50, "kangemi (survey)"),
    (["kasarani", "mwiki", "sunton"],      80, "kasarani belt"),
    (["dandora", "kayole", "saika", "ruai"], 80, "eastlands belt"),
    (["umoja", "donholm", "buruburu", "jacaranda"], 80, "southeast belt"),
    (["utawala", "embakasi", "pipeline", "imara"], 80, "embakasi belt"),
    (["komarocks", "kariobangi"],           80, "kariobangi belt"),
    (["huruma", "mathare"],                 50, "huruma (survey)"),
    (["rounda", "kariobangi south"],        50, "rounda/kario south (survey)"),
    (["eastleigh"],                         50, "eastleigh (survey)"),
    (["pangani", "ngara", "kariokor", "gikomba"], 50, "inner corridor (survey)"),
]

def snap_to_band(fare):
    return min(NAIROBI_BANDS, key=lambda b: abs(b - fare))

def model_fare(route_long, headsign, dist_km):
    """Corridor-first fare model, distance as tiebreaker."""
    combined = (str(route_long) + " " + str(headsign)).lower()
    for keywords, fare, _ in CORRIDOR_RULES:
        if any(kw in combined for kw in keywords):
            return fare
    # Distance fallback (calibrated on combined dataset)
    if dist_km < 3:   return 40
    if dist_km < 6:   return 50
    if dist_km < 10:  return 80
    if dist_km < 16:  return 100
    if dist_km < 25:  return 150
    return 200

# ── 5. Compute final fares for all routes ─────────────────────────────────────
print()
print("=" * 85)
print("FULL ROUTE FARE ADJUSTMENTS")
print("=" * 85)
print(f"{'ID':<8} {'Route':<36} {'km':>5} {'Old':>5} {'Fwd':>5} {'Rev':>5} {'Action':<22} Source")
print("-" * 85)

updates_fwd = {}   # route_id -> new forward fare
updates_rev = {}   # route_id -> new reverse fare
update_src  = {}   # route_id -> source label

fwd_routes = coords[coords.direction == 0].copy()

for _, row in fwd_routes.iterrows():
    rid      = str(row["route_name"])
    old_fare = int(row["fare_KSh"])
    dist     = row["dist_km"]
    rl       = row["route_long"]
    hs       = row["headsign"]
    disp     = clean[clean.id == rid]["display"].values
    disp_str = disp[0][:36] if len(disp) else f"{rl}"[:36]

    # Determine new fwd fare
    if rid in survey_fares and 0 in survey_fares[rid]:
        new_fwd, n = survey_fares[rid][0]
        src_fwd = f"survey ({n} resp)"
    elif rid in survey_fares and 1 in survey_fares[rid]:
        # Only reverse surveyed — use model for forward
        new_fwd = model_fare(rl, hs, dist)
        src_fwd = "model (rev-only survey)"
    elif "estimate" not in str(row["fare_source"]):
        new_fwd = old_fare   # keep verified fare
        src_fwd = row["fare_source"]
    else:
        new_fwd = model_fare(rl, hs, dist)
        src_fwd = "model (calibrated)"

    # Determine new rev fare
    rev_row = coords[(coords.route_name == rid) & (coords.direction == 1)]
    old_rev = int(rev_row.iloc[0]["fare_KSh"]) if not rev_row.empty else old_fare

    if rid in survey_fares and 1 in survey_fares[rid]:
        new_rev, n_r = survey_fares[rid][1]
        src_rev = f"survey ({n_r} resp)"
    else:
        new_rev = new_fwd   # symmetric unless survey says otherwise
        src_rev = src_fwd

    updates_fwd[rid] = new_fwd
    updates_rev[rid] = new_rev
    update_src[rid]  = src_fwd

    action = "KEEP" if (new_fwd == old_fare and new_rev == old_rev) else "UPDATE"
    asym_tag = " [ASYM]" if new_fwd != new_rev else ""
    print(f"{rid:<8} {disp_str:<36} {dist:>5.1f} {old_fare:>5} {new_fwd:>5} {new_rev:>5} {action+asym_tag:<22} {src_fwd}")

# ── 6. Write updated CSVs ─────────────────────────────────────────────────────
def update_coords_csv(df):
    df = df.copy()
    for i, row in df.iterrows():
        rid = str(row["route_name"])
        d   = int(row["direction"])
        if d == 0 and rid in updates_fwd:
            df.at[i, "fare_KSh"]    = updates_fwd[rid]
            df.at[i, "fare_source"] = update_src.get(rid, row["fare_source"])
        elif d == 1 and rid in updates_rev:
            df.at[i, "fare_KSh"]    = updates_rev[rid]
            df.at[i, "fare_source"] = update_src.get(rid, row["fare_source"])
    return df

new_coords = update_coords_csv(coords)
new_coords.to_csv(COORDS_FILE, index=False)

# Update transit data (direction 0 only — it has one row per route)
new_transit = transit.copy()
for i, row in new_transit.iterrows():
    rid = str(row["route_name"])
    if rid in updates_fwd:
        new_transit.at[i, "fare_KSh"]    = updates_fwd[rid]
        new_transit.at[i, "fare_source"] = update_src.get(rid, row["fare_source"])
new_transit.to_csv(TRANSIT, index=False)

# ── 7. Summary ────────────────────────────────────────────────────────────────
changed_fwd = sum(1 for rid in updates_fwd
                  if updates_fwd[rid] != int(coords[(coords.route_name==rid)&(coords.direction==0)]["fare_KSh"].values[0])
                  if not coords[(coords.route_name==rid)&(coords.direction==0)].empty)
asym_routes = [rid for rid in updates_fwd if updates_fwd[rid] != updates_rev.get(rid, updates_fwd[rid])]

print()
print("=" * 55)
print("SUMMARY")
print("=" * 55)
print(f"  Routes updated (forward)    : {changed_fwd}")
print(f"  Routes with direction asymmetry : {len(asym_routes)}")
if asym_routes:
    for rid in asym_routes:
        print(f"    {rid}: fwd={updates_fwd[rid]} KSh  rev={updates_rev[rid]} KSh")
print()
print("  Files written:")
print(f"    {COORDS_FILE}")
print(f"    {TRANSIT}")
print()

# ── 8. Custom routes not in dataset ──────────────────────────────────────────
unmatched = survey[survey.route_id.isna()][["Route","Fare Paid (KSh)","Direction"]].drop_duplicates()
print("=" * 55)
print("CUSTOM ROUTES — add these to the dataset")
print("=" * 55)
for _, r in unmatched.iterrows():
    print(f"  {r['Route']:<40} {int(r['Fare Paid (KSh)']):>5} KSh  |  {r['Direction']}")
print()
print("Done — restart app.py to pick up new fares.")
