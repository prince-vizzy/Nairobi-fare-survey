"""
Apply fare survey responses to nairobi_transit_data.csv and routes_fares_coords.csv.

Usage:
  1. Download "Nairobi Fare Responses" Google Sheet as CSV
  2. Save it as fare_survey_responses.csv in this folder
  3. python apply_survey_fares.py

What it does:
  - Groups survey rows by Route ID, takes the MEDIAN fare (ignores outliers)
  - Requires at least MIN_RESPONSES responses per route before trusting the data
  - Updates fare_KSh and fare_source in both CSVs
  - Prints a before/after diff so you can review every change
"""

import pandas as pd
import os

BASE        = os.path.dirname(os.path.abspath(__file__))
SURVEY_FILE = os.path.join(BASE, "fare_survey_responses.csv")
TRANSIT     = os.path.join(BASE, "nairobi_transit_data.csv")
FARES_COORD = os.path.join(BASE, "routes_fares_coords.csv")

MIN_RESPONSES = 2   # need at least this many reports before updating a route

# ── Load survey ───────────────────────────────────────────────────────────────
if not os.path.exists(SURVEY_FILE):
    print(f"ERROR: {SURVEY_FILE} not found.")
    print("Download your Google Sheet as CSV and save it here as fare_survey_responses.csv")
    raise SystemExit(1)

survey = pd.read_csv(SURVEY_FILE)
print(f"Loaded {len(survey)} survey responses")
print(f"Columns found: {survey.columns.tolist()}\n")

# Normalise column names (Google Sheets export may vary casing/spacing)
survey.columns = [c.strip().lower().replace(" ", "_").replace("(ksh)", "").strip("_")
                  for c in survey.columns]

# Expected columns after normalisation: timestamp, route_id, route, fare_paid, reference_fare, direction
route_id_col  = next((c for c in survey.columns if "route_id" in c or "routeid" in c), None)
fare_paid_col = next((c for c in survey.columns if "fare_paid" in c or "farepaid" in c), None)

if not route_id_col or not fare_paid_col:
    print("ERROR: Could not find 'Route ID' or 'Fare Paid' columns in survey CSV.")
    print(f"Columns available: {survey.columns.tolist()}")
    raise SystemExit(1)

survey[fare_paid_col] = pd.to_numeric(survey[fare_paid_col], errors="coerce")
survey = survey.dropna(subset=[route_id_col, fare_paid_col])
survey[route_id_col] = survey[route_id_col].astype(str).str.strip()

# ── Aggregate: median fare per route, minimum response threshold ──────────────
agg = (
    survey.groupby(route_id_col)[fare_paid_col]
    .agg(median_fare="median", count="count")
    .reset_index()
)
agg = agg[agg["count"] >= MIN_RESPONSES].copy()
agg["median_fare"] = agg["median_fare"].round().astype(int)

print(f"Routes with >= {MIN_RESPONSES} responses: {len(agg)}")
print(f"{'Route ID':<12} {'Responses':<12} {'Survey Median (KSh)'}")
print("-" * 45)
for _, row in agg.iterrows():
    print(f"  {row[route_id_col]:<12} {int(row['count']):<12} {int(row['median_fare'])}")
print()


def apply_updates(csv_path, route_col="route_name"):
    """Read a CSV, apply survey fares, write back, return change summary."""
    df = pd.read_csv(csv_path, dtype={route_col: str})
    df[route_col] = df[route_col].str.strip()

    changes = []
    for _, agg_row in agg.iterrows():
        rid       = str(agg_row[route_id_col]).strip()
        new_fare  = int(agg_row["median_fare"])
        n         = int(agg_row["count"])
        mask      = df[route_col] == rid

        if mask.sum() == 0:
            changes.append({"route": rid, "status": "NOT FOUND in CSV", "old": "-", "new": new_fare})
            continue

        old_fare = int(df.loc[mask, "fare_KSh"].iloc[0])
        if old_fare == new_fare:
            changes.append({"route": rid, "status": "unchanged", "old": old_fare, "new": new_fare})
        else:
            df.loc[mask, "fare_KSh"]    = new_fare
            df.loc[mask, "fare_source"] = f"survey ({n} responses)"
            changes.append({"route": rid, "status": "UPDATED", "old": old_fare, "new": new_fare})

    df.to_csv(csv_path, index=False)
    return changes, df


# ── Apply to both files ───────────────────────────────────────────────────────
print("=" * 55)
print(f"Updating {os.path.basename(TRANSIT)}")
print("=" * 55)
changes1, _ = apply_updates(TRANSIT, route_col="route_name")
updated = [c for c in changes1 if c["status"] == "UPDATED"]
same    = [c for c in changes1 if c["status"] == "unchanged"]
missing = [c for c in changes1 if "NOT FOUND" in c["status"]]

for c in changes1:
    if c["status"] == "UPDATED":
        print(f"  UPDATED  {c['route']:<10}  {c['old']} → {c['new']} KSh")
    elif c["status"] == "unchanged":
        print(f"  same     {c['route']:<10}  {c['old']} KSh (survey agrees)")
    else:
        print(f"  missing  {c['route']:<10}  not found in CSV")

print(f"\n  {len(updated)} routes updated | {len(same)} unchanged | {len(missing)} not found")

print()
print("=" * 55)
print(f"Updating {os.path.basename(FARES_COORD)}")
print("=" * 55)
changes2, _ = apply_updates(FARES_COORD, route_col="route_name")
updated2 = [c for c in changes2 if c["status"] == "UPDATED"]

for c in changes2:
    if c["status"] == "UPDATED":
        print(f"  UPDATED  {c['route']:<10}  {c['old']} → {c['new']} KSh")

print(f"\n  {len(updated2)} rows updated in {os.path.basename(FARES_COORD)}")
print()
print("Done. Restart app.py to pick up the new fares.")
