import os, math, pandas as pd
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

df = pd.read_csv(os.path.join(os.path.dirname(__file__), "nairobi_transit_data.csv"))

def _haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1); dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2)
    return R * 2 * math.asin(math.sqrt(a))

def find_nearest_routes(lat, lng, n=4):
    tmp = df.copy()
    tmp["_d"] = tmp.apply(lambda r: _haversine(lat, lng, r["start_lat"], r["start_lon"]), axis=1)
    return tmp.nsmallest(n, "_d")[["route_name","route_long","headsign","fare_KSh","_d"]].to_dict("records")

# Test: Nairobi CBD (Railways area)
print("=== From Nairobi CBD (Railways) ===")
for r in find_nearest_routes(-1.2921, 36.8219):
    print(f"  Route {r['route_name']:6} | {r['route_long'][:38]:38} | {int(r['_d']*1000):4}m | {r['fare_KSh']} KSh")

# Test: Rongai area
print("\n=== From Ongata Rongai ===")
for r in find_nearest_routes(-1.3978, 36.7452):
    print(f"  Route {r['route_name']:6} | {r['route_long'][:38]:38} | {int(r['_d']*1000):4}m | {r['fare_KSh']} KSh")
