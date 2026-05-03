import pandas as pd
df = pd.read_csv('nairobi_stops.csv')
print(df.head(8).to_string())
linked = df[df['routes'].notna() & (df['routes'] != '')].shape[0]
print(f"\n{len(df)} total stops — {linked} linked to routes")
print("\nTop 15 busiest stops:")
busy = df[df['trip_count'] != '0'].copy()
busy['trip_count'] = pd.to_numeric(busy['trip_count'], errors='coerce').fillna(0)
print(busy.nlargest(15, 'trip_count')[['stop_name','stop_lat','stop_lon','trip_count','routes']].to_string())
