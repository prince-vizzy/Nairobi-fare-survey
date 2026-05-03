import struct, csv, math
from collections import Counter

def read_dbf(path):
    with open(path, 'rb') as f:
        f.read(4)
        num_records = struct.unpack('<I', f.read(4))[0]
        header_size = struct.unpack('<H', f.read(2))[0]
        record_size = struct.unpack('<H', f.read(2))[0]
        f.seek(32)
        fields = []
        while True:
            fd = f.read(32)
            if fd[0] == 0x0D:
                break
            name = fd[:11].decode('latin-1').rstrip('\x00')
            length = fd[16]
            fields.append((name, length))
        f.seek(header_size)
        records = []
        for _ in range(num_records):
            rec = f.read(record_size)
            if rec[0] == 0x2A:
                continue
            offset = 1
            row = {}
            for name, length in fields:
                row[name] = rec[offset:offset+length].decode('latin-1').strip()
                offset += length
            records.append(row)
        return records

def read_shp_polylines(path):
    results = {}
    with open(path, 'rb') as f:
        f.seek(100)
        while True:
            hdr = f.read(8)
            if len(hdr) < 8:
                break
            rec_num     = struct.unpack('>I', hdr[:4])[0]
            content_len = struct.unpack('>I', hdr[4:])[0]
            content     = f.read(content_len * 2)
            shape_type  = struct.unpack('<i', content[:4])[0]
            if shape_type != 3:
                continue
            offset = 4 + 32
            num_parts  = struct.unpack('<i', content[offset:offset+4])[0]; offset += 4
            num_points = struct.unpack('<i', content[offset:offset+4])[0]; offset += 4
            offset += num_parts * 4
            first_x, first_y = struct.unpack('<dd', content[offset:offset+16])
            last_offset = offset + (num_points - 1) * 16
            last_x, last_y   = struct.unpack('<dd', content[last_offset:last_offset+16])
            results[rec_num] = (first_x, first_y, last_x, last_y)
    return results

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))

# Verified fares from web sources
verified = {
    '46':   (30,  'nairobilifestyle.com'),
    '46H':  (80,  'area estimate - Huruma corridor'),
    '46B':  (80,  'area estimate - Huruma corridor'),
    '46K':  (80,  'area estimate - Kawangware corridor'),
    '46Y':  (50,  'nairobilifestyle.com - Yaya short'),
    '046P': (80,  'area estimate - Kawangware'),
    '32A':  (50,  'nairobilifestyle.com (32C ref)'),
    '108':  (45,  'nairobilifestyle.com (avg 30-60 KSh)'),
    '107':  (70,  'nairobilifestyle.com'),
    '107D': (120, 'area estimate - Ruiru branch'),
    '11A':  (40,  'nairobilifestyle.com'),
    '11B':  (70,  'area estimate - Ruaka'),
    '11C':  (40,  'nairobilifestyle.com'),
    '44G':  (70,  'nairobilifestyle.com (avg 60-80)'),
    '44Z':  (70,  'nairobilifestyle.com (avg 60-80)'),
    '44K':  (70,  'nairobilifestyle.com (avg 60-80)'),
    '45K':  (70,  'nairobilifestyle.com'),
    '45G':  (70,  'nairobilifestyle.com'),
    '45P':  (80,  'area estimate - Githurai branch'),
    '58':   (80,  'area estimate - Buruburu'),
}

# Area-based rules: (keywords, fare_KSh, source). First match wins.
area_rules = [
    (['kiserian'],                                         200, 'area estimate'),
    (['limuru'],                                           200, 'area estimate'),
    (['ongata rongai', 'rongai'],                          150, 'viraltea.co.ke'),
    (['kitengela', 'mlolongo'],                            150, 'area estimate'),
    (['ngong'],                                            150, 'viraltea.co.ke'),
    (['ruiru', 'juja', 'thika'],                           150, 'area estimate'),
    (['kiambu', 'githunguri', 'ndumberi'],                 120, 'area estimate'),
    (['karen', 'hardy', 'langata'],                        120, 'area estimate'),
    (['ruaka', 'ndenderu', 'wangige', 'kabete', 'kinoo'],  100, 'area estimate'),
    (['kikuyu', 'dagoretti', 'uthiru'],                    100, 'area estimate'),
    (['kawangware', 'kibera'],                             100, 'kenyan-post.com'),
    (['mathare', 'huruma', 'kariobangi', 'komarocks'],      80, 'kenyan-post.com'),
    (['kangemi'],                                           80, 'area estimate'),
    (['kasarani', 'mwiki', 'sunton'],                       80, 'area estimate'),
    (['dandora', 'kayole', 'saika', 'ruai'],                80, 'area estimate'),
    (['umoja', 'donholm', 'buruburu', 'jacaranda'],         80, 'area estimate'),
    (['utawala', 'embakasi', 'jkia', 'pipeline'],           80, 'area estimate'),
    (['imara daima', 'south b', 'hazina', 'industrial'],    80, 'area estimate'),
    (['ngumo', 'mbagathi', 'nyayo', 'wilson'],              80, 'area estimate'),
    (['eastleigh'],                                         60, 'area estimate'),
    (['pangani', 'ngara', 'kariokor', 'gikomba'],           60, 'area estimate'),
    (['githurai', 'roysambu'],                              70, 'nairobilifestyle.com'),
    (['westlands', 'lavington', 'kileleshwa', 'highridge', 'hurlingham', 'yaya'], 50, 'area estimate'),
    (['muthurwa', 'makadara', 'maringo', 'likoni', 'lunga'],60, 'area estimate'),
]

dbf_records = read_dbf(r'c:/Users/HP/Downloads/GIS_DATA_2019/shapes.dbf')
shp_geom    = read_shp_polylines(r'c:/Users/HP/Downloads/GIS_DATA_2019/shapes.shp')

out_rows = []
for i, rec in enumerate(dbf_records, start=1):
    geom = shp_geom.get(i)
    if not geom:
        continue
    s_lon, s_lat, e_lon, e_lat = geom
    dist_km = haversine_km(s_lat, s_lon, e_lat, e_lon)

    rn       = rec['route_name']
    combined = (rec['route_long'] + ' ' + rec['headsign']).lower()

    fare, source = None, None

    if rn in verified:
        fare, source = verified[rn]

    if fare is None:
        for keywords, f, src in area_rules:
            if any(kw in combined for kw in keywords):
                fare, source = f, src
                break

    if fare is None:
        if dist_km < 8:
            fare, source = 50,  'distance estimate (<8 km)'
        elif dist_km < 15:
            fare, source = 80,  'distance estimate (8-15 km)'
        elif dist_km < 25:
            fare, source = 100, 'distance estimate (15-25 km)'
        else:
            fare, source = 150, 'distance estimate (>25 km)'

    out_rows.append({
        'route_name':  rn,
        'route_long':  rec['route_long'],
        'headsign':    rec['headsign'],
        'direction':   rec['direction'],
        'start_lat':   round(s_lat, 6),
        'start_lon':   round(s_lon, 6),
        'end_lat':     round(e_lat, 6),
        'end_lon':     round(e_lon, 6),
        'dist_km':     round(dist_km, 2),
        'fare_KSh':    fare,
        'fare_source': source,
    })

out = r'c:/Users/HP/Downloads/GIS_DATA_2019/routes_fares_coords.csv'
fields = ['route_name','route_long','headsign','direction',
          'start_lat','start_lon','end_lat','end_lon','dist_km','fare_KSh','fare_source']
with open(out, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(out_rows)

print(f'Saved {len(out_rows)} rows -> {out}')
print()
print(f"{'Route':<8} {'Route Long':<42} {'Dir':<4} {'km':<7} {'KSh':<6} Source")
print('-'*120)
shown = set()
for r in out_rows:
    key = r['route_name']
    if r['direction'] == '0' and key not in shown:
        shown.add(key)
        print(f"{r['route_name']:<8} {r['route_long']:<42} {r['direction']:<4} {r['dist_km']:<7} {r['fare_KSh']:<6} {r['fare_source']}")

print()
print('Fare source breakdown:')
src_counts = Counter(r['fare_source'] for r in out_rows)
for src, count in sorted(src_counts.items(), key=lambda x: -x[1]):
    print(f'  {count:3}  {src}')
