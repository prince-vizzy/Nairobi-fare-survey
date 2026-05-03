"""
Parse stops.dbf -> nairobi_stops.csv
Columns: stop_id, stop_name, stop_lat, stop_lon, trip_count, routes
"""
import struct, csv, os

BASE = os.path.dirname(os.path.abspath(__file__))

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
            name   = fd[:11].decode('latin-1').rstrip('\x00')
            length = fd[16]
            fields.append((name, length))
        f.seek(header_size)
        records = []
        for _ in range(num_records):
            rec = f.read(record_size)
            if rec[0] == 0x2A:
                continue
            offset, row = 1, {}
            for name, length in fields:
                row[name] = rec[offset:offset+length].decode('latin-1').strip()
                offset += length
            records.append(row)
    return records

records = read_dbf(os.path.join(BASE, 'stops.dbf'))

out = os.path.join(BASE, 'nairobi_stops.csv')
with open(out, 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['stop_id', 'stop_name', 'stop_lat', 'stop_lon', 'trip_count', 'routes'])
    for r in records:
        # deduplicate space-separated route names
        raw_routes = r.get('route_nams', '')
        routes = ' '.join(dict.fromkeys(raw_routes.split())) if raw_routes else ''
        try:
            lat = float(r.get('stop_lat', '') or 0)
            lon = float(r.get('stop_lon', '') or 0)
        except ValueError:
            continue
        if lat == 0 and lon == 0:
            continue
        w.writerow([
            r.get('stop_id', ''),
            r.get('stop_name', ''),
            round(lat, 6),
            round(lon, 6),
            r.get('trip_count', '0'),
            routes,
        ])

print(f'Saved {sum(1 for _ in open(out)) - 1} stops -> {out}')
