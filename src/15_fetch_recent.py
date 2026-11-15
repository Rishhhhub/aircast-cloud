"""Download the Vayuayan archive (CPCB live data, hourly-ish snapshots from ~Dec 2025 to now) for every station and build a recent daily station table.
Daily AQI is computed the CPCB way: AQI = max of the pollutant sub-indices (needs PM2.5 or PM10 and at least 3 pollutants); daily value = mean of the day's snapshots.
Outputs: data/raw/vayuayan/*.csv.gz, data/external/live_stations.csv (coordinates), data/processed/recent_station_day.parquet"""
import requests, json, os, gzip, io, time, numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor
B = 'https://data.vayuayan.saketlab.org'; OUT = 'data/raw/vayuayan'; os.makedirs(OUT, exist_ok=True)
H = {'User-Agent': 'AirCast-student-project/1.0'}
st = requests.get(f'{B}/stations.json', headers=H, timeout=60).json()
def get(s):
    f = f"{OUT}/{s['filename']}"
    if os.path.exists(f) and os.path.getsize(f) > 500 and time.time() - os.path.getmtime(f) < 3 * 3600: return f   # re-download anything older than 3 h (the archive grows every hour)
    for t in range(4):
        try:
            r = requests.get(s['url'], headers=H, timeout=90)
            if r.status_code == 200 and r.content[:2] == b'\x1f\x8b': open(f, 'wb').write(r.content); return f
            if r.status_code == 404: return None
        except Exception: pass
        time.sleep(3 * (t + 1))
    return None
with ThreadPoolExecutor(8) as ex: files = list(ex.map(get, st))
print('downloaded', sum(f is not None for f in files), 'of', len(st), flush=True)
geo = requests.get(f'{B}/stations.geojson', headers=H, timeout=90).json()['features']
pd.DataFrame([{'station_id': g['properties']['station_id'], 'station_name': g['properties']['station_name'], 'city': g['properties']['city'], 'state': g['properties']['state'],
               'lon': g['geometry']['coordinates'][0], 'lat': g['geometry']['coordinates'][1], 'is_live': g['properties'].get('is_live')} for g in geo]).to_csv('data/external/live_stations.csv', index=False)
SUB = ['PM2.5_subindex', 'PM10_subindex', 'SO2_subindex', 'NO2_subindex', 'OZONE_subindex', 'CO_subindex', 'NH3_subindex']
POL = ['PM2.5_avg', 'PM10_avg', 'NO2_avg', 'SO2_avg', 'CO_avg', 'OZONE_avg', 'NH3_avg']
rows = []
for f in files:
    if not f: continue
    try: d = pd.read_csv(f, compression='gzip', na_values=['-', ''])
    except Exception: continue
    if d.empty or 'timestamp' not in d: continue
    d['ts'] = pd.to_datetime(d.timestamp, errors='coerce'); d = d.dropna(subset=['ts'])
    sub = d[[c for c in SUB if c in d]].apply(pd.to_numeric, errors='coerce')
    n_ok = sub.notna().sum(axis=1); has_pm = sub[[c for c in ['PM2.5_subindex', 'PM10_subindex'] if c in sub]].notna().any(axis=1)
    d['aqi'] = sub.max(axis=1).where((n_ok >= 3) & has_pm)
    for c in POL:
        if c in d: d[c] = pd.to_numeric(d[c], errors='coerce')
    d['date'] = d.ts.dt.normalize()
    g = d.groupby('date').agg(aqi_mean=('aqi', 'mean'), aqi_max=('aqi', 'max'), aqi_min=('aqi', 'min'), aqi_std=('aqi', 'std'), n=('aqi', 'count'), **{c.replace('_avg', '').replace('.', ''): (c, 'mean') for c in POL if c in d}).reset_index()
    g['station_id'] = d.station_id.iloc[0]; g['station_name'] = d.station_name.iloc[0]; g['city'] = d.city.iloc[0]; g['state'] = d.state.iloc[0]
    rows.append(g)
R = pd.concat(rows); R = R[R.n >= 8].dropna(subset=['aqi_mean']); R.to_parquet('data/processed/recent_station_day.parquet')
print(R.shape, R.station_id.nunique(), 'stations', R.date.min().date(), '->', R.date.max().date())
