"""Estimate the AQI at ANY point (also where there is no station) from the surrounding stations: the deployable version of 21_wind_aware.py.
Context(source, date) holds the stations (with AQI today and yesterday) and the weather of that day; estimate(lat, lon, hide_km) returns the model estimate plus the
baselines (nearest station, inverse-distance) and how far the nearest used station is.
source = 'live' (2026 data: Vayuayan archive + Open-Meteo forecast API) or 'hist' (2017-2023 CPCB data + archived weather)."""
import os, json
import numpy as np, pandas as pd, joblib
from wind_features import pair_geometry, wind_features

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = lambda *a: os.path.join(ROOT, *a)
DIWALI = pd.to_datetime(['2016-10-30', '2017-10-19', '2018-11-07', '2019-10-27', '2020-11-14', '2021-11-04', '2022-10-24', '2023-11-12', '2024-11-01', '2025-10-20', '2026-11-08'])
WXCOLS = ['temperature_2m_mean', 'precipitation_sum', 'wind_direction_10m_dominant', 'pressure_msl_mean', 'cloud_cover_mean', 'relative_humidity_2m_mean', 'dew_point_2m_mean', 'shortwave_radiation_sum', 'wind_speed_10m_max']
_model = None
_elev_cache = {}


def model():
    global _model
    if _model is None:
        _model = joblib.load(P('models/wind_aware.joblib'))
    return _model


def elevation(lat, lon):
    """Open-Meteo elevation API with a rounding cache; falls back to 200 m."""
    import requests
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    keys = [(round(a, 2), round(b, 2)) for a, b in zip(lat, lon)]
    todo = sorted({k for k in keys if k not in _elev_cache})
    for i in range(0, len(todo), 100):
        ch = todo[i:i + 100]
        try:
            r = requests.get('https://api.open-meteo.com/v1/elevation', params={'latitude': ','.join(f'{a:.2f}' for a, _ in ch), 'longitude': ','.join(f'{b:.2f}' for _, b in ch)}, timeout=30).json()['elevation']
            for k, e in zip(ch, r):
                _elev_cache[k] = float(e)
        except Exception:
            for k in ch:
                _elev_cache[k] = 200.0
    return np.array([_elev_cache.get(k, 200.0) for k in keys])


class Context:
    def __init__(self, source, date):
        self.source, self.date = source, pd.Timestamp(date).normalize()
        d0, d1 = self.date, self.date - pd.Timedelta(days=1)
        if source == 'live':
            R = pd.read_parquet(P('data/processed/recent_station_day.parquet')); L = pd.read_csv(P('data/external/live_stations.csv'))
            R = R.merge(L[['station_id', 'lat', 'lon']], on='station_id').dropna(subset=['lat', 'lon'])
            R = R.groupby(['station_id', 'date'], as_index=False).agg(aqi=('aqi_mean', 'mean'), name=('station_name', 'first'), city=('city', 'first'), lat=('lat', 'first'), lon=('lon', 'first'))
            key = 'station_id'
            W = pd.read_parquet(P('data/processed/weather_live.parquet'))
            arc = P('data/external/weather_recent_archive.parquet')
            if os.path.exists(arc):   # archive is authoritative for the past (the forecast API's older past days can be empty)
                a = pd.read_parquet(arc).dropna(subset=['precipitation_sum', 'wind_speed_10m_max'])
                W = pd.concat([a, W[W.date > a.date.max()]], ignore_index=True)
            cxy = R.groupby('city')[['lat', 'lon']].mean()
        else:
            sd = pd.read_parquet(P('data/processed/station_day.parquet'))
            co = pd.read_csv(P('data/external/station_coords_fixed.csv')); co = co[co.approx == 0].drop_duplicates('station').set_index('station')
            sd = sd[sd.station.isin(co.index)].rename(columns={'aqi_mean': 'aqi'}); sd['lat'] = sd.station.map(co.lat); sd['lon'] = sd.station.map(co.lon); sd['name'] = sd.station
            R = sd[['station', 'date', 'aqi', 'name', 'city', 'lat', 'lon']].rename(columns={'station': 'station_id'}); key = 'station_id'
            W = pd.read_parquet(P('data/processed/weather_daily.parquet'))
            cc = json.load(open(P('data/external/city_coords.json'))); cxy = pd.DataFrame({k: {'lat': v['lat'], 'lon': v['lon']} for k, v in cc.items() if v}).T
        st = R[R.date == d0].set_index(key)
        prev = R[R.date == d1].set_index(key).aqi
        self.stations = st[['name', 'city', 'lat', 'lon', 'aqi']].copy()
        self.stations['aqi_prev'] = prev.reindex(self.stations.index).values
        # weather of the day at every city that has it (+ the 3-day rain sum and yesterday's rain)
        w = W[(W.date <= d0) & (W.date >= d0 - pd.Timedelta(days=2))].copy()
        today = w[w.date == d0].set_index('city')[WXCOLS].copy()
        today['precip_3d'] = w.groupby('city').precipitation_sum.sum().reindex(today.index)
        today['post_rain_flag'] = (w[w.date == d1].set_index('city').precipitation_sum.reindex(today.index) > 2).astype(float)
        today = today.join(cxy, how='inner')
        self.weather = today.dropna(subset=['wind_speed_10m_max', 'lat'])
        self.has_weather = len(self.weather) > 0

    def _nearest_weather(self, lat, lon):
        wl, wo = self.weather.lat.values, self.weather.lon.values
        _, _, D = pair_geometry(lat, lon, wl, wo)
        return self.weather.iloc[np.argmin(D, axis=1)]

    def estimate(self, lat, lon, hide_km=0.0):
        """Model estimate + baselines at the given points. hide_km: stations closer than this to the point are ignored (self-test of 'no station nearby')."""
        lat, lon = np.atleast_1d(np.asarray(lat, float)), np.atleast_1d(np.asarray(lon, float))
        M = model(); n = len(lat)
        st = self.stations
        DX, DY, D = pair_geometry(lat, lon, st.lat.values, st.lon.values)
        allowed = D > max(hide_km, 0.01)
        w = self._nearest_weather(lat, lon).reset_index(drop=True)
        sp = w.wind_speed_10m_max.values; wd = np.deg2rad(w.wind_direction_10m_dominant.values)
        wu, wv = -sp * np.sin(wd), -sp * np.cos(wd)
        F = wind_features(DX, DY, D, st.aqi.values, st.aqi_prev.values, wu, wv, allowed)
        X = pd.DataFrame(F)
        X['lat'], X['lon'], X['elevation'] = lat, lon, elevation(lat, lon)
        for c in ['temperature_2m_mean', 'precipitation_sum', 'wind_direction_10m_dominant', 'pressure_msl_mean', 'cloud_cover_mean', 'relative_humidity_2m_mean', 'dew_point_2m_mean', 'shortwave_radiation_sum', 'precip_3d', 'post_rain_flag']:
            X[c] = w[c].values
        d = self.date; north = ((lat >= 24) & (lat <= 32.5) & (lon >= 69) & (lon <= 84.5)).astype(float)
        X['month'] = d.month; X['doy_sin'] = np.sin(2 * np.pi * d.dayofyear / 365.25); X['doy_cos'] = np.cos(2 * np.pi * d.dayofyear / 365.25)
        X['is_stubble_season'] = north * float((d.month == 11) or (d.month == 10 and d.day >= 15))
        X['is_diwali'] = float(any(0 <= (d - t).days <= 0 or (d - t).days == -1 for t in DIWALI)); X['is_diwali_post'] = float(any(1 <= (d - t).days <= 4 for t in DIWALI))
        est = np.clip(M['model'].predict(X[M['cols']]), 0, 500)
        out = pd.DataFrame({'lat': lat, 'lon': lon, 'estimate': est, 'nearest_km': F['near_km'], 'nearest_aqi': F['near_aqi'], 'idw': F['idw2']})
        return out
