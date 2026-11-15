"""LIVE pipeline: recent CPCB station data (Vayuayan archive, to last night) + Open-Meteo forecast API (past 92 days + next days)
 -> same features as training -> (1) out-of-time check on the most recent weeks, (2) 24/48/72 h forecasts for every station.
Outputs: data/processed/live_forecast.parquet, reports/live_check.csv, data/processed/weather_live.parquet.  Safe to rerun any time (refreshes weather; re-run 15_fetch_recent.py first for new AQI)."""
import os, requests, time, json, joblib, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from features_lib import build_features
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
R = pd.read_parquet('data/processed/recent_station_day.parquet'); live = pd.read_csv('data/external/live_stations.csv')
R = R.merge(live[['station_id', 'lat', 'lon']], on='station_id', how='left')
cityxy = R.dropna(subset=['lat']).groupby('city')[['lat', 'lon']].mean(); R = R[R.city.isin(cityxy.index)]
sd = R.rename(columns={'station_id': 'station'})[['station', 'station_name', 'city', 'state', 'date', 'aqi_mean', 'aqi_max', 'aqi_min', 'aqi_std']]
dups = int(sd.duplicated(['station', 'date']).sum()); print('duplicate station-days collapsed:', dups)
sd = sd.groupby(['station', 'date'], as_index=False).agg({'station_name': 'first', 'city': 'first', 'state': 'first', 'aqi_mean': 'mean', 'aqi_max': 'max', 'aqi_min': 'min', 'aqi_std': 'mean'})
VARS = ['temperature_2m_mean', 'temperature_2m_max', 'temperature_2m_min', 'precipitation_sum', 'wind_speed_10m_max', 'wind_direction_10m_dominant', 'pressure_msl_mean', 'cloud_cover_mean',
        'relative_humidity_2m_mean', 'dew_point_2m_mean', 'shortwave_radiation_sum', 'et0_fao_evapotranspiration']
cities = list(cityxy.index); rows = []
for i in range(0, len(cities), 40):
    ch = cities[i:i + 40]
    for t in range(6):
        r = requests.get('https://api.open-meteo.com/v1/forecast', params={'latitude': ','.join(f'{cityxy.lat[c]:.3f}' for c in ch), 'longitude': ','.join(f'{cityxy.lon[c]:.3f}' for c in ch),
                         'daily': ','.join(VARS), 'past_days': 92, 'forecast_days': 6, 'timezone': 'Asia/Kolkata'}, timeout=120)
        if r.status_code == 200: break
        time.sleep(15 * (t + 1))
    else: raise SystemExit('weather fetch failed: ' + r.text[:200])
    js = r.json(); js = js if isinstance(js, list) else [js]
    for c, j in zip(ch, js):
        w = pd.DataFrame(j['daily']).rename(columns={'time': 'date'}); w['city'] = c; rows.append(w)
    print('weather', min(i + 40, len(cities)), '/', len(cities), flush=True)
wx = pd.concat(rows); wx['date'] = pd.to_datetime(wx.date); wx.to_parquet('data/processed/weather_live.parquet')
if os.path.exists('data/external/weather_recent_archive.parquet'):    # the archive is authoritative for the past; the forecast API only for days the archive does not have yet
    arc = pd.read_parquet('data/external/weather_recent_archive.parquet').dropna(subset=['precipitation_sum', 'wind_speed_10m_max'])
    wx = pd.concat([arc, wx[wx.date > arc.date.max()]], ignore_index=True).drop_duplicates(['city', 'date'], keep='first')
    print('weather: archive up to', arc.date.max().date(), '+ forecast API after that')
print('weather covers', wx.date.min().date(), '->', wx.date.max().date())
T = build_features(sd, wx, min_days=30, extend_days=3, keep_missing_today=True)
today = T.groupby('station').apply(lambda g: g.dropna(subset=['aqi_mean']).date.max())
T = T[T.aqi_mean.notna()]
T.to_parquet('data/processed/recent_features.parquet')   # reused by 19_retrain_recent.py
MODELS = {h: joblib.load(f'models/xgb_reg_h{h * 24}.joblib') for h in (1, 2, 3)}
def X(df, h):
    m = MODELS[h]; df = df.rename(columns={c: c.replace(f'fc{h}_', 'fc_') for c in df.columns if c.startswith(f'fc{h}_')}); return df[m.get_booster().feature_names]
# (1) out-of-time check on the latest weeks (weather actually observed/forecast for those days; model never saw 2024+)
res = []
for h in (1, 2, 3):
    e = T[(T.date >= '2026-08-15') & T[f'y_aqi_t{h}'].notna() & T.precipitation_sum.notna()]
    p = np.clip(MODELS[h].predict(X(e, h)), 0, 500); y = e[f'y_aqi_t{h}'].values
    for name, q in (('Persistence', e.aqi_mean.values), ('XGBoost (trained 2017-2022)', p)):
        res.append(dict(horizon_h=h * 24, model=name, n=len(e), RMSE=np.sqrt(mean_squared_error(y, q)), MAE=mean_absolute_error(y, q), R2=r2_score(y, q)))
res = pd.DataFrame(res); res.to_csv('reports/live_check.csv', index=False); print(res.round(3).to_string(index=False))
# (2) live forecasts from each station's last observed day
last = T.merge(today.rename('last').reset_index(), on='station'); last = last[last.date == last['last']]
last = last[last['last'] >= last['last'].max() - pd.Timedelta(days=2)]
out = last[['station', 'station_name', 'city', 'state', 'date', 'aqi_mean']].rename(columns={'date': 'obs_date', 'aqi_mean': 'aqi_obs'}).reset_index(drop=True)
for h in (1, 2, 3): out[f'pred_{h * 24}h'] = np.clip(MODELS[h].predict(X(last, h)), 0, 500)
if os.path.exists('models/interval_adjust.json'):        # prediction ranges + P(AQI > 200); see 20_intervals.py
    latest = T.date.max()
    for h in (1, 2, 3):
        ren = lambda d: d.rename(columns={c: c.replace(f'fc{h}_', 'fc_') for c in d.columns if c.startswith(f'fc{h}_')})
        qm = joblib.load(f'models/xgb_q_h{h * 24}.joblib'); fn = qm.get_booster().feature_names; yh = f'y_aqi_t{h}'
        qq = lambda d: np.sort(np.clip(qm.predict(d[fn]), 0, 500), axis=1)
        # rolling 28-day normalised conformal margin: calibrate on forecasts whose outcomes are already known (issued 28+h .. h days ago)
        cr = ren(T[(T.date > latest - pd.Timedelta(days=h + 28)) & (T.date <= latest - pd.Timedelta(days=h)) & T[yh].notna() & T.precipitation_sum.notna()])
        pc = qq(cr); sc = np.maximum(pc[:, 0] - cr[yh].values, cr[yh].values - pc[:, 2]) / (pc[:, 2] - pc[:, 0] + 5)
        Qn = float(np.quantile(sc, min(1, 0.8 * (1 + 1 / len(sc))))) if len(sc) >= 200 else 0.07
        pq = qq(ren(last)); wd = pq[:, 2] - pq[:, 0] + 5
        out[f'lo_{h * 24}h'] = np.clip(pq[:, 0] - Qn * wd, 0, 500); out[f'hi_{h * 24}h'] = np.clip(pq[:, 2] + Qn * wd, 0, 500)
        out[f'pred_{h * 24}h'] = out[f'pred_{h * 24}h'].clip(out[f'lo_{h * 24}h'], out[f'hi_{h * 24}h'])
        clf = joblib.load(f'models/xgb_clf_h{h * 24}.joblib'); out[f'p_poor_{h * 24}h'] = clf.predict_proba(ren(last)[clf.get_booster().feature_names])[:, 3:].sum(1)
        print(f'h={h * 24}: rolling margin {Qn:.2f} x width from {len(sc)} recent outcomes; mean range width {np.mean(out[f"hi_{h * 24}h"] - out[f"lo_{h * 24}h"]):.0f}')
try:                                                       # v4 production models (29_production_v4.py --final): use them where a city climatology and CAMS exist
    import prod_v4
    from season_physics import load_or_make_thresholds
    thr = load_or_make_thresholds(pd.read_parquet('data/processed/weather_daily.parquet'))
    cams = prod_v4.fetch_cams(cityxy)
    if cams is not None: cams.to_parquet('data/processed/cams_live.parquet')
    for h in (1, 2, 3):
        p, ok = prod_v4.predict(last.reset_index(drop=True), wx, thr, cams, h, borrow=True)   # borrowed climatology for cities without history (31_fallback_clim_eval)
        if p is None: continue
        k = f'pred_{h * 24}h'; orig = out[k].values.copy(); new = np.where(ok, p, orig); delta = new - orig
        out[k + '_original'] = orig; out[k] = new; out[f'model_{h * 24}h'] = np.where(ok, 'v4', 'original')
        if f'lo_{h * 24}h' in out:                          # keep the calibrated range width, centred on the new forecast
            out[f'lo_{h * 24}h'] = np.clip(out[f'lo_{h * 24}h'] + delta, 0, 500); out[f'hi_{h * 24}h'] = np.clip(out[f'hi_{h * 24}h'] + delta, 0, 500)
        print(f'v4 used for {ok.mean() * 100:.0f}% of stations at +{h * 24} h (others: original model)')
except Exception as e:
    print('v4 skipped:', type(e).__name__, str(e)[:200])
out = out.merge(live[['station_id', 'lat', 'lon']].rename(columns={'station_id': 'station'}), on='station', how='left')
out.to_parquet('data/processed/live_forecast.parquet'); print(len(out), 'stations forecast from', out.obs_date.min().date(), 'to', out.obs_date.max().date()); print(out.head(3).round(0).to_string())
