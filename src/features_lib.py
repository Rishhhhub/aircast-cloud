"""Feature engineering as a reusable function (used for the training table AND for live forecasting).
build_features(sd, wx) -> station-day panel with ~76 features and targets (daily resolution; 1/2/3 days = 24/48/72 h).
sd needs: station, city, state, date, aqi_mean, aqi_max, aqi_min, aqi_std.   wx needs: city, date + the Open-Meteo daily variables."""
import pandas as pd, numpy as np, holidays
H = [1, 2, 3]


def build_features(sd, wx, min_days=500, extend_days=0, keep_missing_today=False):
    """extend_days: extend each station's timeline into the future so weather for t+1..t+3 lines up (live mode)."""
    sd = sd[sd.city.isin(wx.city.unique())]
    cnt = sd.groupby('station').date.count()
    sd = sd[sd.station.isin(cnt[cnt >= min_days].index)]
    # continuous daily index per station
    full = []
    for st, g in sd.groupby('station'):
        idx = pd.date_range(g.date.min(), g.date.max() + pd.Timedelta(days=extend_days), freq='D')
        g = g.set_index('date').reindex(idx)
        g['station'] = st; g['city'] = g.city.ffill().bfill(); g['state'] = g.state.ffill().bfill()
        full.append(g.rename_axis('date').reset_index())
    df = pd.concat(full, ignore_index=True)

    # weather (same day) merged by city
    df = df.merge(wx, on=['city', 'date'], how='left')
    wcols = [c for c in wx.columns if c not in ('city', 'date')]
    for c in wcols: df[c] = df.groupby('city')[c].transform(lambda s: s.interpolate(limit=3))
    df = df.sort_values(['station', 'date']).reset_index(drop=True)
    g = df.groupby('station')

    # ---- lag & rolling features of the station's own AQI
    for l in [1, 2, 3, 7, 14]: df[f'aqi_lag{l}'] = g.aqi_mean.shift(l)
    for w in [3, 7, 14, 30]: df[f'aqi_roll_mean{w}'] = g.aqi_mean.transform(lambda s: s.rolling(w, min_periods=max(2, w // 2)).mean())
    df['aqi_roll_std7'] = g.aqi_mean.transform(lambda s: s.rolling(7, min_periods=4).std())
    df['aqi_roll_max7'] = g.aqi_mean.transform(lambda s: s.rolling(7, min_periods=4).max())
    df['aqi_roll_min7'] = g.aqi_mean.transform(lambda s: s.rolling(7, min_periods=4).min())
    df['aqi_diff1'] = df.aqi_mean - df.aqi_lag1
    df['aqi_expanding_mean'] = g.aqi_mean.transform(lambda s: s.expanding(30).mean().shift(1))

    # ---- hyperlocal / neighbour features: other stations in the same city on the same day
    cs = df.groupby(['city', 'date']).aqi_mean.agg(['sum', 'count']).rename(columns={'sum': 'cs', 'count': 'cc'})
    df = df.join(cs, on=['city', 'date'])
    own = df.aqi_mean.fillna(0); ownc = df.aqi_mean.notna().astype(int)
    df['nbr_aqi_mean'] = ((df.cs - own) / (df.cc - ownc).replace(0, np.nan))
    df['n_city_stations'] = df.cc
    df['nbr_aqi_lag1'] = df.groupby('station').nbr_aqi_mean.shift(1)
    df['station_vs_nbr_30d'] = (df.aqi_roll_mean30 - df.groupby('station').nbr_aqi_mean.transform(lambda s: s.rolling(30, min_periods=10).mean()))
    df = df.drop(columns=['cs', 'cc'])

    # ---- temporal
    d = df.date
    df['month'] = d.dt.month; df['dow'] = d.dt.dayofweek; df['dom'] = d.dt.day; df['doy'] = d.dt.dayofyear
    df['is_weekend'] = (df.dow >= 5).astype(int)
    df['doy_sin'] = np.sin(2 * np.pi * df.doy / 365.25); df['doy_cos'] = np.cos(2 * np.pi * df.doy / 365.25)
    df['season'] = df.month.map({12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 2, 10: 3, 11: 3})  # winter, summer, monsoon, post-monsoon
    ih = holidays.India(years=range(2016, 2028))
    df['is_holiday'] = d.isin(pd.to_datetime(list(ih.keys()))).astype(int)

    # ---- event flags
    diwali = pd.to_datetime(['2016-10-30', '2017-10-19', '2018-11-07', '2019-10-27', '2020-11-14', '2021-11-04', '2022-10-24', '2023-11-12', '2024-11-01', '2025-10-20', '2026-11-08'])
    holi = pd.to_datetime(['2017-03-13', '2018-03-02', '2019-03-21', '2020-03-10', '2021-03-29', '2022-03-18', '2023-03-08', '2024-03-25', '2025-03-14', '2026-03-04'])
    def near(dates, lo, hi): return d.apply(lambda x: any(lo <= (x - t).days <= hi for t in dates)).astype(int)
    df['is_diwali'] = near(diwali, -1, 0)
    df['is_diwali_post'] = near(diwali, 1, 4)
    df['is_holi'] = near(holi, -1, 0)
    df['is_new_years_eve'] = ((d.dt.month == 12) & (d.dt.day >= 31) | (d.dt.month == 1) & (d.dt.day == 1)).astype(int)
    north = df.state.isin(['Delhi', 'Punjab', 'Haryana', 'Uttar Pradesh', 'Rajasthan', 'Chandigarh']).astype(int)
    df['is_stubble_season'] = (((d.dt.month == 10) & (d.dt.day >= 15)) | (d.dt.month == 11)).astype(int) * north
    df['is_harvest_season'] = (d.dt.month.isin([4, 5, 10, 11])).astype(int)
    df['post_rain_flag'] = (df.groupby('station').precipitation_sum.shift(1) > 2).astype(int)

    # ---- weather derived + interactions
    wd = np.deg2rad(df.wind_direction_10m_dominant)
    df['wind_u'] = -df.wind_speed_10m_max * np.sin(wd); df['wind_v'] = -df.wind_speed_10m_max * np.cos(wd)
    df['precip_3d'] = g.precipitation_sum.transform(lambda s: s.rolling(3, min_periods=1).sum())
    df['wind_3d_mean'] = g.wind_speed_10m_max.transform(lambda s: s.rolling(3, min_periods=1).mean())
    df['temp_x_hum'] = df.temperature_2m_mean * df.relative_humidity_2m_mean
    df['stubble_x_wind'] = df.is_stubble_season * df.wind_speed_10m_max
    df['rain_x_lag_aqi'] = df.precipitation_sum * df.aqi_lag1
    df['dow_x_weekend'] = df.dow * df.is_weekend

    # ---- forecast-weather features for the target day (stand-in for an NWP forecast available at issue time)
    fc_vars = ['wind_speed_10m_max', 'precipitation_sum', 'temperature_2m_mean', 'relative_humidity_2m_mean', 'cloud_cover_mean', 'pressure_msl_mean']
    for h in H:
        for v in fc_vars: df[f'fc{h}_{v}'] = g[v].shift(-h)

    # ---- targets
    for h in H: df[f'y_aqi_t{h}'] = g.aqi_mean.shift(-h)
    def bucket(a): return pd.cut(a, [0, 50, 100, 200, 300, 400, 500], labels=[0, 1, 2, 3, 4, 5], include_lowest=True).astype('float')
    for h in H: df[f'y_cat_t{h}'] = bucket(df[f'y_aqi_t{h}'])   # 0 Good,1 Satisfactory,2 Moderate,3 Poor,4 Very Poor,5 Severe

    
    if not keep_missing_today:
        df = df[df.aqi_mean.notna()]
    return df
