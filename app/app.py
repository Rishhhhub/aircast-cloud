"""AirCast dashboard: station-level AQI forecast and map tabs.
Run from the project root:  .venv/bin/streamlit run app/app.py
"""
import json, os
import joblib, numpy as np, pandas as pd, plotly.express as px, plotly.graph_objects as go, streamlit as st
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from aqi_field import field_overlay, css_gradient
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHOW_EXTENDED = os.environ.get('AIRCAST_EXTENDED') == '1'      # the extended model and the prediction ranges are hidden unless this is set
CATS = ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe']
CAT_COL = ['#2e9e5b', '#8bc34a', '#f2c230', '#f08a24', '#e0452b', '#8e2a8f']
BOUNDS = [0, 50, 100, 200, 300, 400, 500]
st.set_page_config(page_title='AirCast', layout='wide')


def category(a):
    return CATS[int(np.clip(np.searchsorted(BOUNDS, a, side='left') - 1, 0, 5))]


def legend_bar():
    """Apple-style continuous AQI legend under the maps."""
    st.markdown(f'<div style="max-width:560px;margin:6px 0 2px"><div style="height:12px;border-radius:6px;background:{css_gradient()}"></div>'
                '<div style="display:flex;justify-content:space-between;font-size:12px;opacity:.75;margin-top:3px"><span>0 Good</span><span>50</span><span>100 Moderate</span><span>200 Poor</span><span>300</span><span>400 Severe</span><span>500+</span></div></div>', unsafe_allow_html=True)


def dot_labels(df_, lat, lon, val, names=None):
    """Small white-ringed station dots with the AQI number on them (readable over the colour field)."""
    return go.Scattermap(lat=df_[lat], lon=df_[lon], mode='markers+text', text=[f'{v:.0f}' for v in df_[val]], textfont=dict(size=10, color='#111'),
                         marker=dict(size=15, color='white', opacity=0.92), hovertext=names, hoverinfo='text' if names is not None else 'skip', showlegend=False)


@st.cache_data(show_spinner='Loading data...')
def load_table():
    df = pd.read_parquet(os.path.join(ROOT, 'data/processed/model_table.parquet'))
    co = pd.read_csv(os.path.join(ROOT, 'data/external/station_coords_fixed.csv')).set_index('station')
    return df, co


@st.cache_resource
def load_model(h):
    return joblib.load(os.path.join(ROOT, f'models/xgb_reg_h{h}.joblib'))


def feature_row(df, station, date, h):
    """Features for one station-day, with the target-day weather columns renamed fc{h}_* -> fc_*."""
    r = df[(df.station == station) & (df.date == date)]
    if r.empty:
        return None
    r = r.rename(columns={c: c.replace(f'fc{h}_', 'fc_') for c in r.columns if c.startswith(f'fc{h}_')})
    return r


@st.cache_resource
def load_location_model():
    return joblib.load(os.path.join(ROOT, 'models/location_model.joblib'))


@st.cache_data
def city_centres():
    return json.load(open(os.path.join(ROOT, 'data/external/city_coords.json')))


@st.cache_data
def station_elevation():
    return pd.read_csv(os.path.join(ROOT, 'data/processed/station_location_features.csv')).set_index('station').elevation


@st.cache_data(show_spinner='Looking up elevations...')
def elevations(lats, lons):
    """Open-Meteo elevation API (max 100 points per call); returns None if the lookup fails."""
    import requests
    out = []
    try:
        for i in range(0, len(lats), 100):
            la, lo = lats[i:i + 100], lons[i:i + 100]
            r = requests.get('https://api.open-meteo.com/v1/elevation', params={'latitude': ','.join(f'{x:.5f}' for x in la), 'longitude': ','.join(f'{x:.5f}' for x in lo)}, timeout=30)
            out += r.json()['elevation']
        return np.array(out, dtype=float)
    except Exception:
        return None


def loc_features(base, lat, lon, elev, clat, clon):
    """Feature frame for points (lat, lon): the city's weather/calendar on the chosen day plus static location features."""
    LM = load_location_model()
    n = len(lat)
    X = pd.DataFrame({c: [base[c]] * n for c in LM['wx'] + LM['cal']})
    X['elevation'], X['lat'], X['lon'] = elev, lat, lon
    X['dy'] = (np.asarray(lat) - clat) * 111.19
    X['dx'] = (np.asarray(lon) - clon) * 111.19 * np.cos(np.deg2rad(clat))
    X['dist_centre'] = np.hypot(X.dx, X.dy)
    return X[LM['features']]


@st.cache_data(ttl=300)
def load_live():
    f = os.path.join(ROOT, 'data/processed/live_forecast.parquet')
    return pd.read_parquet(f) if os.path.exists(f) else None


@st.cache_data(ttl=300)
def live_check():
    f = os.path.join(ROOT, 'reports/live_check.csv')
    return pd.read_csv(f) if os.path.exists(f) else None


@st.cache_data(ttl=300)
def live_date_range():
    """First and last day with live station readings and weather (the latest day needs at least 300 stations; today's partial day is left out)."""
    n = pd.read_parquet(os.path.join(ROOT, 'data/processed/recent_station_day.parquet'), columns=['date', 'station_id']).groupby('date').station_id.nunique()
    last_st = n[n >= 300].index.max()
    wl = pd.read_parquet(os.path.join(ROOT, 'data/processed/weather_live.parquet'), columns=['date', 'wind_speed_10m_max']).dropna().date.max()
    return pd.Timestamp('2026-07-14').date(), min(pd.Timestamp(last_st), pd.Timestamp(wl), pd.Timestamp.today().normalize() - pd.Timedelta(days=1)).date()


@st.cache_resource(show_spinner='Loading stations and weather for that day...')
def corridor_ctx(source, date_str):
    from corridor import Context
    return Context(source, date_str)


PRESETS = {'Punjab - Haryana - Delhi corridor': (28.0, 31.6, 74.0, 77.9), 'Delhi NCR': (28.0, 29.2, 76.5, 77.9), 'Mumbai - Pune': (18.3, 19.4, 72.6, 74.2), 'Kolkata region': (22.0, 23.2, 87.8, 89.0)}
# typical error (RMSE, AQI points) of the estimator by how far the nearest station is, from the 2023 held-out-station test (reports/wind_aware.csv); winter in the Punjab-Haryana-Delhi corridor is harder
def typical_error(km, corridor_winter, live=True):
    """Tested RMSE of the estimator by distance to the nearest station. live=True: 2026 test (2025-12..2026-10, 78 days, reports/corridor_live_eval_summary.txt);
    live=False: 2023 held-out-station test (reports/wind_aware.csv). corridor_winter: Punjab-Haryana-Delhi corridor in Oct-Feb."""
    i = 0 if km < 12 else 1 if km < 37 else 2 if km < 75 else 3
    if live:
        return (76, 80, 87, 93)[i] if corridor_winter else (72, 76, 78, 81)[i]
    return (50, 59, 63, 64)[i] if corridor_winter else (46, 49, 51, 52)[i]


df, coords = load_table()
st.title('AirCast: hyperlocal AQI forecast')
st.caption('CSE2032 Machine Learning, BML Munjal University. Station-level forecasts 24, 48 and 72 hours ahead from CPCB data and weather.')

# ------------------------------------------------------------------ sidebar
cities = sorted(df.city.unique(), key=lambda c: -df[df.city == c].station.nunique())
city = st.sidebar.selectbox('City', cities, index=cities.index('Delhi') if 'Delhi' in cities else 0)
stations = sorted(df[df.city == city].station.unique())
station = st.sidebar.selectbox('Station', stations)
dmin, dmax = df[df.station == station].date.min().date(), df[df.station == station].date.max().date()
default = min(max(pd.Timestamp('2023-11-10').date(), dmin), dmax)
date = pd.Timestamp(st.sidebar.date_input('Forecast issued on', default, min_value=dmin, max_value=dmax))
st.sidebar.info('The weather inputs for the forecast days come from the Open-Meteo archive (a stand-in for a real weather forecast).')

tab0, tabB, tab1, tab2 = st.tabs(['Live forecast', 'Between stations', 'Forecast (history)', 'Map'])

# ------------------------------------------------------------------ live tab
with tab0:
    import subprocess, sys
    LF = load_live()
    top = st.columns([3, 1])
    with top[1]:
        if st.button('Refresh live data (about 3 min)'):
            with st.spinner('Downloading the latest station data and weather forecast, then re-running the models...'):
                logs = []
                for script in ('src/15_fetch_recent.py', 'src/16_live_pipeline.py'):
                    r = subprocess.run([sys.executable, os.path.join(ROOT, script)], cwd=ROOT, capture_output=True, text=True)
                    logs.append(r.stdout[-400:] + r.stderr[-300:])
                    if r.returncode != 0:
                        st.error(f'{script} failed:\n' + logs[-1]); break
            st.cache_data.clear(); st.rerun()
    if LF is None:
        st.info('No live forecast yet. Click "Refresh live data", or run src/15_fetch_recent.py and src/16_live_pipeline.py.')
    else:
        obs = LF.obs_date.max()
        # Model switch. "Course model" = the XGBoost regressor of the notebook (same 64 features, trained on 2017-2022). "Extended" = the v4 ensemble (season physics + CAMS correction).
        model_mode = (st.radio('Model', ['Course model (as in the notebook)', 'Extended model (adds CAMS + weather physics)'], horizontal=True, key='live_model',
                               help='Course model: the XGBoost regression from the notebook. Extended: boosted-tree ensemble with rain/stagnation physics and a correction learned from the CAMS atmospheric forecast.')
                      if SHOW_EXTENDED else 'Course model (as in the notebook)')
        if model_mode.startswith('Course') and 'pred_24h_original' in LF.columns:
            LF = LF.copy()
            for h in (24, 48, 72):
                shift = LF[f'pred_{h}h'] - LF[f'pred_{h}h_original']          # the live pipeline moved the range with the v4 forecast; undo that for the course model
                LF[f'pred_{h}h'] = LF[f'pred_{h}h_original']
                if f'lo_{h}h' in LF.columns: LF[f'lo_{h}h'] = (LF[f'lo_{h}h'] - shift).clip(0, 500); LF[f'hi_{h}h'] = (LF[f'hi_{h}h'] - shift).clip(0, 500)
                LF[f'model_{h}h'] = 'original'
        with top[0]:
            st.subheader(f'Live forecast: station data up to {obs.strftime("%d %b %Y")}, weather forecast for the next days')
            st.caption(f'{len(LF)} stations in {LF.city.nunique()} cities' + (f", {int((LF.model_24h == 'v4').mean() * 100)}% forecast by the extended model v4 (boosted-tree ensemble with season physics, recent data and the CAMS forecast), the rest by the course model" if 'model_24h' in LF.columns and (LF.model_24h == 'v4').any() else ', all forecast by the course model (the notebook XGBoost regression)') + f'. The last observed day may be partial. Forecast dates: +24 h = {(obs + pd.Timedelta(days=1)).strftime("%d %b")}, +48 h = {(obs + pd.Timedelta(days=2)).strftime("%d %b")}, +72 h = {(obs + pd.Timedelta(days=3)).strftime("%d %b")}.')
        cl = sorted(LF.city.unique(), key=lambda c: -(LF.city == c).sum())
        lcity = st.selectbox('City', cl, index=cl.index('Delhi') if 'Delhi' in cl else 0, key='live_city')
        sub = LF[LF.city == lcity]
        m = st.columns(4)
        m[0].metric('Observed (city mean)', f'{sub.aqi_obs.mean():.0f}', category(sub.aqi_obs.mean()), delta_color='off')
        has_iv = SHOW_EXTENDED and 'lo_24h' in sub.columns
        for col, h in zip(m[1:], (24, 48, 72)):
            v = sub[f'pred_{h}h'].mean()
            rng = f', likely {sub[f"lo_{h}h"].mean():.0f} to {sub[f"hi_{h}h"].mean():.0f}' if has_iv else ''
            col.metric(f'Forecast +{h} h', f'{v:.0f}', category(v) + rng, delta_color='off')
        if has_iv:
            pp = [sub[f'p_poor_{h}h'].mean() * 100 for h in (24, 48, 72)]
            st.caption(f'Chance that a typical station here is Poor or worse (AQI above 200): +24 h {pp[0]:.0f}%, +48 h {pp[1]:.0f}%, +72 h {pp[2]:.0f}%. "Likely" is an 80% prediction range, averaged over the stations (in tests on the newest weeks such ranges covered about 73 to 77% of outcomes). Very high chances are slightly overconfident: a stated 98% happened about 91% of the time.')
        show = st.radio('Colour the map by', ['Observed', '+24 h', '+48 h', '+72 h'], horizontal=True, key='live_show')
        colname = {'Observed': 'aqi_obs', '+24 h': 'pred_24h', '+48 h': 'pred_48h', '+72 h': 'pred_72h'}[show]
        mp = sub.dropna(subset=['lat', 'lon']).copy()
        if len(mp):
            hov = [f'{n}<br>Observed {a:.0f} | +24h {p1:.0f} | +48h {p2:.0f} | +72h {p3:.0f}' + (f'<br>+24h likely {l1:.0f} to {u1:.0f} | chance of Poor or worse {pp:.0%}' if has_iv else '')
                   for n, a, p1, p2, p3, l1, u1, pp in zip(mp.station_name, mp.aqi_obs, mp.pred_24h, mp.pred_48h, mp.pred_72h, mp.get('lo_24h', mp.pred_24h), mp.get('hi_24h', mp.pred_24h), mp.get('p_poor_24h', mp.pred_24h * 0))]
            fl = go.Figure()
            if len(mp) >= 3:
                src, img_xy, bb = field_overlay(mp.lat, mp.lon, mp[colname])
                fl.update_layout(map=dict(layers=[dict(sourcetype='image', source=src, coordinates=img_xy, below='traces')]))
            fl.add_trace(dot_labels(mp, 'lat', 'lon', colname, hov))
            fl.update_layout(map=dict(style='carto-positron', center=dict(lat=mp.lat.mean(), lon=mp.lon.mean()), zoom=9.4 if len(mp) > 4 else 8.5, layers=fl.layout.map.layers), height=520, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fl, width='stretch')
            legend_bar()
        tb = sub[['station_name', 'aqi_obs', 'pred_24h', 'pred_48h', 'pred_72h'] + (['lo_24h', 'hi_24h', 'p_poor_24h'] if has_iv else [])].copy().round(0)
        tb['Category +24 h'] = tb.pred_24h.map(category)
        if 'model_24h' in sub.columns:
            tb['Model'] = sub.model_24h.map({'v4': 'v4 (ensemble + CAMS)', 'original': 'course model'}).values
        if has_iv:
            tb['Range +24 h'] = tb.lo_24h.astype(int).astype(str) + ' to ' + tb.hi_24h.astype(int).astype(str)
            tb['Chance Poor+ (+24 h)'] = (sub.p_poor_24h * 100).round(0).astype(int).astype(str) + '%'
            tb = tb.drop(columns=['lo_24h', 'hi_24h', 'p_poor_24h'])
        st.dataframe(tb.rename(columns={'station_name': 'Station', 'aqi_obs': 'Observed', 'pred_24h': '+24 h', 'pred_48h': '+48 h', 'pred_72h': '+72 h'}).sort_values('+24 h', ascending=False), hide_index=True)
        with st.expander('Most polluted stations in India, forecast for +24 h'):
            st.dataframe(LF.sort_values('pred_24h', ascending=False).head(15)[['station_name', 'city', 'state', 'aqi_obs', 'pred_24h', 'pred_48h', 'pred_72h']].round(0), hide_index=True)
        chk = live_check()
        if chk is not None:
            e = chk.pivot(index='model', columns='horizon_h', values='RMSE').round(1)
            st.caption('How good is it? Test on 15 Aug to 7 Oct 2026, with models trained only on 2017 to 2022 (RMSE in AQI points; lower is better): ' + '; '.join(f'{int(h)} h: model {e.loc["XGBoost (trained 2017-2022)", h]} vs persistence {e.loc["Persistence", h]}' for h in e.columns) +
                       ('. The extended model scored 34.0 / 44.7 / 49.4 on 2026 stations (see PROJECT_LOG.md)' if not model_mode.startswith('Course') else '') + '. Caveat: for that test the weather for the forecast days was the observed one, so real accuracy with a weather forecast will be somewhat lower.')

# ------------------------------------------------------------------ between stations tab
with tabB:
    st.subheader('AQI between stations: an estimate for places with no station')
    st.caption('Uses the readings of all surrounding stations (today and yesterday), the wind and weather, and the location. In tests where every station within 100 km was hidden, this estimate was far better than distance-weighting or the nearest station (see the report).')
    c1, c2, c3 = st.columns([1.2, 1.2, 1.6])
    mode = c1.radio('Data', ['Live (2026)'], horizontal=False, key='bt_mode')   # the historical mode needs the full multi-year tables, which are left out of the cloud copy to fit the memory limit
    if mode.startswith('Live'):
        wmin, wmax = live_date_range()
        bdate = c2.date_input('Day', wmax, min_value=wmin, max_value=wmax, key='bt_live_date'); src = 'live'
    else:
        bdate = c2.date_input('Day', pd.Timestamp('2022-11-05').date(), min_value=pd.Timestamp('2017-01-02').date(), max_value=pd.Timestamp('2023-12-31').date(), key='bt_hist_date'); src = 'hist'
    region = c3.selectbox('Region', list(PRESETS), key='bt_region')
    la0, la1, lo0, lo1 = PRESETS[region]
    ctx = corridor_ctx(src, str(bdate))
    if not ctx.has_weather or len(ctx.stations) < 20:
        st.info('Not enough station or weather data for that day. Pick another day.')
    else:
        g = 26
        gl, go2 = np.meshgrid(np.linspace(la0, la1, g), np.linspace(lo0, lo1, g), indexing='ij')
        est = ctx.estimate(gl.ravel(), go2.ravel(), hide_km=0.01)
        stn = ctx.stations[(ctx.stations.lat.between(la0 - 0.3, la1 + 0.3)) & (ctx.stations.lon.between(lo0 - 0.3, lo1 + 0.3))].dropna(subset=['aqi'])
        src_img, img_xy, bb = field_overlay(est.lat, est.lon, est.estimate, bbox=(la0 - 0.15, la1 + 0.15, lo0 - 0.15, lo1 + 0.15), bandwidth_km=14.0, fade_km=(30.0, 130.0), pad=0.15,
                                            fade_from=(ctx.stations.lat.values, ctx.stations.lon.values))
        fb = go.Figure()
        fb.add_trace(dot_labels(stn.assign(a=stn.aqi), 'lat', 'lon', 'a', [f'{n}: {v:.0f}' for n, v in zip(stn.name, stn.aqi)]))
        fb.update_layout(map=dict(style='carto-positron', center=dict(lat=(la0 + la1) / 2, lon=(lo0 + lo1) / 2), zoom=6.6 if (la1 - la0) > 2 else 8.3, layers=[dict(sourcetype='image', source=src_img, coordinates=img_xy, below='traces')]),
                         height=560, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fb, width='stretch')
        legend_bar()
        st.caption(f'{len(stn)} stations in view. The colour is the model estimate on a grid; it fades where the nearest real station is far away (more than about 30 km), because the estimate is less certain there.')

        st.markdown('**Estimate at any point**')
        p1, p2, p3 = st.columns([1, 1, 2])
        plat = p1.number_input('Latitude', value=float(round((la0 + la1) / 2, 3)), format='%.3f', step=0.05, key='bt_lat')
        plon = p2.number_input('Longitude', value=float(round((lo0 + lo1) / 2, 3)), format='%.3f', step=0.05, key='bt_lon')
        e = ctx.estimate([plat], [plon], hide_km=0.01).iloc[0]
        corridor_w = (28 <= plat <= 32.5 and 73.5 <= plon <= 78.5 and bdate.month in (10, 11, 12, 1, 2))
        p3.metric('Estimated AQI', f'{e.estimate:.0f}', f'{category(e.estimate)}; typical error about {typical_error(e.nearest_km, corridor_w, src == 'live')}; nearest station {e.nearest_km:.0f} km away (reads {e.nearest_aqi:.0f})', delta_color='off')

        st.markdown('**Self-test: hide the stations around a real station and see how close the estimate gets**')
        s1, s2 = st.columns([2, 1])
        names = stn.sort_values('name').index.tolist()
        pick = s1.selectbox('Station to hide', names, format_func=lambda i: f'{stn.loc[i, "name"]} ({stn.loc[i, "city"]})', key='bt_pick') if names else None
        rad = s2.select_slider('Hide all stations within (km)', options=[0, 25, 50, 100], value=50, key='bt_rad')
        if pick is not None:
            r_ = stn.loc[pick]; ee = ctx.estimate([r_.lat], [r_.lon], hide_km=max(rad, 0.01)).iloc[0]
            hidden = int((np.hypot(*[(ctx.stations.lat - r_.lat) * 111.19, (ctx.stations.lon - r_.lon) * 111.19 * np.cos(np.deg2rad(r_.lat))]) <= max(rad, 0.01)).sum())
            q = st.columns(4)
            q[0].metric('Actual reading', f'{r_.aqi:.0f}', category(r_.aqi), delta_color='off')
            q[1].metric('Our estimate', f'{ee.estimate:.0f}', f'off by {ee.estimate - r_.aqi:+.0f}', delta_color='off')
            q[2].metric('Inverse-distance guess', f'{ee.idw:.0f}' if pd.notna(ee.idw) else 'n/a', f'off by {ee.idw - r_.aqi:+.0f}' if pd.notna(ee.idw) else '', delta_color='off')
            q[3].metric('Nearest remaining station', f'{ee.nearest_aqi:.0f}' if pd.notna(ee.nearest_aqi) else 'n/a', f'{ee.nearest_km:.0f} km away; {hidden} stations hidden', delta_color='off')
        st.caption('One day at one station is anecdotal. Tested numbers with every station within 100 km hidden: in 2023 the typical error was about 52 AQI points (about 64 in winter in the Punjab-Haryana-Delhi corridor) against about 74 (and 142) for inverse-distance weighting; '
                   'on live 2026 data (Dec 2025 to Oct 2026) it was about 81 (93 in the corridor in winter) against 96 (156) for inverse-distance weighting and 109 (137) for the nearest station. '
                   'Wind direction itself did not add measurable skill at this daily resolution; the gain comes from learning how regional levels, weather and location combine.')

# ------------------------------------------------------------------ forecast tab
with tab1:
    row0 = feature_row(df, station, date, 24 // 24)
    if row0 is None:
        st.warning('No data for this station on that day. Pick another date.')
    else:
        today = float(row0.aqi_mean.iloc[0])
        c0, *cs = st.columns(4)
        c0.metric('AQI on ' + date.strftime('%d %b %Y'), f'{today:.0f}', category(today))
        preds = {}
        for col, h in zip(cs, (1, 2, 3)):
            r = feature_row(df, station, date, h)
            m = load_model(h * 24)
            X = r[m.get_booster().feature_names]
            p = float(np.clip(m.predict(X)[0], 0, 500))
            preds[h] = p
            actual = r[f'y_aqi_t{h}'].iloc[0]
            delta = f'actual {actual:.0f}' if pd.notna(actual) else 'no actual yet'
            col.metric(f'Forecast +{h * 24} h', f'{p:.0f}', f'{category(p)} ({delta})', delta_color='off')
        hist = df[(df.station == station) & (df.date <= date) & (df.date > date - pd.Timedelta(days=60))]
        fut = df[(df.station == station) & (df.date > date) & (df.date <= date + pd.Timedelta(days=3))]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist.date, y=hist.aqi_mean, mode='lines', name='Observed AQI', line=dict(color='#2a78d6', width=2)))
        if len(fut):
            fig.add_trace(go.Scatter(x=fut.date, y=fut.aqi_mean, mode='lines+markers', name='Observed later', line=dict(color='#9aa3af', dash='dot')))
        fig.add_trace(go.Scatter(x=[date + pd.Timedelta(days=h) for h in preds], y=list(preds.values()), mode='markers+lines', name='AirCast forecast',
                                 marker=dict(size=10, color='#eb6834'), line=dict(color='#eb6834')))
        for lo, hi, c in zip(BOUNDS[:-1], BOUNDS[1:], CAT_COL):
            fig.add_hrect(y0=lo, y1=hi, fillcolor=c, opacity=0.07, line_width=0)
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), yaxis_title='AQI', yaxis_range=[0, max(300, hist.aqi_mean.max() * 1.1)], legend=dict(orientation='h'))
        st.plotly_chart(fig, width='stretch')

# ------------------------------------------------------------------ map tab
with tab2:
    st.subheader(f'{city}: estimated AQI across the city on {date.strftime("%d %b %Y")}')
    LM = load_location_model()
    cc = city_centres()[city]
    clat, clon = cc['lat'], cc['lon']
    dayrows = df[(df.city == city) & (df.date == date)]
    day = dayrows.join(coords[['lat', 'lon', 'approx']], on='station')
    day = day[day.approx == 0].dropna(subset=['lat'])
    if dayrows.empty:
        st.info('No data for this city on that date. Pick another date in the sidebar.')
    else:
        base = dayrows.iloc[0]
        selev = station_elevation()
        if len(day) >= 3:
            la0, la1, lo0, lo1 = day.lat.min() - 0.04, day.lat.max() + 0.04, day.lon.min() - 0.04, day.lon.max() + 0.04
        else:
            la0, la1, lo0, lo1 = clat - 0.09, clat + 0.09, clon - 0.09, clon + 0.09
        g = 14
        gl, go_ = np.meshgrid(np.linspace(la0, la1, g), np.linspace(lo0, lo1, g), indexing='ij')
        glat, glon = tuple(gl.ravel()), tuple(go_.ravel())
        el = elevations(glat, glon)
        fallback = float(selev.reindex(day.station).median()) if len(day) else 200.0
        el = el if el is not None else np.full(len(glat), fallback if np.isfinite(fallback) else 200.0)
        pred = np.clip(LM['model'].predict(loc_features(base, np.array(glat), np.array(glon), el, clat, clon)), 0, 500)
        shift = 0.0
        calibrate = False
        if len(day):
            calibrate = st.checkbox('Calibrate to the stations\' actual readings today (recommended where stations exist)', value=True)
            ds = day[day.station.isin(selev.dropna().index)]
            if calibrate and len(ds):
                ps = LM['model'].predict(loc_features(base, ds.lat.values, ds.lon.values, selev.reindex(ds.station).values, clat, clon))
                shift = float(ds.aqi_mean.mean() - ps.mean())
        pred = np.clip(pred + shift, 0, 500)
        fm = go.Figure()
        src, img_xy, bb = field_overlay(glat, glon, pred, bbox=(la0 - 0.2, la1 + 0.2, lo0 - 0.2, lo1 + 0.2), bandwidth_km=3.0, fade_km=(3.0, 14.0))
        layers = [dict(sourcetype='image', source=src, coordinates=img_xy, below='traces')]
        if len(day):
            fm.add_trace(dot_labels(day, 'lat', 'lon', 'aqi_mean', [f'{n}: actual AQI {v:.0f}' for n, v in zip(day.station, day.aqi_mean)]))
        fm.update_layout(map=dict(style='carto-positron', center=dict(lat=(la0 + la1) / 2, lon=(lo0 + lo1) / 2), zoom=9.6 if len(day) > 5 else 9.2, layers=layers), height=540, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fm, width='stretch')
        legend_bar()

        st.markdown('**Estimate AQI at any point**')
        c1, c2, c3 = st.columns([1, 1, 2])
        plat = c1.number_input('Latitude', value=float(round((la0 + la1) / 2, 4)), format='%.4f', step=0.005)
        plon = c2.number_input('Longitude', value=float(round((lo0 + lo1) / 2, 4)), format='%.4f', step=0.005)
        pe = elevations((plat,), (plon,))
        pe = pe if pe is not None else np.array([fallback if np.isfinite(fallback) else 200.0])
        pv = float(np.clip(LM['model'].predict(loc_features(base, np.array([plat]), np.array([plon]), pe, clat, clon))[0] + shift, 0, 500))
        c3.metric('Estimated AQI at this point', f'{pv:.0f}', category(pv), delta_color='off')
    st.caption('The model uses no station readings at the chosen point: only the day\'s weather, calendar and location (elevation and position in the city). '
               'In tests on cities it had never seen, its error was about 60 AQI points (RMSE), against about 95 for a month-of-year average. '
               'Inside a city, the pattern between neighbourhoods is weak (it explains roughly 11% of the differences between stations), so calibration to live station readings sets the level and the model only shades it. '
               'Station coordinates are approximate (found by geocoding names); only stations with a name match are drawn.')
