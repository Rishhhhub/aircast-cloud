"""Wind-aware spatial features: estimate the AQI at a point from the AQI of surrounding stations, using the wind to decide which stations matter.
Vectorised over many target points at once.

Conventions
  DX, DY : (n, m) displacement in km from each neighbour j to each target i  (east, north)
  wind   : (u, v) = east / north component of the wind vector (km/h, direction the air moves TOWARD)
  A neighbour is UPWIND of the target when its displacement to the target points along the wind (cos > 0)."""
import numpy as np

R_FLOOR = 2.0                      # km, avoids 1/0 for (almost) co-located points
FEATURES = ['idw1', 'idw2', 'near_aqi', 'near_km', 'near3', 'mean150', 'n50', 'n150',
            'up0', 'up1', 'up_n', 'up_km', 'down0', 'cross0', 'src0', 'src1', 'src_mass', 'idw2_lag', 'wind_speed', 'wind_u', 'wind_v']


def pair_geometry(lat_t, lon_t, lat_n, lon_n):
    """Displacement (neighbour -> target) in km for all pairs: returns DX, DY, D with shape (n_targets, n_neighbours)."""
    lat_t, lon_t, lat_n, lon_n = (np.asarray(a, float) for a in (lat_t, lon_t, lat_n, lon_n))
    mid = np.deg2rad((lat_t[:, None] + lat_n[None, :]) / 2)
    DX = (lon_t[:, None] - lon_n[None, :]) * 111.19 * np.cos(mid)
    DY = (lat_t[:, None] - lat_n[None, :]) * 111.19
    return DX, DY, np.hypot(DX, DY)


def _wmean(vals, valid, w):
    num = (np.where(valid, vals[None, :], 0.0) * w * valid).sum(1)
    den = (w * valid).sum(1)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(den > 0, num / den, np.nan), den


def wind_features(DX, DY, D, a0, a1, wu, wv, allowed, tau_h=12.0, src_sigma=40.0):
    """a0, a1: (m,) AQI of the neighbours on the same day / the day before (NaN = missing).  wu, wv: (n,) wind at the targets.
    allowed: (n, m) bool, False = this neighbour must not be used (self, masked radius, held-out stations)."""
    n = DX.shape[0]
    r = np.maximum(D, R_FLOOR)
    sp = np.hypot(wu, wv)
    ux = np.where(sp > 1e-6, wu / np.maximum(sp, 1e-6), 0.0)
    uy = np.where(sp > 1e-6, wv / np.maximum(sp, 1e-6), 0.0)
    cos = (DX * ux[:, None] + DY * uy[:, None]) / r                  # >0: neighbour is upwind
    v0 = allowed & ~np.isnan(a0)[None, :]
    v1 = allowed & ~np.isnan(a1)[None, :]
    F = {}
    F['idw2'], _ = _wmean(a0, v0, 1 / r ** 2)
    F['idw1'], _ = _wmean(a0, v0, 1 / r)
    F['idw2_lag'], _ = _wmean(a1, v1, 1 / r ** 2)
    # nearest valid neighbour and the three nearest
    rm = np.where(v0, D, np.inf)
    j = np.argmin(rm, 1)
    dmin = rm[np.arange(n), j]
    F['near_km'] = np.where(np.isfinite(dmin), dmin, np.nan)
    F['near_aqi'] = np.where(np.isfinite(dmin), a0[j], np.nan)
    k = min(3, D.shape[1])
    idx3 = np.argpartition(rm, k - 1, axis=1)[:, :k]
    r3 = np.take_along_axis(rm, idx3, 1)
    a3 = np.where(np.isfinite(r3), a0[idx3], np.nan)
    with np.errstate(invalid='ignore'):
        F['near3'] = np.nanmean(a3, 1) if np.isfinite(a3).any() else np.full(n, np.nan)
    inside150 = v0 & (D <= 150)
    F['mean150'], _ = _wmean(a0, inside150, np.ones_like(D))
    F['n50'] = (v0 & (D <= 50)).sum(1).astype(float)
    F['n150'] = inside150.sum(1).astype(float)
    # directional cones (within 250 km, at least 5 km away)
    near_ok = (D >= 5) & (D <= 250)
    up = v0 & near_ok & (cos > 0.5)
    F['up0'], _ = _wmean(a0, up, 1 / r)
    F['up1'], _ = _wmean(a1, v1 & near_ok & (cos > 0.5), 1 / r)
    F['up_n'] = up.sum(1).astype(float)
    F['up_km'] = np.where(up.any(1), np.where(up, D, np.inf).min(1), np.nan)
    F['down0'], _ = _wmean(a0, v0 & near_ok & (cos < -0.5), 1 / r)
    F['cross0'], _ = _wmean(a0, v0 & near_ok & (np.abs(cos) <= 0.5), 1 / r)
    # "where did this air come from": kernel around the source point x - wind * tau (travel capped at 250 km)
    L = np.minimum(sp * tau_h, 250.0)
    dxs = -DX + (ux * L)[:, None]                                    # neighbour position relative to the source point
    dys = -DY + (uy * L)[:, None]
    ws = np.exp(-(dxs ** 2 + dys ** 2) / (2 * src_sigma ** 2))
    F['src0'], mass = _wmean(a0, v0, ws)
    F['src1'], _ = _wmean(a1, v1, ws)
    F['src_mass'] = mass
    F['wind_speed'], F['wind_u'], F['wind_v'] = sp, wu, wv
    return F
