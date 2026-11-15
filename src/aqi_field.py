"""Smooth, Apple-Weather-style AQI colour field: scattered station values -> interpolated raster (RGBA PNG) for a map overlay."""
import base64, io
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

STOPS = [(0, '#2ecc71'), (50, '#a8d94c'), (100, '#f6d13a'), (200, '#f39a2d'), (300, '#e5472f'), (400, '#a8245f'), (500, '#5e1a6a')]   # Good ... Severe, blended continuously


def _hex(h):
    return [int(h[i:i + 2], 16) for i in (1, 3, 5)]


def aqi_rgb(v):
    """Continuous colour for AQI values (array) -> uint8 array (..., 3)."""
    xs = [s[0] for s in STOPS]
    cols = np.array([_hex(s[1]) for s in STOPS], dtype=float)
    v = np.clip(v, 0, 500)
    return np.stack([np.interp(v, xs, cols[:, c]) for c in range(3)], axis=-1).astype(np.uint8)


def css_gradient():
    """CSS linear-gradient string matching the map colours (for the legend bar)."""
    return 'linear-gradient(to right, ' + ', '.join(f'{c} {v / 5:.0f}%' for v, c in STOPS) + ')'


def field_overlay(lat, lon, val, bbox=None, n=300, bandwidth_km=4.5, blur=2.0, fade_km=(7.0, 20.0), max_alpha=0.72, pad=0.2, fade_from=None):
    """lat, lon, val: station (or grid-cell) positions and AQI values. fade_from=(lats, lons): make the colour fade with the distance to these points instead (e.g. the real stations). Returns (data-URI PNG, mapbox image coordinates, bbox).
    Gaussian-kernel blending (Nadaraya-Watson) on a fine grid: smooth, broad gradients with no bullseyes around stations; transparent far from any data point."""
    lat, lon, val = (np.asarray(a, dtype=float) for a in (lat, lon, val))
    ok = ~np.isnan(val)
    lat, lon, val = lat[ok], lon[ok], val[ok]
    if bbox is None:
        bbox = (lat.min() - pad, lat.max() + pad, lon.min() - pad, lon.max() + pad)
    la0, la1, lo0, lo1 = bbox
    ky = 111.19
    kx = 111.19 * np.cos(np.deg2rad((la0 + la1) / 2))
    gy = np.linspace(la1, la0, n)
    gx = np.linspace(lo0, lo1, n)
    F = np.empty((n, n))
    dmin = np.empty((n, n))
    for i in range(0, n, 25):                                  # row blocks keep memory small
        LA, LO = np.meshgrid(gy[i:i + 25], gx, indexing='ij')
        d = np.sqrt(((LA[..., None] - lat) * ky) ** 2 + ((LO[..., None] - lon) * kx) ** 2)
        dm = d.min(-1, keepdims=True)
        w = np.exp(-(d ** 2 - dm ** 2) / (2 * bandwidth_km ** 2))   # shifted by the nearest distance for numerical stability
        F[i:i + 25] = (w * val).sum(-1) / w.sum(-1)
        if fade_from is None:
            dmin[i:i + 25] = d.min(-1)
        else:                                                     # transparency follows the distance to the nearest REAL station
            fl, fo = (np.asarray(a, dtype=float) for a in fade_from)
            dmin[i:i + 25] = np.sqrt(((LA[..., None] - fl) * ky) ** 2 + ((LO[..., None] - fo) * kx) ** 2).min(-1)
    F = gaussian_filter(F, blur)
    alpha = np.clip(1 - (dmin - fade_km[0]) / (fade_km[1] - fade_km[0]), 0, 1) * max_alpha
    alpha = gaussian_filter(alpha, 5)
    ramp = np.clip(np.minimum(np.arange(n), n - 1 - np.arange(n)) / (0.14 * n), 0, 1)   # taper to zero at the image border: no hard edges
    ramp = ramp * ramp * (3 - 2 * ramp)
    alpha = alpha * np.outer(ramp, ramp)
    img = np.dstack([aqi_rgb(F), (alpha * 255).astype(np.uint8)])
    buf = io.BytesIO()
    Image.fromarray(img, 'RGBA').save(buf, 'PNG')
    src = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()
    return src, [[lo0, la1], [lo1, la1], [lo1, la0], [lo0, la0]], bbox
