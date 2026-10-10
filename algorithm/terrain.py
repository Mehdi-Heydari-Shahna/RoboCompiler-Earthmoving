"""Construction-site terrain: a fine deformable soil heightfield around the machine, a coarse visual surround,
procedural soil textures and the initial soil layout.

Grid convention (verified against MuJoCo 3.3.7 hfield ray casts): array[row, col] with rows along y and columns
along x; vertex (r, c) sits at x = xmin + c * dx, y = ymin + r * dx.  MuJoCo stores normalised elevations in
[0, 1]; the geom is placed at z = Z_BASE and the elevation range is ELEV, so z = Z_BASE + data * ELEV.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

Z_BASE = -1.6          # bottom of the soil column represented by the heightfield (m)
PAD = (-7.0, 2.8, 2.3)  # level working platform: x0, x1, |y| half width (m); the undercarriage rests on it
ELEV = 3.2             # elevation range of the heightfield (m): z in [-1.6, 1.6]


@dataclass
class GridSpec:
    center: tuple = (3.0, 3.0)
    half: float = 7.5
    dx: float = 0.06

    @property
    def n(self):
        return int(round(2 * self.half / self.dx)) + 1

    @property
    def xmin(self):
        return self.center[0] - self.half

    @property
    def ymin(self):
        return self.center[1] - self.half

    def xy(self):
        c = self.xmin + self.dx * np.arange(self.n)
        r = self.ymin + self.dx * np.arange(self.n)
        return np.meshgrid(c, r)                    # X[row, col], Y[row, col]


@dataclass
class SiteLayout:
    site_center: tuple = (6.2, 0.0)                 # levelling site (loose pile)
    pile_radii: tuple = (1.25, 0.95)                # base semi-axes of the initial pile (m)
    pile_height: float = 0.62
    dig_center: tuple = (6.9 * np.cos(np.deg2rad(85.)), 6.9 * np.sin(np.deg2rad(85.)))   # dig area in natural ground
    site_footprint: tuple = (4.4, 3.4)
    dig_footprint: tuple = (3.6, 3.2)
    travel_start: tuple = (-3.5, 0.0)
    seed: int = 7


def value_noise(shape, cells, rng, order=3):
    """Smooth value noise: random grid of size cells upsampled bicubically to shape."""
    from scipy.ndimage import zoom
    cells = max(1, int(cells))
    g = rng.standard_normal((cells + 3, cells + 3))
    z = zoom(g, ((shape[0] + 2 * shape[0] / cells) / g.shape[0], (shape[1] + 2 * shape[1] / cells) / g.shape[1]), order=order)
    oy = (z.shape[0] - shape[0]) // 2
    ox = (z.shape[1] - shape[1]) // 2
    return z[oy:oy + shape[0], ox:ox + shape[1]]


def fbm(shape, octaves, rng):
    """Sum of value-noise octaves: octaves = [(cells, amplitude), ...]."""
    out = np.zeros(shape)
    for cells, amp in octaves:
        n = value_noise(shape, cells, rng)
        out += amp * n / max(1e-9, n.std())
    return out


def initial_terrain(grid: GridSpec, layout: SiteLayout):
    """Ground surface h, loose-layer thickness (m) and the pristine ground h_ground."""
    rng = np.random.default_rng(layout.seed)
    X, Y = grid.xy()
    shape = X.shape
    span = 2 * grid.half
    # gentle natural undulation: ~2.5 cm at 6-8 m, ~1 cm at 2 m, 3 mm micro roughness
    ground = fbm(shape, [(int(span / 7), 0.025), (int(span / 2), 0.010), (int(span / 0.5), 0.003)], rng)
    ground -= np.median(ground)
    # prepared working platform where the machine travels and stands: exactly level (PAD), blending into the
    # natural undulation over 1.5 m
    px0, px1, py = PAD
    d_pad = np.hypot(np.clip(np.maximum(px0 - X, X - px1), 0, None), np.clip(np.abs(Y) - py, 0, None))
    ground *= np.clip(d_pad / 1.5, 0, 1) ** 2
    # taper to zero at the grid border so the surround terrain meets it without a step
    edge = np.minimum.reduce([X - grid.xmin, grid.xmin + span - X, Y - grid.ymin, grid.ymin + span - Y])
    ground *= np.clip(edge / 1.5, 0, 1)
    h = ground.copy()
    loose = np.zeros(shape)
    # loose soil pile on the levelling site: rounded cone shaped by the repose relaxation afterwards
    sx, sy = layout.site_center
    ax, ay = layout.pile_radii
    rr = np.sqrt(((X - sx) / ax) ** 2 + ((Y - sy) / ay) ** 2)
    pile = layout.pile_height * np.clip(1 - rr ** 1.6, 0, None) ** 0.9
    pile *= 1 + 0.10 * value_noise(shape, int(span / 0.6), rng)
    pile = np.clip(pile, 0, None)
    h += pile
    loose += pile
    return h, loose, ground


def outer_terrain(n=257, half=200.0, seed=3, inner_half=12.0, inner_center=(3.0, 3.0), hide_margin=None):
    """Coarse visual surround (hills far away, flat near the work area, slightly below the fine grid)."""
    rng = np.random.default_rng(seed)
    x = np.linspace(-half, half, n)
    X, Y = np.meshgrid(x, x)
    hills = fbm((n, n), [(4, 6.0), (10, 2.0), (30, 0.4)], rng)
    hills -= hills.min()
    r = np.hypot(X - inner_center[0], Y - inner_center[1])
    w = np.clip((r - 24.0) / 90.0, 0, 1) ** 1.5
    z = w * hills + (1 - w) * 0.0
    # a few distant earth stockpiles (construction site cues)
    for cx, cy, R, H in ((-38, 26, 7, 3.2), (-30, 40, 5, 2.2), (42, -30, 9, 3.8), (50, 34, 6, 2.5), (-45, -35, 8, 3.0)):
        z += H * np.clip(1 - np.hypot(X - cx, Y - cy) / R, 0, None) ** 1.2
    cell = 2 * half / (n - 1)
    m = cell if hide_margin is None else hide_margin
    inside = (np.abs(X - inner_center[0]) < inner_half - m) & (np.abs(Y - inner_center[1]) < inner_half - m)
    z[inside] = -1.2                      # hidden below the deformable soil grid (also below any pit)
    return z


# ------------------------------------------------------------------------------------------------ textures
SOIL_BASE = np.array([118, 97, 76], float)          # dry, weathered topsoil (sRGB)
SOIL_FRESH = np.array([92, 72, 55], float)          # freshly cut / dumped moist soil
GRAVEL = np.array([150, 140, 128], float)


def periodic_noise(size, scale_px, rng):
    """Smooth, seamlessly tileable noise (unit std): white noise low-pass filtered in the Fourier domain."""
    f = np.fft.fftfreq(size)
    fx, fy = np.meshgrid(f, f)
    spec = np.fft.fft2(rng.standard_normal((size, size))) * np.exp(-2 * (np.pi * np.hypot(fx, fy) * scale_px) ** 2)
    field = np.real(np.fft.ifft2(spec))
    return field / max(field.std(), 1e-12)


def soil_texture(size=2048, seed=11, fresh=False, meters=15.0):
    """Procedural soil albedo, seamlessly tileable (the terrain textures repeat every ``meters``): multi-scale mottling,
    clods, pebbles and dark organic specks."""
    rng = np.random.default_rng(seed + (100 if fresh else 0))
    base = SOIL_FRESH if fresh else SOIL_BASE
    ppm = size / meters                                    # pixels per metre
    lum = 1.0 + 0.045 * periodic_noise(size, 1.2 * ppm, rng) + 0.06 * periodic_noise(size, 0.25 * ppm, rng) \
        + 0.05 * periodic_noise(size, 0.05 * ppm, rng) + 0.035 * periodic_noise(size, 0.012 * ppm, rng)
    img = base[None, None, :] * lum[..., None]
    hue = periodic_noise(size, 1.5 * ppm, rng)
    img[..., 0] *= 1 + 0.025 * hue
    img[..., 2] *= 1 - 0.03 * hue
    if fresh:
        clods = periodic_noise(size, 0.03 * ppm, rng)
        img *= (1 + 0.12 * np.tanh(1.5 * clods))[..., None]
    # pebbles and small stones (wrapped at the tile edges)
    npeb = int((5000 if not fresh else 2500) * (meters / 15.0) ** 2)
    ys = rng.integers(0, size, npeb)
    xs = rng.integers(0, size, npeb)
    rad = np.clip(rng.gamma(2.0, 0.35, npeb) * ppm * 0.010, 0.8, 7.0)
    shade = rng.uniform(0.75, 1.25, npeb)
    for y, x, r, s in zip(ys, xs, rad, shade):
        r0 = int(np.ceil(r))
        yy, xx = np.mgrid[y - r0:y + r0 + 1, x - r0:x + r0 + 1]
        msk = (yy - y) ** 2 + (xx - x) ** 2 <= r * r
        yy, xx = yy[msk] % size, xx[msk] % size
        col = GRAVEL * s * rng.uniform(0.85, 1.05)
        img[yy, xx] = 0.4 * img[yy, xx] + 0.6 * col
    # dark organic specks (2 px blobs, softened)
    nsp = int(9000 * (meters / 15.0) ** 2)
    ys = rng.integers(0, size, nsp)
    xs = rng.integers(0, size, nsp)
    f = rng.uniform(0.65, 0.85, nsp)[:, None]
    for dy, dx_ in ((0, 0), (0, 1), (1, 0), (1, 1)):
        img[(ys + dy) % size, (xs + dx_) % size] *= f
    return np.clip(img, 0, 255).astype(np.uint8)


def surround_texture(size=1024, seed=5):
    """Albedo for the far terrain: dry soil with sparse vegetation patches."""
    rng = np.random.default_rng(seed)
    m1 = fbm((size, size), [(12, 1.0), (40, 0.6), (160, 0.3)], rng)
    veg = np.clip(fbm((size, size), [(20, 1.0), (80, 0.5)], rng) - 0.6, 0, None)
    soil = SOIL_BASE[None, None, :] * (1 + 0.08 * m1)[..., None]
    green = np.array([112, 108, 78], float)
    w = np.clip(veg * 0.8, 0, 0.35)[..., None]
    img = (1 - w) * soil + w * green[None, None, :] * (1 + 0.1 * m1)[..., None]
    return np.clip(img, 0, 255).astype(np.uint8)


def save_png(arr, path):
    Image.fromarray(arr).save(path)
