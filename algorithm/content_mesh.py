"""Render geometry of the soil carried in the bucket: a lumpy free surface (thin closed slab) whose vertices are
recomputed every frame from the content volume and the bucket orientation (fixed topology, uploaded to the GPU).

Partially filled cavity: the surface is the horizontal chord of the cavity section at the fill level (extruded over the
inner width).  Heaped bucket (content above the struck volume, mouth up): a parabolic heap over the mouth line.
"""
from __future__ import annotations

import numpy as np

from .soil import BucketGeometry, clip_polygon, poly_area_centroid

K, M = 26, 18                  # samples along the section chord and across the width
MOUND = 0.07                   # crest height of the rough mound on the carried soil (m)
THICK = 0.03                   # slab thickness (hidden part below the surface)
INSET = 0.035                  # clearance from the side plates


def _grid_faces():
    faces = []
    idx = lambda layer, i, j: layer * K * M + i * M + j
    for i in range(K - 1):
        for j in range(M - 1):
            a, b, c, d = idx(0, i, j), idx(0, i + 1, j), idx(0, i + 1, j + 1), idx(0, i, j + 1)
            faces += [(a, b, c), (a, c, d)]                      # top
            a, b, c, d = idx(1, i, j), idx(1, i + 1, j), idx(1, i + 1, j + 1), idx(1, i, j + 1)
            faces += [(a, c, b), (a, d, c)]                      # bottom (reversed)
    for i in range(K - 1):                                       # side strips j = 0 and j = M-1
        for j, flip in ((0, True), (M - 1, False)):
            a, b = idx(0, i, j), idx(0, i + 1, j)
            c, d = idx(1, i + 1, j), idx(1, i, j)
            faces += [(a, d, c), (a, c, b)] if flip else [(a, b, c), (a, c, d)]
    for j in range(M - 1):                                       # end strips i = 0 and i = K-1
        for i, flip in ((0, False), (K - 1, True)):
            a, b = idx(0, i, j), idx(0, i, j + 1)
            c, d = idx(1, i, j + 1), idx(1, i, j)
            faces += [(a, d, c), (a, c, b)] if flip else [(a, b, c), (a, c, d)]
    return np.asarray(faces, np.int32)


FACES = _grid_faces()
S = np.linspace(0., 1., K)
Y01 = np.linspace(0., 1., M)
_rng = np.random.default_rng(3)
# fixed lumpy relief (clods) on the surface grid, tapering to zero at the rim
_n = _rng.standard_normal((K, M))
from scipy.ndimage import gaussian_filter
LUMPS = gaussian_filter(_n, 0.8) + 0.45 * _rng.standard_normal((K, M))
LUMPS /= np.abs(LUMPS).max()
TAPER = (np.sin(np.pi * S)[:, None] ** 0.5) * (np.sin(np.pi * Y01)[None, :] ** 0.5)


def texcoords():
    u = 0.02 + 0.07 * np.repeat(S[:, None], M, axis=1)
    v = 0.02 + 0.055 * np.repeat(Y01[None, :], K, axis=0)
    uv = np.stack([u, v], axis=-1).reshape(-1, 2)
    return np.vstack([uv, uv])


def surface_section(b: BucketGeometry, R, volume):
    """Section curve (K x 2, bucket frame (x, z)) of the free surface and the unit 'up' direction of the surface in the
    section, or None when the bucket is empty."""
    if volume <= 1e-5:
        return None
    zc = np.array([R[2, 0], R[2, 2]])
    zn = np.linalg.norm(zc)
    up = zc / zn if zn > 1e-6 else np.array([0., 1.])
    target = volume / b.width
    if target < b.area:
        zv = b.V @ zc
        lo, hi = float(zv.min()), float(zv.max())
        for _ in range(36):
            mid = 0.5 * (lo + hi)
            a, _ = poly_area_centroid(clip_polygon(b.V, zc, mid))
            lo, hi = (mid, hi) if a < target else (lo, mid)
        lev = hi
        poly = clip_polygon(b.V, zc, lev)
        on = np.abs(poly @ zc - lev) < 1e-6
        pts = poly[on]
        if len(pts) < 2:
            return None
        # chord end points: extreme points along the chord direction
        tdir = np.array([up[1], -up[0]])
        s = pts @ tdir
        P1, P2 = pts[np.argmin(s)], pts[np.argmax(s)]
        curve = P1[None, :] + S[:, None] * (P2 - P1)[None, :]
        return curve, up, 0.0
    # heaped: parabolic heap over the mouth line, cross-section area = excess / width
    P1, P2 = b.heel, b.lip
    L = np.linalg.norm(P2 - P1)
    excess = (target - b.area)
    hmax = min(1.5 * excess / max(L, 1e-6), 0.45 * L)
    n_out = b.mouth_normal
    curve = P1[None, :] + S[:, None] * (P2 - P1)[None, :] + (4 * hmax * S * (1 - S))[:, None] * n_out[None, :]
    return curve, n_out, hmax


def vertices_world(b: BucketGeometry, R, pos, volume, lump_amp=0.06):
    """(2 K M) x 3 world vertices of the content slab (collapsed inside the shell when empty)."""
    res = surface_section(b, R, volume)
    W = b.width - 2 * min(INSET, 0.1 * b.width)
    ys = -W / 2 + W * Y01
    ex, ey, ez = R[:, 0], R[:, 1], R[:, 2]
    if res is None:
        c = b.centroid
        p = pos + c[0] * ex + c[1] * ez
        return np.repeat(p[None, :], 2 * K * M, axis=0)
    curve, up, hmax = res
    # orientation: (curve direction) x (+y) must point along 'up' in the section
    t = curve[-1] - curve[0]
    tw = t[0] * ex + t[1] * ez
    upw = up[0] * ex + up[1] * ez
    if np.dot(np.cross(tw, ey), upw) < 0:
        curve = curve[::-1]
    rel = LUMPS * TAPER * lump_amp * (1.0 if volume > 0.05 else volume / 0.05)
    # the loose soil in a bucket stands as a rough mound over the fill level (visual; <3 % of the volume)
    mound = MOUND * min(1.0, volume / 0.4) * (np.sin(np.pi * S)[:, None] * np.sin(np.pi * Y01)[None, :] ** 0.6)
    rel = rel + mound
    top = np.empty((K, M, 3))
    bot = np.empty((K, M, 3))
    for i in range(K):
        base = pos + curve[i, 0] * ex + curve[i, 1] * ez
        top[i] = base[None, :] + ys[:, None] * ey[None, :] + (rel[i][:, None] * upw[None, :])
        bot[i] = base[None, :] + ys[:, None] * ey[None, :] - THICK * upw[None, :]
    return np.vstack([top.reshape(-1, 3), bot.reshape(-1, 3)])


def vertex_normals(V):
    f = FACES
    n = np.cross(V[f[:, 1]] - V[f[:, 0]], V[f[:, 2]] - V[f[:, 0]])
    out = np.zeros_like(V)
    for k in range(3):
        np.add.at(out, f[:, k], n)
    ln = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(ln, 1e-12)


def initial_local(b: BucketGeometry):
    """Compile-time vertices in the bucket frame (half-full cavity, bucket mouth up)."""
    R = np.array([[-1., 0., 0.], [0., 1., 0.], [0., 0., -1.]])           # pitch = pi (mouth up)
    Vw = vertices_world(b, R, np.zeros(3), 0.6 * b.struck_volume)
    return Vw @ R                                                          # world -> bucket frame (R orthonormal)
