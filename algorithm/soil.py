"""Continuum soil with bank (in-situ) and loose layers and an explicit soil-tool wrench.

The tool geometry and pose are supplied by the caller; no robot model is included.

  Geometry  - The bucket is the prism of its convex outer section extruded over its width.  At every step each
              heightfield column that the prism intersects below the soil surface is resolved: soil in columns that
              enter through the open mouth while the mouth advances into the soil is transferred into the bucket
              (bank volume x swell factor = loose volume); all other overlap is displaced ahead of the moving shell to
              the first cells outside the bucket footprint (Sumner, O'Brien & Hodgins 1999), and the surface relaxes
              to the angle of repose by cellular avalanching (loose soil at its repose angle, cut bank walls at a
              steeper angle; failed bank soil loosens).
  Cutting   - Resistance of the leading face (cutting plate at the lip when scooping, the leading shell face when
              dozing) from the fundamental earthmoving equation (Reece 1964; McKyes 1985):
                  P = w (gamma d^2 N_g + c d N_c + c_a d N_a + q d N_q),  beta = argmin P,
                  H = P sin(rho + delta) + c_a w d cot(rho),   V = P cos(rho + delta) - c_a w d,
              evaluated per strip across the width; H opposes the horizontal edge velocity, V acts downward.
  Support   - Downward-facing shell parts rest on the soil through a Winkler foundation (subgrade modulus k) capped by
              a bearing pressure (elastic-plastic); the part of the indentation beyond the cap yields (the soil is
              displaced), with Coulomb friction tan(delta) opposing the tangential velocity.
  Content   - Loose volume V in the bucket.  While the mouth is open to the air, the retained region is the cavity below
              the horizontal plane through the lowest mouth point (+ a repose heap when the mouth faces up); the excess
              spills over that point as clods that fly ballistically and are deposited into the heightfield where they
              land.  The content's mass, centroid and inertia are returned for integration into the caller's
              multibody model.
  Accounting- Mass = rho_bank * bank volume + rho_loose * (loose + bucket + airborne + spill buffer) volume, conserved
              to round-off by construction (every transfer converts bank volume with the swell factor).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict

import numpy as np

from .terrain import GridSpec, Z_BASE

G = 9.80665


@dataclass
class SoilParams:
    rho_bank: float = 1600.0              # in-situ bulk density (kg/m3); Engineering ToolBox: common soils 1.2-1.6 t/m3
    swell: float = 1.25                   # loose/bank volume ratio (25 % bulking; earth 20-30 %)
    phi_bank_deg: float = 37.0            # internal friction angle, 'dirt-2' (Servin et al. 2021, Table 1)
    c_bank: float = 4.9e3                 # cohesion (Pa), 'dirt-2'
    phi_loose_deg: float = 35.0           # loose, remoulded soil (assumed)
    c_loose: float = 0.3e3                # (assumed)
    adhesion_ratio: float = 0.5           # c_a / c (assumed)
    delta_ratio: float = 0.65             # soil-steel friction angle / phi (assumed, typical 0.5-0.8)
    repose_loose_deg: float = 35.0        # avalanching slope of loose soil (~ phi_loose)
    repose_bank_deg: float = 70.0         # cut walls of cohesive bank soil stand steeper (heightfield limit)
    k_winkler: float = 2.0e7              # subgrade modulus (N/m3), medium soil
    p_bearing_bank: float = 3.0e5         # bearing pressure cap of bank soil (Pa)
    p_bearing_loose: float = 0.8e5        # bearing pressure cap of loose soil (Pa)
    c_winkler: float = 2.0e5              # contact damping pressure per unit penetration speed (Pa s/m)
    v_reg: float = 0.03                   # speed regularisation of the cutting resistance (m/s)
    clod_volume: float = 0.0008           # loose volume carried by one airborne clod (m3)
    spill_rate: float = 1.5               # maximum spill rate over the lip (m3/s)
    avalanche_iters: int = 2
    relax_every: int = 4                  # physics steps between avalanche sweeps

    @property
    def rho_loose(self):
        return self.rho_bank / self.swell


# ---------------------------------------------------------------------------------------------- earthmoving equation
def fee_factors(rho, beta, phi, delta):
    """McKyes (1985) N-factors for a flat soil surface (alpha = 0); rho may broadcast against beta."""
    den = np.cos(rho + delta) + np.sin(rho + delta) / np.tan(beta + phi)
    Ng = (1 / np.tan(rho) + 1 / np.tan(beta)) / (2 * den)
    Nc = (1 + 1 / np.tan(beta) / np.tan(beta + phi)) / den
    Na = (1 - 1 / np.tan(rho) / np.tan(beta + phi)) / den
    Nq = (1 / np.tan(rho) + 1 / np.tan(beta)) / den
    return Ng, Nc, Na, Nq, den


BETAS = np.deg2rad(np.arange(3.0, 88.0, 1.0))


def fee_force(d, w, rho, gamma, c, ca, phi, delta, q=0.0):
    """Vectorised over strips: arrays d, rho, gamma, c, ca, phi, delta (same shape) -> P, H, V, beta (same shape).

    P is minimised over the failure-plane angle beta (passive failure).  H is the horizontal force opposing the tool
    motion, V the vertical force pushing the tool down (negative = lifting)."""
    d = np.atleast_1d(np.asarray(d, float))
    shp = d.shape
    args = [np.broadcast_to(np.asarray(a, float), shp).reshape(-1, 1) for a in (w, rho, gamma, c, ca, phi, delta, q)]
    w, rho, gamma, c, ca, phi, delta, q = args
    rho = np.clip(rho, np.deg2rad(8.0), np.deg2rad(90.0))
    dd = d.reshape(-1, 1)
    Ng, Nc, Na, Nq, den = fee_factors(rho, BETAS[None, :], phi, delta)
    P = w * (gamma * dd * dd * Ng + c * dd * Nc + ca * dd * Na + q * dd * Nq)
    valid = (den > 1e-6) & np.isfinite(P) & (P > 0) & (BETAS[None, :] + phi < np.pi - 1e-3)
    Pm = np.where(valid, P, np.inf)
    k = np.argmin(Pm, axis=1)
    Pmin = Pm[np.arange(len(k)), k]
    ok = np.isfinite(Pmin) & (dd[:, 0] > 0)
    Pmin = np.where(ok, Pmin, 0.)
    r, dl, cal, wl = rho[:, 0], delta[:, 0], ca[:, 0], w[:, 0]
    H = np.where(ok, Pmin * np.sin(r + dl) + cal * wl * dd[:, 0] / np.tan(r), 0.)
    V = np.where(ok, Pmin * np.cos(r + dl) - cal * wl * dd[:, 0], 0.)
    return Pmin.reshape(shp), H.reshape(shp), V.reshape(shp), BETAS[k].reshape(shp)


# ---------------------------------------------------------------------------------------------- 2-D geometry
def convex_section(points_xz, grow=0.0):
    from scipy.spatial import ConvexHull
    P = np.asarray(points_xz, float)
    if grow > 0:
        ang = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        P = (P[:, None, :] + grow * np.c_[np.cos(ang), np.sin(ang)][None]).reshape(-1, 2)
    hull = ConvexHull(P)
    return P[hull.vertices]                                # counter-clockwise


def clip_polygon(V, a, b):
    """Sutherland-Hodgman: keep the part of convex polygon V (k x 2) with a . p <= b."""
    out = []
    n = len(V)
    for i in range(n):
        p, q = V[i], V[(i + 1) % n]
        fp, fq = a @ p - b, a @ q - b
        if fp <= 0:
            out.append(p)
        if fp * fq < 0:
            t = fp / (fp - fq)
            out.append(p + t * (q - p))
    return np.asarray(out) if out else np.zeros((0, 2))


def poly_area_centroid(V):
    if len(V) < 3:
        return 0., (V.mean(axis=0) if len(V) else np.zeros(2))
    x, y = V[:, 0], V[:, 1]
    xs, ys = np.roll(x, -1), np.roll(y, -1)
    cr = x * ys - xs * y
    A = 0.5 * cr.sum()
    if abs(A) < 1e-12:
        return 0., V.mean(axis=0)
    cx = ((x + xs) * cr).sum() / (6 * A)
    cy = ((y + ys) * cr).sum() / (6 * A)
    return abs(A), np.array([cx, cy])


class BucketGeometry:
    """Bucket section (bucket frame: x pivot->lip, z towards the belly/cavity, y lateral) as a convex prism.

    The prism is the convex hull of the outer shell (grown by ``grow``); its mouth is the hull edge between the heel
    and the lip.  The cavity used for the content is the same section (the shell is nearly convex)."""

    def __init__(self, shell_xz, lip, heel, width, grow=0.0):
        shell_xz = np.asarray(shell_xz, float)
        if shell_xz.ndim != 2 or shell_xz.shape[1] != 2 or len(shell_xz) < 3 or not np.isfinite(shell_xz).all():
            raise ValueError('shell_xz must contain at least three finite (x, z) points')
        if not np.isfinite(width) or width <= 0:
            raise ValueError('width must be finite and positive')
        if not np.isfinite(grow) or grow < 0:
            raise ValueError('grow must be finite and nonnegative')
        lip, heel = np.asarray(lip, float), np.asarray(heel, float)
        if lip.shape != (2,) or heel.shape != (2,) or not np.isfinite([lip, heel]).all():
            raise ValueError('lip and heel must be finite (x, z) points')
        self.V = convex_section(shell_xz, grow)
        self.width = float(width)
        k = len(self.V)
        E = np.roll(self.V, -1, axis=0) - self.V
        N = np.c_[E[:, 1], -E[:, 0]]                       # outward normals of a CCW polygon in (x, z)
        N /= np.linalg.norm(N, axis=1, keepdims=True)
        self.N = N
        self.c = np.einsum('ij,ij->i', N, self.V)          # n . p <= c inside
        lip, heel = np.asarray(lip, float), np.asarray(heel, float)
        il = int(np.argmin(np.linalg.norm(self.V - lip, axis=1)))
        ih = int(np.argmin(np.linalg.norm(self.V - heel, axis=1)))
        if (ih + 1) % k == il:
            self.mouth = ih                                 # edge ih -> il
        elif (il + 1) % k == ih:
            self.mouth = il
        else:
            # lip and heel not adjacent on the hull (grown section): collapse the vertices between them
            raise ValueError('mouth vertices are not adjacent on the hull; reduce grow')
        self.lip, self.heel = self.V[il].copy(), self.V[ih].copy()
        self.il, self.ih = il, ih
        self.area, self.centroid = poly_area_centroid(self.V)
        nb = [(il + 1) % k, (il - 1) % k]
        other = [j for j in nb if j != ih][0]
        self.plate = (self.V[other] - self.lip) / np.linalg.norm(self.V[other] - self.lip)
        self.mouth_normal = self.N[self.mouth]

    @property
    def struck_volume(self):
        return self.area * self.width

# ---------------------------------------------------------------------------------------------- soil model
class SoilModel:
    def __init__(self, grid: GridSpec, h, loose, ground, params: SoilParams, bucket: BucketGeometry, site=None, dig=None,
                 seed=5, soil_base=Z_BASE):
        self.g = grid
        self.p = params
        self.b = bucket
        self.h = np.array(h, float)
        self.loose = np.array(loose, float)
        self.ground = np.array(ground, float)
        self.soil_base = float(soil_base)
        if not np.isfinite(self.soil_base):
            raise ValueError('soil_base must be finite')
        for name, array in (('h', self.h), ('loose', self.loose), ('ground', self.ground)):
            if array.shape != (grid.n, grid.n) or not np.isfinite(array).all():
                raise ValueError(name + ' must be finite with shape (grid.n, grid.n)')
        if np.any(self.loose < 0) or np.any(self.h - self.loose < self.soil_base):
            raise ValueError('soil layers must be nonnegative above soil_base')
        self.h_initial = self.h.copy()
        self.n = grid.n
        self.A = grid.dx * grid.dx
        self.content = 0.0                    # loose volume in the bucket (m3)
        self.spill_buffer = 0.0               # spilled volume not yet released as a clod (m3)
        self.clod_pos = np.zeros((0, 3))
        self.clod_vel = np.zeros((0, 3))
        self.clod_vol = np.zeros(0)
        self.clod_src = np.zeros(0, bool)     # released from the bucket (delivery accounting)
        self.clod_id = np.zeros(0, np.int64)  # identity (render: shape and tumbling)
        self.next_clod_id = 0
        self.dirty = np.zeros((self.n, self.n), bool)
        self.touched = np.zeros((self.n, self.n), bool)    # ever disturbed (fresh-soil rendering)
        self.erode_box = None
        self.site = site                      # (cx, cy, hx, hy) accounting footprints
        self.dig = dig
        self.rng = np.random.default_rng(seed)
        self.stats = dict(scooped_bank=0.0, scooped_loose=0.0, displaced=0.0, spilled=0.0, deposited_site=0.0,
                          deposited_elsewhere=0.0, delivered_site=0.0, yielded=0.0, steps=0, relax_moved=0.0)
        self.last = dict(force=np.zeros(3), torque=np.zeros(3), mode='none', depth=0.0, rho=0.0, H=0.0, V=0.0, support=0.0,
                         contact_cells=0, scoop_cells=0, beta=0.0)
        self.mass0 = self.total_mass()
        self._relax_counter = 0
        self.last_scoop_time = -1e9
        self.time = 0.0
        self.relax_activity = 0.0
        self.tan_bank = np.tan(np.deg2rad(params.repose_bank_deg))
        self.tan_loose = np.tan(np.deg2rad(params.repose_loose_deg))
        self.tool_lo = np.full((self.n, self.n), np.inf)
        self._tool_win = None

    # ------------------------------------------------------------------------------------------ bookkeeping
    def bank_volume(self):
        return float(((self.h - self.loose - self.soil_base).sum()) * self.A)

    def loose_volume(self):
        return float(self.loose.sum() * self.A)

    def total_mass(self):
        p = self.p
        return (p.rho_bank * self.bank_volume()
                + p.rho_loose * (self.loose_volume() + self.content + float(self.clod_vol.sum()) + self.spill_buffer))

    def mass_error(self):
        return abs(self.total_mass() - self.mass0)

    # ------------------------------------------------------------------------------------------ grid helpers
    def ij(self, x, y):
        c = (np.asarray(x, float) - self.g.xmin) / self.g.dx
        r = (np.asarray(y, float) - self.g.ymin) / self.g.dx
        return r, c

    def height_at(self, x, y):
        r, c = self.ij(x, y)
        r0 = np.clip(np.floor(r).astype(int), 0, self.n - 2)
        c0 = np.clip(np.floor(c).astype(int), 0, self.n - 2)
        fr, fc = np.clip(r - r0, 0, 1), np.clip(c - c0, 0, 1)
        h = self.h
        return ((1 - fr) * (1 - fc) * h[r0, c0] + (1 - fr) * fc * h[r0, c0 + 1] + fr * (1 - fc) * h[r0 + 1, c0]
                + fr * fc * h[r0 + 1, c0 + 1])

    def _window(self, xmin, xmax, ymin, ymax, pad=2):
        ra, ca = self.ij(xmin, ymin)
        rb, cb = self.ij(xmax, ymax)
        r0 = int(max(0, np.floor(min(ra, rb)) - pad))
        r1 = int(min(self.n, np.ceil(max(ra, rb)) + pad + 1))
        c0 = int(max(0, np.floor(min(ca, cb)) - pad))
        c1 = int(min(self.n, np.ceil(max(ca, cb)) + pad + 1))
        return r0, r1, c0, c1

    def _mark(self, r0, r1, c0, c1):
        r0, c0 = max(0, r0), max(0, c0)
        r1, c1 = min(self.n, r1), min(self.n, c1)
        if self.erode_box is None:
            self.erode_box = [r0, r1, c0, c1]
        else:
            b = self.erode_box
            self.erode_box = [min(b[0], r0), max(b[1], r1), min(b[2], c0), max(b[3], c1)]

    def _remove(self, rr, cc, amount):
        """Remove soil thickness ``amount`` (m) from columns; returns the per-cell loose-equivalent volume (m3)."""
        amount = np.clip(amount, 0., np.maximum(self.h[rr, cc] - self.soil_base, 0.))
        take_loose = np.minimum(amount, self.loose[rr, cc])
        take_bank = amount - take_loose
        self.loose[rr, cc] -= take_loose
        self.h[rr, cc] -= amount
        self.dirty[rr, cc] = True
        self.touched[rr, cc] = True
        return (take_loose + take_bank * self.p.swell) * self.A, float(take_bank.sum() * self.A), float(take_loose.sum() * self.A)

    def _add_loose(self, rr, cc, vol):
        dz = vol / self.A
        np.add.at(self.h, (rr, cc), dz)
        np.add.at(self.loose, (rr, cc), dz)
        self.dirty[rr, cc] = True
        self.touched[rr, cc] = True

    # ------------------------------------------------------------------------------------------ bucket interaction
    def prism_columns(self, R, pos, pad=2):
        """Heightfield columns crossed by the bucket prism: window, lower/upper bounds and the lower-bound face."""
        b = self.b
        ex, ey, ez = R[:, 0], R[:, 1], R[:, 2]
        W = b.width
        sec = pos[None, :] + b.V[:, :1] * ex[None, :] + b.V[:, 1:] * ez[None, :]
        corners = np.vstack([sec - W / 2 * ey[None, :], sec + W / 2 * ey[None, :]])
        r0, r1, c0, c1 = self._window(corners[:, 0].min(), corners[:, 0].max(), corners[:, 1].min(), corners[:, 1].max(), pad=pad)
        if r1 <= r0 or c1 <= c0:
            return None
        self._footprint = corners[:, :2]
        X = self.g.xmin + self.g.dx * np.arange(c0, c1)
        Y = self.g.ymin + self.g.dx * np.arange(r0, r1)
        XX, YY = np.meshgrid(X, Y)
        qx, qy = XX - pos[0], YY - pos[1]
        # a point (x, y, t) of the column has body coordinates u = u0 + t ex_z, w = w0 + t ez_z, s = s0 + t ey_z
        u0 = qx * ex[0] + qy * ex[1] - pos[2] * ex[2]
        w0 = qx * ez[0] + qy * ez[1] - pos[2] * ez[2]
        s0 = qx * ey[0] + qy * ey[1] - pos[2] * ey[2]
        lo = np.full(XX.shape, -np.inf)
        hi = np.full(XX.shape, np.inf)
        bind = np.full(XX.shape, -1)
        bhi = np.full(XX.shape, -1)
        if abs(ey[2]) > 1e-9:
            ta = (-W / 2 - s0) / ey[2]
            tb = (W / 2 - s0) / ey[2]
            lo = np.maximum(lo, np.minimum(ta, tb))
            hi = np.minimum(hi, np.maximum(ta, tb))
            bind[:] = -2                                                 # side plates bound first
            bhi[:] = -2
        else:
            out = np.abs(s0) > W / 2
            lo[out], hi[out] = np.inf, -np.inf
        for k in range(len(b.V)):
            nx, nz = b.N[k]
            coef = nx * ex[2] + nz * ez[2]
            rhs = b.c[k] - (nx * u0 + nz * w0)
            if abs(coef) < 1e-12:
                bad = rhs < 0
                lo[bad], hi[bad] = np.inf, -np.inf
                continue
            t = rhs / coef
            if coef > 0:
                upd = t < hi
                bhi = np.where(upd, k, bhi)
                hi = np.minimum(hi, t)
            else:
                upd = t > lo
                bind = np.where(upd, k, bind)
                lo = np.maximum(lo, t)
        return (r0, r1, c0, c1), XX, YY, lo, hi, bind, bhi

    def step(self, dt, R, pos, v_origin, omega, allow_scoop=True):
        """Soil-bucket interaction for one physics step.

        R, pos: bucket body rotation (columns = body axes in world) and origin (world); v_origin, omega: velocity of the
        body origin and angular velocity (world).  Returns (force, torque about the body origin) on the bucket (world).
        """
        p, b = self.p, self.b
        self.stats['steps'] += 1
        self.time += dt
        F = np.zeros(3)
        T = np.zeros(3)
        self.last.update(force=F, torque=T, mode='none', depth=0.0, H=0.0, V=0.0, support=0.0, contact_cells=0, scoop_cells=0)
        ex, ez = R[:, 0], R[:, 2]
        W = b.width
        sec = pos[None, :] + b.V[:, :1] * ex[None, :] + b.V[:, 1:] * ez[None, :]
        zmin = sec[:, 2].min() - W / 2 * abs(R[2, 1])
        # cheap rejection: bucket above the soil in its bounding window
        wr = self._window(sec[:, 0].min() - W, sec[:, 0].max() + W, sec[:, 1].min() - W, sec[:, 1].max() + W, pad=1)
        if wr[1] <= wr[0] or wr[3] <= wr[2] or zmin > self.h[wr[0]:wr[1], wr[2]:wr[3]].max() + 0.005:
            self.clear_tool()
            return F, T
        cols = self.prism_columns(R, pos)
        if cols is None:
            return F, T
        (r0, r1, c0, c1), XX, YY, lo, hi, bind, bhi = cols
        hw = self.h[r0:r1, c0:c1]
        footprint = lo < hi
        self._set_tool((r0, r1, c0, c1), np.where(footprint, lo, np.inf))
        inside = footprint & (hw > lo)
        m_out = b.mouth_normal[0] * ex + b.mouth_normal[1] * ez
        lipw = pos + b.lip[0] * ex + b.lip[1] * ez
        v_lip = v_origin + np.cross(omega, lipw - pos)
        mouth_adv = bool(v_lip @ m_out > 0.02) and bool(allow_scoop) and self.content < b.struck_volume
        if not inside.any():
            # no overlap at this instant: the blade may still be pressed against the soil face ahead of it
            Fc, Tc = self._cutting(R, pos, v_origin, omega, scooping=mouth_adv)
            self.last['force'], self.last['torque'] = Fc, Tc
            return Fc, Tc
        rr, cc = np.nonzero(inside)
        t_lo = lo[rr, cc]
        t_hi = hi[rr, cc]
        hcol = hw[rr, cc]
        ov = hcol - t_lo                                    # all soil above the lower prism bound (no overhangs)
        k_lo = bind[rr, cc]
        k_hi = bhi[rr, cc]
        R_abs, C_abs = rr + r0, cc + c0
        xs, ys = XX[rr, cc], YY[rr, cc]
        pts = np.c_[xs, ys, t_lo]
        vpts = v_origin[None, :] + np.cross(omega[None, :], pts - pos[None, :])
        # outward face normals (world); side plates: +/- ey
        kk = np.clip(k_lo, 0, len(b.V) - 1)
        Nw = b.N[kk]
        nwo = Nw[:, :1] * ex[None, :] + Nw[:, 1:] * ez[None, :]
        side = k_lo == -2
        if side.any():
            sgn = np.sign(((pts[side] - pos) @ R[:, 1]))
            nwo[side] = sgn[:, None] * R[:, 1][None, :]
        # ------------------------------------------------ scooping through the mouth
        # soil enters the cavity through the open mouth when the mouth plane bounds the column from below (the mouth
        # advances into the soil) or from above with soil standing above it (the cut layer in front of an advancing or
        # upward-facing mouth falls into the cavity); the whole column above the lower bound is taken (no overhangs)
        mouth_lo = k_lo == b.mouth
        mouth_hi = (k_hi == b.mouth) & (hcol >= t_hi - 1e-9)
        p_m = np.where(mouth_lo[:, None], pts, np.c_[xs, ys, np.minimum(t_hi, hcol)])
        v_m = v_origin[None, :] + np.cross(omega[None, :], p_m - pos[None, :])
        advancing = (v_m @ m_out) > 0.02
        cand = (mouth_lo & advancing) | (mouth_hi & (advancing | (m_out[2] > 0.25)))
        room = self.content < b.struck_volume
        scoop = cand & bool(allow_scoop) & room
        n_scoop = int(scoop.sum())
        if n_scoop:
            take = ov[scoop]
            lf = np.minimum(take, self.loose[R_abs[scoop], C_abs[scoop]])
            vol_in = float(((take - lf) * p.swell + lf).sum() * self.A)
            room_v = b.struck_volume - self.content
            if vol_in > room_v:
                take = take * max(room_v, 0.) / vol_in           # fill exactly to the struck volume
            _, vb, vl = self._remove(R_abs[scoop], C_abs[scoop], take)
            self.content += vb * p.swell + vl
            self.stats['scooped_bank'] += vb
            self.stats['scooped_loose'] += vl
            self.last_scoop_time = self.time
        is_mouth = mouth_lo | mouth_hi
        # ------------------------------------------------ shell contact: Winkler support + yield, or displacement
        shell = ~scoop
        down = (nwo[:, 2] < -0.35) & shell & ~cand          # a full bucket pushes the soil in front of its mouth
        push = shell & ~down
        if down.any():
            idx = np.nonzero(down)[0]
            vn = -vpts[idx, 2]                               # penetration speed (positive downward)
            lf = np.clip(self.loose[R_abs[idx], C_abs[idx]] / 0.10, 0, 1)
            pcap = (1 - lf) * p.p_bearing_bank + lf * p.p_bearing_loose
            pres = np.minimum(p.k_winkler * ov[idx], pcap) + p.c_winkler * np.clip(vn, 0, None) * (ov[idx] > 0.002)
            fz = np.clip(pres, 0, None) * self.A
            vt = vpts[idx].copy()
            vt[:, 2] = 0.
            sp = np.linalg.norm(vt, axis=1)
            mu = np.tan(np.deg2rad(p.delta_ratio * self._phi(R_abs[idx], C_abs[idx])))
            ft = -(mu * fz * np.tanh(sp / p.v_reg))[:, None] * vt / np.maximum(sp, 1e-9)[:, None]
            fcell = ft
            fcell[:, 2] += fz
            F += fcell.sum(axis=0)
            T += np.cross(pts[idx] - pos, fcell).sum(axis=0)
            self.last['support'] = float(fz.sum())
            y_ov = ov[idx] - pcap / p.k_winkler             # plastic yield beyond the bearing cap
            yl = y_ov > 0
            if yl.any():
                ii = idx[yl]
                self._displace(R_abs[ii], C_abs[ii], y_ov[yl], vpts[ii], xs[ii], ys[ii])
                self.stats['yielded'] += float(y_ov[yl].sum() * self.A)
        if push.any():
            ii = np.nonzero(push)[0]
            self._displace(R_abs[ii], C_abs[ii], ov[ii], vpts[ii], xs[ii], ys[ii])
        self.last['contact_cells'] = int(len(rr))
        self.last['scoop_cells'] = n_scoop
        # ------------------------------------------------ cutting / dozing resistance (fundamental earthmoving equation)
        mouth_active = bool((is_mouth & advancing).any())
        Fc, Tc = self._cutting(R, pos, v_origin, omega, scooping=n_scoop > 0 or mouth_active or mouth_adv)
        F += Fc
        T += Tc
        self._mark(r0 - 2, r1 + 2, c0 - 2, c1 + 2)
        self.last['force'], self.last['torque'] = F, T
        return F, T

    def _set_tool(self, win, lo):
        """Lower surface of the bucket over its footprint (+inf elsewhere): avalanching may not fill cells above it."""
        if self._tool_win is not None:
            a0, a1, b0, b1 = self._tool_win
            self.tool_lo[a0:a1, b0:b1] = np.inf
        r0, r1, c0, c1 = win
        self.tool_lo[r0:r1, c0:c1] = lo
        self._tool_win = win

    def clear_tool(self):
        if self._tool_win is not None:
            a0, a1, b0, b1 = self._tool_win
            self.tool_lo[a0:a1, b0:b1] = np.inf
            self._tool_win = None

    def _phi(self, rr, cc):
        loose_frac = np.clip(self.loose[rr, cc] / 0.05, 0, 1)
        return (1 - loose_frac) * self.p.phi_bank_deg + loose_frac * self.p.phi_loose_deg

    def _soil_props(self, x, y, zb):
        """Soil constants of cuts whose bottoms are at zb (loose layer fraction of the cut thickness)."""
        r, c = self.ij(x, y)
        r = np.clip(np.round(r).astype(int), 0, self.n - 1)
        c = np.clip(np.round(c).astype(int), 0, self.n - 1)
        hs = self.h[r, c]
        thick = np.maximum(hs - zb, 1e-6)
        fl = np.clip(self.loose[r, c] / thick, 0, 1)
        p = self.p
        phi = np.deg2rad((1 - fl) * p.phi_bank_deg + fl * p.phi_loose_deg)
        coh = (1 - fl) * p.c_bank + fl * p.c_loose
        rho = (1 - fl) * p.rho_bank + fl * p.rho_loose
        return phi, coh, rho

    def _cutting(self, R, pos, v_origin, omega, scooping):
        p, b = self.p, self.b
        ex, ey, ez = R[:, 0], R[:, 1], R[:, 2]
        W = b.width
        if scooping:
            e2, plate = b.lip, b.plate
        else:
            # dozing: the lowest hull vertex leads; its adjacent face that faces the motion is the blade
            hz = [(pos + v[0] * ex + v[1] * ez)[2] for v in b.V]
            i = int(np.argmin(hz))
            e2 = b.V[i]
            plate = None
        pe = pos + e2[0] * ex + e2[1] * ez
        ve = v_origin + np.cross(omega, pe - pos)
        vh = np.array([ve[0], ve[1], 0.])
        sp = float(np.linalg.norm(vh))
        if sp < 1e-4:
            return np.zeros(3), np.zeros(3)
        u = vh / sp
        if plate is None:
            best = -np.inf
            for j in ((i + 1) % len(b.V), (i - 1) % len(b.V)):
                dvec = b.V[j] - e2
                dvec = dvec / np.linalg.norm(dvec)
                dw = dvec[0] * ex + dvec[1] * ez
                score = -(dw @ u)
                if score > best:
                    best, plate = score, dvec
        dw = plate[0] * ex + plate[1] * ez                 # blade direction from the edge upwards/backwards
        back = float(-(dw @ u))
        up = float(dw[2])
        rho = float(np.arctan2(up, back)) if up > 0 else np.deg2rad(8.0)
        rho = float(np.clip(rho, np.deg2rad(8.0), np.deg2rad(90.0)))
        ns = 9
        s = np.linspace(-W / 2 + W / (2 * ns), W / 2 - W / (2 * ns), ns)
        P = pe[None, :] + s[:, None] * ey[None, :]
        ahead = P + u[None, :] * self.g.dx * 1.5
        hs = self.height_at(ahead[:, 0], ahead[:, 1])
        d = hs - P[:, 2]
        act = d > 0.002
        if not act.any():
            return np.zeros(3), np.zeros(3)
        phi, coh, rho_s = self._soil_props(ahead[act, 0], ahead[act, 1], P[act, 2])
        delta = p.delta_ratio * phi
        ca = p.adhesion_ratio * coh
        _, H, V, beta = fee_force(d[act], W / ns, rho, rho_s * G, coh, ca, phi, delta)
        reg = np.tanh(sp / p.v_reg)
        f = reg * (-H[:, None] * u[None, :] + np.c_[np.zeros((len(H), 2)), -V])
        F = f.sum(axis=0)
        T = np.cross(P[act] - pos[None, :], f).sum(axis=0)
        self.last.update(mode='scoop' if scooping else 'doze', depth=float(d[act].mean()), rho=float(np.rad2deg(rho)),
                         H=float(H.sum() * reg), V=float(V.sum() * reg), beta=float(np.rad2deg(beta.mean())))
        return F, T

    def _displace(self, rr, cc, amount, vpts, xs, ys):
        """Move overlapped soil ahead of the moving shell: to the first cells outside the (convex) bucket footprint
        along the horizontal motion (radially from the contact centroid when the motion is vertical)."""
        if len(rr) == 0:
            return
        from scipy.spatial import ConvexHull
        vol, vb, vl = self._remove(rr, cc, amount)
        self.stats['displaced'] += float(vol.sum())
        vh = vpts[:, :2]
        sp = np.linalg.norm(vh, axis=1)
        cen = np.array([xs.mean(), ys.mean()])
        rad = np.c_[xs - cen[0], ys - cen[1]]
        rn = np.linalg.norm(rad, axis=1)
        rad = np.where(rn[:, None] > 1e-6, rad / np.maximum(rn, 1e-9)[:, None], np.array([1., 0.]))
        dirs = np.where(sp[:, None] > 0.01, vh / np.maximum(sp, 1e-9)[:, None], rad)
        P = np.c_[xs, ys]
        try:
            hull = ConvexHull(self._footprint)
            eq = hull.equations                                  # n . p + c <= 0 inside
            nd = dirs @ eq[:, :2].T                              # (cells, edges)
            dist = -(P @ eq[:, :2].T + eq[:, 2][None, :])        # >= 0 inside
            with np.errstate(divide='ignore', invalid='ignore'):
                t = np.where(nd > 1e-9, dist / nd, np.inf)
            texit = np.clip(t.min(axis=1), 0., 4.0)
        except Exception:
            texit = np.zeros(len(rr))
        target = P + dirs * (texit + 0.6 * self.g.dx)[:, None]
        r, c = self.ij(target[:, 0], target[:, 1])
        tr = np.clip(np.round(r).astype(int), 0, self.n - 1)
        tc = np.clip(np.round(c).astype(int), 0, self.n - 1)
        self._add_loose(tr, tc, vol)
        self._mark(int(tr.min()) - 2, int(tr.max()) + 3, int(tc.min()) - 2, int(tc.max()) + 3)

    # ------------------------------------------------------------------------------------------ bucket content
    @staticmethod
    def _vertical(R):
        return np.array([R[2, 0], R[2, 2]])                # world z of the bucket-frame (x, z) section axes

    def spill_point(self, R):
        b = self.b
        zc = self._vertical(R)
        return b.lip if (b.lip @ zc) <= (b.heel @ zc) else b.heel

    def heap_volume(self, R):
        """Repose heap above an upward-facing mouth: triangular section over the horizontal mouth length."""
        b = self.b
        m_out = b.mouth_normal[0] * R[:, 0] + b.mouth_normal[1] * R[:, 2]
        if m_out[2] <= 0.5:
            return 0.0
        dv = b.lip - b.heel
        wv = dv[0] * R[:, 0] + dv[1] * R[:, 2]
        L = float(np.hypot(wv[0], wv[1]))
        return L * L * np.tan(np.deg2rad(self.p.repose_loose_deg)) / 4 * min(1.0, (m_out[2] - 0.5) / 0.4) * b.width * 0.85

    def capacity(self, R):
        """Loose volume the bucket retains at orientation R when its mouth is free (cavity below the spill point
        + repose heap on an upward-facing mouth)."""
        b = self.b
        zc = self._vertical(R)
        z_spill = min(b.lip @ zc, b.heel @ zc)
        area, _ = poly_area_centroid(clip_polygon(b.V, zc, z_spill))
        return area * b.width + self.heap_volume(R)

    def fill_level(self, R, volume=None):
        """Level (world-vertical coordinate in the section, relative to the bucket origin) that holds ``volume`` in
        the cavity filled from its lowest point; None if the cavity (without heap) is overfull."""
        b = self.b
        vol = self.content if volume is None else volume
        zc = self._vertical(R)
        zv = b.V @ zc
        target = vol / b.width
        if target <= 0:
            return float(zv.min())
        if target >= b.area:
            return None
        lo, hi = float(zv.min()), float(zv.max())
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            a, _ = poly_area_centroid(clip_polygon(b.V, zc, mid))
            if a < target:
                lo = mid
            else:
                hi = mid
        return hi

    def fill_region(self, R):
        """Section polygon (bucket frame) occupied by the content (cavity part; the heap is not included)."""
        b = self.b
        if self.content <= 1e-9:
            return np.zeros((0, 2)), None
        zc = self._vertical(R)
        lev = self.fill_level(R)
        if lev is None:
            return b.V.copy(), float((b.V @ zc).max())
        return clip_polygon(b.V, zc, lev), lev

    def content_mass_properties(self, R):
        """Mass (kg), centroid (bucket frame) and inertia about the centroid (bucket frame) of the content."""
        m = self.content * self.p.rho_loose
        if m <= 0:
            return 0., np.zeros(3), np.zeros((3, 3))
        poly, _ = self.fill_region(R)
        if len(poly) >= 3:
            a, cxz = poly_area_centroid(poly)
            # second moments of the section polygon about its centroid (exact for a polygon)
            x, z = poly[:, 0] - cxz[0], poly[:, 1] - cxz[1]
            xs, zs = np.roll(x, -1), np.roll(z, -1)
            cr = x * zs - xs * z
            sgn = 1.0 if cr.sum() >= 0 else -1.0                               # orientation of the clipped polygon
            Ixx_s = sgn * ((z * z + z * zs + zs * zs) * cr).sum() / 12          # int z^2 dA about the centroid
            Izz_s = sgn * ((x * x + x * xs + xs * xs) * cr).sum() / 12          # int x^2 dA
            Ixz_s = sgn * ((x * zs + 2 * x * z + 2 * xs * zs + xs * z) * cr).sum() / 24   # int x z dA
            aa = max(a, 1e-9)
            kx2, kz2, kxz = Izz_s / aa, Ixx_s / aa, Ixz_s / aa                 # per unit area
            ky2 = self.b.width ** 2 / 12
            I = m * np.array([[ky2 + kz2, 0., -kxz], [0., kx2 + kz2, 0.], [-kxz, 0., kx2 + ky2]])
        else:
            cxz = self.b.centroid
            I = m * 0.01 * np.eye(3)
        return m, np.array([cxz[0], 0., cxz[1]]), I

    def mouth_blocked(self, R, pos):
        """The spill point is buried (soil ahead of the mouth holds the content) or the bucket has just scooped."""
        if self.time - self.last_scoop_time < 0.3:
            return True
        e = self.spill_point(R)
        pw = pos + e[0] * R[:, 0] + e[1] * R[:, 2]
        return bool(pw[2] < self.height_at(pw[0], pw[1]) + 0.03)

    def spill(self, dt, R, pos, v_origin, omega):
        """Release content above the retained capacity over the spill point as airborne clods."""
        b = self.b
        if self.content <= 1e-9 and self.spill_buffer > 1e-12 and not self.mouth_blocked(R, pos):
            self.spill_buffer, vol = 0.0, self.spill_buffer
            e = self.spill_point(R)
            P = (pos + e[0] * R[:, 0] + e[1] * R[:, 2])[None, :]
            self.clod_pos = np.vstack([self.clod_pos, P])
            self.clod_vel = np.vstack([self.clod_vel, (v_origin + np.cross(omega, P[0] - pos))[None, :]])
            self.clod_vol = np.r_[self.clod_vol, vol]
            self.clod_src = np.r_[self.clod_src, True]
            self.clod_id = np.r_[self.clod_id, self.next_clod_id]
            self.next_clod_id += 1
            return 0.0
        if self.content <= 1e-9 or self.mouth_blocked(R, pos):
            return 0.0
        excess = self.content - self.capacity(R)
        if excess <= 1e-9:
            return 0.0
        amount = min(excess, self.p.spill_rate * dt)
        self.content -= amount
        self.spill_buffer += amount
        self.stats['spilled'] += amount
        n = int(self.spill_buffer / self.p.clod_volume)
        if n <= 0:
            return amount
        vols = np.full(n, self.p.clod_volume)
        self.spill_buffer -= vols.sum()
        ex, ey, ez = R[:, 0], R[:, 1], R[:, 2]
        e = self.spill_point(R)
        m_out = b.mouth_normal[0] * ex + b.mouth_normal[1] * ez
        s = self.rng.uniform(-b.width / 2 * 0.92, b.width / 2 * 0.92, n)
        base = pos + e[0] * ex + e[1] * ez
        P = base[None, :] + s[:, None] * ey[None, :] + 0.03 * m_out[None, :] - 0.02 * np.array([0, 0, 1.])
        V = (v_origin[None, :] + np.cross(omega[None, :], P - pos[None, :]) + 0.3 * m_out[None, :]
             + self.rng.normal(0, 0.10, (n, 3)))
        self.clod_pos = np.vstack([self.clod_pos, P])
        self.clod_vel = np.vstack([self.clod_vel, V])
        self.clod_vol = np.r_[self.clod_vol, vols]
        self.clod_src = np.r_[self.clod_src, np.ones(n, bool)]
        self.clod_id = np.r_[self.clod_id, self.next_clod_id + np.arange(n)]
        self.next_clod_id += n
        return amount

    def advance_clods(self, dt):
        if len(self.clod_vol) == 0:
            return
        self.clod_vel[:, 2] -= G * dt
        self.clod_pos += self.clod_vel * dt
        hz = self.height_at(self.clod_pos[:, 0], self.clod_pos[:, 1])
        land = self.clod_pos[:, 2] <= hz
        if land.any():
            P = self.clod_pos[land]
            vol = self.clod_vol[land]
            src = self.clod_src[land]
            r, c = self.ij(P[:, 0], P[:, 1])
            r = np.clip(np.round(r).astype(int), 1, self.n - 2)
            c = np.clip(np.round(c).astype(int), 1, self.n - 2)
            w = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], float)
            w /= w.sum()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    self._add_loose(r + dr, c + dc, vol * w[dr + 1, dc + 1])
            self._mark(int(r.min()) - 2, int(r.max()) + 3, int(c.min()) - 2, int(c.max()) + 3)
            if self.site is not None:
                inside = self._in_box(P[:, 0], P[:, 1], self.site)
                self.stats['deposited_site'] += float(vol[inside].sum())
                self.stats['delivered_site'] += float(vol[inside & src].sum())
                self.stats['deposited_elsewhere'] += float(vol[~inside].sum())
            keep = ~land
            self.clod_pos, self.clod_vel = self.clod_pos[keep], self.clod_vel[keep]
            self.clod_vol, self.clod_src, self.clod_id = self.clod_vol[keep], self.clod_src[keep], self.clod_id[keep]

    @staticmethod
    def _in_box(x, y, box):
        cx, cy, hx, hy = box
        return (np.abs(x - cx) <= hx) & (np.abs(y - cy) <= hy)

    # ------------------------------------------------------------------------------------------ avalanching
    def relax(self, force=False, iters=None, box=None):
        """Angle-of-repose relaxation (cellular avalanching) on the recently modified box."""
        self._relax_counter += 1
        if box is None:
            if self.erode_box is None:
                return 0.0
            if not force and self._relax_counter % self.p.relax_every:
                return 0.0
            box = self.erode_box
            self.erode_box = None
        iters = iters or self.p.avalanche_iters
        r0, r1, c0, c1 = box
        r0, c0 = max(1, r0 - 4), max(1, c0 - 4)
        r1, c1 = min(self.n - 1, r1 + 4), min(self.n - 1, c1 + 4)
        if r1 - r0 < 2 or c1 - c0 < 2:
            return 0.0
        dx = self.g.dx
        sw = self.p.swell
        h = self.h[r0 - 1:r1 + 1, c0 - 1:c1 + 1]
        L = self.loose[r0 - 1:r1 + 1, c0 - 1:c1 + 1]
        TL = self.tool_lo[r0 - 1:r1 + 1, c0 - 1:c1 + 1]
        h_before = h.copy()
        tool = bool(np.isfinite(TL).any())
        moved = 0.0
        H0, W0 = h.shape
        for _ in range(iters):
            for (dr, dc) in ((0, 1), (1, 0), (1, 1), (1, -1)):
                dist = dx * np.hypot(dr, dc)
                ra = slice(1, H0 - 1)
                ca = slice(1, W0 - 1)
                rb = slice(1 + dr, H0 - 1 + dr)
                cb = slice(1 + dc, W0 - 1 + dc)
                diff = h[ra, ca] - h[rb, cb]
                if tool:
                    room_a = np.where(np.isfinite(TL[ra, ca]), 0., np.inf)  # nothing flows in under the bucket
                    room_b = np.where(np.isfinite(TL[rb, cb]), 0., np.inf)
                # loose soil slides down in either direction beyond the repose slope
                exl = np.abs(diff) - self.tan_loose * dist
                act = exl > 1e-5
                if act.any():
                    from_a = diff > 0
                    avail = np.where(from_a, L[ra, ca], L[rb, cb])
                    amt = np.minimum(0.25 * exl * act, 0.5 * avail)
                    if tool:
                        amt = np.minimum(amt, 0.5 * np.where(from_a, room_b, room_a))
                    s = np.where(from_a, amt, -amt)
                    h[ra, ca] -= s
                    h[rb, cb] += s
                    L[ra, ca] -= s
                    L[rb, cb] += s
                    moved += float(amt.sum())
                    diff = h[ra, ca] - h[rb, cb]
                # bank walls (bank surface h - loose) beyond the steep limit fail; the failed bank soil loosens (swell)
                bank_a = h[ra, ca] - L[ra, ca] - self.soil_base
                bank_b = h[rb, cb] - L[rb, cb] - self.soil_base
                dbank = bank_a - bank_b
                exb = np.abs(dbank) - self.tan_bank * dist
                actb = exb > 1e-5
                if actb.any():
                    from_a = dbank > 0
                    amt = np.minimum(0.25 * exb * actb / sw, 0.5 * np.where(from_a, bank_a, bank_b))
                    if tool:
                        amt = np.minimum(amt, 0.5 * np.where(from_a, room_b, room_a) / sw)
                    amt = np.maximum(amt, 0.)
                    ta = np.where(from_a, amt, 0.)
                    tb = np.where(from_a, 0., amt)
                    h[ra, ca] += -ta + tb * sw
                    h[rb, cb] += -tb + ta * sw
                    L[ra, ca] += tb * sw
                    L[rb, cb] += ta * sw
                    moved += float(amt.sum())
        np.clip(L, 0., None, out=L)
        self.relax_activity = moved * self.A
        self.stats['relax_moved'] += moved * self.A
        if moved * self.A > 1e-6:
            changed = np.abs(h - h_before) > 1e-7
            self.dirty[r0 - 1:r1 + 1, c0 - 1:c1 + 1] |= changed
            self.touched[r0 - 1:r1 + 1, c0 - 1:c1 + 1] |= changed & (np.abs(h - self.h_initial[r0 - 1:r1 + 1, c0 - 1:c1 + 1]) > 0.004)
            rows = np.nonzero(changed.any(axis=1))[0]
            cols_ = np.nonzero(changed.any(axis=0))[0]
            if len(rows):
                self._mark(r0 - 1 + rows[0], r0 + rows[-1] + 1, c0 - 1 + cols_[0], c0 + cols_[-1] + 1)   # keep relaxing while moving
        return moved * self.A

    def settle(self, box=None, max_sweeps=4000, tol=1e-7):
        """Relax a region to rest (initial pile, tests)."""
        if box is None:
            box = [1, self.n - 1, 1, self.n - 1]
        for k in range(max_sweeps):
            moved = self.relax(box=box, iters=1)
            if moved < tol:
                return k
        return max_sweeps

    # ------------------------------------------------------------------------------------------ metrics and state
    def box_slices(self, box):
        cx, cy, hx, hy = box
        r0, r1, c0, c1 = self._window(cx - hx, cx + hx, cy - hy, cy + hy, pad=0)
        return slice(r0, r1), slice(c0, c1)

    def metrics(self, grade=None):
        out = dict(bucket_volume=self.content, bucket_mass=self.content * self.p.rho_loose,
                   airborne_volume=float(self.clod_vol.sum()) + self.spill_buffer, airborne_count=int(len(self.clod_vol)),
                   soil_speed=float(np.linalg.norm(self.clod_vel, axis=1).max()) if len(self.clod_vol) else 0.0,
                   relax_activity=self.relax_activity, mass_error=self.mass_error())
        if self.site is not None:
            rs, cs = self.box_slices(self.site)
            dh = self.h[rs, cs] - self.ground[rs, cs]
            cov = dh > 0.02
            out.update(site_volume=float(np.clip(dh, 0, None).sum() * self.A), site_peak=float(dh.max()),
                       site_rms=float(np.sqrt(np.mean((dh[cov] - dh[cov].mean()) ** 2))) if cov.any() else 0.0,
                       site_coverage=float(cov.mean()), site_mean_height=float(dh[cov].mean()) if cov.any() else 0.0,
                       site_mass=float(np.clip(dh, 0, None).sum() * self.A * self.p.rho_loose))
            if grade is not None:
                out['above_grade_volume'] = float(np.clip(dh - grade, 0, None).sum() * self.A)
        if self.dig is not None:
            rs, cs = self.box_slices(self.dig)
            dd = self.ground[rs, cs] - self.h[rs, cs]
            out.update(pit_volume=float(np.clip(dd, 0, None).sum() * self.A), pit_depth=float(dd.max()),
                       pit_bank_volume=float(np.clip(self.ground[rs, cs] - (self.h[rs, cs] - self.loose[rs, cs]), 0, None).sum() * self.A))
        return out

    def take_dirty(self, tol=3e-4):
        """Cells whose height changed by more than ``tol`` since they were last reported (render diffs)."""
        if not hasattr(self, 'h_rec'):
            self.h_rec = self.h_initial.copy()
        idx = np.flatnonzero(self.dirty)
        self.dirty[:] = False
        if len(idx) == 0:
            return idx.astype(np.int32), np.zeros(0, np.float32)
        hv = self.h.ravel()[idx]
        big = np.abs(hv - self.h_rec.ravel()[idx]) > tol
        idx, hv = idx[big], hv[big]
        self.h_rec.ravel()[idx] = hv
        # cells below the tolerance stay pending for a later report
        return idx.astype(np.int32), hv.astype(np.float32)

    def state(self):
        return dict(h=self.h.copy(), loose=self.loose.copy(), touched=self.touched.copy(), content=np.array(self.content),
                    spill_buffer=np.array(self.spill_buffer), clod_pos=self.clod_pos.copy(), clod_vel=self.clod_vel.copy(),
                    clod_vol=self.clod_vol.copy(), clod_src=self.clod_src.copy(), clod_id=self.clod_id.copy(),
                    next_clod_id=np.array(self.next_clod_id), mass0=np.array(self.mass0),
                    stats=np.array(json.dumps(self.stats)), rng=np.array(json.dumps(self.rng.bit_generator.state)),
                    scalars=np.array([self.time, self.last_scoop_time, self._relax_counter, self.relax_activity]),
                    erode_box=np.array(self.erode_box if self.erode_box is not None else [-1, -1, -1, -1]))

    def restore(self, st):
        self.h[:] = st['h']
        self.loose[:] = st['loose']
        self.touched[:] = st['touched']
        self.content = float(st['content'])
        self.spill_buffer = float(st['spill_buffer'])
        self.clod_pos, self.clod_vel = np.array(st['clod_pos']).reshape(-1, 3), np.array(st['clod_vel']).reshape(-1, 3)
        self.clod_vol, self.clod_src = np.array(st['clod_vol']), np.array(st['clod_src'], bool)
        self.clod_id, self.next_clod_id = np.array(st['clod_id'], np.int64), int(st['next_clod_id'])
        self.mass0 = float(st['mass0'])
        self.stats = json.loads(str(st['stats']))
        self.rng.bit_generator.state = json.loads(str(st['rng']))
        self.time, self.last_scoop_time, rc, self.relax_activity = [float(x) for x in st['scalars']]
        self._relax_counter = int(rc)
        eb = [int(x) for x in st['erode_box']]
        self.erode_box = None if eb[0] < 0 else eb
        self.clear_tool()
        self.dirty[:] = True

    def describe(self):
        return dict(params=asdict(self.p), rho_loose=self.p.rho_loose, soil_base=self.soil_base, grid=dict(center=list(self.g.center), half=self.g.half,
                    dx=self.g.dx, n=self.n), bucket=dict(section=self.b.V.tolist(), width=self.b.width,
                                                          struck_volume=self.b.struck_volume, lip=self.b.lip.tolist(),
                                                          heel=self.b.heel.tolist()))
