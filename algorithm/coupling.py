"""Tool kinematics, payload coupling, and soil-state recording.

Tool geometry and the MuJoCo model are supplied by the caller. No robot model
or mesh assets are constructed by this module.
"""
from __future__ import annotations

import numpy as np
import mujoco

from .terrain import GridSpec
from .soil import SoilModel

def ground_function(grid: GridSpec, ground):
    ground = np.asarray(ground, float)

    def fn(x, y):
        c = (np.asarray(x, float) - grid.xmin) / grid.dx
        r = (np.asarray(y, float) - grid.ymin) / grid.dx
        r0 = np.clip(np.floor(r).astype(int), 0, grid.n - 2)
        c0 = np.clip(np.floor(c).astype(int), 0, grid.n - 2)
        fr, fc = np.clip(r - r0, 0, 1), np.clip(c - c0, 0, 1)
        g = ground
        return ((1 - fr) * (1 - fc) * g[r0, c0] + (1 - fr) * fc * g[r0, c0 + 1] + fr * (1 - fc) * g[r0 + 1, c0]
                + fr * fc * g[r0 + 1, c0 + 1])
    return fn


class ToolCoupler:
    """Exact tool kinematics of the current state (scratch MjData, the plant data is not touched), the soil wrench
    as a Cartesian force/torque at the tool CoM, and the payload (content) mass properties of the tool body."""

    def __init__(self, m, *, tool_body):
        self.m = m
        self.dk = mujoco.MjData(m)
        self.bid = m.body(tool_body).id
        self.base = dict(mass=float(m.body_mass[self.bid]), ipos=m.body_ipos[self.bid].copy(),
                         iquat=m.body_iquat[self.bid].copy(), inertia=m.body_inertia[self.bid].copy())
        Rb = np.zeros(9)
        mujoco.mju_quat2Mat(Rb, self.base['iquat'])
        Rb = Rb.reshape(3, 3)
        self.I_body = Rb @ np.diag(self.base['inertia']) @ Rb.T     # tool inertia about its CoM, body frame
        chain = []
        b = self.bid
        while True:
            chain.append(b)
            if b == 0:
                break
            b = int(m.body_parentid[b])
        self.ancestors = np.array(chain)
        self.subtree0 = m.body_subtreemass[self.ancestors].copy()
        self.payload = 0.0
        self.energy_injected = 0.0
        self._res = np.zeros(6)

    def kinematics(self, d):
        dk = self.dk
        dk.qpos[:] = d.qpos
        dk.qvel[:] = d.qvel
        mujoco.mj_kinematics(self.m, dk)
        mujoco.mj_comPos(self.m, dk)
        mujoco.mj_comVel(self.m, dk)
        mujoco.mj_objectVelocity(self.m, dk, mujoco.mjtObj.mjOBJ_XBODY, self.bid, self._res, 0)
        R = dk.xmat[self.bid].reshape(3, 3).copy()
        pos = dk.xpos[self.bid].copy()
        return R, pos, self._res[3:].copy(), self._res[:3].copy(), dk.xipos[self.bid].copy()

    def wrench_at_com(self, F, T_origin, pos, com):
        """xfrc_applied row: force and torque about the body CoM (MuJoCo applies xfrc_applied at xipos)."""
        return np.r_[F, T_origin + np.cross(pos - com, F)]

    def _energy(self, d):
        dk = self.dk
        dk.qpos[:] = d.qpos
        dk.qvel[:] = d.qvel
        mujoco.mj_kinematics(self.m, dk)
        mujoco.mj_comPos(self.m, dk)
        mujoco.mj_crb(self.m, dk)
        mujoco.mj_energyPos(self.m, dk)
        mujoco.mj_energyVel(self.m, dk)
        return float(dk.energy.sum())

    def set_payload(self, d, mass, centroid, inertia_c):
        """Tool body = tool + content (rigid payload).  Returns the mechanical energy change of this model update
        at the current state (bookkeeping of the energy balance)."""
        mass = float(mass)
        centroid = np.asarray(centroid, float)
        inertia_c = np.asarray(inertia_c, float)
        if not np.isfinite(mass) or mass < 0 or centroid.shape != (3,) or not np.all(np.isfinite(centroid)):
            raise ValueError('Payload mass and centroid must be finite, with nonnegative mass')
        if inertia_c.shape != (3, 3) or not np.all(np.isfinite(inertia_c)) or not np.allclose(inertia_c, inertia_c.T):
            raise ValueError('Payload inertia must be a finite symmetric 3-by-3 matrix')
        if np.linalg.eigvalsh(inertia_c).min() < -1e-10:
            raise ValueError('Payload inertia must be positive semidefinite')
        m = self.m
        if abs(mass - self.payload) < 1e-9 and mass == 0.0:
            return 0.0
        e0 = self._energy(d)
        mb, cb, Ib = self.base['mass'], self.base['ipos'], self.I_body
        M = mb + mass
        c = (mb * cb + mass * centroid) / M
        def shift(mm, r):
            return mm * ((r @ r) * np.eye(3) - np.outer(r, r))
        I = Ib + shift(mb, cb - c) + (inertia_c + shift(mass, centroid - c) if mass > 0 else 0.)
        I = 0.5 * (I + I.T)
        w, V = np.linalg.eigh(I)
        if np.linalg.det(V) < 0:
            V[:, 0] = -V[:, 0]
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, V.ravel())
        m.body_mass[self.bid] = M
        m.body_ipos[self.bid] = c
        m.body_iquat[self.bid] = q
        m.body_inertia[self.bid] = np.maximum(w, 1e-6)
        m.body_subtreemass[self.ancestors] = self.subtree0 + mass
        self.payload = mass
        de = self._energy(d) - e0
        self.energy_injected += de
        return de


class RenderRecorder:
    """Per-sample data for offline rendering: terrain diffs, airborne clods, tool content volume."""

    def __init__(self):
        self.idx, self.h, self.idx_off = [], [], [0]
        self.clod_p, self.clod_v, self.clod_i, self.clod_off = [], [], [], [0]
        self.content = []

    def sample(self, sm: SoilModel):
        i, h = sm.take_dirty()
        self.idx.append(i)
        self.h.append(h)
        self.idx_off.append(self.idx_off[-1] + len(i))
        self.clod_p.append(sm.clod_pos.astype(np.float32))
        self.clod_v.append(sm.clod_vol.astype(np.float32))
        self.clod_i.append(sm.clod_id.astype(np.int64))
        self.clod_off.append(self.clod_off[-1] + len(sm.clod_vol))
        self.content.append(sm.content)

    def arrays(self, start=0):
        """Arrays of the samples from index ``start`` (checkpoint chunks)."""
        io = np.asarray(self.idx_off[start:]) - self.idx_off[start]
        co = np.asarray(self.clod_off[start:]) - self.clod_off[start]
        cat = lambda lst, dt, shp: np.concatenate(lst[start:]).astype(dt) if len(lst[start:]) else np.zeros(shp, dt)
        return dict(terrain_idx=cat(self.idx, np.int32, (0,)), terrain_h=cat(self.h, np.float32, (0,)), terrain_off=io.astype(np.int64),
                    clod_pos=cat(self.clod_p, np.float32, (0, 3)).reshape(-1, 3), clod_vol=cat(self.clod_v, np.float32, (0,)),
                    clod_id=cat(self.clod_i, np.int64, (0,)),
                    clod_off=co.astype(np.int64), content=np.asarray(self.content[start:], float))

    def __len__(self):
        return len(self.content)


def concat_render_chunks(parts):
    """Join per-chunk render arrays into one set (offsets rebased)."""
    out = dict(terrain_idx=[], terrain_h=[], terrain_off=[np.zeros(1, np.int64)], clod_pos=[], clod_vol=[], clod_id=[],
               clod_off=[np.zeros(1, np.int64)], content=[])
    tb = cb = 0
    for p in parts:
        n = len(p['content'])
        if n == 0:
            continue
        out['terrain_idx'].append(p['terrain_idx'])
        out['terrain_h'].append(p['terrain_h'])
        out['terrain_off'].append(p['terrain_off'][1:n + 1] + tb)
        tb += int(p['terrain_off'][n])
        out['clod_pos'].append(p['clod_pos'])
        out['clod_vol'].append(p['clod_vol'])
        out['clod_id'].append(p['clod_id'])
        out['clod_off'].append(p['clod_off'][1:n + 1] + cb)
        cb += int(p['clod_off'][n])
        out['content'].append(p['content'])
    res = {}
    for k, v in out.items():
        res[k] = np.concatenate(v) if len(v) else np.zeros(0)
    res['clod_pos'] = res['clod_pos'].reshape(-1, 3)
    return res
