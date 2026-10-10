"""Planar reference kinematics with caller-supplied geometry and joint limits."""
from __future__ import annotations

import numpy as np


class PlanarChain:
    """Three rotary joints: two links and a terminal tool frame.

    ``pivot_rz`` is the first joint position. ``link_lengths`` gives the two
    distances between joint pivots. Joint limits and angles are in radians;
    lengths and tool-frame points use the caller's consistent length unit.
    Tool points may be planar ``(r, z)`` or spatial ``(x, y, z)`` coordinates.
    """

    def __init__(self, pivot_rz, link_lengths, joint_lower, joint_upper,
                 *, elbow_sign=-1, limit_margin=np.deg2rad(1.0)):
        self.pivot = np.asarray(pivot_rz, float).copy()
        self.L = np.asarray(link_lengths, float).copy()
        self.lower = np.asarray(joint_lower, float).copy()
        self.upper = np.asarray(joint_upper, float).copy()
        if self.pivot.shape != (2,) or self.L.shape != (2,):
            raise ValueError('Supply a planar pivot and two link lengths')
        if self.lower.shape != (3,) or self.upper.shape != (3,):
            raise ValueError('Supply lower and upper limits for three joints')
        if not all(np.all(np.isfinite(x)) for x in (self.pivot, self.L, self.lower, self.upper)):
            raise ValueError('Geometry and joint limits must be finite')
        if np.any(self.L <= 0) or np.any(self.lower >= self.upper):
            raise ValueError('Link lengths must be positive and joint limits ordered')
        if elbow_sign not in (-1, 1):
            raise ValueError('elbow_sign must be -1 or 1')
        self.elbow_sign = int(elbow_sign)
        self.limit_margin = float(limit_margin)
        if not np.isfinite(self.limit_margin) or self.limit_margin < 0:
            raise ValueError('limit_margin must be finite and nonnegative')
        if np.any(self.lower + self.limit_margin >= self.upper - self.limit_margin):
            raise ValueError('The joint-limit margin leaves no feasible interval')

    @staticmethod
    def _point(point):
        p = np.asarray(point, float)
        if p.shape == (3,):
            p = p[[0, 2]]
        if p.shape != (2,) or not np.all(np.isfinite(p)):
            raise ValueError('Supply a finite planar or spatial tool-frame point')
        return p

    @staticmethod
    def _angles(q3):
        q = np.asarray(q3, float)
        if q.shape != (3,) or not np.all(np.isfinite(q)):
            raise ValueError('Supply three finite joint angles')
        return q

    def fk(self, q3, point):
        """Position ``(r, z)`` and pitch of a caller-supplied tool-frame point."""
        t = np.cumsum(self._angles(q3))
        point = self._point(point)
        p = self.pivot.copy()
        p += self.L[0] * np.array([np.cos(t[0]), np.sin(t[0])])
        p += self.L[1] * np.array([np.cos(t[1]), np.sin(t[1])])
        c, s = np.cos(t[2]), np.sin(t[2])
        p += np.array([c * point[0] - s * point[1], s * point[0] + c * point[1]])
        return p, float(t[2])

    def jacobian(self, q3, point):
        """Central-difference derivative of position and pitch."""
        q3 = self._angles(q3)
        J = np.zeros((3, 3))
        e = 1e-7
        for i in range(3):
            dq = np.zeros(3)
            dq[i] = e
            pp, phip = self.fk(q3 + dq, point)
            pm, phim = self.fk(q3 - dq, point)
            J[:2, i] = (pp - pm) / (2 * e)
            J[2, i] = (phip - phim) / (2 * e)
        return J

    def ik(self, target_rz, pitch, point, check=True):
        """Joint angles placing a supplied tool point at the target and pitch."""
        target = np.asarray(target_rz, float)
        pitch = float(pitch)
        if target.shape != (2,) or not np.all(np.isfinite(target)) or not np.isfinite(pitch):
            raise ValueError('Supply a finite planar target and pitch')
        point = self._point(point)
        c, s = np.cos(pitch), np.sin(pitch)
        pivot = target - np.array([c * point[0] - s * point[1], s * point[0] + c * point[1]])
        rel = pivot - self.pivot
        d2 = rel @ rel
        l0, l1 = self.L
        cos_joint = (d2 - l0 ** 2 - l1 ** 2) / (2 * l0 * l1)
        if cos_joint > 1. + 1e-12 or cos_joint < -1. - 1e-12:
            raise ValueError(f'Target out of reach: distance {np.sqrt(d2):.6g}')
        q1 = self.elbow_sign * np.arccos(np.clip(cos_joint, -1., 1.))
        k1 = l0 + l1 * np.cos(q1)
        k2 = l1 * np.sin(q1)
        q0 = np.arctan2(rel[1], rel[0]) - np.arctan2(k2, k1)
        q2 = pitch - q0 - q1
        q2 = (q2 + np.pi) % (2 * np.pi) - np.pi
        q = np.array([q0, q1, q2])
        if check:
            self.check_limits(q)
            p, phi = self.fk(q, point)
            if np.max(np.abs(p - target)) > 1e-9 or abs((phi - pitch + np.pi) % (2 * np.pi) - np.pi) > 1e-9:
                raise ValueError('IK verification failed')
        return q

    def check_limits(self, q3, margin=None):
        q3 = self._angles(q3)
        margin = self.limit_margin if margin is None else float(margin)
        if not np.isfinite(margin) or margin < 0:
            raise ValueError('The joint-limit margin must be finite and nonnegative')
        if np.any(q3 < self.lower + margin) or np.any(q3 > self.upper - margin):
            raise ValueError(f'Joint limits violated: {q3} outside {self.lower}..{self.upper}')
        return True


def smooth(t, start, end):
    """Quintic time scaling and its first two derivatives on [start, end]."""
    d = float(end) - float(start)
    if not np.isfinite(d) or d <= 0:
        raise ValueError('Time-scaling duration must be finite and positive')
    s = np.clip((t - start) / d, 0., 1.)
    f = s ** 3 * (10 - 15 * s + 6 * s * s)
    df = 30 * s * s * (1 - s) ** 2 / d
    ddf = 60 * s * (1 - 3 * s + 2 * s * s) / d ** 2
    return f, df, ddf
