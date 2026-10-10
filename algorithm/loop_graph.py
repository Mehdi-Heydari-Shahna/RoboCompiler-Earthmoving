"""Loop-closure evaluators and graph interfaces for PACDM assembly.

A cut revolute joint uses the relative-transform residual
``r = log((T_minus D)^-1 T_plus)`` and its SE(3) differential. Each cut is
computed in the lowest common ancestor frame of its two bodies. Only edges
and coordinate columns on the fundamental cycle enter that closure.
"""
from __future__ import annotations

import math

import numpy as np

from .pacdm_core import PACDM, inv, log, inv_left

_I3 = np.eye(3)
_I6 = np.eye(6)


def rot(K, K2, angle):
    return _I3 + math.sin(angle) * K + (1. - math.cos(angle)) * K2


def _cross(a, b):
    a0, a1, a2 = a.tolist()
    b0, b1, b2 = b.tolist()
    return np.array((a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0))


def _skew(v):
    x, y, z = v
    return np.array(((0., -z, y), (z, 0., -x), (-y, x, 0.)))


def se3_log(R, p):
    """Same formula and branch guard as the original ``pacdm.log`` (4x4 -> 6)."""
    (r00, r01, r02), (r10, r11, r12), (r20, r21, r22) = R.tolist()
    u0, u1, u2 = (r21 - r12) / 2, (r02 - r20) / 2, (r10 - r01) / 2
    s = math.sqrt(u0 * u0 + u1 * u1 + u2 * u2)
    theta = math.atan2(s, min(1., max(-1., (r00 + r11 + r22 - 1) / 2)))
    if math.pi - theta < 1e-6:
        raise ValueError('SE3 logarithm near pi branch')
    if s < 1e-14:
        w = (u0, u1, u2)
    else:
        f = theta / s
        w = (f * u0, f * u1, f * u2)
    K = _skew(w)
    if theta < 1e-4:
        beta = 1 / 12 + theta ** 2 / 720 + theta ** 4 / 30240
    else:
        beta = (1 - theta / 2 / math.tan(theta / 2)) / theta ** 2
    v = (_I3 - K / 2 + beta * (K @ K)) @ p
    out = np.empty(6)
    out[:3] = w
    out[3:] = v
    return out


def se3_inv_left(x):
    """Same as the original ``pacdm.inv_left``; its series branch is inlined."""
    W, V = _skew(x[:3].tolist()), _skew(x[3:].tolist())
    if math.sqrt(4. * float(x[:3] @ x[:3]) + 2. * float(x[3:] @ x[3:])) <= 1e-3:
        Z = np.zeros((6, 6))
        Z[:3, :3] = W
        Z[3:, :3] = V
        Z[3:, 3:] = W
        Z2 = Z @ Z
        return _I6 - Z / 2 + Z2 / 12 - (Z2 @ Z2) / 720
    return inv_left(x)


def adjoint(R, p):
    A = np.zeros((6, 6))
    A[:3, :3] = R
    A[3:, 3:] = R
    A[3:, :3] = _skew(p.tolist()) @ R
    return A


class LoopKinematics:
    """Precompiled LCA-relative traversal of every fundamental cycle.

    ``comp`` is a :class:`algorithm.compiler.Compilation`. Cuts sharing an LCA share
    one traversal; each group only visits the edges on its cycles' paths.
    """

    def __init__(self, comp, cut_indices=None):
        s = comp.structure
        self.n = len(s['coordinate_ids'])
        cuts = list(range(len(s['cuts']))) if cut_indices is None else list(cut_indices)
        self.cut_indices = cuts
        groups = {}
        for c in cuts:
            groups.setdefault(s['cuts'][c]['lca'], []).append(c)
        self.row_of = {c: 6 * k for k, c in enumerate(cuts)}
        self.position_of = {c: k for k, c in enumerate(cuts)}
        self.rows = 6 * len(cuts)
        self.groups = []
        for lca, members in groups.items():
            order = []
            for c in members:
                for e in s['cuts'][c]['minus_path'] + s['cuts'][c]['plus_path']:
                    if e not in order:
                        order.append(e)
            order.sort(key=lambda e: s['edges'][e]['depth'])
            steps = []
            for e in order:
                edge = s['edges'][e]
                steps.append((edge['parent'], edge['child'], edge['kind'], edge['enter_R'], edge['enter_p'],
                              edge['leave_R'], edge['leave_p'], edge['axis'], edge['K'], edge['K2'],
                              edge['coord']))
            plans = []
            for c in members:
                cut = s['cuts'][c]
                plus = [s['edges'][e]['coord'] for e in cut['plus_path'] if s['edges'][e]['coord'] is not None]
                minus = [s['edges'][e]['coord'] for e in cut['minus_path'] if s['edges'][e]['coord'] is not None]
                cols = np.array(plus + minus + [cut['coord']], int)
                signs = np.array([1.] * len(plus) + [-1.] * (len(minus) + 1))
                plans.append((c, cut['base'], cut['follower'], np.asarray(cut['T_A'], float),
                              np.asarray(cut['T_B'], float), cut['axis'], cut['K'], cut['K2'], cut['coord'],
                              plus + minus, cols, signs, self.row_of[c], self.position_of[c]))
            self.groups.append((lca, steps, plans))

    def evaluate(self, q, defects=None, columns=None, need_jac=True):
        """Residual (6 per cut), Jacobian (rows x n, or selected columns) and deltas.

        With ``need_jac=False`` no twist column, inverse left Jacobian or
        Jacobian entry is formed (used by the finite-difference baselines).
        """
        r = np.zeros(self.rows)
        J = np.zeros((self.rows, self.n)) if need_jac else None
        deltas = np.zeros((len(self.cut_indices), 4, 4))
        for lca, steps, plans in self.groups:
            R = {lca: _I3}
            P = {lca: np.zeros(3)}
            xi = {}
            for parent, child, kind, eR, ep, lR, lp, axis, K, K2, k in steps:
                Rp = R[parent]
                Rpre = Rp @ eR
                ppre = P[parent] + Rp @ ep
                if kind == 'revolute':
                    if need_jac:
                        w = Rpre @ axis
                        t = np.empty(6)
                        t[:3] = w
                        t[3:] = _cross(ppre, w)
                        xi[k] = t
                    Rj = Rpre @ rot(K, K2, q[k])
                    pj = ppre
                elif kind == 'prismatic':
                    w = Rpre @ axis
                    if need_jac:
                        t = np.zeros(6)
                        t[3:] = w
                        xi[k] = t
                    Rj = Rpre
                    pj = ppre + w * q[k]
                else:
                    Rj, pj = Rpre, ppre
                R[child] = Rj @ lR
                P[child] = pj + Rj @ lp
            for c, base, follower, TA, TB, axis, K, K2, kc, path_coords, cols, signs, row, position in plans:
                Rb, pb = R[base], P[base]
                Rpre = Rb @ TA[:3, :3]
                ppre = pb + Rb @ TA[:3, 3]
                Rm = Rpre @ rot(K, K2, q[kc])
                pm = ppre
                if defects is not None:
                    Dm = defects[position]
                    pm = pm + Rm @ Dm[:3, 3]
                    Rm = Rm @ Dm[:3, :3]
                Rf = R[follower]
                Rplus = Rf @ TB[:3, :3]
                pplus = P[follower] + Rf @ TB[:3, 3]
                Rrev = Rm.T
                prev = -(Rrev @ pm)
                Rd = Rrev @ Rplus
                pd = Rrev @ pplus + prev
                res = se3_log(Rd, pd)
                r[row:row + 6] = res
                delta = deltas[position]
                delta[:3, :3] = Rd
                delta[:3, 3] = pd
                delta[3, 3] = 1.
                if need_jac:
                    w = Rpre @ axis
                    xcut = np.empty(6)
                    xcut[:3] = w
                    xcut[3:] = _cross(ppre, w)
                    X = np.empty((6, len(cols)))
                    for j, k in enumerate(path_coords):
                        X[:, j] = xi[k]
                    X[:, -1] = xcut
                    J[row:row + 6, cols] = (se3_inv_left(res) @ adjoint(Rrev, prev)) @ (X * signs)
        if need_jac and columns is not None:
            return r, J[:, columns], deltas
        return r, J, deltas


class TreeWorld:
    """World poses of every body along the compiled tree (for lift and reports)."""

    def __init__(self, comp):
        s = comp.structure
        self.steps = [(e['parent'], e['child'], e['kind'], e['enter_R'], e['enter_p'], e['leave_R'],
                       e['leave_p'], e['axis'], e['K'], e['K2'], e['coord'])
                      for e in sorted(s['edges'], key=lambda e: e['depth'])]
        self.root = s['root']
        self.cuts = s['cuts']

    def poses(self, q):
        R = {self.root: _I3}
        P = {self.root: np.zeros(3)}
        for parent, child, kind, eR, ep, lR, lp, axis, K, K2, k in self.steps:
            Rp = R[parent]
            Rpre = Rp @ eR
            ppre = P[parent] + Rp @ ep
            if kind == 'revolute':
                Rj, pj = Rpre @ rot(K, K2, q[k]), ppre
            elif kind == 'prismatic':
                Rj, pj = Rpre, ppre + Rpre @ axis * q[k]
            else:
                Rj, pj = Rpre, ppre
            R[child] = Rj @ lR
            P[child] = pj + Rj @ lp
        out = {}
        for body in R:
            T = np.eye(4)
            T[:3, :3] = R[body]
            T[:3, 3] = P[body]
            out[body] = T
        return out

    def cut_angles(self, q):
        """Cut coordinates implied by the tree coordinates (angle about the cut axis)."""
        poses = self.poses(q)
        out = []
        for cut in self.cuts:
            A = poses[cut['base']] @ np.asarray(cut['T_A'])
            B = poses[cut['follower']] @ np.asarray(cut['T_B'])
            out.append(float(cut['axis'] @ log(inv(A) @ B)[:3]))
        return np.asarray(out)


class GlobalLoopGraph:
    """All cut residuals as one PACDM problem (the generated 'global' evaluator)."""

    def __init__(self, comp, independent=None):
        s = comp.structure
        self.comp = comp
        self.ids = list(s['coordinate_ids'])
        self.n = len(self.ids)
        self.nt = s['tree_coordinates']
        ind = s['independent'] if independent is None else independent
        self.active = np.array(ind, int)
        self.passive = np.setdiff1d(np.arange(self.n), self.active)
        self.lower = np.asarray(s['lower'], float)
        self.upper = np.asarray(s['upper'], float)
        self.kin = LoopKinematics(comp)
        self.world = TreeWorld(comp)
        self.calls = 0

    def residual(self, q, defects=None):
        self.calls += 1
        return self.kin.evaluate(q, defects)

    def residual_only(self, q):
        self.calls += 1
        return self.kin.evaluate(q, None, need_jac=False)[0]

    def lift(self, q):
        """Complete a configuration whose tree coordinates are closed: set cut angles."""
        out = np.asarray(q, float).copy()
        cut_coords = [c['coord'] for c in self.comp.structure['cuts']]
        angles = self.world.cut_angles(out)
        seed = np.asarray(self.comp.structure['seed'])
        for k, angle in zip(cut_coords, angles):
            out[k] = angle + 2 * np.pi * np.round((seed[k] - angle) / (2 * np.pi))
        return out


class ModuleLoopGraph:
    """One generated loop module as its own PACDM problem.

    Local coordinates: the module inputs (active, independent coordinates of its
    cycles) followed by its dependent coordinates (passive). Only the module's
    cuts are evaluated, each in its LCA frame.
    """

    def __init__(self, comp, module):
        s = comp.structure
        self.module = module
        self.inputs = list(module['input_indices'])
        self.dependents = list(module['dependent_indices'])
        self.global_index = np.array(self.inputs + self.dependents, int)
        self.n = len(self.global_index)
        self.active = np.arange(len(self.inputs))
        self.passive = np.arange(len(self.inputs), self.n)
        self.lower = np.asarray(s['lower'], float)[self.global_index]
        self.upper = np.asarray(s['upper'], float)[self.global_index]
        self.kin = LoopKinematics(comp, module['cut_indices'])
        self.nglobal = len(s['coordinate_ids'])
        self.calls = 0

    def residual(self, x, defects=None):
        self.calls += 1
        q = np.zeros(self.nglobal)
        q[self.global_index] = x
        return self.kin.evaluate(q, defects, columns=self.global_index)

    def residual_only(self, x):
        self.calls += 1
        q = np.zeros(self.nglobal)
        q[self.global_index] = x
        return self.kin.evaluate(q, None, need_jac=False)[0]


def pacdm_for(graph):
    return PACDM(graph)
