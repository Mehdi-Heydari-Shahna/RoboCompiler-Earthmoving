"""Compiled graph assembly, tangent lift, curvature and reduced dynamics.

For independent rates u, the augmented tangent lift is v = N u and
v_dot = N u_dot + c. Reduced rigid-body terms are M_r = N_tree.T M N_tree
and h_r = N_tree.T (h + M c_tree). Actuator ports and optional channel groups
are supplied by the caller's physical records.
"""
from __future__ import annotations

import time

import numpy as np
from .compiler import compile_graph
from .loop_graph import GlobalLoopGraph, TreeWorld
from .pacdm_core import PACDM
from .tree_dynamics import SourceBiasDynamics, _skew


def normmax(x):
    return float(np.max(np.abs(x))) if np.size(x) else 0.


class CompiledInterface:
    """One compiled partition of the physical graph with online PACDM assembly."""

    def __init__(self, physical, partition='joint_space', *, gravity=None, polish_tolerance=5e-13):
        self.physical = physical
        self.partition = partition
        self.comp = compile_graph(physical, partition)
        s = self.comp.structure
        self.coordinate_ids = list(s['coordinate_ids'])
        self.tree_ids = list(self.comp.cmg['tree_joint_ids'])
        self.cut_ids = list(self.comp.cmg['cut_joint_ids'])
        self.independent_ids = list(self.comp.cmg['independent_ids'])
        self.n, self.nt, self.nu = len(self.coordinate_ids), len(self.tree_ids), len(self.independent_ids)
        self.graph = GlobalLoopGraph(self.comp)
        self.world = TreeWorld(self.comp)
        self.pacdm = PACDM(self.graph)
        self.active = np.asarray(self.graph.active, int)
        self.passive = np.asarray(self.graph.passive, int)
        self.tree_rows = np.array([self.coordinate_ids.index(k) for k in self.tree_ids], int)
        self.gravity = np.asarray(gravity if gravity is not None else physical['gravity_m_s2'], float)
        self.dyn = SourceBiasDynamics(self.comp.cmg, self.tree_ids)
        self.polish_tolerance = polish_tolerance
        # actuator ports (moving joints with a declared actuator), in physical record order
        self.actuators = [a for a in physical.get('actuators', [])]
        self.port_ids = [a['joint_id'] for a in self.actuators]
        self.E = np.zeros((self.nt, len(self.port_ids)))
        for col, jid in enumerate(self.port_ids):
            self.E[self.tree_ids.index(jid), col] = 1.
        # Without actuator_groups, every actuator is its own control channel.
        groups = physical.get('actuator_groups') or {a['id']: [a['joint_id']] for a in self.actuators}
        self.group_ids = [g for g in groups if any(j in self.port_ids for j in groups[g])]
        self.G = np.zeros((len(self.port_ids), len(self.group_ids)))     # port effort = G @ channel effort
        for c, g in enumerate(self.group_ids):
            for jid in groups[g]:
                if jid in self.port_ids:
                    self.G[self.port_ids.index(jid), c] = 1.
        self.seed = np.asarray(s['seed_closed'], float).copy()
        self.N, self.mapping = self.pacdm.mapping(self.seed)
        if self.N is None:
            raise ValueError(f'Compiled seed has no admissible tangent map: {self.mapping}')
        self.seed = self.polish(self.seed, np.asarray(self.mapping['rows'], int))
        self.N, self.mapping = self.pacdm.mapping(self.seed)
        self.stats = dict(evaluations=0, continuation_steps=0, fallback_count=0, closure_max=0., tangent_max=0.,
                          assembly_wall_s=0., assembly_wall_max_s=0.)

    # ------------------------------------------------------------------ assembly (shadow configuration)
    def polish(self, q, rows):
        """Refine dependent coordinates while preserving independent coordinates."""
        q = q.copy()
        for _ in range(6):
            residual, jacobian, _ = self.graph.residual(q)
            if normmax(residual) < self.polish_tolerance:
                break
            q[self.graph.passive] -= np.linalg.solve(jacobian[np.ix_(rows, self.graph.passive)], residual[rows])
        if np.any(q < self.graph.lower) or np.any(q > self.graph.upper):
            raise ValueError('PACDM polish left the numerical branch guards')
        return q

    def assemble(self, u_measured, step=0.02):
        """Bounded prediction/correction continuation from the last accepted closed configuration."""
        started = time.perf_counter()
        target_all = np.asarray(u_measured, float)
        if target_all.shape != (self.nu,) or not np.all(np.isfinite(target_all)):
            raise ValueError('Independent coordinates must be a finite vector matching the compiled partition')
        if not np.isfinite(step) or step <= 0:
            raise ValueError('Continuation step must be finite and positive')
        start = self.seed[self.active].copy()
        count = max(1, int(np.ceil(normmax(target_all - start) / step)))
        for index in range(1, count + 1):
            target = target_all if index == count else start + (target_all - start) * (index / count)
            rows = np.asarray(self.mapping['rows'], int)
            predicted = self.seed[self.passive] + self.N[self.passive] @ (target - self.seed[self.active])
            candidate, info = self.pacdm.correct(target, predicted, None, rows, maxiter=12)
            if not info['success']:
                self.stats['fallback_count'] += 1
                candidate, info = self.pacdm.acquire(target, self.seed)
            if not info['success']:
                raise ValueError(f'Online PACDM closure failed: {info}')
            candidate = self.polish(candidate, rows)
            tangent, mapping = self.pacdm.mapping(candidate)
            if tangent is None:
                raise ValueError(f'Online PACDM rank/full-closure gate failed: {mapping}')
            self.seed, self.N, self.mapping = candidate, tangent, mapping
            self.stats['closure_max'] = max(self.stats['closure_max'], float(mapping['residual_inf']))
            self.stats['tangent_max'] = max(self.stats['tangent_max'], float(mapping['tangent_residual']))
            self.stats['continuation_steps'] += 1
        self.stats['evaluations'] += 1
        elapsed = time.perf_counter() - started
        self.stats['assembly_wall_s'] += elapsed
        self.stats['assembly_wall_max_s'] = max(self.stats['assembly_wall_max_s'], elapsed)
        return self.seed.copy()

    def reconstruct(self, u, seed=None):
        """Closed augmented configuration for prescribed independent coordinates (homotopy acquire)."""
        q, info = self.pacdm.acquire(np.asarray(u, float), self.seed if seed is None else np.asarray(seed, float))
        if not info['success']:
            raise ValueError(f'PACDM acquisition failed: {info}')
        q = self.polish(q, np.asarray(info['mapping']['rows'], int))
        return q

    # ------------------------------------------------------------------ tangent lift and curvature
    def lift(self, q, u_dot=None):
        """Augmented tangent map N and, with u_dot, the velocity-product term c."""
        N, info = self.pacdm.mapping(q)
        if N is None:
            raise ValueError(f'Tangent map rejected: {info}')
        out = dict(N=N, info=info, N_tree=N[self.tree_rows])
        if u_dot is not None:
            v = N @ np.asarray(u_dot, float)
            J = self.graph.residual(q)[1]
            eps = 1e-5 / max(1., normmax(v))
            Jp = self.graph.residual(q + eps * v)[1]
            Jm = self.graph.residual(q - eps * v)[1]
            gamma = ((Jp - Jm) / (2 * eps)) @ v
            c = np.zeros(self.n)
            rows = np.asarray(info['rows'], int)
            c[self.passive] = -np.linalg.solve(J[np.ix_(rows, self.passive)], gamma[rows])
            out.update(v=v, c=c, v_tree=v[self.tree_rows], c_tree=c[self.tree_rows], J=J, gamma=gamma)
        return out

    def two_point_closure(self, evaluation):
        """Independent analytic two-point closure (residual, J, Jdot) on the tree coordinates."""
        return self.dyn.closure(evaluation, self.cut_ids)

    def reference_reduction(self, q_tree, v_tree):
        """All-row least-squares reference of N_tree and c_tree from the two-point closure."""
        ev = self.dyn.evaluate(q_tree, v_tree, self.gravity)
        r, J, Jd = self.two_point_closure(ev)
        active = [self.tree_ids.index(k) for k in self.independent_ids]
        passive = [i for i in range(self.nt) if i not in active]
        n = np.zeros((self.nt, len(active)))
        n[active] = np.eye(len(active))
        n[passive] = np.linalg.lstsq(J[:, passive], -J[:, active], rcond=1e-12)[0]
        nd = np.zeros_like(n)
        nd[passive] = np.linalg.lstsq(J[:, passive], -Jd @ n, rcond=1e-12)[0]
        c = nd @ v_tree[active]
        s = np.linalg.svd(J, compute_uv=False)
        return dict(N_tree=n, c_tree=c, residual=r, J=J, Jdot=Jd, rank=int(np.sum(s > 1e-10 * s[0])) if s.size else 0,
                    tangent_residual=normmax(J @ n), acceleration_residual=normmax(J @ c + Jd @ v_tree))

    # ------------------------------------------------------------------ reduced dynamics
    def dynamics(self, q, u_dot, gravity=None):
        """Tree M, h and reduced M_r, h_r at a closed augmented configuration q with independent rate u_dot."""
        gravity = self.gravity if gravity is None else np.asarray(gravity, float)
        k = self.lift(q, u_dot)
        qt, vt = q[self.tree_rows], k['v_tree']
        ev = self.dyn.evaluate(qt, vt, gravity)
        M, h = ev['mass_matrix'], ev['bias_forces']
        Nt, ct = k['N_tree'], k['c_tree']
        Mr = Nt.T @ M @ Nt
        Mr = 0.5 * (Mr + Mr.T)
        hr = Nt.T @ (h + M @ ct)
        return dict(q=q, q_tree=qt, v_tree=vt, u_dot=np.asarray(u_dot, float), N=k['N'], N_tree=Nt, c=k['c'], c_tree=ct,
                    M=M, h=h, Mr=Mr, hr=hr, evaluation=ev, mapping=k['info'],
                    kinetic_energy=ev['kinetic_energy'], potential_energy=ev['potential_energy'],
                    gravity_r=Nt.T @ ev['gravity_compensation'])

    def port_maps(self, N_tree):
        """A = E^T N: independent velocity -> port velocity; A^T: port effort -> reduced effort."""
        A = self.E.T @ N_tree
        return dict(A=A, E=self.E, G=self.G, channel_map=A.T @ self.G)

    def forward(self, dynamics, port_efforts, extra_reduced=None):
        """Independent accelerations from port efforts (and optional extra reduced effort)."""
        tau_r = dynamics['N_tree'].T @ (self.E @ np.asarray(port_efforts, float))
        if extra_reduced is not None:
            tau_r = tau_r + np.asarray(extra_reduced, float)
        udd = np.linalg.solve(dynamics['Mr'], tau_r - dynamics['hr'])
        a_tree = dynamics['N_tree'] @ udd + dynamics['c_tree']
        return dict(udd=udd, a_tree=a_tree, tau_r=tau_r)


def floating_rows(dyn_eval, cmg, q_tree, base_position, base_rotation, v_full, gravity):
    """Floating-base rigid-body sums: M (6+nt), h (6+nt).  World-aligned base coupling.

    ``dyn_eval`` is a SourceBiasDynamics.evaluate result computed with zero gravity for the
    tree at q_tree with velocity v_full[6:]; base linear velocity is world aligned, angular
    velocity world aligned, both expressed at the virtual root frame origin.
    """
    v = np.asarray(v_full, float)
    g = np.asarray(gravity, float)
    R = np.asarray(base_rotation, float)
    p = np.asarray(base_position, float)
    omega = v[3:6]
    n = len(v)
    M = np.zeros((n, n))
    h = np.zeros(n)
    gg = np.zeros(n)
    poses = {}
    for b in cmg['bodies']:
        if b['kind'] != 'rigid_body':
            continue
        name = b['id']
        T0 = dyn_eval['poses'][name]
        offset = R @ T0[:3, 3]
        T = np.eye(4)
        T[:3, :3] = R @ T0[:3, :3]
        T[:3, 3] = p + offset
        poses[name] = T
        J0 = dyn_eval['jacobians'][name]
        D0 = dyn_eval['body_jacobian_dots'][name]
        J = np.zeros((6, n))
        J[:3, :3] = np.eye(3)
        J[:3, 3:6] = -_skew(offset)
        J[3:, 3:6] = np.eye(3)
        J[:3, 6:] = R @ J0[:3]
        J[3:, 6:] = R @ J0[3:]
        od = np.cross(omega, offset) + R @ (J0[:3] @ v[6:])
        D = np.zeros_like(J)
        D[:3, 3:6] = -_skew(od)
        D[:3, 6:] = _skew(omega) @ R @ J0[:3] + R @ D0[:3]
        D[3:, 6:] = _skew(omega) @ R @ J0[3:] + R @ D0[3:]
        vel = J @ v
        r = T[:3, :3] @ np.asarray(b['com_m'])
        rd = np.cross(vel[3:], r)
        Jc = J[:3] - _skew(r) @ J[3:]
        Dc = D[:3] - _skew(rd) @ J[3:] - _skew(r) @ D[3:]
        mb = float(b['mass_kg'])
        I = T[:3, :3] @ np.asarray(b['inertia_com_kg_m2']) @ T[:3, :3].T
        w = vel[3:]
        M += mb * Jc.T @ Jc + J[3:].T @ I @ J[3:]
        grav = -mb * Jc.T @ g
        gg += grav
        h += mb * Jc.T @ (Dc @ v) + J[3:].T @ (I @ (D[3:] @ v) + np.cross(w, I @ w)) + grav
    return dict(mass_matrix=M, bias_forces=h, gravity_compensation=gg, poses=poses)
