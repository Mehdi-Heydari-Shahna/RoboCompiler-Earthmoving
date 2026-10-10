"""Build a Pinocchio backend from caller-supplied physical graph records.

COM inertias are attached through fixed edges. Each cut revolute joint uses
two point-pair constraints, at its origin and at a numerical axis lever.
Constrained acceleration is evaluated with Pinocchio's proximal solver.
"""
from __future__ import annotations

import numpy as np
import pinocchio as pin

from .tree_dynamics import SourceDynamics


def _se3(matrix):
    matrix = np.asarray(matrix, float)
    return pin.SE3(matrix[:3, :3].copy(), matrix[:3, 3].copy())


def _inverse(matrix):
    out = np.eye(4)
    out[:3, :3] = matrix[:3, :3].T
    out[:3, 3] = -out[:3, :3] @ matrix[:3, 3]
    return out


class PinModel:
    def __init__(self, cmg, cut_ids, lever=0.1, gravity=None, point_frames=None):
        self.cmg = cmg
        self.bodies = {b['id']: b for b in cmg['bodies']}
        self.joints = {j['id']: j for j in cmg['joints']}
        self.cut_ids = list(cut_ids)
        if len(set(self.cut_ids)) != len(self.cut_ids) or any(
                jid not in self.joints or self.joints[jid]['type'] != 'revolute' for jid in self.cut_ids):
            raise ValueError('Cut IDs must identify distinct revolute joints')
        tree = [jid for jid in self.joints if jid not in self.cut_ids]
        SourceDynamics(cmg, [jid for jid in tree if self.joints[jid]['type'] != 'fixed'])
        gravity = np.asarray(cmg['gravity_m_s2'] if gravity is None else gravity, float)
        if gravity.shape != (3,) or not np.all(np.isfinite(gravity)):
            raise ValueError('Gravity must be a finite three-component acceleration vector')
        if not np.isfinite(lever) or lever <= 0:
            raise ValueError('The numerical axis lever must be finite and positive')
        adjacency = {b: [] for b in self.bodies}
        for jid in sorted(tree):
            j = self.joints[jid]
            adjacency[j['base_body']].append((jid, j['follower_body'], 1))
            adjacency[j['follower_body']].append((jid, j['base_body'], -1))
        self.model = pin.Model()
        self.model.name = 'physical_graph_pinocchio'
        self.model.gravity = pin.Motion.Zero()
        self.model.gravity.linear = np.asarray(gravity, float)
        root = cmg['root_body']
        self.body_frame, self.support, self.placement, self.joint_index, self.direction = {}, {}, {}, {}, {}
        self.support[root] = 0
        self.placement[root] = np.eye(4)
        self.body_frame[root] = int(self.model.addFrame(pin.Frame(root, 0, 0, pin.SE3.Identity(), pin.FrameType.BODY), False))
        seen = {root}

        def visit(parent):
            for jid, child, direction in adjacency[parent]:
                if child in seen:
                    continue
                seen.add(child)
                joint = self.joints[jid]
                base = np.asarray(joint['T_BJ'], float)
                follower = np.asarray(joint['T_FJ'], float)
                parent_attach, child_attach = (base, follower) if direction == 1 else (follower, base)
                joint_placement = self.placement[parent] @ parent_attach
                body_placement = _inverse(child_attach)
                if joint['type'] == 'fixed':
                    support = self.support[parent]
                    body_placement = joint_placement @ body_placement
                else:
                    axis = direction * np.asarray(joint['axis'], float)
                    kind = (pin.JointModelRevoluteUnaligned(axis) if joint['type'] == 'revolute'
                            else pin.JointModelPrismaticUnaligned(axis))
                    support = int(self.model.addJoint(self.support[parent], kind, _se3(joint_placement), jid))
                    self.joint_index[jid] = support
                    self.direction[jid] = direction
                record = self.bodies[child]
                self.body_frame[child] = int(self.model.addFrame(pin.Frame(child, support, self.body_frame[parent],
                                                                           _se3(body_placement), pin.FrameType.BODY), False))
                if record['kind'] == 'rigid_body':
                    self.model.appendBodyToJoint(support, pin.Inertia(float(record['mass_kg']), np.asarray(record['com_m'], float),
                                                                     np.asarray(record['inertia_com_kg_m2'], float)), _se3(body_placement))
                self.support[child] = support
                self.placement[child] = body_placement
                visit(child)

        visit(root)
        if seen != set(self.bodies):
            raise ValueError('CMG graph is disconnected')
        self.tree_ids = [self.model.names[i] for i in range(1, self.model.njoints)]
        self.lever = float(lever)
        self.constraints = []
        for jid in self.cut_ids:
            joint = self.joints[jid]
            axis = np.asarray(joint['axis'], float)
            for k, distance in enumerate((0., self.lever)):
                ends = []
                for body_key, transform_key in (('base_body', 'T_BJ'), ('follower_body', 'T_FJ')):
                    transform = np.asarray(joint[transform_key], float)
                    local = transform[:3, 3] + transform[:3, :3] @ (distance * axis)
                    body = joint[body_key]
                    placement = self.placement[body]
                    at_joint = placement[:3, 3] + placement[:3, :3] @ local
                    ends.append((self.support[body], pin.SE3(np.eye(3), at_joint)))
                cm = pin.RigidConstraintModel(pin.ContactType.CONTACT_3D, self.model, ends[0][0], ends[0][1], ends[1][0], ends[1][1],
                                              pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
                cm.name = f'closure_{jid}_{k}'
                self.constraints.append(cm)
        self.point_frames = {}
        for name, (body, local) in (point_frames or {}).items():
            placement = self.placement[body]
            at_joint = placement[:3, 3] + placement[:3, :3] @ np.asarray(local, float)
            self.point_frames[name] = int(self.model.addFrame(pin.Frame('point_' + name, self.support[body], self.body_frame[body],
                                                                        pin.SE3(np.eye(3), at_joint), pin.FrameType.OP_FRAME), False))
        self.index = {jid: i for i, jid in enumerate(self.tree_ids)}
        self.data = self.model.createData()
        self.constraint_datas = [c.createData() for c in self.constraints]
        pin.initConstraintDynamics(self.model, self.data, self.constraints)
        self.prox = pin.ProximalSettings(1e-12, 1e-10, 10)

    # ------------------------------------------------------------------ closure
    def _endpoints(self, q):
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        out = []
        for cm in self.constraints:
            a = self.data.oMi[cm.joint1_id].act(cm.joint1_placement).translation
            b = self.data.oMi[cm.joint2_id].act(cm.joint2_placement).translation
            out.append((a, b))
        return out

    def closure_residual(self, q):
        endpoints = self._endpoints(q)
        return np.concatenate([a - b for a, b in endpoints]) if endpoints else np.zeros(0)

    def closure_jacobian(self, q):
        pin.computeJointJacobians(self.model, self.data, q)
        rows = []
        for cm in self.constraints:
            rows.append(self._point_jacobian(q, cm.joint1_id, cm.joint1_placement) - self._point_jacobian(q, cm.joint2_id, cm.joint2_placement))
        return np.vstack(rows) if rows else np.zeros((0, self.model.nv))

    def _point_jacobian(self, q, joint_id, placement):
        J = pin.getJointJacobian(self.model, self.data, joint_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        offset = self.data.oMi[joint_id].rotation @ placement.translation
        return J[:3] - pin.skew(offset) @ J[3:]

    def gamma(self, q, v):
        """Constraint acceleration drift Jdot v (world-aligned point accelerations at zero joint acceleration)."""
        pin.forwardKinematics(self.model, self.data, q, v, np.zeros(self.model.nv))
        out = []
        for cm in self.constraints:
            accs = []
            for jid, pl in ((cm.joint1_id, cm.joint1_placement), (cm.joint2_id, cm.joint2_placement)):
                a = pin.getClassicalAcceleration(self.model, self.data, jid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
                vel = pin.getVelocity(self.model, self.data, jid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
                r = self.data.oMi[jid].rotation @ pl.translation
                accs.append(a.linear + np.cross(a.angular, r) + np.cross(vel.angular, np.cross(vel.angular, r)))
            out.append(accs[0] - accs[1])
        return np.concatenate(out) if out else np.zeros(0)

    def mass_bias(self, q, v):
        M = pin.crba(self.model, self.data, q)
        M = np.triu(M) + np.triu(M, 1).T
        h = pin.nonLinearEffects(self.model, self.data, q, v)
        return M, h

    def constrained_acceleration(self, q, v, tau):
        return np.array(pin.constraintDynamics(self.model, self.data, q, v, tau, self.constraints, self.constraint_datas, self.prox))

    def point(self, q, name):
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        return self.data.oMf[self.point_frames[name]].translation.copy()
