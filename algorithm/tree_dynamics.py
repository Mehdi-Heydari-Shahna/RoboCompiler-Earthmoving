"""NumPy rigid-body dynamics from caller-supplied CMG records.

Revolute, prismatic and fixed tree joints are supported. Evaluation provides
mass, energy, gravity, analytic velocity bias and two-point loop kinematics.
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy

import numpy as np


def _array(value, shape, name):
    try:
        if np.iscomplexobj(value):
            raise ValueError(f"{name} must be real-valued")
        array = np.asarray(value, dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} must be a finite numeric array") from exc
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must have shape {shape} with finite entries")
    return array.copy()


def _skew(vector):
    x, y, z = vector
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def _transform(value, name):
    matrix = _array(value, (4, 4), name)
    rotation = matrix[:3, :3]
    if (not np.allclose(matrix[3], [0., 0., 0., 1.], rtol=0., atol=1e-10)
            or np.linalg.norm(rotation.T @ rotation - np.eye(3)) > 1e-8
            or abs(np.linalg.det(rotation) - 1.) > 1e-8):
        raise ValueError(f"{name} must be a proper rigid transform")
    return matrix


def _inverse(matrix):
    inverse = np.eye(4)
    inverse[:3, :3] = matrix[:3, :3].T
    inverse[:3, 3] = -inverse[:3, :3] @ matrix[:3, 3]
    return inverse


def _motion(kind, axis, value):
    matrix = np.eye(4)
    if kind == 'revolute':
        cross = _skew(axis)
        matrix[:3, :3] += np.sin(value) * cross + (1. - np.cos(value)) * cross @ cross
    elif kind == 'prismatic':
        matrix[:3, 3] = axis * value
    return matrix


class SourceDynamics:
    """Compile a physical-record tree and evaluate standard rigid-body sums.

    ``evaluate(q, velocity, gravity)`` accepts exact-shape one-dimensional
    arrays in SI units.  Jacobians are linear-first, world-aligned, at physical
    body origins.  ``gravity_compensation`` is the generalized force needed
    to balance gravity: it equals the gradient of ``potential_energy``.
    ``gravity`` is a world acceleration vector, for example [0, 0, -9.81].
    No hydraulic effort or unknown source effort is assigned by this class.
    """

    def __init__(self, cmg, tree_ids):
        self.cmg = deepcopy(cmg)
        self.tree_ids = list(tree_ids)
        self.nv = len(self.tree_ids)
        if (any(not isinstance(jid, str) for jid in self.tree_ids)
                or len(set(self.tree_ids)) != self.nv):
            raise ValueError("Tree IDs must be unique strings")
        body_records = self.cmg['bodies']
        joint_records = self.cmg['joints']
        self._bodies = {body['id']: body for body in body_records}
        if len(self._bodies) != len(body_records):
            raise ValueError("Duplicate physical body IDs")
        self.root_body = self.cmg['root_body']
        if self.root_body not in self._bodies:
            raise ValueError("Root body is absent")
        self.body_ids = list(self._bodies)
        self.physical_body_ids = []
        self._inertias = {}
        for body_id, body in self._bodies.items():
            if body['kind'] == 'reference_frame':
                if body_id != self.root_body:
                    raise ValueError("Only the root may be a reference frame")
                continue
            if body['kind'] != 'rigid_body':
                raise ValueError(f"Unsupported physical body kind at {body_id}")
            mass_array = _array(body['mass_kg'], (), f"{body_id} mass")
            mass = float(mass_array)
            if mass <= 0.:
                raise ValueError(f"{body_id} mass must be positive")
            com = _array(body['com_m'], (3,), f"{body_id} COM")
            inertia = _array(body['inertia_com_kg_m2'], (3, 3), f"{body_id} inertia")
            if not np.allclose(inertia, inertia.T, rtol=0., atol=1e-10):
                raise ValueError(f"{body_id} inertia must be symmetric")
            if np.min(np.linalg.eigvalsh(inertia)) < -1e-10:
                raise ValueError(f"{body_id} inertia must be positive semidefinite")
            self._inertias[body_id] = (mass, com, inertia)
            self.physical_body_ids.append(body_id)

        joints = {joint['id']: joint for joint in joint_records}
        if len(joints) != len(joint_records):
            raise ValueError("Duplicate physical joint IDs")
        if any(jid not in joints or joints[jid]['type'] == 'fixed'
               for jid in self.tree_ids):
            raise ValueError("Every tree ID must identify a moving source joint")
        index = {jid: column for column, jid in enumerate(self.tree_ids)}
        selected = set(self.tree_ids)
        selected.update(jid for jid, record in joints.items() if record['type'] == 'fixed')
        if len(selected) != len(self._bodies) - 1:
            raise ValueError("Selected joints and fixed connections must form a spanning tree")
        adjacency = {body_id: [] for body_id in self._bodies}
        for jid in sorted(selected):
            record = joints[jid]
            kind = record['type']
            if kind not in ('revolute', 'prismatic', 'fixed'):
                raise ValueError(f"Unsupported source joint type at {jid}")
            base, follower = record['base_body'], record['follower_body']
            if base not in adjacency or follower not in adjacency or base == follower:
                raise ValueError(f"Invalid source joint endpoints at {jid}")
            base_attach = _transform(record['T_BJ'], f"{jid} T_BJ")
            follower_attach = _transform(record['T_FJ'], f"{jid} T_FJ")
            axis = _array(record['axis'], (3,), f"{jid} axis")
            if kind != 'fixed' and abs(np.linalg.norm(axis) - 1.) > 1e-10:
                raise ValueError(f"{jid} moving axis must be a unit vector")
            adjacency[base].append((jid, follower, kind, base_attach,
                                    _inverse(follower_attach), axis, 1.))
            adjacency[follower].append((jid, base, kind, follower_attach,
                                        _inverse(base_attach), axis, -1.))
        self._traversal = []
        visited = {self.root_body}
        queue = deque([self.root_body])
        while queue:
            parent = queue.popleft()
            for jid, child, kind, attach, child_inverse, axis, direction in adjacency[parent]:
                if child in visited:
                    continue
                visited.add(child)
                queue.append(child)
                self._traversal.append((parent, child, kind, attach, child_inverse,
                                        direction * axis, index.get(jid)))
        if len(visited) != len(self._bodies):
            raise ValueError("Selected source tree is disconnected or cyclic")

    def evaluate(self, q, velocity, gravity):
        q = _array(q, (self.nv,), 'q')
        velocity = _array(velocity, (self.nv,), 'velocity')
        gravity = _array(gravity, (3,), 'gravity')
        poses = {self.root_body: np.eye(4)}
        jacobians = {self.root_body: np.zeros((6, self.nv))}
        for parent, child, kind, attach, child_inverse, axis, column in self._traversal:
            parent_pose = poses[parent]
            joint_pose = parent_pose @ attach
            value = q[column] if column is not None else 0.
            child_pose = joint_pose @ _motion(kind, axis, value) @ child_inverse
            child_position = child_pose[:3, 3]
            shift = child_position - parent_pose[:3, 3]
            parent_jacobian = jacobians[parent]
            child_jacobian = parent_jacobian.copy()
            child_jacobian[:3] -= _skew(shift) @ parent_jacobian[3:]
            if column is not None:
                world_axis = joint_pose[:3, :3] @ axis
                if kind == 'revolute':
                    child_jacobian[:3, column] += np.cross(
                        world_axis, child_position - joint_pose[:3, 3])
                    child_jacobian[3:, column] += world_axis
                elif kind == 'prismatic':
                    child_jacobian[:3, column] += world_axis
            poses[child] = child_pose
            jacobians[child] = child_jacobian

        mass_matrix = np.zeros((self.nv, self.nv))
        compensation = np.zeros(self.nv)
        potential = 0.
        kinetic = 0.
        com_positions = {}
        com_jacobians = {}
        contributions = {}
        for body_id, (mass, local_com, local_inertia) in self._inertias.items():
            pose = poses[body_id]
            rotation = pose[:3, :3]
            offset = rotation @ local_com
            com_position = pose[:3, 3] + offset
            body_jacobian = jacobians[body_id]
            linear_com = body_jacobian[:3] - _skew(offset) @ body_jacobian[3:]
            angular = body_jacobian[3:]
            inertia_world = rotation @ local_inertia @ rotation.T
            body_mass = mass * linear_com.T @ linear_com + angular.T @ inertia_world @ angular
            body_compensation = -mass * linear_com.T @ gravity
            body_potential = -mass * float(gravity @ com_position)
            linear_velocity = linear_com @ velocity
            angular_velocity = angular @ velocity
            body_kinetic = (0.5 * mass * float(linear_velocity @ linear_velocity)
                            + 0.5 * float(angular_velocity @ inertia_world @ angular_velocity))
            mass_matrix += body_mass
            compensation += body_compensation
            potential += body_potential
            kinetic += body_kinetic
            com_positions[body_id] = com_position
            com_jacobians[body_id] = linear_com
            contributions[body_id] = {
                'mass_matrix': body_mass,
                'gravity_compensation': body_compensation,
                'potential_energy': body_potential,
                'kinetic_energy': body_kinetic,
            }
        if not (np.all(np.isfinite(mass_matrix)) and np.all(np.isfinite(compensation))
                and np.isfinite(potential) and np.isfinite(kinetic)):
            raise ValueError("Source rigid-body evaluation produced nonfinite values")
        return {
            'poses': poses,
            'jacobians': jacobians,
            'mass_matrix': mass_matrix,
            'gravity_compensation': compensation,
            'potential_energy': float(potential),
            'kinetic_energy': float(kinetic),
            'body_com_positions': com_positions,
            'body_com_jacobians': com_jacobians,
            'body_contributions': contributions,
        }


class SourceBiasDynamics(SourceDynamics):
    """Independent rigid-body M, energy, gravity and analytic velocity bias.

    ``evaluate(q, velocity, gravity)`` extends ``SourceDynamics.evaluate`` with
    ``body_jacobian_dots``, ``body_com_jacobian_dots``, ``coriolis_forces`` and
    ``bias_forces``.  The vector called coriolis_forces includes centrifugal
    terms; no particular Coriolis matrix factorization is implied.

    ``closure(evaluation, cut_ids, axis_lever_m=0.1)`` returns residual, J and
    Jdot for two-point revolute closures in the supplied cut-ID order.
    """

    def evaluate(self, q, velocity, gravity):
        # The inherited implementation independently validates the source
        # records and input shapes and computes all first-order kinematics.
        q = _array(q, (self.nv,), 'q')
        velocity = _array(velocity, (self.nv,), 'velocity')
        gravity = _array(gravity, (3,), 'gravity')
        result = super().evaluate(q, velocity, gravity)
        poses, jacobians = result['poses'], result['jacobians']
        body_velocities = {body: jacobian @ velocity
                           for body, jacobian in jacobians.items()}
        jacobian_dots = {self.root_body: np.zeros((6, self.nv))}

        for parent, child, kind, attach, _, axis, column in self._traversal:
            parent_pose, child_pose = poses[parent], poses[child]
            parent_jacobian = jacobians[parent]
            parent_dot = jacobian_dots[parent]
            parent_velocity, child_velocity = body_velocities[parent], body_velocities[child]
            shift = child_pose[:3, 3] - parent_pose[:3, 3]
            shift_dot = child_velocity[:3] - parent_velocity[:3]

            child_dot = parent_dot.copy()
            child_dot[:3] -= (_skew(shift_dot) @ parent_jacobian[3:]
                             + _skew(shift) @ parent_dot[3:])
            if column is not None:
                joint_pose = parent_pose @ attach
                world_axis = joint_pose[:3, :3] @ axis
                axis_dot = np.cross(parent_velocity[3:], world_axis)
                if kind == 'revolute':
                    joint_offset = joint_pose[:3, 3] - parent_pose[:3, 3]
                    joint_velocity = (parent_velocity[:3]
                                      + np.cross(parent_velocity[3:], joint_offset))
                    lever = child_pose[:3, 3] - joint_pose[:3, 3]
                    lever_dot = child_velocity[:3] - joint_velocity
                    child_dot[:3, column] += (np.cross(axis_dot, lever)
                                             + np.cross(world_axis, lever_dot))
                    child_dot[3:, column] += axis_dot
                elif kind == 'prismatic':
                    child_dot[:3, column] += axis_dot
            jacobian_dots[child] = child_dot

        com_jacobian_dots = {}
        body_bias_accelerations = {}
        body_com_bias_accelerations = {}
        body_coriolis_contributions = {}
        coriolis = np.zeros(self.nv)
        for body_id, (mass, local_com, local_inertia) in self._inertias.items():
            rotation = poses[body_id][:3, :3]
            offset = rotation @ local_com
            omega = body_velocities[body_id][3:]
            offset_dot = np.cross(omega, offset)
            body_jacobian, body_dot = jacobians[body_id], jacobian_dots[body_id]
            com_dot = (body_dot[:3] - _skew(offset_dot) @ body_jacobian[3:]
                       - _skew(offset) @ body_dot[3:])
            com_bias = com_dot @ velocity
            angular_bias = body_dot[3:] @ velocity
            inertia_world = rotation @ local_inertia @ rotation.T
            force = mass * com_bias
            torque = inertia_world @ angular_bias + np.cross(omega, inertia_world @ omega)
            body_term = (result['body_com_jacobians'][body_id].T @ force
                         + body_jacobian[3:].T @ torque)
            coriolis += body_term
            com_jacobian_dots[body_id] = com_dot
            body_bias_accelerations[body_id] = body_dot @ velocity
            body_com_bias_accelerations[body_id] = com_bias
            body_coriolis_contributions[body_id] = body_term

        bias = coriolis + result['gravity_compensation']
        arrays = [coriolis, bias, *jacobian_dots.values(), *com_jacobian_dots.values(),
                  *body_velocities.values(), *body_bias_accelerations.values(),
                  *body_com_bias_accelerations.values(), *body_coriolis_contributions.values()]
        if any(not np.all(np.isfinite(value)) for value in arrays):
            raise ValueError('Source bias evaluation produced nonfinite values')
        result.update({
            'configuration': q,
            'velocity': velocity,
            'gravity': gravity,
            'body_velocities': body_velocities,
            'body_jacobian_dots': jacobian_dots,
            'body_com_jacobian_dots': com_jacobian_dots,
            'body_bias_accelerations': body_bias_accelerations,
            'body_com_bias_accelerations': body_com_bias_accelerations,
            'body_coriolis_contributions': body_coriolis_contributions,
            'coriolis_forces': coriolis,
            'bias_forces': bias,
        })
        return result

    @staticmethod
    def _point(evaluation, body_id, local_point):
        pose = evaluation['poses'][body_id]
        body_jacobian = evaluation['jacobians'][body_id]
        body_dot = evaluation['body_jacobian_dots'][body_id]
        omega = evaluation['body_velocities'][body_id][3:]
        offset = pose[:3, :3] @ local_point
        offset_dot = np.cross(omega, offset)
        point = pose[:3, 3] + offset
        jacobian = body_jacobian[:3] - _skew(offset) @ body_jacobian[3:]
        derivative = (body_dot[:3] - _skew(offset_dot) @ body_jacobian[3:]
                      - _skew(offset) @ body_dot[3:])
        return point, jacobian, derivative

    def closure(self, evaluation, cut_ids, axis_lever_m=0.1):
        """Evaluate analytic two-point closure kinematics from raw attachments.

        For each revolute joint: three base-minus-follower origin rows followed
        by three base-minus-follower axis-point rows.  The lever is a numerical
        length in metres, not measured component geometry.  Redundant rows are
        retained.  Cut joints must be omitted from this instance's source tree.
        """
        lever = _array(axis_lever_m, (), 'axis_lever_m')
        if float(lever) <= 0.:
            raise ValueError('axis_lever_m must be positive')
        cut_ids = list(cut_ids)
        if any(not isinstance(jid, str) for jid in cut_ids) or len(set(cut_ids)) != len(cut_ids):
            raise ValueError('Cut IDs must be unique strings')
        joints = {joint['id']: joint for joint in self.cmg['joints']}
        residuals, jacobians, derivatives = [], [], []
        for jid in cut_ids:
            if jid not in joints or joints[jid]['type'] != 'revolute' or jid in self.tree_ids:
                raise ValueError('Each cut ID must identify a source revolute joint outside the tree')
            joint = joints[jid]
            axis = _array(joint['axis'], (3,), f'{jid} axis')
            if abs(np.linalg.norm(axis) - 1.) > 1e-10:
                raise ValueError(f'{jid} moving axis must be a unit vector')
            base_transform = _transform(joint['T_BJ'], f'{jid} T_BJ')
            follower_transform = _transform(joint['T_FJ'], f'{jid} T_FJ')
            for distance in (0., float(lever)):
                base_local = base_transform[:3, 3] + base_transform[:3, :3] @ (distance * axis)
                follower_local = (follower_transform[:3, 3]
                                  + follower_transform[:3, :3] @ (distance * axis))
                base, base_jacobian, base_dot = self._point(evaluation, joint['base_body'], base_local)
                follower, follower_jacobian, follower_dot = self._point(
                    evaluation, joint['follower_body'], follower_local)
                residuals.append(base - follower)
                jacobians.append(base_jacobian - follower_jacobian)
                derivatives.append(base_dot - follower_dot)
        if not cut_ids:
            return np.zeros(0), np.zeros((0, self.nv)), np.zeros((0, self.nv))
        result = np.concatenate(residuals), np.vstack(jacobians), np.vstack(derivatives)
        if any(not np.all(np.isfinite(value)) for value in result):
            raise ValueError('Source closure evaluation produced nonfinite values')
        return result
