"""Compiled-interface motion control and differential-drive travel control.

The motion controller assembles measured independent coordinates in a shadow
configuration, evaluates the tangent lift, curvature, reduced inertia and
floating-base coupling, and maps requested efforts to actuator channels by
virtual work. Plant state is never overwritten. All mechanism data and native
joint names come from the caller's compiled interface and MuJoCo model.
"""
from __future__ import annotations

import time

import mujoco
import numpy as np

from .reduced import CompiledInterface, floating_rows, normmax


def _body_id(model, body):
    index = int(model.body(body).id) if isinstance(body, str) else int(body)
    if not 0 <= index < model.nbody:
        raise ValueError('Base body ID is outside the supplied model')
    return index


def _scalar_joint_ids(model, names):
    ids = np.asarray([model.joint(name).id for name in names], dtype=int)
    scalar = (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))
    if any(int(model.jnt_type[index]) not in scalar for index in ids):
        raise ValueError('The compiled graph requires scalar hinge or slide joints')
    return ids


def _vector(value, length, label):
    vector = np.asarray(value, float)
    if vector.shape != (length,) or not np.all(np.isfinite(vector)):
        raise ValueError(f'{label} must be a finite vector of length {length}')
    return vector


def _gain(value, length, label):
    try:
        vector = np.broadcast_to(np.asarray(value, float), (length,)).copy()
    except ValueError as exc:
        raise ValueError(f'{label} must be scalar or have length {length}') from exc
    if not np.all(np.isfinite(vector)) or np.any(vector < 0):
        raise ValueError(f'{label} must be finite and nonnegative')
    return vector


class ArmController:
    """Control independent scalar coordinates through compiled actuator groups.

    ``interface`` is a caller-created :class:`CompiledInterface`. Its tree and
    independent IDs must name scalar joints in ``model``. ``base_body`` names
    the body carrying the floating free joint, at the compiled root-frame
    origin. The compiled channel map must be square and nonsingular. Gains are
    supplied as scalars or arrays in independent-coordinate order.
    """

    def __init__(self, interface, model, *, base_body, kp, kd, feedforward=True,
                 observer_gain=0., observer_limit=None, base_coupling=True,
                 base_acceleration_filter_s=0.):
        if not isinstance(interface, CompiledInterface):
            raise TypeError('interface must be a caller-created CompiledInterface')
        self.ci = interface
        self.cmg = self.ci.comp.cmg
        self.m = model
        self.nu = len(self.ci.independent_ids)
        self.nchannels = len(self.ci.group_ids)
        if self.nu == 0 or self.nchannels != self.nu:
            raise ValueError('A nonempty square independent-coordinate/channel partition is required')
        joint_ids = _scalar_joint_ids(model, self.ci.independent_ids)
        self.qi, self.vi = model.jnt_qposadr[joint_ids], model.jnt_dofadr[joint_ids]
        self.base_id = _body_id(model, base_body)
        if int(model.body_jntnum[self.base_id]) != 1:
            raise ValueError('The supplied base body must carry one free joint')
        base_joint = int(model.body_jntadr[self.base_id])
        if int(model.jnt_type[base_joint]) != int(mujoco.mjtJoint.mjJNT_FREE):
            raise ValueError('The supplied base body must carry a floating free joint')
        self.base_qi = int(model.jnt_qposadr[base_joint])
        self.base_vi = int(model.jnt_dofadr[base_joint])
        self.kp = _gain(kp, self.nu, 'kp')
        self.kd = _gain(kd, self.nu, 'kd')
        self.feedforward = bool(feedforward)
        self.base_coupling = bool(base_coupling)
        self.observer_gain = float(observer_gain)
        if not np.isfinite(self.observer_gain) or not 0 <= self.observer_gain <= 1:
            raise ValueError('observer_gain must lie between zero and one')
        self.observer_limit = None if observer_limit is None else _gain(observer_limit, self.nu, 'observer_limit')
        self.disturbance = np.zeros(self.nu)
        self.acceleration_filter_s = float(base_acceleration_filter_s)
        if not np.isfinite(self.acceleration_filter_s) or self.acceleration_filter_s < 0:
            raise ValueError('base_acceleration_filter_s must be finite and nonnegative')
        self._filtered_acceleration = None
        self._last_sample_time = None
        self.tree_index = {key: index for index, key in enumerate(self.ci.tree_ids)}
        self.port_index = [self.ci.tree_ids.index(key) for key in self.ci.port_ids]
        self.stats = dict(evaluations=0, shadow_max=0., shadow_velocity_max=0., channel_condition_max=0.,
                          command_peak=np.zeros(self.nchannels), coupling_max=0., base_acceleration_peak=np.zeros(6),
                          wall_s=0., wall_max_s=0., virtual_power_defect_max=0.)
        self.last = None
        # Native addresses of the compiled tree joints for shadow-state comparison.
        tree_joint_ids = _scalar_joint_ids(model, self.ci.tree_ids)
        self.tree_qi = model.jnt_qposadr[tree_joint_ids]
        self.tree_vi = model.jnt_dofadr[tree_joint_ids]

    def base_state(self, data):
        qi, vi = self.base_qi, self.base_vi
        quat = data.qpos[qi + 3:qi + 7]
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, quat / np.linalg.norm(quat))
        rotation = rotation.reshape(3, 3)
        velocity = data.qvel[vi:vi + 6].copy()
        acceleration = data.qacc[vi:vi + 6].copy()
        omega, alpha = rotation @ velocity[3:], rotation @ acceleration[3:]
        return dict(position=data.qpos[qi:qi + 3].copy(), rotation=rotation,
                    velocity=np.r_[velocity[:3], omega], acceleration=np.r_[acceleration[:3], alpha])

    def evaluate(self, desired, velocity, acceleration, data, applied_ports=None):
        desired = _vector(desired, self.nu, 'desired')
        velocity = _vector(velocity, self.nu, 'velocity')
        acceleration = _vector(acceleration, self.nu, 'acceleration')
        if applied_ports is not None:
            applied_ports = _vector(applied_ports, len(self.ci.port_ids), 'applied_ports')
        started = time.perf_counter()
        u = data.qpos[self.qi].copy()
        ud = data.qvel[self.vi].copy()
        q = self.ci.assemble(u)
        lift = self.ci.lift(q, ud)
        tangent, curvature, tree_velocity = lift['N_tree'], lift['c_tree'], lift['v_tree']
        tree_position = q[self.ci.tree_rows]
        base = self.base_state(data)
        base_acceleration = base['acceleration'].copy()
        sample_time = float(data.time)
        if self.acceleration_filter_s > 0:
            if self._filtered_acceleration is None:
                self._filtered_acceleration = base_acceleration.copy()
            elif self._last_sample_time is not None and sample_time > self._last_sample_time:
                weight = -np.expm1(-(sample_time - self._last_sample_time) / self.acceleration_filter_s)
                self._filtered_acceleration += weight * (base_acceleration - self._filtered_acceleration)
            base_acceleration = self._filtered_acceleration.copy()
        self._last_sample_time = sample_time
        base['used_acceleration'] = base_acceleration
        local = self.ci.dyn.evaluate(tree_position, tree_velocity, np.zeros(3))
        floating = floating_rows(local, self.cmg, tree_position, base['position'], base['rotation'],
                                 np.r_[base['velocity'], tree_velocity], self.ci.gravity)
        mass, bias = floating['mass_matrix'], floating['bias_forces']
        mass_qq, mass_qb, bias_q = mass[6:, 6:], mass[6:, :6], bias[6:]
        coupling = mass_qb @ base_acceleration if self.base_coupling else np.zeros(self.ci.nt)
        reduced_mass = tangent.T @ mass_qq @ tangent
        reduced_mass = 0.5 * (reduced_mass + reduced_mass.T)
        reduced_bias = tangent.T @ (bias_q + mass_qq @ curvature + coupling)
        port_map = self.ci.E.T @ tangent
        channel_map = port_map.T @ self.ci.G
        error = desired - u
        velocity_error = velocity - ud
        requested = acceleration + self.kp * error + self.kd * velocity_error
        if self.observer_gain > 0. and applied_ports is not None:
            # Residual of the reduced equation using measured acceleration and applied port efforts.
            measured_acceleration = data.qacc[self.vi].copy()
            residual = reduced_mass @ measured_acceleration + reduced_bias - port_map.T @ applied_ports
            self.disturbance += self.observer_gain * (residual - self.disturbance)
            if self.observer_limit is not None:
                self.disturbance = np.clip(self.disturbance, -self.observer_limit, self.observer_limit)
        # M_r u_dd + h_r = tau_applied + tau_external: compensate the estimated external load.
        if self.feedforward:
            reduced_effort = reduced_mass @ requested + reduced_bias - self.disturbance
        else:
            reduced_effort = reduced_mass @ (self.kp * error + self.kd * velocity_error) - self.disturbance
        condition = float(np.linalg.cond(channel_map))
        channels = np.linalg.solve(channel_map, reduced_effort)
        ports = self.ci.G @ channels
        # Virtual power: port power equals independent-coordinate power.
        defect = abs(float(ports @ (port_map @ ud)) - float(channels @ (channel_map.T @ ud)))
        native_q = data.qpos[self.tree_qi]
        native_v = data.qvel[self.tree_vi]
        stats = self.stats
        stats['evaluations'] += 1
        stats['shadow_max'] = max(stats['shadow_max'], normmax(native_q - tree_position))
        stats['shadow_velocity_max'] = max(stats['shadow_velocity_max'], normmax(native_v - tree_velocity))
        stats['channel_condition_max'] = max(stats['channel_condition_max'], condition)
        stats['command_peak'] = np.maximum(stats['command_peak'], np.abs(channels))
        stats['coupling_max'] = max(stats['coupling_max'], normmax(coupling))
        stats['base_acceleration_peak'] = np.maximum(stats['base_acceleration_peak'], np.abs(base['acceleration']))
        stats['virtual_power_defect_max'] = max(stats['virtual_power_defect_max'], defect)
        elapsed = time.perf_counter() - started
        stats['wall_s'] += elapsed
        stats['wall_max_s'] = max(stats['wall_max_s'], elapsed)
        self.last = dict(u=u, ud=ud, q=q, Mr=reduced_mass, hr=reduced_bias, C=channel_map, A=port_map,
                         channels=channels, requested=requested, error=error, coupling=coupling, base=base,
                         gravity_r=tangent.T @ floating['gravity_compensation'][6:])
        return channels

    def diagnostics(self):
        stats = dict(self.stats)
        stats['command_peak'] = stats['command_peak'].tolist()
        stats['base_acceleration_peak'] = stats['base_acceleration_peak'].tolist()
        stats.update(self.ci.stats)
        stats.update(method='online PACDM prediction/correction with reduced floating-base feedforward',
                     polish_tolerance=self.ci.polish_tolerance, feedforward=self.feedforward,
                     observer_gain=self.observer_gain, base_coupling=self.base_coupling,
                     base_acceleration_filter_s=self.acceleration_filter_s, kp=self.kp.tolist(), kd=self.kd.tolist(),
                     mean_wall_s=stats['wall_s'] / max(1, stats['evaluations']))
        return stats


class TravelController:
    """Bounded base-pose tracking with a PI mean-wheel-speed servo.

    All native body/joint names, drive geometry, bounds and gains are supplied
    by the caller. Left and right wheel velocities use the same forward sign.
    """

    def __init__(self, model, *, base_body, left_joints, right_joints, radius, gauge,
                 forward_bound, yaw_rate_bound, position_gain, heading_gain,
                 lateral_gain, yaw_damping_gain, kp, ki, integral_bound):
        self.m = model
        self.base_id = _body_id(model, base_body)
        left_joints, right_joints = list(left_joints), list(right_joints)
        if not left_joints or not right_joints:
            raise ValueError('At least one wheel joint is required on each side')
        if len(set(left_joints + right_joints)) != len(left_joints) + len(right_joints):
            raise ValueError('Wheel joints must be unique across both sides')
        self.left = _scalar_joint_ids(model, left_joints)
        self.right = _scalar_joint_ids(model, right_joints)
        if any(int(model.jnt_type[index]) != int(mujoco.mjtJoint.mjJNT_HINGE)
               for index in np.r_[self.left, self.right]):
            raise ValueError('Wheel joints must be hinges')
        self.vi_left, self.vi_right = model.jnt_dofadr[self.left], model.jnt_dofadr[self.right]
        self.radius, self.gauge = float(radius), float(gauge)
        if not np.all(np.isfinite([self.radius, self.gauge])) or min(self.radius, self.gauge) <= 0:
            raise ValueError('radius and gauge must be finite and positive')
        tuning = dict(forward_bound=forward_bound, yaw_rate_bound=yaw_rate_bound,
                      position_gain=position_gain, heading_gain=heading_gain,
                      lateral_gain=lateral_gain, yaw_damping_gain=yaw_damping_gain,
                      kp=kp, ki=ki, ibound=integral_bound)
        for name, value in tuning.items():
            value = float(value)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
            setattr(self, name, value)
        self.integral = np.zeros(2)
        self.last = {}

    def pose(self, data):
        rotation = data.xmat[self.base_id].reshape(3, 3)
        position = data.xpos[self.base_id]
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        return np.r_[position[:2], yaw]

    def evaluate(self, data, target, rate, dt, *, update_outer=True):
        target, rate = _vector(target, 3, 'target'), _vector(rate, 3, 'rate')
        dt = float(dt)
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('dt must be finite and positive')
        if update_outer or not self.last:
            pose = self.pose(data)
            yaw = pose[2]
            forward = np.array([np.cos(yaw), np.sin(yaw)])
            lateral = np.array([-np.sin(yaw), np.cos(yaw)])
            error = target[:2] - pose[:2]
            heading = np.arctan2(np.sin(target[2] - yaw), np.cos(target[2] - yaw))
            reference_forward = np.array([np.cos(target[2]), np.sin(target[2])])
            reference_speed = float(rate[:2] @ reference_forward)
            rotation = data.xmat[self.base_id].reshape(3, 3)
            forward_3d = rotation[:, 0]
            forward_derivative = np.cross(data.cvel[self.base_id, :3], forward_3d)
            yaw_rate = float((forward_3d[0] * forward_derivative[1] - forward_3d[1] * forward_derivative[0]) /
                             max(1e-8, forward_3d[0] ** 2 + forward_3d[1] ** 2))
            speed_command = np.clip(rate[:2] @ forward + self.position_gain * (error @ forward),
                                    -self.forward_bound, self.forward_bound)
            steering = self.lateral_gain * reference_speed * float(error @ lateral)
            damping = self.yaw_damping_gain * (float(rate[2]) - yaw_rate)
            omega = np.clip(rate[2] + self.heading_gain * heading + steering + damping,
                            -self.yaw_rate_bound, self.yaw_rate_bound)
            desired = np.array([speed_command - 0.5 * self.gauge * omega,
                                speed_command + 0.5 * self.gauge * omega]) / self.radius
            self.last = dict(pose=pose, target=target.copy(), desired=desired, error=error,
                             heading_error=heading, yaw_rate=yaw_rate)
        desired = self.last['desired']
        speed = np.array([data.qvel[self.vi_left].mean(), data.qvel[self.vi_right].mean()])
        error = desired - speed
        self.integral = np.clip(self.integral + dt * error, -self.ibound, self.ibound)
        torque = self.kp * error + self.ki * self.integral
        self.last.update(speed=speed, command=torque)
        return torque
