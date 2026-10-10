"""Pressure-state hydraulic cylinders and first-order rotary drive dynamics.

Geometry, pressure, flow, response, friction and drive limits are supplied by
the caller. This module contains no machine-specific actuator parameters.
"""
from __future__ import annotations

import numpy as np


def _finite_scalar(value, name, *, positive=False, nonnegative=False):
    value = float(value)
    if not np.isfinite(value) or (positive and value <= 0) or (nonnegative and value < 0):
        requirement = 'positive' if positive else 'nonnegative' if nonnegative else 'finite'
        raise ValueError(f'{name} must be finite and {requirement}')
    return value


def _vector(value, size, name, *, positive=False, nonnegative=False):
    value = np.asarray(value, float)
    if value.shape != (size,) or not np.all(np.isfinite(value)):
        raise ValueError(f'{name} must be a finite vector of length {size}')
    if (positive and np.any(value <= 0)) or (nonnegative and np.any(value < 0)):
        raise ValueError(f'{name} has an invalid sign')
    return value.copy()


class HydraulicBank:
    """Double-acting cylinders plus one first-order rotary drive.

    ``parameters`` must provide every key in ``required_parameters``. Cylinder
    vectors have the length of ``area_A_m2`` and share its order. Dead volume
    may be scalar or an array shaped ``(number_of_cylinders, 2)``. All other
    non-vector parameters are scalars. Units are identified in parameter keys;
    backpressure factors are dimensionless. No physical values are inferred.
    """

    required_parameters = (
        'supply_pressure_Pa', 'bulk_modulus_Pa', 'area_A_m2', 'area_B_m2',
        'closed_m', 'stroke_m', 'dead_volume_m3', 'valve_max_flow_m3_s',
        'pressure_response_s', 'leakage_m3_s_Pa', 'minimum_command_pressure_Pa',
        'cylinder_coulomb_N', 'cylinder_viscous_N_s_m', 'rotary_torque_limit_Nm',
        'rotary_response_s', 'rotary_coulomb_Nm', 'rotary_viscous_Nm_s_rad',
        'friction_smoothing_speed_SI', 'stroke_domain_tolerance_m',
        'backpressure_factor', 'backpressure_cap',
    )

    def __init__(self, parameters, initial_force, initial_rotary_torque=0.):
        missing = [key for key in self.required_parameters if key not in parameters]
        if missing:
            raise ValueError('Missing actuator parameters: ' + ', '.join(missing))
        self.p = {key: parameters[key] for key in self.required_parameters}
        area = np.asarray(self.p['area_A_m2'], float)
        if area.ndim != 1 or area.size == 0:
            raise ValueError('area_A_m2 must be a nonempty cylinder vector')
        self.n = int(area.size)
        positive_vectors = ('area_A_m2', 'area_B_m2', 'stroke_m', 'valve_max_flow_m3_s')
        nonnegative_vectors = ('closed_m', 'cylinder_coulomb_N', 'cylinder_viscous_N_s_m')
        for key in positive_vectors:
            self.p[key] = _vector(self.p[key], self.n, key, positive=True)
        for key in nonnegative_vectors:
            self.p[key] = _vector(self.p[key], self.n, key, nonnegative=True)
        if np.any(self.p['area_B_m2'] > self.p['area_A_m2']):
            raise ValueError('area_B_m2 cannot exceed area_A_m2')
        positive_scalars = ('supply_pressure_Pa', 'bulk_modulus_Pa', 'pressure_response_s',
                            'rotary_response_s', 'friction_smoothing_speed_SI')
        nonnegative_scalars = ('leakage_m3_s_Pa', 'minimum_command_pressure_Pa',
                               'rotary_torque_limit_Nm', 'rotary_coulomb_Nm',
                               'rotary_viscous_Nm_s_rad', 'stroke_domain_tolerance_m',
                               'backpressure_factor', 'backpressure_cap')
        for key in positive_scalars:
            self.p[key] = _finite_scalar(self.p[key], key, positive=True)
        for key in nonnegative_scalars:
            self.p[key] = _finite_scalar(self.p[key], key, nonnegative=True)
        try:
            dead_volume = np.broadcast_to(np.asarray(self.p['dead_volume_m3'], float), (self.n, 2)).copy()
        except ValueError as exc:
            raise ValueError('dead_volume_m3 must be scalar or shaped (number_of_cylinders, 2)') from exc
        if not np.all(np.isfinite(dead_volume)) or np.any(dead_volume <= 0):
            raise ValueError('dead_volume_m3 must be finite and positive')
        self.p['dead_volume_m3'] = dead_volume
        self.A, self.B = self.p['area_A_m2'], self.p['area_B_m2']
        self.ps = self.p['supply_pressure_Pa']
        self.qmax = self.p['valve_max_flow_m3_s']
        self.closed, self.stroke = self.p['closed_m'], self.p['stroke_m']
        self.backpressure_factor = self.p['backpressure_factor']
        self.backpressure_cap = self.p['backpressure_cap']
        if self.backpressure_cap > 1:
            raise ValueError('backpressure_cap must be a fraction between zero and one')
        if self.p['minimum_command_pressure_Pa'] > self.ps:
            raise ValueError('minimum_command_pressure_Pa cannot exceed supply pressure')
        self.pressure = self.targets(_vector(initial_force, self.n, 'initial_force'))
        self.rotary_limit = self.p['rotary_torque_limit_Nm']
        initial_rotary_torque = _finite_scalar(initial_rotary_torque, 'initial_rotary_torque')
        self.rotary = float(np.clip(initial_rotary_torque, -self.rotary_limit, self.rotary_limit))
        self.saturation_count = 0
        self.relief_count = 0

    def targets(self, force, velocity=None):
        force = _vector(force, self.n, 'force')
        low = self.p['minimum_command_pressure_Pa']
        pressure_a = np.where(force >= 0, (force + self.B * low) / self.A, low)
        pressure_b = np.where(force >= 0, low, (self.A * low - force) / self.B)
        if velocity is not None:
            velocity = _vector(velocity, self.n, 'velocity')
            discharge_a = np.minimum(self.A * velocity, 0)
            discharge_b = np.minimum(-self.B * velocity, 0)
            # Q = q_max sqrt(p / p_s) requires p >= p_s (Q / q_max)^2 for meter-out flow.
            # Equal force preload in both chambers preserves the commanded force before pressure clipping.
            backpressure_a = np.minimum(low + self.backpressure_factor * self.ps * (discharge_a / self.qmax) ** 2,
                                        self.backpressure_cap * self.ps)
            backpressure_b = np.minimum(low + self.backpressure_factor * self.ps * (discharge_b / self.qmax) ** 2,
                                        self.backpressure_cap * self.ps)
            preload = np.maximum.reduce([self.A * (backpressure_a - pressure_a),
                                         self.B * (backpressure_b - pressure_b), np.zeros(self.n)])
            pressure_a = pressure_a + preload / self.A
            pressure_b = pressure_b + preload / self.B
        return np.clip(np.column_stack((pressure_a, pressure_b)), 0, self.ps)

    def volumes(self, position):
        extension = _vector(position, self.n, 'position') - self.closed
        tolerance = self.p['stroke_domain_tolerance_m']
        if np.any(extension < -tolerance) or np.any(extension > self.stroke + tolerance):
            raise ValueError('Cylinder left its supplied stroke domain: ' + str(extension))
        extension = np.clip(extension, 0., self.stroke)
        return self.p['dead_volume_m3'] + np.column_stack((self.A * extension,
                                                         self.B * (self.stroke - extension)))

    def stored_energy(self, position):
        return float(np.sum(self.volumes(position) * self.pressure ** 2) / (2 * self.p['bulk_modulus_Pa']))

    def evaluate(self, position, velocity, force_command, rotary_command, rotary_velocity, dt):
        velocity = _vector(velocity, self.n, 'velocity')
        command = _vector(force_command, self.n, 'force_command')
        rotary_command = _finite_scalar(rotary_command, 'rotary_command')
        rotary_velocity = _finite_scalar(rotary_velocity, 'rotary_velocity')
        dt = _finite_scalar(dt, 'dt', positive=True)
        pressure = self.pressure
        volume = self.volumes(position)
        beta = self.p['bulk_modulus_Pa']
        volume_rate = np.column_stack((self.A * velocity, -self.B * velocity))
        target = self.targets(command, velocity)
        leak = self.p['leakage_m3_s_Pa'] * (pressure[:, 0] - pressure[:, 1])
        leakage_flow = np.column_stack((leak, -leak))
        request = volume_rate + leakage_flow + volume / beta * (target - pressure) / self.p['pressure_response_s']
        capacity = self.qmax[:, None] * np.sqrt(np.maximum(np.where(request >= 0, self.ps - pressure, pressure), 0) / self.ps)
        flow = np.clip(request, -capacity, capacity)
        unbounded = pressure + dt * beta / volume * (flow - volume_rate - leakage_flow)
        bounded = np.clip(unbounded, 0, self.ps)
        relief = (unbounded - bounded) * volume / (dt * beta)
        pressure_derivative = beta / volume * (flow - volume_rate - leakage_flow - relief)
        epsilon = self.p['friction_smoothing_speed_SI']
        friction = self.p['cylinder_coulomb_N'] * np.tanh(velocity / epsilon) + self.p['cylinder_viscous_N_s_m'] * velocity
        rotary_friction = self.p['rotary_coulomb_Nm'] * np.tanh(rotary_velocity / epsilon) + self.p['rotary_viscous_Nm_s_rad'] * rotary_velocity
        cylinder_force = self.A * pressure[:, 0] - self.B * pressure[:, 1] - friction
        rotary_torque = self.rotary - rotary_friction
        rotary_derivative = (np.clip(rotary_command, -self.rotary_limit, self.rotary_limit) - self.rotary) / self.p['rotary_response_s']
        supply = self.ps * np.maximum(flow, 0).sum()
        fluid_input = np.sum(pressure * flow)
        saturated = bool(np.any(np.abs(request) > capacity + 1e-15))
        limited = bool(np.any(target <= 0) | np.any(target >= self.ps))
        self.saturation_count += int(saturated)
        self.relief_count += int(np.any(relief != 0))
        power = dict(supply=float(supply), throttle=float(supply - fluid_input),
                     leakage=float(np.sum(leak * (pressure[:, 0] - pressure[:, 1]))),
                     relief=float(np.sum(pressure * relief)),
                     friction=float(friction @ velocity + rotary_friction * rotary_velocity),
                     mechanical=float(cylinder_force @ velocity + rotary_torque * rotary_velocity))
        return dict(cylinder_force=cylinder_force, rotary_torque=rotary_torque,
                    pressure_derivative=pressure_derivative, rotary_derivative=rotary_derivative,
                    flow=flow, power=power, valve_saturated=saturated,
                    pressure_limit_active=limited, pressure_max=float(pressure.max()))

    def advance(self, item, dt):
        dt = _finite_scalar(dt, 'dt', positive=True)
        self.pressure = self.pressure + dt * item['pressure_derivative']
        self.rotary = self.rotary + dt * item['rotary_derivative']
        if self.pressure.min() < -1e-6 or self.pressure.max() > self.ps + 1e-6:
            raise ValueError('Pressure bound violated')
        self.pressure = np.clip(self.pressure, 0, self.ps)


class TrackDrive:
    """First-order per-wheel torque response for two drive sides.

    ``rollers_per_side`` is a positive integer or a pair of positive integers.
    Velocity vectors list every left-side wheel before every right-side wheel.
    Limits, response time and viscous resistance are provided by the caller.
    """

    def __init__(self, *, torque_limit, response_s, rollers_per_side, viscous):
        self.limit = _finite_scalar(torque_limit, 'torque_limit', nonnegative=True)
        self.tau = _finite_scalar(response_s, 'response_s', positive=True)
        self.viscous = _finite_scalar(viscous, 'viscous', nonnegative=True)
        counts = np.asarray(rollers_per_side)
        if counts.ndim == 0:
            counts = np.repeat(counts, 2)
        if counts.shape != (2,) or not np.all(np.isfinite(counts)) or np.any(counts < 1) or np.any(counts != counts.astype(int)):
            raise ValueError('rollers_per_side must be a positive integer or pair of positive integers')
        self.nrollers = counts.astype(int)
        self.torque = np.zeros(2)

    def evaluate(self, command_per_roller, roller_speeds, dt):
        command = _vector(command_per_roller, 2, 'command_per_roller')
        speeds = _vector(roller_speeds, int(self.nrollers.sum()), 'roller_speeds')
        _finite_scalar(dt, 'dt', positive=True)
        bounded_command = np.clip(command, -self.limit, self.limit)
        derivative = (bounded_command - self.torque) / self.tau
        effort = np.repeat(self.torque, self.nrollers) - self.viscous * speeds
        return dict(effort=effort, derivative=derivative, saturated=bool(np.any(np.abs(command) > self.limit)))

    def advance(self, item, dt):
        dt = _finite_scalar(dt, 'dt', positive=True)
        self.torque = np.clip(self.torque + dt * item['derivative'], -self.limit, self.limit)
