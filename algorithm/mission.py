"""Trajectory phases and guarded execution for caller-supplied models and tasks."""
from __future__ import annotations

from copy import copy

import numpy as np

from .kinematics import smooth


def _vector(value, name):
    value = np.asarray(value, float).copy()
    if value.ndim != 1 or value.size == 0 or not np.all(np.isfinite(value)):
        raise ValueError(f'{name} must be a nonempty finite vector')
    return value


class Phase:
    def __init__(self, name, duration, kind='arm', guard=None, timeout_extension=8.):
        self.name, self.duration, self.kind, self.guard = name, float(duration), kind, guard
        self.timeout_extension = float(timeout_extension)
        if not np.isfinite(self.duration) or self.duration <= 0:
            raise ValueError('Phase duration must be finite and positive')
        if not np.isfinite(self.timeout_extension) or self.timeout_extension < 0:
            raise ValueError('Timeout extension must be finite and nonnegative')
        self.start_u = None

    def reference(self, tau):
        raise NotImplementedError


class Hold(Phase):
    def __init__(self, name, duration, u, kind='arm', guard=None, **kw):
        super().__init__(name, duration, kind, guard, **kw)
        self.u = _vector(u, 'u')

    def reference(self, tau):
        return self.u.copy(), np.zeros_like(self.u), np.zeros_like(self.u)


class JointMove(Phase):
    def __init__(self, name, duration, u0, u1, guard=None, **kw):
        super().__init__(name, duration, 'arm', guard, **kw)
        self.u0, self.u1 = _vector(u0, 'u0'), _vector(u1, 'u1')
        if self.u0.shape != self.u1.shape:
            raise ValueError('Joint endpoints must have the same shape')

    def reference(self, tau):
        f, df, ddf = smooth(tau, 0., self.duration)
        delta = self.u1 - self.u0
        return self.u0 + f * delta, df * delta, ddf * delta


class PathMove(Phase):
    """A straight planar tool-point path with linear pitch and quintic timing.

    ``chain.ik(target_rz, pitch, point)`` must return the terminal-chain joints.
    The leading reference coordinate is the supplied swing angle.
    """

    def __init__(self, name, duration, chain, swing, point, rz0, rz1, pitch0, pitch1, guard=None, **kw):
        super().__init__(name, duration, 'arm', guard, **kw)
        self.chain, self.swing, self.point = chain, float(swing), _vector(point, 'point')
        self.rz0, self.rz1 = _vector(rz0, 'rz0'), _vector(rz1, 'rz1')
        if self.rz0.shape != (2,) or self.rz1.shape != (2,):
            raise ValueError('Planar path endpoints must have two coordinates')
        self.pitch0, self.pitch1 = float(pitch0), float(pitch1)
        if not all(np.isfinite(x) for x in (self.swing, self.pitch0, self.pitch1)):
            raise ValueError('Swing and pitch values must be finite')
        self.u0 = self._u(0.)
        self.u1 = self._u(duration)

    def _u(self, tau):
        f, _, _ = smooth(tau, 0., self.duration)
        rz = self.rz0 + f * (self.rz1 - self.rz0)
        pitch = self.pitch0 + f * (self.pitch1 - self.pitch0)
        return np.r_[self.swing, self.chain.ik(rz, pitch, self.point)]

    def reference(self, tau, h=1e-3):
        if not np.isfinite(h) or h <= 0:
            raise ValueError('The derivative step must be finite and positive')
        tau = float(np.clip(tau, 0., self.duration))
        u = self._u(tau)
        up, um = self._u(min(tau + h, self.duration)), self._u(max(tau - h, 0.))
        hp, hm = min(tau + h, self.duration) - tau, tau - max(tau - h, 0.)
        if hp > 0 and hm > 0:
            ud = (up - um) / (hp + hm)
            udd = 2 * (hm * up - (hp + hm) * u + hp * um) / (hp * hm * (hp + hm))
        else:
            ud, udd = np.zeros_like(u), np.zeros_like(u)
        return u, ud, udd


class SwingMove(Phase):
    """Quintic movement of a caller-selected rotary coordinate."""

    def __init__(self, name, duration, u_hold, swing0, swing1, guard=None, *, coordinate_index=0, **kw):
        super().__init__(name, duration, 'arm', guard, **kw)
        self.u0 = _vector(u_hold, 'u_hold')
        if not isinstance(coordinate_index, (int, np.integer)) or not 0 <= coordinate_index < self.u0.size:
            raise ValueError('coordinate_index is outside the reference vector')
        if not np.isfinite(swing0) or not np.isfinite(swing1):
            raise ValueError('Rotary endpoints must be finite')
        self.u0[coordinate_index] = swing0
        self.u1 = self.u0.copy()
        self.u1[coordinate_index] = swing1

    def reference(self, tau):
        f, df, ddf = smooth(tau, 0., self.duration)
        delta = self.u1 - self.u0
        return self.u0 + f * delta, df * delta, ddf * delta


class Travel(Phase):
    def __init__(self, name, duration, u_hold, pose0, pose1, guard=None, **kw):
        super().__init__(name, duration, 'travel', guard, **kw)
        self.u = _vector(u_hold, 'u_hold')
        self.pose0, self.pose1 = _vector(pose0, 'pose0'), _vector(pose1, 'pose1')
        if self.pose0.shape != (3,) or self.pose1.shape != (3,):
            raise ValueError('Base poses must be planar (x, y, yaw) vectors')

    def reference(self, tau):
        return self.u.copy(), np.zeros_like(self.u), np.zeros_like(self.u)

    def base_reference(self, tau):
        f, df, _ = smooth(tau, 0., self.duration)
        delta = self.pose1 - self.pose0
        return self.pose0 + f * delta, df * delta


class Mission:
    """Execute supplied phases; geometry and actuator checks are caller callbacks.

    ``validate_state(u)`` checks model feasibility and may raise or return False.
    ``actuator_rates(u, ud)`` returns a rate vector in the same order as
    ``rate_limits``; each row gives positive and negative speed allowances.
    These rate checks sample each phase and stretch its duration as needed.
    ``guards`` maps labels to callbacks ``guard(observation, mission)``.
    A callable ``Phase.guard`` uses the same signature. Named material guards
    consume ``load_mass`` and ``material_speed`` observations.
    """

    def __init__(self, phases, initial_base_pose, *, validate_state=None,
                 actuator_rates=None, rate_limits=None, guards=None,
                 tracking_tolerance=0.035, settled_speed=0.06, load_quantum=None,
                 base_position_tolerance=0.12, base_yaw_tolerance=0.05,
                 emptied_speed=0.25):
        self.validate_state = validate_state
        self.actuator_rates = actuator_rates
        self.rate_limits = None if rate_limits is None else np.asarray(rate_limits, float).copy()
        if (self.actuator_rates is None) != (self.rate_limits is None):
            raise ValueError('Supply actuator_rates and rate_limits together')
        if self.rate_limits is not None:
            if self.rate_limits.ndim != 2 or self.rate_limits.shape[1] != 2 or self.rate_limits.shape[0] == 0:
                raise ValueError('rate_limits must have shape (number of rates, 2)')
            if not np.all(np.isfinite(self.rate_limits)) or np.any(self.rate_limits <= 0):
                raise ValueError('Rate allowances must be finite and positive')
        self.guards = {} if guards is None else dict(guards)
        self.tol = float(tracking_tolerance)
        self.settled_speed = float(settled_speed)
        self.load_quantum = None if load_quantum is None else float(load_quantum)
        self.base_position_tolerance = float(base_position_tolerance)
        self.base_yaw_tolerance = float(base_yaw_tolerance)
        self.emptied_speed = float(emptied_speed)
        for value in (self.tol, self.settled_speed, self.base_position_tolerance,
                      self.base_yaw_tolerance, self.emptied_speed):
            if not np.isfinite(value) or value <= 0:
                raise ValueError('Guard tolerances must be finite and positive')
        if self.load_quantum is not None and (not np.isfinite(self.load_quantum) or self.load_quantum <= 0):
            raise ValueError('load_quantum must be finite and positive')
        self.base_target = _vector(initial_base_pose, 'initial_base_pose')
        if self.base_target.shape != (3,):
            raise ValueError('The initial base pose must be (x, y, yaw)')
        self.base_rate = np.zeros_like(self.base_target)
        self.phases = []
        self.events, self.failures = [], []
        self.phase, self.entered, self.done, self.finished_at = 0, 0., False, None
        for phase in phases:
            self.add_phase(phase)
        if not self.phases:
            raise ValueError('Supply at least one phase')
        self.last = self.phases[0].reference(0.)

    def _validate(self, u):
        if self.validate_state is not None:
            result = self.validate_state(np.asarray(u, float).copy())
            if isinstance(result, (bool, np.bool_)) and not result:
                raise ValueError('The supplied model rejected a reference state')
            return result
        return None

    def add_phase(self, phase):
        """Append a phase after sampled feasibility, rate, and continuity checks."""
        phase = copy(phase)
        known_guards = {'tracking_loose', 'load_aware', 'material_settled', 'soil_settled',
                        'base_arrived', 'emptied', 'loaded'}
        if phase.guard is not None and not callable(phase.guard) and phase.guard not in known_guards and phase.guard not in self.guards:
            raise ValueError(f'Unknown phase guard: {phase.guard}')
        if phase.guard in ('load_aware', 'emptied') and phase.guard not in self.guards and self.load_quantum is None:
            raise ValueError(f'{phase.guard} requires load_quantum')
        for _ in range(12):
            factor = self._rate_factor(phase)
            if factor <= 1.0:
                break
            phase.duration = float(np.ceil(phase.duration * factor * 10.) / 10.)
        else:
            raise ValueError(f'{phase.name}: could not satisfy actuator rate limits')
        u_now = _vector(phase.reference(0.)[0], 'phase start reference')
        if self.phases:
            prev = self.phases[-1]
            u_prev = prev.reference(prev.duration)[0]
            if np.shape(u_prev) != u_now.shape or np.max(np.abs(u_prev - u_now)) > 1e-6:
                raise ValueError(f'Reference discontinuity entering {phase.name}')
        for tau in np.linspace(0., phase.duration, 7):
            self._validate(phase.reference(tau)[0])
        self.phases.append(phase)
        self.names = [p.name for p in self.phases]
        self.duration_nominal = float(sum(p.duration for p in self.phases))
        return phase

    def _add(self, make, duration):
        """Append a phase built by the supplied ``make(duration)`` callback."""
        return self.add_phase(make(duration))

    def _rate_factor(self, phase):
        if self.actuator_rates is None:
            return 0.
        worst = 0.
        for tau in np.linspace(0., phase.duration, 41):
            u, ud, _ = phase.reference(tau)
            rates = np.asarray(self.actuator_rates(u, ud), float)
            if rates.shape != (self.rate_limits.shape[0],) or not np.all(np.isfinite(rates)):
                raise ValueError('actuator_rates returned an invalid rate vector')
            limits = np.where(rates >= 0, self.rate_limits[:, 0], self.rate_limits[:, 1])
            worst = max(worst, float(np.max(np.abs(rates) / limits)))
        return worst

    def _end_u(self):
        prev = self.phases[-1]
        return prev.reference(prev.duration)[0]

    def references(self, t):
        ph = self.phases[self.phase]
        tau = np.clip(float(t) - self.entered, 0., ph.duration)
        reference = ph.reference(tau)
        if ph.kind == 'travel':
            self.base_target, self.base_rate = ph.base_reference(tau)
        else:
            self.base_rate = np.zeros_like(self.base_target)
        self.last = reference
        return reference, self.base_target.copy(), self.base_rate.copy()

    def update(self, t, obs):
        """Advance the phase machine using tracking and optional guard observations."""
        if self.done:
            return
        t = float(t)
        ph = self.phases[self.phase]
        tau = t - self.entered
        err = float(obs['tracking_error'])
        if not np.isfinite(err) or err < 0:
            raise ValueError('tracking_error must be finite and nonnegative')
        ready = tau >= ph.duration
        guard = ph.guard
        if guard == 'tracking_loose':
            ready = ready and err < 3 * self.tol
        elif guard == 'load_aware' and guard not in self.guards:
            ready = ready and (err < 3 * self.tol or obs['load_mass'] >= 3 * self.load_quantum)
        else:
            ready = ready and err < self.tol
        if callable(guard):
            ready = ready and bool(guard(obs, self))
        elif guard in self.guards:
            ready = ready and bool(self.guards[guard](obs, self))
        elif guard in ('material_settled', 'soil_settled'):
            ready = ready and obs['material_speed'] < self.settled_speed
        elif guard == 'base_arrived':
            target, _ = ph.base_reference(min(max(tau, 0.), ph.duration)) if ph.kind == 'travel' else (self.base_target, self.base_rate)
            pose = _vector(obs['base_pose'], 'base_pose')
            if pose.shape != (3,):
                raise ValueError('base_pose must be (x, y, yaw)')
            yaw_error = (pose[2] - target[2] + np.pi) % (2 * np.pi) - np.pi
            ready = ready and np.linalg.norm(pose[:2] - target[:2]) < self.base_position_tolerance and abs(yaw_error) < self.base_yaw_tolerance
        elif guard == 'emptied':
            ready = ready and obs['load_mass'] < 0.5 * self.load_quantum and obs['material_speed'] < self.emptied_speed
        elif guard == 'loaded':
            ready = ready and obs['load_mass'] > 0.
        if not ready:
            if tau > ph.duration + ph.timeout_extension:
                detail = f'tracking error {err:.4f}'
                if guard == 'loaded' and guard not in self.guards and obs['load_mass'] <= 0.:
                    detail += ', no load detected'
                self.failures.append(f'Phase timeout: {ph.name} ({detail})')
                self.done, self.finished_at = True, t
            return
        self.events.append(dict(time_s=t, phase=ph.name, tracking_error=err,
                                observations=dict(obs)))
        if self.phase == len(self.phases) - 1:
            self.done, self.finished_at = True, t
        else:
            self.phase += 1
            self.entered = t
            self.phases[self.phase].start_u = self.last[0].copy()
