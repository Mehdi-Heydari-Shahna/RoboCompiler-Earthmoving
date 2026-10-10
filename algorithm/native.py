"""Residual-controlled MuJoCo forward dynamics and implicitfast integration.

This implementation requires MuJoCo 3.3.7 and a caller-supplied model using the
native Newton solver. It retries the numerical initializer without changing
physical inputs and reports any accepted degraded solve.
"""
import numpy as np
import mujoco

SUPPORTED_VERSION='3.3.7'


def validate_model(model):
    if mujoco.__version__ != SUPPORTED_VERSION:
        raise RuntimeError('native_step was verified only against MuJoCo '+SUPPORTED_VERSION)
    if model.opt.integrator != mujoco.mjtIntegrator.mjINT_IMPLICITFAST:
        raise ValueError('native_step requires implicitfast; no integrator is substituted')
    if model.nplugin:
        raise ValueError('Native plugin callbacks are outside this audited pipeline scope')
    for name in ('get_mjcb_control','get_mjcb_passive'):
        if hasattr(mujoco,name) and getattr(mujoco,name)() is not None:
            raise ValueError('Stateful callbacks are outside this audited pipeline scope')


def checked_forward(model,data):
    """The pre-integrator checked forward segment of native mj_step.

    Model suitability is checked once with validate_model before entering a loop.
    """
    mujoco.mj_checkPos(model,data)
    mujoco.mj_checkVel(model,data)
    mujoco.mj_forward(model,data)


def advance_after_forward(model,data):
    """Consume the existing full forward solution using the native integrator.

    PRECONDITION: checked_forward has run for the current, unchanged inputs. No
    manual activation, time, warmstart, position or velocity assignment occurs.
    """
    mujoco.mj_checkAcc(model,data)
    if int(model.opt.enableflags) & int(mujoco.mjtEnableBit.mjENBL_FWDINV):
        mujoco.mj_compareFwdInv(model,data)
    mujoco.mj_implicit(model,data)



FORCE_TOLERANCE=1e-8
DEGRADED_TOLERANCE=1e-4
STATS=dict(forward_calls=0,warm_restarts=0,cold_restarts=0,degraded_steps=0,degraded_max_residual=0.,degraded_first_time_s=None)
PHYSICAL_FIELDS=('qpos','qvel','act','ctrl','qfrc_applied','xfrc_applied',
                 'mocap_pos','mocap_quat','eq_active','userdata')


def force_residual(model,data):
    """Registered infinity-norm balance, independently evaluated from native M."""
    ma=np.zeros(model.nv)
    mujoco.mj_mulM(model,data,ma,data.qacc)
    residual=ma-data.qfrc_smooth-data.qfrc_constraint
    scale=max(1.,float(np.max(abs(ma),initial=0)),float(np.max(abs(data.qfrc_bias),initial=0)))
    value=float(np.max(abs(residual),initial=0))/scale
    if not np.isfinite(value):
        raise FloatingPointError('Nonfinite native force-balance residual')
    return value


def accurate_forward(model,data):
    """Run native forward dynamics, retrying numerical initialization if needed.

    Returns residuals and restart counts. Any mutation of physical inputs, or a
    remaining residual above DEGRADED_TOLERANCE, raises; a remaining residual
    between FORCE_TOLERANCE and DEGRADED_TOLERANCE is accepted as a counted and
    printed degraded step (see the module docstring). Model options are never
    changed. In particular, this is not an alternate integrator, and it is not
    bitwise equivalent to a single unconverged solve.
    """
    validate_model(model)
    if model.opt.solver != mujoco.mjtSolver.mjSOL_NEWTON:
        raise ValueError('Residual-controlled restart was validated with native Newton only')
    before={key:getattr(data,key).copy() for key in PHYSICAL_FIELDS}
    before_time=float(data.time)
    checked_forward(model,data)
    history=[force_residual(model,data)]
    retries=cold=0
    if history[-1]>FORCE_TOLERANCE:
        # Keep the current native solution, not its previous-step initializer.
        data.qacc_warmstart[:]=data.qacc
        checked_forward(model,data)
        history.append(force_residual(model,data));retries+=1
    if history[-1]>FORCE_TOLERANCE:
        data.qacc_warmstart[:]=0.
        checked_forward(model,data)
        history.append(force_residual(model,data));retries+=1;cold+=1
    STATS['forward_calls']+=1
    STATS['warm_restarts']+=int(retries>0)
    STATS['cold_restarts']+=cold
    degraded=int(FORCE_TOLERANCE<history[-1]<=DEGRADED_TOLERANCE)
    if degraded:
        STATS['degraded_steps']+=1
        STATS['degraded_max_residual']=max(STATS['degraded_max_residual'],history[-1])
        if STATS['degraded_first_time_s'] is None:
            STATS['degraded_first_time_s']=float(data.time)
        print(f'native solve: degraded acceptance at t={data.time:.4f} s, residual history {history}, '
              f'solver iterations {int(data.solver_niter[0])}, ncon {data.ncon}',flush=True)
    changed=[key for key,value in before.items() if not np.array_equal(value,getattr(data,key))]
    if float(data.time)!=before_time:changed.append('time')
    if changed:
        raise RuntimeError('Native forward changed physical inputs: '+', '.join(changed))
    if history[-1]>DEGRADED_TOLERANCE:
        raise RuntimeError(f'Native force solve did not converge after {retries} restarts: {history}')
    return dict(initial_residual=history[0],final_residual=history[-1],
                retry_count=retries,cold_count=cold,degraded=degraded,residual_history=history)


