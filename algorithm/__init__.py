"""Graph compilation, constrained dynamics, and model-driven simulation tools.

The physical graph is supplied by the caller. Bodies use ``id``, ``mass_kg``,
``com_m``, and ``inertia_com_kg_m2``. Joints use ``id``, ``type`` (revolute,
prismatic, or fixed), ``body_a``, ``body_b``, ``T_AJ``, ``T_BJ``, and ``axis``.
The generated reference frame is named ``world``. Supply named moving-joint
values in ``seed.joints``, positive ``branch_window.revolute_rad`` and
``branch_window.prismatic_m``, ``gravity_m_s2``, ``actuators``, and a mapping
of partition names to requested independent joint IDs in ``partition_requests``.
Actuators use ``id`` and ``joint_id``; ``actuator_groups`` maps channel IDs to
actuated joint IDs when grouped control is required.

MuJoCo, hydraulic, tool, trajectory, and environment interfaces are available
in their respective modules. These interfaces accept caller-supplied models
and parameters; the package contains no robot definitions or assets.
"""

from .compiler import Compilation, ModelError, compile_graph
from .reduced import CompiledInterface
from .model_input import load_graph

__all__ = ["Compilation", "CompiledInterface", "ModelError", "compile_graph", "load_graph"]
