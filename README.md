# Graph-Based Modeling and Simulation of a Hydraulic Excavator Using RoboCompiler

Mehdi Heydari Shahna · Seihun Kim · Soyi Jung · Soohyun Park · Jouni Mattila · Joongheon Kim

This repository accompanies the manuscript above. It extends the [RoboCompiler](https://github.com/Mehdi-Heydari-Shahna/RoboCompiler) workflow to a hydraulic excavator with coupled hydraulic, soil, and material-transfer models. The physical graph retains four hydraulic cylinders, the bucket rocker–coupler transmission, five mechanical loops, and the associated actuator effort ports. Compiled assembly, motion, and effort maps connect the mechanism to a reduced-dynamics controller and a **MuJoCo** plant with pressure-state actuators and a mobile undercarriage, while a separate **Pinocchio** representation supports independent mechanics checks.

## Main components

- A machine-specific physical graph connecting closed-chain reduction and actuator-effort maps to hydraulic actuation, control, and native simulation.
- A layered bank/loose-soil model coupling cutting resistance, support, excavation, retention, spill, deposition, and terrain relaxation to bucket motion, with explicit soil-mass accounting.
- Carried soil updates the plant's bucket mass, centroid, and inertia.
- A guarded construction-site mission combining travel, three grading stages, and two excavation cycles.

## Repository contents

| Path | Content |
|---|---|
| `algorithm/` | Python package: graph compilation, PACDM closure, reduced dynamics, control, hydraulics, soil, and terrain |
| `videos/site_overview.mp4` | Video of the simulated site mission |

## Demonstration video

[Site overview](videos/site_overview.mp4) (4 min 37 s, shown at twice real time): approach travel, three levelling stages, two dig–lift–slew–dump cycles, and the finished site with the levelled pad, spoil heap, and excavation.

## Code overview

| Module | Content |
|---|---|
| `compiler.py` | Compiles physical graph records into a tree, loop cuts, and coordinate partitions |
| `loop_graph.py` | Loop-closure evaluators and graph interfaces for PACDM assembly |
| `pacdm_core.py` | Path-Assembly Closure Differential Mapping (PACDM) with angular-first SE(3) twists |
| `reduced.py` | Compiled graph assembly, tangent lift, curvature, and reduced dynamics |
| `tree_dynamics.py` | NumPy rigid-body dynamics from canonical mechanism graph (CMG) records |
| `controller.py` | Compiled-interface motion control and differential-drive travel control |
| `hydraulics.py` | Pressure-state hydraulic cylinders and first-order rotary drive dynamics |
| `soil.py` | Continuum soil with bank and loose layers and an explicit soil–tool wrench |
| `terrain.py`, `environment.py` | Deformable soil heightfield, construction-site terrain, and lighting for a MuJoCo model |
| `content_mesh.py` | Render geometry of the soil carried in the bucket |
| `coupling.py` | Tool kinematics, payload coupling, and soil-state recording |
| `mission.py` | Trajectory phases and guarded execution |
| `kinematics.py` | Planar reference kinematics |
| `native.py` | Residual-controlled MuJoCo forward dynamics |
| `pinocchio_model.py` | Pinocchio backend built from the same physical graph records |
| `model_input.py`, `__main__.py` | JSON model input and the command-line compiler |

The package contains no robot definitions or assets. The physical graph (JSON), the MuJoCo model, and the machine-specific actuator parameters are supplied by the caller; the record fields are listed in `algorithm/__init__.py`. Default soil properties are defined in `SoilParams` in `soil.py`.

## Getting started

Use Python 3.12 on Linux (on Windows, use WSL: Pinocchio 3.8.0 cannot be installed from PyPI on Windows). The code uses NumPy, SciPy, Pillow, MuJoCo 3.3.7, and Pinocchio 3.8.0 (`pin` on PyPI, used by `pinocchio_model.py`). From the repository root:

```bash
python -m pip install numpy scipy pillow mujoco==3.3.7 pin==3.8.0 cmeel-urdfdom==4.0.1 cmeel-tinyxml2==10.0.0
python -m algorithm MODEL.json --output compiled.json
```

`cmeel-urdfdom` and `cmeel-tinyxml2` are pinned because a fresh install of Pinocchio 3.8.0 otherwise fails to import (`liburdfdom_sensor.so.4.0` not found).

`MODEL.json` is a caller-supplied physical graph. The command prints a compilation summary and, with `--output`, writes the compiled records and plan.

## Paper and citation

Please cite the accompanying manuscript when using this work:

Mehdi Heydari Shahna, Seihun Kim, Soyi Jung, Soohyun Park, Jouni Mattila, and Joongheon Kim. **Graph-Based Modeling and Simulation of a Hydraulic Excavator Using RoboCompiler.** 2026.

RoboCompiler: Mehdi Heydari Shahna, Joongheon Kim, and Jouni Mattila. **RoboCompiler: Graph-Native Compilation of Closed-Chain Robots for Consistent Modeling, Control, and Simulation.** 2026. https://doi.org/10.48550/arXiv.2609.35717

## Acknowledgment

This research was supported by Mechanical Equipment Industry Technology Development Program through the Korea Planning & Evaluation of Industrial Technology (KEIT) funded by the Korea government (Ministry of Trade, Industry and Energy (MOTIE)) (RS-2024-00442168).

## License

Released under the Apache License 2.0; see [LICENSE](LICENSE).
