---
title: "nuPlan Dataset & Devkit — The Closed-Loop Planning Benchmark"
description: "A detailed guide to nuPlan and nuplan-devkit — the data format, scenario/simulation/metrics stack, and how the closed-loop planning score is computed."
---

> A detailed guide to nuPlan (Motional's large-scale ML planning benchmark) and `nuplan-devkit`: the data format, the scenario/simulation/metrics stack, and how the closed-loop score is computed. See the companion notebook [nuplan_dataset.ipynb](nuplan_dataset.ipynb) for runnable illustrations of the data hierarchy, scenario sampling, closed-loop rollout, and the score formula.

Most autonomous-driving datasets (nuScenes, Waymo Open, Argoverse) were built for **perception** or **short-horizon motion forecasting**: you get a few seconds of context and predict a few seconds ahead, scored **open-loop** against a logged trajectory. But a planner that scores well open-loop can still crash in reality, because open-loop evaluation never lets the planner's own mistakes compound — every step is re-anchored to the ground-truth log. nuPlan was the first benchmark to fix this: 1500 hours of real driving, a lightweight **closed-loop** simulator where the ego car actually follows the planner's output and the world reacts, and planning-specific metrics (collisions, drivable-area compliance, progress, comfort). This is the benchmark family that NAVSIM and the PDM planner grew out of, so understanding nuPlan is the key to understanding modern planning evaluation.

---

## Table of Contents
1. [What nuPlan Is (and Why Closed-Loop Matters)](#1-what-nuplan-is-and-why-closed-loop-matters)
2. [Dataset Composition](#2-dataset-composition)
3. [Data Format: SQLite Logs, Sensor Blobs, GPKG Maps](#3-data-format-sqlite-logs-sensor-blobs-gpkg-maps)
4. [The Devkit Architecture](#4-the-devkit-architecture)
5. [Scenarios and the ScenarioBuilder](#5-scenarios-and-the-scenariobuilder)
6. [The Simulation Framework](#6-the-simulation-framework)
7. [Metrics and the Closed-Loop Score](#7-metrics-and-the-closed-loop-score)
8. [The Training Framework](#8-the-training-framework)
9. [nuBoard: Visualization and Debugging](#9-nuboard-visualization-and-debugging)
10. [nuPlan vs. NAVSIM](#10-nuplan-vs-navsim)
11. [Common Pitfalls](#11-common-pitfalls)
12. [References](#12-references)

---

## 1. What nuPlan Is (and Why Closed-Loop Matters)

nuPlan is a **planning benchmark**: the task is to output a future ego trajectory (or control actions) given the current scene, map, and agent tracks. Unlike perception datasets, it de-emphasises raw sensors and instead ships **auto-labeled tracks** (3D boxes + states) so researchers can focus on the planning problem. Evaluation happens in three modes:

| Mode | What the ego does | What other agents do | What it tests |
|---|---|---|---|
| **Open-loop (OL)** | Trajectory compared to the log; ego is *not* actually moved | Replayed from log | Imitation accuracy (like L2 displacement) |
| **Closed-loop non-reactive (CL-NR)** | Ego is driven by the planner; state evolves | Replayed from log (they ignore the ego) | Does the planner's own trajectory stay safe over time? |
| **Closed-loop reactive (CL-R)** | Ego is driven by the planner | Background agents react via an **IDM** policy | Does the planner cope with agents that respond to it? |

### Key insight
> **Open-loop success is necessary but not sufficient.** In closed-loop, small errors accumulate: a slightly-off heading this step changes the state the planner sees next step. This is exactly the *distribution shift* that dooms naive imitation learning, and it is why a rule-based planner (PDM-Closed) was able to beat learned planners on the nuPlan closed-loop leaderboard — a finding that reshaped the field.

**Why this matters:** the entire value of nuPlan is that it exposes failure modes (compounding error, reactive-agent interaction) that open-loop L2 metrics on nuScenes simply cannot see.

---

## 2. Dataset Composition

| Property | Value |
|---|---|
| Total driving | ~1500 hours of human driving |
| Cities | Boston, Pittsburgh, Las Vegas, Singapore (4) |
| Scenario types | ~75 auto-labeled tags (e.g. `starting_left_turn`, `high_magnitude_speed`, `near_multiple_vehicles`, `stopping_at_traffic_light`) |
| Developer | Motional |
| Sensors (subset) | 8 cameras (`CAM_F0/B0/L0/L1/L2/R0/R1/R2`) + merged LiDAR point cloud |

Las Vegas contributes the largest share of the mileage (dense, complex urban driving — the Strip), which is why Vegas scenarios dominate the challenge splits.

### Splits

| Split | Purpose | Rough size |
|---|---|---|
| `nuplan_mini` | Smoke-testing / tutorials | tens of GB |
| `nuplan_trainval` | Training + validation | the bulk of the ~1500 h |
| `nuplan_test` | Held-out challenge evaluation | separate logs |
| sensor blobs | Optional camera `.jpg` + LiDAR `.pcd` | very large (TB-scale); **skip unless you need raw sensors** |

> **The tracks live in the `.db` files; the sensor blobs are separate and enormous.** For pure planning research you download only the logs + maps and never touch the sensor blobs — the same lightweight-download trick used for NAVSIM's `navmini`.

**Why this matters:** you can do meaningful planning work with a small fraction of the full download, because the planning signal (ego pose, agent tracks, map, traffic lights) is all in the compact SQLite logs.

---

## 3. Data Format: SQLite Logs, Sensor Blobs, GPKG Maps

nuPlan made a deliberate design change from nuScenes: instead of a web of JSON files, **each driving log is a single SQLite `.db` file**. This scales to 1500 hours and lets you query with SQL / SQLAlchemy.

### Key tables in a log `.db`

| Table | Contents |
|---|---|
| `lidar_pc` | One row per LiDAR keyframe: timestamp, ego token, pointer to the `.pcd` blob. The temporal backbone. |
| `ego_pose` | Ego global pose (x, y, z, qw..qz) and velocities/accelerations per timestamp |
| `lidar_box` | 3D bounding boxes (auto-labeled tracks): size, pose, velocity, linked to a `track` |
| `track` | Persistent object identity across frames + category |
| `traffic_light_status` | Per-lane-connector traffic-light state (green/red/unknown) over time |
| `scene` | A ~20 s segment of a log |
| `scenario_tag` | Which of the ~75 scenario types fire at a given time (this is how you query "all left turns") |
| `category` | Object class taxonomy (vehicle, pedestrian, bicycle, ...) |

```python
# The devkit wraps these tables; you rarely write raw SQL, but you can:
import sqlite3
con = sqlite3.connect("2021.07.16.20.45.29_veh-35_01095_01486.db")
cur = con.execute("SELECT type, COUNT(*) FROM scenario_tag GROUP BY type ORDER BY 2 DESC")
for scenario_type, n in cur.fetchall():
    print(scenario_type, n)
```

### Maps

Maps are **HD semantic maps**, one per city, shipped as `nuplan-maps-v1.0.json` plus a `.gpkg` (GeoPackage — a SQLite-based GIS format) per location. Layers include lanes, lane-connectors (intersection paths), roadblocks, crosswalks, stop lines, and drivable-area polygons. The devkit exposes them through an `AbstractMap` API with spatial queries:

```python
from nuplan.common.maps.abstract_map import SemanticMapLayer
# nearest lanes, objects within a radius, lane-graph successors, route roadblocks, ...
lanes = map_api.get_proximal_map_objects(point, radius=50.0,
                                         layers=[SemanticMapLayer.LANE])
```

**Why this matters:** the SQLite-per-log format is the practical reason nuPlan can be 1500 h and still queryable; the GPKG maps give planners the lane graph they need for route-relative progress metrics.

---

## 4. The Devkit Architecture

`nuplan-devkit` is the official Python SDK. It is organised into three layers, orchestrated by **Hydra** configs and (for training) **PyTorch Lightning**.

```
nuplan/
├── database/     # Load .db logs: NuPlanDB, tables, ORM-style access to boxes/poses/tracks
├── common/       # Shared primitives
│   ├── actor_state/   # EgoState, StateSE2 (x,y,heading), DynamicCarState, kinematics
│   ├── maps/          # AbstractMap, lane graph, GPKG loaders, SemanticMapLayer
│   └── geometry/      # transforms, interpolation, convert between frames
└── planning/     # The benchmark itself
    ├── scenario_builder/  # AbstractScenario + builders (turn a .db into query-able Scenarios)
    ├── simulation/        # Closed-loop engine: planners, controllers, observations, callbacks
    ├── metrics/           # Metric engine + the aggregator that produces the final score
    ├── training/          # Feature/target builders, caching, Lightning models
    ├── nuboard/           # Bokeh dashboard for visual + metric inspection
    └── script/            # Hydra entry points: run_simulation.py, run_training.py, run_nuboard.py
```

### Key insight
> **Everything is a registered, config-swappable component.** A planner, an ego controller, an observation model, a metric — each is an abstract base class with Hydra config. You run an experiment by composing YAML (`+simulation=closed_loop_reactive_agents planner=idm_planner`), not by editing code. This is the same Registry+Config philosophy as OpenMMLab's mmEngine.

**Why this matters:** you can benchmark your planner against every baseline, in every simulation mode, without touching the engine — you only implement one class (`AbstractPlanner`) and point a config at it.

---

## 5. Scenarios and the ScenarioBuilder

The atomic unit of evaluation is a **Scenario**: a short (typically ~15 s) slice of a log with a defined initial state, a duration, and a database handle for querying ground truth.

```python
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario

class AbstractScenario:
    def get_number_of_iterations(self) -> int: ...              # simulation steps
    def get_ego_state_at_iteration(self, i) -> EgoState: ...    # GT ego (for OL / init)
    def get_tracked_objects_at_iteration(self, i): ...          # other agents at step i
    def get_traffic_light_status_at_iteration(self, i): ...
    def get_expert_ego_trajectory(self): ...                    # the human "expert" route
    def get_route_roadblock_ids(self): ...                      # the mission route on the map
    @property
    def map_api(self): ...
```

You obtain scenarios through a **ScenarioBuilder + ScenarioFilter**:

```python
# Conceptually: "give me up to 500 left-turn scenarios from the val logs"
scenario_filter = ScenarioFilter(
    scenario_types=["starting_left_turn"],
    num_scenarios_per_type=500,
    ...
)
builder = NuPlanScenarioBuilder(data_root, map_root, ...)
scenarios = builder.get_scenarios(scenario_filter, worker)
```

| Concept | Role |
|---|---|
| `scenario_type` | One of ~75 tags; lets you evaluate on curated difficult situations |
| `ScenarioFilter` | Selects/subsamples scenarios (by type, token, log, count) |
| expert trajectory | The human's actual path — the reference for the **progress** metric and OL scoring |
| route roadblock ids | The intended mission route; progress is measured *along this route*, not straight-line |

**Why this matters:** because scenarios are typed, nuPlan reports scores **per scenario type** — you learn not just "how good" but "good at what" (e.g. strong on lane-follow, weak on unprotected left turns).

---

## 6. The Simulation Framework

Closed-loop simulation is a discrete-time loop (typically 10 Hz). Each step:

1. Build the current `PlannerInput` (ego state, observations, map, route, history buffer).
2. Call `planner.compute_trajectory()` → a proposed trajectory.
3. An **ego controller** tracks that trajectory to produce the *next* ego state.
4. An **observation** model advances the other agents.
5. Metrics accumulate; repeat until the scenario ends or a terminal condition fires.

```python
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner

class MyPlanner(AbstractPlanner):
    def initialize(self, initialization) -> None: ...          # route, map, mission
    def compute_trajectory(self, current_input):               # <-- your algorithm
        # current_input.history: past ego states + observations
        # current_input.traffic_light_data, ...
        return InterpolatedTrajectory(states)                  # future ego states
```

### The swappable pieces

| Component | Purpose | Common choices |
|---|---|---|
| **Planner** | Produces the trajectory | `SimplePlanner` (constant velocity), `IDMPlanner` (rule-based car-following), `MLPlanner` (wraps a trained model), community `PDMPlanner` |
| **Ego controller** | Turns a trajectory into the next ego state | `PerfectTrackingController` (ego = planned state) or a two-stage **LQR + kinematic bicycle** controller (realistic dynamics) |
| **Observation** | Evolves background agents | `TracksObservation` (replay log = non-reactive) or `IDMAgents` (reactive) |
| **Simulation manager** | Time stepping / termination | fixed-horizon stepper |

### Key insight
> **Non-reactive vs. reactive is just the observation model swap.** CL-NR uses `TracksObservation` (agents replay the log and drive *through* you if you deviate); CL-R uses `IDMAgents` (a rule-based intelligent-driver-model that brakes/follows in response to the ego). Same engine, one config line different.

**Why this matters:** the ego controller is what makes closed-loop *hard* — with a realistic LQR+bicycle controller, your planned trajectory is only approximately executed, so a jittery or dynamically-infeasible plan degrades over the horizon.

---

## 7. Metrics and the Closed-Loop Score

This is the heart of nuPlan and the direct ancestor of NAVSIM's PDMS. The closed-loop score for one scenario is a **hierarchical, gated** aggregation: a set of **multiplier** (gate) metrics that can zero out the score, times a **weighted average** of quality metrics.

$$
\text{Score} = \Big(\prod_{m \in \text{multipliers}} m\Big) \times \frac{\sum_{w} \text{weight}_w \cdot \text{metric}_w}{\sum_{w} \text{weight}_w}
$$

### Multiplier (gate) metrics — each in $[0,1]$, applied multiplicatively

| Metric | Meaning | Effect |
|---|---|---|
| **No at-fault Collisions (NC)** | Ego did not cause a collision | Severe collision → 0 (score killed) |
| **Drivable Area Compliance (DAC)** | Ego stayed on drivable surface | Off-road → 0 |
| **Driving Direction Compliance (DDC)** | Ego did not drive against traffic | Large backward distance → 0, moderate → 0.5 |
| **Making Progress (MP)** | Ego progressed above a minimum threshold | Stuck / no progress → 0 |

### Weighted-average (quality) metrics

| Metric | Weight | Meaning |
|---|---|---|
| **Ego Progress along expert route (EP)** | 5 | Ratio of ego progress to the expert's, measured *along the route* |
| **Time to Collision (TTC)** | 5 | Fraction of time TTC stayed above a safe bound |
| **Speed-limit Compliance (SC)** | 4 | Penalised by magnitude/duration of speed-limit violations |
| **Comfort (C)** | 2 | Within bounds on longitudinal/lateral accel, jerk, yaw rate/accel |

```python
# The scoring shape, in miniature:
multipliers = [nc, dac, ddc, mp]           # each in [0, 1]
weighted = (5*ep + 5*ttc + 4*sc + 2*comfort) / (5 + 5 + 4 + 2)
score = np.prod(multipliers) * weighted    # one scenario's closed-loop score
final = np.mean([score_per_scenario ...])  # aggregate over all scenarios
```

### Key insight
> **The gate design encodes a safety hierarchy.** No amount of comfort or progress can rescue a run that hits someone or drives off-road — those multipliers send the whole product to 0. Only *among safe, legal, progressing* runs do the weighted quality metrics differentiate planners. This is exactly the structure NAVSIM's PDMS/EPDMS inherits.

**Why this matters:** it explains counterintuitive leaderboard results — a conservative planner that never violates a gate but makes modest progress can outrank an aggressive learned planner that occasionally collides, because collisions zero the score.

---

## 8. The Training Framework

For learned planners, `nuplan.planning.training` provides a Lightning-based pipeline built around two abstractions:

| Abstraction | Role |
|---|---|
| **FeatureBuilder** | Turns a raw `Scenario` into model input tensors — e.g. `RasterFeatureBuilder` (BEV multi-channel image) or a vector/agent-graph builder (VectorMap, agents), à la VectorNet/UrbanDriver |
| **TargetBuilder** | Turns a `Scenario` into supervision — e.g. the future ego trajectory |

```
Scenario ──FeatureBuilder──► features ─┐
                                        ├─► LightningModule ─► loss ─► planner
Scenario ──TargetBuilder───► targets ──┘
```

- **Feature caching:** builders are run once and cached to disk (`run_training.py cache=...`), so training epochs read tensors instead of re-querying `.db` files — essential at 1500 h scale.
- **Baseline models:** a raster CNN planner and an `UrbanDriver`-style vectorised open-loop model ship with the devkit; a trained model is wrapped by `MLPlanner` to plug straight into the simulator.
- **The open-/closed-loop gap:** models trained purely by imitation (open-loop L2) routinely score poorly closed-loop — motivating post-hoc trajectory selection, rule-based fallbacks, or hybrid planners like PDM.

**Why this matters:** the same `Scenario` object feeds both training and simulation, so there is no train/eval data-format skew — but the *metric* skew (L2 loss vs. gated closed-loop score) is real and is the central research tension.

---

## 9. nuBoard: Visualization and Debugging

`run_nuboard.py` launches a **Bokeh** dashboard that reads the pickled simulation outputs and metric files. It gives you:

- **Scenario rendering:** BEV playback of ego + agents + map + the planned vs. driven trajectory per time step.
- **Metric breakdown:** per-scenario and per-scenario-type score tables, so you can sort by worst cases.
- **Comparison:** overlay multiple experiments (e.g. your planner vs. IDM baseline).

**Why this matters:** closed-loop failures are temporal and interactive — a number tells you the DAC gate fired, but nuBoard shows you the ego clipping a curb at second 7. It is the primary debugging surface.

---

## 10. nuPlan vs. NAVSIM

NAVSIM is built **on top of `nuplan-devkit`** and reuses its data classes, maps, and the PDM-score machinery — so the two are best understood together.

| Aspect | nuPlan | NAVSIM |
|---|---|---|
| Data source | Own 1500 h logs (4 cities) | Re-annotated **OpenScene** (a nuPlan resample) + sensors |
| Simulation | Full closed-loop over ~15 s | **One-shot** pseudo-closed-loop: score a single trajectory via non-reactive PDM simulation |
| Cost | Heavy (real rollouts, reactive agents) | Light (no per-step re-planning loop) |
| Metric | Closed-loop score (NC/DAC/DDC/MP × EP/TTC/SC/C) | **PDMS / EPDMS** — same metric family, extended (e.g. lane-keeping, extended comfort, history penalties) |
| Focus | Planning from tracks | End-to-end sensor→trajectory |

### Key insight
> **NAVSIM's PDMS is nuPlan's closed-loop score, decoupled from the expensive rollout.** NAVSIM freezes the world (non-reactive), simulates the ego's single proposed trajectory with the PDM controller, and scores it with the same gated formula. If you understand nuPlan's score, you understand PDMS — the weights and gates carry over almost verbatim.

**Why this matters:** the reason to learn nuPlan even if you work in NAVSIM is that NAVSIM inherits nuPlan's coordinate frames, `EgoState`/map APIs, and scoring philosophy — nuPlan is the substrate.

---

## 11. Common Pitfalls

- **Confusing open-loop skill with closed-loop skill.** A low L2 displacement does *not* imply a high closed-loop score; the ranking often inverts. Report the mode explicitly (OL / CL-NR / CL-R).
- **Downloading the sensor blobs when you don't need them.** For track-based planning you need only logs + maps. The camera/LiDAR blobs are TB-scale and usually unnecessary.
- **Ignoring the ego controller.** With the LQR+bicycle controller the ego does *not* perfectly follow your plan; dynamically infeasible or jittery trajectories degrade over the horizon and tank the comfort/progress metrics.
- **Forgetting the score is gated.** Optimising average displacement or comfort is pointless if a collision or off-road event zeroes the multiplier. Fix gate violations first.
- **Measuring progress as straight-line distance.** Progress (EP) is measured **along the route roadblocks**, relative to the expert — you must respect `get_route_roadblock_ids()`, not Euclidean distance to a goal.
- **Non-reactive surprises.** In CL-NR, logged agents ignore you and may "drive through" the ego if you deviate from the human path — an at-fault-collision attribution nuance to keep in mind.
- **Re-querying `.db` files every epoch.** Always cache features; SQLite queries per sample are far too slow at dataset scale.
- **Map/version mismatches.** The `.db` logs, `nuplan-maps-v1.0`, and devkit version must line up; a stale map root produces silent lane-graph errors.

---

## 12. References

- Caesar et al., "nuPlan: A closed-loop ML-based planning benchmark for autonomous vehicles," CVPR 2021 ADP3 Workshop. arXiv: https://arxiv.org/abs/2106.11810
- nuPlan devkit (official): https://github.com/motional/nuplan-devkit
- nuPlan devkit docs: https://nuplan-devkit.readthedocs.io/
- Motional announcement: https://motional.com/news/nuplan-closed-loop-ml-based-planning-benchmark-autonomous-vehicles
- Dauner et al., "Parting with Misconceptions about Learning-based Vehicle Motion Planning" (PDM), CoRL 2023. arXiv: https://arxiv.org/abs/2306.07962
- Dauner et al., "NAVSIM: Data-Driven Non-Reactive Autonomous Vehicle Simulation and Benchmarking," NeurIPS 2024. arXiv: https://arxiv.org/abs/2406.15349
- Companion NAVSIM material in this repo: [driving_benchmarks.md](../driving_benchmarks/driving_benchmarks.md), [navsim_hands_on.md](../driving_benchmarks/navsim_hands_on.md)
