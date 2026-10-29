# 🦾 ROBO-SIM — Interactive Robotic Arm Simulation and Motion Planning

A browser-based simulator for a two-link planar robotic arm. You can move its
joints, set a target, drop obstacles into the workspace, solve the inverse
kinematics analytically, plan a collision-free path through the arm's
**configuration space**, and compare A\* against Dijkstra on identical problems.

Everything is computed in Python from first principles — forward and inverse
kinematics, exact planar geometry, capsule-based collision detection and both
graph-search planners are written out in this repository rather than imported
from a robotics library.

---

## Project overview

The interesting problem here is not "where does the arm's tip go?" but **"how
does the whole arm get there without hitting anything?"**

The naive approach — route the end effector around the obstacles like a dot on
a map — is wrong for a robot arm. The tip is not the only part that can
collide. Rotating the shoulder joint sweeps the *entire* arm through a large
region of the workspace, so a tip trajectory that looks perfectly clear may
only be executable by dragging the upper link straight through an obstacle.

ROBO-SIM therefore plans in **joint space**: it searches over pairs of joint
angles rather than over tip positions. A state enters the search graph only if
the complete arm — both thick links and the gripper — is collision free in that
pose. Obstacle avoidance then falls out of an ordinary graph search, and any
path found is executable by construction.

---

## Features

Everything listed here is implemented and covered by the test suite.

**Simulation and mathematics**

- Two-link planar arm with a fixed base, configurable link lengths, two
  revolute joints and an end effector.
- Forward kinematics with live numeric readouts of the elbow and tip positions.
- Analytical inverse kinematics returning both elbow-up and elbow-down
  solutions, filtered against joint limits and obstacles.
- Explicit separation of three distinct questions: geometric reachability,
  joint-limit feasibility, and collision-free reachability.
- Reachable-workspace annulus drawn on the chart.

**Obstacles and collisions**

- Add, list, remove and clear rectangular obstacles through the interface.
- Links modelled as **capsules** (segment + radius), the gripper as a disc, so
  physical thickness is respected rather than ignored.
- Exact segment-to-rectangle distances, including the corner cases — no
  conservative approximation anywhere.
- Boundary contact counts as a collision.
- **Transition validation**: every motion is sampled at 1° of joint rotation
  and checked along its whole length, because two safe poses can be separated
  by unsafe ones.

**Planning**

- Configuration-space grid at 3°, 5° or 10° resolution.
- 8-connected neighbours with Euclidean joint-space edge costs.
- A\* with a documented admissible and consistent heuristic.
- Dijkstra implemented independently, sharing the same graph, costs, goal set
  and collision rules so the comparison is fair.
- Multiple goal states (a target is normally reachable two ways).
- Distinct, honest failure reporting for each different cause of failure.

**Interface**

- Four tabs: Simulator, Algorithm comparison, Experiments, How it works.
- Browser-side Plotly animation with Play/Pause and a frame slider, plus a
  server-side frame scrubber.
- Stale plans are invalidated the moment anything they depended on changes.
- Full metrics panel after each plan.

**Experiments**

- Five deterministic benchmark scenarios.
- Command-line runner independent of Streamlit.
- CSV download in the browser; CSV file output locally.
- Charts for planning time, expanded states, path cost, tip path length,
  success rate, and performance versus resolution.

---

## Live application

**Not yet deployed.** No public URL exists for this project at the time of
writing, and none is invented here. Once you have deployed it by following the
[deployment guide](#deployment-guide) below, replace this section with the real
URL that Streamlit Community Cloud gives you.

---

## Technology stack

| Technology | Role |
|---|---|
| **Python 3.13** | Development and test runtime (3.11–3.13 should all work). |
| **Streamlit 1.66.1** | Web application framework: layout, widgets, session state. |
| **Plotly 7.1.0** | Interactive arm visualization, animation frames, charts. |
| **Python standard library** | `math` for trigonometry, `heapq` for the priority queues, `csv` for export, `time.perf_counter` for timing, `dataclasses` for the models, `argparse` for the CLI, `unittest` for the tests. |

That is the complete dependency list: two packages. There is no NumPy, no
Pandas, no SciPy and no JavaScript build step. The arrays here are small enough
that plain Python lists and dictionaries are faster than the overhead of
crossing into NumPy, and writing the mathematics out explicitly makes it
readable, which is the point of the project.

---

## Local installation

macOS, from a terminal:

```bash
git clone YOUR_REPOSITORY_URL
cd robo-sim

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`YOUR_REPOSITORY_URL` and the directory name are placeholders until you have
actually created the repository.

### Running locally

```bash
streamlit run streamlit_app.py
```

Streamlit prints a local URL (normally `http://localhost:8501`) and opens it in
your browser.

### Running tests

```bash
python -m unittest discover -s tests -v
```

### Compilation check

```bash
python -m compileall -q .
```

### Running experiments

```bash
python -m robo_sim.experiments
```

This writes `results/experiment_results.csv` and prints a summary table. Useful
options:

```bash
python -m robo_sim.experiments --output results/my_run.csv
python -m robo_sim.experiments --resolutions 3 5 10
python -m robo_sim.experiments --algorithms "A*" --no-write
```

---

## Mathematical background

### Coordinate convention

+X points right, +Y points up — ordinary mathematical axes, used identically in
the computational core and in the Plotly figures, with no screen-coordinate
flip anywhere. θ₁ is measured from the +X axis; **θ₂ is measured relative to
link 1**, so link 2 points along θ₁ + θ₂. Angles are degrees in every public
interface and converted to radians only inside the trigonometry.

### Forward kinematics

Given the joint angles, where is the arm?

```
x₁ = bₓ + L₁·cos(θ₁)
y₁ = b_y + L₁·sin(θ₁)

x₂ = x₁ + L₂·cos(θ₁ + θ₂)
y₂ = y₁ + L₂·sin(θ₁ + θ₂)
```

`(x₁, y₁)` is the elbow and `(x₂, y₂)` the end effector.

### Inverse kinematics

The reverse question, solved analytically rather than by iteration. With `r`
the distance from the base to the target and `r² = x² + y²`:

```
cos(θ₂) = (r² − L₁² − L₂²) / (2·L₁·L₂)

θ₂ = ±arccos(cos θ₂)

θ₁ = atan2(y, x) − atan2( L₂·sin(θ₂),  L₁ + L₂·cos(θ₂) )
```

The `±` gives the two classic **elbow-up** and **elbow-down** configurations.
ROBO-SIM computes both, filters them against the joint limits and the
obstacles, and prefers whichever requires the least joint movement from the
current pose.

The `arccos` argument is clamped to `[−1, 1]` **only** when it is outside by
less than `1e-9`, which is attributable to floating-point rounding. A larger
excursion means the target is genuinely unreachable and is reported as such.
An impossible target is never silently clamped onto the workspace boundary.

### Reachable workspace

The tip can reach exactly the annulus between `|L₁ − L₂|` and `L₁ + L₂`,
centred on the base. With the defaults (170 and 130) that is radii 40 to 300.
Inside radius 40 is a dead zone the arm physically cannot fold into.

Three questions are kept strictly separate throughout, because they have
different answers and different remedies:

1. Is the target **geometrically** reachable?
2. Is it reachable **within the configured joint limits**?
3. Is there a **collision-free pose and path** that reaches it?

### Joint constraints

| Parameter | Default |
|---|---:|
| Base | (0, 0) |
| Link 1 length | 170 u |
| Link 2 length | 130 u |
| Joint 1 range | −180° … +180° |
| Joint 2 range | −150° … +150° |
| Joint 1 initial | 35° |
| Joint 2 initial | −60° |
| Link collision radius | 7.5 u |
| End-effector radius | 9.0 u |
| Planning resolution | 5° |
| Target tolerance | 12 u |

All lengths are abstract **simulation units (u)**, not millimetres. Only the
ratios between them matter.

### Obstacle collision checking

Each link is a **capsule**: the set of points within `link_radius` of its
centre line, which is exactly the shape of a physical link of uniform
thickness. The gripper is a disc of its own radius at the tip.

Modelling thickness is not cosmetic. A centre line can miss an obstacle by 5
units while a link of radius 7.5 clearly collides with it — the test suite
checks precisely that case. Drawing thick links while collision-checking
zero-width lines would make the simulator confidently wrong.

Distances to the rectangles are computed **exactly**, via
`segment_to_rect_distance`. When a segment and a rectangle are disjoint they
are two convex sets, so the minimum distance is attained either at a segment
endpoint or at a rectangle corner; both families are checked. That handles the
corner regions exactly, so there is no conservative approximation whose
behaviour near corners would need separate justification. Contact exactly at
the collision radius is treated as a collision.

### Configuration-space path planning

A state is a pair of joint-angle grid indices `(i, j)` mapping to angles
`joint1_min + i·resolution` and `joint2_min + j·resolution`. A state enters the
graph only if the whole arm is valid there — within joint limits and clear of
every obstacle.

---

## Algorithms

### The graph

- **Nodes** — grid states, lazily validated and cached. At 5° there are
  73 × 61 = 4,453 of them.
- **Edges** — 8-connected: each index may change by −1, 0 or +1, excluding the
  no-move case. An edge exists only if both endpoints are valid **and** the
  interpolated sweep between them is collision free.
- **Edge cost** — the Euclidean change in joint angles, in degrees:

  ```
  c = √(Δθ₁² + Δθ₂²)
  ```

  So a straight step costs `resolution` and a diagonal step costs
  `resolution·√2`, which prices diagonal shortcuts correctly. Costs are
  computed from the real grid angles, never from index differences.

The −180° and +180° states of joint 1 are deliberately **not** connected.
They are the same physical direction, but a joint with those configured limits
cannot pass between them, and wrapping would silently authorise motions the
robot could not perform.

### Collision checking between neighbouring configurations

Validating only the endpoints of a move is unsafe: the arm can be clear before
a move and clear after it while sweeping straight through an obstacle in
between. Every edge is therefore interpolated — both joints simultaneously and
linearly, exactly as the animation moves them — and sampled at no coarser than
**1°** of joint rotation, with the complete arm geometry checked at each
sample. Since the finest planning grid is 3°, every planned edge is checked at
least three times internally.

The same `collision.py` functions are used by the manual joint controls,
inverse kinematics, the planners, the pre-animation validation and the unit
tests, so these components cannot disagree about what counts as a collision.

### Dijkstra

Orders its frontier by accumulated cost `g` alone, growing a uniform-cost ball
outwards from the start until it reaches a goal state.

### A\*

Orders by `f = g + h`, where `h` is the straight-line distance in joint-angle
space to the nearest goal pose:

```
h(n) = min over goals g of  √( (θ₁ⁿ − θ₁ᵍ)² + (θ₂ⁿ − θ₂ᵍ)² )
```

**Admissible** — any route from `n` to a goal is a chain of moves, and by the
triangle inequality the sum of their Euclidean lengths is at least the
straight-line distance, so `h` can never overestimate the true remaining cost.

**Consistent** — for any edge `(n, n')` of cost `c`, the minimum distance to the
goal set changes by at most `c`, so `h(n) ≤ c + h(n')`. A consistent heuristic
means the first goal popped is optimal and no state ever needs reopening.

Crucially, `h` is measured in **degrees**, the same unit as the edge costs.
Mixing units there — for example using the tip's distance in simulation units
as a heuristic for a cost in degrees — is the classic way to make A\* quietly
non-optimal.

Because both planners are optimal on the same graph, they return paths of
**equal cost**. The interesting difference between them is how many states each
had to expand to get there.

### Goal states

The goal is a **set**, not a single pose. A target is normally reachable both
elbow-up and elbow-down, and the tolerance admits a neighbourhood around each.
Goal generation scans the grid, applies the cheap tip-position test first and
runs the expensive full-body collision test only on the states that pass it.

If no grid pose puts the tip within tolerance, that is reported as *no discrete
goal state at this resolution* — explicitly **not** as "the target is
impossible" — with a suggestion to try a finer grid or a larger tolerance.

### Honest failure reporting

"No path" has several very different causes, and conflating them would mislead
the user. Each gets its own status code and message:

| Status | Meaning |
|---|---|
| `start_invalid` | The arm is already in collision, or outside its joint limits. |
| `start_snap_failed` | The current pose is valid but the nearest grid pose, or the move onto it, is not. |
| `target_unreachable` | The target is outside the geometric workspace annulus. |
| `no_goal_state` | No collision-free grid pose puts the tip within tolerance. |
| `no_path` | The search exhausted the reachable component without reaching a goal. |
| `grid_too_large` | The requested resolution would exceed the state-count safety limit. |

A `no_path` result never claims that no continuous collision-free motion
exists. A discretised search failing is evidence about *this grid*, not a proof
about the continuous problem.

---

## Experimental methodology

### Scenarios

Five deterministic scenarios, defined in `robo_sim/experiments.py`. Each one's
expected behaviour was established by running the planner and observing the
outcome, not by assumption; the reasoning is recorded in each scenario's
`expectation` field and shown in the Experiments tab.

| ID | Description | Observed outcome |
|---|---|---|
| `S1_free_space` | Reachable target, no obstacles | Path found |
| `S2_obstacle_near_route` | One obstacle beside the direct route | Path found, higher cost than S1 |
| `S3_three_obstacles` | Three obstacles around the workspace | Path found |
| `S4_detour_required` | Direct route blocked, detour available | Path found, higher cost than S2 |
| `S5_no_discrete_path` | Goal poses exist but are sealed off | No path (by construction) |

S1, S2 and S4 share a target, which isolates the effect of the obstacle: the
cost rises as the obstacle moves onto the route. S5 is S3's first obstacle plus
a second one that seals the only gap the arm used, so its failure is a
consequence of the geometry rather than an arbitrary assertion.

### Determinism

No randomness is used anywhere: fixed rectangles, fixed angles, a regular grid,
and heap ties broken by an insertion counter. The structural metrics — expanded
states, discovered states, path cost, path length — reproduce **exactly** from
run to run.

Wall-clock `planning_time_ms` is the one exception. It depends on the machine,
the Python build and what else the computer is doing, so it should be read
alongside the expanded-state counts rather than on its own. Planning time is
hardware- and implementation-dependent, and this is an unoptimised pure-Python
collision checker.

### Metric definitions

| Metric | Definition |
|---|---|
| **Planning duration** | Time around the search loop via `time.perf_counter()`, in milliseconds. Excludes building the problem. |
| **Expanded states** | States popped from the priority queue and processed. A state popped again after being finalised is skipped and not counted. Both planners use this identical rule. |
| **Discovered states** | Distinct states ever assigned a cost estimate. |
| **Joint-space path cost** | Sum of the planner's own edge costs, in degrees. |
| **Total joint travel** | `Σ(|Δθ₁| + |Δθ₂|)` — a Manhattan sum, which is why it differs from the path cost. |
| **End-effector path length** | Summed tip-to-tip distance along the path, in simulation units. |
| **Final target error** | Distance from the final tip position to the requested target. |
| **Collision validated** | Whether the returned path passed independent full-path validation. Blank in the CSV when no path existed to validate — which is not the same as a failed validation. |

Failed runs stay in the dataset. A success rate computed over only the
successes would be meaningless.

### Reproducing

```bash
python -m robo_sim.experiments --resolutions 3 5 10 --output results/sweep.csv
```

Or press **Run all scenarios** / **Run resolution sweep** in the Experiments
tab and use the CSV download button. The hosted app's filesystem is temporary,
so the web interface offers downloads rather than writing files server-side.

---

## Folder structure

```text
robo-sim/
├── streamlit_app.py          Web interface only: layout, widgets, session state
├── requirements.txt          Two pinned dependencies
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml           Dark engineering theme
├── robo_sim/                 The computational core — no Streamlit imports
│   ├── __init__.py
│   ├── config.py             Defaults, limits, palette
│   ├── models.py             Immutable dataclasses and status codes
│   ├── kinematics.py         Forward and inverse kinematics
│   ├── geometry.py           Exact planar geometry primitives
│   ├── collision.py          Configuration and transition validity
│   ├── path_planning.py      Joint-space graph, A*, Dijkstra, metrics
│   ├── experiments.py        Scenarios, runner, CSV, CLI entry point
│   └── visualization.py      Plotly figure builders
├── tests/
│   ├── test_kinematics.py
│   ├── test_geometry.py
│   ├── test_collision.py
│   ├── test_path_planning.py
│   ├── test_experiments.py
│   └── test_app.py           Streamlit AppTest interaction tests
└── results/
    └── .gitkeep
```

The `robo_sim` package never imports Streamlit — a test asserts this. That is
what lets the mathematics be run, scripted and tested with no browser involved,
and it is why `streamlit_app.py` contains interface code only.

---

## Deployment guide

ROBO-SIM is built for [Streamlit Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy),
which is free and needs no credit card. The app requires no secrets, no
database, no external APIs and no user accounts.

1. **Create a GitHub repository.** On [github.com/new](https://github.com/new),
   make a new repository (for example `robo-sim`) and set it to **Public** so
   the source code can be read by anyone.
2. **Push this project to it:**
   ```bash
   git init
   git add .
   git commit -m "Add ROBO-SIM robotic arm simulator"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/robo-sim.git
   git push -u origin main
   ```
   If the directory already has history and a remote, check with
   `git remote -v` that it is the repository you intend before pushing, and do
   not force-push over unrelated work.
3. **Sign in to [share.streamlit.io](https://share.streamlit.io)** with your
   GitHub account and authorise access to the repository.
4. **Click "Create app" → "Deploy a public app from GitHub".**
5. Select your **repository** and the **`main`** branch.
6. Set the **main file path** to `streamlit_app.py`.
7. Open **Advanced settings** and set the **Python version** to **3.13**
   (3.11 or 3.12 also work; the pinned dependencies were tested on 3.13).
   Leave the secrets box empty — this app needs none.
8. **Click Deploy.** The first build installs the two dependencies and takes a
   couple of minutes.
9. **Open the generated public URL** (`https://YOUR-APP-NAME.streamlit.app`).
10. **Test it in a normal browser** — move the joints, solve IK, add an
    obstacle, plan a path, play the animation, run the experiments and download
    the CSV.
11. **To update the hosted app**, push to the same branch; Streamlit Cloud
    rebuilds automatically.

Each visitor gets an independent `st.session_state`, so one person's robot,
obstacles and experiments never affect another's. No mutable state is held at
module level.

---

## Verification

Being precise about what was and was not checked:

**Automated, actually executed** — the full suite of **207 tests** passes:

```
python -m unittest discover -s tests -v
...
Ran 207 tests in ~37s
OK
```

- `test_kinematics.py` — forward kinematics at known poses, FK→IK round trips,
  both elbow solutions, workspace boundaries, joint-limit and
  obstacle-blocked targets, non-finite inputs.
- `test_geometry.py` — point/segment/rectangle distances, intersections,
  capsule and disc overlap, boundary contact, corner cases.
- `test_collision.py` — collision-free poses, each link crossing an obstacle,
  the thickness case a centre line would miss, end-effector-only collisions,
  exact boundary contact, and transitions that are valid at both ends but
  sweep through an obstacle in between.
- `test_path_planning.py` — grid construction, 8-connectivity, the deliberate
  −180°/+180° non-connection, edge costs, goal sets, heuristic admissibility
  and consistency checked against real optimal costs, both planners, agreement
  of their path costs, and every failure status code.
- `test_experiments.py` — scenario definitions, measured output, failure
  records, CSV headers and round-tripping, and the CLI.
- `test_app.py` — the Streamlit interface driven through `AppTest`: startup,
  joint control, resets, IK success and failure, obstacle management,
  planning, plan invalidation, the algorithm comparison, and the CSV download.

`python -m compileall -q .` also passes cleanly.

**Manually checked in a local browser** — the app was started with
`streamlit run streamlit_app.py` and loaded at `http://localhost:8501`.

**Not verified** — the application has **never been deployed**, so nothing
about its behaviour on Streamlit Community Cloud has been confirmed. The
Plotly animation's Play/Pause buttons are generated by Plotly's standard
`updatemenus` mechanism and are not exercised by the automated tests, which
cannot drive browser-side JavaScript; the server-side frame scrubber is tested
and works as a documented fallback. Cross-browser rendering in Safari, Firefox
and Chrome has not been systematically checked.

---

## Limitations

These are real, and worth understanding before drawing conclusions from the
simulator.

- **Discretisation.** The planner searches a grid. Failing to find a path at 5°
  does **not** prove that no continuous collision-free motion exists — the grid
  may simply be too coarse to represent a narrow gap. A finer resolution
  sometimes succeeds where a coarser one fails.
- **Target tolerance.** The tip finishes *within a tolerance* of the target,
  not exactly on it, because the goal must be a grid pose. The residual error
  is always reported and never described as zero when it is not.
- **Transition sampling.** Moves are checked every 1° of joint rotation. An
  obstacle much thinner than the distance the arm covers between samples could
  in principle be stepped over. With a 300 unit arm the tip moves about 5 units
  per sample, so obstacles should be comfortably larger than that.
- **Kinematics only.** There are no masses, forces, velocities, accelerations,
  motor torque limits, control error, backlash or sensor noise. A real
  manipulator needs all of those, and a kinematically valid path is only the
  starting point for a real trajectory.
- **Two links, one plane.** Real manipulators typically have six or seven
  degrees of freedom in three dimensions. The configuration space here is
  two-dimensional, which is exactly what makes it possible to visualise and
  reason about.
- **Axis-aligned rectangular obstacles only.** No rotated boxes, polygons or
  curves.
- **Static world.** Obstacles do not move during planning or playback.
- **Performance.** Collision checking is pure Python. At 3° resolution a
  difficult scenario can take several seconds.

---

## Future improvements

None of these are implemented; they are directions the project could take.

- **Dynamic obstacles** and replanning while the arm is moving.
- **A three-link arm**, which makes the configuration space three-dimensional
  and the visualisation much more interesting.
- **Trajectory smoothing** — shortcutting the grid-aligned path and then
  re-validating, to remove the staircase artefacts of an 8-connected grid.
- **Sampling-based planners** such as RRT or PRM, for comparison against the
  grid search in higher dimensions.
- **Velocity and acceleration limits**, turning a geometric path into a
  time-parameterised trajectory.
- **Rotated and polygonal obstacles**, using a general convex-distance routine.
- **3D simulation** with a full six-degree-of-freedom manipulator.

---

## Where to start reading

To understand the mathematics and the algorithms, read the source in this
order:

1. **`robo_sim/kinematics.py`** — the forward and inverse kinematics. Start
   here; everything else builds on it.
2. **`robo_sim/geometry.py`** — the exact planar-geometry primitives, in
   particular `segment_to_rect_distance`.
3. **`robo_sim/collision.py`** — how poses and, importantly, *transitions*
   between poses are validated.
4. **`robo_sim/path_planning.py`** — the configuration-space graph, the
   heuristic and both planners. The module docstring explains why planning
   happens in joint space.
5. **`robo_sim/experiments.py`** — how the benchmarks are defined and measured.

The matching test files are good companion reading: each one states, as an
executable assertion, what the module is supposed to do.

---

## License

Released under the [MIT License](LICENSE) — © 2026 Gatik Jain.

You may use, copy, modify and distribute this code, including commercially,
provided the copyright notice and licence text are kept. The software is
provided without warranty.
