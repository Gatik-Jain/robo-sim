"""ROBO-SIM -- interactive robotic arm simulation and motion planning.

Run locally with::

    streamlit run streamlit_app.py

This file is the *interface only*. Every piece of mathematics lives in the
``robo_sim`` package: kinematics, geometry, collision checking, path planning
and the experiment runner. That separation is what lets the same code be
exercised by ``python -m unittest`` without a browser.

All per-visitor state lives in ``st.session_state``, which Streamlit keeps
separate for each browser session. Nothing mutable is held at module level, so
one visitor's robot cannot affect another's.
"""

from __future__ import annotations

import math
from typing import Optional

import streamlit as st

from robo_sim import config
from robo_sim.collision import is_configuration_valid, validate_path
from robo_sim.experiments import (
    build_scenarios,
    results_to_csv,
    run_experiments,
    run_resolution_sweep,
)
from robo_sim.kinematics import (
    end_effector_position,
    forward_kinematics,
    inverse_kinematics,
    is_geometrically_reachable,
)
from robo_sim.models import (
    PLAN_OK,
    InvalidInputError,
    Rectangle,
    RobotSpec,
    describe_obstacles,
)
from robo_sim.path_planning import GridTooLargeError, PlanningProblem, plan
from robo_sim.visualization import (
    add_animation,
    build_robot_figure,
    grouped_bar_chart,
    resolution_chart,
    subsample,
    success_rate_chart,
)

st.set_page_config(
    page_title="ROBO-SIM",
    page_icon="🦾",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==========================================================================
# Session state
# ==========================================================================

DEFAULT_STATE = {
    "theta1": config.JOINT1_INITIAL,
    "theta2": config.JOINT2_INITIAL,
    "link1": config.LINK1_LENGTH,
    "link2": config.LINK2_LENGTH,
    "target_x": config.TARGET_X_INITIAL,
    "target_y": config.TARGET_Y_INITIAL,
    "tolerance": config.TARGET_TOLERANCE,
    "resolution": config.DEFAULT_RESOLUTION,
    "obstacles": [],
    "plan_result": None,
    "comparison": None,
    "experiment_results": None,
    "sweep_results": None,
    "frame": 0,
    "status": ("info", "Ready. Move the joints, set a target, then plan a path."),
    "ik_result": None,
    "show_workspace": True,
    "show_labels": False,
    "scenario_loaded": None,
}


def init_state() -> None:
    for key, value in DEFAULT_STATE.items():
        if key not in st.session_state:
            # Lists must be copied, or every session would share one list.
            st.session_state[key] = list(value) if isinstance(value, list) else value


# Settings whose value invalidates an existing plan when it changes. Each is
# the key of a *keyed widget*, so the widget and the stored value are the same
# thing and a programmatic reset survives the next rerun.
#
# Binding widgets to session-state keys matters for more than tidiness: a
# Streamlit widget without a key keeps its own internal value, which overrides
# the `value=` argument on every later rerun. Resetting the robot by assigning
# to session_state would then appear to do nothing, because the slider would
# immediately write its old value back.
TRACKED_SETTINGS = (
    ("theta1", "the joint angles changed"),
    ("theta2", "the joint angles changed"),
    ("link1", "the link lengths changed"),
    ("link2", "the link lengths changed"),
    ("target_x", "the target moved"),
    ("target_y", "the target moved"),
    ("resolution", "the planning resolution changed"),
    ("tolerance", "the target tolerance changed"),
)


def snapshot_settings() -> dict[str, object]:
    return {key: st.session_state[key] for key, _ in TRACKED_SETTINGS}


def detect_setting_changes() -> None:
    """Invalidate a stored plan when anything it depended on has changed.

    Called once per rerun, after the widgets have been drawn, by comparing the
    current values against the snapshot taken at the end of the previous run.
    """
    previous = st.session_state.get("_settings_snapshot")
    current = snapshot_settings()
    st.session_state["_settings_snapshot"] = current

    if previous is None:
        return

    for key, reason in TRACKED_SETTINGS:
        if previous[key] != current[key]:
            if key in ("target_x", "target_y"):
                st.session_state.ik_result = None
            invalidate_plan(reason)
            return


def queue_update(**values: object) -> None:
    """Schedule changes to widget-backed state for the next rerun.

    Streamlit forbids assigning to a widget's session-state key once that
    widget has been drawn in the current run. Inverse kinematics, the reset
    buttons and the example loader all need to move controls that live in the
    sidebar, which has already rendered by the time they run, so their changes
    are queued here and applied by :func:`apply_pending` at the very top of the
    next run -- before any widget exists.
    """
    st.session_state.setdefault("_pending", {}).update(values)


def apply_pending() -> None:
    pending = st.session_state.pop("_pending", None)
    if pending:
        for key, value in pending.items():
            st.session_state[key] = value


def set_status(kind: str, message: str) -> None:
    st.session_state.status = (kind, message)


def invalidate_plan(reason: str) -> None:
    """Discard a plan that no longer describes the current situation.

    Anything that changes the robot, the target, the obstacles, the tolerance
    or the resolution makes an existing path meaningless. Showing a stale path
    as though it were still valid would be the single most misleading thing
    this interface could do, so it is cleared immediately.
    """
    if st.session_state.plan_result is not None:
        st.session_state.plan_result = None
        st.session_state.frame = 0
        set_status("warning", f"Previous path discarded: {reason}.")
    st.session_state.comparison = None


def current_spec() -> RobotSpec:
    return RobotSpec(
        link1_length=st.session_state.link1,
        link2_length=st.session_state.link2,
    )


def current_pose() -> tuple[float, float]:
    return (st.session_state.theta1, st.session_state.theta2)


def current_target() -> tuple[float, float]:
    return (st.session_state.target_x, st.session_state.target_y)


def current_obstacles() -> list[Rectangle]:
    return list(st.session_state.obstacles)


# ==========================================================================
# Example scenarios
# ==========================================================================

EXAMPLE_SCENARIOS = {
    "— none —": None,
    "Obstacle beside the route": (
        [(230.0, 290.0, -120.0, -40.0)],
        (150.0, -200.0),
    ),
    "Detour required": ([(210.0, 280.0, -150.0, -60.0)], (150.0, -200.0)),
    "Three obstacles": (
        [
            (200.0, 250.0, -70.0, 20.0),
            (-80.0, 40.0, 180.0, 260.0),
            (-270.0, -170.0, -80.0, 40.0),
        ],
        (170.0, -160.0),
    ),
    "Sealed off (no path)": (
        [(200.0, 250.0, -70.0, 20.0), (175.0, 215.0, 25.0, 50.0)],
        (170.0, -160.0),
    ),
}


def load_example(name: str) -> None:
    entry = EXAMPLE_SCENARIOS.get(name)
    if entry is None:
        return
    rects, target = entry
    st.session_state.obstacles = [Rectangle(*r) for r in rects]
    st.session_state.scenario_loaded = name
    queue_update(
        target_x=float(target[0]),
        target_y=float(target[1]),
        theta1=config.JOINT1_INITIAL,
        theta2=config.JOINT2_INITIAL,
        link1=config.LINK1_LENGTH,
        link2=config.LINK2_LENGTH,
    )
    invalidate_plan("an example scenario was loaded")
    set_status("info", f"Loaded the example scenario “{name}”.")


# ==========================================================================
# Sidebar: robot, environment and planning settings
# ==========================================================================


def render_sidebar() -> None:
    sidebar = st.sidebar
    sidebar.markdown("### 🦾 Robot")

    sidebar.slider(
        "Joint 1 — shoulder (°)",
        min_value=float(config.JOINT1_MIN),
        max_value=float(config.JOINT1_MAX),
        step=1.0,
        key="theta1",
        help="Measured from the +X axis, counter-clockwise.",
    )
    sidebar.slider(
        "Joint 2 — elbow (°)",
        min_value=float(config.JOINT2_MIN),
        max_value=float(config.JOINT2_MAX),
        step=1.0,
        key="theta2",
        help="Measured relative to link 1, not to the X axis.",
    )

    col_a, col_b = sidebar.columns(2)
    # `value` is passed as well as `key` on every number_input in this app.
    # The key alone is not enough: st.number_input sends `min_value` to the
    # browser as its initial display value regardless of what session_state
    # holds, so the box would read 20 while the robot actually used 170.
    # Passing both keeps the displayed number and the stored number identical,
    # including after a programmatic reset. (st.slider does not have this
    # problem, which is why the sliders only need a key.)
    col_a.number_input(
        "Link 1 (u)",
        min_value=float(config.MIN_LINK_LENGTH),
        max_value=float(config.MAX_LINK_LENGTH),
        step=5.0,
        value=float(st.session_state.link1),
        key="link1",
    )
    col_b.number_input(
        "Link 2 (u)",
        min_value=float(config.MIN_LINK_LENGTH),
        max_value=float(config.MAX_LINK_LENGTH),
        step=5.0,
        value=float(st.session_state.link2),
        key="link2",
    )

    if sidebar.button("↺ Reset robot", width="stretch"):
        queue_update(
            theta1=config.JOINT1_INITIAL,
            theta2=config.JOINT2_INITIAL,
            link1=config.LINK1_LENGTH,
            link2=config.LINK2_LENGTH,
        )
        invalidate_plan("the robot was reset")
        set_status("info", "Robot restored to its default dimensions and pose.")
        st.rerun()

    sidebar.divider()
    sidebar.markdown("### ⚙️ Planning")

    sidebar.select_slider(
        "Grid resolution (°)",
        options=list(config.SUPPORTED_RESOLUTIONS),
        key="resolution",
        help=(
            "The joint-angle step of the search grid. A finer grid represents "
            "narrow gaps better and gives smoother paths, but the state count "
            "grows roughly with the square of the inverse step, so planning "
            "takes noticeably longer."
        ),
    )
    sidebar.slider(
        "Target tolerance (u)",
        min_value=float(config.MIN_TARGET_TOLERANCE),
        max_value=float(config.MAX_TARGET_TOLERANCE),
        step=1.0,
        key="tolerance",
        help=(
            "How close the tip must finish to the target for a grid pose to "
            "count as a goal. Too tight a value and no grid pose qualifies."
        ),
    )

    resolution = st.session_state.resolution
    grid1 = int(round((config.JOINT1_MAX - config.JOINT1_MIN) / resolution)) + 1
    grid2 = int(round((config.JOINT2_MAX - config.JOINT2_MIN) / resolution)) + 1
    sidebar.caption(
        f"Search graph: {grid1} × {grid2} = {grid1 * grid2:,} joint configurations."
    )

    sidebar.divider()
    sidebar.markdown("### 👁 View")
    st.session_state.show_workspace = sidebar.checkbox(
        "Reachable workspace", value=st.session_state.show_workspace
    )
    st.session_state.show_labels = sidebar.checkbox(
        "Joint labels", value=st.session_state.show_labels
    )

    sidebar.divider()
    sidebar.caption(
        "All lengths are abstract **simulation units (u)**, not millimetres. "
        "Only the ratios between them matter."
    )


# ==========================================================================
# Simulator tab
# ==========================================================================


def render_status_banner() -> None:
    kind, message = st.session_state.status
    {"success": st.success, "warning": st.warning, "error": st.error}.get(
        kind, st.info
    )(message)


def render_target_controls() -> None:
    st.markdown("#### 🎯 Target")
    spec = current_spec()

    # Widgets inside a form only write back to session_state when the form is
    # submitted, which is exactly the "set target deliberately" behaviour
    # wanted here: typing a coordinate does not move the target until the
    # button is pressed.
    with st.form("target_form"):
        col_a, col_b = st.columns(2)
        col_a.number_input(
            "Target X (u)",
            step=5.0,
            min_value=-float(config.OBSTACLE_COORD_LIMIT),
            max_value=float(config.OBSTACLE_COORD_LIMIT),
            value=float(st.session_state.target_x),
            key="target_x",
        )
        col_b.number_input(
            "Target Y (u)",
            step=5.0,
            min_value=-float(config.OBSTACLE_COORD_LIMIT),
            max_value=float(config.OBSTACLE_COORD_LIMIT),
            value=float(st.session_state.target_y),
            key="target_y",
        )
        if st.form_submit_button("Set target", width="stretch"):
            set_status(
                "info",
                f"Target updated to "
                f"({st.session_state.target_x:g}, {st.session_state.target_y:g}).",
            )

    target = current_target()
    distance = math.hypot(target[0] - spec.base_x, target[1] - spec.base_y)
    reachable = is_geometrically_reachable(spec, target)
    st.caption(
        f"Distance from base **{distance:.1f} u** · reachable band "
        f"{spec.min_reach:.0f}–{spec.max_reach:.0f} u · "
        + ("✅ geometrically reachable" if reachable else "❌ outside the workspace")
    )

    col_ik, col_plan = st.columns(2)
    if col_ik.button("Solve IK", width="stretch", type="secondary"):
        run_inverse_kinematics()
        st.rerun()
    if col_plan.button("Plan path", width="stretch", type="primary"):
        run_planner(st.session_state.get("algorithm_choice", "A*"))
        st.rerun()

    st.radio(
        "Algorithm",
        options=["A*", "Dijkstra"],
        horizontal=True,
        key="algorithm_choice",
    )

    render_ik_report()


def run_inverse_kinematics() -> None:
    spec = current_spec()
    try:
        result = inverse_kinematics(
            spec, current_target(), obstacles=current_obstacles(), current=current_pose()
        )
    except InvalidInputError as exc:
        st.session_state.ik_result = None
        set_status("error", f"Invalid target: {exc}")
        return

    st.session_state.ik_result = result

    if result.success:
        queue_update(
            theta1=round(result.chosen.theta1_deg, 2),
            theta2=round(result.chosen.theta2_deg, 2),
        )
        invalidate_plan("inverse kinematics moved the arm")
        set_status("success", f"Inverse kinematics solved. {result.message}")
    elif not result.geometrically_reachable:
        set_status("error", f"Target unreachable. {result.message}")
    elif result.limit_blocked:
        set_status("error", f"Target violates the joint limits. {result.message}")
    else:
        set_status("error", f"No collision-free solution. {result.message}")


def render_ik_report() -> None:
    result = st.session_state.ik_result
    if result is None:
        return

    with st.expander("Inverse-kinematics detail", expanded=True):
        st.write(
            f"**Target** ({result.target[0]:g}, {result.target[1]:g}) · "
            f"**Geometrically reachable:** {'yes' if result.geometrically_reachable else 'no'} · "
            f"**Analytical solutions:** {len(result.solutions)}"
        )
        if not result.solutions:
            st.caption(result.message)
            return

        rows = []
        for solution in result.solutions:
            if solution.collision_free is None:
                collision = "not checked"
            else:
                collision = "clear" if solution.collision_free else "collides"
            rows.append(
                {
                    "Elbow": solution.elbow,
                    "Joint 1 (°)": round(solution.theta1_deg, 2),
                    "Joint 2 (°)": round(solution.theta2_deg, 2),
                    "Within limits": "yes" if solution.within_joint_limits else "no",
                    "Obstacles": collision,
                    "Tip error (u)": round(solution.position_error, 4),
                    "Joint travel (°)": round(solution.joint_travel, 1),
                    "Selected": "◀" if solution is result.chosen else "",
                }
            )
        st.dataframe(rows, hide_index=True, width="stretch")

        if result.chosen is not None:
            error = result.chosen.position_error
            if error < 1e-6:
                st.caption(
                    f"The selected pose places the tip on the target to within "
                    f"{error:.2e} u — floating-point rounding only."
                )
            else:
                st.caption(f"Residual tip error: {error:.4f} u.")


def render_obstacle_controls() -> None:
    st.markdown("#### 🧱 Environment")

    example = st.selectbox(
        "Example scenario",
        options=list(EXAMPLE_SCENARIOS),
        index=0,
        help=(
            "Predefined layouts, provided for demonstration. Loading one "
            "replaces your own obstacles and target."
        ),
    )
    if example != "— none —" and st.button(
        f"Load “{example}”", width="stretch"
    ):
        load_example(example)
        st.rerun()

    with st.form("obstacle_form", clear_on_submit=False):
        st.caption("Add a rectangular obstacle")
        col_a, col_b = st.columns(2)
        min_x = col_a.number_input("Min X", value=200.0, step=10.0)
        max_x = col_b.number_input("Max X", value=250.0, step=10.0)
        min_y = col_a.number_input("Min Y", value=-70.0, step=10.0)
        max_y = col_b.number_input("Max Y", value=20.0, step=10.0)

        if st.form_submit_button("➕ Add obstacle", width="stretch"):
            add_obstacle(min_x, max_x, min_y, max_y)
            st.rerun()

    obstacles = current_obstacles()
    if not obstacles:
        st.caption("No obstacles. The arm may move freely within its joint limits.")
        return

    st.caption(f"{len(obstacles)} obstacle(s)")
    for index, rect in enumerate(obstacles):
        col_text, col_button = st.columns([5, 1])
        col_text.markdown(
            f"<div style='padding:4px 0;font-size:0.85rem'>"
            f"<code>{index + 1}</code> {rect.describe()}</div>",
            unsafe_allow_html=True,
        )
        if col_button.button("✕", key=f"remove_obstacle_{index}", help="Remove"):
            st.session_state.obstacles.pop(index)
            invalidate_plan("an obstacle was removed")
            set_status("info", f"Removed obstacle {index + 1}.")
            st.rerun()

    if st.button("🗑 Clear all obstacles", width="stretch"):
        st.session_state.obstacles = []
        invalidate_plan("all obstacles were cleared")
        set_status("info", "All obstacles removed.")
        st.rerun()

    if st.button("↺ Reset environment", width="stretch"):
        st.session_state.obstacles = []
        st.session_state.ik_result = None
        queue_update(
            target_x=config.TARGET_X_INITIAL, target_y=config.TARGET_Y_INITIAL
        )
        invalidate_plan("the environment was reset")
        set_status("info", "Environment reset: obstacles cleared, target restored.")
        st.rerun()


def add_obstacle(min_x: float, max_x: float, min_y: float, max_y: float) -> None:
    if len(st.session_state.obstacles) >= config.MAX_OBSTACLES:
        set_status(
            "error",
            f"At most {config.MAX_OBSTACLES} obstacles are allowed. Remove one first.",
        )
        return

    limit = config.OBSTACLE_COORD_LIMIT
    if any(abs(v) > limit for v in (min_x, max_x, min_y, max_y)):
        set_status("error", f"Obstacle coordinates must stay within ±{limit:g} u.")
        return

    try:
        rect = Rectangle(min_x, max_x, min_y, max_y)
    except InvalidInputError as exc:
        set_status("error", str(exc))
        return

    if rect.width < config.MIN_OBSTACLE_SIZE or rect.height < config.MIN_OBSTACLE_SIZE:
        set_status(
            "error",
            f"Obstacles must be at least {config.MIN_OBSTACLE_SIZE:g} u in each "
            f"direction.",
        )
        return

    st.session_state.obstacles.append(rect)
    invalidate_plan("an obstacle was added")
    set_status("success", f"Added an obstacle at {rect.describe()}.")


# ==========================================================================
# Planning
# ==========================================================================


def build_problem() -> PlanningProblem:
    return PlanningProblem(
        spec=current_spec(),
        obstacles=current_obstacles(),
        target=current_target(),
        resolution_deg=st.session_state.resolution,
        target_tolerance=st.session_state.tolerance,
    )


def run_planner(algorithm: str) -> None:
    """Run one planner and store the result, or report why it could not."""
    try:
        with st.spinner(f"Planning with {algorithm}…"):
            result = plan(algorithm, build_problem(), current_pose())
    except GridTooLargeError as exc:
        set_status("error", str(exc))
        return
    except (InvalidInputError, KeyError) as exc:
        set_status("error", f"Could not plan: {exc}")
        return

    st.session_state.plan_result = result
    st.session_state.frame = 0

    if result.path_found:
        set_status("success", result.message)
    else:
        set_status("error", result.message)


def run_comparison() -> None:
    """Run both planners on an identical problem and keep both results.

    Each planner gets its own freshly built problem, so neither inherits the
    other's warm collision cache. Without that the second planner would look
    much faster than it is.
    """
    results = {}
    try:
        with st.spinner("Running A* and Dijkstra on the same scenario…"):
            for algorithm in ("A*", "Dijkstra"):
                results[algorithm] = plan(algorithm, build_problem(), current_pose())
    except GridTooLargeError as exc:
        set_status("error", str(exc))
        return

    st.session_state.comparison = results
    found = [name for name, r in results.items() if r.path_found]
    if len(found) == 2:
        set_status("success", "Both planners completed. See the comparison tab.")
    elif found:
        set_status(
            "warning",
            f"Only {found[0]} found a path. The comparison shows both outcomes.",
        )
    else:
        set_status("error", "Neither planner found a path under these settings.")


# ==========================================================================
# Main visualization and playback
# ==========================================================================


def render_workspace() -> None:
    spec = current_spec()
    theta1, theta2 = current_pose()
    obstacles = current_obstacles()
    result = st.session_state.plan_result

    validity = is_configuration_valid(spec, theta1, theta2, obstacles)
    colliding = not validity.valid

    trajectory = None
    trail = None
    poses = None

    if result is not None and result.path_found:
        # Re-validate against the environment as it is *now*. The plan is
        # invalidated whenever anything changes, so this should always pass;
        # it is a cheap guard against ever showing an unsafe path as safe.
        recheck = validate_path(spec, result.path, obstacles)
        if not recheck.valid:
            st.session_state.plan_result = None
            set_status(
                "warning",
                f"The stored path is no longer valid and has been discarded. "
                f"{recheck.reason}",
            )
            result = None
        else:
            indices = subsample(result.path, config.MAX_ANIMATION_FRAMES)
            poses = [result.path[i] for i in indices]
            trajectory = [end_effector_position(spec, t1, t2) for t1, t2 in result.path]

    if poses:
        tip_positions = [end_effector_position(spec, t1, t2) for t1, t2 in poses]
        frame = min(st.session_state.frame, len(poses) - 1)
        theta1, theta2 = poses[frame]
        trail = tip_positions[: frame + 1]
        colliding = False

    figure = build_robot_figure(
        spec=spec,
        theta1=theta1,
        theta2=theta2,
        target=current_target(),
        obstacles=obstacles,
        trajectory=trajectory,
        trail=trail,
        show_workspace=st.session_state.show_workspace,
        colliding=colliding,
        show_joint_labels=st.session_state.show_labels,
    )

    if poses and len(poses) > 1:
        tip_positions = [end_effector_position(spec, t1, t2) for t1, t2 in poses]
        add_animation(figure, spec, poses, tip_positions)

    st.plotly_chart(
        figure,
        width="stretch",
        config={"displayModeBar": True, "scrollZoom": True},
    )

    if colliding:
        st.error(f"⚠️ Collision in the current configuration. {validity.reason}")

    render_readouts(spec, colliding)

    if poses and len(poses) > 1:
        render_playback_controls(poses)


def render_readouts(spec: RobotSpec, colliding: bool) -> None:
    theta1, theta2 = current_pose()
    geometry = forward_kinematics(spec, theta1, theta2)
    tip = geometry.end_effector
    target = current_target()
    error = math.hypot(tip[0] - target[0], tip[1] - target[1])

    columns = st.columns(5)
    columns[0].metric("Joint 1", f"{theta1:.1f}°")
    columns[1].metric("Joint 2", f"{theta2:.1f}°")
    columns[2].metric("Elbow", f"({geometry.joint1[0]:.1f}, {geometry.joint1[1]:.1f})")
    columns[3].metric("End effector", f"({tip[0]:.1f}, {tip[1]:.1f})")
    columns[4].metric(
        "Tip → target",
        f"{error:.1f} u",
        delta="within tolerance"
        if error <= st.session_state.tolerance
        else f"{error - st.session_state.tolerance:.1f} u over",
        delta_color="normal" if error <= st.session_state.tolerance else "inverse",
    )
    if colliding:
        st.caption(
            "These readouts describe the pose you have selected, which is "
            "currently in collision — it is shown in red and cannot be used as "
            "a planning start."
        )


def render_playback_controls(poses) -> None:
    st.markdown("##### ▶ Trajectory playback")
    st.caption(
        "Use the **Play** button and slider under the chart for browser-side "
        "animation, or scrub frames here. Both show the same validated path."
    )

    frame = st.slider(
        "Frame",
        min_value=0,
        max_value=len(poses) - 1,
        value=min(st.session_state.frame, len(poses) - 1),
        key="frame_slider",
        label_visibility="collapsed",
    )
    if frame != st.session_state.frame:
        st.session_state.frame = frame
        st.rerun()

    col_a, col_b, col_c = st.columns(3)
    if col_a.button("⏮ Reset to start", width="stretch"):
        st.session_state.frame = 0
        st.rerun()
    if col_b.button("⏭ Jump to end", width="stretch"):
        st.session_state.frame = len(poses) - 1
        set_status("success", "Animation complete — the arm is at the final pose.")
        st.rerun()
    if col_c.button("🗑 Clear trajectory", width="stretch"):
        st.session_state.plan_result = None
        st.session_state.frame = 0
        set_status("info", "Trajectory cleared.")
        st.rerun()

    pose = poses[min(st.session_state.frame, len(poses) - 1)]
    st.caption(
        f"Frame {st.session_state.frame + 1} of {len(poses)} · "
        f"J1 {pose[0]:.1f}°, J2 {pose[1]:.1f}°"
    )


def render_metrics_panel() -> None:
    result = st.session_state.plan_result
    st.markdown("#### 📊 Planning result")

    if result is None:
        st.caption("No current plan. Set a target and press **Plan path**.")
        return

    if not result.path_found:
        st.error(f"**{result.algorithm}** — no path. {result.message}")
        st.caption(
            "Metrics below reflect the search that was actually performed; "
            "path metrics are zero because no path exists to measure."
        )

    columns = st.columns(4)
    columns[0].metric("Algorithm", result.algorithm)
    columns[1].metric("Path found", "yes" if result.path_found else "no")
    columns[2].metric("Planning time", f"{result.planning_time_ms:.1f} ms")
    columns[3].metric("Expanded states", f"{result.expanded_states:,}")

    columns = st.columns(4)
    columns[0].metric("Discovered states", f"{result.discovered_states:,}")
    columns[1].metric("Configurations", f"{result.path_length:,}")
    columns[2].metric("Joint-space cost", f"{result.path_cost:.2f}°")
    columns[3].metric("Joint travel", f"{result.total_joint_travel:.1f}°")

    columns = st.columns(4)
    columns[0].metric("Tip path length", f"{result.ee_path_length:.1f} u")
    columns[1].metric("Final tip error", f"{result.final_target_error:.2f} u")
    columns[2].metric(
        "Collision check",
        {True: "passed", False: "FAILED", None: "not run"}[result.collision_validated],
    )
    columns[3].metric("Goal poses", f"{result.goal_states:,}")

    if result.snapped_start and result.snapped_start != current_pose():
        st.caption(
            f"ℹ️ The search grid does not contain the exact current pose "
            f"({current_pose()[0]:.1f}°, {current_pose()[1]:.1f}°). It was snapped "
            f"to ({result.snapped_start[0]:.1f}°, {result.snapped_start[1]:.1f}°), "
            f"and the move onto the grid was collision checked separately."
        )


# ==========================================================================
# Tabs
# ==========================================================================


def tab_simulator() -> None:
    render_status_banner()
    left, right = st.columns([2.1, 1], gap="medium")
    with left:
        render_workspace()
    with right:
        render_target_controls()
        st.divider()
        render_obstacle_controls()
    st.divider()
    render_metrics_panel()


def tab_comparison() -> None:
    st.markdown("### Algorithm comparison")
    st.markdown(
        "A\\* and Dijkstra are run on **exactly the same problem**: the same "
        "start pose, link lengths, joint limits, target, tolerance, obstacles, "
        "grid resolution and collision rules. The only difference is how each "
        "orders its search frontier."
    )

    if st.button("Run A* and Dijkstra on the current scenario", type="primary"):
        run_comparison()
        st.rerun()

    comparison = st.session_state.comparison
    if not comparison:
        st.info(
            "No comparison yet. Set up a scenario in the Simulator tab, then run "
            "the comparison here."
        )
        return

    astar = comparison["A*"]
    dijkstra = comparison["Dijkstra"]

    rows = [
        ("Path found", "yes" if astar.path_found else "no", "yes" if dijkstra.path_found else "no"),
        ("Planning time (ms)", f"{astar.planning_time_ms:.2f}", f"{dijkstra.planning_time_ms:.2f}"),
        ("Expanded states", f"{astar.expanded_states:,}", f"{dijkstra.expanded_states:,}"),
        ("Discovered states", f"{astar.discovered_states:,}", f"{dijkstra.discovered_states:,}"),
        ("Path configurations", f"{astar.path_length:,}", f"{dijkstra.path_length:,}"),
        ("Joint-space cost (°)", f"{astar.path_cost:.3f}", f"{dijkstra.path_cost:.3f}"),
        ("Total joint travel (°)", f"{astar.total_joint_travel:.2f}", f"{dijkstra.total_joint_travel:.2f}"),
        ("Tip path length (u)", f"{astar.ee_path_length:.2f}", f"{dijkstra.ee_path_length:.2f}"),
        ("Final tip error (u)", f"{astar.final_target_error:.3f}", f"{dijkstra.final_target_error:.3f}"),
        (
            "Collision validated",
            {True: "passed", False: "FAILED", None: "not run"}[astar.collision_validated],
            {True: "passed", False: "FAILED", None: "not run"}[dijkstra.collision_validated],
        ),
    ]
    st.dataframe(
        [{"Metric": m, "A*": a, "Dijkstra": d} for m, a, d in rows],
        hide_index=True,
        width="stretch",
    )

    if astar.path_found and dijkstra.path_found:
        chart = grouped_bar_chart(
            [
                _as_scenario_row("current", astar),
                _as_scenario_row("current", dijkstra),
            ],
            "expanded_states",
            "States expanded on this scenario",
            "Expanded states",
        )
        st.plotly_chart(chart, width="stretch")

        cost_gap = abs(astar.path_cost - dijkstra.path_cost)
        if cost_gap < 1e-6:
            st.success(
                f"Both planners returned a path of cost **{astar.path_cost:.3f}°**. "
                "They agree, which is the expected outcome: both are optimal on "
                "this graph, and A*'s heuristic is admissible so it does not "
                "trade optimality for speed."
            )
        else:
            st.warning(
                f"The costs differ by {cost_gap:.6f}°. With an admissible "
                "heuristic they should match; a gap this size is either "
                "floating-point noise or a genuine defect worth investigating."
            )

        faster = "A*" if astar.planning_time_ms < dijkstra.planning_time_ms else "Dijkstra"
        fewer = "A*" if astar.expanded_states < dijkstra.expanded_states else "Dijkstra"
        st.markdown(
            f"On **this** scenario {fewer} expanded fewer states "
            f"({min(astar.expanded_states, dijkstra.expanded_states):,} vs "
            f"{max(astar.expanded_states, dijkstra.expanded_states):,}) and "
            f"{faster} finished sooner in wall-clock terms. "
            "Neither is a general law. A\\* usually expands fewer states because "
            "its heuristic pulls the search towards the goal, but how much it "
            "helps depends on the obstacle layout, the grid resolution and how "
            "informative the heuristic is in that particular configuration space. "
            "Wall-clock time also varies between repeated runs on the same "
            "machine, so read it alongside the expanded-state counts rather than "
            "on its own."
        )
    else:
        st.warning(
            "At least one planner found no path, so the cost comparison is not "
            "meaningful. The outcome of each run is shown above rather than "
            "hidden."
        )
        for name, result in comparison.items():
            if not result.path_found:
                st.caption(f"**{name}** — `{result.status}`: {result.message}")


def _as_scenario_row(scenario_id: str, result):
    """Adapt a PlanResult to the small interface the chart helpers use."""

    class _Row:
        pass

    row = _Row()
    row.scenario_id = scenario_id
    row.algorithm = result.algorithm
    row.path_found = result.path_found
    row.expanded_states = result.expanded_states
    row.discovered_states = result.discovered_states
    row.planning_time_ms = result.planning_time_ms
    row.path_cost = result.path_cost
    row.ee_path_length = result.ee_path_length
    row.resolution_deg = result.resolution_deg
    return row


def tab_experiments() -> None:
    st.markdown("### Experiments")
    st.markdown(
        "Five deterministic scenarios, each run with both planners. Every number "
        "below is measured during the run — nothing is cached from a previous "
        "session or written in by hand. Failed runs stay in the dataset."
    )

    with st.expander("Scenario definitions", expanded=False):
        for scenario in build_scenarios():
            st.markdown(
                f"**{scenario.scenario_id}** — {scenario.description}  \n"
                f"Target ({scenario.target[0]:g}, {scenario.target[1]:g}) · "
                f"obstacles: `{describe_obstacles(scenario.obstacles)}`  \n"
                f"_{scenario.expectation}_"
            )

    col_a, col_b = st.columns(2)
    run_default = col_a.button(
        f"Run all scenarios at {config.DEFAULT_RESOLUTION:g}°",
        type="primary",
        width="stretch",
    )
    run_sweep = col_b.button(
        "Run resolution sweep (3°, 5°, 10°)", width="stretch"
    )

    if run_default:
        progress = st.progress(0.0, text="Starting…")
        results = run_experiments(
            progress=lambda done, total, label: progress.progress(
                done / total, text=f"{label} ({done}/{total})"
            )
        )
        progress.empty()
        st.session_state.experiment_results = results
        set_status("success", f"Experiment complete: {len(results)} runs.")

    if run_sweep:
        progress = st.progress(0.0, text="Starting…")
        st.caption(
            "The 3° grid has about 12,000 states, so this takes noticeably "
            "longer than the single-resolution run."
        )
        results = run_resolution_sweep(
            progress=lambda done, total, label: progress.progress(
                done / total, text=f"{label} ({done}/{total})"
            )
        )
        progress.empty()
        st.session_state.sweep_results = results
        set_status("success", f"Resolution sweep complete: {len(results)} runs.")

    results = st.session_state.experiment_results
    if results:
        st.divider()
        st.markdown("#### Results at a single resolution")
        render_results_table(results)
        st.download_button(
            "⬇ Download results as CSV",
            data=results_to_csv(results),
            file_name="robo_sim_experiment_results.csv",
            mime="text/csv",
            width="stretch",
        )

        st.plotly_chart(
            grouped_bar_chart(
                results,
                "planning_time_ms",
                "Planning time by scenario and algorithm",
                "Time (ms)",
                log_y=True,
            ),
            width="stretch",
        )
        st.caption(
            "Logarithmic scale. Planning time is hardware- and "
            "implementation-dependent: it reflects this machine, this Python "
            "build and this pure-Python collision checker, so treat the ratios "
            "as more meaningful than the absolute values."
        )
        st.plotly_chart(
            grouped_bar_chart(
                results, "expanded_states", "States expanded", "Expanded states"
            ),
            width="stretch",
        )
        st.plotly_chart(
            grouped_bar_chart(
                results, "path_cost", "Joint-space path cost", "Cost (degrees)"
            ),
            width="stretch",
        )
        st.plotly_chart(
            grouped_bar_chart(
                results,
                "ee_path_length",
                "End-effector path length",
                "Tip travel (units)",
            ),
            width="stretch",
        )
        st.plotly_chart(success_rate_chart(results), width="stretch")

    sweep = st.session_state.sweep_results
    if sweep:
        st.divider()
        st.markdown("#### Resolution sweep")
        st.plotly_chart(resolution_chart(sweep), width="stretch")
        st.plotly_chart(
            resolution_chart(
                sweep, "planning_time_ms", "Planning time, mean (ms)"
            ),
            width="stretch",
        )
        st.caption(
            "Averaged over the scenarios in which that algorithm found a path, "
            "with the count shown above each point. Scenarios that fail at one "
            "resolution and succeed at another change which runs are averaged, "
            "so the raw per-run data below is the authoritative record."
        )
        render_results_table(sweep)
        st.download_button(
            "⬇ Download sweep results as CSV",
            data=results_to_csv(sweep),
            file_name="robo_sim_resolution_sweep.csv",
            mime="text/csv",
            width="stretch",
        )

    if not results and not sweep:
        st.info("No experiment results yet. Run one of the batches above.")


def render_results_table(results) -> None:
    st.dataframe(
        [
            {
                "Scenario": r.scenario_id,
                "Algorithm": r.algorithm,
                "Res (°)": r.resolution_deg,
                "Found": "yes" if r.path_found else "no",
                "Status": r.status,
                "Time (ms)": round(r.planning_time_ms, 2),
                "Expanded": r.expanded_states,
                "Discovered": r.discovered_states,
                "Cost (°)": round(r.path_cost, 2),
                "Poses": r.path_configurations,
                "Tip path (u)": round(r.ee_path_length, 1),
                "Tip error (u)": round(r.final_target_error, 2),
                "Validated": {True: "yes", False: "NO", None: "—"}[
                    r.collision_validated
                ],
            }
            for r in results
        ],
        hide_index=True,
        width="stretch",
    )


def tab_how_it_works() -> None:
    st.markdown("### How it works")

    st.markdown("#### Forward kinematics")
    st.markdown(
        "Given the joint angles, where is the arm? Joint 1 is measured from the "
        "+X axis and joint 2 **relative to link 1**, so link 2 points along "
        "$\\theta_1 + \\theta_2$."
    )
    st.latex(
        r"""
        \begin{aligned}
        x_1 &= b_x + L_1\cos\theta_1 \\
        y_1 &= b_y + L_1\sin\theta_1 \\
        x_2 &= x_1 + L_2\cos(\theta_1 + \theta_2) \\
        y_2 &= y_1 + L_2\sin(\theta_1 + \theta_2)
        \end{aligned}
        """
    )

    st.markdown("#### Inverse kinematics")
    st.markdown(
        "The reverse question, solved analytically rather than iteratively. "
        "With $r$ the distance from the base to the target:"
    )
    st.latex(
        r"""
        \begin{aligned}
        \cos\theta_2 &= \frac{r^2 - L_1^2 - L_2^2}{2 L_1 L_2} \\
        \theta_2 &= \pm\arccos(\cos\theta_2) \\
        \theta_1 &= \operatorname{atan2}(y, x)
                  - \operatorname{atan2}\!\big(L_2\sin\theta_2,\;
                    L_1 + L_2\cos\theta_2\big)
        \end{aligned}
        """
    )
    st.markdown(
        "The $\\pm$ gives the two classic **elbow-up** and **elbow-down** poses. "
        "ROBO-SIM computes both, filters them against the joint limits and the "
        "obstacles, and prefers whichever moves the joints least."
    )

    st.markdown("#### Reachable workspace")
    st.markdown(
        "The tip can reach exactly the annulus between $|L_1 - L_2|$ and "
        "$L_1 + L_2$ from the base. Anything outside it is rejected outright — "
        "an unreachable target is never quietly clamped onto the boundary. "
        "Three questions are kept strictly separate, because they have "
        "different answers and different remedies:"
    )
    st.markdown(
        "1. Is the target **geometrically** reachable?\n"
        "2. Is it reachable **within the joint limits**?\n"
        "3. Is there a **collision-free pose and path** that gets there?"
    )

    st.markdown("#### Collision checking")
    st.markdown(
        "Each link is a **capsule**: every point within the link radius of its "
        "centre line. The gripper is a disc. Modelling thickness matters — a "
        "centre line can miss an obstacle by 5 units while a link of radius 7.5 "
        "clearly hits it. Distances to the rectangles are computed exactly, "
        "including near corners, so there is no conservative margin to reason "
        "about, and contact exactly at the boundary counts as a collision."
    )
    st.markdown(
        "A **transition** between two safe poses is not automatically safe: the "
        "arm can sweep through an obstacle in between. Every move is therefore "
        f"sampled at no coarser than {config.TRANSITION_STEP_DEG:g}° of joint "
        "rotation and checked at each sample. Endpoint-only checking is never "
        "used anywhere in this project."
    )

    st.markdown("#### Configuration-space planning")
    st.markdown(
        "Routing the tip around obstacles like a dot on a map would be wrong: "
        "rotating joint 1 sweeps the *whole* arm, so a clear-looking tip path "
        "may be impossible to execute. ROBO-SIM instead searches the space of "
        "**joint angles**. A state is a pair of grid indices; a state only "
        "enters the graph if the complete arm is collision free in that pose. "
        "Obstacle avoidance then falls out of an ordinary graph search, and any "
        "path found is executable by construction."
    )
    st.markdown(
        "Neighbours are 8-connected (each index may change by −1, 0 or +1, "
        "excluding no movement). The edge cost is the Euclidean change in joint "
        "angles, so a diagonal step costs $\\sqrt{2}$ times a straight one:"
    )
    st.latex(r"c = \sqrt{\Delta\theta_1^2 + \Delta\theta_2^2}")
    st.markdown(
        "The −180° and +180° states of joint 1 are deliberately **not** joined. "
        "They are the same physical direction, but a joint with those limits "
        "cannot pass between them, and wrapping would silently authorise "
        "motions the robot could not perform."
    )

    st.markdown("#### A\\* and Dijkstra")
    st.markdown(
        "**Dijkstra** orders its frontier by accumulated cost $g$ alone, "
        "growing a uniform-cost ball outwards from the start. **A\\*** orders by "
        "$f = g + h$, where $h$ is the straight-line distance in joint-angle "
        "space to the nearest goal pose:"
    )
    st.latex(
        r"h(n) = \min_{g \in G} \sqrt{(\theta_1^n - \theta_1^g)^2 "
        r"+ (\theta_2^n - \theta_2^g)^2}"
    )
    st.markdown(
        "This heuristic is **admissible** — any route to a goal is a chain of "
        "moves whose total Euclidean length is at least the straight-line "
        "distance — and **consistent**, since the minimum distance to the goal "
        "set changes by at most the cost of any single edge. It is measured in "
        "degrees, the same unit as the edge costs; mixing units there is the "
        "classic way to make A\\* quietly non-optimal. Both planners therefore "
        "return paths of equal cost, and the interesting difference between "
        "them is how many states each had to expand to get there."
    )
    st.markdown(
        "The goal is a **set**, not a single pose: a target is usually "
        "reachable both elbow-up and elbow-down, and the tolerance admits a "
        "neighbourhood around each."
    )

    st.markdown("#### Metric definitions")
    st.markdown(
        "- **Planning duration** — measured around the search with "
        "`time.perf_counter()`. Excludes building the problem.\n"
        "- **Expanded states** — states popped from the priority queue and "
        "processed. A state popped again after being finalised is skipped and "
        "not counted; both planners use this same rule.\n"
        "- **Discovered states** — distinct states ever given a cost estimate.\n"
        "- **Joint-space path cost** — the sum of the planner's own edge costs.\n"
        "- **Total joint travel** — $\\sum(|\\Delta\\theta_1| + |\\Delta\\theta_2|)$, "
        "a Manhattan sum, which is why it differs from the path cost.\n"
        "- **End-effector path length** — summed tip-to-tip distance along the "
        "path, in simulation units.\n"
        "- **Target error** — distance from the final tip position to the "
        "requested target."
    )

    st.markdown("#### Honest limitations")
    st.markdown(
        "- The search is **discrete**. Failing to find a path at 5° does not "
        "prove no continuous motion exists; it may simply mean the grid is too "
        "coarse to represent the gap.\n"
        "- The tip finishes **within a tolerance** of the target, not exactly "
        "on it. The residual error is always reported.\n"
        f"- Transition checking samples at {config.TRANSITION_STEP_DEG:g}°; an "
        "obstacle much thinner than the distance the arm covers between samples "
        "could in principle be stepped over.\n"
        "- This is **kinematics only** — no masses, forces, velocities, "
        "accelerations, motor limits, control error or sensor noise. A real "
        "robot would need all of those."
    )


# ==========================================================================
# Entry point
# ==========================================================================


def main() -> None:
    init_state()
    # Must run before any widget is created, since it writes widget-backed
    # keys that Streamlit locks once the widget exists.
    apply_pending()
    # Streamlit syncs keyed widget values into session_state before the script
    # body runs, so comparing here -- ahead of the widgets being drawn --
    # catches sidebar moves and target-form submissions alike.
    detect_setting_changes()

    st.markdown(
        """
        <div style="padding:2px 0 10px 0">
          <h1 style="margin:0;font-size:2.0rem;letter-spacing:-0.02em">
            🦾 ROBO-SIM
          </h1>
          <p style="margin:2px 0 0 0;opacity:0.75;font-size:0.98rem">
            Interactive two-link robotic arm simulation, inverse kinematics and
            collision-aware motion planning in configuration space.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "**How to use:** move the joints in the sidebar · set a target and press "
        "**Solve IK** to jump there, or **Plan path** to route around obstacles · "
        "add obstacles on the right · compare the two planners and run the "
        "benchmarks in the other tabs."
    )

    render_sidebar()

    simulator, comparison, experiments, how = st.tabs(
        ["Simulator", "Algorithm comparison", "Experiments", "How it works"]
    )
    with simulator:
        tab_simulator()
    with comparison:
        tab_comparison()
    with experiments:
        tab_experiments()
    with how:
        tab_how_it_works()


if __name__ == "__main__":
    main()
else:
    # Streamlit executes this module as a script rather than importing it, so
    # the guard above is enough. The call here covers `streamlit run`, which
    # sets __name__ to "__main__" anyway; keeping both is harmless and makes
    # the file importable for a smoke test.
    pass
