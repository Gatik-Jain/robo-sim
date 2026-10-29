"""Plotly figure builders for the arm, the environment and the charts.

This module draws; it never computes geometry. Every arm position comes from
:mod:`robo_sim.kinematics`, so what is on screen is exactly what the collision
checker and the planner reasoned about.

Plotly uses ordinary Cartesian axes, so the mathematical convention (+X right,
+Y up) is kept throughout -- there is no screen-coordinate flip anywhere.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import plotly.graph_objects as go

from . import config
from .kinematics import forward_kinematics
from .models import Rectangle, RobotSpec, ScenarioResult

Pose = tuple[float, float]

# Fixed trace order. The animation replaces traces by index, so this order is
# part of the contract between `build_robot_figure` and `add_animation`.
TRACE_WORKSPACE = 0
TRACE_TRAJECTORY = 1
TRACE_TRAIL = 2
TRACE_LINK1 = 3
TRACE_LINK2 = 4
TRACE_JOINTS = 5
TRACE_END_EFFECTOR = 6
TRACE_BASE = 7
TRACE_TARGET = 8

ANIMATED_TRACES = [TRACE_TRAIL, TRACE_LINK1, TRACE_LINK2, TRACE_JOINTS, TRACE_END_EFFECTOR]


def _axis_range(spec: RobotSpec, obstacles: Sequence[Rectangle]) -> tuple[float, float]:
    """A square view that contains the whole workspace and every obstacle."""
    reach = spec.max_reach * 1.15
    lo = min(spec.base_x, spec.base_y) - reach
    hi = max(spec.base_x, spec.base_y) + reach
    for rect in obstacles:
        lo = min(lo, rect.min_x, rect.min_y)
        hi = max(hi, rect.max_x, rect.max_y)
    margin = (hi - lo) * 0.04
    return (lo - margin, hi + margin)


def _arm_traces(
    spec: RobotSpec, theta1: float, theta2: float, colliding: bool
) -> dict[str, dict]:
    """The x/y data for the five traces that move during an animation."""
    geometry = forward_kinematics(spec, theta1, theta2)
    base, joint1, tip = geometry.base, geometry.joint1, geometry.end_effector
    link_colour = config.COLOR_COLLISION if colliding else None

    return {
        "link1": {
            "x": [base[0], joint1[0]],
            "y": [base[1], joint1[1]],
            "line": {"color": link_colour or config.COLOR_LINK1},
        },
        "link2": {
            "x": [joint1[0], tip[0]],
            "y": [joint1[1], tip[1]],
            "line": {"color": link_colour or config.COLOR_LINK2},
        },
        "joints": {"x": [base[0], joint1[0]], "y": [base[1], joint1[1]]},
        "tip": {"x": [tip[0]], "y": [tip[1]]},
    }


def _obstacle_shapes(obstacles: Sequence[Rectangle]) -> list[dict]:
    """Obstacles as Plotly shapes so they stay put during animation."""
    return [
        {
            "type": "rect",
            "x0": rect.min_x,
            "x1": rect.max_x,
            "y0": rect.min_y,
            "y1": rect.max_y,
            "fillcolor": config.COLOR_OBSTACLE,
            "opacity": 0.55,
            "line": {"color": config.COLOR_OBSTACLE_LINE, "width": 1.5},
            "layer": "below",
        }
        for rect in obstacles
    ]


def _workspace_trace(spec: RobotSpec, show: bool) -> go.Scatter:
    """The annulus the tip can reach, drawn as two circles.

    A single trace with a ``None`` break keeps the outer and inner boundaries
    together so the whole thing toggles as one legend entry.
    """
    xs: list[Optional[float]] = []
    ys: list[Optional[float]] = []
    if show:
        for radius in (spec.max_reach, spec.min_reach):
            if radius <= 0:
                continue
            for step in range(181):
                angle = math.radians(step * 2.0)
                xs.append(spec.base_x + radius * math.cos(angle))
                ys.append(spec.base_y + radius * math.sin(angle))
            xs.append(None)
            ys.append(None)

    return go.Scatter(
        x=xs,
        y=ys,
        mode="lines",
        name="Reachable workspace",
        line=dict(color=config.COLOR_WORKSPACE, width=1, dash="dot"),
        hoverinfo="skip",
        showlegend=show,
    )


def build_robot_figure(
    spec: RobotSpec,
    theta1: float,
    theta2: float,
    target: Optional[tuple[float, float]] = None,
    obstacles: Sequence[Rectangle] = (),
    trajectory: Optional[Sequence[tuple[float, float]]] = None,
    trail: Optional[Sequence[tuple[float, float]]] = None,
    show_workspace: bool = True,
    colliding: bool = False,
    show_joint_labels: bool = False,
    height: int = 620,
    title: Optional[str] = None,
) -> go.Figure:
    """The main simulation view.

    ``trajectory`` is the planned tip path, drawn distinctly from the arm's
    current links. ``trail`` is the portion of that path already travelled
    during playback.
    """
    arm = _arm_traces(spec, theta1, theta2, colliding)
    geometry = forward_kinematics(spec, theta1, theta2)

    traj_x = [p[0] for p in (trajectory or [])]
    traj_y = [p[1] for p in (trajectory or [])]
    trail_x = [p[0] for p in (trail or [])]
    trail_y = [p[1] for p in (trail or [])]

    figure = go.Figure()

    # Order here must match the TRACE_* constants.
    figure.add_trace(_workspace_trace(spec, show_workspace))

    figure.add_trace(
        go.Scatter(
            x=traj_x,
            y=traj_y,
            mode="lines",
            name="Planned trajectory",
            line=dict(color=config.COLOR_TRAJECTORY, width=2, dash="dash"),
            hovertemplate="trajectory (%{x:.1f}, %{y:.1f})<extra></extra>",
            showlegend=bool(traj_x),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=trail_x,
            y=trail_y,
            mode="lines",
            name="Travelled",
            line=dict(color=config.COLOR_END_EFFECTOR, width=3),
            opacity=0.75,
            hoverinfo="skip",
            showlegend=bool(trail_x),
        )
    )
    figure.add_trace(
        go.Scatter(
            **arm["link1"],
            mode="lines",
            name="Link 1",
            line_width=2 * spec.link_radius,
            hovertemplate="link 1<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            **arm["link2"],
            mode="lines",
            name="Link 2",
            line_width=2 * spec.link_radius,
            hovertemplate="link 2<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            **arm["joints"],
            mode="markers",
            name="Joints",
            marker=dict(
                size=13,
                color=config.COLOR_JOINT,
                line=dict(color=config.COLOR_BACKGROUND, width=2),
            ),
            hovertemplate="joint (%{x:.1f}, %{y:.1f})<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            **arm["tip"],
            mode="markers",
            name="End effector",
            marker=dict(
                size=2 * spec.end_effector_radius,
                color=config.COLOR_END_EFFECTOR,
                line=dict(color=config.COLOR_BACKGROUND, width=2),
            ),
            hovertemplate="tip (%{x:.1f}, %{y:.1f})<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[spec.base_x],
            y=[spec.base_y],
            mode="markers",
            name="Base",
            marker=dict(size=20, color=config.COLOR_BASE, symbol="square"),
            hovertemplate="base (%{x:.1f}, %{y:.1f})<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[target[0]] if target else [],
            y=[target[1]] if target else [],
            mode="markers",
            name="Target",
            marker=dict(
                size=16,
                color=config.COLOR_TARGET,
                symbol="x-thin",
                line=dict(color=config.COLOR_TARGET, width=3),
            ),
            hovertemplate="target (%{x:.1f}, %{y:.1f})<extra></extra>",
            showlegend=target is not None,
        )
    )

    if show_joint_labels:
        for label, (px, py) in (
            ("J1", geometry.joint1),
            ("TCP", geometry.end_effector),
        ):
            figure.add_annotation(
                x=px,
                y=py,
                text=label,
                showarrow=False,
                yshift=20,
                font=dict(color=config.COLOR_TEXT, size=11),
            )

    axis_range = _axis_range(spec, obstacles)
    axis_style = dict(
        range=list(axis_range),
        zeroline=True,
        zerolinecolor=config.COLOR_AXIS,
        zerolinewidth=1.5,
        gridcolor=config.COLOR_GRID,
        griddash="dot",
        showline=True,
        linecolor=config.COLOR_AXIS,
        color=config.COLOR_TEXT,
    )

    figure.update_layout(
        title=title,
        shapes=_obstacle_shapes(obstacles),
        xaxis=dict(title="X (simulation units)", **axis_style),
        # Equal scaling, so a circle looks like a circle and a square obstacle
        # is not stretched into a rectangle.
        yaxis=dict(
            title="Y (simulation units)",
            scaleanchor="x",
            scaleratio=1,
            **axis_style,
        ),
        height=height,
        margin=dict(l=10, r=10, t=40 if title else 16, b=10),
        paper_bgcolor=config.COLOR_PANEL,
        plot_bgcolor=config.COLOR_BACKGROUND,
        font=dict(color=config.COLOR_TEXT, size=12),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            x=0,
            bgcolor="rgba(0,0,0,0)",
        ),
        hovermode="closest",
        dragmode="pan",
    )
    return figure


def add_animation(
    figure: go.Figure,
    spec: RobotSpec,
    poses: Sequence[Pose],
    tip_positions: Sequence[tuple[float, float]],
    frame_ms: int = 60,
) -> go.Figure:
    """Attach Plotly animation frames with native Play/Pause and a slider.

    Playback runs entirely in the browser: Plotly swaps pre-computed frames, so
    the Python process is not involved and nothing blocks. There is no
    ``time.sleep`` anywhere in ROBO-SIM.

    ``poses`` must already have been collision validated; this function only
    draws. For long paths the caller may sub-sample the frames for display,
    which is safe because validation covers the full path at a finer step.
    """
    frames = []
    for index, (theta1, theta2) in enumerate(poses):
        arm = _arm_traces(spec, theta1, theta2, colliding=False)
        travelled = tip_positions[: index + 1]
        frames.append(
            go.Frame(
                name=str(index),
                data=[
                    go.Scatter(
                        x=[p[0] for p in travelled], y=[p[1] for p in travelled]
                    ),
                    go.Scatter(**arm["link1"]),
                    go.Scatter(**arm["link2"]),
                    go.Scatter(**arm["joints"]),
                    go.Scatter(**arm["tip"]),
                ],
                traces=ANIMATED_TRACES,
            )
        )

    figure.frames = frames

    play_args = [
        None,
        {
            "frame": {"duration": frame_ms, "redraw": False},
            "fromcurrent": True,
            "transition": {"duration": 0},
            "mode": "immediate",
        },
    ]
    pause_args = [
        [None],
        {
            "frame": {"duration": 0, "redraw": False},
            "mode": "immediate",
            "transition": {"duration": 0},
        },
    ]

    figure.update_layout(
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "showactive": False,
                "x": 0.0,
                "y": -0.08,
                "xanchor": "left",
                "yanchor": "top",
                "pad": {"t": 4, "r": 8},
                "bgcolor": config.COLOR_PANEL,
                "bordercolor": config.COLOR_AXIS,
                "font": {"color": config.COLOR_TEXT},
                "buttons": [
                    {"label": "▶ Play", "method": "animate", "args": play_args},
                    {"label": "❚❚ Pause", "method": "animate", "args": pause_args},
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "x": 0.14,
                "len": 0.86,
                "y": -0.08,
                "xanchor": "left",
                "yanchor": "top",
                "pad": {"t": 4},
                "currentvalue": {
                    "prefix": "Frame ",
                    "font": {"color": config.COLOR_TEXT, "size": 12},
                },
                "font": {"color": config.COLOR_TEXT, "size": 10},
                "bgcolor": config.COLOR_GRID,
                "activebgcolor": config.COLOR_END_EFFECTOR,
                "bordercolor": config.COLOR_AXIS,
                "steps": [
                    {
                        "label": str(index),
                        "method": "animate",
                        "args": [
                            [str(index)],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    }
                    for index in range(len(poses))
                ],
            }
        ],
        margin=dict(l=10, r=10, t=16, b=70),
    )
    return figure


def subsample(poses: Sequence[Pose], limit: int) -> list[int]:
    """Indices of at most ``limit`` evenly spaced poses, keeping both ends.

    Used only to keep animation payloads small. The *validated* path is always
    the full one.
    """
    count = len(poses)
    if count <= limit:
        return list(range(count))
    step = (count - 1) / (limit - 1)
    indices = sorted({int(round(i * step)) for i in range(limit)})
    if indices[-1] != count - 1:
        indices.append(count - 1)
    return indices


# --------------------------------------------------------------------------
# Experiment charts
# --------------------------------------------------------------------------

ALGORITHM_COLORS = {"A*": config.COLOR_LINK1, "Dijkstra": config.COLOR_END_EFFECTOR}


def _chart_layout(figure: go.Figure, title: str, y_title: str, height: int) -> go.Figure:
    figure.update_layout(
        title=title,
        barmode="group",
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor=config.COLOR_PANEL,
        plot_bgcolor=config.COLOR_BACKGROUND,
        font=dict(color=config.COLOR_TEXT, size=12),
        xaxis=dict(gridcolor=config.COLOR_GRID, color=config.COLOR_TEXT),
        yaxis=dict(
            title=y_title, gridcolor=config.COLOR_GRID, color=config.COLOR_TEXT
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
    )
    return figure


def grouped_bar_chart(
    results: Sequence[ScenarioResult],
    value_field: str,
    title: str,
    y_title: str,
    log_y: bool = False,
    height: int = 360,
) -> go.Figure:
    """One bar per scenario, grouped by algorithm, from measured values.

    Runs that found no path are drawn at zero and labelled "no path", so a
    failure is visible rather than silently missing from the chart.
    """
    scenarios = sorted({r.scenario_id for r in results})
    figure = go.Figure()

    for algorithm in sorted({r.algorithm for r in results}):
        rows = {r.scenario_id: r for r in results if r.algorithm == algorithm}
        values, labels = [], []
        for scenario_id in scenarios:
            row = rows.get(scenario_id)
            if row is None:
                values.append(0.0)
                labels.append("")
                continue
            value = getattr(row, value_field)
            values.append(value)
            labels.append("no path" if not row.path_found else f"{value:,.6g}")

        figure.add_trace(
            go.Bar(
                name=algorithm,
                x=scenarios,
                y=values,
                text=labels,
                textposition="outside",
                marker_color=ALGORITHM_COLORS.get(algorithm),
                hovertemplate=f"{algorithm}<br>%{{x}}<br>{y_title}: %{{y:.4g}}<extra></extra>",
            )
        )

    if log_y:
        figure.update_yaxes(type="log")
    return _chart_layout(figure, title, y_title, height)


def success_rate_chart(
    results: Sequence[ScenarioResult], height: int = 320
) -> go.Figure:
    """Share of the selected scenarios in which each algorithm found a path."""
    figure = go.Figure()
    algorithms = sorted({r.algorithm for r in results})
    rates, labels = [], []
    for algorithm in algorithms:
        rows = [r for r in results if r.algorithm == algorithm]
        found = sum(1 for r in rows if r.path_found)
        rates.append(100.0 * found / len(rows) if rows else 0.0)
        labels.append(f"{found}/{len(rows)}")

    figure.add_trace(
        go.Bar(
            x=algorithms,
            y=rates,
            text=labels,
            textposition="outside",
            marker_color=[ALGORITHM_COLORS.get(a) for a in algorithms],
            hovertemplate="%{x}<br>%{y:.0f}% of scenarios<extra></extra>",
        )
    )
    figure.update_yaxes(range=[0, 109])
    return _chart_layout(
        figure,
        "Success rate across the selected scenarios",
        "Scenarios with a path found (%)",
        height,
    )


def resolution_chart(
    results: Sequence[ScenarioResult],
    value_field: str = "expanded_states",
    y_title: str = "Expanded states (mean)",
    height: int = 360,
) -> go.Figure:
    """Mean metric against planning resolution, one line per algorithm.

    Only successful runs are averaged, and the point label records how many
    runs went into each average, so a mean over fewer scenarios is visible.
    """
    figure = go.Figure()
    resolutions = sorted({r.resolution_deg for r in results})

    for algorithm in sorted({r.algorithm for r in results}):
        xs, ys, labels = [], [], []
        for resolution in resolutions:
            rows = [
                r
                for r in results
                if r.algorithm == algorithm
                and r.resolution_deg == resolution
                and r.path_found
            ]
            if not rows:
                continue
            xs.append(resolution)
            ys.append(sum(getattr(r, value_field) for r in rows) / len(rows))
            labels.append(f"n={len(rows)}")

        figure.add_trace(
            go.Scatter(
                name=algorithm,
                x=xs,
                y=ys,
                text=labels,
                mode="lines+markers+text",
                textposition="top center",
                textfont=dict(size=10, color=config.COLOR_TEXT),
                line=dict(color=ALGORITHM_COLORS.get(algorithm), width=2),
                marker=dict(size=9),
                hovertemplate=f"{algorithm}<br>%{{x}}° grid<br>%{{y:.4g}}<extra></extra>",
            )
        )

    figure.update_xaxes(title="Planning resolution (degrees)", autorange="reversed")
    return _chart_layout(
        figure, "Search effort versus angular resolution", y_title, height
    )
