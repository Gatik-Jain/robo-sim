"""Deterministic benchmark scenarios and the experiment runner.

This module has no dependency on Streamlit and can be run straight from a
terminal::

    python -m robo_sim.experiments
    python -m robo_sim.experiments --output results/my_run.csv --resolutions 5 10

Every number it reports is measured during the run. Nothing here is estimated,
remembered or hard-coded, and a planner that fails is recorded as a failure
rather than dropped from the dataset -- a success rate computed over only the
successes would be meaningless.

**Determinism.** No randomness is used anywhere. Scenarios are fixed
rectangles and fixed angles, the graph is a regular grid, and heap ties are
broken by an insertion counter, so the structural metrics (expanded states,
path cost, path length) reproduce exactly from run to run. Wall-clock
``planning_time_ms`` is the one exception: it depends on the machine, the
Python build and what else the computer is doing, so it should be read
alongside the expanded-state counts rather than on its own.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence

from . import config
from .models import (
    CSV_FIELDNAMES,
    Rectangle,
    RobotSpec,
    Scenario,
    ScenarioResult,
    describe_obstacles,
)
from .path_planning import GridTooLargeError, PLANNERS, PlanningProblem, plan

DEFAULT_START = (config.JOINT1_INITIAL, config.JOINT2_INITIAL)
DEFAULT_OUTPUT = Path("results") / "experiment_results.csv"


def build_scenarios(
    spec: Optional[RobotSpec] = None,
    resolution_deg: float = config.DEFAULT_RESOLUTION,
    target_tolerance: float = config.TARGET_TOLERANCE,
) -> tuple[Scenario, ...]:
    """The five built-in scenarios.

    Each one was chosen by running the planner and confirming the outcome, not
    by assuming it. The ``expectation`` text records what was observed and why
    it happens, so a future change that breaks one of them is noticeable.
    """
    spec = spec or RobotSpec()
    common = dict(
        spec=spec,
        start=DEFAULT_START,
        resolution_deg=resolution_deg,
        target_tolerance=target_tolerance,
    )

    return (
        Scenario(
            scenario_id="S1_free_space",
            description="Reachable target, no obstacles",
            target=(150.0, -200.0),
            obstacles=(),
            expectation=(
                "A path exists. With nothing in the way the optimal motion is a "
                "near-straight line in joint space, so A* should expand far fewer "
                "states than Dijkstra for the same path cost."
            ),
            **common,
        ),
        Scenario(
            scenario_id="S2_obstacle_near_route",
            description="One obstacle beside the direct route",
            target=(150.0, -200.0),
            obstacles=(Rectangle(230.0, 290.0, -120.0, -40.0),),
            expectation=(
                "A path exists but must bend around the obstacle, so its cost is "
                "higher than S1's. Same target as S1, which isolates the cost of "
                "the obstacle."
            ),
            **common,
        ),
        Scenario(
            scenario_id="S3_three_obstacles",
            description="Three obstacles around the workspace",
            target=(170.0, -160.0),
            obstacles=(
                Rectangle(200.0, 250.0, -70.0, 20.0),
                Rectangle(-80.0, 40.0, 180.0, 260.0),
                Rectangle(-270.0, -170.0, -80.0, 40.0),
            ),
            expectation=(
                "A path exists. The first obstacle forces the elbow to fold before "
                "the arm swings down; the other two remove large parts of the "
                "configuration space without blocking that particular route."
            ),
            **common,
        ),
        Scenario(
            scenario_id="S4_detour_required",
            description="Direct route blocked, detour available",
            target=(150.0, -200.0),
            obstacles=(Rectangle(210.0, 280.0, -150.0, -60.0),),
            expectation=(
                "The straight joint-space route is blocked, so the planner must "
                "detour. Same target as S1 and S2; the cost ordering S1 < S2 < S4 "
                "shows the obstacle's effect growing as it moves onto the route."
            ),
            **common,
        ),
        Scenario(
            scenario_id="S5_no_discrete_path",
            description="Goal poses exist but are sealed off",
            target=(170.0, -160.0),
            obstacles=(
                Rectangle(200.0, 250.0, -70.0, 20.0),
                Rectangle(175.0, 215.0, 25.0, 50.0),
            ),
            expectation=(
                "Collision-free goal poses exist and the start pose is valid, but "
                "the second obstacle seals the only gap the arm used in S3, so the "
                "search exhausts the reachable component and reports no path. This "
                "is a property of this grid and these obstacles; it is not a claim "
                "that no continuous motion could exist."
            ),
            **common,
        ),
    )


def run_scenario(
    scenario: Scenario, algorithm: str
) -> ScenarioResult:
    """Run one algorithm on one scenario and record what happened.

    A fresh :class:`PlanningProblem` is built for every run so that no planner
    inherits a warm collision cache from another. That keeps the timings
    comparable; sharing a problem would make whichever planner ran second look
    artificially fast.
    """
    try:
        problem = PlanningProblem(
            spec=scenario.spec,
            obstacles=scenario.obstacles,
            target=scenario.target,
            resolution_deg=scenario.resolution_deg,
            target_tolerance=scenario.target_tolerance,
        )
        result = plan(algorithm, problem, scenario.start)
    except GridTooLargeError as exc:
        from .models import PLAN_GRID_TOO_LARGE, PlanResult

        result = PlanResult(
            algorithm=algorithm,
            path_found=False,
            status=PLAN_GRID_TOO_LARGE,
            message=str(exc),
            resolution_deg=scenario.resolution_deg,
        )

    spec = scenario.spec
    return ScenarioResult(
        scenario_id=scenario.scenario_id,
        scenario_description=scenario.description,
        link1_length=spec.link1_length,
        link2_length=spec.link2_length,
        start_theta1=scenario.start[0],
        start_theta2=scenario.start[1],
        target_x=scenario.target[0],
        target_y=scenario.target[1],
        joint1_min=spec.joint1_min,
        joint1_max=spec.joint1_max,
        joint2_min=spec.joint2_min,
        joint2_max=spec.joint2_max,
        resolution_deg=scenario.resolution_deg,
        target_tolerance=scenario.target_tolerance,
        obstacles=describe_obstacles(scenario.obstacles),
        algorithm=algorithm,
        path_found=result.path_found,
        status=result.status,
        planning_time_ms=round(result.planning_time_ms, 4),
        expanded_states=result.expanded_states,
        discovered_states=result.discovered_states,
        path_cost=round(result.path_cost, 6),
        path_configurations=result.path_length,
        ee_path_length=round(result.ee_path_length, 6),
        total_joint_travel=round(result.total_joint_travel, 6),
        final_target_error=round(result.final_target_error, 6),
        collision_validated=result.collision_validated,
    )


def run_experiments(
    scenarios: Optional[Sequence[Scenario]] = None,
    algorithms: Sequence[str] = ("A*", "Dijkstra"),
    progress: Optional[callable] = None,
) -> list[ScenarioResult]:
    """Run every algorithm on every scenario.

    ``progress`` is an optional callback ``(done, total, label)`` used by the
    web interface to drive a progress bar. The runner itself never imports
    Streamlit.
    """
    scenarios = list(scenarios if scenarios is not None else build_scenarios())
    results: list[ScenarioResult] = []
    total = len(scenarios) * len(algorithms)
    done = 0

    for scenario in scenarios:
        for algorithm in algorithms:
            results.append(run_scenario(scenario, algorithm))
            done += 1
            if progress is not None:
                progress(done, total, f"{scenario.scenario_id} · {algorithm}")

    return results


def run_resolution_sweep(
    resolutions: Iterable[float] = config.SUPPORTED_RESOLUTIONS,
    algorithms: Sequence[str] = ("A*", "Dijkstra"),
    progress: Optional[callable] = None,
) -> list[ScenarioResult]:
    """Run the full scenario set once per planning resolution.

    This is what produces the "search performance versus angular resolution"
    chart: finer grids represent the free space more faithfully but contain
    many more states, so planning gets slower.
    """
    resolutions = list(resolutions)
    results: list[ScenarioResult] = []
    scenario_sets = [build_scenarios(resolution_deg=r) for r in resolutions]
    total = sum(len(s) for s in scenario_sets) * len(algorithms)
    done = 0

    for resolution, scenarios in zip(resolutions, scenario_sets):
        for scenario in scenarios:
            for algorithm in algorithms:
                results.append(run_scenario(scenario, algorithm))
                done += 1
                if progress is not None:
                    progress(
                        done,
                        total,
                        f"{resolution:g}° · {scenario.scenario_id} · {algorithm}",
                    )

    return results


# --------------------------------------------------------------------------
# CSV output
# --------------------------------------------------------------------------


def results_to_rows(results: Sequence[ScenarioResult]) -> list[dict[str, object]]:
    """Flatten results into dictionaries matching :data:`CSV_FIELDNAMES`."""
    rows: list[dict[str, object]] = []
    for result in results:
        row = {name: getattr(result, name) for name in CSV_FIELDNAMES}
        # An un-run validation is genuinely unknown, which is different from
        # "validated and failed". Write it as an empty cell, not as False.
        if row["collision_validated"] is None:
            row["collision_validated"] = ""
        rows.append(row)
    return rows


def results_to_csv(results: Sequence[ScenarioResult]) -> str:
    """Render results as CSV text, ready for a download button."""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(results_to_rows(results))
    return buffer.getvalue()


def write_csv(results: Sequence[ScenarioResult], path: Path) -> Path:
    """Write results to ``path``, creating the parent directory if needed.

    Only useful locally: a hosted Streamlit app's filesystem is temporary, so
    the web interface offers a download instead of writing files.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(results_to_csv(results), encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# Command line entry point
# --------------------------------------------------------------------------


def _summarise(results: Sequence[ScenarioResult]) -> str:
    """A readable summary table for the terminal."""
    header = (
        f"{'scenario':<24} {'algorithm':<9} {'found':<6} {'ms':>9} "
        f"{'expanded':>9} {'cost':>9} {'poses':>6} {'err u':>7}"
    )
    lines = [header, "-" * len(header)]
    for r in results:
        lines.append(
            f"{r.scenario_id:<24} {r.algorithm:<9} {str(r.path_found):<6} "
            f"{r.planning_time_ms:>9.2f} {r.expanded_states:>9d} "
            f"{r.path_cost:>9.2f} {r.path_configurations:>6d} "
            f"{r.final_target_error:>7.2f}"
        )
    successes = sum(1 for r in results if r.path_found)
    lines.append("-" * len(header))
    lines.append(
        f"{successes}/{len(results)} runs found a path "
        f"({100.0 * successes / len(results):.0f}% success rate)."
    )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m robo_sim.experiments",
        description="Run the ROBO-SIM planning benchmarks and write a CSV.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"CSV output path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--resolutions",
        "-r",
        type=float,
        nargs="+",
        default=None,
        metavar="DEG",
        help=(
            "Run the scenario set once per resolution instead of only at the "
            f"default {config.DEFAULT_RESOLUTION:g}°."
        ),
    )
    parser.add_argument(
        "--algorithms",
        "-a",
        nargs="+",
        default=["A*", "Dijkstra"],
        choices=sorted(PLANNERS),
        help="Which planners to run (default: both).",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Print the summary without writing a CSV file.",
    )
    args = parser.parse_args(argv)

    def progress(done: int, total: int, label: str) -> None:
        print(f"  [{done}/{total}] {label}", file=sys.stderr, flush=True)

    print("Running ROBO-SIM planning experiments...", file=sys.stderr)
    if args.resolutions:
        results = run_resolution_sweep(
            args.resolutions, tuple(args.algorithms), progress=progress
        )
    else:
        results = run_experiments(algorithms=tuple(args.algorithms), progress=progress)

    print()
    print(_summarise(results))

    if not args.no_write:
        written = write_csv(results, args.output)
        print()
        print(f"Wrote {len(results)} rows to {written}")

    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via the CLI
    raise SystemExit(main())
