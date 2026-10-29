"""Tests for the experiment runner and its CSV output.

These run the real planners on the real scenarios. They are the slowest tests
in the suite, which is the price of checking measured output rather than
mocked output.
"""

import csv
import io
import tempfile
import unittest
from pathlib import Path

from robo_sim import config
from robo_sim.experiments import (
    build_scenarios,
    main,
    results_to_csv,
    results_to_rows,
    run_experiments,
    run_scenario,
    write_csv,
)
from robo_sim.models import CSV_FIELDNAMES, PLAN_NO_PATH, PLAN_OK


class ScenarioDefinitionTests(unittest.TestCase):
    def setUp(self):
        self.scenarios = build_scenarios()

    def test_five_scenarios_are_defined(self):
        self.assertEqual(len(self.scenarios), 5)

    def test_scenario_ids_are_unique(self):
        ids = [s.scenario_id for s in self.scenarios]
        self.assertEqual(len(ids), len(set(ids)))

    def test_every_scenario_documents_its_expected_behaviour(self):
        for scenario in self.scenarios:
            self.assertTrue(scenario.description)
            self.assertTrue(scenario.expectation)

    def test_scenarios_cover_the_required_variety(self):
        obstacle_counts = sorted(len(s.obstacles) for s in self.scenarios)
        self.assertEqual(obstacle_counts[0], 0)      # a scenario with none
        self.assertGreaterEqual(obstacle_counts[-1], 3)  # one with several

    def test_resolution_can_be_overridden(self):
        for scenario in build_scenarios(resolution_deg=10.0):
            self.assertEqual(scenario.resolution_deg, 10.0)


class RunnerTests(unittest.TestCase):
    def test_single_run_produces_a_populated_record(self):
        scenario = build_scenarios()[0]
        result = run_scenario(scenario, "A*")
        self.assertEqual(result.scenario_id, scenario.scenario_id)
        self.assertEqual(result.algorithm, "A*")
        self.assertTrue(result.path_found)
        self.assertEqual(result.status, PLAN_OK)
        self.assertGreater(result.planning_time_ms, 0.0)
        self.assertGreater(result.expanded_states, 0)
        self.assertGreater(result.path_cost, 0.0)
        self.assertGreater(result.path_configurations, 1)
        self.assertIs(result.collision_validated, True)

    def test_scenario_parameters_are_copied_into_the_record(self):
        scenario = build_scenarios()[1]
        result = run_scenario(scenario, "A*")
        self.assertEqual(result.target_x, scenario.target[0])
        self.assertEqual(result.target_y, scenario.target[1])
        self.assertEqual(result.link1_length, scenario.spec.link1_length)
        self.assertEqual(result.resolution_deg, scenario.resolution_deg)
        self.assertEqual(result.target_tolerance, scenario.target_tolerance)
        self.assertIn("230", result.obstacles)

    def test_obstacle_free_scenario_records_none(self):
        result = run_scenario(build_scenarios()[0], "A*")
        self.assertEqual(result.obstacles, "none")

    def test_failures_are_recorded_rather_than_dropped(self):
        failing = build_scenarios()[4]  # S5, designed to have no path
        result = run_scenario(failing, "A*")
        self.assertFalse(result.path_found)
        self.assertEqual(result.status, PLAN_NO_PATH)
        self.assertEqual(result.path_cost, 0.0)
        self.assertEqual(result.path_configurations, 0)
        self.assertIsNone(result.collision_validated)
        # The search really ran, so effort was still spent and recorded.
        self.assertGreater(result.expanded_states, 0)

    def test_full_run_covers_every_scenario_and_algorithm(self):
        results = run_experiments()
        self.assertEqual(len(results), 5 * 2)
        self.assertEqual({r.algorithm for r in results}, {"A*", "Dijkstra"})

    def test_both_algorithms_agree_on_cost_wherever_both_succeed(self):
        """A cross-check on optimality using the real benchmark scenarios."""
        results = run_experiments()
        by_scenario: dict[str, dict[str, object]] = {}
        for result in results:
            by_scenario.setdefault(result.scenario_id, {})[result.algorithm] = result

        compared = 0
        for scenario_id, runs in by_scenario.items():
            astar, dijkstra = runs["A*"], runs["Dijkstra"]
            self.assertEqual(astar.path_found, dijkstra.path_found, scenario_id)
            if astar.path_found:
                self.assertAlmostEqual(
                    astar.path_cost, dijkstra.path_cost, places=6, msg=scenario_id
                )
                compared += 1
        self.assertGreaterEqual(compared, 4)

    def test_progress_callback_is_invoked_once_per_run(self):
        calls = []
        run_experiments(
            scenarios=build_scenarios()[:2],
            algorithms=("A*",),
            progress=lambda done, total, label: calls.append((done, total, label)),
        )
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[-1][0], calls[-1][1])

    def test_runner_does_not_import_streamlit(self):
        """The computational core must work headlessly.

        Checked in a subprocess: within this process ``tests/test_app.py`` has
        already imported Streamlit, so inspecting ``sys.modules`` here would
        prove nothing.
        """
        import subprocess
        import sys
        from pathlib import Path

        project_root = Path(__file__).resolve().parent.parent
        script = (
            "import sys;"
            "from robo_sim.experiments import run_experiments, build_scenarios;"
            "run_experiments(scenarios=build_scenarios()[:1], algorithms=('A*',));"
            "leaked = [m for m in sys.modules if m.split('.')[0] == 'streamlit'];"
            "print('LEAKED' if leaked else 'CLEAN')"
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=300,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("CLEAN", completed.stdout)


class CsvTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = run_experiments(
            scenarios=build_scenarios()[:2], algorithms=("A*", "Dijkstra")
        )
        cls.text = results_to_csv(cls.results)

    def test_header_matches_the_declared_field_names(self):
        first_line = self.text.splitlines()[0]
        self.assertEqual(first_line.split(","), list(CSV_FIELDNAMES))

    def test_one_data_row_per_result(self):
        rows = list(csv.DictReader(io.StringIO(self.text)))
        self.assertEqual(len(rows), len(self.results))

    def test_values_come_from_the_actual_run(self):
        rows = list(csv.DictReader(io.StringIO(self.text)))
        for row, result in zip(rows, self.results):
            self.assertEqual(row["scenario_id"], result.scenario_id)
            self.assertEqual(row["algorithm"], result.algorithm)
            self.assertEqual(row["path_found"], str(result.path_found))
            self.assertAlmostEqual(
                float(row["path_cost"]), result.path_cost, places=6
            )
            self.assertEqual(int(row["expanded_states"]), result.expanded_states)

    def test_output_reads_back_into_python_with_usable_types(self):
        rows = list(csv.DictReader(io.StringIO(self.text)))
        for row in rows:
            float(row["planning_time_ms"])
            int(row["expanded_states"])
            float(row["final_target_error"])
            self.assertIn(row["path_found"], ("True", "False"))

    def test_unknown_validation_is_blank_not_false(self):
        """A plan that never ran is not the same as one that failed validation."""
        results = run_experiments(
            scenarios=build_scenarios()[4:5], algorithms=("A*",)
        )
        rows = results_to_rows(results)
        self.assertEqual(rows[0]["collision_validated"], "")

    def test_write_csv_creates_the_file_and_parent_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "out.csv"
            written = write_csv(self.results, path)
            self.assertTrue(written.exists())
            rows = list(csv.DictReader(io.StringIO(written.read_text(encoding="utf-8"))))
            self.assertEqual(len(rows), len(self.results))


class CommandLineTests(unittest.TestCase):
    def test_cli_writes_a_readable_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cli.csv"
            exit_code = main(["--output", str(path)])
            self.assertEqual(exit_code, 0)
            self.assertTrue(path.exists())
            rows = list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8"))))
            self.assertEqual(len(rows), 10)
            self.assertEqual(list(rows[0]), list(CSV_FIELDNAMES))

    def test_cli_can_run_a_single_algorithm_without_writing(self):
        exit_code = main(["--algorithms", "A*", "--no-write"])
        self.assertEqual(exit_code, 0)

    def test_cli_rejects_an_unknown_algorithm(self):
        with self.assertRaises(SystemExit):
            main(["--algorithms", "Greedy", "--no-write"])


class ResolutionSweepTests(unittest.TestCase):
    def test_sweep_runs_each_resolution(self):
        from robo_sim.experiments import run_resolution_sweep

        results = run_resolution_sweep(resolutions=(10.0,), algorithms=("A*",))
        self.assertEqual(len(results), 5)
        self.assertEqual({r.resolution_deg for r in results}, {10.0})

    def test_supported_resolutions_are_the_documented_ones(self):
        self.assertEqual(config.SUPPORTED_RESOLUTIONS, (3.0, 5.0, 10.0))
        self.assertIn(config.DEFAULT_RESOLUTION, config.SUPPORTED_RESOLUTIONS)


if __name__ == "__main__":
    unittest.main()
