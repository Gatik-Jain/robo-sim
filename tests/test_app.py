"""Interaction tests for the Streamlit interface.

These use ``streamlit.testing.v1.AppTest``, which runs the real app script in
process and exposes its widgets, without a browser. They verify wiring and
state handling -- that pressing a button calls the right function and that
stale results are invalidated.

They do **not** verify anything visual. Whether the Plotly animation's Play
button behaves correctly in Chrome or Safari is a separate, manual check; see
the README's "Verification" section for what was and was not confirmed that
way.

If Streamlit is not installed the whole module is skipped, so the mathematical
core can still be tested in a bare environment.
"""

import unittest
from pathlib import Path

try:
    from streamlit.testing.v1 import AppTest

    STREAMLIT_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the environment
    STREAMLIT_AVAILABLE = False

from robo_sim import config


# AppTest resolves a relative path against the *calling* file, so the app is
# addressed absolutely from the project root instead.
APP_PATH = Path(__file__).resolve().parent.parent / "streamlit_app.py"


def fresh_app(timeout: int = 180):
    app = AppTest.from_file(str(APP_PATH), default_timeout=timeout)
    app.run()
    return app


def slider(app, key):
    """Select a slider by key.

    Index-based selection is unreliable here: the playback "Frame" slider only
    exists once a path has been planned, which shifts every other index.
    """
    for candidate in app.slider:
        if candidate.key == key:
            return candidate
    raise AssertionError(
        f"No slider with key {key!r}. Found: {[s.key for s in app.slider]}"
    )


def button(app, label):
    for candidate in app.button:
        if candidate.label == label:
            return candidate
    raise AssertionError(
        f"No button labelled {label!r}. Found: {[b.label for b in app.button]}"
    )


@unittest.skipUnless(STREAMLIT_AVAILABLE, "Streamlit is not installed")
class AppStartupTests(unittest.TestCase):
    def setUp(self):
        self.app = fresh_app()

    def test_app_runs_without_exceptions(self):
        self.assertEqual(list(self.app.exception), [])

    def test_four_tabs_are_present(self):
        self.assertEqual(len(self.app.tabs), 4)

    def test_session_state_starts_at_the_documented_defaults(self):
        state = self.app.session_state
        self.assertEqual(state["theta1"], config.JOINT1_INITIAL)
        self.assertEqual(state["theta2"], config.JOINT2_INITIAL)
        self.assertEqual(state["link1"], config.LINK1_LENGTH)
        self.assertEqual(state["link2"], config.LINK2_LENGTH)
        self.assertEqual(state["resolution"], config.DEFAULT_RESOLUTION)
        self.assertEqual(state["obstacles"], [])
        self.assertIsNone(state["plan_result"])

    def test_end_effector_readout_matches_forward_kinematics(self):
        from robo_sim.kinematics import end_effector_position
        from robo_sim.models import RobotSpec

        expected = end_effector_position(
            RobotSpec(), config.JOINT1_INITIAL, config.JOINT2_INITIAL
        )
        shown = [m.value for m in self.app.metric if m.label == "End effector"][0]
        self.assertEqual(shown, f"({expected[0]:.1f}, {expected[1]:.1f})")

    def test_number_inputs_display_the_value_they_actually_use(self):
        """Guard against the displayed number drifting from the stored one.

        ``st.number_input`` sends ``min_value`` to the browser as its initial
        display value even when session_state holds something else, so a
        key-only widget showed "20.00" in the box while the robot used a
        170 unit link. ``proto.default`` is what the browser renders, so it is
        asserted here rather than ``.value``, which reads session state and
        would have passed throughout.
        """
        for key in ("link1", "link2", "target_x", "target_y"):
            with self.subTest(key=key):
                widget = [n for n in self.app.number_input if n.key == key][0]
                self.assertEqual(
                    widget.proto.default, self.app.session_state[key]
                )

    def test_two_sessions_do_not_share_state(self):
        """Each visitor must get an independent simulation."""
        other = fresh_app()
        self.app.session_state["theta1"] = 99.0
        self.assertEqual(other.session_state["theta1"], config.JOINT1_INITIAL)


@unittest.skipUnless(STREAMLIT_AVAILABLE, "Streamlit is not installed")
class JointControlTests(unittest.TestCase):
    def test_moving_a_joint_updates_the_readout(self):
        app = fresh_app()
        slider(app, "theta1").set_value(90.0).run()
        self.assertEqual(app.session_state["theta1"], 90.0)
        shown = [m.value for m in app.metric if m.label == "Joint 1"][0]
        self.assertEqual(shown, "90.0°")

    def test_reset_robot_restores_the_defaults(self):
        app = fresh_app()
        slider(app, "theta1").set_value(120.0).run()
        button(app, "↺ Reset robot").click().run()
        self.assertEqual(app.session_state["theta1"], config.JOINT1_INITIAL)
        self.assertEqual(app.session_state["theta2"], config.JOINT2_INITIAL)


@unittest.skipUnless(STREAMLIT_AVAILABLE, "Streamlit is not installed")
class InverseKinematicsUiTests(unittest.TestCase):
    def test_solving_ik_moves_the_arm_onto_a_reachable_target(self):
        app = fresh_app()
        app.session_state["target_x"] = 200.0
        app.session_state["target_y"] = 60.0
        app.run()
        button(app, "Solve IK").click().run()

        result = app.session_state["ik_result"]
        self.assertIsNotNone(result)
        self.assertTrue(result.success, result.message)
        self.assertEqual(app.session_state["status"][0], "success")

        from robo_sim.kinematics import target_error
        from robo_sim.models import RobotSpec

        error = target_error(
            RobotSpec(),
            app.session_state["theta1"],
            app.session_state["theta2"],
            (200.0, 60.0),
        )
        self.assertLess(error, 0.2)  # the stored angles are rounded to 2 dp

    def test_unreachable_target_reports_an_error_without_moving_the_arm(self):
        app = fresh_app()
        before = (app.session_state["theta1"], app.session_state["theta2"])
        app.session_state["target_x"] = 500.0
        app.session_state["target_y"] = 0.0
        app.run()
        button(app, "Solve IK").click().run()

        self.assertEqual(list(app.exception), [])
        self.assertEqual(app.session_state["status"][0], "error")
        self.assertIn("unreachable", app.session_state["status"][1].lower())
        self.assertEqual(
            (app.session_state["theta1"], app.session_state["theta2"]), before
        )


@unittest.skipUnless(STREAMLIT_AVAILABLE, "Streamlit is not installed")
class ObstacleUiTests(unittest.TestCase):
    def test_adding_an_obstacle(self):
        app = fresh_app()
        button(app, "➕ Add obstacle").click().run()
        self.assertEqual(len(app.session_state["obstacles"]), 1)
        self.assertEqual(app.session_state["status"][0], "success")

    def test_clearing_all_obstacles(self):
        app = fresh_app()
        button(app, "➕ Add obstacle").click().run()
        button(app, "🗑 Clear all obstacles").click().run()
        self.assertEqual(app.session_state["obstacles"], [])

    def test_reset_environment_restores_the_default_target(self):
        app = fresh_app()
        button(app, "➕ Add obstacle").click().run()
        app.session_state["target_x"] = 10.0
        app.run()
        button(app, "↺ Reset environment").click().run()
        self.assertEqual(app.session_state["obstacles"], [])
        self.assertEqual(app.session_state["target_x"], config.TARGET_X_INITIAL)

    def test_loading_an_example_scenario_populates_the_environment(self):
        from streamlit_app import load_example

        app = fresh_app()
        app.session_state["obstacles"] = []
        # load_example works on st.session_state, so drive it through the app.
        app.selectbox[0].set_value("Three obstacles").run()
        button(app, "Load “Three obstacles”").click().run()
        self.assertEqual(len(app.session_state["obstacles"]), 3)
        self.assertEqual(app.session_state["scenario_loaded"], "Three obstacles")


@unittest.skipUnless(STREAMLIT_AVAILABLE, "Streamlit is not installed")
class PlanningUiTests(unittest.TestCase):
    def test_planning_stores_a_validated_path(self):
        app = fresh_app()
        app.session_state["target_x"] = 150.0
        app.session_state["target_y"] = -200.0
        app.run()
        button(app, "Plan path").click().run()

        result = app.session_state["plan_result"]
        self.assertIsNotNone(result)
        self.assertTrue(result.path_found, result.message)
        self.assertIs(result.collision_validated, True)
        self.assertEqual(app.session_state["status"][0], "success")

    def test_moving_a_joint_invalidates_an_existing_plan(self):
        """A stale path must never be left on screen as though still valid."""
        app = fresh_app()
        app.session_state["target_x"] = 150.0
        app.session_state["target_y"] = -200.0
        app.run()
        button(app, "Plan path").click().run()
        self.assertIsNotNone(app.session_state["plan_result"])

        slider(app, "theta1").set_value(100.0).run()
        self.assertIsNone(app.session_state["plan_result"])
        self.assertEqual(app.session_state["status"][0], "warning")

    def test_adding_an_obstacle_invalidates_an_existing_plan(self):
        app = fresh_app()
        app.session_state["target_x"] = 150.0
        app.session_state["target_y"] = -200.0
        app.run()
        button(app, "Plan path").click().run()
        self.assertIsNotNone(app.session_state["plan_result"])

        button(app, "➕ Add obstacle").click().run()
        self.assertIsNone(app.session_state["plan_result"])

    def test_changing_the_resolution_invalidates_an_existing_plan(self):
        app = fresh_app()
        app.session_state["target_x"] = 150.0
        app.session_state["target_y"] = -200.0
        app.run()
        button(app, "Plan path").click().run()
        self.assertIsNotNone(app.session_state["plan_result"])

        app.session_state["resolution"] = 10.0
        app.run()
        self.assertIsNone(app.session_state["plan_result"])

    def test_unreachable_target_fails_without_crashing(self):
        app = fresh_app()
        app.session_state["target_x"] = 500.0
        app.session_state["target_y"] = 0.0
        app.run()
        button(app, "Plan path").click().run()

        self.assertEqual(list(app.exception), [])
        self.assertEqual(app.session_state["status"][0], "error")
        self.assertFalse(app.session_state["plan_result"].path_found)

    def test_comparison_runs_both_planners_on_one_scenario(self):
        app = fresh_app()
        app.session_state["target_x"] = 150.0
        app.session_state["target_y"] = -200.0
        app.run()
        button(app, "Run A* and Dijkstra on the current scenario").click().run()

        comparison = app.session_state["comparison"]
        self.assertIsNotNone(comparison)
        self.assertEqual(set(comparison), {"A*", "Dijkstra"})
        self.assertTrue(comparison["A*"].path_found)
        self.assertTrue(comparison["Dijkstra"].path_found)
        self.assertAlmostEqual(
            comparison["A*"].path_cost, comparison["Dijkstra"].path_cost, places=6
        )


@unittest.skipUnless(STREAMLIT_AVAILABLE, "Streamlit is not installed")
class ExperimentUiTests(unittest.TestCase):
    def test_running_the_experiments_offers_a_csv_download(self):
        app = fresh_app(timeout=600)
        button(app, "Run all scenarios at 5°").click().run()

        results = app.session_state["experiment_results"]
        self.assertEqual(len(results), 10)

        labels = [d.label for d in app.download_button]
        self.assertIn("⬇ Download results as CSV", labels)

        # AppTest serves the download payload by URL and does not expose the
        # bytes, so the CSV is checked through the same function that supplies
        # the button, applied to the results the app actually stored.
        from robo_sim.experiments import results_to_csv
        from robo_sim.models import CSV_FIELDNAMES

        text = results_to_csv(results)
        self.assertEqual(text.splitlines()[0].split(","), list(CSV_FIELDNAMES))
        self.assertEqual(len(text.strip().splitlines()), len(results) + 1)


if __name__ == "__main__":
    unittest.main()
