import unittest

import numpy as np

from learning.observation import MultiScaleCPPObservationBuilder, MultiScaleCPPObservationConfig
from learning.reinforcement.cpp_env import CPPDiscreteEnv, CPPDiscreteEnvConfig


def _known_map() -> np.ndarray:
    grid = np.zeros((16, 16), dtype=np.int32)
    grid[3:6, 7] = 1
    grid[10, 2:8] = 1
    grid[12:14, 12:14] = 1
    return grid


class DTMTwoModeTest(unittest.TestCase):
    def test_two_mode_is_axis2_alias_for_observation(self):
        occupancy = _known_map()
        explored = np.zeros_like(occupancy, dtype=bool)
        explored[7, 7] = True

        common = dict(
            dtm_coarse_mode="bfs",
            dtm_connectivity=4,
            include_cell_phase_channels=True,
        )
        axis_builder = MultiScaleCPPObservationBuilder(
            MultiScaleCPPObservationConfig(dtm_output_mode="axis2", **common),
            include_dtm=True,
        )
        two_builder = MultiScaleCPPObservationBuilder(
            MultiScaleCPPObservationConfig(dtm_output_mode="two", **common),
            include_dtm=True,
        )

        self.assertEqual(axis_builder.channel_names_by_level, two_builder.channel_names_by_level)
        axis_levels = axis_builder.build_levels(occupancy, robot_pos=(7, 7), explored=explored)
        two_levels = two_builder.build_levels(occupancy, robot_pos=(7, 7), explored=explored)

        self.assertEqual(axis_levels.keys(), two_levels.keys())
        for level_id in axis_levels:
            np.testing.assert_array_equal(axis_levels[level_id], two_levels[level_id])

    def test_two_mode_boundary_exit_channel_count_and_scores(self):
        env = CPPDiscreteEnv(
            np.zeros((8, 8), dtype=np.int32),
            config=CPPDiscreteEnvConfig(
                include_dtm=True,
                observation=MultiScaleCPPObservationConfig(dtm_output_mode="two"),
            ),
        )
        self.assertEqual(env._dtm_channel_count(), 2)
        np.testing.assert_allclose(
            env._exit_scores_from_dtm(np.array([0.5, 0.2], dtype=np.float32)),
            (0.2, 0.5, 0.2, 0.5),
        )

    def test_six_boundary_exit_scores_follow_side_pair_order(self):
        env = CPPDiscreteEnv(
            np.zeros((8, 8), dtype=np.int32),
            config=CPPDiscreteEnvConfig(
                include_dtm=True,
                observation=MultiScaleCPPObservationConfig(dtm_output_mode="six"),
            ),
        )

        # Channel 0 is U<->R, so both up and right exits must be active.
        self.assertEqual(env._exit_scores_from_dtm(np.array([1, 0, 0, 0, 0, 0], dtype=np.float32)), (1.0, 1.0, 0.0, 0.0))
        # Channel 5 is D<->L, so both down and left exits must be active.
        self.assertEqual(env._exit_scores_from_dtm(np.array([0, 0, 0, 0, 0, 1], dtype=np.float32)), (0.0, 0.0, 1.0, 1.0))


if __name__ == "__main__":
    unittest.main()
