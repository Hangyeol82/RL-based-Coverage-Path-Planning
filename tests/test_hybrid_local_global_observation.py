import numpy as np

from learning.observation import HybridLocalGlobalCPPObservationConfig
from learning.reinforcement.cpp_env import CPPDiscreteEnvConfig
from paper_training.cpp_env import PaperCPPDiscreteGymEnv
from paper_training.robot_state_observation import RobotStateObservationConfig


def _grid() -> np.ndarray:
    grid = np.zeros((128, 128), dtype=np.int32)
    grid[32:96, 64] = 1
    grid[72, 18:82] = 1
    return grid


def _config(*, include_dtm: bool) -> CPPDiscreteEnvConfig:
    return CPPDiscreteEnvConfig(
        sensor_range=3,
        max_steps=100,
        include_dtm=include_dtm,
        maps_observation_mode="hybrid_local_global",
        hybrid_observation=HybridLocalGlobalCPPObservationConfig(
            local_crop_size=41,
            global_coarse_size=16,
            global_coarse_sizes=(64, 32, 16),
            dtm_output_mode="six",
        ),
        robot_state=RobotStateObservationConfig(
            include_position=False,
            include_action_history=True,
            action_history_len=10,
            include_progress=False,
            include_stagnation=False,
        ),
        use_cell_phase_features=False,
        use_action_mask=True,
    )


def _assert_robot_marker(obs: dict, *, size: int, robot_pos: tuple[int, int]) -> None:
    marker = obs[f"global_map_{size}"][3]
    expected = np.zeros((size, size), dtype=np.float32)
    expected[(robot_pos[0] * size) // 128, (robot_pos[1] * size) // 128] = 1.0
    assert np.array_equal(marker, expected)


def test_hybrid_observation_shapes_without_dtm() -> None:
    start = (65, 97)
    env = PaperCPPDiscreteGymEnv(_grid(), start_pos=start, config=_config(include_dtm=False))
    obs, _info = env.reset()

    assert set(obs.keys()) == {"local_map", "global_map_64", "global_map_32", "global_map_16", "robot_state"}
    assert obs["local_map"].shape == (5, 41, 41)
    assert obs["global_map_64"].shape == (4, 64, 64)
    assert obs["global_map_32"].shape == (4, 32, 32)
    assert obs["global_map_16"].shape == (4, 16, 16)
    _assert_robot_marker(obs, size=64, robot_pos=start)
    _assert_robot_marker(obs, size=32, robot_pos=start)
    _assert_robot_marker(obs, size=16, robot_pos=start)
    assert obs["robot_state"].shape == (50,)
    assert env.action_masks().shape == (4,)


def test_hybrid_observation_shapes_with_six_dtm() -> None:
    start = (65, 97)
    env = PaperCPPDiscreteGymEnv(_grid(), start_pos=start, config=_config(include_dtm=True))
    obs, _info = env.reset()

    assert set(obs.keys()) == {"local_map", "global_map_64", "global_map_32", "global_map_16", "robot_state"}
    assert obs["local_map"].shape == (5, 41, 41)
    assert obs["global_map_64"].shape == (10, 64, 64)
    assert obs["global_map_32"].shape == (10, 32, 32)
    assert obs["global_map_16"].shape == (10, 16, 16)
    _assert_robot_marker(obs, size=64, robot_pos=start)
    _assert_robot_marker(obs, size=32, robot_pos=start)
    _assert_robot_marker(obs, size=16, robot_pos=start)
    assert obs["robot_state"].shape == (50,)
    assert env.action_masks().shape == (4,)
