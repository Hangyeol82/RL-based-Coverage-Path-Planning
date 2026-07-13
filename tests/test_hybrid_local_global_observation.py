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


def test_hybrid_observation_shapes_without_dtm() -> None:
    env = PaperCPPDiscreteGymEnv(_grid(), start_pos=(1, 1), config=_config(include_dtm=False))
    obs, _info = env.reset()

    assert set(obs.keys()) == {"local_map", "global_map_64", "global_map_32", "global_map_16", "robot_state"}
    assert obs["local_map"].shape == (5, 41, 41)
    assert obs["global_map_64"].shape == (3, 64, 64)
    assert obs["global_map_32"].shape == (3, 32, 32)
    assert obs["global_map_16"].shape == (3, 16, 16)
    assert obs["robot_state"].shape == (50,)
    assert env.action_masks().shape == (4,)


def test_hybrid_observation_shapes_with_six_dtm() -> None:
    env = PaperCPPDiscreteGymEnv(_grid(), start_pos=(1, 1), config=_config(include_dtm=True))
    obs, _info = env.reset()

    assert set(obs.keys()) == {"local_map", "global_map_64", "global_map_32", "global_map_16", "robot_state"}
    assert obs["local_map"].shape == (5, 41, 41)
    assert obs["global_map_64"].shape == (9, 64, 64)
    assert obs["global_map_32"].shape == (9, 32, 32)
    assert obs["global_map_16"].shape == (9, 16, 16)
    assert obs["robot_state"].shape == (50,)
    assert env.action_masks().shape == (4,)
