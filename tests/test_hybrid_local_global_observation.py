import numpy as np

from learning.observation import (
    HybridLocalGlobalCPPObservationConfig,
    MultiScaleCPPObservationBuilder,
    MultiScaleCPPObservationConfig,
)
from learning.reinforcement.cpp_env import CPPDiscreteEnvConfig
from paper_training.cpp_env import PaperCPPDiscreteGymEnv
from paper_training.robot_state_observation import RobotStateObservationConfig


def _grid() -> np.ndarray:
    grid = np.zeros((128, 128), dtype=np.int32)
    grid[32:96, 64] = 1
    grid[72, 18:82] = 1
    return grid


def _config(
    *,
    include_dtm: bool,
    global_sizes: tuple[int, ...] = (64, 32, 16),
    global_view_mode: str = "full",
    global_window_sizes: tuple[int, ...] = (),
) -> CPPDiscreteEnvConfig:
    return CPPDiscreteEnvConfig(
        sensor_range=3,
        max_steps=100,
        include_dtm=include_dtm,
        maps_observation_mode="hybrid_local_global",
        hybrid_observation=HybridLocalGlobalCPPObservationConfig(
            local_crop_size=41,
            global_coarse_size=16,
            global_coarse_sizes=global_sizes,
            global_view_mode=global_view_mode,
            global_window_sizes=global_window_sizes,
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


def _assert_centered_robot_marker(obs: dict, *, size: int, window_size: int) -> None:
    marker = obs[f"global_map_{size}"][3]
    expected = np.zeros((size, size), dtype=np.float32)
    center_idx = window_size // 2
    expected[(center_idx * size) // window_size, (center_idx * size) // window_size] = 1.0
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


def test_hybrid_centered_pyramid_observation_shapes_with_six_dtm() -> None:
    start = (65, 97)
    global_sizes = (16, 12, 8)
    global_windows = (64, 96, 128)
    env = PaperCPPDiscreteGymEnv(
        _grid(),
        start_pos=start,
        config=_config(
            include_dtm=True,
            global_sizes=global_sizes,
            global_view_mode="centered",
            global_window_sizes=global_windows,
        ),
    )
    obs, _info = env.reset()

    assert set(obs.keys()) == {"local_map", "global_map_16", "global_map_12", "global_map_8", "robot_state"}
    assert obs["local_map"].shape == (5, 41, 41)
    assert obs["global_map_16"].shape == (10, 16, 16)
    assert obs["global_map_12"].shape == (10, 12, 12)
    assert obs["global_map_8"].shape == (10, 8, 8)
    _assert_centered_robot_marker(obs, size=16, window_size=64)
    _assert_centered_robot_marker(obs, size=12, window_size=96)
    _assert_centered_robot_marker(obs, size=8, window_size=128)
    assert obs["robot_state"].shape == (50,)
    assert env.action_masks().shape == (4,)


def test_hybrid_centered_dtm_matches_multiscale_block4_crop() -> None:
    start = (65, 97)
    grid = _grid()

    hybrid = PaperCPPDiscreteGymEnv(
        grid,
        start_pos=start,
        config=_config(
            include_dtm=True,
            global_sizes=(16,),
            global_view_mode="centered",
            global_window_sizes=(64,),
        ),
    )
    obs, _info = hybrid.reset()

    reference_builder = MultiScaleCPPObservationBuilder(
        MultiScaleCPPObservationConfig(
            local_blocks=(4,),
            local_window_size=16,
            dtm_output_mode="six",
            include_cell_phase_channels=False,
            exclude_dtm_level0=False,
        ),
        include_dtm=True,
    )
    ref = reference_builder.build_levels(
        hybrid.core_env.known_map,
        robot_pos=start,
        explored=hybrid.core_env.explored,
    )[0]

    np.testing.assert_array_equal(obs["global_map_16"][4:10], ref[3:9])


def test_hybrid_viewport_shifts_window_inside_map_bounds() -> None:
    start = (5, 5)
    env = PaperCPPDiscreteGymEnv(
        _grid(),
        start_pos=start,
        config=_config(
            include_dtm=True,
            global_sizes=(32, 16, 12, 8),
            global_view_mode="viewport",
            global_window_sizes=(128, 64, 96, 128),
        ),
    )
    obs, _info = env.reset()

    assert set(obs.keys()) == {
        "local_map",
        "global_map_32",
        "global_map_16",
        "global_map_12",
        "global_map_8",
        "robot_state",
    }
    assert obs["global_map_32"].shape == (10, 32, 32)
    assert obs["global_map_16"].shape == (10, 16, 16)
    assert obs["global_map_12"].shape == (10, 12, 12)
    assert obs["global_map_8"].shape == (10, 8, 8)
    marker16 = obs["global_map_16"][3]
    assert marker16[1, 1] == 1.0
    assert marker16[8, 8] == 0.0
