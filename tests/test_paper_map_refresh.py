import unittest

import numpy as np

from learning.reinforcement.cpp_env import CPPDiscreteEnvConfig
from paper_training.cpp_env import PaperCPPDiscreteGymEnv
from paper_training.offline_cpp_env import OfflinePaperCPPDiscreteGymEnv


def _map_with_obstacle(row: int, col: int) -> np.ndarray:
    grid = np.zeros((8, 8), dtype=np.int32)
    grid[row, col] = 1
    return grid


class PaperMapRefreshTest(unittest.TestCase):
    def _make_env(self, env_cls):
        maps = (
            _map_with_obstacle(1, 1),
            _map_with_obstacle(2, 2),
            _map_with_obstacle(3, 3),
        )
        metadata = tuple(
            {
                "map_index": idx,
                "family": f"family_{idx}",
                "generator": f"generator_{idx}",
                "level": idx + 1,
            }
            for idx in range(len(maps))
        )
        return env_cls(
            grid_map=maps[0],
            start_pos=(0, 0),
            config=CPPDiscreteEnvConfig(max_steps=4),
            grid_map_pool=maps,
            grid_map_metadata_pool=metadata,
            episode_map_refresh=True,
            map_refresh_mode="cycle",
            map_refresh_seed=0,
        )

    def _assert_cycle(self, env_cls):
        env = self._make_env(env_cls)
        try:
            indices = []
            families = []
            for seed in (0, None, None, None):
                env.reset(seed=seed)
                indices.append(env.core_env._paper_episode_map_index)
                families.append(env.core_env._paper_episode_map_metadata["family"])
            self.assertEqual(indices, [0, 1, 2, 0])
            self.assertEqual(families, ["family_0", "family_1", "family_2", "family_0"])
        finally:
            env.close()

    def test_online_env_cycles_map_pool_on_automatic_reset(self):
        self._assert_cycle(PaperCPPDiscreteGymEnv)

    def test_offline_env_cycles_map_pool_on_automatic_reset(self):
        self._assert_cycle(OfflinePaperCPPDiscreteGymEnv)


if __name__ == "__main__":
    unittest.main()
