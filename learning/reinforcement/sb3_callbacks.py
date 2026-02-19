from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Sequence

import numpy as np

try:
    from stable_baselines3.common.callbacks import BaseCallback
except Exception as exc:  # pragma: no cover
    BaseCallback = object  # type: ignore[assignment]
    _SB3_IMPORT_ERROR = exc
else:
    _SB3_IMPORT_ERROR = None


class RewardBreakdownCallback(BaseCallback):
    """
    Collect rollout-level means for reward breakdown terms from env info.
    """

    TRACK_KEYS: Sequence[str] = (
        "reward_area",
        "reward_tv_i",
        "reward_tv_g",
        "reward_coll",
        "reward_const",
        "reward_total",
        "coverage_ratio",
        "collision",
    )

    def __init__(self, verbose: int = 0):
        if _SB3_IMPORT_ERROR is not None:
            raise ImportError(
                "stable_baselines3 is required for RewardBreakdownCallback"
            ) from _SB3_IMPORT_ERROR
        super().__init__(verbose=verbose)
        self._cur: DefaultDict[str, List[float]] = defaultdict(list)
        self.history: List[Dict[str, float]] = []
        self.rollout_idx: int = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if isinstance(infos, dict):
            infos = [infos]

        for info in infos:
            if not isinstance(info, dict):
                continue
            for key in self.TRACK_KEYS:
                if key in info:
                    self._cur[key].append(float(info[key]))
        return True

    def _on_rollout_end(self) -> None:
        self.rollout_idx += 1
        rec: Dict[str, float] = {
            "rollout": float(self.rollout_idx),
            "timesteps": float(self.num_timesteps),
        }
        for key in self.TRACK_KEYS:
            vals = self._cur.get(key, [])
            if len(vals) > 0:
                rec[key] = float(np.mean(vals))
                self.logger.record(f"reward_breakdown/{key}", rec[key])
        self.history.append(rec)
        self._cur.clear()

    def save_json(self, path: str):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "rollouts": self.history,
            "track_keys": list(self.TRACK_KEYS),
        }
        with p.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def save_csv(self, path: str):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if len(self.history) == 0:
            with p.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["rollout", "timesteps", *self.TRACK_KEYS])
            return

        keys = ["rollout", "timesteps", *self.TRACK_KEYS]
        with p.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in self.history:
                rec = {k: row.get(k, "") for k in keys}
                writer.writerow(rec)
