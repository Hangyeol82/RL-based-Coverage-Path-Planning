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
    Collect rollout-level means from env info.
    - `coverage_ratio` is step-mean over the rollout.
    - `episode_final_coverage_ratio` is mean of per-episode terminal coverage values.
    """

    TRACK_KEYS: Sequence[str] = (
        "reward_area",
        "reward_tv_i",
        "reward_tv_g",
        "reward_coll",
        "reward_const",
        "reward_turn",
        "reward_overlap",
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
        self._step_count: int = 0
        self._override_step_count: int = 0
        self._episode_final_coverage: List[float] = []
        self._episode_final_collision: List[float] = []
        self._episode_final_steps: List[float] = []
        self.history: List[Dict[str, float]] = []
        self.rollout_idx: int = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if isinstance(infos, dict):
            infos = [infos]

        for info in infos:
            if not isinstance(info, dict):
                continue
            self._step_count += 1
            if bool(info.get("action_overridden", False)):
                self._override_step_count += 1
            for key in self.TRACK_KEYS:
                if key in info:
                    self._cur[key].append(float(info[key]))
            if str(info.get("done_reason", "")):
                if "coverage_ratio" in info:
                    self._episode_final_coverage.append(float(info["coverage_ratio"]))
                if "collision" in info:
                    self._episode_final_collision.append(float(info["collision"]))
                if "steps" in info:
                    self._episode_final_steps.append(float(info["steps"]))
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

        if self._step_count > 0:
            rec["action_override_rate"] = float(self._override_step_count) / float(self._step_count)
            self.logger.record("reward_breakdown/action_override_rate", rec["action_override_rate"])

        done_count = len(self._episode_final_coverage)
        rec["episode_done_count"] = float(done_count)
        self.logger.record("reward_breakdown/episode_done_count", rec["episode_done_count"])
        if done_count > 0:
            rec["episode_final_coverage_ratio"] = float(np.mean(self._episode_final_coverage))
            rec["episode_final_collision"] = (
                float(np.mean(self._episode_final_collision))
                if self._episode_final_collision
                else 0.0
            )
            rec["episode_final_steps"] = (
                float(np.mean(self._episode_final_steps))
                if self._episode_final_steps
                else float("nan")
            )
            self.logger.record("reward_breakdown/episode_final_coverage_ratio", rec["episode_final_coverage_ratio"])
            self.logger.record("reward_breakdown/episode_final_collision", rec["episode_final_collision"])
            self.logger.record("reward_breakdown/episode_final_steps", rec["episode_final_steps"])

        self.history.append(rec)
        self._cur.clear()
        self._step_count = 0
        self._override_step_count = 0
        self._episode_final_coverage.clear()
        self._episode_final_collision.clear()
        self._episode_final_steps.clear()

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
        base_keys = [
            "rollout",
            "timesteps",
            *self.TRACK_KEYS,
            "action_override_rate",
            "episode_done_count",
            "episode_final_coverage_ratio",
            "episode_final_collision",
            "episode_final_steps",
        ]
        if len(self.history) == 0:
            with p.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(base_keys)
            return

        extra = sorted({k for row in self.history for k in row.keys() if k not in base_keys})
        keys = base_keys + extra
        with p.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in self.history:
                rec = {k: row.get(k, "") for k in keys}
                writer.writerow(rec)
