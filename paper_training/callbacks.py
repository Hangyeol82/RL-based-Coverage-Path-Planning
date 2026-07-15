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


class PaperMetricsCallback(BaseCallback):
    """
    Rollout logger for paper experiments.

    Step metrics are averaged over every environment step in the rollout.
    Episode metrics are averaged only over episodes that terminated during the
    rollout. step_to_XX is averaged over successful episodes for that threshold;
    success_XX is the success rate among terminated episodes.
    """

    STEP_MEAN_KEYS: Sequence[str] = (
        "reward_new_cell",
        "reward_area",
        "reward_tv_i",
        "reward_tv_g",
        "reward_coll",
        "reward_const",
        "reward_turn",
        "reward_overlap",
        "reward_milestone",
        "reward_hole",
        "reward_revisit_burden_shape",
        "reward_total",
        "offline_full_map",
        "offline_known_ratio",
        "coverage_ratio",
        "collision",
        "turn_event",
        "turn_ratio",
        "revisit_ratio",
        "overlap_ratio",
        "no_progress_streak_paper",
        "stagnation_detected",
        "loop_detected",
        "loop_or_stagnation_detected",
        "executed_hole_risk",
        "coverage_hole_count",
        "coverage_hole_known_mass",
        "coverage_hole_open_mass",
        "hole_refresh_ms",
        "hole_risk_ms",
        "heuristic_assist_enabled",
        "heuristic_active",
        "heuristic_selected",
        "heuristic_overridden",
        "heuristic_path_found",
        "heuristic_no_progress_streak_before",
        "heuristic_no_progress_streak",
        "heuristic_known_uncovered_count",
        "heuristic_path_length",
        "heuristic_astar_expansions",
        "heuristic_target_row",
        "heuristic_target_col",
        "heuristic_target_known_uncovered",
        "heuristic_target_frontier",
        "heuristic_target_cached",
        "revisit_burden_phi",
        "revisit_burden_phi_prev",
        "revisit_burden_phi_delta",
        "revisit_burden_value",
        "revisit_burden_value_prev",
        "revisit_burden_value_delta",
        "revisit_burden_target_count",
        "revisit_burden_reachable_target_count",
        "revisit_burden_unreachable_target_count",
        "revisit_burden_max_cost",
        "revisit_burden_compute_ms",
        "episode_map_index",
    )
    EPISODE_MEAN_KEYS: Sequence[str] = (
        "episode_final_coverage_ratio",
        "episode_final_collision",
        "episode_final_steps",
        "episode_turn_count",
        "episode_turn_ratio",
        "episode_revisit_count",
        "episode_revisit_ratio",
        "episode_overlap_ratio",
        "episode_coverage_auc",
        "episode_stagnation_step_count",
        "episode_loop_step_count",
        "episode_loop_or_stagnation_step_count",
        "episode_loop_or_stagnation_ratio",
    )
    TREND_KEYS: Sequence[str] = (
        "episode_final_coverage_ratio",
        "success_90",
        "success_95",
        "success_99",
        "step_to_90",
        "step_to_95",
        "step_to_99",
        "episode_turn_ratio",
        "episode_overlap_ratio",
        "episode_coverage_auc",
        "episode_loop_or_stagnation_ratio",
    )
    TREND_WINDOW = 10

    def __init__(self, verbose: int = 0):
        if _SB3_IMPORT_ERROR is not None:
            raise ImportError("stable_baselines3 is required for PaperMetricsCallback") from _SB3_IMPORT_ERROR
        super().__init__(verbose=verbose)
        self._step_sums: DefaultDict[str, float] = defaultdict(float)
        self._step_counts: DefaultDict[str, int] = defaultdict(int)
        self._episode: DefaultDict[str, List[float]] = defaultdict(list)
        self._step_count = 0
        self._override_step_count = 0
        self._episode_group: Dict[str, Dict[str, DefaultDict[str, List[float]]]] = {
            "family": {},
            "generator": {},
            "level": {},
            "map_index": {},
        }
        self.rollout_idx = 0
        self.history: List[Dict[str, float]] = []

    @staticmethod
    def _safe_token(raw: object) -> str:
        token = str(raw).strip().lower() or "unknown"
        out = []
        for ch in token:
            out.append(ch if ch.isalnum() else "_")
        return "".join(out).strip("_") or "unknown"

    def _group_bucket(self, kind: str, raw: object) -> DefaultDict[str, List[float]]:
        token = self._safe_token(raw)
        groups = self._episode_group[kind]
        if token not in groups:
            groups[token] = defaultdict(list)
        return groups[token]

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
            for key in self.STEP_MEAN_KEYS:
                if key in info:
                    self._step_sums[key] += float(info[key])
                    self._step_counts[key] += 1

            if str(info.get("done_reason", "")):
                episode_vals: Dict[str, float] = {}
                if "coverage_ratio" in info:
                    episode_vals["episode_final_coverage_ratio"] = float(info["coverage_ratio"])
                    self._episode["episode_final_coverage_ratio"].append(
                        episode_vals["episode_final_coverage_ratio"]
                    )
                if "collision" in info:
                    episode_vals["episode_final_collision"] = float(info["collision"])
                    self._episode["episode_final_collision"].append(episode_vals["episode_final_collision"])
                if "steps" in info:
                    episode_vals["episode_final_steps"] = float(info["steps"])
                    self._episode["episode_final_steps"].append(episode_vals["episode_final_steps"])
                for key in self.EPISODE_MEAN_KEYS:
                    if key in {
                        "episode_final_coverage_ratio",
                        "episode_final_collision",
                        "episode_final_steps",
                    }:
                        continue
                    if key in info:
                        episode_vals[key] = float(info[key])
                        self._episode[key].append(episode_vals[key])
                for suffix in ("90", "95", "99"):
                    success_key = f"episode_success_{suffix}"
                    step_key = f"episode_step_to_{suffix}"
                    if success_key in info:
                        episode_vals[f"success_{suffix}"] = float(info[success_key])
                        self._episode[success_key].append(episode_vals[f"success_{suffix}"])
                    if step_key in info:
                        val = float(info[step_key])
                        if np.isfinite(val):
                            episode_vals[f"step_to_{suffix}"] = val
                            self._episode[step_key].append(val)
                group_items = (
                    ("family", info.get("map_family", "unknown")),
                    ("generator", info.get("map_generator", "unknown")),
                    ("level", info.get("map_level", "unknown")),
                    ("map_index", int(float(info.get("episode_map_index", -1)))),
                )
                for kind, group_val in group_items:
                    bucket = self._group_bucket(kind, group_val)
                    bucket["episode_count"].append(1.0)
                    for metric_key, metric_val in episode_vals.items():
                        if np.isfinite(float(metric_val)):
                            bucket[metric_key].append(float(metric_val))
        return True

    @staticmethod
    def _mean(vals: List[float]) -> float:
        if not vals:
            return float("nan")
        return float(np.mean(np.asarray(vals, dtype=np.float64)))

    @staticmethod
    def _slope(vals: List[float]) -> float:
        ys = np.asarray([v for v in vals if np.isfinite(float(v))], dtype=np.float64)
        if ys.size < 3:
            return float("nan")
        xs = np.arange(ys.size, dtype=np.float64)
        return float(np.polyfit(xs, ys, deg=1)[0])

    @staticmethod
    def _std(vals: List[float]) -> float:
        ys = np.asarray([v for v in vals if np.isfinite(float(v))], dtype=np.float64)
        if ys.size < 2:
            return float("nan")
        return float(np.std(ys))

    def _on_rollout_end(self) -> None:
        self.rollout_idx += 1
        rec: Dict[str, float] = {
            "rollout": float(self.rollout_idx),
            "timesteps": float(self.num_timesteps),
        }
        for key in self.STEP_MEAN_KEYS:
            count = int(self._step_counts.get(key, 0))
            if count > 0:
                rec[key] = float(self._step_sums.get(key, 0.0)) / float(count)
                self.logger.record(f"paper_metrics/{key}", rec[key])

        if self._step_count > 0:
            rec["action_override_rate"] = float(self._override_step_count) / float(self._step_count)
            self.logger.record("paper_metrics/action_override_rate", rec["action_override_rate"])

        done_count = len(self._episode.get("episode_final_coverage_ratio", []))
        rec["episode_done_count"] = float(done_count)
        self.logger.record("paper_metrics/episode_done_count", rec["episode_done_count"])
        if done_count > 0:
            for key in self.EPISODE_MEAN_KEYS:
                vals = self._episode.get(key, [])
                if vals:
                    rec[key] = self._mean(vals)
                    self.logger.record(f"paper_metrics/{key}", rec[key])
            for suffix in ("90", "95", "99"):
                success_key = f"episode_success_{suffix}"
                step_key = f"episode_step_to_{suffix}"
                success_vals = self._episode.get(success_key, [])
                step_vals = self._episode.get(step_key, [])
                if success_vals:
                    rec[f"success_{suffix}"] = self._mean(success_vals)
                    rec[f"step_to_{suffix}"] = self._mean(step_vals)
                    rec[f"step_to_{suffix}_count"] = float(len(step_vals))
                    self.logger.record(f"paper_metrics/success_{suffix}", rec[f"success_{suffix}"])
                    self.logger.record(f"paper_metrics/step_to_{suffix}", rec[f"step_to_{suffix}"])

            for kind, groups in self._episode_group.items():
                for token, metrics in sorted(groups.items()):
                    count = float(len(metrics.get("episode_count", [])))
                    count_key = f"episode_count_{kind}_{token}"
                    rec[count_key] = count
                    self.logger.record(f"paper_metrics/{count_key}", count)
                    for metric_key, vals in sorted(metrics.items()):
                        if metric_key == "episode_count" or not vals:
                            continue
                        out_key = f"{kind}_{token}_{metric_key}"
                        rec[out_key] = self._mean(vals)
                        self.logger.record(f"paper_metrics/{out_key}", rec[out_key])

        rows = [*self.history, rec]
        for key in self.TREND_KEYS:
            vals = [float(row[key]) for row in rows[-self.TREND_WINDOW :] if key in row]
            if vals:
                slope_key = f"trend{self.TREND_WINDOW}_{key}_slope"
                std_key = f"trend{self.TREND_WINDOW}_{key}_std"
                rec[slope_key] = self._slope(vals)
                rec[std_key] = self._std(vals)
                self.logger.record(f"paper_metrics/{slope_key}", rec[slope_key])
                self.logger.record(f"paper_metrics/{std_key}", rec[std_key])

        self.history.append(rec)
        self._step_sums.clear()
        self._step_counts.clear()
        self._episode.clear()
        self._episode_group = {
            "family": {},
            "generator": {},
            "level": {},
            "map_index": {},
        }
        self._step_count = 0
        self._override_step_count = 0

    def save_json(self, path: str):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "rollouts": self.history,
            "step_mean_keys": list(self.STEP_MEAN_KEYS),
            "episode_mean_keys": list(self.EPISODE_MEAN_KEYS),
            "thresholds": [0.90, 0.95, 0.99],
        }
        with p.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def save_csv(self, path: str):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        base_keys = [
            "rollout",
            "timesteps",
            *self.STEP_MEAN_KEYS,
            "action_override_rate",
            "episode_done_count",
            *self.EPISODE_MEAN_KEYS,
            "success_90",
            "step_to_90",
            "step_to_90_count",
            "success_95",
            "step_to_95",
            "step_to_95_count",
            "success_99",
            "step_to_99",
            "step_to_99_count",
        ]
        extra = sorted({k for row in self.history for k in row.keys() if k not in base_keys})
        keys = base_keys + extra
        with p.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in self.history:
                writer.writerow({k: row.get(k, "") for k in keys})
