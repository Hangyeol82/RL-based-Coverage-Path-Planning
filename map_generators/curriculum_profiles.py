from __future__ import annotations

from copy import deepcopy
from typing import Dict, Mapping, MutableMapping, Tuple


StageConfig = Dict[str, object]
StageMap = Dict[int, StageConfig]


def _with_label(config: Mapping[str, object], difficulty_label: str) -> StageConfig:
    out = dict(config)
    out["difficulty_label"] = str(difficulty_label)
    return out


LEGACY_FOUR_STAGE: StageMap = {
    1: _with_label(
        {
            "types": ["rect", "u_shape"],
            "overlap_prob": 0.08,
            "obs_per_1k_range": (3, 5),
            "size_ratio_range": (0.06, 0.11),
            "min_u_shape": 2,
            "u_shape_min_size": (3, 3),
        },
        "1",
    ),
    2: _with_label(
        {
            "types": ["rect", "u_shape"],
            "overlap_prob": 0.18,
            "obs_per_1k_range": (5, 8),
            "size_ratio_range": (0.08, 0.14),
            "min_u_shape": 2,
            "u_shape_min_size": (3, 3),
        },
        "2",
    ),
    3: _with_label(
        {
            "types": ["rect", "u_shape", "false_hole"],
            "overlap_prob": 0.28,
            "obs_per_1k_range": (8, 12),
            "size_ratio_range": (0.10, 0.18),
            "min_u_shape": 2,
            "u_shape_min_size": (3, 3),
        },
        "3",
    ),
    4: _with_label(
        {
            "types": ["rect", "u_shape", "false_hole"],
            "overlap_prob": 0.38,
            "obs_per_1k_range": (11, 16),
            "size_ratio_range": (0.12, 0.22),
            "min_u_shape": 2,
            "u_shape_min_size": (3, 3),
        },
        "4",
    ),
}


# 64x64 curriculum intended to stay within the "old stage 1~2" difficulty band:
#   level 1   ~= legacy stage 1
#   level 2   ~= midway between legacy stage 1 and 2
#   level 3   ~= legacy stage 2
RANDOM64_EASYMID2: StageMap = {
    1: _with_label(
        {
            "types": ["rect", "u_shape"],
            "overlap_prob": 0.08,
            "obs_per_1k_range": (3, 5),
            "size_ratio_range": (0.06, 0.11),
            "min_u_shape": 2,
            "u_shape_min_size": (3, 3),
        },
        "1",
    ),
    2: _with_label(
        {
            "types": ["rect", "u_shape"],
            "overlap_prob": 0.12,
            "obs_per_1k_range": (4, 6),
            "size_ratio_range": (0.07, 0.12),
            "min_u_shape": 2,
            "u_shape_min_size": (3, 3),
        },
        "1.5",
    ),
    3: _with_label(
        {
            "types": ["rect", "u_shape"],
            "overlap_prob": 0.18,
            "obs_per_1k_range": (5, 8),
            "size_ratio_range": (0.08, 0.14),
            "min_u_shape": 2,
            "u_shape_min_size": (3, 3),
        },
        "2",
    ),
}


_PROFILE_TABLE: Dict[str, Dict[str, object]] = {
    "legacy4": {
        "profile_name": "legacy4",
        "description": "Original 4-stage random curriculum used for 32x32 experiments.",
        "stages": LEGACY_FOUR_STAGE,
        "default_stages": (1, 2, 3, 4),
    },
    "random64_easymid2": {
        "profile_name": "random64_easymid2",
        "description": "64x64 3-level curriculum with semantics 1 / 1.5 / 2.",
        "stages": RANDOM64_EASYMID2,
        "default_stages": (1, 2, 3),
    },
}


def available_curriculum_profiles() -> Tuple[str, ...]:
    return tuple(sorted(_PROFILE_TABLE.keys()))


def get_curriculum_profile(name: str) -> Dict[str, object]:
    key = str(name).strip().lower()
    if key not in _PROFILE_TABLE:
        raise KeyError(
            f"Unknown curriculum profile: {name}. "
            f"Available: {', '.join(available_curriculum_profiles())}"
        )
    return deepcopy(_PROFILE_TABLE[key])


def default_stages_for_profile(name: str) -> Tuple[int, ...]:
    profile = get_curriculum_profile(name)
    return tuple(int(x) for x in profile["default_stages"])


def stage_difficulty_label(name: str, stage: int) -> str:
    profile = get_curriculum_profile(name)
    stages: MutableMapping[int, StageConfig] = profile["stages"]  # type: ignore[assignment]
    if int(stage) not in stages:
        raise KeyError(f"Stage {stage} is not defined for profile {name}")
    label = stages[int(stage)].get("difficulty_label", str(stage))
    return str(label)


def stage_token(name: str, stage: int) -> str:
    label = stage_difficulty_label(name, stage)
    return label.replace(".", "p")
