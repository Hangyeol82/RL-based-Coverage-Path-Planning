import sys

from run_ppo_shapegrid_curriculum_paper import main


def _has_flag(flag: str) -> bool:
    return any(arg == flag or arg.startswith(flag + "=") for arg in sys.argv[1:])


def _has_any(flags: tuple[str, ...]) -> bool:
    return any(_has_flag(flag) for flag in flags)


def _ensure_arg(flag: str, value: str) -> None:
    if not _has_flag(flag):
        sys.argv.extend([flag, value])


def _ensure_switch(flag: str, *, conflicts: tuple[str, ...] = ()) -> None:
    if not _has_any((flag, *conflicts)):
        sys.argv.append(flag)


if __name__ == "__main__":
    _ensure_arg("--generator-curriculum", "mixed_paper")
    _ensure_arg("--total-timesteps", "50000000")
    _ensure_arg("--chunk-timesteps", "100000")
    _ensure_arg("--map-size", "128")
    _ensure_arg("--maps-per-chunk", "12")
    _ensure_arg("--phase-timesteps", "5000000,10000000,15000000,20000000")
    _ensure_arg("--phase-level-probs", "1:1.0;1:0.2,2:0.8;2:0.2,3:0.8;3:0.2,4:0.8")
    _ensure_arg("--family-weights", "object:1,trail_grid:1,room_corridor:1")
    _ensure_arg("--object-generator-weights", "shape_grid:1,macro_detail:1")
    _ensure_arg("--model-size", "xlarge")
    _ensure_arg("--map-refresh-mode", "cycle")
    _ensure_arg("--curriculum-mode", "fixed")
    _ensure_arg("--init-from-bc", "")
    _ensure_switch("--no-robot-state-position", conflicts=("--robot-state-position",))
    main()
