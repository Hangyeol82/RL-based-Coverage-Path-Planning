import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List


REPO_ROOT = Path(__file__).resolve().parent

# Server CPU-affinity note for long paper runs:
# - lscpu/nvidia-smi topo on the current server showed GPU0/GPU1 affinity to
#   NUMA node0: logical CPUs 0-27,56-83.
# - Safe first scaling step: launch with `taskset -c 0-27` and `--num-envs 28`.
# - If rollout collection remains CPU-bound, benchmark `taskset -c 0-27,56-83`
#   with `--num-envs 40` before trying all 56 logical CPUs.
# - Keep `taskset` CPU count and `--num-envs` aligned; changing `--num-envs`
#   also changes PPO rollout batch size (`num_envs * n_steps`).


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Paper-training preset for shape-grid online CPP PPO experiments. "
            "This wrapper fixes the observation/reward cleanup choices used for "
            "main paper runs and forwards only experiment-scale parameters."
        )
    )
    p.add_argument("--python-bin", type=str, default=sys.executable)
    p.add_argument(
        "--variant",
        type=str,
        default="dtm-two",
        choices=[
            "baseline",
            "dtm-two",
            "dtm-axis2",
            "dtm-six",
            "dtm-two-hole-obs",
            "dtm-axis2-hole-obs",
            "dtm-two-hole-reward",
            "dtm-axis2-hole-reward",
            "dtm-two-hole-full",
            "dtm-axis2-hole-full",
            "dtm-six-hole-full",
        ],
        help="Main/ablation paper variants.",
    )
    p.add_argument(
        "--coverage-hole-penalty-scale",
        type=float,
        default=-1.0,
        help="Override hole penalty scale. Negative uses the variant default.",
    )
    p.add_argument("--total-timesteps", type=int, default=20_000_000)
    p.add_argument("--chunk-timesteps", type=int, default=500_000)
    p.add_argument("--stage-timesteps", type=int, default=5_000_000)
    p.add_argument("--phase-levels", type=str, default="2,3,4,4")
    p.add_argument("--curriculum-mode", type=str, default="fixed", choices=["fixed", "adaptive"])
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--map-seed-mode", type=str, default="chunk", choices=["fixed", "chunk"])
    p.add_argument("--map-size", type=int, default=128)
    p.add_argument("--maps-per-chunk", type=int, default=1)
    p.add_argument("--map-refresh-mode", type=str, default="cycle", choices=["cycle", "random"])
    p.add_argument("--episode-success-threshold", type=float, default=1.0)
    p.add_argument("--sensor-range", type=int, default=3)
    p.add_argument("--max-episode-steps", type=int, default=30_000)
    p.add_argument("--device", type=str, default="cuda", choices=["auto", "cpu", "cuda"])
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--vec-env", type=str, default="subproc", choices=["auto", "dummy", "subproc"])
    p.add_argument(
        "--subproc-start-method",
        type=str,
        default="forkserver",
        choices=["auto", "spawn", "fork", "forkserver"],
    )
    p.add_argument("--n-steps", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--n-epochs", type=int, default=4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--maps-encoder-mode", type=str, default="sgcnn", choices=["sgcnn", "independent"])
    p.add_argument("--model-size", type=str, default="xlarge", choices=["small", "large", "xlarge"])
    p.add_argument("--dtm-coarse-mode", type=str, default="bfs", choices=["bfs", "aggregate", "aggregate_transfer"])
    p.add_argument("--dtm-connectivity", type=int, default=4, choices=[4, 8])
    p.add_argument("--obs-unknown-policy", type=str, default="keep", choices=["keep", "as_free", "as_obstacle"])
    p.add_argument("--metric-stagnation-threshold", type=int, default=30)
    p.add_argument("--metric-loop-window", type=int, default=12)
    p.add_argument("--run-tag", type=str, default="")
    p.add_argument("--out-dir", type=str, default="")
    p.add_argument("--init-from-bc", type=str, default="", help="Default is random init for clean paper runs.")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Print the resolved command without running it.")
    return p.parse_args()


def _default_tag(args: argparse.Namespace) -> str:
    tt_m = int(args.total_timesteps) // 1_000_000
    return (
        f"paper_shapegrid{int(args.map_size)}_{tt_m}m_"
        f"{args.variant}_{args.model_size}_sensor{int(args.sensor_range)}"
        f"_maps{int(args.maps_per_chunk)}"
    )


def _build_cmd(args: argparse.Namespace) -> List[str]:
    runner = REPO_ROOT / "run_ppo_shapegrid_curriculum_paper.py"
    hole_signals = False
    hole_penalty_scale = 0.0
    if args.variant == "baseline":
        include_dtm = False
        dtm_output_mode = "axis2"
    elif args.variant == "dtm-two":
        include_dtm = True
        dtm_output_mode = "two"
    elif args.variant == "dtm-axis2":
        include_dtm = True
        dtm_output_mode = "axis2"
    elif args.variant == "dtm-six":
        include_dtm = True
        dtm_output_mode = "six"
    elif args.variant == "dtm-two-hole-obs":
        include_dtm = True
        dtm_output_mode = "two"
        hole_signals = True
    elif args.variant == "dtm-axis2-hole-obs":
        include_dtm = True
        dtm_output_mode = "axis2"
        hole_signals = True
    elif args.variant == "dtm-two-hole-reward":
        include_dtm = True
        dtm_output_mode = "two"
        hole_penalty_scale = 0.1
    elif args.variant == "dtm-axis2-hole-reward":
        include_dtm = True
        dtm_output_mode = "axis2"
        hole_penalty_scale = 0.1
    elif args.variant == "dtm-two-hole-full":
        include_dtm = True
        dtm_output_mode = "two"
        hole_signals = True
        hole_penalty_scale = 0.1
    elif args.variant == "dtm-axis2-hole-full":
        include_dtm = True
        dtm_output_mode = "axis2"
        hole_signals = True
        hole_penalty_scale = 0.1
    elif args.variant == "dtm-six-hole-full":
        include_dtm = True
        dtm_output_mode = "six"
        hole_signals = True
        hole_penalty_scale = 0.1
    else:
        raise ValueError(f"Unsupported variant: {args.variant}")
    if float(args.coverage_hole_penalty_scale) >= 0.0:
        hole_penalty_scale = float(args.coverage_hole_penalty_scale)

    cmd = [
        str(args.python_bin),
        str(runner),
        "--total-timesteps",
        str(int(args.total_timesteps)),
        "--chunk-timesteps",
        str(int(args.chunk_timesteps)),
        "--stage-timesteps",
        str(int(args.stage_timesteps)),
        "--phase-levels",
        str(args.phase_levels),
        "--curriculum-mode",
        str(args.curriculum_mode),
        "--seed",
        str(int(args.seed)),
        "--map-seed-mode",
        str(args.map_seed_mode),
        "--map-size",
        str(int(args.map_size)),
        "--maps-per-chunk",
        str(int(args.maps_per_chunk)),
        "--map-refresh-mode",
        str(args.map_refresh_mode),
        "--episode-success-threshold",
        str(float(args.episode_success_threshold)),
        "--sensor-range",
        str(int(args.sensor_range)),
        "--max-episode-steps",
        str(int(args.max_episode_steps)),
        "--device",
        str(args.device),
        "--num-envs",
        str(int(args.num_envs)),
        "--vec-env",
        str(args.vec_env),
        "--subproc-start-method",
        str(args.subproc_start_method),
        "--n-steps",
        str(int(args.n_steps)),
        "--batch-size",
        str(int(args.batch_size)),
        "--n-epochs",
        str(int(args.n_epochs)),
        "--gamma",
        str(float(args.gamma)),
        "--gae-lambda",
        str(float(args.gae_lambda)),
        "--clip-range",
        str(float(args.clip_range)),
        "--ent-coef",
        str(float(args.ent_coef)),
        "--vf-coef",
        str(float(args.vf_coef)),
        "--lr",
        str(float(args.lr)),
        "--maps-encoder-mode",
        str(args.maps_encoder_mode),
        "--model-size",
        str(args.model_size),
        "--dtm-coarse-mode",
        str(args.dtm_coarse_mode),
        "--dtm-output-mode",
        dtm_output_mode,
        "--dtm-connectivity",
        str(int(args.dtm_connectivity)),
        "--obs-unknown-policy",
        str(args.obs_unknown_policy),
        "--cell-phase-channels",
        "--no-boundary-exit-features",
        "--no-milestone-reward",
        "--no-robot-state-position",
        "--robot-state-action-history",
        "--no-robot-state-progress",
        "--no-robot-state-stagnation",
        "--robot-state-action-history-len",
        "5",
        "--coverage-hole-penalty-scale",
        str(float(hole_penalty_scale)),
        "--metric-stagnation-threshold",
        str(int(args.metric_stagnation_threshold)),
        "--metric-loop-window",
        str(int(args.metric_loop_window)),
        "--action-mask",
        "--init-from-bc",
        str(args.init_from_bc),
        "--run-tag",
        args.run_tag.strip() or _default_tag(args),
    ]
    if include_dtm:
        cmd.append("--include-dtm")
    if hole_signals:
        cmd.append("--hole-signals")
    else:
        cmd.append("--no-hole-signals")
    if args.out_dir:
        cmd += ["--out-dir", str(args.out_dir)]
    if args.overwrite:
        cmd.append("--overwrite")
    return cmd


def main() -> int:
    args = _parse_args()
    cmd = _build_cmd(args)
    print("Paper training command:")
    print(shlex.join(cmd), flush=True)
    if args.dry_run:
        return 0
    return int(subprocess.run(cmd, check=False).returncode)


if __name__ == "__main__":
    raise SystemExit(main())
