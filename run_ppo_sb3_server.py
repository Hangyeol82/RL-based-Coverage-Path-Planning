import argparse
import subprocess
import sys
from typing import List


def _parse_args():
    p = argparse.ArgumentParser(
        description="Server-oriented PPO launcher (multiprocessing defaults for 6-core desktop).",
    )
    p.add_argument("--total-timesteps", type=int, default=50000)
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--num-envs", type=int, default=6)
    p.add_argument("--vec-env", type=str, default="subproc", choices=["auto", "dummy", "subproc"])
    p.add_argument(
        "--subproc-start-method",
        type=str,
        default="spawn",
        choices=["auto", "spawn", "fork", "forkserver"],
    )

    p.add_argument("--map-source", type=str, default="file", choices=["random", "custom", "file"])
    p.add_argument("--map-file", type=str, default="map/indoor_seed101.txt")
    p.add_argument("--map-size", type=int, default=64)
    p.add_argument("--map-stage", type=int, default=3, choices=[1, 2, 3, 4])
    p.add_argument("--sensor-range", type=int, default=2)
    p.add_argument("--max-episode-steps", type=int, default=2000)

    p.add_argument("--n-steps", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--n-epochs", type=int, default=4)

    p.add_argument("--include-dtm", action="store_true")
    p.add_argument(
        "--init-from-bc",
        type=str,
        default="",
        help="Warm-start PPO feature encoder from BC checkpoint (.pt).",
    )
    p.add_argument("--init-from-bc-strict", action="store_true")
    p.add_argument("--save-model", type=str, default="learning/checkpoints/rl/ppo_sb3_server_latest")
    p.add_argument(
        "--save-breakdown-json",
        type=str,
        default="learning/checkpoints/rl/logs/ppo_breakdown_server.json",
    )
    p.add_argument(
        "--save-breakdown-csv",
        type=str,
        default="learning/checkpoints/rl/logs/ppo_breakdown_server.csv",
    )

    return p.parse_known_args()


def _build_cmd(args: argparse.Namespace, extra: List[str]) -> List[str]:
    cmd = [
        sys.executable,
        "run_ppo_sb3.py",
        "--total-timesteps",
        str(args.total_timesteps),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--num-envs",
        str(args.num_envs),
        "--vec-env",
        args.vec_env,
        "--subproc-start-method",
        args.subproc_start_method,
        "--map-source",
        args.map_source,
        "--map-size",
        str(args.map_size),
        "--map-stage",
        str(args.map_stage),
        "--sensor-range",
        str(args.sensor_range),
        "--max-episode-steps",
        str(args.max_episode_steps),
        "--n-steps",
        str(args.n_steps),
        "--batch-size",
        str(args.batch_size),
        "--n-epochs",
        str(args.n_epochs),
        "--save-model",
        args.save_model,
        "--save-breakdown-json",
        args.save_breakdown_json,
        "--save-breakdown-csv",
        args.save_breakdown_csv,
    ]
    if args.map_source == "file":
        cmd += ["--map-file", args.map_file]
    if args.include_dtm:
        cmd += ["--include-dtm"]
    if args.init_from_bc:
        cmd += ["--init-from-bc", args.init_from_bc]
    if args.init_from_bc_strict:
        cmd += ["--init-from-bc-strict"]
    cmd += extra
    return cmd


def main():
    args, extra = _parse_args()
    cmd = _build_cmd(args, extra)
    print("Launching:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
