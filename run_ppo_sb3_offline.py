import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List


REPO_ROOT = Path(__file__).resolve().parent


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run offline/full-map PPO variants. This is a thin launcher around "
            "run_ppo_sb3_paper.py with --full-map-observation enabled."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "--variant",
        choices=["baseline", "dtm", "both"],
        default="both",
        help="offline baseline, offline DTM, or both sequentially.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="offline",
        help="Output tag used when default save paths are generated.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="learning/checkpoints/rl",
        help="Root directory for default offline run outputs.",
    )
    parser.add_argument(
        "--paper-script",
        type=str,
        default=str(REPO_ROOT / "run_ppo_sb3_paper.py"),
        help="Path to the paper PPO training script.",
    )
    args, passthrough = parser.parse_known_args()
    return args, passthrough


def _has_flag(args: Iterable[str], flag: str) -> bool:
    return any(str(x) == flag or str(x).startswith(flag + "=") for x in args)


def _run_variant(args, passthrough: List[str], variant: str):
    if "--include-dtm" in passthrough:
        raise ValueError("Do not pass --include-dtm directly; use --variant dtm or --variant both.")
    if "--full-map-observation" in passthrough:
        raise ValueError("run_ppo_sb3_offline.py always enables --full-map-observation.")

    output_dir = Path(args.output_root) / f"{args.tag}_{variant}"
    cmd = [
        sys.executable,
        str(args.paper_script),
        "--full-map-observation",
    ]
    if variant == "dtm":
        cmd.append("--include-dtm")

    if not _has_flag(passthrough, "--save-model"):
        cmd += ["--save-model", str(output_dir / "model")]
    if not _has_flag(passthrough, "--save-breakdown-json"):
        cmd += ["--save-breakdown-json", str(output_dir / "logs" / "breakdown.json")]
    if not _has_flag(passthrough, "--save-breakdown-csv"):
        cmd += ["--save-breakdown-csv", str(output_dir / "logs" / "breakdown.csv")]

    cmd += list(passthrough)
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def main():
    args, passthrough = _parse_args()
    variants = ["baseline", "dtm"] if args.variant == "both" else [args.variant]
    for variant in variants:
        _run_variant(args, passthrough, variant)


if __name__ == "__main__":
    main()
