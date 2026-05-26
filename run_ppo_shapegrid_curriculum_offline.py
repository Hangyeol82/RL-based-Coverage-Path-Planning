import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Tuple


REPO_ROOT = Path(__file__).resolve().parent


def _parse_args() -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Run shape-grid curriculum PPO in offline/full-map mode. "
            "This wraps run_ppo_shapegrid_curriculum_paper.py, enables "
            "--full-map-observation, and can run baseline and DTM variants "
            "with identical curriculum/hyperparameter settings."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument(
        "--variant",
        choices=["baseline", "dtm", "both"],
        default="both",
        help="Run offline baseline, offline DTM, or both sequentially.",
    )
    parser.add_argument(
        "--run-tag-base",
        type=str,
        default="shapegrid_offline",
        help=(
            "Base run tag. The launcher appends _baseline or _dtm unless "
            "only one variant is selected."
        ),
    )
    parser.add_argument(
        "--runner",
        type=str,
        default=str(REPO_ROOT / "run_ppo_shapegrid_curriculum_paper.py"),
        help="Path to run_ppo_shapegrid_curriculum_paper.py.",
    )
    args, passthrough = parser.parse_known_args()
    return args, list(passthrough)


def _has_flag(args: Iterable[str], flag: str) -> bool:
    return any(str(x) == flag or str(x).startswith(flag + "=") for x in args)


def _pop_option(args: List[str], option: str) -> Tuple[List[str], str]:
    out: List[str] = []
    value = ""
    i = 0
    while i < len(args):
        item = str(args[i])
        if item == option:
            if i + 1 >= len(args):
                raise ValueError(f"{option} requires a value")
            value = str(args[i + 1])
            i += 2
            continue
        if item.startswith(option + "="):
            value = item.split("=", 1)[1]
            i += 1
            continue
        out.append(item)
        i += 1
    return out, value


def _variant_tag(base_tag: str, variant: str, variants: List[str]) -> str:
    base = str(base_tag).strip() or "shapegrid_offline"
    if len(variants) == 1:
        return base
    return f"{base}_{variant}"


def _variant_path(raw: str, variant: str, variants: List[str]) -> str:
    if len(variants) == 1:
        return raw
    p = Path(raw)
    name = f"{p.name}_{variant}" if p.name else variant
    return str(p.with_name(name))


def _run_variant(args: argparse.Namespace, passthrough: List[str], variant: str, variants: List[str]):
    if _has_flag(passthrough, "--include-dtm"):
        raise ValueError("Do not pass --include-dtm directly; use --variant dtm or --variant both.")
    if _has_flag(passthrough, "--full-map-observation"):
        raise ValueError("This launcher always enables --full-map-observation.")

    cleaned, passthrough_tag = _pop_option(passthrough, "--run-tag")
    cleaned, passthrough_out_dir = _pop_option(cleaned, "--out-dir")
    base_tag = passthrough_tag or str(args.run_tag_base)
    run_tag = _variant_tag(base_tag, variant, variants)

    cmd = [
        str(args.python_bin),
        str(args.runner),
        "--python-bin",
        str(args.python_bin),
        "--full-map-observation",
        "--run-tag",
        run_tag,
    ]
    if variant == "dtm":
        cmd.append("--include-dtm")
    if passthrough_out_dir:
        cmd += ["--out-dir", _variant_path(passthrough_out_dir, variant, variants)]
    cmd += cleaned

    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def main():
    args, passthrough = _parse_args()
    variants = ["baseline", "dtm"] if args.variant == "both" else [str(args.variant)]
    for variant in variants:
        _run_variant(args, passthrough, variant, variants)


if __name__ == "__main__":
    main()
