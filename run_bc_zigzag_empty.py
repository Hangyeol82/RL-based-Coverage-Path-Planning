import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from learning.common import (
    FusedMAPSStateEncoderConfig,
    MultiLevelMAPSEncoderConfig,
    RobotStateEncoderConfig,
)
from learning.imitation import BCPolicy, BCPolicyConfig
from learning.imitation.epsilon_rollout import (
    BCTensorDataset,
    build_bc_tensors_from_rollout,
    collect_serpentine_rollout,
)
from learning.observation import (
    MultiScaleCPPObservationBuilder,
    MultiScaleCPPObservationConfig,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train BC on deterministic zigzag teacher rollout over an empty map.",
    )
    parser.add_argument("--map-size", type=int, default=32)
    parser.add_argument("--sensor-range", type=int, default=2, help="2 -> 5x5 sensing")
    parser.add_argument("--sweep-axis", type=str, default="row", choices=["row", "col"])
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="0 means full serpentine coverage path length.",
    )

    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--action-dim", type=int, default=4)
    parser.add_argument("--maps-encoder-mode", type=str, default="sgcnn", choices=["sgcnn", "independent"])
    parser.add_argument("--include-dtm", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    parser.add_argument("--save-dir", type=str, default="learning/checkpoints/bc")
    parser.add_argument("--save-name", type=str, default="bc_zigzag_empty_latest.pt")
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def _select_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _batch_slice_levels(levels: Dict[int, torch.Tensor], idx: torch.Tensor) -> Dict[int, torch.Tensor]:
    return {lv: x[idx] for lv, x in levels.items()}


def _evaluate_accuracy(
    policy: BCPolicy,
    dataset: BCTensorDataset,
    device: torch.device,
) -> Tuple[float, float]:
    policy.eval()
    with torch.no_grad():
        levels = {lv: x.to(device) for lv, x in dataset.levels.items()}
        state = dataset.robot_state.to(device)
        actions = dataset.actions.to(device)
        logits = policy(levels, state)
        loss = torch.nn.functional.cross_entropy(logits, actions).item()
        pred = torch.argmax(logits, dim=1)
        acc = (pred == actions).float().mean().item()
    return float(loss), float(acc)


def main():
    args = _parse_args()
    _set_seed(args.seed)
    device = _select_device(args.device)

    if args.map_size <= 1:
        raise ValueError("--map-size must be >= 2")
    grid = np.zeros((args.map_size, args.map_size), dtype=np.int32)
    max_steps = None if int(args.max_steps) <= 0 else int(args.max_steps)

    transitions, teacher_path = collect_serpentine_rollout(
        grid_map=grid,
        start_node=(0, 0),
        sensor_range=int(args.sensor_range),
        sweep_axis=args.sweep_axis,
        max_steps=max_steps,
        strict_empty=True,
    )
    obs_cfg = MultiScaleCPPObservationConfig()
    maps_builder = MultiScaleCPPObservationBuilder(obs_cfg, include_dtm=bool(args.include_dtm))
    dataset = build_bc_tensors_from_rollout(transitions, maps_builder=maps_builder)
    if dataset.size == 0:
        raise RuntimeError("Collected zero transitions from zigzag teacher")

    enc_cfg = FusedMAPSStateEncoderConfig(
        maps=MultiLevelMAPSEncoderConfig(
            num_levels=len(dataset.levels),
            in_channels_per_level=int(dataset.levels[0].shape[1]),
            conv_channels=(16, 32),
            level_embed_dim=64,
            mode=args.maps_encoder_mode,
        ),
        robot_state=RobotStateEncoderConfig(input_dim=int(dataset.robot_state.shape[1]), hidden_dims=(64, 64)),
        fusion_hidden_dims=(256, 256),
    )
    policy = BCPolicy(BCPolicyConfig(action_dim=args.action_dim, encoder=enc_cfg)).to(device)
    optim = torch.optim.Adam(policy.parameters(), lr=args.lr)

    print(f"Device: {device}")
    print(f"Teacher: serpentine({args.sweep_axis}) on empty {args.map_size}x{args.map_size}")
    print(f"Observation mode: {'dtm' if args.include_dtm else 'baseline'}")
    print(f"Teacher path length: {len(teacher_path)}")
    print(f"Transitions: {dataset.size}")
    for lv in sorted(dataset.levels.keys()):
        print(f"Level {lv} tensor shape: {tuple(dataset.levels[lv].shape)}")
    print(f"Robot state shape: {tuple(dataset.robot_state.shape)}")
    print(f"Action shape: {tuple(dataset.actions.shape)}")

    n = dataset.size
    for epoch in range(1, args.epochs + 1):
        policy.train()
        perm = torch.randperm(n)
        total_loss = 0.0

        for start in range(0, n, args.batch_size):
            idx = perm[start : start + args.batch_size]
            levels_b = _batch_slice_levels(dataset.levels, idx)
            levels_b = {lv: x.to(device) for lv, x in levels_b.items()}
            state_b = dataset.robot_state[idx].to(device)
            action_b = dataset.actions[idx].to(device)

            optim.zero_grad(set_to_none=True)
            loss = policy.loss(levels_b, state_b, action_b)
            loss.backward()
            optim.step()
            total_loss += float(loss.item()) * int(action_b.shape[0])

        if epoch == 1 or (epoch % args.log_every == 0) or epoch == args.epochs:
            train_loss, train_acc = _evaluate_accuracy(policy, dataset, device)
            mean_loss = total_loss / float(n)
            print(
                f"[Epoch {epoch:03d}] minibatch_loss={mean_loss:.6f} "
                f"full_loss={train_loss:.6f} acc={train_acc * 100.0:.2f}%"
            )

    final_loss, final_acc = _evaluate_accuracy(policy, dataset, device)
    print("Done.")
    print(f"Final train loss: {final_loss:.6f}")
    print(f"Final train accuracy: {final_acc * 100.0:.2f}%")

    if not args.no_save:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = save_dir / args.save_name
        checkpoint = {
            "model_state_dict": policy.state_dict(),
            "optimizer_state_dict": optim.state_dict(),
            "config": vars(args),
            "metrics": {
                "final_train_loss": final_loss,
                "final_train_accuracy": final_acc,
                "num_transitions": dataset.size,
                "teacher_path_length": len(teacher_path),
                "teacher_kind": "serpentine_empty",
            },
        }
        torch.save(checkpoint, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
