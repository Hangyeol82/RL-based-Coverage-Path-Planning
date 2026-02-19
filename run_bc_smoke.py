import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from learning.imitation import BCPolicy, BCPolicyConfig
from learning.imitation.epsilon_rollout import (
    BCTensorDataset,
    build_bc_tensors_from_rollout,
    collect_epsilon_rollout,
)
from run_cstar_custom_map import CUSTOM_MAP_TEXT, parse_custom_map


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BC smoke test using E* rollout on a single fixed map.",
    )
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--epsilon", type=float, default=0.5)
    parser.add_argument("--sensor-range", type=int, default=5)
    parser.add_argument("--action-dim", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--save-dir", type=str, default="learning/checkpoints/bc")
    parser.add_argument("--save-name", type=str, default="bc_smoke_latest.pt")
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

    grid = parse_custom_map(CUSTOM_MAP_TEXT)
    if grid[0, 0] == 1:
        grid[0, 0] = 0

    transitions, teacher_path = collect_epsilon_rollout(
        grid_map=grid,
        start_node=(0, 0),
        epsilon=args.epsilon,
        sensor_range=args.sensor_range,
        max_steps=args.max_steps,
    )
    dataset = build_bc_tensors_from_rollout(transitions)
    if dataset.size == 0:
        raise RuntimeError("Collected zero transitions from E* rollout")

    policy = BCPolicy(BCPolicyConfig(action_dim=args.action_dim)).to(device)
    optim = torch.optim.Adam(policy.parameters(), lr=args.lr)

    print(f"Device: {device}")
    print(f"Teacher path length: {len(teacher_path)}")
    print(f"Transitions: {dataset.size}")
    for lv in range(6):
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
    print("If accuracy is very high on this single map rollout, model wiring is likely correct.")

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
            },
        }
        torch.save(checkpoint, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
