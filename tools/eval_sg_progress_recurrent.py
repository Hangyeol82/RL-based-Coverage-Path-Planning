import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if not hasattr(np, '_core'):
    sys.modules.setdefault('numpy._core', np.core)
    sys.modules.setdefault('numpy._core.multiarray', np.core.multiarray)
    sys.modules.setdefault('numpy._core.numeric', np.core.numeric)
    sys.modules.setdefault('numpy._core.umath', np.core.umath)

RUNS = {
    'sg_min1': 'paper_hybrid_edge_tvi03_sgtv005_step0_new05_sgcnn8_l34_100m_baseline_lstm256_penalty3_lr1e4_ent003_rsfix_mask_mid_b512_seed9001',
    'sg_min20': 'paper_hybrid_edge_tvi03_sgtv005min20_step0_new05_sgcnn8_l34_100m_baseline_lstm256_penalty3_lr1e4_ent003_rsfix_mask_mid_b512_seed9001',
}
CHUNKS = {
    'sg_min1': [10, 20, 30, 40, 50],
    'sg_min20': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
}
MAPS = ['map001', 'map002', 'map003']

REPO = Path.cwd()
MAX_STEPS = 8000
CPPDiscreteEnvConfig = None
HybridLocalGlobalCPPObservationConfig = None
MultiScaleCPPObservationConfig = None
CPPRewardConfig = None
RobotStateObservationConfig = None
PaperCPPDiscreteGymEnv = None
RecurrentPPO = None
MaskablePPO = None
PPO = None


def load_repo_imports(repo: Path) -> None:
    global REPO
    global CPPDiscreteEnvConfig, HybridLocalGlobalCPPObservationConfig, MultiScaleCPPObservationConfig
    global CPPRewardConfig, RobotStateObservationConfig, PaperCPPDiscreteGymEnv
    global RecurrentPPO, MaskablePPO, PPO

    REPO = repo.resolve()
    sys.path.insert(0, str(REPO))

    from learning.reinforcement.cpp_env import CPPDiscreteEnvConfig as _CPPDiscreteEnvConfig
    from learning.observation import (
        HybridLocalGlobalCPPObservationConfig as _HybridLocalGlobalCPPObservationConfig,
        MultiScaleCPPObservationConfig as _MultiScaleCPPObservationConfig,
    )
    from learning.reinforcement.reward import CPPRewardConfig as _CPPRewardConfig
    from paper_training.robot_state_observation import RobotStateObservationConfig as _RobotStateObservationConfig
    from paper_training.cpp_env import PaperCPPDiscreteGymEnv as _PaperCPPDiscreteGymEnv
    from stable_baselines3 import PPO as _PPO

    try:
        from sb3_contrib import RecurrentPPO as _RecurrentPPO, MaskablePPO as _MaskablePPO
    except Exception:
        _RecurrentPPO = None
        _MaskablePPO = None

    CPPDiscreteEnvConfig = _CPPDiscreteEnvConfig
    HybridLocalGlobalCPPObservationConfig = _HybridLocalGlobalCPPObservationConfig
    MultiScaleCPPObservationConfig = _MultiScaleCPPObservationConfig
    CPPRewardConfig = _CPPRewardConfig
    RobotStateObservationConfig = _RobotStateObservationConfig
    PaperCPPDiscreteGymEnv = _PaperCPPDiscreteGymEnv
    RecurrentPPO = _RecurrentPPO
    MaskablePPO = _MaskablePPO
    PPO = _PPO


def parse_map(path: Path) -> np.ndarray:
    rows = [list(map(int, line.split())) for line in path.read_text().splitlines() if line.strip()]
    return np.asarray(rows, dtype=np.int32)


def turn_ratio(path):
    if len(path) < 3:
        return 0.0
    turns = 0
    denom = 0
    for a, b, c in zip(path[:-2], path[1:-1], path[2:]):
        d1 = (b[0] - a[0], b[1] - a[1])
        d2 = (c[0] - b[0], c[1] - b[1])
        if d1 == (0, 0) or d2 == (0, 0):
            continue
        denom += 1
        if d1 != d2:
            turns += 1
    return turns / max(1, denom)


def path_metrics(grid, path, cov_hist):
    free = grid == 0
    free_total = int(free.sum())
    visited = np.zeros_like(free, dtype=bool)
    revisit = 0
    threshold_info = {thr: {'success': False, 'step': math.nan, 'overlap': math.nan} for thr in [0.90, 0.95, 0.99]}
    for i, (r, c) in enumerate(path[1:], start=1):
        if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] and free[r, c]:
            if visited[r, c]:
                revisit += 1
            visited[r, c] = True
        cov = float(visited.sum()) / max(1, free_total)
        for thr in threshold_info:
            if (not threshold_info[thr]['success']) and cov >= thr:
                threshold_info[thr] = {'success': True, 'step': i, 'overlap': revisit / max(1, i)}
    steps = max(1, len(path) - 1)
    return {
        'coverage_ratio': float(visited.sum()) / max(1, free_total),
        'coverage_auc': float(np.nanmean(cov_hist)) if cov_hist else math.nan,
        'overlap_ratio': revisit / steps,
        'turn_ratio': turn_ratio(path),
        'success_90': threshold_info[0.90]['success'],
        'success_95': threshold_info[0.95]['success'],
        'success_99': threshold_info[0.99]['success'],
        'step_to_90': threshold_info[0.90]['step'],
        'step_to_95': threshold_info[0.95]['step'],
        'step_to_99': threshold_info[0.99]['step'],
        'overlap_to_90': threshold_info[0.90]['overlap'],
        'overlap_to_95': threshold_info[0.95]['overlap'],
        'overlap_to_99': threshold_info[0.99]['overlap'],
    }


def build_env(grid, semi_min, max_steps):
    reward_cfg = CPPRewardConfig(
        new_cell_reward=0.5,
        collision_reward=-10.0,
        constant_reward=0.0,
        constant_reward_always=False,
        revisit_penalty=-0.2,
        local_tv_reward_scale=0.3,
        local_tv_normalizer=1.0,
        global_tv_reward_scale=0.0,
        semi_global_tv_reward_scale=0.05,
        semi_global_tv_horizon=32,
        semi_global_tv_clip=2.0,
        semi_global_tv_window_size=41,
        semi_global_tv_min_new_cells=semi_min,
        tv_obstacle_edge_weight=0.0,
        turn_change_penalty=-0.05,
    )
    cfg = CPPDiscreteEnvConfig(
        sensor_range=3,
        local_tv_mode='edge',
        local_tv_patch_radius=3,
        max_steps=max_steps,
        collision_ends_episode=False,
        stop_on_full_coverage=True,
        include_dtm=False,
        action_mask_mode='penalty',
        invalid_action_penalty=-3.0,
        maps_observation_mode='hybrid_local_global',
        observation=MultiScaleCPPObservationConfig(
            unknown_policy='keep', dtm_coarse_mode='bfs', dtm_output_mode='six', dtm_connectivity=4,
            include_cell_phase_channels=False,
        ),
        hybrid_observation=HybridLocalGlobalCPPObservationConfig(
            local_crop_size=41,
            global_coarse_size=16,
            global_coarse_sizes=(16,12,8,4),
            global_view_mode='viewport',
            global_window_sizes=(64,96,128,128),
            unknown_policy='keep',
            dtm_connectivity=4,
            dtm_output_mode='six',
        ),
        robot_state=RobotStateObservationConfig(
            action_history_len=10,
            include_position=True,
            include_action_history=True,
            include_action_mask=True,
            include_progress=False,
            include_stagnation=False,
        ),
        use_cell_phase_features=False,
        use_action_mask=True,
        heuristic_assist_enabled=False,
        reward=reward_cfg,
    )
    return PaperCPPDiscreteGymEnv(grid_map=grid, start_pos=(0,0), config=cfg)


def load_model(model_path, env, device='cuda'):
    custom_objects = {
        'action_space': env.action_space,
        'observation_space': env.observation_space,
        '_last_obs': None,
        '_last_original_obs': None,
        '_last_episode_starts': None,
        'ep_info_buffer': [],
        'ep_success_buffer': [],
    }
    errors=[]
    if RecurrentPPO is not None:
        try:
            return RecurrentPPO.load(str(model_path), env=env, device=device, custom_objects=custom_objects), 'RecurrentPPO'
        except Exception as e:
            errors.append(f'RecurrentPPO: {e}')
    try:
        return PPO.load(str(model_path), env=env, device=device, custom_objects=custom_objects), 'PPO'
    except Exception as e:
        errors.append(f'PPO: {e}')
    if MaskablePPO is not None:
        try:
            return MaskablePPO.load(str(model_path), env=env, device=device, custom_objects=custom_objects), 'MaskablePPO'
        except Exception as e:
            errors.append(f'MaskablePPO: {e}')
    raise RuntimeError('model load failed: ' + ' | '.join(errors))


def rollout(model, algo, env, max_steps):
    obs, _ = env.reset()
    state = None
    episode_start = np.ones((1,), dtype=bool)
    path = [(0,0)]
    cov_hist=[]
    total_reward=0.0
    invalid_count=0
    collision_count=0
    final_info={}
    for _ in range(max_steps):
        if algo == 'RecurrentPPO':
            action, state = model.predict(obs, state=state, episode_start=episode_start, deterministic=True)
            episode_start[:] = False
        else:
            action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        total_reward += float(reward)
        final_info=dict(info)
        cov_hist.append(float(info.get('coverage_ratio', np.nan)))
        if float(info.get('invalid_action_penalty', 0.0)) < 0:
            invalid_count += 1
        if bool(info.get('collision', False)):
            collision_count += 1
        pos=info.get('position', path[-1])
        path.append((int(pos[0]), int(pos[1])))
        if terminated or truncated:
            break
    m=path_metrics(env.core_env.true_map, path, cov_hist)
    steps=max(1, len(path)-1)
    m.update({
        'steps': steps,
        'total_reward': total_reward,
        'invalid_rate': invalid_count/steps,
        'collision_rate': collision_count/steps,
        'done_reason': str(final_info.get('done_reason','')),
        'path': [[int(r), int(c)] for r,c in path],
    })
    return m


def save_plot(grid, path, out_png, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p=np.asarray(path, dtype=np.int32)
    fig, ax = plt.subplots(figsize=(7,7))
    ax.imshow(grid == 1, cmap='gray_r', origin='upper')
    if len(p)>1:
        ax.plot(p[:,1], p[:,0], color='tab:red', linewidth=1.0)
    ax.scatter(p[0,1], p[0,0], color='tab:blue', s=28)
    ax.scatter(p[-1,1], p[-1,0], color='tab:green', s=28)
    ax.set_title(title, fontsize=10)
    ax.set_xlim(-0.5, grid.shape[1]-0.5); ax.set_ylim(grid.shape[0]-0.5, -0.5)
    fig.tight_layout(); out_png.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out_png, dpi=130); plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate recurrent PPO checkpoints every N chunks on fixed maps."
    )
    parser.add_argument("--repo", type=Path, default=Path.cwd(), help="Repository root.")
    parser.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Output directory. Defaults to <repo>/log_analysis/logs/semi_global_progress_eval_tvi03.",
    )
    parser.add_argument("--max-steps", type=int, default=8000, help="Max rollout steps per map.")
    parser.add_argument("--device", default="cuda", help="Model device for SB3 load/predict.")
    return parser.parse_args()


def main():
    args = parse_args()
    repo = args.repo.resolve()
    load_repo_imports(repo)
    out_root = args.out_root or repo / 'log_analysis' / 'logs' / 'semi_global_progress_eval_tvi03'
    out_root.mkdir(parents=True, exist_ok=True)
    rows=[]
    for label, tag in RUNS.items():
        semi_min = 1 if label == 'sg_min1' else 20
        run_dir = repo/'learning/checkpoints/rl'/f'mixed_curriculum_{tag}'
        map_paths=[]
        for name in MAPS:
            found=sorted((run_dir/'maps/chunk01_pool').glob(f'{name}_*.txt'))
            if not found:
                raise FileNotFoundError(f'map not found {label} {name}')
            map_paths.append(found[0])
        for chunk in CHUNKS[label]:
            model_path=run_dir/'models'/f'chunk{chunk:02d}.zip'
            if not model_path.exists():
                print(f'[SKIP] missing {model_path}', flush=True); continue
            for map_path in map_paths:
                map_id=map_path.name.split('_')[0]
                out_dir=out_root/label/f'chunk{chunk:03d}'
                out_json=out_dir/f'{map_id}.json'
                out_png=out_dir/f'{map_id}.png'
                if out_json.exists():
                    data=json.loads(out_json.read_text())
                    rows.append({k:v for k,v in data.items() if k != 'path'})
                    continue
                grid=parse_map(map_path)
                env=build_env(grid, semi_min, args.max_steps)
                try:
                    model, algo=load_model(model_path, env, device=args.device)
                    res=rollout(model, algo, env, args.max_steps)
                    res.update({'run':label,'tag':tag,'chunk':chunk,'map_id':map_id,'map_file':str(map_path),'model':str(model_path),'algo':algo})
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_json.write_text(json.dumps(res, indent=2), encoding='utf-8')
                    save_plot(grid, res['path'], out_png, f'{label} chunk{chunk:03d} {map_id}')
                    rows.append({k:v for k,v in res.items() if k != 'path'})
                    print(f"[OK] {label} chunk{chunk:03d} {map_id} cov={res['coverage_ratio']:.4f} ov={res['overlap_ratio']:.4f} ov99={res['overlap_to_99']}", flush=True)
                finally:
                    env.close()
    df=pd.DataFrame(rows)
    df.to_csv(out_root/'summary_per_map.csv', index=False)
    summary=(df.groupby(['run','chunk'], as_index=False)
        .agg(coverage_ratio=('coverage_ratio','mean'), coverage_auc=('coverage_auc','mean'),
             success_90=('success_90','mean'), success_95=('success_95','mean'), success_99=('success_99','mean'),
             overlap_to_90=('overlap_to_90','mean'), overlap_to_95=('overlap_to_95','mean'), overlap_to_99=('overlap_to_99','mean'),
             overlap_ratio=('overlap_ratio','mean'), turn_ratio=('turn_ratio','mean'), invalid_rate=('invalid_rate','mean'), steps=('steps','mean')))
    summary.to_csv(out_root/'summary_by_chunk.csv', index=False)
    print('\n=== SUMMARY BY CHUNK ===')
    print(summary.to_string(index=False))

if __name__ == '__main__':
    main()
