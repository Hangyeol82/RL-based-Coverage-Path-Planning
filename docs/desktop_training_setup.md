# Desktop Training Setup (i5-10400F + GTX 1060 6GB)

This project now includes a desktop profile for SB3 PPO training.

## 1) On your desktop: clone and enter repo

```bash
git clone <your-repo-url>
cd "RL based online CPP"
```

## 2) Create training environment

```bash
bash tools/setup_desktop_rl_env.sh rlcpp
```

If you want CPU-only:

```bash
USE_GPU=0 bash tools/setup_desktop_rl_env.sh rlcpp
```

## 3) Activate env and run baseline vs DTM

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rlcpp

PYTHON=$(which python) \
TOTAL_TIMESTEPS=50000 \
MAP_FILE=map/indoor_seed101.txt \
bash tools/run_desktop_baseline_vs_dtm.sh
```

Or run server-oriented multiprocessing launcher directly:

```bash
python run_ppo_sb3_server.py --total-timesteps 50000
python run_ppo_sb3_server.py --total-timesteps 50000 --include-dtm \
  --save-model learning/checkpoints/rl/ppo_sb3_server_dtm
```

## 3.1) 10-map benchmark (baseline vs DTM)

If `map/manifest.json` contains your 10 maps (5 indoor + 5 random), run:

```bash
python run_ppo_sb3_benchmark10.py \
  --manifest map/manifest.json \
  --groups indoor,random \
  --total-timesteps 50000 \
  --num-envs 6 \
  --vec-env subproc \
  --subproc-start-method spawn \
  --seed-mode map
```

Outputs:

- `learning/checkpoints/rl/benchmark10_<timestamp>/reports/per_map_comparison.csv`
- `learning/checkpoints/rl/benchmark10_<timestamp>/reports/summary.json`

Output folder:

- `learning/checkpoints/rl/desktop_<timestamp>/`
- models: `ppo_baseline.zip`, `ppo_dtm.zip`
- logs: `logs/baseline.json`, `logs/dtm.json`, csv files

## 4) Quick compare from logs

```bash
python tools/summarize_breakdown.py \
  --baseline learning/checkpoints/rl/desktop_<timestamp>/logs/baseline.json \
  --dtm learning/checkpoints/rl/desktop_<timestamp>/logs/dtm.json
```

## 5) Practical recommendations for this hardware

- Keep `n_steps=512`, `batch_size=128` as default.
- Use one environment first (`DummyVecEnv([env])`) for stable debugging.
- Run long jobs in `tmux`:

```bash
tmux new -s rlcpp
# run training command
# detach: Ctrl+b then d
```

## 6) For later lab server migration

- Keep same scripts and only change:
  - `PYTHON` path
  - `TOTAL_TIMESTEPS`
  - map set / seeds
- This keeps desktop and lab results directly comparable.
