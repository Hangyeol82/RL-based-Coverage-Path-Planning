Log analysis quickstart

1) Put rollout JSON files into `log_analysis/logs/`.
   - Example filename: `indoor_seed101_baseline.json`
   - Expected format: top-level key `rollouts` with a list of records.

2) Plot one log file:

```bash
python log_analysis/plot_rollout_log.py \
  --input log_analysis/logs/indoor_seed101_baseline.json \
  --smooth-window 3 \
  --show
```

3) Output PNG:
   - Default: same folder as input, named `<input_stem>_plot.png`
   - You can override with `--output`.

Notes:
- X-axis defaults to `timesteps`. Use `--x-axis rollout` to plot by rollout index.
- The script draws: `reward_total`, `coverage_ratio`, `collision`, `reward_area`, `reward_tv_i` when present.
