# PML Project

```bash
python -m digress.cli.train --run-name my_run
python -m digress.cli.sample outputs/runs/my_run/checkpoints/best.pt
python -m digress.cli.evaluate_test outputs/runs/my_run/checkpoints/best.pt
python -m digress.cli.evaluate_baselines --image
python -m digress.cli.report_runs outputs/runs/my_run --out outputs/runs/my_run/report
```

## Cluster

From the repo root (`mkdir -p logs` once); arguments after the model are passed to `train`:

```bash
sbatch --account=<account> --job-name=gt_20k \
    cluster/train.sh gt_20k simple_graph_transformer --steps 20000
```

Logs go to `logs/<job-name>_<job-id>.out`, the container defaults to `~/pml.sif` (override with `CONTAINER=...`).
Resubmitting with the same run name resumes from `last.pt`.