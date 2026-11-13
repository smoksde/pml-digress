# PML Project

```bash
python -m digress.cli.train --run-name my_run
python -m digress.cli.sample outputs/runs/my_run/checkpoints/best.pt
python -m digress.cli.evaluate_test outputs/runs/my_run/checkpoints/best.pt
python -m digress.cli.evaluate_baselines --image
python -m digress.cli.report_runs outputs/runs/my_run --out outputs/runs/my_run/report
```