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

## Report assets

```bash
rm -rf outputs/report_assets # remove if you want to generate new report assets
python -m digress.cli.evaluate_test outputs/runs/<run_name>/checkpoints/best.pt   # per run, after training changed
python -m digress.cli.evaluate_baselines --image                              # after data changes
python -m digress.cli.report_dataset --out outputs/report_assets/data          # slow (~15-20 min)
python -m digress.cli.report_runs outputs/runs/<run_name_1> outputs/runs/<run_name_2> \
    --labels "<label1>" "<label2>" --out outputs/report_assets/runs           # up to 3 runs
cd outputs && zip -r report_assets.zip report_assets
```

`report_runs` reads the results of `evaluate_test` and `evaluate_baselines`, so run those first.

## Branch begin_ms2

Refactor/ extension of MS1 code with the MOSES dataset, already placed at /home/space/datasets/moses on cluster.
Still have to test out some changes, new DataInfo obj/class exposing some params like atoms, valences, atom weights, max atoms, aromatic bonds
or not (for MOSES). New moses dataset py file, addition of uniform transition model to marginal transitions (enable second/ other dataset,
batched sampling, sampling change for unifrom support(using each diffusio's own limit dist. instead of data marginal to supoprt uniform).
digress model w/o the spectral/ structural features for now, train changes regarding logging/confifgs, train_apply module as required by task sheet. some more evals, ...
