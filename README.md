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

## Milestone 2 stuff

`train_apply.train_apply(method='digress', dataset='qm9', ...)` trains (model selection on the validation loss) and
returns generated test-time samples plus test metrics; every other argument has a default:

```python
from train_apply import train_apply
out = train_apply("digress", "qm9")                       # paper setting
out = train_apply("digress", "qm9", transition="uniform") # noise-model ablation
out = train_apply("digress", "qm9", node_index_features=True)  # without permutation equivariance
out = train_apply("mlp", "qm9")                           # Milestone-1 denoiser in the same pipeline
```

### Datasets

| `--dataset` | data | split | max atoms | atom types |
|---|---|---|---|---|
| `qm9` | all usable QM9 molecules (`--subset-size 50000` = Milestone 1 data) | random 81/9/10 | 9 | C N O F |
| `moses` | MOSES benchmark, downloaded to `MOSES_DIR` (default `/home/space/datasets/moses`) | train / test_scaffolds (validation) / test | 27 | C N S O F Cl Br |

Molecules are kekulized; `--aromatic` keeps aromatic bonds as an additional class like the DiGress reference code (moses).
Charged molecules are filtered out as in Milestone 1 (MOSES contains none). Build MOSES cache once before training
(parsing of all ~1.9M SMILES in MOSES takes some time; parallel with flag `--workers`):

```bash
python -m digress.cli.prepare_data --dataset moses --workers 8
```

`--subset-size N` on MOSES trains on a seeded subset of N train molecules (validation/test unchanged).

## Run settings and ablation flags (`python -m digress.cli.train --help`)

Fixed once a run exists (stored in the checkpoint; resuming with a different explicit value is going to cause an error):
`--dataset`, `--subset-size`, `--aromatic`, `--diffusion-steps` (default 500), `--transition {marginal,uniform}`,
`--optimizer {adam,adamw}`, `--weight-decay`.

Model (new M2 runs only): `--model {digress,mlp,equivariant_mlp,simple_graph_transformer}` (default `digress`, sizes are taken from
the paper: 9 layers on QM9, 12 on MOSES), `--n-layers`, `--hidden-dim`, `--dropout`,
`--node-index-features` (removing equivariance), and the planned feature flags `--extra-features
{cycles,eigenvalues,all}` and `--molecular-features` ; these are wired through (stored in model config, input
dimensions are reserved in `digress/features/`) but raises `NotImplementedError` until implemented.

Training: `--steps`, `--lr` / `--batch-size` (defaults are per model and dataset), `--val-size` (cap per validation pass),
`--eval-samples`, `--max-minutes`.

## Metrics

`evaluate_test` writes `test_results.json`, `test_samples.pt` and `test_samples.smi`. Besides the M1 metrics
it reports the relaxed validity / uniqueness / novelty used in DiGress (by the authors' reference) and the MOSES distribution metrics SNN, Frag, Scaf
(from the MOSES paper), against the test split; FCD and Filters are added when `fcd_torch` / `molsets` are installed.

## Cluster

From the repo root (`mkdir -p logs` once); arguments after the model are passed to `train`:

```bash
sbatch --account=<account> --job-name=gt_20k \
    cluster/train.sh gt_20k simple_graph_transformer --steps 20000
sbatch --account=<account> --job-name=digress_qm9 \
    cluster/train.sh digress_qm9 digress --dataset qm9 --steps 100000 --max-minutes 110
sbatch --account=<account> cluster/prepare_data.sh --dataset moses \   # once, before MOSES training
sbatch --account=<account> --job-name=digress_moses \
    cluster/train.sh digress_moses_firstTestRun digress --dataset moses --steps 100000 --max-minutes 110
```

Logs go to `logs/<job-name>_<job-id>.out`, the container defaults to `~/pml.sif` (override with `CONTAINER=...`).


Milestone 1 runs (`outputs/runs/gt_20k`, ...) still load: they are treated as QM9 only, 50k subset with T=200 and Adam.