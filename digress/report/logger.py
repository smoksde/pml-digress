import csv
import json
import time

import torch

from digress.report.runinfo import run_info

SAMPLE_FIELDS = ["step", "valid_connected", "validity", "connected", "uniqueness", "novelty", "fail_valence",
                 "fail_other", "bonds_per_mol", "tv_atom_types", "tv_bond_types"]


def metric_fields(val_ts):
    return ["step", "train_loss", "train_atom", "train_bond", "val_loss", "val_atom", "val_bond",
            *(f"val_t{t}" for t in val_ts), "grad_norm", "lr", "steps_per_sec", "gpu_mem_gb", "elapsed_min"]


class CsvLog:

    def __init__(self, path, fields, keep_until_step):
        self.path, self.fields = path, fields
        rows = []
        if path.exists():
            with open(path) as f:
                rows = [r for r in csv.DictReader(f) if int(r["step"]) <= keep_until_step]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fields)
            writer.writeheader()
            writer.writerows(rows)

    def write(self, row):
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, self.fields).writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()})


class RunLogger:

    def __init__(self, run_dir, start_step, val_ts):
        self.run_dir = run_dir
        (run_dir / "samples").mkdir(parents=True, exist_ok=True)
        self.metrics = CsvLog(run_dir / "metrics.csv", metric_fields(val_ts), start_step)
        self.samples = CsvLog(run_dir / "samples.csv", SAMPLE_FIELDS, start_step)
        self.t0 = time.time()
        self._reset_window()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def _reset_window(self):
        # running totals since the last metrics row: (loss, atom_loss, bond_loss, grad_norm), seconds, steps
        self.sums, self.train_time, self.steps = torch.zeros(4), 0.0, 0

    def write_config(self, settings):
        text = json.dumps({**settings, **run_info()}, indent=2, default=str) + "\n"
        (self.run_dir / "config.json").write_text(text)

    def add_train_step(self, stats, seconds):
        self.sums += stats
        self.train_time += seconds
        self.steps += 1

    def log_metrics(self, step, num_steps, val_losses, val_per_t, lr):
        train_loss, train_atom, train_bond, grad = (self.sums / self.steps).tolist()
        val, val_atom, val_bond = val_losses
        self.metrics.write({
            "step": step, "train_loss": train_loss, "train_atom": train_atom, "train_bond": train_bond,
            "val_loss": val, "val_atom": val_atom, "val_bond": val_bond, **val_per_t, "grad_norm": grad,
            "lr": lr, "steps_per_sec": self.steps / self.train_time,
            "gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0,
            "elapsed_min": (time.time() - self.t0) / 60,
        })
        print(f"step {step}/{num_steps}  train {train_loss:.4f}  val {val:.4f}  grad {grad:.2f}", flush=True)
        self._reset_window()

    def log_samples(self, step, scores, samples):
        try:  # drawing is optional: broken early samples or a missing drawing backend must not stop training
            from digress.report.molecules import graphs_from_sample, show

            atoms, adj, mask = (x[:16] for x in samples)
            show(graphs_from_sample(atoms, adj, mask), outfile=self.run_dir / "samples" / f"step_{step:06d}.png",
                 mols_per_row=8, sub_img_size=(160, 140))
        except (RuntimeError, ImportError) as err:
            print(f"  could not draw samples: {err}", flush=True)
        self.samples.write({"step": step, **{k: scores[k] for k in SAMPLE_FIELDS[1:]}})
        print(f"  samples @ {step}: valid+connected {scores['valid_connected']:.2f}  validity {scores['validity']:.2f}  "
              f"connected {scores['connected']:.2f}  bonds/mol {scores['bonds_per_mol']:.1f}", flush=True)
