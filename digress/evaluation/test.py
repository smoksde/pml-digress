"""
new file evaluation/test.py: the final evaluation of a trained run on the test split (digress.cli.evaluate_test and train_apply.py)
"""

import torch

from digress.config import EVAL_SAMPLES, REFERENCE_SIZE, TEST_EVAL_SIZE
from digress.data.dataset import subsample
from digress.evaluation.denoising import denoising_metrics, per_entry_reference
from digress.evaluation.metrics import relaxed_smiles
from digress.evaluation.moses_metrics import moses_metrics
from digress.training.loop import _evaluating, sample_quality, split_loss, val_ts


def evaluate_run(run, edge_weight, n_samples=EVAL_SAMPLES, max_test=TEST_EVAL_SIZE, distribution_metrics=True,
                 reference_size=REFERENCE_SIZE):
    """Test loss overall and per t, per-noise-level denoising metrics, and quality of the n_samples generated
    graphs (validity, ..., TVs against test split and optionally MOSES distribution metrics like in the MOSES paper).
    Returns (results dict, (atoms, adj, mask), generated SMILES with None for invalid graphs)."""
    data = run.data
    test_idx = subsample(data.test_idx, max_test)
    (loss, atom, bond), per_t = split_loss(run, edge_weight, test_idx, "test")
    samples, (atoms, adj, mask) = sample_quality(run, n_samples, data.test_idx)
    smiles = [relaxed_smiles(a, b, m, run.info) for a, b, m in zip(atoms, adj, mask)]
    reference, floor = per_entry_reference(run, edge_weight, val_ts(run.T))
    with _evaluating(run.model):
        denoising = denoising_metrics(run, test_idx)
    results = {
        "dataset": data.info.key, "n_test": len(data.test_idx), "n_test_evaluated": len(test_idx),
        "edge_loss_weight": edge_weight, "test_loss": loss, "test_atom": atom, "test_bond": bond, **per_t,
        "samples": {"n": n_samples, **samples}, "denoising": denoising,
        "per_entry_reference": reference, "marginal_floor": floor,
    }
    if distribution_metrics:
        ref = [data.smiles[i] for i in subsample(data.test_idx, reference_size).tolist()]
        results["distribution"] = moses_metrics([s for s in smiles if s], ref)
    return results, (atoms, adj, mask), smiles


def save_samples(path, atoms, adj, mask, smiles):
    torch.save({"atoms": atoms, "adj": adj, "mask": mask}, path)
    path.with_suffix(".smi").write_text("\n".join(s or "" for s in smiles) + "\n")
