import random
import time
from contextlib import contextmanager

import torch

from digress.config import (
    CHECKPOINT_INTERVAL,
    DEVICE,
    DIFFUSION_STEPS,
    EVAL_SAMPLES,
    LOG_INTERVAL,
    MAX_ATOMS,
)
from digress.config import RANDOM_SEED, VAL_T_BUCKETS
from digress.data.dataset import get_dataset
from digress.data.graphs import edge_vector_to_adj
from digress.diffusion.sampling import generate
from digress.evaluation.metrics import (
    distribution_distances,
    evaluate,
    graph_histograms,
)
from digress.models.conditioning import normalize
from digress.training.checkpoint import build_run, load_checkpoint, save_checkpoint
from digress.training.loss import denoise_loss

VAL_TS = torch.linspace(1, DIFFUSION_STEPS, VAL_T_BUCKETS).round().long().tolist()


def train_step(run, data, batch_size, edge_weight):
    batch = data.sample_batch(batch_size)
    t = torch.randint(1, DIFFUSION_STEPS + 1, (batch_size,), device=DEVICE)
    loss, atom_loss, bond_loss = denoise_loss(run, batch, t, edge_weight)
    run.opt.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(run.model.parameters(), float("inf"))
    run.opt.step()
    return torch.tensor(
        [loss.item(), atom_loss.item(), bond_loss.item(), grad_norm.item()]
    )


@contextmanager
def _evaluating(model):
    model.eval()
    with torch.no_grad(), torch.random.fork_rng():
        torch.manual_seed(RANDOM_SEED)
        yield
    model.train()


def validation_loss(run, edge_weight, batch_size=500):
    return split_loss(run, edge_weight, get_dataset().val_idx, "val", batch_size)


def split_loss(run, edge_weight, idx, prefix, batch_size=500, by_t=True):
    # mean (loss, atom, bond) over a split with seeded noise, plus the total loss at each fixed t in VAL_TS
    data = get_dataset()

    def mean_loss(t_fixed=None):
        totals, count = torch.zeros(3), 0
        for start in range(0, len(idx), batch_size):
            batch = data.batch(idx[start : start + batch_size])
            n = batch[0].shape[0]
            t = (
                torch.full((n,), t_fixed, device=DEVICE)
                if t_fixed
                else torch.randint(1, DIFFUSION_STEPS + 1, (n,), device=DEVICE)
            )
            totals += torch.stack(denoise_loss(run, batch, t, edge_weight)).cpu() * n
            count += n
        return (totals / count).tolist()

    with _evaluating(run.model):
        overall = mean_loss()
        per_t = {f"{prefix}_t{t}": mean_loss(t)[0] for t in VAL_TS} if by_t else {}
    return overall, per_t


def reference_histograms(data, idx):
    return graph_histograms(
        data.atoms[idx],
        edge_vector_to_adj(data.edge_vecs[idx], MAX_ATOMS),
        data.masks[idx],
    )


def sample_quality(run, n=EVAL_SAMPLES, ref_idx=None):
    data = get_dataset()
    ref_idx = data.val_idx if ref_idx is None else ref_idx
    with _evaluating(run.model):
        y_in = None
        if run.model.y_dim:
            idx = data.train_idx[torch.randint(len(data.train_idx), (n,))]
            y_in = normalize(data.y[idx], data.y_mean, data.y_std).to(DEVICE)
        atoms, bonds, mask, _ = generate(
            run.model, run.atom_diffusion, run.bond_diffusion, run.marginals, n, y_in
        )
    adj = edge_vector_to_adj(bonds, MAX_ATOMS)
    scores = evaluate(atoms.cpu(), adj.cpu(), mask.cpu(), data.train_smiles_flat)
    scores.update(
        distribution_distances(
            graph_histograms(atoms, adj, mask), reference_histograms(data, ref_idx)
        )
    )
    return scores, (atoms.cpu(), adj.cpu(), mask.cpu())


def is_due(step, interval, num_steps):
    return step % interval == 0 or step == num_steps


def setup_run(run_dir, lr, seed, model_type=None, hidden_dim=None):
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    torch.manual_seed(seed)
    return build_run(
        load_checkpoint(run_dir / "checkpoints" / "last.pt"), lr, model_type, hidden_dim
    )


def train(run, run_dir, logger, num_steps, batch_size, eval_interval, edge_weight):
    data = get_dataset()
    ckpt_dir = run_dir / "checkpoints"
    for step in range(run.start_step + 1, num_steps + 1):
        step_start = time.time()
        logger.add_train_step(
            train_step(run, data, batch_size, edge_weight), time.time() - step_start
        )

        if is_due(step, LOG_INTERVAL, num_steps):
            val_losses, val_per_t = validation_loss(run, edge_weight)
            train_eval, _ = split_loss(
                run,
                edge_weight,
                data.train_idx[: len(data.val_idx)],
                "train_eval",
                by_t=False,
            )
            logger.log_metrics(
                step,
                num_steps,
                val_losses,
                val_per_t,
                train_eval,
                lr=run.opt.param_groups[0]["lr"],
            )
            if val_losses[0] < run.best_val:
                run.best_val = val_losses[0]
                save_checkpoint(ckpt_dir / "best.pt", run, data, step)
        if is_due(step, eval_interval, num_steps):
            logger.log_samples(step, *sample_quality(run))
        if is_due(step, CHECKPOINT_INTERVAL, num_steps):
            save_checkpoint(ckpt_dir / "last.pt", run, data, step)
