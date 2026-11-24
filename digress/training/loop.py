"""
changes: TrainConfig collect atgs which may change betweens resuming runs
(steps, lr, val_size, max minutes)

val_ts(T) replaces VAL_TS
--max-minutes saves last.pt with some time left so it should not break anything
if the job is terminated.

setup_run() functionality moved to training/api.py
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass

import torch

from digress.config import (
    BATCH_SIZE,
    CHECKPOINT_INTERVAL,
    DEVICE,
    EDGE_LOSS_WEIGHT,
    EVAL_INTERVAL,
    EVAL_SAMPLES,
    GRAD_CLIP,
    LEARNING_RATE,
    LOG_INTERVAL,
    RANDOM_SEED,
    REFERENCE_SIZE,
    TRAIN_STEPS,
    VAL_SIZE,
    VAL_T_BUCKETS,
)
from digress.data.dataset import subsample
from digress.data.graphs import edge_vector_to_adj
from digress.diffusion.sampling import generate
from digress.evaluation.metrics import distribution_distances, evaluate, graph_histograms
from digress.models.conditioning import normalize
from digress.training.checkpoint import save_checkpoint
from digress.training.loss import denoise_loss


@dataclass
class TrainConfig:
    """Args of a training call which are not part of the run itself (i.e. may change between runs)"""

    steps: int = TRAIN_STEPS
    batch_size: int = BATCH_SIZE
    lr: float = LEARNING_RATE
    seed: int = RANDOM_SEED
    edge_loss_weight: float = EDGE_LOSS_WEIGHT
    grad_clip: float = GRAD_CLIP  # 0 is off/ no clipping
    log_interval: int = LOG_INTERVAL
    eval_interval: int = EVAL_INTERVAL
    checkpoint_interval: int = CHECKPOINT_INTERVAL
    eval_samples: int = EVAL_SAMPLES
    val_size: int = VAL_SIZE  # cap on molecules per validation-loss pass
    max_minutes: float = None  # stop early and save last.pt after this much time. resubmit to resume
    ema_decay: float = 0.0  # placeholder, not yet added.

    def __post_init__(self):
        if self.ema_decay:
            raise NotImplementedError("EMA of the weights planned but not implemented yet; use --ema-decay 0 or omit")


def val_ts(T, buckets=VAL_T_BUCKETS):
    return torch.linspace(1, T, buckets).round().long().tolist()


def train_step(run, batch_size, edge_weight, grad_clip):
    batch = run.data.sample_batch(batch_size)
    t = torch.randint(1, run.T + 1, (batch_size,), device=DEVICE)
    loss, atom_loss, bond_loss = denoise_loss(run, batch, t, edge_weight)
    run.opt.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(run.model.parameters(), grad_clip or float("inf"))
    run.opt.step()
    return torch.tensor([loss.item(), atom_loss.item(), bond_loss.item(), grad_norm.item()])


@contextmanager
def _evaluating(model):
    was_training = model.training
    model.eval()
    with torch.no_grad(), torch.random.fork_rng(devices=[torch.cuda.current_device()] if torch.cuda.is_available() else []):
        torch.manual_seed(RANDOM_SEED)
        yield
    model.train(was_training)


def split_loss(run, edge_weight, idx, prefix, batch_size=500, by_t=True):
    # mean (loss, atom, bond) over a split with seeded noise, plus the total loss at each fixed t in val_ts(T)
    data = run.data

    def mean_loss(t_fixed=None):
        totals, count = torch.zeros(3), 0
        for start in range(0, len(idx), batch_size):
            batch = data.batch(idx[start : start + batch_size])
            n = batch[0].shape[0]
            t = (
                torch.full((n,), t_fixed, device=DEVICE)
                if t_fixed
                else torch.randint(1, run.T + 1, (n,), device=DEVICE)
            )
            totals += torch.stack(denoise_loss(run, batch, t, edge_weight)).cpu() * n
            count += n
        return (totals / count).tolist()

    with _evaluating(run.model):
        overall = mean_loss()
        per_t = {f"{prefix}_t{t}": mean_loss(t)[0] for t in val_ts(run.T)} if by_t else {}
    return overall, per_t


def reference_histograms(data, idx, cap=REFERENCE_SIZE):
    return graph_histograms(*data.graphs(subsample(idx, cap)), info=data.info)


def sample_conditions(run, n):
    if not run.model.y_dim:
        return None
    data = run.data
    idx = data.train_idx[torch.randint(len(data.train_idx), (n,))]
    return normalize(data.y[idx], data.y_mean, data.y_std).to(DEVICE)


def sample_quality(run, n=EVAL_SAMPLES, ref_idx=None):
    data = run.data
    ref_idx = data.val_idx if ref_idx is None else ref_idx
    with _evaluating(run.model):
        atoms, bonds, mask, _ = generate(run.model, run.atom_diffusion, run.bond_diffusion,
                                         run.marginals["atom_count_dist"], n, run.info, sample_conditions(run, n))
    adj = edge_vector_to_adj(bonds, run.info.max_atoms)
    scores = evaluate(atoms.cpu(), adj.cpu(), mask.cpu(), data.train_smiles_flat, run.info)
    scores.update(distribution_distances(graph_histograms(atoms, adj, mask, run.info),
                                         reference_histograms(data, ref_idx)))
    return scores, (atoms.cpu(), adj.cpu(), mask.cpu())


def is_due(step, interval, num_steps):
    return step % interval == 0 or step == num_steps


def train(run, run_dir, logger, cfg):
    # Training until cfg.steps or cfg.max_minutes; Returns True if all steps done
    data = run.data
    ckpt_dir = run_dir / "checkpoints"
    val_idx = subsample(data.val_idx, cfg.val_size)
    train_eval_idx = subsample(data.train_idx, len(val_idx))
    t_start = time.time()
    run.model.train()
    step = run.start_step
    for step in range(run.start_step + 1, cfg.steps + 1):
        step_start = time.time()
        logger.add_train_step(train_step(run, cfg.batch_size, cfg.edge_loss_weight, cfg.grad_clip),
                              time.time() - step_start)

        if is_due(step, cfg.log_interval, cfg.steps):
            val_losses, val_per_t = split_loss(run, cfg.edge_loss_weight, val_idx, "val")
            train_eval, _ = split_loss(run, cfg.edge_loss_weight, train_eval_idx, "train_eval", by_t=False)
            logger.log_metrics(step, cfg.steps, val_losses, val_per_t, train_eval, lr=run.opt.param_groups[0]["lr"])
            if val_losses[0] < run.best_val:
                run.best_val = val_losses[0]
                save_checkpoint(ckpt_dir / "best.pt", run, step)
        if is_due(step, cfg.eval_interval, cfg.steps):
            logger.log_samples(step, *sample_quality(run, cfg.eval_samples))
        out_of_time = cfg.max_minutes is not None and (time.time() - t_start) / 60 > cfg.max_minutes
        if is_due(step, cfg.checkpoint_interval, cfg.steps) or out_of_time:
            save_checkpoint(ckpt_dir / "last.pt", run, step)
        if out_of_time:
            print(f"stopping at step {step}/{cfg.steps} after {cfg.max_minutes} min; resubmit to resume", flush=True)
            return False
    return step >= cfg.steps
