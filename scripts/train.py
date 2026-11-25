"""H30: train the spline RFM policy on all Tool Hang windows, with the validation RFM loss and EMA checkpoints (plan.md 9.3).

Run from the repo root:
    uv run python scripts/train.py --run checkpoints/h30_m1 --steps 100000      # milestone 1
    uv run python scripts/train.py --run checkpoints/h30                        # full run (config budget)
    uv run python scripts/train.py --run checkpoints/h31_m1 --steps 100000 --time-base speed   # H31
    uv run python scripts/train.py --run checkpoints/h30 --resume               # continue after a stop
The W&B key comes only from the WANDB_API_KEY environment variable.
"""

import argparse
import json
import os
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from bezier_rfm.datasets import window_loader as wl
from bezier_rfm.flow_matching import spline
from bezier_rfm.models.transformer import SplineTransformer
from bezier_rfm.training import decay_mask, make_block, save_checkpoint

N_BIN = 10


class Probe:
    """Fixed windows and flow times for the RFM loss of one window set."""

    def __init__(self, idx, cfg, seed):
        """Draw the fixed windows of a set.

        Args:
            idx (np.ndarray): Window indices of the set.
            cfg (dict): Run config (bin_draws).
            seed (int): Seed of the fixed choice.
        """
        rng = np.random.default_rng(seed)
        # Unused draw: keeps the loss windows equal to those of H30.
        rng.choice(idx, 64, replace=False)
        nb = cfg["bin_draws"]
        self.bin_rows = rng.choice(idx, nb)
        # Stratified flow times, so every bin gets nb / N_BIN draws.
        self.t = jnp.asarray((np.arange(nb) % N_BIN + rng.random(nb)) / N_BIN, jnp.float32)


def offline_loss(model, data, stats, probe, key):
    """Compute the RFM loss per term and flow-time bin on a probe.

    Args:
        model (nnx.Module): EMA model in eval mode.
        data (tuple): Device targets (spline.State) and conditioning (spline.Cond) of all windows.
        stats (spline.Stats): Scales.
        probe (Probe): Fixed windows and flow times.
        key (jax.Array): Fixed PRNG key of the prior draws.

    Returns:
        dict: Scalar losses.
    """
    x1, cond = data
    # First half of the split, as in H30.
    kb, _ = jax.random.split(key)
    xb1, cb = wl.take(x1, probe.bin_rows), wl.take(cond, probe.bin_rows)
    xb0 = spline.sample_prior(kb, cb.q0_pos, cb.q0_rot, stats, cb.log_t_base)
    terms = np.asarray(_loss(model, xb0, xb1, probe.t, cb, stats))
    bins = np.minimum((np.asarray(probe.t) * N_BIN).astype(int), N_BIN - 1)
    out = {f"loss/{n}": float(terms[:, j].mean()) for j, n in enumerate(spline.TERMS)}
    out["loss/total"] = float(terms.sum(1).mean())
    for b in range(N_BIN):
        for j, n in enumerate(spline.TERMS):
            out[f"loss_bin{b}/{n}"] = float(terms[bins == b, j].mean())
    return out


@jax.jit
def _loss(model, x0, x1, t, cond, stats):
    return spline.loss(model, x0, x1, t, cond, stats)


def save_resume(path, carry, block):
    """Write the full training state for resume, atomically.

    Args:
        path (Path): Output file.
        carry (tuple): Training carry (state, ema, step).
        block (int): Number of finished blocks.
    """
    leaves = [np.asarray(jax.random.key_data(a)) if jnp.issubdtype(a.dtype, jax.dtypes.prng_key) else np.asarray(a)
              for a in jax.tree.leaves(carry)]
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump({"block": block, "leaves": leaves}, f)
    os.replace(tmp, path)


def load_resume(path, carry):
    """Read a resume file into the structure of a fresh carry.

    Args:
        path (Path): Resume file from save_resume.
        carry (tuple): Fresh carry with the same structure.

    Returns:
        tuple: Restored carry on the device and the number of finished blocks.
    """
    with open(path, "rb") as f:
        d = pickle.load(f)
    ref, treedef = jax.tree.flatten(carry)
    leaves = [jax.random.wrap_key_data(v, impl=jax.random.key_impl(r)) if jnp.issubdtype(r.dtype, jax.dtypes.prng_key)
              else jnp.asarray(v) for v, r in zip(d["leaves"], ref)]
    return jax.tree.unflatten(treedef, leaves), d["block"]


def init_wandb(cfg, run_dir, mode):
    """Start or resume the W&B run of this training run.

    Args:
        cfg (dict): Run config with wandb_id and wandb_project.
        run_dir (Path): Run directory.
        mode (str): online, offline or disabled.

    Returns:
        wandb.Run: The run, or None when disabled.
    """
    if mode == "disabled":
        return None
    import wandb

    # Shared mode lets scripts/eval.py write to the same run at the same time.
    settings = wandb.Settings(mode="shared", x_primary=True, x_label="train") if mode == "online" else \
        wandb.Settings(mode=mode)
    run = wandb.init(project=cfg["wandb_project"], name=run_dir.name, id=cfg["wandb_id"], resume="allow", config=cfg,
                     dir=str(run_dir), settings=settings)
    run.define_metric("trainer/step")
    for prefix in ("train/*", "val/*", "trainset/*"):
        run.define_metric(prefix, step_metric="trainer/step")
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tool_hang.json")
    ap.add_argument("--run", required=True, help="run directory, e.g. checkpoints/h30")
    ap.add_argument("--steps", type=int, help="gradient steps (overrides the config)")
    ap.add_argument("--dropout", type=float, help="Transformer dropout (overrides the config)")
    ap.add_argument("--time-base", choices=["none", "speed"], help="duration target (overrides the config): log T, or log(T / T_base(s)) (H31)")
    ap.add_argument("--n-evals", type=int, help="evaluation points (overrides the config; smoke tests only)")
    ap.add_argument("--block-steps", type=int, help="steps per jitted block (overrides the config; smoke tests only)")
    ap.add_argument("--resume", action="store_true", help="continue from <run>/resume.pkl with the saved config")
    ap.add_argument("--wandb", choices=["online", "offline", "disabled"], default="online")
    args = ap.parse_args()
    run_dir = Path(args.run)

    # ---- config: the saved one on resume, else the file plus flags
    if args.resume:
        cfg = json.loads((run_dir / "config.json").read_text())
    else:
        if (run_dir / "resume.pkl").exists():
            raise SystemExit(f"{run_dir} has a resume file; pass --resume or choose a new --run")
        cfg = json.loads(Path(args.config).read_text())
        for k in ("steps", "dropout", "n_evals", "block_steps", "time_base"):
            if getattr(args, k) is not None:
                cfg[k] = getattr(args, k)
    n_blocks = cfg["steps"] // cfg["block_steps"]
    blocks_per_eval = n_blocks // cfg["n_evals"]
    if n_blocks * cfg["block_steps"] != cfg["steps"] or blocks_per_eval * cfg["n_evals"] != n_blocks:
        raise SystemExit("steps must be a multiple of n_evals * block_steps")

    # ---- data, split, scales
    w = wl.load_windows(cfg["windows"])
    val = wl.val_demos(len(np.unique(w["demo"])), cfg["val_ratio"], cfg["split_seed"])
    tr_idx, va_idx = wl.split(w, val)
    if args.resume:
        stats, saved_val = wl.load_stats(run_dir / "stats.json")
        if not np.array_equal(saved_val, val):
            raise SystemExit("the validation demos differ from stats.json")
    else:
        stats = spline.fit_stats({k: v[tr_idx] for k, v in w.items()}, cfg["time_base"])
        run_dir.mkdir(parents=True, exist_ok=True)
        wl.save_stats(run_dir / "stats.json", stats, val)
        stats, _ = wl.load_stats(run_dir / "stats.json")
    data = wl.device_data(w, stats)
    probes = {"val": Probe(va_idx, cfg, 1), "trainset": Probe(tr_idx, cfg, 2)}
    if not args.resume:
        import secrets

        cfg.update({"val_demos": val.tolist(), "n_train_windows": len(tr_idx), "n_val_windows": len(va_idx),
                    "wandb_id": secrets.token_hex(4)})
        (run_dir / "config.json").write_text(json.dumps(cfg, indent=1))
    print(f"validation demos {val.tolist()}; {len(tr_idx)} training and {len(va_idx)} validation windows", flush=True)

    # ---- model, optimizer, loss
    model = SplineTransformer(nnx.Rngs(params=cfg["seed"], dropout=cfg["seed"] + 1), dropout=cfg["dropout"])
    ema_model = SplineTransformer(nnx.Rngs(params=0, dropout=0), dropout=cfg["dropout"])
    ema_model.eval()
    tr_dev = jnp.asarray(tr_idx)
    x1_all, cond_all = data

    def loss_fn(m, key):
        ki, k0, kt = jax.random.split(key, 3)
        # Batch indices are drawn inside the scan from the training windows on the device.
        idx = tr_dev[jax.random.randint(ki, (cfg["batch"],), 0, tr_dev.shape[0])]
        x1, c = wl.take(x1_all, idx), wl.take(cond_all, idx)
        x0 = spline.sample_prior(k0, c.q0_pos, c.q0_rot, stats, c.log_t_base)
        terms = spline.loss(m, x0, x1, jax.random.uniform(kt, (cfg["batch"],)), c, stats).mean(0)
        return terms.sum(), terms

    _, carry, run_block, schedule = make_block(
        model, loss_fn, cfg["steps"], cfg["lr"], cfg["warmup"], tuple(cfg["betas"]), cfg["weight_decay"],
        cfg["ema_power"], cfg["ema_max"], wd_mask=decay_mask)
    start = 0
    if args.resume:
        carry, start = load_resume(run_dir / "resume.pkl", carry)
        print(f"resumed after block {start} (step {start * cfg['block_steps']})", flush=True)
    wb = init_wandb(cfg, run_dir, args.wandb)
    key = jax.random.PRNGKey(cfg["seed"] + 2)
    log = open(run_dir / "train_log.txt", "a")
    t_last = time.time()

    for b in range(start, n_blocks):
        carry, (loss, aux) = run_block(carry, jax.random.split(jax.random.fold_in(key, b), cfg["block_steps"]))
        loss, aux = np.asarray(loss), np.asarray(aux)
        step = (b + 1) * cfg["block_steps"]
        dt = time.time() - t_last
        t_last = time.time()
        mem = jax.devices()[0].memory_stats() or {}
        rec = {"trainer/step": step, "train/loss": float(loss.mean()), "train/lr": float(schedule(step)),
               "train/step_time_ms": 1e3 * dt / cfg["block_steps"],
               "train/peak_mem_gb": mem.get("peak_bytes_in_use", 0) / 1e9,
               **{f"train/{n}": float(a) for n, a in zip(spline.TERMS, aux.mean(0))}}
        line = (f"block {b} step {step} loss {rec['train/loss']:.4f} "
                + " ".join(f"{n} {rec['train/' + n]:.4f}" for n in spline.TERMS)
                + f" lr {rec['train/lr']:.2e} {rec['train/step_time_ms']:.1f} ms/step peak {rec['train/peak_mem_gb']:.2f} GB")
        if (b + 1) % blocks_per_eval == 0:
            ema = carry[1]
            nnx.update(ema_model, ema)
            t0 = time.time()
            for name, probe in probes.items():
                m = offline_loss(ema_model, data, stats, probe, jax.random.PRNGKey(100))
                rec.update({f"{name}/{k}": v for k, v in m.items()})
            save_checkpoint(run_dir, jax.device_get(ema), name=f"ema_{step:07d}.pkl")
            save_resume(run_dir / "resume.pkl", carry, b + 1)
            line += (f" | val loss {rec['val/loss/total']:.4f} trainset loss {rec['trainset/loss/total']:.4f}"
                     f" | eval {time.time() - t0:.0f}s")
            with open(run_dir / "metrics.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
            t_last = time.time()
        if wb is not None:
            wb.log(rec)
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()
    log.close()
    if wb is not None:
        wb.finish()


if __name__ == "__main__":
    main()
