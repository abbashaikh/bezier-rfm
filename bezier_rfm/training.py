"""Training loop: AdamW with warmup and cosine decay, DP EMA of the weights, dropout, donated lax.scan blocks."""

import json
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx


def ema_decay(step, power=0.75, max_value=0.9999):
    """Return the Diffusion Policy EMA decay at an optimizer step (0 at step 0).

    Args:
        step (jnp.ndarray): Number of EMA updates done before this one.
        power (float): Decay power.
        max_value (float): Upper bound of the decay.

    Returns:
        jnp.ndarray: Decay in [0, max_value].
    """
    value = 1.0 - (1.0 + step) ** -power
    return jnp.where(step <= 0, 0.0, jnp.clip(value, 0.0, max_value))


def decay_mask(params):
    """Mark the parameters that get weight decay: Linear and attention kernels only, as in DP-T.

    Args:
        params (nnx.State): Parameters.

    Returns:
        pytree: Booleans with the structure of params.
    """
    def is_kernel(path, _):
        names = [k.key for k in path if isinstance(k, jax.tree_util.DictKey)]
        return names[-1] == "kernel"

    return jax.tree_util.tree_map_with_path(is_kernel, params)


def make_block(model, loss_fn, total_steps, lr, warmup, betas=(0.9, 0.95), weight_decay=1e-3, ema_power=0.75,
               ema_max=0.9999, wd_mask=None):
    """Build AdamW with warmup and cosine decay, the initial carry, and a jitted block of optimizer steps.

    Args:
        model (nnx.Module): Model with dropout; put in train mode.
        loss_fn (callable): loss_fn(model, key) -> (scalar loss, aux array); draws its own batch from key.
        total_steps (int): Length of the cosine schedule.
        lr (float): Peak learning rate.
        warmup (int): Linear warmup steps.
        betas (tuple): AdamW (b1, b2).
        weight_decay (float): AdamW weight decay.
        ema_power (float): EMA decay power.
        ema_max (float): EMA decay upper bound.
        wd_mask (callable): Weight-decay mask of the parameters (e.g. decay_mask); None decays all.

    Returns:
        tuple: graphdef, carry (state, ema, step), run_block(carry, keys (S, 2)) -> (carry, (loss (S,), aux (S, ...))), schedule.
    """
    schedule = optax.warmup_cosine_decay_schedule(0.0, lr, warmup, total_steps)
    model.train()
    opt = nnx.Optimizer(model, optax.adamw(schedule, b1=betas[0], b2=betas[1], weight_decay=weight_decay,
                                           mask=wd_mask), wrt=nnx.Param)
    graphdef, state = nnx.split((model, opt))
    ema = jax.tree.map(jnp.copy, nnx.state(model, nnx.Param))

    def step(carry, key):
        state, ema, n = carry
        model, opt = nnx.merge(graphdef, state)
        (loss, aux), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model, key)
        opt.update(model, grads)
        d = ema_decay(n, ema_power, ema_max)
        ema = jax.tree.map(lambda e, p: d * e + (1.0 - d) * p, ema, nnx.state(model, nnx.Param))
        return (nnx.state((model, opt)), ema, n + 1), (loss, aux)

    # Donating the carry keeps one copy of the weights and optimizer state on the GPU.
    run_block = jax.jit(lambda carry, ks: jax.lax.scan(step, carry, ks), donate_argnums=(0,))
    return graphdef, (state, ema, jnp.zeros((), jnp.int32)), run_block, schedule


def train(model, loss_fn, key, n_blocks, block_steps, lr, warmup, betas=(0.9, 0.95), weight_decay=1e-3,
          ema_power=0.75, ema_max=0.9999, log_fn=None):
    """Train a model in train mode and keep a DP EMA of its parameters.

    Args:
        model (nnx.Module): Model with dropout; updated in place to the last raw weights, left in eval mode.
        loss_fn (callable): loss_fn(model, key) -> (scalar loss, aux array); draws its own batch from key.
        key (jax.Array): PRNG key.
        n_blocks (int): Number of jitted blocks.
        block_steps (int): Optimizer steps per block.
        lr (float): Peak learning rate.
        warmup (int): Linear warmup steps.
        betas (tuple): AdamW (b1, b2).
        weight_decay (float): AdamW weight decay.
        ema_power (float): EMA decay power.
        ema_max (float): EMA decay upper bound.
        log_fn (callable): Called after each block as log_fn(block, losses (block_steps,), aux (block_steps, ...), lr, ema (host nnx.State)).

    Returns:
        tuple: EMA parameters (host nnx.State), losses (n_blocks * block_steps,), aux (n_blocks * block_steps, ...).
    """
    graphdef, carry, run_block, schedule = make_block(model, loss_fn, n_blocks * block_steps, lr, warmup, betas,
                                                      weight_decay, ema_power, ema_max)
    losses, auxs = [], []
    for b in range(n_blocks):
        carry, (loss, aux) = run_block(carry, jax.random.split(jax.random.fold_in(key, b), block_steps))
        losses.append(np.asarray(loss))
        auxs.append(np.asarray(aux))
        if log_fn is not None:
            log_fn(b, losses[-1], auxs[-1], float(schedule((b + 1) * block_steps)), jax.device_get(carry[1]))

    trained, _ = nnx.merge(graphdef, carry[0])
    nnx.update(model, nnx.state(trained))
    model.eval()
    return jax.device_get(carry[1]), np.concatenate(losses), np.concatenate(auxs)


def save_checkpoint(run_dir, params, config=None, name="ema.pkl", **arrays):
    """Save parameters, the run config and arrays in the checkpoints/<run>/ pattern.

    Args:
        run_dir (str): Run directory, e.g. checkpoints/<run>.
        params (nnx.State): Parameters to save.
        config (dict): Run config written to config.json; None to skip.
        name (str): File name of the parameters.
        **arrays (np.ndarray): Arrays written to losses.npz; none to skip.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / name, "wb") as f:
        pickle.dump(jax.tree.map(np.asarray, nnx.to_pure_dict(params)), f)
    if config is not None:
        (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    if arrays:
        np.savez(run_dir / "losses.npz", **arrays)


def load_params(model, path):
    """Load saved parameters into a model in place.

    Args:
        model (nnx.Module): Model with the same structure.
        path (str): Parameter file from save_checkpoint.
    """
    with open(path, "rb") as f:
        pure = pickle.load(f)
    params = nnx.state(model, nnx.Param)
    nnx.replace_by_pure_dict(params, pure)
    nnx.update(model, params)
