# -----------------------------------------------------------------------------
# Riemannian flow matching (RFM), after Chen & Lipman, "Flow Matching on General
# Geometries" (docs/papers/RFM.md) and RFMP (docs/papers/RFMP.md). Reference code
# (read only, not a dependency): facebookresearch/riemannian-fm @ b6ac1e9.
# Ported unchanged from the old xyz repo (xyz/flow_matching/riemannian.py @ 52260ad,
# H12/H13 on S^2); the LASA chunk-policy helpers and S^2 priors were left there.
# Time convention: t = 0 is noise, t = 1 is data.
# The manifold is passed as an object mf with exp(p, v), log(p, q), proj(p, v)
# and projx(x). Points and tangent vectors have the same shape, broadcast over
# trailing dims, e.g. (n, 3) or (n, J, 3); times t have shape (n,).
# -----------------------------------------------------------------------------

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx


def _bcast(t, x):
    # t (n,) -> (n, 1, ..., 1) to broadcast against x (n, ...).
    return t.reshape(t.shape + (1,) * (x.ndim - 1))


def geodesic_path(mf, x0, x1, t):
    """Compute the point x_t on the geodesic from noise to data (RFM Eq. 15).

    Args:
        mf (object): Manifold with exp, log, proj, projx.
        x0 (jnp.ndarray): Noise points, shape (n, ..., D).
        x1 (jnp.ndarray): Data points, shape (n, ..., D).
        t (jnp.ndarray): Times in [0, 1], shape (n,).

    Returns:
        jnp.ndarray: Points x_t, shape (n, ..., D).
    """
    return mf.exp(x0, _bcast(t, x0) * mf.log(x0, x1))


def geodesic_velocity(mf, x0, x1, t):
    """Compute the conditional target velocity d/dt x_t of the geodesic path.

    Args:
        mf (object): Manifold with exp, log, proj, projx.
        x0 (jnp.ndarray): Noise points, shape (n, ..., D).
        x1 (jnp.ndarray): Data points, shape (n, ..., D).
        t (jnp.ndarray): Times in [0, 1], shape (n,).

    Returns:
        jnp.ndarray: Velocities, tangent at x_t, shape (n, ..., D).
    """
    # Forward mode in t: no 1/(1 - t) factor, so it is finite up to t = 1.
    _, dx = jax.jvp(
        lambda tt: geodesic_path(mf, x0, x1, tt),
        (t,),
        (jnp.ones_like(t),)
    )
    return dx


def exp_euler_sample(mf, v_fn, x0, n_steps):
    """Integrate dx/dt = v_fn(x, t) from t = 0 to 1 with exp-map Euler steps (RFMP Alg. 1).

    Args:
        mf (object): Manifold with exp, log, proj, projx.
        v_fn (callable): Tangent field v_fn(x (n, ..., D), t (n,)) -> (n, ..., D).
        x0 (jnp.ndarray): Noise points at t = 0, shape (n, ..., D).
        n_steps (int): Number of Euler steps.

    Returns:
        jnp.ndarray: Samples at t = 1, shape (n, ..., D).
    """
    dt = 1.0 / n_steps

    def step(x, k):
        t = jnp.full(x.shape[:1], k * dt)
        # projx after each step removes float32 drift off the manifold, as in
        # riemannian-fm @ b6ac1e9 (manifm/solvers.py, projx_integrator, projx=True).
        return mf.projx(mf.exp(x, dt * v_fn(x, t))), None

    x1, _ = jax.lax.scan(step, x0, jnp.arange(n_steps))
    return x1


def tangent_field(mf, model, o=None, c=None):
    """Wrap an ambient-output model as a tangent field v(x, t) = proj_x model(x, t[, c, o]) (RFM Eq. 26).

    Args:
        mf (object): Manifold with exp, log, proj, projx.
        model (callable): Ambient field model(x, t) -> same shape as x; with o, model(x (n, T_p*D), t, c=c, o=(n, T_o*D)).
        o (jnp.ndarray): Observations, shape (n, T_o, D); None for an unconditional point model.
        c (jnp.ndarray): Class labels, shape (n,) int; used only with o; None for no label.

    Returns:
        callable: Tangent field v_fn(x, t) -> same shape as x.
    """
    if o is None:
        return lambda x, t: mf.proj(x, model(x, t))
    # Chunk policy: flatten the chunk (n, T_p, D) and the observation for the MLP,
    # reshape the output back and project each point onto its own tangent space.
    o_flat = o.reshape(o.shape[0], -1)
    return lambda x, t: mf.proj(x, model(x.reshape(x.shape[0], -1), t, c=c, o=o_flat).reshape(x.shape))


def rfm_loss(mf, v_fn, x0, x1, t):
    """Compute the per-sample RCFM loss |v(x_t, t) - d/dt x_t|^2 (RFM Eq. 10).

    Args:
        mf (object): Manifold with exp, log, proj, projx.
        v_fn (callable): Tangent field v_fn(x (n, ..., D), t (n,)) -> (n, ..., D).
        x0 (jnp.ndarray): Noise points, shape (n, ..., D).
        x1 (jnp.ndarray): Data points, shape (n, ..., D).
        t (jnp.ndarray): Times in [0, 1], shape (n,).

    Returns:
        jnp.ndarray: Per-sample losses, shape (n,).
    """
    xt = geodesic_path(mf, x0, x1, t)
    # Squared norm over the last axis (the embedded metric, e.g. of S^2 in R^3 or R^n), mean over any chunk axes.
    sq = jnp.sum(jnp.square(v_fn(xt, t) - geodesic_velocity(mf, x0, x1, t)), axis=-1)
    return jnp.mean(sq.reshape(sq.shape[0], -1), axis=-1)


def train_policy(mf, model, sample_x0, sample_x1, val, key, n_blocks, block_steps, batch, lr, warmup=500,
                 weight_decay=1e-3, betas=(0.95, 0.999), ema_decay=0.999, val_batch=1024, log_fn=None):
    """Train a chunk policy with AdamW (warmup + cosine) and an EMA of the weights; keep the best EMA on validation.

    Args:
        mf (object): Manifold with exp, log, proj, projx.
        model (nnx.Module): Chunk model, as in tangent_field with o; updated in place to the best EMA weights.
        sample_x0 (callable): Prior sampler sample_x0(key, n) -> (n, T_p, D).
        sample_x1 (callable): Data sampler sample_x1(key, n) -> (x1 (n, T_p, D), o (n, T_o, D), c (n,)).
        val (tuple): Fixed validation set (x0, x1, o, c, t), shapes (m, T_p, D), (m, T_p, D), (m, T_o, D), (m,), (m,).
        key (jax.Array): PRNG key.
        n_blocks (int): Number of blocks; validation runs after each block.
        block_steps (int): Optimizer steps per block.
        batch (int): Batch size.
        lr (float): Peak learning rate.
        warmup (int): Linear warmup steps.
        weight_decay (float): AdamW weight decay.
        betas (tuple): AdamW (b1, b2).
        ema_decay (float): EMA decay per step.
        val_batch (int): Batch size of the validation pass.
        log_fn (callable): Called after each block as log_fn(block, train_losses (block_steps,), val_loss, is_best, lr, best_ema (host nnx.State)).

    Returns:
        train_losses (np.ndarray): Batch loss per step, shape (n_blocks * block_steps,).
        val_losses (np.ndarray): EMA validation loss per block, shape (n_blocks,).
    """
    schedule = optax.warmup_cosine_decay_schedule(0.0, lr, warmup, n_blocks * block_steps)
    opt = nnx.Optimizer(model, optax.adamw(schedule, b1=betas[0], b2=betas[1], weight_decay=weight_decay), wrt=nnx.Param)
    graphdef, state = nnx.split((model, opt))
    ema = jax.tree.map(jnp.copy, nnx.state(model, nnx.Param))

    def step(carry, key):
        state, ema = carry
        model, opt = nnx.merge(graphdef, state)
        k0, k1, kt = jax.random.split(key, 3)
        x1, o, c = sample_x1(k1, batch)
        x0 = sample_x0(k0, batch)
        t = jax.random.uniform(kt, (batch,))
        loss_fn = lambda m: jnp.mean(rfm_loss(mf, tangent_field(mf, m, o, c), x0, x1, t))
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        opt.update(model, grads)
        ema = jax.tree.map(lambda e, p: ema_decay * e + (1 - ema_decay) * p, ema, nnx.state(model, nnx.Param))
        return (nnx.state((model, opt)), ema), loss

    # Donating the carry lets the 65M model fit a 4 GB GPU (the old state is not kept next to the new one).
    run_block = jax.jit(lambda carry, ks: jax.lax.scan(step, carry, ks), donate_argnums=(0,))

    # The model state is all nnx.Param, so the EMA state alone rebuilds a model.
    model_graphdef = nnx.graphdef(model)

    @jax.jit
    def val_loss_fn(ema):
        m = nnx.merge(model_graphdef, ema)
        x0, x1, o, c, t = val
        # Static slices keep the validation activations at val_batch samples.
        total = 0.0
        for i in range(0, x0.shape[0], val_batch):
            sl = slice(i, i + val_batch)
            total += jnp.sum(rfm_loss(mf, tangent_field(mf, m, o[sl], c[sl]), x0[sl], x1[sl], t[sl]))
        return total / x0.shape[0]

    carry = (state, ema)
    train_losses, val_losses, best, best_ema = [], [], jnp.inf, None
    for b in range(n_blocks):
        carry, losses = run_block(carry, jax.random.split(jax.random.fold_in(key, b), block_steps))
        v = float(val_loss_fn(carry[1]))
        train_losses.append(np.asarray(losses))
        val_losses.append(v)
        is_best = v < best
        if is_best:
            # Host copy: keeps GPU memory flat.
            best, best_ema = v, jax.device_get(carry[1])
        if log_fn is not None:
            log_fn(b, train_losses[-1], v, is_best, float(schedule((b + 1) * block_steps)), best_ema)

    nnx.update(model, best_ema)
    return np.concatenate(train_losses), np.asarray(val_losses)
