# -----------------------------------------------------------------------------
# MLP velocity field with a Pi0-style sine-cosine time embedding.
# Ported unchanged from the old xyz repo (xyz/models/velocity_mlp.py @ 52260ad, H11).
# -----------------------------------------------------------------------------

import jax
import jax.numpy as jnp
from flax import nnx


def posemb_sincos(t, dim, min_period=4e-3, max_period=4.0):
    """Embed scalar times as sine-cosine features, as in Pi0.

    Args:
        t (jnp.ndarray): Times, shape (n,).
        dim (int): Embedding width, even.
        min_period (float): Shortest period.
        max_period (float): Longest period.

    Returns:
        jnp.ndarray: Embeddings [sin, cos], shape (n, dim).
    """
    fraction = jnp.linspace(0.0, 1.0, dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    arg = t[:, None] * (2 * jnp.pi / period)
    return jnp.concatenate([jnp.sin(arg), jnp.cos(arg)], axis=-1)


class VelocityMLP(nnx.Module):
    """MLP velocity field v(x, t[, c][, o]) on concat[x, posemb(t)[, onehot(c)][, o]] with swish layers."""

    def __init__(self, d, rngs, width=64, depth=3, t_dim=32, c_dim=0, o_dim=0):
        """Build the layers.

        Args:
            d (int): Data dimension.
            rngs (nnx.Rngs): Parameter initialization keys.
            width (int): Hidden width.
            depth (int): Number of hidden layers.
            t_dim (int): Time-embedding width.
            c_dim (int): Number of condition classes; 0 for no condition.
            o_dim (int): Width of a continuous condition vector; 0 for none.
        """
        self.t_dim = t_dim
        self.c_dim = c_dim
        self.o_dim = o_dim
        dims = [d + t_dim + c_dim + o_dim] + [width] * depth
        self.hidden = nnx.List([nnx.Linear(a, b, rngs=rngs) for a, b in zip(dims[:-1], dims[1:])])
        self.out = nnx.Linear(width, d, rngs=rngs)

    def __call__(self, x, t, c=None, o=None):
        """Evaluate the field.

        Args:
            x (jnp.ndarray): Points, shape (n, d).
            t (jnp.ndarray): Times in [0, 1], shape (n,).
            c (jnp.ndarray): Class labels, shape (n,) int; used only if c_dim > 0.
            o (jnp.ndarray): Continuous condition, shape (n, o_dim); used only if o_dim > 0.

        Returns:
            jnp.ndarray: Velocities, shape (n, d).
        """
        h = jnp.concatenate([x, posemb_sincos(t, self.t_dim)], axis=-1)
        if self.c_dim:
            h = jnp.concatenate([h, jax.nn.one_hot(c, self.c_dim)], axis=-1)
        if self.o_dim:
            h = jnp.concatenate([h, o], axis=-1)
        for layer in self.hidden:
            h = nnx.swish(layer(h))
        return self.out(h)
