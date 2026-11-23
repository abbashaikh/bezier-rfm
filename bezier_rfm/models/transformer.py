"""Diffusion Policy-style Transformer over K segment tokens and one z token, with cross-attention to conditioning tokens."""

import jax
import jax.numpy as jnp
from flax import nnx

from bezier_rfm.models.velocity_mlp import posemb_sincos

# DP-T initialization: normal(0, 0.02) weights, zero biases.
_INIT = nnx.initializers.normal(0.02)


def _linear(a, b, rngs):
    return nnx.Linear(a, b, kernel_init=_INIT, rngs=rngs)


class DecoderLayer(nnx.Module):
    """Pre-norm layer: self-attention, cross-attention to memory, feed-forward; dropout on each branch."""

    def __init__(self, width, heads, dropout, rngs):
        """Build the layer.

        Args:
            width (int): Token width.
            heads (int): Attention heads.
            dropout (float): Dropout rate of attention weights and residual branches.
            rngs (nnx.Rngs): Parameter and dropout keys.
        """
        mha = lambda: nnx.MultiHeadAttention(heads, width, dropout_rate=dropout, kernel_init=_INIT, decode=False,
                                             rngs=rngs)
        self.self_attn, self.cross_attn = mha(), mha()
        self.norms = nnx.List([nnx.LayerNorm(width, rngs=rngs) for _ in range(3)])
        self.ff1, self.ff2 = _linear(width, 4 * width, rngs), _linear(4 * width, width, rngs)
        self.drops = nnx.List([nnx.Dropout(dropout, rngs=rngs) for _ in range(4)])

    def __call__(self, x, mem):
        """Apply the layer.

        Args:
            x (jnp.ndarray): Action tokens, shape (n, L, width).
            mem (jnp.ndarray): Conditioning tokens, shape (n, M, width).

        Returns:
            jnp.ndarray: Action tokens, shape (n, L, width).
        """
        # No mask: action tokens attend to one another in both directions.
        h = self.norms[0](x)
        x = x + self.drops[0](self.self_attn(h, h, h))
        x = x + self.drops[1](self.cross_attn(self.norms[1](x), mem, mem))
        h = self.drops[2](nnx.gelu(self.ff1(self.norms[2](x))))
        return x + self.drops[3](self.ff2(h))


class SplineTransformer(nnx.Module):
    """Velocity network: K segment tokens plus a z token, conditioned on 2 observation frames, start state and flow time."""

    def __init__(self, rngs, seg_in=29, seg_out=17, obs_dim=53, start_dim=18, k=3, width=256, depth=8, heads=4,
                 dropout=0.3):
        """Build the layers.

        Args:
            rngs (nnx.Rngs): Keys with streams params and dropout.
            seg_in (int): Segment-token input width.
            seg_out (int): Segment-token output width.
            obs_dim (int): Observation width per frame.
            start_dim (int): Start-state width.
            k (int): Number of segment tokens.
            width (int): Token width.
            depth (int): Number of decoder layers.
            heads (int): Attention heads.
            dropout (float): Attention dropout rate.
        """
        self.width = width
        self.seg_in, self.z_in = _linear(seg_in, width, rngs), _linear(1, width, rngs)
        self.obs_in, self.start_in = _linear(obs_dim, width, rngs), _linear(start_dim, width, rngs)
        self.pos_emb = nnx.Param(_INIT(rngs.params(), (k + 1, width)))
        self.cond_pos_emb = nnx.Param(_INIT(rngs.params(), (4, width)))
        # DP-T condition encoder for n_cond_layers = 0: Linear, Mish, Linear.
        self.cond1, self.cond2 = _linear(width, 4 * width, rngs), _linear(4 * width, width, rngs)
        self.layers = nnx.List([DecoderLayer(width, heads, dropout, rngs) for _ in range(depth)])
        self.norm = nnx.LayerNorm(width, rngs=rngs)
        self.seg_head, self.z_head = _linear(width, seg_out, rngs), _linear(width, 1, rngs)

    def __call__(self, seg, ztok, obs, start, t):
        """Predict the normalized flow velocity of every action token.

        Args:
            seg (jnp.ndarray): Segment-token inputs, shape (n, K, seg_in).
            ztok (jnp.ndarray): z-token input, shape (n, 1).
            obs (jnp.ndarray): Normalized observations, shape (n, 2, obs_dim).
            start (jnp.ndarray): Normalized start state, shape (n, start_dim).
            t (jnp.ndarray): Flow times, shape (n,).

        Returns:
            tuple: Segment outputs, shape (n, K, seg_out), and z output, shape (n, 1).
        """
        x = jnp.concatenate([self.seg_in(seg), self.z_in(ztok)[:, None]], 1) + self.pos_emb[...]
        mem = jnp.concatenate([posemb_sincos(t, self.width)[:, None], self.obs_in(obs), self.start_in(start)[:, None]], 1)
        mem = self.cond2(jax.nn.mish(self.cond1(mem + self.cond_pos_emb[...])))
        for layer in self.layers:
            x = layer(x, mem)
        x = self.norm(x)
        return self.seg_head(x[:, :-1]), self.z_head(x[:, -1])
