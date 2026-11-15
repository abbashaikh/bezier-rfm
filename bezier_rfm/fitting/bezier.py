"""Piecewise cubic geodesic Bézier curves on R^3 x SO(3) with one duration per segment."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from bezier_rfm.fitting import so3


class Controls(NamedTuple):
    """Knot poses q_k and forward handles H_k, k = 0..K."""

    q_pos: jnp.ndarray  # (K + 1, 3)
    q_rot: jnp.ndarray  # (K + 1, 3, 3)
    h_pos: jnp.ndarray  # (K + 1, 3)
    h_rot: jnp.ndarray  # (K + 1, 3, 3)


def phase_h(durations, ks):
    """Return equal segment durations inside each phase.

    Args:
        durations (jnp.ndarray): Phase durations T_i in s, shape (n,).
        ks (tuple): Number of segments K_i per phase, length n.

    Returns:
        jnp.ndarray: Segment durations h, shape (K,).
    """
    durations = jnp.asarray(durations)
    return jnp.concatenate([jnp.full(k, durations[i] / k) for i, k in enumerate(ks)])


def knot_times(h):
    """Return the knot times of segment durations.

    Args:
        h (jnp.ndarray): Segment durations in s, shape (K,).

    Returns:
        jnp.ndarray: Knot times tau, shape (K + 1,).
    """
    return jnp.concatenate([jnp.zeros(1), jnp.cumsum(h)])


def segment_controls(c, h):
    """Build the four controls of every segment, with duration-ratio handles for C1 in seconds.

    Args:
        c (Controls): Knot poses and forward handles.
        h (jnp.ndarray): Segment durations, shape (K,).

    Returns:
        tuple: Control positions, shape (K, 4, 3), and rotations, shape (K, 4, 3, 3).
    """
    # rho_{k+1} = h_k / h_{k+1}, with h_K := h_{K-1} for the last knot.
    rho = h / jnp.concatenate([h[1:], h[-1:]])
    log_pos = c.h_pos[1:] - c.q_pos[1:]
    log_rot = jax.vmap(so3.log)(jnp.swapaxes(c.q_rot[1:], 1, 2) @ c.h_rot[1:])
    p2_pos = c.q_pos[1:] - rho[:, None] * log_pos
    p2_rot = c.q_rot[1:] @ jax.vmap(so3.exp)(-rho[:, None] * log_rot)
    pos = jnp.stack([c.q_pos[:-1], c.h_pos[:-1], p2_pos, c.q_pos[1:]], axis=1)
    rot = jnp.stack([c.q_rot[:-1], c.h_rot[:-1], p2_rot, c.q_rot[1:]], axis=1)
    return pos, rot


def segment_point(pos, rot, u):
    """Evaluate one cubic segment with geodesic de Casteljau.

    Args:
        pos (jnp.ndarray): Control positions, shape (4, 3).
        rot (jnp.ndarray): Control rotations, shape (4, 3, 3).
        u (float): Local parameter in [0, 1].

    Returns:
        tuple: Position, shape (3,), and rotation, shape (3, 3).
    """

    def interp(a, b):
        return a @ so3.exp(u * so3.log(a.T @ b))

    for _ in range(3):
        pos = [(1 - u) * pos[i] + u * pos[i + 1] for i in range(len(pos) - 1)]
        rot = [interp(rot[i], rot[i + 1]) for i in range(len(rot) - 1)]
    return pos[0], rot[0]


def decode(c, h, t):
    """Evaluate the curve at times in seconds.

    Args:
        c (Controls): Knot poses and forward handles.
        h (jnp.ndarray): Segment durations in s, shape (K,).
        t (jnp.ndarray): Times in [0, sum(h)], shape (M,).

    Returns:
        tuple: Positions, shape (M, 3), and rotations, shape (M, 3, 3).
    """
    h = jnp.asarray(h)
    tau = knot_times(h)
    pos, rot = segment_controls(c, h)
    k = jnp.clip(jnp.searchsorted(tau, t, side="right") - 1, 0, len(h) - 1)
    u = (t - tau[k]) / h[k]
    return jax.vmap(segment_point)(pos[k], rot[k], u)


def velocity(c, h, t):
    """Return the linear and body-frame angular velocity of the curve, in seconds.

    Args:
        c (Controls): Knot poses and forward handles.
        h (jnp.ndarray): Segment durations in s, shape (K,).
        t (jnp.ndarray): Times, shape (M,).

    Returns:
        tuple: Linear velocity (m/s), shape (M, 3), and angular velocity (rad/s), shape (M, 3).
    """

    def point(s):
        p, r = decode(c, h, s[None])
        return p[0], r[0]

    def one(s):
        (_, r), (dp, dr) = point(s), jax.jacfwd(point)(s)
        return dp, so3.vee(r.T @ dr)

    return jax.vmap(one)(t)
