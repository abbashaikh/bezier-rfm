"""Batched pose maps on R^3 x SO(3): x = (p, R), tangent xi = (v world, omega body)."""

import jax
import jax.numpy as jnp

from bezier_rfm.fitting import so3


def _batched(f, x, n_trail):
    # Flatten the leading dims of x (all but its n_trail last), vmap f, and restore them.
    lead = x.shape[: x.ndim - n_trail]
    out = jax.vmap(f)(x.reshape((-1,) + x.shape[len(lead):]))
    return out.reshape(lead + out.shape[1:])


def rot_exp(w):
    """Map rotation vectors to rotation matrices.

    Args:
        w (jnp.ndarray): Rotation vectors, shape (..., 3).

    Returns:
        jnp.ndarray: Rotation matrices, shape (..., 3, 3).
    """
    return _batched(so3.exp, w, 1)


def rot_log(r):
    """Map rotation matrices (angle below pi) to rotation vectors.

    Args:
        r (jnp.ndarray): Rotation matrices, shape (..., 3, 3).

    Returns:
        jnp.ndarray: Rotation vectors, shape (..., 3).
    """
    return _batched(so3.log, r, 2)


def exp(p, r, v, w):
    """Move poses along tangent vectors: (p + v, R Exp(omega)).

    Args:
        p (jnp.ndarray): Positions, shape (..., 3).
        r (jnp.ndarray): Rotations, shape (..., 3, 3).
        v (jnp.ndarray): World-frame translations, shape (..., 3).
        w (jnp.ndarray): Body-frame rotation vectors, shape (..., 3).

    Returns:
        tuple: Positions, shape (..., 3), and rotations, shape (..., 3, 3).
    """
    return p + v, r @ rot_exp(w)


def log(p, r, p2, r2):
    """Return the tangent vector at (p, R) that points to (p2, R2): (p2 - p, Log(R^T R2)).

    Args:
        p (jnp.ndarray): Base positions, shape (..., 3).
        r (jnp.ndarray): Base rotations, shape (..., 3, 3).
        p2 (jnp.ndarray): Target positions, shape (..., 3).
        r2 (jnp.ndarray): Target rotations, shape (..., 3, 3).

    Returns:
        tuple: World-frame translations, shape (..., 3), and body-frame rotation vectors, shape (..., 3).
    """
    return p2 - p, rot_log(jnp.swapaxes(r, -1, -2) @ r2)


def project_rot(r):
    """Project matrices to the nearest rotation matrices.

    Args:
        r (jnp.ndarray): Near-rotation matrices, shape (..., 3, 3).

    Returns:
        jnp.ndarray: Rotation matrices, shape (..., 3, 3).
    """
    u, _, vt = jnp.linalg.svd(r)
    # Flip the last singular direction if the product is a reflection.
    d = jnp.sign(jnp.linalg.det(u @ vt))
    u = u.at[..., :, 2].multiply(d[..., None])
    return u @ vt
