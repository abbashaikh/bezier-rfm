"""Exponential and logarithm maps of SO(3) in JAX, with finite values and gradients at the identity."""

import jax.numpy as jnp

# Below this squared angle (or squared sine), the maps use Taylor series.
SMALL = 1e-8


def hat(w):
    """Return the skew-symmetric matrix of a rotation vector.

    Args:
        w (jnp.ndarray): Rotation vector, shape (3,).

    Returns:
        jnp.ndarray: Skew-symmetric matrix, shape (3, 3).
    """
    return jnp.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


def vee(k):
    """Return the vector of the antisymmetric part of a 3 x 3 matrix.

    Args:
        k (jnp.ndarray): Matrix, shape (3, 3).

    Returns:
        jnp.ndarray: Vector, shape (3,).
    """
    return 0.5 * jnp.array([k[2, 1] - k[1, 2], k[0, 2] - k[2, 0], k[1, 0] - k[0, 1]])


def exp(w):
    """Map a rotation vector to a rotation matrix.

    Args:
        w (jnp.ndarray): Rotation vector, shape (3,).

    Returns:
        jnp.ndarray: Rotation matrix, shape (3, 3).
    """
    theta2 = w @ w
    small = theta2 < SMALL
    # Double where: the unused branch must not see theta = 0, or its gradient is NaN.
    theta = jnp.sqrt(jnp.where(small, 1.0, theta2))
    a = jnp.where(small, 1.0 - theta2 / 6.0 + theta2**2 / 120.0, jnp.sin(theta) / theta)
    b = jnp.where(small, 0.5 - theta2 / 24.0 + theta2**2 / 720.0, (1.0 - jnp.cos(theta)) / theta**2)
    k = hat(w)
    return jnp.eye(3) + a * k + b * (k @ k)


def log(r):
    """Map a rotation matrix with angle below pi to its rotation vector.

    Args:
        r (jnp.ndarray): Rotation matrix, shape (3, 3).

    Returns:
        jnp.ndarray: Rotation vector, shape (3,).
    """
    # w = sin(theta) * axis.
    w = vee(r)
    c = 0.5 * (jnp.trace(r) - 1.0)
    s2 = w @ w
    # A small sine also occurs near theta = pi (c < 0), where the series below is wrong.
    small = (s2 < SMALL) & (c > 0.0)
    s = jnp.sqrt(jnp.where(small, 1.0, s2))
    # Near theta = 0, theta / sin(theta) = arcsin(s) / s.
    factor = jnp.where(small, 1.0 + s2 / 6.0 + 3.0 * s2**2 / 40.0, jnp.arctan2(s, c) / s)
    return factor * w
