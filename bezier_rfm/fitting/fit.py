"""Fit knot poses and handles to commanded poses with SciPy TRF in local Lie-group charts."""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from bezier_rfm.fitting import so3
from bezier_rfm.fitting.bezier import Controls, decode, knot_times

# Per-coordinate bound on a local rotation step, so that each step is at most 0.5 rad.
ROT_BOUND = 0.5 / np.sqrt(2)
CONTROL_DT = 0.05
TRF_OPTIONS = {"method": "trf", "loss": "linear", "x_scale": "jac", "ftol": 1e-10, "xtol": 1e-10, "gtol": 1e-8, "max_nfev": 1000}


def free_indices(k):
    """Return where q_1..q_{K-1} and H_0..H_K sit in the free-pose order (H_0, q_1, H_1, ..., H_K).

    Args:
        k (int): Number of segments K.

    Returns:
        tuple: Indices of the free knots, shape (K - 1,), and of the handles, shape (K + 1,).
    """
    q = 2 * np.arange(1, k) - 1
    h = np.concatenate([[0], 2 * np.arange(1, k), [2 * k - 1]])
    return q, h


def split_free(c, fix_h0=False):
    """Return the free poses of the controls, without q_0 and q_K (and without H_0 if it is fixed).

    Args:
        c (Controls): Knot poses and forward handles.
        fix_h0 (bool): Whether H_0 is fixed to q_0.

    Returns:
        tuple: Free positions, shape (F, 3), and rotations, shape (F, 3, 3); F = 2K, or 2K - 1 if fix_h0.
    """
    k = len(c.q_pos) - 1
    qi, hi = free_indices(k)
    pos, rot = np.zeros((2 * k, 3)), np.zeros((2 * k, 3, 3))
    pos[qi], rot[qi] = np.asarray(c.q_pos)[1:-1], np.asarray(c.q_rot)[1:-1]
    pos[hi], rot[hi] = np.asarray(c.h_pos), np.asarray(c.h_rot)
    # H_0 is index 0 of the free-pose order.
    return (pos[1:], rot[1:]) if fix_h0 else (pos, rot)


def join_free(free_pos, free_rot, ends_pos, ends_rot, fix_h0=False):
    """Insert the fixed q_0 and q_K (and H_0 = q_0 if it is fixed) and return the full controls.

    Args:
        free_pos (jnp.ndarray): Free positions, shape (F, 3).
        free_rot (jnp.ndarray): Free rotations, shape (F, 3, 3).
        ends_pos (jnp.ndarray): Positions of q_0 and q_K, shape (2, 3).
        ends_rot (jnp.ndarray): Rotations of q_0 and q_K, shape (2, 3, 3).
        fix_h0 (bool): Whether H_0 is fixed to q_0; then F = 2K - 1, else F = 2K.

    Returns:
        Controls: Knot poses and forward handles.
    """
    if fix_h0:
        # H_0 = q_0 gives a start velocity of zero.
        free_pos = jnp.concatenate([ends_pos[:1], free_pos])
        free_rot = jnp.concatenate([ends_rot[:1], free_rot])
    qi, hi = free_indices(len(free_pos) // 2)
    q_pos = jnp.concatenate([ends_pos[:1], free_pos[qi], ends_pos[1:]])
    q_rot = jnp.concatenate([ends_rot[:1], free_rot[qi], ends_rot[1:]])
    return Controls(q_pos, q_rot, free_pos[hi], free_rot[hi])


def _residuals(delta, anchor_pos, anchor_rot, ends_pos, ends_rot, h, t, p, r, scale, fix_h0=False):
    """Return the stacked residuals [p - p_hat, Log(R_hat^T R)] * scale for local chart coordinates.

    Args:
        delta (jnp.ndarray): Chart coordinates, 6 per free pose, shape (6F,).
        anchor_pos (jnp.ndarray): Chart anchor positions, shape (F, 3).
        anchor_rot (jnp.ndarray): Chart anchor rotations, shape (F, 3, 3).
        ends_pos (jnp.ndarray): Positions of q_0 and q_K, shape (2, 3).
        ends_rot (jnp.ndarray): Rotations of q_0 and q_K, shape (2, 3, 3).
        h (jnp.ndarray): Segment durations in s, shape (K,).
        t (jnp.ndarray): Sample times, shape (M,).
        p (jnp.ndarray): Target positions, shape (M, 3).
        r (jnp.ndarray): Target rotations, shape (M, 3, 3).
        scale (float): Residual scale, 1 / sqrt(N).
        fix_h0 (bool): Whether H_0 is fixed to q_0.

    Returns:
        jnp.ndarray: Residuals, shape (6M,).
    """
    d = delta.reshape(-1, 6)
    free_pos = anchor_pos + d[:, :3]
    free_rot = anchor_rot @ jax.vmap(so3.exp)(d[:, 3:])
    c = join_free(free_pos, free_rot, ends_pos, ends_rot, fix_h0)
    p_hat, r_hat = decode(c, h, t)
    rot = jax.vmap(so3.log)(jnp.swapaxes(r_hat, 1, 2) @ r)
    return scale * jnp.concatenate([p - p_hat, rot], axis=1).ravel()


residuals = jax.jit(_residuals, static_argnames="fix_h0")
jacobian = jax.jit(jax.jacfwd(_residuals), static_argnames="fix_h0")


def fit(c0, h, t, p, r, n, max_charts=20, rel_tol=1e-12, trf=None, fix_h0=False):
    """Fit the free controls by TRF, re-centering the charts after each local solve.

    Args:
        c0 (Controls): Initial controls; q_0 and q_K stay fixed (and H_0 = q_0 if fix_h0).
        h (np.ndarray): Segment durations in s, shape (K,).
        t (np.ndarray): Sample times without the two endpoints, shape (M,).
        p (np.ndarray): Target positions, shape (M, 3).
        r (np.ndarray): Target rotations, shape (M, 3, 3).
        n (int): Number of samples N including the endpoints.
        max_charts (int): Largest number of local solves.
        rel_tol (float): Stop when a solve lowers the cost by less than this fraction.
        trf (dict | None): Overrides of TRF_OPTIONS.
        fix_h0 (bool): Whether H_0 is fixed to q_0, so the start velocity is zero.

    Returns:
        tuple: Fitted Controls (NumPy arrays) and a diagnostics dict.
    """
    ends_pos = np.stack([c0.q_pos[0], c0.q_pos[-1]])
    ends_rot = np.stack([c0.q_rot[0], c0.q_rot[-1]])
    free_pos, free_rot = split_free(c0, fix_h0)
    options = {**TRF_OPTIONS, **(trf or {})}
    n_free = len(free_pos)
    ub = np.tile(np.r_[np.full(3, np.inf), np.full(3, ROT_BOUND)], n_free)
    data = (ends_pos, ends_rot, np.asarray(h, float), t, p, r, 1.0 / np.sqrt(n))
    cost = 0.5 * float(np.sum(np.asarray(residuals(np.zeros(6 * n_free), free_pos, free_rot, *data, fix_h0=fix_h0)) ** 2))
    history = [{"cost": cost}]
    for _ in range(max_charts):
        anchors = (free_pos, free_rot)
        sol = least_squares(
            lambda x: np.asarray(residuals(x, *anchors, *data, fix_h0=fix_h0)),
            np.zeros(6 * n_free),
            jac=lambda x: np.asarray(jacobian(x, *anchors, *data, fix_h0=fix_h0)),
            bounds=(-ub, ub),
            **options,
        )
        d = sol.x.reshape(-1, 6)
        free_pos = free_pos + d[:, :3]
        free_rot = free_rot @ np.asarray(jax.vmap(so3.exp)(d[:, 3:]))
        active = int(np.count_nonzero(sol.active_mask))
        history.append({"cost": float(sol.cost), "status": int(sol.status), "nfev": int(sol.nfev), "njev": int(sol.njev), "active": active})
        # Without an active bound, the solve ended inside its chart, so a new chart gives the same point.
        if active == 0 or cost - sol.cost <= rel_tol * cost:
            break
        cost = float(sol.cost)
    c = Controls(*(np.asarray(a) for a in join_free(free_pos, free_rot, ends_pos, ends_rot, fix_h0)))
    last = history[-1]
    return c, {"cost": last["cost"], "status": last["status"], "active": last["active"], "charts": len(history) - 1, "history": history}


def initialize(p, r, events, h, ks, t0=0.0):
    """Build the initial controls from one demonstration's commands (plan section 2).

    Args:
        p (np.ndarray): Commanded positions, shape (N, 3).
        r (np.ndarray): Commanded rotations, shape (N, 3, 3).
        events (list): Action indices of the event knots, length n - 1 (empty for one phase).
        h (np.ndarray): Segment durations in s, shape (K,); each phase must sum to its duration.
        ks (tuple): Number of segments K_i per phase, length n.
        t0 (float): Time of knot 0, s; the knots end at the last command.

    Returns:
        Controls: Initial controls (NumPy arrays).
    """
    n = len(p)
    bounds = [0, *events, n - 1]
    h = np.asarray(h, float)
    tau = t0 + np.asarray(knot_times(h))
    k = len(h)
    # Knot index -> action index for the knots on the action grid: the ends and every event knot.
    grid_knots = dict(zip(np.concatenate([[0], np.cumsum(ks)]).tolist(), bounds))
    if t0 > 0:
        # Knot 0 is off the grid and takes the geodesic interpolation of its two neighbouring commands.
        del grid_knots[0]
    log_rot = lambda a, b: Rotation.from_matrix(a.T @ b).as_rotvec()
    q_pos, q_rot, v = np.zeros((k + 1, 3)), np.zeros((k + 1, 3, 3)), np.zeros((k + 1, 6))
    for i, x in enumerate(tau / CONTROL_DT):
        # The ends and the event knots are on the action grid and take the command itself.
        grid = grid_knots.get(i)
        if grid is not None:
            q_pos[i], q_rot[i] = p[grid], r[grid]
            lo, hi = max(grid - 1, 0), min(grid + 1, n - 1)
        else:
            j = min(int(np.floor(x)), n - 2)
            a = x - j
            q_pos[i] = (1 - a) * p[j] + a * p[j + 1]
            q_rot[i] = r[j] @ Rotation.from_rotvec(a * log_rot(r[j], r[j + 1])).as_matrix()
            lo, hi = j, j + 1
        # One-sided at the two ends, central elsewhere.
        log_hi = np.r_[p[hi] - q_pos[i], log_rot(q_rot[i], r[hi])]
        log_lo = np.r_[p[lo] - q_pos[i], log_rot(q_rot[i], r[lo])]
        v[i] = (log_hi - log_lo) / (CONTROL_DT * (hi - lo))
    step = np.r_[h, h[-1]][:, None] / 3 * v
    h_pos = q_pos + step[:, :3]
    h_rot = q_rot @ Rotation.from_rotvec(step[:, 3:]).as_matrix()
    return Controls(q_pos, q_rot, h_pos, h_rot)


def position_lstsq(c, h, t, p):
    """Solve the position part of the fit in closed form; it is linear in the free positions.

    Args:
        c (Controls): Controls that supply q_0, q_K and the rotations.
        h (np.ndarray): Segment durations in s, shape (K,).
        t (np.ndarray): Sample times without the two endpoints, shape (M,).
        p (np.ndarray): Target positions, shape (M, 3).

    Returns:
        np.ndarray: Optimal free positions in the order of split_free, shape (2K, 3).
    """
    free_pos, free_rot = split_free(c)
    ends_pos = np.stack([c.q_pos[0], c.q_pos[-1]])
    ends_rot = np.stack([c.q_rot[0], c.q_rot[-1]])
    decoded = lambda x: decode(join_free(x, free_rot, ends_pos, ends_rot), h, t)[0].ravel()
    zero = np.zeros_like(free_pos)
    # The decoded positions are affine in the free positions, so this Jacobian is exact.
    a = np.asarray(jax.jacfwd(decoded)(zero)).reshape(-1, zero.size)
    x = np.linalg.lstsq(a, p.ravel() - np.asarray(decoded(zero)), rcond=None)[0]
    return x.reshape(zero.shape)
