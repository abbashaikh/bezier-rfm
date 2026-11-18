"""Knot placement for H26: the speed-rule segment clock and error-driven knot insertion (B-spline Policy)."""

import numpy as np
from scipy.interpolate import make_lsq_spline
from scipy.spatial.transform import Rotation

CONTROL_DT = 0.05


def clock_h(w, c, h_min=0.15):
    """Place knots at the integer ticks of the segment clock ds/dt = min(w, c/h_min) / c.

    Args:
        w (np.ndarray): Weight per action, shape (N,).
        c (float): Global clock constant.
        h_min (float): Shortest segment, s.

    Returns:
        np.ndarray: Segment durations h (s), shape (K,).
    """
    t = CONTROL_DT * np.arange(len(w))
    rate = np.minimum(w, c / h_min) / c
    s = np.concatenate([[0.0], np.cumsum(0.5 * (rate[1:] + rate[:-1]) * CONTROL_DT)])
    # The clock ends between two ticks; K = round(s(T)), and the clock is rescaled so that s(T) = K.
    k = max(1, int(round(s[-1])))
    tau = np.interp(s[-1] * np.arange(1, k) / k, s, t)
    return np.diff(np.concatenate([[0.0], tau, [t[-1]]]))


def grid_tau(w, c, offset, h_min=0.15):
    """Return the knot times of one shifted tick grid of the segment clock, ending at the last command.

    Args:
        w (np.ndarray): Weight per action, shape (N,).
        c (float): Global clock constant.
        offset (float): Grid offset in ticks, in [0, 1).
        h_min (float): Shortest segment, s.

    Returns:
        np.ndarray: Knot times tau (s), shape (K + 1,); tau[0] > 0 for offset > 0, tau[-1] = last command time.
    """
    t = CONTROL_DT * np.arange(len(w))
    rate = np.minimum(w, c / h_min) / c
    s = np.concatenate([[0.0], np.cumsum(0.5 * (rate[1:] + rate[:-1]) * CONTROL_DT)])
    # Same rescale as clock_h: s(T) = K, so the ticks of all grids interleave on one clock.
    k = max(1, int(round(s[-1])))
    s = s * k / s[-1]
    # The part before the first tick is dropped; the last knot is always the last command.
    tau = np.interp(np.arange(offset, k), s, t)
    # A final partial interval (offset > 0) shorter than h_min is merged into the segment before it.
    if offset > 0 and t[-1] - tau[-1] < h_min and len(tau) > 1:
        tau = tau[:-1]
    return np.concatenate([tau, [t[-1]]])


def euclidean_proxy(actions):
    """Return [p, 2q] with a sign-continuous unit quaternion, so that 2|dq| is about the rotation angle.

    Args:
        actions (np.ndarray): Absolute actions, shape (N, 7).

    Returns:
        np.ndarray: Proxy data, shape (N, 7).
    """
    q = Rotation.from_rotvec(actions[:, 3:6]).as_quat()
    for i in range(1, len(q)):
        if q[i] @ q[i - 1] < 0:
            q[i] = -q[i]
    return np.concatenate([actions[:, :3], 2.0 * q], axis=1)


def c1_residuals(y, knot_idx):
    """Fit a C1 cubic spline (double interior knots) by least squares and return the per-step squared error.

    Args:
        y (np.ndarray): Data on the action grid, shape (N, d).
        knot_idx (np.ndarray): Sorted knot indices on the grid, including 0 and N-1, shape (K+1,).

    Returns:
        np.ndarray: Squared error per step, summed over d, shape (N,).
    """
    x = CONTROL_DT * np.arange(len(y))
    tk = x[knot_idx]
    t = np.concatenate([[tk[0]] * 4, np.repeat(tk[1:-1], 2), [tk[-1]] * 4])
    spline = make_lsq_spline(x, y, t, k=3)
    return np.sum((spline(x) - y) ** 2, axis=1)


def error_knots(y, k, h_min=0.15):
    """Insert knots one at a time where the interval error sum is largest (FITPACK rule) until there are k segments.

    Args:
        y (np.ndarray): Data on the action grid, shape (N, d).
        k (int): Number of segments.
        h_min (float): Shortest segment, s.

    Returns:
        np.ndarray: Segment durations h (s), shape (K,).
    """
    n_min = int(round(h_min / CONTROL_DT))
    knots = np.array([0, len(y) - 1])
    while len(knots) - 1 < k:
        r = c1_residuals(y, knots)
        # Interval j holds the steps knots[j] <= i < knots[j+1]; the last step belongs to the last interval.
        sums = np.add.reduceat(r[:-1], knots[:-1])
        sums[-1] += r[-1]
        # The new knot is the middle grid step of the interval (fpknot); both halves must keep h >= h_min.
        sums[np.diff(knots) < 2 * n_min] = -np.inf
        j = int(np.argmax(sums))
        if not np.isfinite(sums[j]):
            raise ValueError("no interval can be split without breaking h_min")
        knots = np.insert(knots, j + 1, (knots[j] + knots[j + 1]) // 2)
    return CONTROL_DT * np.diff(knots)


def step_density(h, n):
    """Return the knot density 1/h of the segment that contains each action step.

    Args:
        h (np.ndarray): Segment durations (s), shape (K,).
        n (int): Number of action steps N.

    Returns:
        np.ndarray: Segments per second at each step, shape (N,).
    """
    tau = np.concatenate([[0.0], np.cumsum(h)])
    k = np.clip(np.searchsorted(tau, CONTROL_DT * np.arange(n), side="right") - 1, 0, len(h) - 1)
    return 1.0 / h[k]


def dense_intervals(d_s, d_e, ratio=2.0, min_len=0.5):
    """Return the step intervals where the E density is at least ratio times the S density for at least min_len.

    Args:
        d_s (np.ndarray): S density per step, shape (N,).
        d_e (np.ndarray): E density per step, shape (N,).
        ratio (float): Smallest E/S density ratio.
        min_len (float): Shortest interval, s.

    Returns:
        list: (start, end) step indices, end exclusive.
    """
    edges = np.flatnonzero(np.diff(np.concatenate([[0], (d_e >= ratio * d_s).astype(int), [0]])))
    return [(a, b) for a, b in zip(edges[::2], edges[1::2]) if (b - a) * CONTROL_DT >= min_len]
