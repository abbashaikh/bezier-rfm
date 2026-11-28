"""Place segment durations by equidistribution of a weight over time, with the gripper events as knots."""

import numpy as np
from scipy.spatial.transform import Rotation

CONTROL_DT = 0.05


def release_weight(actions, events, m=3.0, last=2.0):
    """Return weight m in the last seconds before every gripper release, 1 elsewhere.

    Args:
        actions (np.ndarray): Absolute actions, shape (N, 7).
        events (list): Action indices of the gripper events.
        m (float): Weight before a release.
        last (float): Length of the weighted window before a release, s.

    Returns:
        np.ndarray: Weight per action, shape (N,).
    """
    t = CONTROL_DT * np.arange(len(actions))
    w = np.ones(len(actions))
    for e in events:
        # A release is an event after which the gripper command is lower (open).
        if actions[e, 6] < actions[e - 1, 6]:
            w[(t >= t[e] - last) & (t <= t[e])] = m
    return w


def lever_length(actions, lengths):
    """Return the lever length L per action from the gripper state.

    Args:
        actions (np.ndarray): Absolute actions, shape (N, 7).
        lengths (tuple): L while empty, then L for each held interval in order (m).

    Returns:
        np.ndarray: L per action, shape (N,).
    """
    # A closed gripper command (> 0) holds an object; held intervals are [start, end).
    edges = np.diff(np.concatenate([[0], (actions[:, 6] > 0).astype(int), [0]]))
    L = np.full(len(actions), float(lengths[0]))
    for j, (s, e) in enumerate(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))):
        # Held intervals after the listed ones (regrasps) keep the last listed L.
        L[s:e] = lengths[1 + min(j, len(lengths) - 2)]
    return L


def speed_weight(actions, v0=0.02, smooth=5, lengths=None):
    """Return the weight 1 / (v_eff + v0) of the smoothed commanded speed, v_eff = sqrt(v^2 + (L w)^2).

    Args:
        actions (np.ndarray): Absolute actions, shape (N, 7).
        v0 (float): Speed offset, m/s.
        smooth (int): Length of the moving-average window, steps.
        lengths (tuple | None): Lever lengths for lever_length (m); None uses the linear speed only.

    Returns:
        np.ndarray: Weight per action, shape (N,).
    """
    kernel = np.ones(smooth) / smooth
    v = np.linalg.norm(np.gradient(actions[:, :3], CONTROL_DT, axis=0), axis=1)
    v = np.convolve(np.pad(v, smooth // 2, mode="edge"), kernel, mode="valid")
    if lengths is None:
        return 1.0 / (v + v0)
    # Geodesic angular speed on SO(3): central differences inside, one-sided at the ends, as np.gradient.
    # Relative rotations avoid the false jumps of the rotation vector near pi.
    rot = Rotation.from_rotvec(actions[:, 3:6])
    w = np.empty(len(actions))
    w[1:-1] = (rot[:-2].inv() * rot[2:]).magnitude() / (2 * CONTROL_DT)
    w[0] = (rot[0].inv() * rot[1]).magnitude() / CONTROL_DT
    w[-1] = (rot[-2].inv() * rot[-1]).magnitude() / CONTROL_DT
    w = np.convolve(np.pad(w, smooth // 2, mode="edge"), kernel, mode="valid")
    return 1.0 / (np.hypot(v, lever_length(actions, lengths) * w) + v0)


def allocate(mass, k):
    """Split k segments over phases in proportion to their weight, at least one per phase.

    Args:
        mass (np.ndarray): Weight integral per phase, shape (n,).
        k (int): Total number of segments.

    Returns:
        np.ndarray: Segments per phase, shape (n,).
    """
    share = k * mass / mass.sum()
    ks = np.maximum(1, np.floor(share).astype(int))
    # Largest remainders first; take from the phases with the most segments when the floor of 1 overshoots.
    while ks.sum() < k:
        ks[np.argmax(share - ks)] += 1
    while ks.sum() > k:
        ks[np.argmax(np.where(ks > 1, ks - share, -np.inf))] -= 1
    return ks


def place(w, events, k):
    """Place k segments so that each covers an equal weight integral inside its phase.

    Args:
        w (np.ndarray): Weight per action, shape (N,).
        events (list): Action indices of the event knots.
        k (int): Total number of segments.

    Returns:
        tuple: Segment durations h (s), shape (K,), and segments per phase, length n.
    """
    t = CONTROL_DT * np.arange(len(w))
    # Cumulative weight, trapezoid rule on the action grid.
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (w[1:] + w[:-1]) * CONTROL_DT)])
    bounds = [0, *events, len(w) - 1]
    ks = allocate(np.diff(cum[bounds]), k)
    tau = [0.0]
    for i, ki in enumerate(ks):
        a, b = bounds[i], bounds[i + 1]
        levels = np.linspace(cum[a], cum[b], ki + 1)[1:-1]
        tau += np.interp(levels, cum[a : b + 1], t[a : b + 1]).tolist() + [t[b]]
    return np.diff(tau), tuple(int(x) for x in ks)


def place_capped(w, events, k, h_min=0.15, iterations=60):
    """Place segments with the largest weight cap that keeps every segment at least h_min long.

    Args:
        w (np.ndarray): Weight per action, shape (N,).
        events (list): Action indices of the event knots.
        k (int): Total number of segments.
        h_min (float): Shortest allowed segment, s.
        iterations (int): Bisection steps on log(cap).

    Returns:
        tuple: Segment durations h (s), shape (K,), segments per phase, and the weight cap.
    """
    h, ks = place(w, events, k)
    if h.min() >= h_min:
        return h, ks, float(w.max())
    # A cap at min(w) gives equal segments inside each phase; bisect for the largest cap that still meets h_min.
    lo, hi = np.log(w.min()), np.log(w.max())
    best = place(np.full_like(w, w.min()), events, k)
    if best[0].min() < h_min:
        raise ValueError("h_min cannot be met even with equal segments")
    best = (*best, float(w.min()))
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        h, ks = place(np.minimum(w, np.exp(mid)), events, k)
        if h.min() >= h_min:
            lo, best = mid, (h, ks, float(np.exp(mid)))
        else:
            hi = mid
    return best
