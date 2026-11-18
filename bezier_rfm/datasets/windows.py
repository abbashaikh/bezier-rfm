"""Cut K-segment training windows from one clock fit per demo and grid, with a braking tail at the demo end (H27)."""

import numpy as np
from scipy.spatial.transform import Rotation

CONTROL_DT = 0.05
OBS_KEYS = ("object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")


def rot_exp(w):
    """Return rotation matrices of rotation vectors.

    Args:
        w (np.ndarray): Rotation vectors, shape (..., 3).

    Returns:
        np.ndarray: Rotation matrices, shape (..., 3, 3).
    """
    return Rotation.from_rotvec(np.reshape(w, (-1, 3))).as_matrix().reshape(*np.shape(w)[:-1], 3, 3)


def rot_log(r):
    """Return rotation vectors of rotation matrices.

    Args:
        r (np.ndarray): Rotation matrices, shape (..., 3, 3).

    Returns:
        np.ndarray: Rotation vectors, shape (..., 3).
    """
    return Rotation.from_matrix(np.reshape(r, (-1, 3, 3))).as_rotvec().reshape(*np.shape(r)[:-2], 3)


def knot_velocity(q_pos, q_rot, h_pos, h_rot, h_next):
    """Return the velocity in seconds at a knot, from its forward handle.

    Args:
        q_pos (np.ndarray): Knot position, shape (3,).
        q_rot (np.ndarray): Knot rotation, shape (3, 3).
        h_pos (np.ndarray): Forward handle position, shape (3,).
        h_rot (np.ndarray): Forward handle rotation, shape (3, 3).
        h_next (float): Duration of the segment that starts at the knot, s.

    Returns:
        np.ndarray: Linear (m/s) and body-frame angular (rad/s) velocity, shape (6,).
    """
    return 3.0 / h_next * np.r_[h_pos - q_pos, rot_log(q_rot.T @ h_rot)]


def braking_tail(q_pos, q_rot, vel, n_seg, a_max=0.8, alpha_max=1.0, max_pos=0.05, max_rot=np.radians(5.0), h_min=0.15):
    """Build n_seg tail segments that brake from vel at constant deceleration along one geodesic.

    Args:
        q_pos (np.ndarray): Last real knot position, shape (3,).
        q_rot (np.ndarray): Last real knot rotation, shape (3, 3).
        vel (np.ndarray): Velocity at the last real knot, linear and body-frame angular, shape (6,).
        n_seg (int): Number of tail segments.
        a_max (float): Linear deceleration cap, m/s^2.
        alpha_max (float): Angular deceleration cap, rad/s^2.
        max_pos (float): Largest position travel of a full stop, m.
        max_rot (float): Largest rotation travel of a full stop, rad.
        h_min (float): Shortest segment, s.

    Returns:
        tuple: Tail knots (pos (n_seg, 3), rot (n_seg, 3, 3)), their velocities (n_seg, 6), durations (n_seg,),
            and a dict with t_b, the braking segment duration and the full-stop flag.
    """
    t_b = max(np.linalg.norm(vel[:3]) / a_max, np.linalg.norm(vel[3:]) / alpha_max)
    # Full stop: one braking segment that ends exactly at the stop (at most the caps; stretched to h_min), then holds.
    h_b = max(t_b, h_min)
    full = np.linalg.norm(vel[:3]) * h_b / 2 <= max_pos and np.linalg.norm(vel[3:]) * h_b / 2 <= max_rot
    if full:
        h = np.r_[h_b, np.full(n_seg - 1, h_min)]
        stop = h_b
    else:
        h = np.full(n_seg, h_min)
        stop = t_b
        # Every tail knot must lie before the stop, so that each segment is one exact quadratic.
        if n_seg * h_min > t_b:
            raise ValueError("the braking tail would stop inside a segment")
    # Displacement along the geodesic exp_q(d(t) vel): d'(t) = 1 - t / stop until the stop, then 0.
    t = np.minimum(np.cumsum(h), stop)
    d = t - t**2 / (2 * stop)
    d_dot = 1 - t / stop
    pos = q_pos + d[:, None] * vel[:3]
    rot = q_rot @ rot_exp(d[:, None] * vel[3:])
    # On a one-parameter geodesic, the body-frame angular velocity keeps the direction of vel[3:].
    return (pos, rot), d_dot[:, None] * vel, h, {"t_b": float(t_b), "h_brake": float(h[0]), "full_stop": bool(full)}


def extend_fit(fit, n_seg, **tail_args):
    """Append a braking tail to one demo fit, keeping the curve of the real segments unchanged.

    Args:
        fit (dict): q_pos (L+1, 3), q_rot (L+1, 3, 3), H_pos (L+1, 3), H_rot (L+1, 3, 3), h (L,).
        n_seg (int): Number of tail segments.
        **tail_args: Caps and limits of braking_tail.

    Returns:
        dict: Extended q_pos, q_rot, H_pos, H_rot (L+1+n_seg, ...), h (L+n_seg,), and the tail info.
    """
    q_pos, q_rot, h = fit["q_pos"], fit["q_rot"], fit["h"]
    # The fit stores the last handle with h_L := h_{L-1}.
    vel = knot_velocity(q_pos[-1], q_rot[-1], fit["H_pos"][-1], fit["H_rot"][-1], h[-1])
    (t_pos, t_rot), t_vel, t_h, info = braking_tail(q_pos[-1], q_rot[-1], vel, n_seg, **tail_args)
    all_h = np.r_[h, t_h]
    k_pos, k_rot = np.r_[q_pos, t_pos], np.concatenate([q_rot, t_rot])
    last = len(h)
    # Handles of the last real knot and of the tail knots come from their velocities: H = exp_q((h_next / 3) v).
    # The last real handle is rebuilt for the new h_next, so its velocity, and so the last real segment, stay the same.
    h_next = np.r_[t_h, t_h[-1]]
    step = h_next[:, None] / 3 * np.r_[vel[None], t_vel]
    H_pos = np.r_[fit["H_pos"][:-1], k_pos[last:] + step[:, :3]]
    H_rot = np.concatenate([fit["H_rot"][:-1], k_rot[last:] @ rot_exp(step[:, 3:])])
    return {"q_pos": k_pos, "q_rot": k_rot, "H_pos": H_pos, "H_rot": H_rot, "h": all_h, "n_real": len(h), **info}


def cut_windows(ext, k):
    """Cut every K-segment window of an extended fit, with the last handle set for rho = 1 at the window end.

    Args:
        ext (dict): Extended fit from extend_fit.
        k (int): Segments per window K.

    Returns:
        dict: Per window (W = number of starts): q_pos (W, K+1, 3), q_rot (W, K+1, 3, 3), H_pos, H_rot,
            h (W, K), knot (W,), n_tail (W,).
    """
    q_pos, q_rot, H_pos, H_rot, h = (ext[key] for key in ("q_pos", "q_rot", "H_pos", "H_rot", "h"))
    n_real = ext["n_real"]
    starts = np.arange(len(h) - k + 1)
    idx = starts[:, None] + np.arange(k + 1)
    w = {"q_pos": q_pos[idx], "q_rot": q_rot[idx], "H_pos": H_pos[idx].copy(), "H_rot": H_rot[idx].copy(), "h": h[idx[:, :-1]]}
    # The decoder uses h_K := h_{K-1}, so rho = 1 at the window end; the stored handle is the mirror of the
    # incoming control P2 = exp_q(-rho log_q H) about q, i.e. exp_q(rho log_q H) with rho = h_{j+K-1} / h_{j+K}.
    end = starts + k
    h_after = np.r_[h, h[-1]][end]
    rho = h[end - 1] / h_after
    log_pos = H_pos[end] - q_pos[end]
    log_rot = rot_log(np.swapaxes(q_rot[end], 1, 2) @ H_rot[end])
    w["H_pos"][:, -1] = q_pos[end] + rho[:, None] * log_pos
    w["H_rot"][:, -1] = q_rot[end] @ rot_exp(rho[:, None] * log_rot)
    w["knot"] = starts
    w["n_tail"] = np.maximum(0, end - n_real)
    return w


def tail_travel(ext, w):
    """Return the position and rotation distance of each window's end knot from the last real knot.

    Args:
        ext (dict): Extended fit from extend_fit.
        w (dict): Windows from cut_windows.

    Returns:
        tuple: Position distance (m), shape (W,), and rotation distance (rad), shape (W,); 0 for windows without tail.
    """
    # The tail moves monotonically along one geodesic, so the window end is its farthest point.
    last = ext["n_real"]
    pos = np.linalg.norm(w["q_pos"][:, -1] - ext["q_pos"][last], axis=1)
    rot = np.linalg.norm(rot_log(ext["q_rot"][last].T @ w["q_rot"][:, -1]), axis=1)
    return np.where(w["n_tail"] > 0, pos, 0.0), np.where(w["n_tail"] > 0, rot, 0.0)


def window_targets(w, t0, gripper, obs, samples_per_seg=4):
    """Return the duration, logits, start velocity, gripper samples and observation pair of each window.

    Args:
        w (dict): Windows from cut_windows.
        t0 (float): Time of knot 0 of the fit in the demo, s.
        gripper (np.ndarray): Raw gripper commands at 20 Hz, shape (N,).
        obs (np.ndarray): Concatenated observations at 20 Hz, shape (N, D).
        samples_per_seg (int): Gripper samples per segment, at u = i / samples_per_seg.

    Returns:
        dict: z (W,), logits (W, K), v0 (W, 6), gripper (W, K * samples_per_seg), obs (W, 2, D), obs_dt (W,), t_start (W,).
    """
    h = w["h"]
    tau = np.concatenate([np.zeros((len(h), 1)), np.cumsum(h, axis=1)], axis=1)
    # Absolute time of each window start: knot j of the fit.
    starts_abs = t0 + np.r_[0.0, np.cumsum(w["h"][:, 0])][: len(h)]
    log_h = np.log(h)
    u = np.arange(samples_per_seg) / samples_per_seg
    t_g = starts_abs[:, None] + (tau[:, :-1, None] + u * h[:, :, None]).reshape(len(h), -1)
    n = len(gripper)
    # Latest recorded command at or before each sample time; past the demo end it is the last command.
    g_idx = np.minimum(np.floor(t_g / CONTROL_DT + 1e-9).astype(int), n - 1)
    o_idx = np.minimum(np.floor(starts_abs / CONTROL_DT + 1e-9).astype(int), n - 1)
    v0 = np.stack([knot_velocity(w["q_pos"][i, 0], w["q_rot"][i, 0], w["H_pos"][i, 0], w["H_rot"][i, 0], h[i, 0])
                   for i in range(len(h))])
    return {
        "z": np.log(h.sum(axis=1)),
        "logits": log_h - log_h.mean(axis=1, keepdims=True),
        "v0": v0,
        "gripper": gripper[g_idx],
        "obs": obs[np.stack([np.maximum(o_idx - 1, 0), o_idx], axis=1)],
        "obs_dt": starts_abs - CONTROL_DT * o_idx,
        "t_start": starts_abs,
    }
