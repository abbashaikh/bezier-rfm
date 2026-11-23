"""Receding execution of K-segment windows as 20 Hz absolute OSC commands, with m = 1 (plan 5.2, 9.2)."""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial.transform import Rotation

from bezier_rfm.datasets.windows import CONTROL_DT, knot_velocity, rot_exp
from bezier_rfm.fitting import bezier

# Grid times within this distance of a knot count as on the knot.
TIME_TOL = 1e-9
# Compiled once per number of command times, which takes only a few values.
_decode = jax.jit(bezier.decode)


def window_durations(z, logits):
    """Return the segment durations h = T softmax(logits).

    Args:
        z (float): Log of the total duration T in s.
        logits (np.ndarray): Segment logits, shape (K,).

    Returns:
        np.ndarray: Segment durations in s, shape (K,).
    """
    e = np.exp(logits - np.max(logits))
    return np.exp(z) * e / e.sum()


def window_controls(window, q0_pos, q0_rot, v0):
    """Build the controls of one window, with q_0 from the caller and H_0 = exp_{q0}((h_0 / 3) v_0).

    Args:
        window (dict): q_pos (K+1, 3), q_rot (K+1, 3, 3), H_pos (K+1, 3), H_rot (K+1, 3, 3), z (), logits (K,);
            index 0 of the poses is ignored.
        q0_pos (np.ndarray): Start position, shape (3,).
        q0_rot (np.ndarray): Start rotation, shape (3, 3).
        v0 (np.ndarray): Start velocity, linear (m/s) and body-frame angular (rad/s), shape (6,).

    Returns:
        tuple: bezier.Controls and the durations h, shape (K,).
    """
    h = window_durations(window["z"], window["logits"])
    step = h[0] / 3 * np.asarray(v0)
    c = bezier.Controls(
        q_pos=np.concatenate([q0_pos[None], window["q_pos"][1:]]),
        q_rot=np.concatenate([q0_rot[None], window["q_rot"][1:]]),
        h_pos=np.concatenate([(q0_pos + step[:3])[None], window["H_pos"][1:]]),
        h_rot=np.concatenate([(q0_rot @ rot_exp(step[3:]))[None], window["H_rot"][1:]]),
    )
    return c, h


def grid_steps(t_from, t_to, closed):
    """Return the 20 Hz command indices with times in [t_from, t_to), or [t_from, t_to] when closed.

    Args:
        t_from (float): Start time in s.
        t_to (float): End time in s.
        closed (bool): Include a command at t_to.

    Returns:
        np.ndarray: Command indices, shape (M,).
    """
    first = int(np.ceil(t_from / CONTROL_DT - TIME_TOL))
    last = int(np.floor(t_to / CONTROL_DT + TIME_TOL)) if closed else int(np.ceil(t_to / CONTROL_DT - TIME_TOL)) - 1
    return np.arange(first, last + 1)


def gripper_at(t_rel, h, samples):
    """Return the latest gripper sample at or before each time, thresholded at zero.

    Args:
        t_rel (np.ndarray): Times from the window start in s, shape (M,).
        h (np.ndarray): Segment durations in s, shape (K,).
        samples (np.ndarray): Gripper samples, shape (K * S,), at u = i / S of each segment.

    Returns:
        np.ndarray: Gripper commands in {-1, 1}, shape (M,).
    """
    per_seg = len(samples) // len(h)
    tau = np.r_[0.0, np.cumsum(h)]
    t_s = (tau[:-1, None] + np.arange(per_seg) / per_seg * h[:, None]).reshape(-1)
    idx = np.searchsorted(t_s, t_rel + TIME_TOL, side="right") - 1
    return np.where(samples[np.maximum(idx, 0)] > 0, 1.0, -1.0)


def to_commands(pos, rot, grip):
    """Pack poses and gripper values into absolute OSC actions.

    Args:
        pos (np.ndarray): Positions, shape (M, 3).
        rot (np.ndarray): Rotations, shape (M, 3, 3).
        grip (np.ndarray): Gripper commands, shape (M,).

    Returns:
        np.ndarray: Actions [pos, rotation vector, gripper], shape (M, 7).
    """
    # Same rotation-vector convention as scripts/convert_abs.py.
    return np.concatenate([pos, Rotation.from_matrix(rot).as_rotvec(), grip[:, None]], axis=1)


class Executor:
    """Turn one window per knot into 20 Hz commands: execute segment 0, hand knot 1 to the next window."""

    def __init__(self, q0_pos, q0_rot, v0=None, t0=0.0):
        """Start at a pose and velocity at time t0.

        Args:
            q0_pos (np.ndarray): Start position, shape (3,).
            q0_rot (np.ndarray): Start rotation, shape (3, 3).
            v0 (np.ndarray): Start velocity, shape (6,); None means rest.
            t0 (float): Time of the start knot in s.

        Returns:
            None: Nothing.
        """
        self.q_pos, self.q_rot = np.asarray(q0_pos, float), np.asarray(q0_rot, float)
        self.v = np.zeros(6) if v0 is None else np.asarray(v0, float)
        self.t = float(t0)
        self.log = []

    def obs_frames(self):
        """Return the observation indices for the next window: the latest frame at or before the start knot and the one before.

        Args:
            None: Nothing.

        Returns:
            tuple: Frame indices (2,) and obs_dt, the start time minus the latest frame time in s.
        """
        o = int(np.floor(self.t / CONTROL_DT + TIME_TOL))
        return np.array([max(o - 1, 0), o]), self.t - CONTROL_DT * o

    def step(self, window, last=False):
        """Execute segment 0 of a window (all segments when last) and move the start knot to knot 1.

        Args:
            window (dict): Window poses, z, logits and gripper samples (see window_controls).
            last (bool): Execute the whole window and close the end time.

        Returns:
            tuple: Command indices (M,) and absolute actions (M, 7).
        """
        c, h = window_controls(window, self.q_pos, self.q_rot, self.v)
        n_seg = len(h) if last else 1
        t_end = self.t + h[:n_seg].sum()
        idx = grid_steps(self.t, t_end, closed=last)
        t_rel = CONTROL_DT * idx - self.t
        pos, rot = (np.asarray(a) for a in _decode(c, jnp.asarray(h), jnp.asarray(t_rel)))
        cmds = to_commands(pos, rot, gripper_at(t_rel, h, window["gripper"]))
        self.log.append({"t": self.t, "controls": c, "h": h})
        # Hand the planned pose and velocity at knot 1 to the next window; for knot 1 the next duration is h_1.
        k = n_seg
        h_next = h[k] if k < len(h) else h[-1]
        self.v = knot_velocity(c.q_pos[k], c.q_rot[k], c.h_pos[k], c.h_rot[k], h_next)
        self.q_pos, self.q_rot, self.t = c.q_pos[k], c.q_rot[k], t_end
        return idx, cmds
