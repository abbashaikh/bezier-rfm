"""Plot pose, error, and gripper-cube diagnostics for one Lift demonstration."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from bezier_rfm.eval.lift_diagnostics import CONTROL_DT, achieved_errors, phase_labels, tracking_errors

# Fixed colour per trace slot: recording first, then each evaluated trace in order.
SLOT_COLORS = ["#555555", "#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
PHASE_SHADES = {"first": "#f2f2f2", "approach": "#ffffff", "grasp": "#e8f0fb", "lift": "#fdeee6", "final": "#f2f2f2"}


def state_times(trace):
    """Return the time of every state row.

    Args:
        trace (dict): Trace arrays.

    Returns:
        np.ndarray: Times in seconds from the first state, shape (N + 1,).
    """
    return CONTROL_DT * np.arange(len(trace["valid"]))


def shade_phases(ax, recorded, phases):
    """Shade the task phase of every action interval and label each phase once.

    Args:
        ax (matplotlib.axes.Axes): Target axes.
        recorded (dict): Recorded trace.
        phases (np.ndarray): Phase label per action, shape (N,).

    Returns:
        None: Nothing.
    """
    starts = np.r_[0, np.flatnonzero(phases[1:] != phases[:-1]) + 1]
    ends = np.r_[starts[1:], len(phases)]
    for s, e in zip(starts, ends):
        ax.axvspan(recorded["t_start"][s], recorded["t_end"][e - 1], color=PHASE_SHADES[phases[s]], zorder=0, lw=0)
        ax.axvline(recorded["t_start"][s], color="#cccccc", lw=0.8, zorder=0)
        if phases[s] in ("approach", "grasp", "lift"):
            ax.text(recorded["t_start"][s] + 0.01, 0.98, phases[s], transform=ax.get_xaxis_transform(), va="top", fontsize=7, color="#777777")


def same_commands(trace, recorded):
    """Return whether a trace uses exactly the recorded commands.

    Args:
        trace (dict): Evaluated trace.
        recorded (dict): Recorded trace.

    Returns:
        bool: True when positions, rotations, and gripper commands are equal.
    """
    return all(np.array_equal(trace[k], recorded[k]) for k in ("cmd_pos", "cmd_rot", "gripper_cmd"))


def plot_command(ax, trace, values, color, label):
    """Draw a command as a step held over each action interval.

    Args:
        ax (matplotlib.axes.Axes): Target axes.
        trace (dict): Trace with t_start and t_end.
        values (np.ndarray): One value per action, shape (N,).
        color (str): Line colour.
        label (str): Legend label.

    Returns:
        None: Nothing.
    """
    ax.step(np.r_[trace["t_start"], trace["t_end"][-1]], np.r_[values, values[-1]], where="post", color=color, lw=1.2, ls="--", label=label)


def plot_states(ax, trace, values, color, label, ms=3):
    """Draw per-state values as dots joined by a thin line, skipping invalid states.

    Args:
        ax (matplotlib.axes.Axes): Target axes.
        trace (dict): Trace arrays.
        values (np.ndarray): One value per state, shape (N + 1,).
        color (str): Line colour.
        label (str): Legend label.
        ms (float): Marker size.

    Returns:
        None: Nothing.
    """
    ok = trace["valid"]
    ax.plot(state_times(trace)[ok], values[ok], "o-", color=color, ms=ms, lw=1, label=label)


def relative_rotvec(reference, rot):
    """Return the rotation vector of R_ref^T R for every row.

    Args:
        reference (np.ndarray): Reference rotation matrix, shape (3, 3).
        rot (np.ndarray): Rotation matrices, NaN rows allowed, shape (M, 3, 3).

    Returns:
        np.ndarray: Rotation vectors in degrees, NaN for NaN rows, shape (M, 3).
    """
    out = np.full((len(rot), 3), np.nan)
    ok = np.isfinite(rot).all(axis=(1, 2))
    out[ok] = np.degrees(Rotation.from_matrix(reference.T @ rot[ok]).as_rotvec())
    return out


def plot_pose(path, name, recorded, traces):
    """Save the pose figure: commanded and achieved position and orientation over time.

    Args:
        path (Path): Output image path.
        name (str): Demonstration name.
        recorded (dict): Recorded trace.
        traces (dict): Label -> evaluated trace, in slot order.

    Returns:
        None: Nothing.
    """
    phases = phase_labels(recorded, len(recorded["cmd_pos"]))
    ref = recorded["ach_rot"][0]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
    for col in range(3):
        for row in range(2):
            shade_phases(axes[row, col], recorded, phases)
        axes[0, col].set_ylabel(f"{'xyz'[col]} (m)")
        axes[1, col].set_ylabel(f"rotation vector {'xyz'[col]} (deg)")
    for slot, (label, trace) in enumerate([("recorded", recorded), *traces.items()]):
        color = SLOT_COLORS[slot]
        rot_cmd, rot_ach = relative_rotvec(ref, trace["cmd_rot"]), relative_rotvec(ref, trace["ach_rot"])
        for col in range(3):
            # A trace that replays the recorded commands would draw the same line again.
            if slot == 0 or not same_commands(trace, recorded):
                plot_command(axes[0, col], trace, trace["cmd_pos"][:, col], color, f"command ({label})")
                plot_command(axes[1, col], trace, rot_cmd[:, col], color, f"command ({label})")
            # Larger recorded markers stay visible under a replay that follows the recording closely.
            ms = 6 if slot == 0 else 3
            plot_states(axes[0, col], trace, trace["ach_pos"][:, col], color, f"achieved ({label})", ms)
            plot_states(axes[1, col], trace, rot_ach[:, col], color, f"achieved ({label})", ms)
    for ax in axes[1]:
        ax.set_xlabel("time (s)")
    axes[0, 0].legend(fontsize=7, loc="best")
    fig.suptitle(
        f"{name}: grip-site pose. Dashed steps: command held over its 0.05 s interval. Dots: achieved pose at each state.\n"
        "Orientation is the rotation vector of R_0^T R, with R_0 the recorded orientation at t = 0."
    )
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_errors(path, name, recorded, traces):
    """Save the error figure: tracking error and difference from the recording over time.

    Args:
        path (Path): Output image path.
        name (str): Demonstration name.
        recorded (dict): Recorded trace.
        traces (dict): Label -> evaluated trace, in slot order.

    Returns:
        None: Nothing.
    """
    n = len(recorded["cmd_pos"])
    phases = phase_labels(recorded, n)
    # Each error belongs to action t and is measured at state t + 1, the end of its interval.
    t_end = recorded["t_end"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 7), sharex=True)
    track = {"recorded": tracking_errors(recorded), **{label: tracking_errors(trace) for label, trace in traces.items()}}
    # The recording was driven by the stale goal at t = 0, so its tracking error there is not meaningful.
    track["recorded"][0] = np.nan
    panels = [
        (axes[0, 0], "tracking position error (mm)", {k: v[:, 0] * 1000 for k, v in track.items()}, False),
        (axes[0, 1], "tracking orientation error (deg)", {k: np.degrees(v[:, 1]) for k, v in track.items()}, False),
        (axes[1, 0], "achieved position vs recording (m)", {k: achieved_errors(t, recorded)[:, 0] for k, t in traces.items()}, True),
        (axes[1, 1], "achieved orientation vs recording (rad)", {k: achieved_errors(t, recorded)[:, 1] for k, t in traces.items()}, True),
    ]
    contact = recorded["contact"][1:]
    for ax, ylabel, series, log in panels:
        for t in np.flatnonzero(contact):
            ax.axvspan(recorded["t_start"][t], t_end[t], color="#ffe9b3", lw=0, zorder=0)
        for t in range(1, n):
            if phases[t] != phases[t - 1]:
                ax.axvline(recorded["t_start"][t], color="#cccccc", lw=0.8, zorder=0)
                ax.text(recorded["t_start"][t] + 0.01, 0.98, phases[t], transform=ax.get_xaxis_transform(), va="top", fontsize=7, color="#777777")
        for label, values in series.items():
            slot = 0 if label == "recorded" else 1 + list(traces).index(label)
            ax.plot(t_end, np.maximum(values, 1e-16) if log else values, "o-", color=SLOT_COLORS[slot], ms=3, lw=1, label=label)
            if np.isfinite(values).any():
                worst = int(np.nanargmax(values))
                ax.plot(t_end[worst], values[worst], "v", color=SLOT_COLORS[slot], ms=9, mec="black")
        if log:
            ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("time of the achieved state t + 1 (s)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"{name}: errors of action t, measured at the end of its interval. Shaded: gripper touches the cube. Triangles: maximum. Values below 1e-16 drawn at 1e-16.")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_gripper_cube(path, name, recorded, traces):
    """Save the gripper and cube figure: gripper command, finger opening, cube height, and gripper-to-cube distance.

    Args:
        path (Path): Output image path.
        name (str): Demonstration name.
        recorded (dict): Recorded trace.
        traces (dict): Label -> evaluated trace, in slot order.

    Returns:
        None: Nothing.
    """
    phases = phase_labels(recorded, len(recorded["cmd_pos"]))
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    for ax in axes:
        shade_phases(ax, recorded, phases)
        ax.grid(alpha=0.3)
    for slot, (label, trace) in enumerate([("recorded", recorded), *traces.items()]):
        color = SLOT_COLORS[slot]
        if slot == 0 or not same_commands(trace, recorded):
            plot_command(axes[0], trace, trace["gripper_cmd"], color, label)
        ms = 6 if slot == 0 else 3
        plot_states(axes[1], trace, (trace["finger_q"][:, 0] - trace["finger_q"][:, 1]) * 100, color, label, ms)
        plot_states(axes[2], trace, trace["cube_height"] * 100, color, label, ms)
        plot_states(axes[3], trace, trace["gripper_to_cube"] * 100, color, label, ms)
    axes[2].axhline(4, color="#999999", ls=":", lw=1, label="lift rule (4 cm)")
    for ax, ylabel in zip(axes, ("gripper command (-1 open, 1 close)", "finger opening (cm)", "cube height above table (cm)", "grip site to cube (cm)")):
        ax.set_ylabel(ylabel)
    axes[-1].set_xlabel("time (s)")
    axes[2].legend(fontsize=8, loc="center left")
    fig.suptitle(f"{name}: gripper and cube. Gripper command held over each 0.05 s interval. Dots: state values.")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_diagnostics(out_dir, name, recorded, traces):
    """Save the pose, error, and gripper-cube figures of one demonstration.

    Args:
        out_dir (Path): Output folder.
        name (str): Demonstration name.
        recorded (dict): Recorded trace.
        traces (dict): Label -> evaluated trace, in slot order (at most 4).

    Returns:
        list: Paths of the saved figures.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{name}_{kind}.png" for kind in ("pose", "errors", "gripper_cube")]
    plot_pose(paths[0], name, recorded, traces)
    plot_errors(paths[1], name, recorded, traces)
    plot_gripper_cube(paths[2], name, recorded, traces)
    return paths
