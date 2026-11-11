"""Replay whole Lift demonstrations under delta and absolute actions without restoring intermediate states."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from bezier_rfm.datasets.robomimic import arm_controller, demo_keys, load_demo_model, make_env, read_result, require
from bezier_rfm.eval.lift_diagnostics import PHASES, phase_labels
from experiment_a_one_step import pose_errors
from inspect_lift_demo import table

EPISODES = ["delta_stale", "delta_refreshed", "absolute_stale", "absolute_refreshed"]
STARTS = ["stale", "refreshed"]
PAIRS = ["recording_vs_delta", "recording_vs_absolute", "delta_vs_absolute"]
ERRORS = ["eef_pos_m", "eef_rot_rad", "cube_pos_m"]
GOAL_TOL = 1e-8
CUBE_POS = slice(10, 13)
FINGER_Q = slice(8, 10)
COLORS = {"recording": "#555555", "delta": "#2a78d6", "absolute": "#eb6834", "delta_vs_absolute": "#1baf7a"}


def read_frame(env):
    """Read the state, grip-site pose, contact, and success flag.

    Args:
        env (EnvRobosuite): The environment wrapper.

    Returns:
        dict: Output of read_result, plus contact (bool) and success (bool).
    """
    frame = read_result(env)
    base = env.env
    frame["contact"] = bool(base.check_contact(base.robots[0].gripper["right"], base.cube))
    frame["success"] = bool(base._check_success())
    return frame


def stack(frames):
    """Stack a list of frames into arrays.

    Args:
        frames (list): Outputs of read_frame.

    Returns:
        dict: Arrays with one leading row per frame.
    """
    return {key: np.array([f[key] for f in frames]) for key in frames[0]}


def render(renderer, env):
    """Render the front camera for the current simulator data.

    Args:
        renderer (mujoco.Renderer | None): Renderer for this model, or None.
        env (EnvRobosuite): The environment wrapper.

    Returns:
        np.ndarray | None: RGB image, shape (H, W, 3), or None.
    """
    if renderer is None:
        return None
    renderer.update_scene(env.env.sim.data._data, camera="frontview")
    return renderer.render().copy()


def make_renderer(env, video):
    """Create a renderer for the currently loaded model.

    Args:
        env (EnvRobosuite): The environment wrapper.
        video (bool): Whether to render.

    Returns:
        mujoco.Renderer | None: The renderer, or None.
    """
    if not video:
        return None
    import mujoco

    return mujoco.Renderer(env.env.sim.model._model, 480, 480)


def record_trajectory(env, demo, video):
    """Read every recorded state without stepping.

    Args:
        env (EnvRobosuite): The environment wrapper.
        demo (h5py.Group): One stored demonstration.
        video (bool): Whether to render images.

    Returns:
        tuple: Stacked frames, length N, and a list of images.
    """
    load_demo_model(env, demo)
    renderer = make_renderer(env, video)
    frames, images = [], []
    for state in demo["states"][:]:
        env.reset_to({"states": state})
        frames.append(read_frame(env))
        images.append(render(renderer, env))
    return stack(frames), images


def replay(env, demo, actions, refresh, video):
    """Restore s_0 once and run all actions in sequence.

    Args:
        env (EnvRobosuite): The environment wrapper.
        demo (h5py.Group): One stored demonstration.
        actions (np.ndarray): Actions, shape (N, 7).
        refresh (bool): Whether the controller rereads the pose before the first step.
        video (bool): Whether to render images.

    Returns:
        tuple: Stacked frames, length N + 1, logged goals, and a list of images.
    """
    load_demo_model(env, demo)
    if refresh:
        # After a model load, new_update stays False, so the first step would use a pose from before reset_to.
        arm_controller(env).new_update = True
    renderer = make_renderer(env, video)
    frames, images = [read_frame(env)], [render(renderer, env)]
    goal_pos, goal_rot = [], []
    for action in actions:
        env.step(action)
        controller = arm_controller(env)
        goal_pos.append(controller.goal_pos.copy())
        goal_rot.append(controller.goal_ori.copy())
        frames.append(read_frame(env))
        images.append(render(renderer, env))
    return stack(frames), {"pos": np.array(goal_pos), "rot": np.array(goal_rot)}, images


def trajectory_errors(a, b):
    """Return per-row grip-site and cube errors between two aligned trajectories.

    Args:
        a (dict): Stacked frames, M rows.
        b (dict): Stacked frames, M rows.

    Returns:
        np.ndarray: Errors per entry of ERRORS, shape (M, 3).
    """
    rows = []
    for i in range(len(a["pos"])):
        cube = np.linalg.norm(a["state"][i, CUBE_POS] - b["state"][i, CUBE_POS])
        rows.append(pose_errors(a["pos"][i], a["rot"][i], b["pos"][i], b["rot"][i]) + [float(cube)])
    return np.array(rows)


def goal_gaps(goals, stored):
    """Return the distance between logged controller goals and stored absolute actions.

    Args:
        goals (dict): Logged goal positions, shape (N, 3), and rotations, shape (N, 3, 3).
        stored (np.ndarray): Stored absolute actions, shape (N, 7).

    Returns:
        np.ndarray: Position gap in meters and rotation gap in radians, shape (N, 2).
    """
    stored_rot = Rotation.from_rotvec(stored[:, 3:6]).as_matrix()
    return np.array([pose_errors(goals["pos"][t], goals["rot"][t], stored[t, :3], stored_rot[t]) for t in range(len(stored))])




def run_demo(envs, source_demo, abs_demo, video):
    """Run the recording read and the four continuous replays of one demonstration.

    Args:
        envs (dict): Environments named delta and absolute.
        source_demo (h5py.Group): Demonstration with delta actions.
        abs_demo (h5py.Group): Demonstration with absolute actions.
        video (bool): Whether to render images.

    Returns:
        dict: Trajectories, errors, goal gaps, phases, and images.
    """
    delta, absolute = source_demo["actions"][:], abs_demo["actions"][:]
    require(np.array_equal(source_demo["states"][:], abs_demo["states"][:]), "the two files hold different states")
    n = len(delta)
    recording, rec_images = record_trajectory(envs["delta"], source_demo, video)
    runs, goals, images = {}, {}, {"recording": rec_images}
    for name in EPISODES:
        kind, start = name.split("_")
        actions = delta if kind == "delta" else absolute
        runs[name], goals[name], images[name] = replay(envs[kind], source_demo, actions, start == "refreshed", video)

    errors = {}
    for start in STARTS:
        d, a = runs[f"delta_{start}"], runs[f"absolute_{start}"]
        after = lambda run: {key: value[1:] for key, value in run.items()}
        # The file has no recorded state after the final action, so that row is NaN.
        for kind, run in (("delta", d), ("absolute", a)):
            rec = np.full((n, len(ERRORS)), np.nan)
            rec[:-1] = trajectory_errors({k: v[1:] for k, v in recording.items()}, {k: v[1:n] for k, v in run.items()})
            errors[f"{start}/recording_vs_{kind}"] = rec
        errors[f"{start}/delta_vs_absolute"] = trajectory_errors(after(d), after(a))
    return {
        "n": n,
        "recording": recording,
        "runs": runs,
        "goals": goals,
        "errors": errors,
        "goal_gaps": {name: goal_gaps(goals[name], absolute) for name in EPISODES},
        "phase": phase_labels(recording, n),
        "absolute": absolute,
        "images": images,
    }


def write_video(path, result):
    """Write the recording, delta, and absolute stale-start replays side by side.

    Args:
        path (Path): Output .mp4 path.
        result (dict): Output of run_demo with images.

    Returns:
        None: Nothing.
    """
    import cv2
    import imageio

    panels = [("recording", "Recording (states)"), ("delta_stale", "Delta replay"), ("absolute_stale", "Absolute replay")]
    frames = []
    for t in range(result["n"] + 1):
        row = []
        for key, label in panels:
            seq = result["images"][key]
            # The recording has N states, so its last image is held for the final frame.
            image = seq[min(t, len(seq) - 1)].copy()
            cv2.putText(image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(image, f"t = {t}", (10, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            row.append(image)
        frames.append(np.concatenate(row, axis=1))
    imageio.mimwrite(path, frames, fps=20, macro_block_size=1)


def phase_lines(ax, phase):
    """Draw vertical lines where the task phase changes.

    Args:
        ax (matplotlib.axes.Axes): Target axes.
        phase (np.ndarray): Phase label per action, shape (N,).

    Returns:
        None: Nothing.
    """
    for t in range(1, len(phase)):
        if phase[t] != phase[t - 1]:
            ax.axvline(t - 0.5, color="#bbbbbb", lw=1, ls="--")
            ax.text(t - 0.4, 1.0, phase[t], transform=ax.get_xaxis_transform(), va="top", fontsize=7, color="#777777")


def plot_demo(out_dir, name, result, table_height):
    """Save error, gripper, cube, and drift plots for one demonstration.

    Args:
        out_dir (Path): Output folder.
        name (str): Demonstration name.
        result (dict): Output of run_demo.
        table_height (float): Table-top height in meters.

    Returns:
        list: Paths of the saved figures.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(result["n"])
    phase = result["phase"]
    paths = []

    fig, axes = plt.subplots(len(ERRORS), 2, figsize=(12, 8), sharex=True, sharey="row")
    for col, start in enumerate(STARTS):
        for row, metric in enumerate(ERRORS):
            ax = axes[row, col]
            for pair, color in zip(PAIRS, (COLORS["delta"], COLORS["absolute"], COLORS["delta_vs_absolute"])):
                ax.plot(t, np.maximum(result["errors"][f"{start}/{pair}"][:, row], 1e-16), color=color, lw=2, label=pair.replace("_", " "))
            ax.set_yscale("log")
            ax.grid(alpha=0.3)
            phase_lines(ax, phase)
            if col == 0:
                ax.set_ylabel(metric)
        axes[0, col].set_title(f"{start} start")
        axes[-1, col].set_xlabel("action t (compared after the step)")
    axes[0, 0].legend(fontsize=8, loc="lower right")
    fig.suptitle(f"{name}: errors over time (values below 1e-16 drawn at 1e-16)")
    fig.tight_layout()
    paths.append(out_dir / f"{name}_errors.png")
    fig.savefig(paths[-1], dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharex=True, sharey="row")
    rec = result["recording"]
    k = np.arange(len(rec["state"]))
    for col, start in enumerate(STARTS):
        for row, (label, read) in enumerate(
            (("finger opening (m)", lambda s: s[:, FINGER_Q][:, 0] - s[:, FINGER_Q][:, 1]), ("cube height above table (m)", lambda s: s[:, 12] - table_height))
        ):
            ax = axes[row, col]
            ax.plot(k, read(rec["state"]), color=COLORS["recording"], lw=2, label="recording")
            for kind in ("delta", "absolute"):
                run = result["runs"][f"{kind}_{start}"]
                ax.plot(np.arange(len(run["state"])), read(run["state"]), color=COLORS[kind], lw=2, ls="--" if kind == "absolute" else "-", label=kind)
            if row == 1:
                ax.axhline(0.04, color="#999999", lw=1, ls=":")
            ax.grid(alpha=0.3)
            phase_lines(ax, phase)
            if col == 0:
                ax.set_ylabel(label)
        axes[0, col].set_title(f"{start} start")
        axes[-1, col].set_xlabel("state index")
    axes[1, 0].legend(fontsize=8, loc="upper left")
    fig.suptitle(f"{name}: gripper and cube (dotted line: lift rule, 0.04 m)")
    fig.tight_layout()
    paths.append(out_dir / f"{name}_gripper_cube.png")
    fig.savefig(paths[-1], dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for col, start in enumerate(STARTS):
        ax = axes[col]
        gap = result["goal_gaps"][f"delta_{start}"][:, 0]
        ax.plot(t, np.maximum(gap, 1e-16), color=COLORS["delta"], lw=2, label="delta goal - stored absolute goal")
        # Achieved-pose drift before action t is the pose error after action t - 1.
        drift = np.r_[np.nan, result["errors"][f"{start}/recording_vs_delta"][:-1, 0]]
        ax.plot(t, np.maximum(drift, 1e-16), color=COLORS["recording"], marker="o", ms=4, lw=0, label="pose drift before the step")
        interface = result["goal_gaps"][f"absolute_{start}"][:, 0]
        ax.plot(t, np.maximum(interface, 1e-16), color=COLORS["absolute"], lw=2, label="absolute goal - stored absolute goal")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        phase_lines(ax, phase)
        ax.set_title(f"{start} start")
        ax.set_xlabel("action t")
    axes[0].set_ylabel("goal position gap (m)")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle(f"{name}: logged controller goals (values below 1e-16 drawn at 1e-16)")
    fig.tight_layout()
    paths.append(out_dir / f"{name}_goals.png")
    fig.savefig(paths[-1], dpi=120)
    plt.close(fig)
    return paths


def outcome(result, table_height):
    """Summarize lift success and final cube height for one demonstration.

    Args:
        result (dict): Output of run_demo.
        table_height (float): Table-top height in meters.

    Returns:
        dict: Recording success, and per-episode final success, ever success, and final cube height.
    """
    out = {"recording_final": bool(result["recording"]["success"][-1])}
    for name in EPISODES:
        run = result["runs"][name]
        out[f"{name}_final"] = bool(run["success"][-1])
        out[f"{name}_ever"] = bool(run["success"].any())
        out[f"{name}_height"] = float(run["state"][-1, 12] - table_height)
    return out


def phase_rows(names, results):
    """Build median, 95th-percentile, and maximum rows per start, pair, and phase.

    Args:
        names (list): Demonstration names.
        results (list): Outputs of run_demo.

    Returns:
        list: Table rows.
    """
    phase = np.concatenate([r["phase"] for r in results])
    rows = []
    for start in STARTS:
        for pair in PAIRS:
            values = np.concatenate([r["errors"][f"{start}/{pair}"] for r in results])
            for p in PHASES:
                block = values[phase == p]
                if np.isnan(block).all():
                    continue
                cells = []
                for j in range(len(ERRORS)):
                    col = block[:, j]
                    cells.append(f"{np.nanmedian(col):.1e} / {np.nanpercentile(col, 95):.1e} / {np.nanmax(col):.1e}")
                rows.append([start, pair.replace("_", " "), p, int((~np.isnan(block[:, 0])).sum()), *cells])
    return rows


def write_report(path, names, results, outcomes, figures, video):
    """Write the Experiment B Markdown report.

    Args:
        path (Path): Output report path.
        names (list): Demonstration names.
        results (list): Outputs of run_demo.
        outcomes (list): Outputs of outcome.
        figures (list): Saved figure paths.
        video (Path | None): Saved video path.

    Returns:
        None: Nothing.
    """
    lines = [
        "# Experiment B: continuous replay",
        "",
        f"Demonstrations: {len(names)}. Control steps: {sum(r['n'] for r in results)}. "
        "Each replay restores s_0 once with the stored model and then runs every action in sequence.",
        "",
        "Starts: **stale** steps right after the model load, as the recording appears to have done. "
        "**refreshed** sets `new_update = True` first, so the first step reads the restored pose.",
        "",
        "Recorded poses come from the recorded states with `sim.forward()`. Replay pose after action t is compared with recorded state t + 1. "
        "The final action has no recorded next state, so it is judged by lift success and cube height only.",
        "",
        "## Lift success",
        "",
    ]
    rec_ok = [o["recording_final"] for o in outcomes]
    lines.append(f"Recorded final state meets the lift rule: {sum(rec_ok)} of {len(names)}.")
    lines.append("")
    rows = []
    for name in EPISODES:
        heights = np.array([o[f"{name}_height"] for o in outcomes])
        rows.append([name, sum(o[f"{name}_final"] for o in outcomes), sum(o[f"{name}_ever"] for o in outcomes), f"{np.median(heights):.4f}", f"{heights.min():.4f}"])
    lines.append(table(["episode", "lifted at final step", "lifted at any step", "median final height (m)", "min final height (m)"], rows))
    lines.append("")

    lines += ["## H17 check", ""]
    for start in STARTS:
        broken = [n for n, o in zip(names, outcomes) if o["recording_final"] and o[f"delta_{start}_final"] and not o[f"absolute_{start}_final"]]
        other = [n for n, o in zip(names, outcomes) if o["recording_final"] and o[f"absolute_{start}_final"] and not o[f"delta_{start}_final"]]
        lines.append(f"- {start}: delta lifts but absolute does not in {len(broken)} demos {broken[:20]}; absolute lifts but delta does not in {len(other)} demos {other[:20]}.")
    failures = [(n, [e for e in EPISODES if not o[f"{e}_final"]]) for n, o in zip(names, outcomes) if not all(o[f"{e}_final"] for e in EPISODES)]
    lines.append(f"- Demos with any failed episode: {len(failures)}.")
    for n, failed in failures[:30]:
        lines.append(f"  - {n}: {', '.join(failed)}")
    lines.append("")

    lines += ["## Goal log", ""]
    rows = []
    for name in EPISODES:
        gaps = np.concatenate([r["goal_gaps"][name] for r in results])
        rows.append([name, f"{np.median(gaps[:, 0]):.1e}", f"{gaps[:, 0].max():.1e}", f"{np.median(gaps[:, 1]):.1e}", f"{gaps[:, 1].max():.1e}"])
    lines.append(table(["episode", "median pos gap (m)", "max pos gap (m)", "median rot gap (rad)", "max rot gap (rad)"], rows))
    interface = max(np.concatenate([r["goal_gaps"][f"absolute_{s}"] for r in results]).max() for s in STARTS)
    lines += ["", f"Interface check (absolute runs, goal equals stored action within {GOAL_TOL:g}): {'pass' if interface <= GOAL_TOL else 'FAIL'} (max {interface:.1e}).", ""]
    lines.append("In delta runs, the gap equals how far the achieved pose has drifted from the recorded pose before each step.")
    lines.append("")

    lines += ["## Errors by phase", "", "Each cell is median / 95th percentile / maximum.", ""]
    lines.append(table(["start", "pair", "phase", "steps", *ERRORS], phase_rows(names, results)))
    lines.append("")

    lines += ["## Largest errors", ""]
    for start in STARTS:
        for pair in ("recording_vs_absolute", "delta_vs_absolute"):
            worst = sorted(((np.nanmax(r["errors"][f"{start}/{pair}"][:, 0]), n) for n, r in zip(names, results)), reverse=True)[:5]
            lines.append(f"- {start}, {pair.replace('_', ' ')}, eef position: " + ", ".join(f"{n} {v * 1000:.2f} mm" for v, n in worst))
    lines.append("")

    if figures or video:
        lines += ["## Figures", ""]
        lines += [f"- `{p.name}`" for p in figures]
        if video:
            lines.append(f"- `{video.name}`: recording, delta replay, and absolute replay with the stale start.")
        lines.append("")
    path.write_text("\n".join(lines))


def main():
    """Run Experiment B and write the report, results, figures, and video.

    Args:
        None: Arguments come from the command line.

    Returns:
        None: Nothing.
    """
    root = Path("data/robomimic/datasets/lift/ph")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=root / "low_dim_v15.hdf5")
    parser.add_argument("--absolute", type=Path, default=root / "low_dim_abs.hdf5")
    parser.add_argument("--demos", nargs="+", default=["demo_0"], help='Demonstration names, or "all"')
    parser.add_argument("--plots", nargs="*", default=["demo_0"], help="Demonstrations to plot")
    parser.add_argument("--video", default="demo_0", help="Demonstration to render, or none")
    parser.add_argument("--out-dir", type=Path, default=Path("docs/continuous-replay"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.source, "r") as source_file, h5py.File(args.absolute, "r") as abs_file:
        source, output = source_file["data"], abs_file["data"]
        names = demo_keys(source) if args.demos == ["all"] else args.demos
        obs_keys = source[f"{names[0]}/obs"].keys()
        envs = {
            "delta": make_env(json.loads(output.attrs["env_args_delta"]), obs_keys),
            "absolute": make_env(json.loads(output.attrs["env_args"]), obs_keys),
        }
        require(arm_controller(envs["delta"]).input_type == "delta", "delta environment is not delta")
        require(arm_controller(envs["absolute"]).input_type == "absolute", "absolute environment is not absolute")
        table_height = float(envs["delta"].env.model.mujoco_arena.table_offset[2])

        results, outcomes, figures, video_path = [], [], [], None
        for i, name in enumerate(names):
            render_this = name == args.video
            result = run_demo(envs, source[name], output[name], render_this)
            outcomes.append(outcome(result, table_height))
            if name in args.plots:
                figures += plot_demo(args.out_dir, name, result, table_height)
            if render_this:
                video_path = args.out_dir / f"{name}_replay.mp4"
                write_video(video_path, result)
            result["images"] = None
            results.append(result)
            if i % 25 == 0:
                print(f"Finished {name}", flush=True)

    suffix = "" if args.demos == ["all"] else "_" + "_".join(names)
    write_report(args.out_dir / f"report{suffix}.md", names, results, outcomes, figures, video_path)
    np.savez(
        args.out_dir / f"results{suffix}.npz",
        errors_names=np.array(ERRORS),
        demo=np.concatenate([[name] * r["n"] for name, r in zip(names, results)]),
        t=np.concatenate([np.arange(r["n"]) for r in results]),
        phase=np.concatenate([r["phase"] for r in results]),
        **{f"errors_{key.replace('/', '_')}": np.concatenate([r["errors"][key] for r in results]) for key in results[0]["errors"]},
        **{f"goal_gap_{name}": np.concatenate([r["goal_gaps"][name] for r in results]) for name in EPISODES},
        **{f"outcome_{key}": np.array([o[key] for o in outcomes]) for key in outcomes[0]},
        demo_names=np.array(names),
    )
    print(f"Report: {args.out_dir / f'report{suffix}.md'}")


if __name__ == "__main__":
    main()
