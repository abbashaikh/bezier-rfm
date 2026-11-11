"""Write baseline tracking and replay metrics for the demonstrations in a traces file."""

import argparse
from pathlib import Path

import h5py
import numpy as np

from bezier_rfm.datasets.robomimic import demo_keys
from bezier_rfm.eval.lift_diagnostics import (
    ACHIEVED_METRICS,
    PHASES,
    achieved_errors,
    command_fidelity,
    phase_labels,
    read_trace,
    summarize,
    tracking_errors,
)
from bezier_rfm.eval.lift_plots import plot_diagnostics
from inspect_lift_demo import table

# Report scale and unit per metric name.
UNITS = {"m": (1000.0, "mm"), "rad": (np.degrees(1.0), "deg")}


def scaled(name):
    """Return the display scale and unit of a metric.

    Args:
        name (str): Metric name ending in "_m" or "_rad".

    Returns:
        tuple: Scale (float) and unit (str).
    """
    return UNITS[name.rsplit("_", 1)[1]]


def summary_rows(label, values, names, phases, mask=None):
    """Build table rows with count, RMS, median, 95th percentile, and maximum per phase.

    Args:
        label (str): Row label.
        values (np.ndarray): Errors per action, shape (N, M).
        names (list): Metric name per column, length M.
        phases (np.ndarray): Phase label per action, shape (N,).
        mask (np.ndarray | None): Actions to include, shape (N,).

    Returns:
        list: Table rows.
    """
    rows = []
    for j, name in enumerate(names):
        scale, unit = scaled(name)
        stats = summarize(values[:, j], phases, mask)
        for phase in [*PHASES, "all"]:
            if phase in stats:
                s = stats[phase]
                rows.append([label, f"{name.rsplit('_', 1)[0]} ({unit})", phase, s["count"], *(f"{s[k] * scale:.3g}" for k in ("rms", "median", "p95", "max"))])
    return rows


def demo_section(name, recorded, replay, start):
    """Build the report lines of one demonstration.

    Args:
        name (str): Demonstration name.
        recorded (dict): Recorded trace.
        replay (dict): Replay trace.
        start (str): Replay start.

    Returns:
        list: Markdown lines.
    """
    n = len(recorded["cmd_pos"])
    phases = phase_labels(recorded, n)
    fidelity = command_fidelity(replay)
    # The recording was driven by the stale goal at t = 0, so its tracking error there is not meaningful.
    not_first = np.arange(n) > 0
    headers = ["trace", "metric", "phase", "steps", "RMS", "median", "p95", "max"]
    track = ["position_m", "orientation_rad"]
    last = np.flatnonzero(recorded["valid"])[-1]
    lines = [
        f"## {name}",
        "",
        f"N = {n} actions, replay start: {start}. Phases from the recording: "
        + ", ".join(f"{p} {int((phases == p).sum())}" for p in PHASES)
        + ".",
        "",
        f"- Lift: recorded final state {bool(recorded['success'][last])}, replay final state {bool(replay['success'][-1])}.",
        f"- Cube height above the table top (lift rule 4 cm): state {last}, recorded {recorded['cube_height'][last] * 100:.2f} cm, replay {replay['cube_height'][last] * 100:.2f} cm; replay after the final action {replay['cube_height'][-1] * 100:.2f} cm.",
        f"- Command fidelity (replay goal vs command): max {fidelity[:, 0].max():.1e} m, {fidelity[:, 1].max():.1e} rad.",
        "",
        "### Tracking error, command t vs achieved state t + 1",
        "",
        "The recording row leaves out t = 0, and its final action has no achieved state.",
        "",
        table(headers, summary_rows("recorded", tracking_errors(recorded), track, phases, not_first) + summary_rows("replay", tracking_errors(replay), track, phases)),
        "",
        "### Replay vs recording, achieved state t + 1",
        "",
        table(headers, summary_rows("replay vs recorded", achieved_errors(replay, recorded), ACHIEVED_METRICS, phases)),
        "",
    ]
    return lines


def summary_section(per_demo):
    """Build the report lines that pool all demonstrations.

    Args:
        per_demo (list): One dict of per-step metrics and outcomes per demonstration.

    Returns:
        list: Markdown lines.
    """
    phases = np.concatenate([d["phase"] for d in per_demo])
    headers = ["trace", "metric", "phase", "steps", "RMS", "median", "p95", "max"]
    track = ["position_m", "orientation_rad"]
    pool = lambda key: np.concatenate([d[key] for d in per_demo])
    fidelity = pool("fidelity")
    failed = [d["name"] for d in per_demo if d["lift_recorded"] and not d["lift_replay"]]
    worst = sorted(((np.nanmax(d["achieved"][:, 0]), d["name"]) for d in per_demo), reverse=True)[:5]
    return [
        f"## All {len(per_demo)} demonstrations",
        "",
        f"- Lift: recorded {sum(d['lift_recorded'] for d in per_demo)}, replay {sum(d['lift_replay'] for d in per_demo)}. "
        f"Recorded lift but replay did not: {failed if failed else 'none'}.",
        f"- Command fidelity: max {fidelity[:, 0].max():.1e} m, {fidelity[:, 1].max():.1e} rad.",
        "- Largest replay vs recorded eef position: " + ", ".join(f"{n} {v * 1000:.2f} mm" for v, n in worst) + ".",
        "",
        "### Tracking error, command t vs achieved state t + 1",
        "",
        "The recording rows leave out t = 0, and its final action has no achieved state.",
        "",
        table(headers, summary_rows("recorded", pool("track_recorded"), track, phases) + summary_rows("replay", pool("track_replay"), track, phases)),
        "",
        "### Replay vs recording, achieved state t + 1",
        "",
        table(headers, summary_rows("replay vs recorded", pool("achieved"), ACHIEVED_METRICS, phases)),
        "",
    ]


def main():
    """Write the baseline metrics report.

    Args:
        None: Arguments come from the command line.

    Returns:
        None: Nothing.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, default=Path("docs/diagnostics/baseline_traces.hdf5"))
    parser.add_argument("--figures", nargs="*", default=["demo_0"], help="Demonstrations to plot")
    parser.add_argument("--out-dir", type=Path, default=Path("docs/diagnostics"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Baseline diagnostics",
        "",
        f"Traces: `{args.traces}`. Position errors are in mm and orientation errors in degrees. "
        "Orientation errors are geodesic angles |Log(R_a^T R_b)|.",
        "",
    ]
    per_demo, sections = [], []
    with h5py.File(args.traces, "r") as traces_file:
        for name in demo_keys({k: None for k in traces_file if k != "meta"}):
            recorded, _ = read_trace(traces_file[f"{name}/recorded"])
            replay, attrs = read_trace(traces_file[f"{name}/replay_absolute"])
            n = len(recorded["cmd_pos"])
            track_rec = tracking_errors(recorded)
            # The recording was driven by the stale goal at t = 0, so its tracking error there is not meaningful.
            track_rec[0] = np.nan
            last = np.flatnonzero(recorded["valid"])[-1]
            per_demo.append({
                "name": name,
                "phase": phase_labels(recorded, n),
                "track_recorded": track_rec,
                "track_replay": tracking_errors(replay),
                "achieved": achieved_errors(replay, recorded),
                "fidelity": command_fidelity(replay),
                "lift_recorded": bool(recorded["success"][last]),
                "lift_replay": bool(replay["success"][-1]),
                "height_replay": float(replay["cube_height"][-1]),
            })
            if name in args.figures:
                sections += demo_section(name, recorded, replay, attrs["start"])
                paths = plot_diagnostics(args.out_dir / "figures", name, recorded, {"replay_absolute": replay})
                sections += ["Figures: " + ", ".join(f"`figures/{p.name}`" for p in paths), ""]
    lines += summary_section(per_demo) + sections
    np.savez(
        args.out_dir / "baseline_metrics.npz",
        demo=np.concatenate([[d["name"]] * len(d["phase"]) for d in per_demo]),
        t=np.concatenate([np.arange(len(d["phase"])) for d in per_demo]),
        phase=np.concatenate([d["phase"] for d in per_demo]),
        tracking_recorded=np.concatenate([d["track_recorded"] for d in per_demo]),
        tracking_replay=np.concatenate([d["track_replay"] for d in per_demo]),
        achieved_replay_vs_recorded=np.concatenate([d["achieved"] for d in per_demo]),
        achieved_metrics=np.array(ACHIEVED_METRICS),
        demo_names=np.array([d["name"] for d in per_demo]),
        lift_recorded=np.array([d["lift_recorded"] for d in per_demo]),
        lift_replay=np.array([d["lift_replay"] for d in per_demo]),
        final_height_replay=np.array([d["height_replay"] for d in per_demo]),
    )
    (args.out_dir / "baseline_report.md").write_text("\n".join(lines))
    print(f"Report: {args.out_dir / 'baseline_report.md'}")


if __name__ == "__main__":
    main()
