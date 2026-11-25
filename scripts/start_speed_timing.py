"""Measure how start linear speed predicts segment and window durations.

Run from the repository root:
    uv run python scripts/start_speed_timing.py --run checkpoints/h30_m1 --out docs/tool_hang/h30_m1/start_speed
"""

import argparse
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import spearmanr

from bezier_rfm.datasets.windows import CONTROL_DT

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def load_rows(path, with_angular=False):
    """Load grid-zero windows with two recorded observations and no braking tail.

    Args:
        path (str): H30 window HDF5 path.
        with_angular (bool): Include angular speed in degrees per second.
    Returns:
        tuple: Demo IDs, speeds by source, duration targets in ms, and row counts.
    """
    with h5py.File(path) as f:
        settings = json.loads(f.attrs["settings"])
        windows = f["windows"]
        grid_zero = windows["grid"][:] == 0
        real = grid_zero & (windows["n_tail"][:] == 0)
        frame = np.floor(windows["t_start"][:] / CONTROL_DT + 1e-9).astype(int)
        keep = real & (frame > 0)
        demo = windows["demo"][keep]
        v0 = windows["v0"][keep]
        h = windows["h"][keep]
        obs = windows["obs"][keep]
        frame = frame[keep]
        np.testing.assert_allclose(h.sum(axis=1), np.exp(windows["z"][keep]))

    speeds = {"fitted": {"linear": np.linalg.norm(v0[:, :3], axis=1) * 1000},
              "observed": {"linear": np.empty(len(demo))}}
    if with_angular:
        speeds["fitted"]["angular"] = np.rad2deg(np.linalg.norm(v0[:, 3:], axis=1))
        speeds["observed"]["angular"] = np.empty(len(demo))
    keys = settings["obs_keys"]
    with h5py.File(settings["demos_file"]) as source:
        for d in np.unique(demo):
            raw = source[f"data/demo_{d}/obs"]
            pos_key = "robot0_eef_pos"
            start = sum(raw[key].shape[1] for key in keys[:keys.index(pos_key)])
            rows = demo == d
            pos_pair = obs[rows, :, start:start + 3]
            pos = raw[pos_key][:]
            current = np.minimum(frame[rows], len(pos) - 1)
            frames = np.column_stack((current - 1, current))
            np.testing.assert_allclose(pos_pair, pos[frames], atol=1e-8)
            speeds["observed"]["linear"][rows] = np.linalg.norm(pos_pair[:, 1] - pos_pair[:, 0], axis=1) * (1000 / CONTROL_DT)
            if with_angular:
                quat_key = "robot0_eef_quat"
                start = sum(raw[key].shape[1] for key in keys[:keys.index(quat_key)])
                quat_pair = obs[rows, :, start:start + 4]
                np.testing.assert_allclose(quat_pair, raw[quat_key][:][frames], atol=1e-8)
                relative = Rotation.from_quat(quat_pair[:, 0]).inv() * Rotation.from_quat(quat_pair[:, 1])
                speeds["observed"]["angular"][rows] = np.rad2deg(relative.magnitude()) / CONTROL_DT

    counts = {"grid_zero": int(grid_zero.sum()), "excluded_tails": int((grid_zero & ~real).sum()),
              "excluded_no_prior_frame": int((real & ~keep).sum()), "included": len(demo)}
    return demo, speeds, {"h0": h[:, 0] * 1000, "T": h.sum(axis=1) * 1000}, counts


def speed_bins(speed, bins):
    """Group speeds into populated quantile bins.

    Args:
        speed (np.ndarray): Speeds, shape (n,).
        bins (int): Number of quantile bins to form.
    Returns:
        list: Boolean row masks in order of increasing speed.
    """
    edges = np.unique(np.quantile(speed, np.linspace(0, 1, bins + 1)))
    groups = np.searchsorted(edges[1:-1], speed, side="right")
    return [groups == i for i in np.unique(groups)]


def fit_curve(speed, duration, bins=10):
    """Estimate median duration and spread in speed quantile bins.

    Args:
        speed (np.ndarray): Training speeds, shape (n,).
        duration (np.ndarray): Training durations in ms, shape (n,).
        bins (int): Number of quantile bins to form.
    Returns:
        tuple: Median speed, p10, median and p90 duration for each populated bin.
    """
    masks = speed_bins(speed, bins)
    centers = np.array([np.median(speed[mask]) for mask in masks])
    p10, median, p90 = np.array([np.quantile(duration[mask], [0.1, 0.5, 0.9]) for mask in masks]).T
    return centers, p10, median, p90


def score(speed, target, predicted, baseline):
    """Score a speed curve and a constant median against recorded durations.

    Args:
        speed (np.ndarray): Speeds in mm/s, shape (n,).
        target (np.ndarray): Recorded durations in ms, shape (n,).
        predicted (np.ndarray): Predicted durations in ms, shape (n,).
        baseline (float): Training median duration in ms.
    Returns:
        dict: Sample count, rank correlation, and absolute error quantiles.
    """
    result = {"windows": len(target), "spearman": float(spearmanr(speed, target).statistic)}
    for name, prediction in (("constant", baseline), ("speed_curve", predicted)):
        median, p90 = np.quantile(np.abs(prediction - target), [0.5, 0.9])
        result[name] = {"median_ms": float(median), "p90_ms": float(p90)}
    return result


def main():
    """Fit and evaluate the linear-speed duration curves on the H30 split."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads((args.run / "config.json").read_text())
    val_demos = json.loads((args.run / "stats.json").read_text())["val_demos"]
    np.testing.assert_array_equal(cfg["val_demos"], val_demos)
    demo, speeds, targets, counts = load_rows(cfg["windows"])
    val = np.isin(demo, val_demos)
    train = ~val
    groups = {"train": train, "validation": val, **{f"demo_{d}": demo == d for d in val_demos}}
    results = {"windows_file": cfg["windows"], "val_demos": val_demos, "counts": counts, "results": {}}

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True, sharey="col", layout="constrained")
    for row, (speed_name, values) in enumerate(speeds.items()):
        speed = values["linear"]
        results["results"][speed_name] = {}
        for col, (target_name, target) in enumerate(targets.items()):
            centers, p10, median, p90 = fit_curve(speed[train], target[train])
            baseline = float(np.median(target[train]))
            predicted = np.interp(speed, centers, median)
            results["results"][speed_name][target_name] = {
                "constant_ms": baseline,
                "training_bins": {"speed_mm_s": centers.tolist(), "median_ms": median.tolist(),
                                  "p10_ms": p10.tolist(), "p90_ms": p90.tolist()},
                "scores": {name: score(speed[mask], target[mask], predicted[mask], baseline)
                           for name, mask in groups.items()},
            }
            ax = axes[row, col]
            ax.fill_between(centers, p10, p90, color="0.86", label="Training p10–p90")
            ax.plot(centers, median, "k-", lw=1.8, label="Training median curve")
            for d in val_demos:
                mask = demo == d
                ax.scatter(speed[mask], target[mask], s=10, alpha=0.50, label=f"Demo {d}")
            duration_label = "First segment" if target_name == "h0" else "Total window"
            ax.set_title(f"{duration_label} · {speed_name} speed")
            if col == 0:
                ax.set_ylabel("First segment duration (ms)")
            else:
                ax.set_ylabel("Total duration (ms)")
            ax.grid(alpha=0.15)

    for ax in axes[-1]:
        ax.set_xlabel("Start linear speed (mm/s)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=3, fontsize=9)
    fig.suptitle("Tool Hang · Start linear speed and duration")
    args.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out / "start_speed.png", dpi=150)
    plt.close(fig)
    (args.out / "start_speed.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
