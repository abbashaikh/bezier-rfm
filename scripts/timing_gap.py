"""Compare H30 checkpoint timing errors with cross-demo nearest neighbours.

Run from the repository root:
    uv run python scripts/timing_gap.py --run checkpoints/h30_m1 --out docs/tool_hang/h30_m1/timing
"""

import argparse
import json
from pathlib import Path

import jax
import matplotlib
import numpy as np
from flax import nnx
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation

from bezier_rfm.datasets import window_loader as wl
from bezier_rfm.flow_matching import spline
from bezier_rfm.models.transformer import SplineTransformer
from bezier_rfm.training import load_params

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def timing_errors(h, target):
    """Return signed duration differences and the H28 max timing error.

    Args:
        h (np.ndarray): Durations in seconds, shape (..., K).
        target (np.ndarray): Reference durations in seconds, broadcastable to h.
    Returns:
        dict: Signed segment and total errors and absolute max error, in ms.
    """
    segment = (h - target) * 1000
    total = segment.sum(axis=-1)
    return {"segment_ms": segment, "total_ms": total,
            "max_ms": np.maximum(np.abs(segment).max(axis=-1), np.abs(total))}


def timing_summary(errors):
    """Summarize duration errors across windows and draws.

    Args:
        errors (dict): Duration errors in ms from timing_errors.
    Returns:
        dict: Error quantiles, signed bias, and segment error medians.
    """
    total = errors["total_ms"]
    maximum = errors["max_ms"]
    return {
        "max_median_ms": float(np.median(maximum)),
        "max_p90_ms": float(np.quantile(maximum, 0.9)),
        "total_abs_median_ms": float(np.median(np.abs(total))),
        "total_abs_p90_ms": float(np.quantile(np.abs(total), 0.9)),
        "total_signed_median_ms": float(np.median(total)),
        "segment_abs_median_ms": np.median(np.abs(errors["segment_ms"]).reshape(-1, spline.K), axis=0).tolist(),
        "total_is_largest_fraction": float(np.mean(np.abs(total) == maximum)),
    }


def plot_timing(distance, match, sample, path):
    """Plot neighbour and model errors against conditioning distance.

    Args:
        distance (np.ndarray): Nearest-neighbour distances, shape (n,).
        match (dict): Target-pair timing errors in ms.
        sample (dict): Model timing errors in ms, with a draw axis.
        path (Path): Output PNG path.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), sharey=True, layout="constrained")
    draws = sample["max_ms"].shape[1]
    for ax, key, label in zip(axes, ("total_ms", "max_ms"), ("Absolute total-duration error", "Max timing error")):
        ax.scatter(distance, np.abs(match[key]), s=23, alpha=0.75, label="Nearest training target")
        ax.scatter(distance, np.median(np.abs(sample[key]), axis=1), s=20, alpha=0.75,
                   marker="x", label=f"Model (median of {draws} draws)")
        ax.set(xlabel="Conditioning distance (RMS across inputs)", ylabel=f"{label} (ms)", ylim=(0, None))
        ax.axhline(50, color="0.4", lw=1, ls="--", label="50 ms threshold")
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_yscale("symlog", linthresh=200)
    axes[0].set_yticks([0, 50, 100, 200, 400, 800, 1200], labels=[0, 50, 100, 200, 400, 800, 1200])
    axes[1].tick_params(labelleft=True)
    axes[1].legend(fontsize=8)
    fig.suptitle("H30 milestone 1 · Dropout 0.3 · 100k EMA · 64 fixed validation windows")
    fig.supxlabel("Timing axis is linear up to 200 ms and logarithmic above it.", fontsize=9)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    """Run the cross-demo timing comparison on the final milestone weights."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cfg = json.loads((args.run / "config.json").read_text())
    stats, val_demos = wl.load_stats(args.run / "stats.json")
    w = wl.load_windows(cfg["windows"])
    train, _ = wl.split(w, val_demos)
    fixed = np.array(cfg["fixed_val"])
    np.testing.assert_array_equal(val_demos, cfg["val_demos"])
    np.testing.assert_array_equal(np.isin(w["demo"][fixed], val_demos), True)
    cond = spline.make_cond(w, stats)

    step = cfg["steps"]
    checkpoint = args.run / f"ema_{step:07d}.pkl"
    model = SplineTransformer(nnx.Rngs(params=0, dropout=0), dropout=cfg["dropout"])
    load_params(model, checkpoint)
    model.eval()
    draws = cfg["fixed_draws"]
    probe = wl.take(cond, np.repeat(fixed, draws))
    # Match the sampling key in train.offline_metrics.
    _, key = jax.random.split(jax.random.PRNGKey(100))
    prior = spline.sample_prior(key, probe.q0_pos, probe.q0_rot, stats)
    run_sampler = jax.jit(spline.sample, static_argnums=4)
    samples = run_sampler(model, prior, probe, stats, cfg["nfe"])
    sampled_h = np.asarray(spline.durations(samples)).reshape(len(fixed), draws, spline.K)
    sampled = timing_errors(sampled_h, w["h"][fixed, None, :])
    model_summary = timing_summary(sampled)
    with (args.run / "metrics.jsonl").open() as f:
        log = next(row for row in map(json.loads, f) if row["trainer/step"] == step)
    logged = [log[f"val/sample/time_ms_{q}"] for q in ("median", "p90")]
    np.testing.assert_allclose([model_summary["max_median_ms"], model_summary["max_p90_ms"]], logged, atol=1, rtol=0)

    # Match on the normalized observations and start token, with equal weight per coordinate.
    features = np.column_stack((np.asarray(cond.obs).reshape(len(w["demo"]), -1), np.asarray(cond.start)))
    distances = cdist(features[fixed], features[train], metric="sqeuclidean")
    nearest = distances.argmin(axis=1)
    matched = train[nearest]
    distance = np.sqrt(distances[np.arange(len(fixed)), nearest] / features.shape[1])
    neighbour = timing_errors(w["h"][matched], w["h"][fixed])
    demos = w["demo"][fixed]
    start_pos_mm = 1000 * np.linalg.norm(w["q_pos"][fixed, 0] - w["q_pos"][matched, 0], axis=1)
    fixed_rot = Rotation.from_matrix(w["q_rot"][fixed, 0])
    matched_rot = Rotation.from_matrix(w["q_rot"][matched, 0])
    start_rot_deg = np.rad2deg((fixed_rot.inv() * matched_rot).magnitude())
    start_speed_mm_s = 1000 * np.linalg.norm(w["v0"][:, :3], axis=1)
    start_speed_gap_mm_s = np.abs(start_speed_mm_s[fixed] - start_speed_mm_s[matched])

    groups = {"all": np.arange(len(fixed))}
    groups.update({f"distance_quartile_{i}": rows for i, rows in enumerate(np.array_split(distance.argsort(), 4), 1)})
    groups.update({f"demo_{demo}": np.flatnonzero(demos == demo) for demo in val_demos})
    summary = {}
    for name, rows in groups.items():
        summary[name] = {
            "windows": len(rows),
            "windows_by_demo": {str(demo): int(np.sum(demos[rows] == demo)) for demo in val_demos},
            "distance_range": [float(distance[rows].min()), float(distance[rows].max())],
            "start_position_median_mm": float(np.median(start_pos_mm[rows])),
            "start_rotation_median_deg": float(np.median(start_rot_deg[rows])),
            "start_speed_gap_median_mm_s": float(np.median(start_speed_gap_mm_s[rows])),
            "neighbour": timing_summary({key: value[rows] for key, value in neighbour.items()}),
            "model": timing_summary({key: value[rows] for key, value in sampled.items()}),
        }

    results = {
        "checkpoint": str(checkpoint), "dropout": cfg["dropout"], "sampler_steps": cfg["nfe"], "draws": draws,
        "logged_max_timing_ms": dict(zip(("median", "p90"), logged)),
        "summary": summary,
        "pairs": {"validation_row": fixed.tolist(), "validation_demo": demos.tolist(),
                  "training_row": matched.tolist(), "training_demo": w["demo"][matched].tolist(),
                  "distance": distance.tolist(), "start_position_mm": start_pos_mm.tolist(),
                  "start_rotation_deg": start_rot_deg.tolist(), "start_speed_gap_mm_s": start_speed_gap_mm_s.tolist(),
                  "target_h_s": w["h"][fixed].tolist(),
                  "neighbour_h_s": w["h"][matched].tolist(), "sampled_h_s": sampled_h.tolist()},
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "timing.json").write_text(json.dumps(results, indent=2) + "\n")
    plot_timing(distance, neighbour, sampled, args.out / "timing.png")


if __name__ == "__main__":
    main()
