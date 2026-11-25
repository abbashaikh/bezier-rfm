"""Test whether start angular speed improves the linear-speed timing baseline.

Run from the repository root:
    uv run python scripts/angular_speed_timing.py --run checkpoints/h30_m1 --out docs/tool_hang/h30_m1/angular_speed
"""

import argparse
import json
from pathlib import Path

import numpy as np

from start_speed_timing import fit_curve, load_rows, plt, speed_bins


def predict(linear, angular, target, train):
    """Predict held-out duration with and without an angular-speed correction.

    Args:
        linear (np.ndarray): Start linear speeds in mm/s, shape (n,).
        angular (np.ndarray): Start angular speeds in degrees/s, shape (n,).
        target (np.ndarray): Recorded durations in ms, shape (n,).
        train (np.ndarray): Boolean training mask, shape (n,).
    Returns:
        tuple: Linear and corrected duration predictions for held-out rows, in ms.
    """
    centers, _, median, _ = fit_curve(linear[train], target[train])
    baseline = np.interp(linear, centers, median)
    residual = target[train] - baseline[train]
    corrections = []
    # Fit the angular effect separately at each linear-speed decile.
    for mask in speed_bins(linear[train], 10):
        local_residual = residual[mask]
        angular_centers, _, correction, _ = fit_curve(angular[train][mask], local_residual, bins=5)
        # Remove the offset that the linear-speed bin alone could explain.
        correction -= np.median(local_residual)
        corrections.append(np.interp(angular[~train], angular_centers, correction))
    adjusted = np.array([np.interp(speed, centers, offsets)
                         for speed, offsets in zip(linear[~train], np.column_stack(corrections))])
    return baseline[~train], baseline[~train] + adjusted


def plot_changes(scores, path):
    """Plot the change in held-out timing error after adding angular speed.

    Args:
        scores (dict): Error quantiles by input source, timing target, and demo.
        path (Path): Output PNG path.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True, layout="constrained")
    for ax, (source, targets) in zip(axes, scores.items()):
        groups = list(targets["h0"])
        x = np.arange(len(groups))
        width = 0.19
        for i, (target, quantile, label) in enumerate((
            ("h0", "median_ms", "First segment median"), ("h0", "p90_ms", "First segment p90"),
            ("T", "median_ms", "Total duration median"), ("T", "p90_ms", "Total duration p90"),
        )):
            changes = [targets[target][g]["linear_and_angular"][quantile] - targets[target][g]["linear"][quantile]
                       for g in groups]
            ax.bar(x + (i - 1.5) * width, changes, width, label=label)
        ax.axhline(0, color="0.2", linewidth=0.8)
        ax.set_title(f"{source.capitalize()} start speeds")
        ax.set_xticks(x, ["All" if g == "validation" else g.replace("demo_", "Demo ") for g in groups])
        ax.grid(axis="y", alpha=0.18)
    axes[0].set_ylabel("Change in absolute error (ms)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=2, fontsize=9)
    fig.suptitle("Tool Hang · Effect of adding angular speed · Negative values mean lower error")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    """Compare timing predictions on the same rows as the linear-speed experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads((args.run / "config.json").read_text())
    val_demos = json.loads((args.run / "stats.json").read_text())["val_demos"]
    np.testing.assert_array_equal(cfg["val_demos"], val_demos)
    demo, speeds, targets, counts = load_rows(cfg["windows"], with_angular=True)
    val = np.isin(demo, val_demos)
    held_out = demo[val]
    groups = {"validation": np.ones(len(held_out), dtype=bool),
              **{f"demo_{d}": held_out == d for d in val_demos}}
    scores = {source: {} for source in speeds}
    for source, speed in speeds.items():
        for name, target in targets.items():
            linear, adjusted = predict(speed["linear"], speed["angular"], target, ~val)
            scores[source][name] = {}
            for group, mask in groups.items():
                result = {"windows": int(mask.sum())}
                for model, prediction in (("linear", linear), ("linear_and_angular", adjusted)):
                    error = np.abs(prediction[mask] - target[val][mask])
                    median, p90 = np.quantile(error, [0.5, 0.9])
                    result[model] = {"median_ms": float(median), "p90_ms": float(p90)}
                scores[source][name][group] = result

    args.out.mkdir(parents=True, exist_ok=True)
    results = {"windows_file": cfg["windows"], "val_demos": val_demos, "counts": counts, "results": scores}
    (args.out / "angular_speed.json").write_text(json.dumps(results, indent=2) + "\n")
    plot_changes(scores, args.out / "angular_speed.png")


if __name__ == "__main__":
    main()
