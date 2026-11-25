"""Compare held-out duration errors of two training runs and the H30.2 speed curve (H31).

Run from the repository root:
    uv run python scripts/timing_compare.py --base checkpoints/h30_m1 --new checkpoints/h31_m1 --out docs/tool_hang/h31_m1/timing
"""

import argparse
import functools
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
from flax import nnx

from bezier_rfm.datasets import window_loader as wl
from bezier_rfm.flow_matching import spline
from bezier_rfm.models.transformer import SplineTransformer
from bezier_rfm.training import load_params
from start_speed_timing import fit_curve

matplotlib.use("Agg")
from matplotlib import pyplot as plt

N_DRAWS = 16
BATCH = 128  # windows per sampler call
QUANTITIES = {"h0": "h_0 error (ms)", "T": "T error (ms)", "share0": "share 0 error (pct points)",
              "share1": "share 1 error (pct points)", "share2": "share 2 error (pct points)",
              "max": "max timing error (ms)"}
STATS = {"draw_median": "single draws, median", "draw_p90": "single draws, p90",
         "mid_median": "median of 16 draws, median", "mid_p90": "median of 16 draws, p90"}
STYLE = {"curve": ("#8a8985", "s"), "base": ("#2a78d6", "o"), "new": ("#eb6834", "D")}


@functools.partial(jax.jit, static_argnums=4)
def sample_h(model, key, cond, stats, nfe):
    """Sample one window duration vector per conditioning row.

    Args:
        model (nnx.Module): EMA model in eval mode.
        key (jax.Array): PRNG key of the prior draws.
        cond (spline.Cond): Conditioning, leading axis n.
        stats (spline.Stats): Scales of the run.
        nfe (int): Sampler steps.

    Returns:
        jnp.ndarray: Segment durations in s, shape (n, K).
    """
    x0 = spline.sample_prior(key, cond.q0_pos, cond.q0_rot, stats, cond.log_t_base)
    return spline.durations(spline.sample(model, x0, cond, stats, nfe))


def run_durations(run, w, idx):
    """Sample N_DRAWS durations per window from the last checkpoint of a run.

    Args:
        run (Path): Training run directory.
        w (dict): Window arrays.
        idx (np.ndarray): Window indices.

    Returns:
        np.ndarray: Durations in s, shape (len(idx), N_DRAWS, K).
    """
    cfg = json.loads((run / "config.json").read_text())
    stats, _ = wl.load_stats(run / "stats.json")
    model = SplineTransformer(nnx.Rngs(params=0, dropout=0), dropout=cfg["dropout"])
    load_params(model, run / f"ema_{cfg['steps']:07d}.pkl")
    model.eval()
    cond = spline.make_cond({k: w[k][idx] for k in ("obs", "q_pos", "q_rot", "v0")}, stats)
    out = []
    for b in range(0, len(idx), BATCH):
        rows = np.repeat(np.arange(b, min(b + BATCH, len(idx))), N_DRAWS)
        # The key does not depend on the run, so both runs use the same prior noise.
        h = sample_h(model, jax.random.fold_in(jax.random.PRNGKey(300), b), wl.take(cond, rows), stats, cfg["nfe"])
        out.append(np.asarray(h, float).reshape(-1, N_DRAWS, spline.K))
    return np.concatenate(out)


def errors(h, target):
    """Compute absolute duration errors of single draws and of the median over draws.

    Args:
        h (np.ndarray): Sampled durations in s, shape (n, N_DRAWS, K).
        target (np.ndarray): Recorded durations in s, shape (n, K).

    Returns:
        dict: Per quantity: draw errors (n, N_DRAWS) and mid errors (n,).
    """
    def parts(a):
        total = a.sum(-1)
        share = 100 * a / total[..., None]
        return {"h0": 1e3 * a[..., 0], "T": 1e3 * total, **{f"share{k}": share[..., k] for k in range(spline.K)}}

    p, t = parts(h), parts(target)
    out = {k: (np.abs(p[k] - t[k][:, None]), np.abs(np.median(p[k], 1) - t[k])) for k in p}
    # Max timing as in H28: largest of the segment errors and the total error.
    e = np.abs(h - target[:, None]).max(-1)
    draw = 1e3 * np.maximum(e, np.abs(h.sum(-1) - target.sum(-1)[:, None]))
    mid = np.median(h, 1)
    mid = 1e3 * np.maximum(np.abs(mid - target).max(-1), np.abs(np.median(h.sum(-1), 1) - target.sum(-1)))
    out["max"] = (draw, mid)
    return out


def curve_errors(w, idx, fit):
    """Compute the errors of the H30.2 speed curves for h_0 and T, and of constant median shares.

    Args:
        w (dict): Window arrays.
        idx (np.ndarray): Scored window indices.
        fit (np.ndarray): Training window indices of the fit (grid 0, no tail, a previous frame).

    Returns:
        dict: Per quantity: errors (n,).
    """
    s, h = spline.start_speed(w["obs"]), w["h"]
    out = {}
    for name, target in (("h0", 1e3 * h[:, 0]), ("T", 1e3 * h.sum(1))):
        centers, _, median, _ = fit_curve(s[fit], target[fit])
        out[name] = np.abs(np.interp(s[idx], centers, median) - target[idx])
    share = 100 * h / h.sum(1, keepdims=True)
    for k in range(spline.K):
        out[f"share{k}"] = np.abs(share[idx, k] - np.median(share[fit, k]))
    return out


def summarize(err, mask):
    """Reduce the errors of a window group to quantiles.

    Args:
        err (dict): Per quantity: (draw errors, mid errors), or one array for the curve.
        mask (np.ndarray): Rows of the group.

    Returns:
        dict: Per quantity: median and p90 of the draw and mid errors.
    """
    out = {}
    for k, e in err.items():
        draw, mid = e if isinstance(e, tuple) else (e, e)
        out[k] = {f"{n}_{q}": float(f(a[mask])) for n, a in (("draw", draw), ("mid", mid))
                  for q, f in (("median", np.median), ("p90", lambda v: np.quantile(v, 0.9)))}
    return out


def plot(res, demos, labels, path):
    """Plot the error quantiles per validation demo and their mean over demos.

    Args:
        res (dict): Per method and group: summaries.
        demos (list): Validation demos.
        labels (dict): Legend label per method.
        path (Path): Output PNG.
    """
    groups = [f"demo_{d}" for d in demos] + ["mean_demo"]
    fig, axes = plt.subplots(len(STATS), len(QUANTITIES), figsize=(20, 12), sharex=True, layout="constrained")
    for i, (stat, stat_label) in enumerate(STATS.items()):
        for j, (q, q_label) in enumerate(QUANTITIES.items()):
            ax = axes[i, j]
            for n, (method, (color, marker)) in enumerate(STYLE.items()):
                if q in res[method]["mean_demo"]:
                    ax.plot(np.arange(len(groups)) + 0.2 * (n - 1), [res[method][g][q][stat] for g in groups], marker,
                            color=color, ms=7, label=labels[method])
            ax.axvline(len(groups) - 1.5, color="0.8", lw=1)
            ax.set_ylim(0, None)
            ax.grid(axis="y", alpha=0.2)
            ax.spines[["top", "right"]].set_visible(False)
            if i == 0:
                ax.set_title(q_label, fontsize=10)
            if j == 0:
                ax.set_ylabel(stat_label)
            ax.set_xticks(range(len(groups)), [str(d) for d in demos] + ["mean"])
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="outside lower center", ncols=3)
    fig.suptitle("Held-out duration errors by validation demo (windows without a braking tail)")
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    """Score both runs and the speed curve on the validation windows, then write timing.json and timing.png."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, required=True, help="reference run, e.g. checkpoints/h30_m1")
    ap.add_argument("--new", type=Path, required=True, help="H31 run")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    cfg = json.loads((args.base / "config.json").read_text())
    _, val = wl.load_stats(args.base / "stats.json")
    np.testing.assert_array_equal(val, wl.load_stats(args.new / "stats.json")[1])
    w = wl.load_windows(cfg["windows"])
    tr, va = wl.split(w, val)
    idx = va[w["n_tail"][va] == 0]
    fit = tr[spline.time_base_rows({k: w[k][tr] for k in ("grid", "n_tail", "t_start")})]
    err = {"curve": curve_errors(w, idx, fit)}
    for name, run in (("base", args.base), ("new", args.new)):
        err[name] = errors(run_durations(run, w, idx), w["h"][idx])
    demo = w["demo"][idx]
    res = {}
    for name, e in err.items():
        res[name] = {f"demo_{d}": summarize(e, demo == d) for d in val}
        res[name]["mean_demo"] = {q: {s: float(np.mean([res[name][f"demo_{d}"][q][s] for d in val])) for s in STATS}
                                  for q in e}
    labels = {"curve": "speed curve (shares: constant)", "base": args.base.name, "new": args.new.name}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "timing.json").write_text(json.dumps({"labels": labels, "windows": len(idx), "results": res}, indent=1) + "\n")
    plot(res, [int(d) for d in val], labels, args.out / "timing.png")
    for stat in STATS:
        for q in QUANTITIES:
            row = " ".join(f"{labels[m]} {res[m]['mean_demo'][q][stat]:.2f}" for m in STYLE if q in res[m]["mean_demo"])
            print(f"{stat:12} {q:7} {row}")


if __name__ == "__main__":
    main()
