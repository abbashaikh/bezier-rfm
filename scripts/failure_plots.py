"""H33: plot where Tool Hang episodes stop, from the npz of scripts/failure_analysis.py.

Run from the repo root:
    uv run python scripts/failure_plots.py docs/tool_hang/h33/h30_1900000.npz
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm

STAGES = ("no frame grasp", "frame grasped", "frame assembled", "tool grasped", "tool hung")


def load(path):
    """Read the episodes of one run.

    Args:
        path (Path): npz file of scripts/failure_analysis.py.

    Returns:
        tuple: Meta dict and a list of episode dicts.
    """
    d = np.load(path)
    eps = []
    for i in range(len(d["start"])):
        e = {k: d[f"{i}/{k}"] for k in ("actions", "achieved_pos", "phases", "call_steps")}
        e.update(start=str(d["start"][i]), kind=str(d["kind"][i]), success=bool(d["success"][i]))
        e["gap"] = np.linalg.norm(e["actions"][:, :3] - e["achieved_pos"], axis=1) * 1e3
        # Furthest milestone reached so far at each step: k + 1 once milestones 0..k have all been seen.
        seen = np.logical_or.accumulate(e["phases"], axis=0)
        e["stage"] = np.cumprod(seen, axis=1).sum(axis=1)
        e["final"] = int(e["stage"][-1])
        e["frame_lost"] = bool(e["phases"][:, 1].any() and not e["phases"][-1, 1])
        eps.append(e)
    return json.loads(str(d["meta"])), eps


def funnel(eps, out):
    """Plot the share of episodes that reach each milestone, for test and training starts.

    Args:
        eps (list): Episode dicts from load.
        out (Path): Output PNG.

    Returns:
        None: Nothing.
    """
    fig, ax = plt.subplots(figsize=(7, 3.5))
    x = np.arange(1, len(STAGES))
    for j, kind in enumerate(("test", "train")):
        sel = [e for e in eps if e["kind"] == kind]
        share = [np.mean([e["final"] >= k for e in sel]) for k in x]
        bars = ax.bar(x + (j - 0.5) * 0.38, share, 0.38, label=f"{kind} (n = {len(sel)})")
        ax.bar_label(bars, [f"{s:.2f}" for s in share], fontsize=8)
    ax.set_xticks(x, STAGES[1:])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("share of episodes that reach it")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def raster(eps, out, max_steps):
    """Plot the stage and the tracking gap over time, one row per failed episode, grouped by final stage.

    Args:
        eps (list): Episode dicts from load.
        out (Path): Output PNG.
        max_steps (int): Step limit of the rollouts.

    Returns:
        None: Nothing.
    """
    fails = sorted([e for e in eps if not e["success"]], key=lambda e: (e["final"], e["start"]))
    stage = np.full((len(fails), max_steps), np.nan)
    gap = np.full((len(fails), max_steps), np.nan)
    for i, e in enumerate(fails):
        stage[i, :len(e["stage"])] = e["stage"]
        gap[i, :len(e["gap"])] = e["gap"]
    fig, axs = plt.subplots(1, 2, figsize=(12, 0.12 * len(fails) + 2), sharey=True)
    cmap = ListedColormap(plt.cm.viridis(np.linspace(0, 1, len(STAGES))))
    im = axs[0].imshow(stage, aspect="auto", interpolation="nearest", cmap=cmap,
                       norm=BoundaryNorm(np.arange(len(STAGES) + 1) - 0.5, len(STAGES)))
    cb = fig.colorbar(im, ax=axs[0], ticks=range(len(STAGES)))
    cb.ax.set_yticklabels(STAGES, fontsize=7)
    im = axs[1].imshow(gap, aspect="auto", interpolation="nearest", cmap="magma", norm=LogNorm(1, 300))
    fig.colorbar(im, ax=axs[1], label="commanded - achieved position (mm)")
    for i, e in enumerate(fails):
        if e["frame_lost"]:
            axs[0].plot(len(e["stage"]) - 1, i, "rx", ms=3)
    axs[0].set_yticks(range(len(fails)), [f"{e['kind'][:2]} {e['start']}" for e in fails], fontsize=5)
    axs[0].set_title("furthest milestone so far (x: frame no longer assembled at the end)")
    axs[1].set_title("tracking gap")
    for ax in axs:
        ax.set_xlabel("step")
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", type=Path, help="episodes from scripts/failure_analysis.py")
    args = ap.parse_args()
    meta, eps = load(args.npz)
    out = args.npz.with_suffix("")
    out.mkdir(parents=True, exist_ok=True)
    funnel(eps, out / "funnel.png")
    raster(eps, out / "failures.png", meta["max_steps"])
    fails = [e for e in eps if not e["success"]]
    print(f"{len(eps)} episodes, {len(fails)} failures")
    for k in range(len(STAGES) - 1):
        sel = [e for e in fails if e["final"] == k]
        if not sel:
            continue
        enter = [int(np.argmax(e["stage"] >= k)) for e in sel]
        print(f"  stop at '{STAGES[k]}': {len(sel)} ({sum(e['kind'] == 'train' for e in sel)} train), "
              f"reached at step median {np.median(enter):.0f}, frame lost {sum(e['frame_lost'] for e in sel)}, "
              f"max gap > 100 mm {sum(e['gap'].max() > 100 for e in sel)}")
    print(f"saved {out}/funnel.png, failures.png")


if __name__ == "__main__":
    main()
