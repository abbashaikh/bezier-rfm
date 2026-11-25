"""H30 milestone 1 follow-up: measure the error between target windows of nearly the same state (data floor).

For each window, take the window of each other grid of the same demo with the nearest start time, and compare
the two targets with the H28 sample metric. A model that samples the true conditional distribution has about the
same error against a target as two such independent targets have against each other.

Run from the repo root:
    uv run python scripts/data_floor.py --out docs/tool_hang/h30_m1/floor
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bezier_rfm.datasets import window_loader as wl
from bezier_rfm.fitting import bezier
from bezier_rfm.manifolds import pose

N_U = 21  # pose check points per segment
THRESH = {"pos_mm": 5.0, "rot_deg": 3.0, "time_ms": 50.0}
DT_EDGES_MS = [0, 10, 25, 50, 100, 200, 1000]
MODEL_M1 = {"pos_mm": 15.6, "rot_deg": 1.70, "time_ms": 99.0}  # milestone 1, val windows at 100k


def spline_points(c, h):
    """Evaluate one spline at N_U matching fractions of each segment.

    Args:
        c (bezier.Controls): Controls.
        h (jnp.ndarray): Durations, shape (K,).

    Returns:
        tuple: Positions (K, N_U, 3) and rotations (K, N_U, 3, 3).
    """
    pos, rot = bezier.segment_controls(c, h)
    u = jnp.linspace(0.0, 1.0, N_U)
    return jax.vmap(lambda a, b: jax.vmap(lambda s: bezier.segment_point(a, b, s))(u))(pos, rot)


def pairs(w):
    """Pair each window with the nearest-start window of each other grid of the same demo.

    Args:
        w (dict): Window arrays.

    Returns:
        tuple: Window indices a and b, each shape (n,), and the start-time gap in ms, shape (n,).
    """
    a, b = [], []
    for d in np.unique(w["demo"]):
        rows = np.flatnonzero(w["demo"] == d)
        for g in np.unique(w["grid"][rows]):
            src = rows[w["grid"][rows] == g]
            for g2 in np.unique(w["grid"][rows]):
                if g2 == g:
                    continue
                dst = rows[w["grid"][rows] == g2]
                ts = w["t_start"][dst]
                j = np.abs(w["t_start"][src][:, None] - ts[None]).argmin(1)
                a.append(src)
                b.append(dst[j])
    a, b = np.concatenate(a), np.concatenate(b)
    return a, b, np.abs(w["t_start"][a] - w["t_start"][b]) * 1e3


@jax.jit
def target_errors(ca, ha, ga, cb, hb, gb):
    """Compute the H28 pose, timing and gripper errors of targets a against targets b.

    Args:
        ca (bezier.Controls): Controls of a, leading axis n.
        ha (jnp.ndarray): Durations of a, shape (n, K).
        ga (jnp.ndarray): Gripper signs of a, shape (n, 12).
        cb (bezier.Controls): Controls of b, leading axis n.
        hb (jnp.ndarray): Durations of b, shape (n, K).
        gb (jnp.ndarray): Gripper signs of b, shape (n, 12).

    Returns:
        dict: pos_mm, rot_deg, time_ms, grip_ok, each shape (n,).
    """
    p, r = jax.vmap(spline_points)(ca, ha)
    tp, tr = jax.vmap(spline_points)(cb, hb)
    pos_mm = jnp.sqrt(jnp.mean(jnp.sum((p - tp) ** 2, -1), (1, 2))) * 1e3
    ang = jnp.linalg.norm(pose.rot_log(jnp.swapaxes(r, -1, -2) @ tr), axis=-1)
    rot_deg = jnp.degrees(jnp.sqrt(jnp.mean(ang**2, (1, 2))))
    t_err = jnp.concatenate([ha - hb, ha.sum(1, keepdims=True) - hb.sum(1, keepdims=True)], 1)
    return {"pos_mm": pos_mm, "rot_deg": rot_deg, "time_ms": jnp.abs(t_err).max(1) * 1e3,
            "grip_ok": jnp.all(ga == gb, 1)}


def errors(w, a, b, chunk=8192):
    """Evaluate target_errors over all pairs in chunks.

    Args:
        w (dict): Window arrays.
        a (np.ndarray): First window indices, shape (n,).
        b (np.ndarray): Second window indices, shape (n,).
        chunk (int): Pairs per call.

    Returns:
        dict: pos_mm, rot_deg, time_ms, grip_ok, each shape (n,).
    """
    f = lambda k, i: jnp.asarray(w[k][i], jnp.float32)
    ctl = lambda i: bezier.Controls(*[f(k, i) for k in ("q_pos", "q_rot", "H_pos", "H_rot")])
    out = []
    for s in range(0, len(a), chunk):
        ia, ib = a[s:s + chunk], b[s:s + chunk]
        e = target_errors(ctl(ia), f("h", ia), f("gripper", ia), ctl(ib), f("h", ib), f("gripper", ib))
        out.append(jax.device_get(e))
    return {k: np.concatenate([o[k] for o in out]) for k in out[0]}


def summary(e, dt, mask):
    """Summarize errors per start-gap bin.

    Args:
        e (dict): Errors per pair.
        dt (np.ndarray): Start-time gaps in ms, shape (n,).
        mask (np.ndarray): Pairs to include, shape (n,).

    Returns:
        list: One dict per bin with counts, medians, p90 and rates.
    """
    rows = []
    for lo, hi in zip(DT_EDGES_MS[:-1], DT_EDGES_MS[1:]):
        m = mask & (dt >= lo) & (dt < hi)
        if not m.any():
            continue
        ok = (e["pos_mm"] <= THRESH["pos_mm"]) & (e["rot_deg"] <= THRESH["rot_deg"]) & \
             (e["time_ms"] <= THRESH["time_ms"]) & e["grip_ok"]
        r = {"dt_ms": f"{lo}-{hi}", "n": int(m.sum())}
        for k in ("pos_mm", "rot_deg", "time_ms"):
            r[f"{k}_median"] = float(np.median(e[k][m]))
            r[f"{k}_p90"] = float(np.quantile(e[k][m], 0.9))
        r["grip_acc"] = float(e["grip_ok"][m].mean())
        r["pass"] = float(ok[m].mean())
        rows.append(r)
    return rows


def plot(e, dt, val_mask, path):
    """Plot median pair errors against the start-time gap, with the milestone 1 model errors.

    Args:
        e (dict): Errors per pair.
        dt (np.ndarray): Start-time gaps in ms, shape (n,).
        val_mask (np.ndarray): Pairs of validation demos, shape (n,).
        path (Path): Output PNG.
    """
    edges = np.array(DT_EDGES_MS[:-1] + [300])
    mid = 0.5 * (edges[:-1] + edges[1:])
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, k, lab in zip(axs, ("pos_mm", "rot_deg", "time_ms"), ("position RMS (mm)", "rotation RMS (deg)",
                                                                    "max timing error (ms)")):
        for m, name, c in ((np.ones_like(val_mask), "all demos", "tab:blue"), (val_mask, "val demos", "tab:orange")):
            med = [np.median(e[k][m & (dt >= lo) & (dt < hi)]) if (m & (dt >= lo) & (dt < hi)).any() else np.nan
                   for lo, hi in zip(edges[:-1], edges[1:])]
            ax.plot(mid, med, "o-", color=c, label=f"target pairs, {name}")
        ax.axhline(MODEL_M1[k], color="k", ls="--", lw=1, label="model, milestone 1 (val)")
        ax.axhline(THRESH[k], color="tab:red", ls=":", lw=1, label="pass threshold")
        ax.set_xlabel("start-time gap of the pair (ms)")
        ax.set_ylabel(f"median {lab}")
        ax.set_ylim(bottom=0)
    axs[0].legend(fontsize=7)
    fig.suptitle("Target vs target error for windows of nearly the same state (other grid, same demo)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tool_hang.json")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    w = wl.load_windows(cfg["windows"])
    val = wl.val_demos(int(w["demo"].max()) + 1, cfg["val_ratio"], cfg["split_seed"])
    a, b, dt = pairs(w)
    e = errors(w, a, b)
    val_mask = np.isin(w["demo"][a], val)
    res = {"val_demos": val.tolist(), "all": summary(e, dt, np.ones_like(val_mask)), "val": summary(e, dt, val_mask)}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "floor.json").write_text(json.dumps(res, indent=1))
    plot(e, dt, val_mask, args.out / "floor.png")
    for name in ("all", "val"):
        print(f"== {name} demos")
        for r in res[name]:
            print(f"  dt {r['dt_ms']:>9} ms n {r['n']:6d}  pos {r['pos_mm_median']:5.1f} (p90 {r['pos_mm_p90']:5.1f}) mm"
                  f"  rot {r['rot_deg_median']:4.2f} deg  time {r['time_ms_median']:5.0f} (p90 {r['time_ms_p90']:4.0f}) ms"
                  f"  grip {r['grip_acc']:.2f}  pass {r['pass']:.3f}")


if __name__ == "__main__":
    main()
