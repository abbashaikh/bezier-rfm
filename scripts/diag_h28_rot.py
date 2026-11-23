"""H28.2: diagnose the rotation-error floor of an H28 checkpoint (no training).

Run from the repo root:
    uv run python scripts/diag_h28_rot.py --run checkpoints/h28_1_sf_d0_20k --out docs/tool_hang/h28_2_diag
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

sys.path.insert(0, str(Path(__file__).parent))
from train_h28 import KEYS, N_U, build_controls, select_windows, spline_points, val_demos  # noqa: E402

from bezier_rfm.fitting import bezier  # noqa: E402
from bezier_rfm.flow_matching import spline  # noqa: E402
from bezier_rfm.manifolds import pose  # noqa: E402
from bezier_rfm.models.transformer import SplineTransformer  # noqa: E402
from bezier_rfm.training import load_params  # noqa: E402

POSES = ("q1", "q2", "q3", "H1", "H2", "H3")


def table(header, rows):
    """Format a markdown table.

    Args:
        header (list): Column names.
        rows (list): Rows of cells.

    Returns:
        list: Markdown lines.
    """
    fmt = lambda c: f"{c:.4g}" if isinstance(c, float) else str(c)
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + \
        ["| " + " | ".join(fmt(c) for c in r) + " |" for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default="data/robomimic/datasets/tool_hang/ph/windows_K3_c2.90_g4_all.hdf5")
    ap.add_argument("--run", default="checkpoints/h28_1_sf_d0_20k")
    ap.add_argument("--out", default="docs/tool_hang/h28_2_diag")
    ap.add_argument("--nfe", type=int, default=100)
    args = ap.parse_args()
    run, out = Path(args.run), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((run / "config.json").read_text())
    sf = cfg.get("rot_frame", "world") == "start"
    sr = cfg.get("rot_input", "matrix") == "scaled"

    with h5py.File(args.windows) as f:
        w = {k: f["windows"][k][()] for k in KEYS}
    train_mask = ~np.isin(w["demo"], val_demos(len(np.unique(w["demo"]))))
    stats = spline.fit_stats({k: v[train_mask] for k, v in w.items()})
    sel = select_windows(w, train_mask)
    assert sel.tolist() == cfg["windows_idx"]
    ws = {k: v[sel] for k, v in w.items()}
    n_win = len(sel)
    x1_win, cond_win = spline.target_state(ws), spline.make_cond(ws, stats)

    model = SplineTransformer(nnx.Rngs(params=0, dropout=1), dropout=cfg["dropout"])
    load_params(model, run / "ema.pkl")
    model.eval()
    lines = [f"# H28.2 rotation diagnostics, run `{run}` (rot frame {'start' if sf else 'world'}, rot input {'scaled' if sr else 'matrix'})", "",
             f"rot_scale {float(stats.rot_scale):.4f} rad ({np.degrees(float(stats.rot_scale)):.2f} deg), "
             f"pos_scale {float(stats.pos_scale) * 1e3:.1f} mm", ""]

    # ---- (a) loss per pose and flow-time bin, (d) target angles; same draws as train_h28
    n_bin, per = 10, 256
    ib = jnp.repeat(jnp.arange(n_win), per)
    tb = (jnp.tile(jnp.arange(per), n_win) + jax.random.uniform(jax.random.PRNGKey(7), (n_win * per,))) / per
    xb1, cb = jax.tree.map(lambda a: a[ib], x1_win), jax.tree.map(lambda a: a[ib], cond_win)
    xb0 = spline.sample_prior(jax.random.PRNGKey(8), cb.q0_pos, cb.q0_rot, stats)
    xt, u = spline.geodesic(xb0, xb1, tb)
    pred = jax.jit(lambda m: spline.velocity(m, xt, tb, cb, stats, sf, sr))(model)
    ev = np.asarray(((pred.v - u.v) / stats.pos_scale) ** 2).mean(-1)  # (n, 6)
    ew = np.asarray(((pred.w - u.w) / stats.rot_scale) ** 2).mean(-1)
    bins = np.minimum((np.asarray(tb) * n_bin).astype(int), n_bin - 1)
    lines += ["## (a) Normalized squared velocity error per pose (EMA, 2048 draws)", ""]
    lines += table(["term", *POSES, "all"],
                   [["pos", *ev.mean(0).tolist(), float(ev.mean())], ["rot", *ew.mean(0).tolist(), float(ew.mean())]])
    lines += ["", "Rotation term per flow-time bin and pose:", ""]
    lines += table(["t bin", *POSES, "all", "pos all"],
                   [[f"{b / n_bin:.1f}-{(b + 1) / n_bin:.1f}", *ew[bins == b].mean(0).tolist(),
                     float(ew[bins == b].mean()), float(ev[bins == b].mean())] for b in range(n_bin)])
    # Bias against spread: error of the mean prediction per (window, t bin) vs the total.
    ang = np.linalg.norm(np.asarray(u.w), axis=-1)
    lines += ["", f"(d) Target rotation angle |omega| (rad): mean {ang.mean():.3f}, p99 {np.quantile(ang, 0.99):.3f}, "
              f"max {ang.max():.3f} (cut locus at pi = 3.142).", ""]

    # ---- (b) spread of the segment-token inputs over the geodesic states
    seg, _ = spline.features(xt, cb, stats, sf, sr)
    seg = np.asarray(seg)  # (n, K, 29): q dp 0:3, q rot 3:12, H dp 12:15, H rot 15:24, ell 24, g 25:29
    sd = seg.std(0).mean(0)
    rel_rot = np.asarray(jnp.swapaxes(cb.q0_rot, -1, -2)[:, None] @ xt.rot if sf else xt.rot)
    scaled = (rel_rot - np.eye(3)) / float(stats.rot_scale)
    lines += ["## (b) Input spread: std over the 2048 geodesic states, mean over the K tokens", ""]
    lines += table(["input", "entries", "std per entry"], [
        ["q position (p - p0)/pos_scale", "3", " ".join(f"{s:.3f}" for s in sd[0:3])],
        ["q rotation R entries", "9", " ".join(f"{s:.3f}" for s in sd[3:12])],
        ["H position", "3", " ".join(f"{s:.3f}" for s in sd[12:15])],
        ["H rotation R entries", "9", " ".join(f"{s:.3f}" for s in sd[15:24])],
        ["ell / ell_scale", "1", f"{sd[24]:.3f}"],
        ["gripper", "4", " ".join(f"{s:.3f}" for s in sd[25:29])],
        ["(R - I)/rot_scale, all 6 poses", "9",
         " ".join(f"{s:.3f}" for s in scaled.reshape(-1, 9).std(0))],
    ])
    lines += [""]

    # ---- (c) sampled errors: per generated pose (bias vs spread) and on the spline
    ns = 10
    isamp = jnp.repeat(jnp.arange(n_win), ns)
    cs = jax.tree.map(lambda a: a[isamp], cond_win)
    x0s = spline.sample_prior(jax.random.PRNGKey(100), cs.q0_pos, cs.q0_rot, stats)
    xs = jax.jit(lambda m: spline.sample(m, x0s, cs, stats, args.nfe, sf, sr))(model)
    x1s = jax.tree.map(lambda a: a[isamp], x1_win)
    # Error vector in the target body frame: Log(R_target^T R_sample).
    e_rot = np.degrees(np.asarray(pose.rot_log(jnp.swapaxes(x1s.rot, -1, -2) @ xs.rot))).reshape(n_win, ns, 6, 3)
    e_pos = (np.asarray(xs.pos - x1s.pos) * 1e3).reshape(n_win, ns, 6, 3)
    rms = lambda e: np.sqrt(np.mean(np.sum(e**2, -1), (0, 1)))
    # Bias = norm of the per-window mean error; spread = RMS deviation of samples from that mean.
    bias = lambda e: np.sqrt(np.mean(np.sum(e.mean(1) ** 2, -1), 0))
    spread = lambda e: np.sqrt(np.mean(np.sum((e - e.mean(1, keepdims=True)) ** 2, -1), (0, 1)))
    lines += [f"## (c) Sampled generated poses ({args.nfe} steps, 10 samples x 8 windows)", ""]
    lines += table(["error", *POSES], [
        ["rot RMS (deg)", *rms(e_rot).tolist()], ["rot bias (deg)", *bias(e_rot).tolist()],
        ["rot spread (deg)", *spread(e_rot).tolist()],
        ["pos RMS (mm)", *rms(e_pos).tolist()], ["pos bias (mm)", *bias(e_pos).tolist()],
        ["pos spread (mm)", *spread(e_pos).tolist()]])
    lines += ["", "Rotation RMS per pose and window (deg):", ""]
    lines += table(["window", *POSES], [[i, *np.sqrt(np.mean(np.sum(e_rot[i] ** 2, -1), 0)).tolist()]
                                        for i in range(n_win)])

    jax.config.update("jax_enable_x64", True)
    with jax.default_device(jax.devices("cpu")[0]):
        f64 = lambda tree: jax.tree.map(lambda a: jnp.asarray(np.asarray(a, np.float64)), tree)
        x64, c64 = f64(xs), f64(cs._replace(obs=cs.q0_pos, start=cs.q0_pos))
        pts = jax.jit(jax.vmap(lambda *a: spline_points(*build_controls(*a))))
        _, r, _, _ = pts(x64.pos, x64.rot, x64.z, x64.ell, c64.q0_pos, c64.q0_rot, c64.v0)
        tgt_c = bezier.Controls(*[jnp.asarray(ws[k]) for k in ("q_pos", "q_rot", "H_pos", "H_rot")])
        tr = jax.jit(jax.vmap(spline_points))(tgt_c, jnp.asarray(ws["h"]))[1][np.asarray(isamp)]
        a = np.degrees(np.linalg.norm(np.asarray(pose.rot_log(jnp.swapaxes(r, -1, -2) @ tr)), axis=-1))
    # a: (n, K, N_U) angle error on the spline at matching fractions.
    seg_rms = np.sqrt(np.mean(a**2, (0, 2)))
    u_rms = np.sqrt(np.mean(a**2, (0, 1)))
    lines += ["", "Spline rotation error (deg), RMS over samples:", ""]
    lines += table(["segment", "0", "1", "2", "all"], [["RMS", *seg_rms.tolist(), float(np.sqrt(np.mean(a**2)))]])
    lines += ["", "By fraction u within the segment (all segments):", ""]
    pick = [0, 5, 10, 15, 20]
    lines += table(["u", *[f"{k / (N_U - 1):.2f}" for k in pick]], [["RMS", *u_rms[pick].tolist()]])

    # ---- (e) handle offset q_i^-1 H_i: sampled vs target, and the duration ratio rho of the reflected control
    off = lambda x: np.asarray(pose.log(x.pos[:, :3], x.rot[:, :3], x.pos[:, 3:], x.rot[:, 3:])[1])
    offp = lambda x: np.asarray(x.pos[:, 3:] - x.pos[:, :3])
    d_rot = np.degrees(off(xs) - off(x1s))  # (n, K, 3)
    d_pos = (offp(xs) - offp(x1s)) * 1e3
    e_q = e_rot.reshape(-1, 6, 3)
    # Correlation of the q and H error vectors (1 = errors move together, so the offset error cancels).
    corr = np.sum(e_q[:, :3] * e_q[:, 3:]) / np.sqrt(np.sum(e_q[:, :3] ** 2) * np.sum(e_q[:, 3:] ** 2))
    ht = ws["h"]
    rho = ht / np.concatenate([ht[:, 1:], ht[:, -1:]], 1)
    lines += ["", "## (e) Handle offset Log(q_i^T H_i): sampled minus target", ""]
    lines += table(["", "i=1", "i=2", "i=3"], [
        ["target offset angle RMS (deg)", *np.degrees(np.sqrt(np.mean(np.sum(off(x1s) ** 2, -1), 0))).tolist()],
        ["offset rot error RMS (deg)", *np.sqrt(np.mean(np.sum(d_rot**2, -1), 0)).tolist()],
        ["target offset RMS (mm)", *(np.sqrt(np.mean(np.sum(offp(x1s) ** 2, -1), 0)) * 1e3).tolist()],
        ["offset pos error RMS (mm)", *np.sqrt(np.mean(np.sum(d_pos**2, -1), 0)).tolist()],
        ["rho = h_k/h_k+1 (min-max over windows)", *[f"{rho[:, k].min():.2f}-{rho[:, k].max():.2f}" for k in range(3)]],
    ])
    lines += ["", f"Correlation of q_i and H_i rotation error vectors: {corr:.3f}"]
    (out / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
