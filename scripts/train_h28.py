"""H28: overfit eight Tool Hang windows with closed-form RFM, then check recovery, continuity and plot (plan.md 9.1).

Run from the repo root:
    uv run python scripts/train_h28.py                          # train 5,000 updates, evaluate with 100 steps
    uv run python scripts/train_h28.py --skip-train --nfe 50 20 10   # same checkpoint and prior draws
"""

import argparse
import json
import time
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from flax import nnx

from bezier_rfm.fitting import bezier, so3
from bezier_rfm.flow_matching import spline
from bezier_rfm.manifolds import pose
from bezier_rfm.models.transformer import SplineTransformer
from bezier_rfm.training import load_params, save_checkpoint, train

WINDOWS = "data/robomimic/datasets/tool_hang/ph/windows_K3_c2.90_g4_all.hdf5"
KEYS = ("q_pos", "q_rot", "H_pos", "H_rot", "h", "z", "logits", "v0", "gripper", "obs", "demo", "grid", "knot",
        "n_tail")
N_U = 21  # pose check points per segment
THRESH = {"pos_mm": 5.0, "rot_deg": 3.0, "time_ms": 50.0, "jump": 1e-4}


def val_demos(n_demos, val_ratio=0.02, seed=42):
    """Pick validation demos with the Diffusion Policy rule (get_val_mask).

    Args:
        n_demos (int): Number of demos.
        val_ratio (float): Validation fraction.
        seed (int): RNG seed.

    Returns:
        np.ndarray: Validation demo indices, shape (n_val,).
    """
    n_val = min(max(1, round(n_demos * val_ratio)), n_demos - 1)
    return np.random.default_rng(seed=seed).choice(n_demos, size=n_val, replace=False)


def select_windows(w, train_mask):
    """Select two consecutive grid-0, tail-free windows from each of four training demos.

    Args:
        w (dict): All window arrays.
        train_mask (np.ndarray): Training-demo mask, shape (N,).

    Returns:
        np.ndarray: Window indices [A0, B0, A1, B1, ...], shape (8,); B follows A by one knot.
    """
    ok = train_mask & (w["grid"] == 0) & (w["n_tail"] == 0)
    idx = {(d, k): i for i, (d, k) in enumerate(zip(w["demo"], w["knot"])) if ok[i]}
    T = np.exp(w["z"])
    pairs = [(a, idx[(d, k + 1)]) for (d, k), a in idx.items() if (d, k + 1) in idx]
    pairs = np.array(pairs)
    mean_t = T[pairs].mean(1)
    switch = np.abs(np.diff(w["gripper"][pairs[:, 0], :4], axis=1)).sum(1) > 0
    q_short, q_long = np.quantile(T[ok], [0.05, 0.95])
    chosen, used = [], set()
    # Two pairs whose first window switches the gripper in its first segment, two without; one short, one long each.
    for sw in (True, False):
        for q in (q_short, q_long):
            cand = np.where((switch == sw) & ~np.isin(w["demo"][pairs[:, 0]], list(used)))[0]
            p = cand[np.argmin(np.abs(mean_t[cand] - q))]
            used.add(w["demo"][pairs[p, 0]])
            chosen.extend(pairs[p])
    return np.array(chosen)


def segment_velocity(pos, rot, h_k, u):
    """Return the linear and body angular velocity in seconds of one segment at local parameter u.

    Args:
        pos (jnp.ndarray): Control positions, shape (4, 3).
        rot (jnp.ndarray): Control rotations, shape (4, 3, 3).
        h_k (float): Segment duration, s.
        u (float): Local parameter in [0, 1].

    Returns:
        jnp.ndarray: Linear (m/s) and angular (rad/s) velocity, shape (6,).
    """
    (_, r), (dp, dr) = jax.jvp(lambda s: bezier.segment_point(pos, rot, s), (u,), (jnp.ones_like(u),))
    return jnp.concatenate([dp, so3.vee(r.T @ dr)]) / h_k


def spline_points(c, h):
    """Evaluate a spline at N_U matching fractions of each segment and its one-sided knot velocities.

    Args:
        c (bezier.Controls): Controls.
        h (jnp.ndarray): Durations, shape (K,).

    Returns:
        tuple: Positions (K, N_U, 3), rotations (K, N_U, 3, 3), left and right velocities (K, 6) at u = 1 and u = 0.
    """
    pos, rot = bezier.segment_controls(c, h)
    u = jnp.linspace(0.0, 1.0, N_U)
    p, r = jax.vmap(lambda a, b: jax.vmap(lambda s: bezier.segment_point(a, b, s))(u))(pos, rot)
    left = jax.vmap(lambda a, b, hk: segment_velocity(a, b, hk, jnp.array(1.0)))(pos, rot, h)
    right = jax.vmap(lambda a, b, hk: segment_velocity(a, b, hk, jnp.array(0.0)))(pos, rot, h)
    return p, r, left, right


def build_controls(pos, rot, z, ell, q0p, q0r, v0):
    """Build Bézier controls and durations of one sample in the dtype of its inputs.

    Args:
        pos (jnp.ndarray): Generated positions, shape (2K, 3).
        rot (jnp.ndarray): Generated rotations, shape (2K, 3, 3).
        z (jnp.ndarray): Log duration, shape ().
        ell (jnp.ndarray): Logits, shape (K,).
        q0p (jnp.ndarray): Start position, shape (3,).
        q0r (jnp.ndarray): Start rotation, shape (3, 3).
        v0 (jnp.ndarray): Start velocity, shape (6,).

    Returns:
        tuple: Controls and durations, shape (K,).
    """
    x = spline.State(pos[None], rot[None], z[None], ell[None], jnp.zeros((1, spline.N_GRIP), pos.dtype))
    cond = spline.Cond(None, None, q0p[None], q0r[None], v0[None])
    return spline.controls(x, cond, 0)


def knot1_state(x):
    """Return the pose and planned velocity at knot 1 of samples, for the handover.

    Args:
        x (spline.State): Samples.

    Returns:
        tuple: Position (n, 3), rotation (n, 3, 3), velocity (n, 6).
    """
    h = spline.durations(x)
    dv, dw = pose.log(x.pos[:, 0], x.rot[:, 0], x.pos[:, spline.K], x.rot[:, spline.K])
    return x.pos[:, 0], x.rot[:, 0], 3.0 / h[:, 1:2] * jnp.concatenate([dv, dw], -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default=WINDOWS)
    ap.add_argument("--run", default="checkpoints/h28")
    ap.add_argument("--out", default="docs/tool_hang/h28")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--block-steps", type=int, default=500)
    ap.add_argument("--draws", type=int, default=4, help="prior draws per window per batch (batch = 8 x draws)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--n-samples", type=int, default=10)
    ap.add_argument("--nfe", type=int, nargs="+", default=[100])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()
    run, out = Path(args.run), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- data, split, statistics, selection
    with h5py.File(args.windows) as f:
        w = {k: f["windows"][k][()] for k in KEYS}
    val = val_demos(len(np.unique(w["demo"])))
    train_mask = ~np.isin(w["demo"], val)
    stats = spline.fit_stats({k: v[train_mask] for k, v in w.items()})
    sel = select_windows(w, train_mask)
    ws = {k: v[sel] for k, v in w.items()}
    n_win = len(sel)
    print(f"validation demos {sorted(val.tolist())}; {train_mask.sum()} training windows")
    print("selected windows (demo, knot, T, gripper seg 0):")
    for i in range(n_win):
        print(f"  {i}: demo {ws['demo'][i]} knot {ws['knot'][i]} T {np.exp(ws['z'][i]):.3f} s  g0 {ws['gripper'][i, :4]}")
    x1_win, cond_win = spline.target_state(ws), spline.make_cond(ws, stats)

    model = SplineTransformer(nnx.Rngs(params=args.seed, dropout=args.seed + 1))
    config = {**vars(args), "val_demos": sorted(val.tolist()), "windows_idx": sel.tolist(),
              "stats": {k: np.asarray(v).tolist() for k, v in stats._asdict().items()}}

    # ---- training
    rep = jnp.repeat(jnp.arange(n_win), args.draws)
    x1_b, cond_b = jax.tree.map(lambda a: a[rep], x1_win), jax.tree.map(lambda a: a[rep], cond_win)

    def loss_fn(m, key):
        k0, kt = jax.random.split(key)
        x0 = spline.sample_prior(k0, cond_b.q0_pos, cond_b.q0_rot, stats)
        t = jax.random.uniform(kt, rep.shape)
        terms = spline.loss(m, x0, x1_b, t, cond_b, stats).mean(0)
        return terms.sum(), terms

    if not args.skip_train:
        run.mkdir(parents=True, exist_ok=True)
        log = open(run / "train_log.txt", "w")
        t_start = time.time()

        def log_fn(b, losses, aux, lr, ema):
            line = (f"block {b} step {(b + 1) * args.block_steps} loss {losses.mean():.4f} "
                    + " ".join(f"{n} {a:.4f}" for n, a in zip(spline.TERMS, aux.mean(0)))
                    + f" lr {lr:.2e} time {time.time() - t_start:.0f}s")
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()

        ema, losses, terms = train(model, loss_fn, jax.random.PRNGKey(args.seed + 2), args.steps // args.block_steps,
                                   args.block_steps, args.lr, args.warmup, log_fn=log_fn)
        save_checkpoint(run, ema, config, losses=losses, terms=terms)
        log.close()
    load_params(model, run / "ema.pkl")
    model.eval()
    d = np.load(run / "losses.npz")
    losses, terms = d["losses"], d["terms"]

    # ---- RFM loss by flow-time bin (EMA weights, fixed draws)
    n_bin, per = 10, 256
    nb = n_win * per
    ib = jnp.repeat(jnp.arange(n_win), per)
    tb = (jnp.tile(jnp.arange(per), n_win) + jax.random.uniform(jax.random.PRNGKey(7), (nb,))) / per
    xb1, cb = jax.tree.map(lambda a: a[ib], x1_win), jax.tree.map(lambda a: a[ib], cond_win)
    xb0 = spline.sample_prior(jax.random.PRNGKey(8), cb.q0_pos, cb.q0_rot, stats)
    bin_terms = np.asarray(jax.jit(lambda m: spline.loss(m, xb0, xb1, tb, cb, stats))(model))
    bins = np.minimum((np.asarray(tb) * n_bin).astype(int), n_bin - 1)
    binned = np.stack([bin_terms[bins == b].mean(0) for b in range(n_bin)])

    # ---- sampling (float32 on the default device)
    ns = args.n_samples
    isamp = jnp.repeat(jnp.arange(n_win), ns)
    cs = jax.tree.map(lambda a: a[isamp], cond_win)
    x0s = spline.sample_prior(jax.random.PRNGKey(100), cs.q0_pos, cs.q0_rot, stats)
    # Handover: window B = A + 1 knot, conditioned on A's sampled knot 1 and B's recorded observations.
    a_rows = np.concatenate([np.arange(p * ns, (p + 1) * ns) for p in range(0, n_win, 2)])
    b_win = np.repeat(np.arange(1, n_win, 2), ns)
    key_h = jax.random.PRNGKey(101)
    sample_fn = jax.jit(lambda m, x0, c, n: spline.sample(m, x0, c, stats, n), static_argnums=3)
    results = {}
    for nfe in args.nfe:
        xs = sample_fn(model, x0s, cs, nfe)
        hp, hr, hv = knot1_state(jax.tree.map(lambda a: a[a_rows], xs))
        ch = spline.make_cond_from(cond_win.obs[b_win], hp, hr, hv, stats)
        xh = sample_fn(model, spline.sample_prior(key_h, hp, hr, stats), ch, nfe)
        results[nfe] = (jax.device_get(xs), jax.device_get(ch), jax.device_get(xh))

    # ---- metrics in float64 on the CPU (continuity limit 1e-4 is near float32 noise)
    jax.config.update("jax_enable_x64", True)
    with jax.default_device(jax.devices("cpu")[0]):
        f64 = lambda tree: jax.tree.map(lambda a: None if a is None else jnp.asarray(np.asarray(a, np.float64)), tree)
        pts = jax.jit(jax.vmap(lambda *a: spline_points(*build_controls(*a))))
        tgt_c = bezier.Controls(*[jnp.asarray(ws[k]) for k in ("q_pos", "q_rot", "H_pos", "H_rot")])
        tgt = jax.jit(jax.vmap(spline_points))(tgt_c, jnp.asarray(ws["h"]))
        cond64 = f64(cond_win)
        for nfe, (xs, ch, xh) in results.items():
            report(nfe, f64(xs), f64(ch), f64(xh), cond64, ws, tgt, pts, isamp, a_rows, b_win, out / f"nfe{nfe}",
                   losses, terms, binned)


def report(nfe, xs, ch, xh, cond, ws, tgt, pts, isamp, a_rows, b_win, out, losses, terms, binned):
    """Compute the pass criteria, write the table and metrics, and make the plots for one step count.

    Args:
        nfe (int): Number of Euler steps.
        xs (spline.State): Samples on the recorded inputs, float64, (n_win * ns).
        ch (spline.Cond): Handover conditioning, float64.
        xh (spline.State): Handover samples of the B windows, float64.
        cond (spline.Cond): Recorded conditioning of the windows, float64.
        ws (dict): Selected window arrays.
        tgt (tuple): Target spline points from spline_points, batched over windows.
        pts (callable): Batched build_controls + spline_points.
        isamp (jnp.ndarray): Window of each sample, shape (n_win * ns,).
        a_rows (np.ndarray): Sample rows of the A windows used for the handover.
        b_win (np.ndarray): B window of each handover sample.
        out (Path): Output directory.
        losses (np.ndarray): Training losses per step.
        terms (np.ndarray): Training loss terms per step, shape (steps, 5).
        binned (np.ndarray): EMA loss terms per flow-time bin, shape (10, 5).
    """
    out.mkdir(parents=True, exist_ok=True)
    isamp = np.asarray(isamp)
    n_win, ns = len(ws["z"]), len(isamp) // len(ws["z"])
    cs = jax.tree.map(lambda a: a[isamp], cond)
    p, r, left, right = pts(xs.pos, xs.rot, xs.z, xs.ell, cs.q0_pos, cs.q0_rot, cs.v0)
    tp, tr = tgt[0][isamp], tgt[1][isamp]
    pos_rms = np.sqrt(np.mean(np.sum((p - tp) ** 2, -1), (1, 2))) * 1e3
    ang = np.linalg.norm(np.asarray(pose.rot_log(jnp.swapaxes(r, -1, -2) @ tr)), axis=-1)
    rot_rms = np.degrees(np.sqrt(np.mean(ang**2, (1, 2))))
    h = np.asarray(spline.durations(xs))
    ht = ws["h"][isamp]
    t_err = np.abs(np.concatenate([h - ht, h.sum(1, keepdims=True) - ht.sum(1, keepdims=True)], 1)).max(1) * 1e3
    g_ok = np.all(np.where(np.asarray(xs.g) > 0, 1.0, -1.0) == ws["gripper"][isamp], 1)
    pose_ok = (pos_rms <= THRESH["pos_mm"]) & (rot_rms <= THRESH["rot_deg"])
    time_ok = t_err <= THRESH["time_ms"]
    ok = pose_ok & time_ok & g_ok
    # Internal knots 1, 2: left of segment k-1 against right of segment k.
    jump = np.abs(np.asarray(left[:, :-1] - right[:, 1:]))
    jump_lin, jump_ang = jump[..., :3].max((1, 2)), jump[..., 3:].max((1, 2))
    # Handover: A's planned velocity at knot 1 (right side) against B's start velocity.
    ph, rh, lh, rgh = pts(xh.pos, xh.rot, xh.z, xh.ell, ch.q0_pos, ch.q0_rot, ch.v0)
    hjump = np.abs(np.asarray(right[a_rows, 1] - rgh[:, 0]))
    hpos_rms = np.sqrt(np.mean(np.sum((ph - tgt[0][b_win]) ** 2, -1), (1, 2))) * 1e3

    rows, win_pass = [], []
    for i in range(n_win):
        s = isamp == i
        hs = b_win == i
        n_ok = int(ok[s].sum())
        win_pass.append(n_ok >= 8 * ns // 10)
        rows.append({
            "window": i, "demo": int(ws["demo"][i]), "knot": int(ws["knot"][i]), "T_s": float(np.exp(ws["z"][i])),
            "pose_pass": int(pose_ok[s].sum()), "time_pass": int(time_ok[s].sum()), "grip_pass": int(g_ok[s].sum()),
            "all_pass": n_ok, "window_pass": bool(win_pass[-1]),
            "pos_rms_mm_max": float(pos_rms[s].max()), "rot_rms_deg_max": float(rot_rms[s].max()),
            "time_err_ms_max": float(t_err[s].max()),
            "knot_jump_lin_max": float(jump_lin[s].max()), "knot_jump_ang_max": float(jump_ang[s].max()),
            "handover_jump_lin_max": float(hjump[hs, :3].max()) if hs.any() else None,
            "handover_jump_ang_max": float(hjump[hs, 3:].max()) if hs.any() else None,
            "handover_pos_rms_mm_max": float(hpos_rms[hs].max()) if hs.any() else None,
        })
    cont_ok = bool(max(jump_lin.max(), jump_ang.max(), hjump.max()) < THRESH["jump"])
    summary = {"nfe": nfe, "windows_pass": int(sum(win_pass)), "n_windows": n_win, "recovery_pass": bool(all(win_pass)),
               "knot_jump_max": float(max(jump_lin.max(), jump_ang.max())), "handover_jump_max": float(hjump.max()),
               "continuity_pass": cont_ok, "thresholds": THRESH, "rows": rows,
               "final_train_loss": float(losses[-100:].mean()), "binned_loss": binned.tolist()}
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))

    lines = [f"# H28 recovery, {nfe} exp-Euler steps", "",
             f"Recovery: {sum(win_pass)}/{n_win} windows pass (>= 8 of {ns} samples pass pose, timing and gripper).",
             f"Continuity: max knot jump {summary['knot_jump_max']:.2e}, max handover jump {summary['handover_jump_max']:.2e}"
             f" (limit 1e-4 m/s and rad/s): {'pass' if cont_ok else 'FAIL'}.", "",
             "| win | demo | knot | T (s) | pose | timing | gripper | all | pass | max pos RMS (mm) | max rot RMS (deg) "
             "| max timing err (ms) | max knot jump | max handover jump |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for q in rows:
        hj = "—" if q["handover_jump_lin_max"] is None else f"{max(q['handover_jump_lin_max'], q['handover_jump_ang_max']):.1e}"
        lines.append(f"| {q['window']} | {q['demo']} | {q['knot']} | {q['T_s']:.3f} | {q['pose_pass']}/{ns} | "
                     f"{q['time_pass']}/{ns} | {q['grip_pass']}/{ns} | {q['all_pass']}/{ns} | "
                     f"{'yes' if q['window_pass'] else 'NO'} | {q['pos_rms_mm_max']:.2f} | {q['rot_rms_deg_max']:.2f} | "
                     f"{q['time_err_ms_max']:.1f} | {max(q['knot_jump_lin_max'], q['knot_jump_ang_max']):.1e} | {hj} |")
    (out / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    plot_overlay(xs, cs, ws, isamp, p, r, tgt, out / "overlay.png")
    plot_errors(pos_rms, rot_rms, t_err, isamp, n_win, out / "errors.png")
    plot_handover(xs, cs, xh, ch, a_rows, b_win, ns, out / "handover.png")
    plot_losses(losses, terms, binned, out / "losses.png")


def _times(h):
    # Times of the N_U matching fractions of each segment, shape (K, N_U).
    tau = np.r_[0.0, np.cumsum(h)]
    return tau[:-1, None] + np.linspace(0, 1, N_U)[None] * h[:, None]


def plot_overlay(xs, cs, ws, isamp, p, r, tgt, path):
    """Overlay sampled and target splines: position, rotation offset from q_0, and gripper, with segment boundaries."""
    n_win = len(ws["z"])
    h = np.asarray(spline.durations(xs))
    rel = lambda rr, r0: np.asarray(pose.rot_log(jnp.swapaxes(r0, -1, -2)[:, None, None] @ rr))
    rs = rel(r, cs.q0_rot)
    rt = rel(tgt[1], jnp.asarray(ws["q_rot"][:, 0]))
    fig, axes = plt.subplots(n_win, 3, figsize=(16, 2.4 * n_win), squeeze=False)
    cols = ["tab:red", "tab:green", "tab:blue"]
    for i in range(n_win):
        ht = ws["h"][i]
        tt = _times(ht).ravel()
        ax_p, ax_r, ax_g = axes[i]
        for j in np.where(isamp == i)[0]:
            ts = _times(h[j]).ravel()
            for a in range(3):
                ax_p.plot(ts, (np.asarray(p[j, :, :, a]).ravel() - ws["q_pos"][i, 0, a]) * 100, color=cols[a], lw=0.6, alpha=0.5)
                ax_r.plot(ts, np.degrees(rs[j, :, :, a].ravel()), color=cols[a], lw=0.6, alpha=0.5)
            ax_g.plot(np.r_[0, np.cumsum(h[j])][:-1].repeat(4) + np.tile(np.arange(4) / 4, 3) * h[j].repeat(4),
                      np.asarray(xs.g[j]), ".", color="tab:orange", ms=3, alpha=0.5)
            for ax in (ax_p, ax_r, ax_g):
                for tk in np.cumsum(h[j])[:-1]:
                    ax.axvline(tk, color="tab:orange", lw=0.4, ls="--", alpha=0.4)
        for a in range(3):
            ax_p.plot(tt, (tgt[0][i, :, :, a].ravel() - ws["q_pos"][i, 0, a]) * 100, color="k", lw=1.2, ls=":")
            ax_r.plot(tt, np.degrees(rt[i, :, :, a].ravel()), color="k", lw=1.2, ls=":")
        tg = np.r_[0, np.cumsum(ht)][:-1].repeat(4) + np.tile(np.arange(4) / 4, 3) * ht.repeat(4)
        ax_g.step(np.r_[tg, ht.sum()], np.r_[ws["gripper"][i], ws["gripper"][i, -1]], where="post", color="k", lw=1.2)
        for ax in (ax_p, ax_r, ax_g):
            for tk in np.cumsum(ht)[:-1]:
                ax.axvline(tk, color="k", lw=0.8)
        ax_p.set_ylabel(f"win {i} (demo {ws['demo'][i]})\np - p0 (cm)")
        ax_r.set_ylabel("Log(R0ᵀR) (deg)")
        ax_g.set_ylabel("gripper g")
    axes[0, 0].set_title("position (x red, y green, z blue); dotted black = target")
    axes[0, 1].set_title("rotation offset from q0; dotted black = target")
    axes[0, 2].set_title("gripper: orange = sampled g, black = target commands")
    for ax in axes[-1]:
        ax.set_xlabel("t (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_errors(pos_rms, rot_rms, t_err, isamp, n_win, path):
    """Plot per-sample pose and timing errors against the thresholds."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
    for ax, v, th, lab in zip(axes, (pos_rms, rot_rms, t_err), (THRESH["pos_mm"], THRESH["rot_deg"], THRESH["time_ms"]),
                              ("position RMS (mm)", "rotation RMS (deg)", "max timing error (ms)")):
        jit = (np.random.default_rng(0).random(len(v)) - 0.5) * 0.3
        ax.scatter(isamp + jit, v, s=10)
        ax.axhline(th, color="r", lw=1)
        ax.set_yscale("log")
        ax.set_xticks(range(n_win))
        ax.set_xlabel("window")
        ax.set_ylabel(lab)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_handover(xs, cs, xh, ch, a_rows, b_win, ns, path):
    """Plot linear and angular speed across the handover: A over its first segment, then B from A's knot 1."""
    vel = lambda c, h, t: np.asarray(jnp.concatenate(bezier.velocity(c, h, t), -1))
    n_pairs = len(a_rows) // ns
    fig, axes = plt.subplots(2, n_pairs, figsize=(4 * n_pairs, 5), squeeze=False)
    for pi in range(n_pairs):
        for s in range(min(ns, 3)):
            j, jh = a_rows[pi * ns + s], pi * ns + s
            ca, ha = build_controls(xs.pos[j], xs.rot[j], xs.z[j], xs.ell[j], cs.q0_pos[j], cs.q0_rot[j], cs.v0[j])
            cb, hb = build_controls(xh.pos[jh], xh.rot[jh], xh.z[jh], xh.ell[jh], ch.q0_pos[jh], ch.q0_rot[jh], ch.v0[jh])
            ta = jnp.linspace(0, float(ha.sum()) * (1 - 1e-6), 200)
            tb = jnp.linspace(0, float(hb.sum()) * (1 - 1e-6), 200)
            va, vb = vel(ca, ha, ta), vel(cb, hb, tb)
            for row, sl in ((0, slice(0, 3)), (1, slice(3, 6))):
                ax = axes[row, pi]
                line, = ax.plot(ta, np.linalg.norm(va[:, sl], axis=1), lw=0.8, ls="--")
                ax.plot(tb + float(ha[0]), np.linalg.norm(vb[:, sl], axis=1), lw=1.2, color=line.get_color())
                ax.axvline(float(ha[0]), color="k", lw=0.5)
        axes[0, pi].set_title(f"pair {pi}: A dashed, B solid")
        axes[0, pi].set_ylabel("|v| (m/s)")
        axes[1, pi].set_ylabel("|ω| (rad/s)")
        axes[1, pi].set_xlabel("t from A start (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_losses(losses, terms, binned, path):
    """Plot the training loss per step and the EMA RFM loss per flow-time bin."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    k = 50
    smooth = lambda a: np.convolve(a, np.ones(k) / k, mode="valid")
    axes[0].plot(smooth(losses), color="k", label="total")
    for n, a in zip(spline.TERMS, terms.T):
        axes[0].plot(smooth(a), lw=0.8, label=n)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("update")
    axes[0].set_title(f"training loss ({k}-step mean)")
    axes[0].legend()
    centers = (np.arange(len(binned)) + 0.5) / len(binned)
    axes[1].plot(centers, binned.sum(1), "k-o", label="total")
    for n, a in zip(spline.TERMS, binned.T):
        axes[1].plot(centers, a, "-o", ms=3, lw=0.8, label=n)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("flow time t")
    axes[1].set_title("EMA RFM loss by flow-time bin")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
