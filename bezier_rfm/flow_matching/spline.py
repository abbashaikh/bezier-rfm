"""Closed-form RFM on the K-segment spline state (poses on R^3 x SO(3), timing, gripper)."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from bezier_rfm.fitting import bezier
from bezier_rfm.manifolds import pose

K = 3
N_GRIP = 12
TERMS = ("pos", "rot", "z", "ell", "grip")


class State(NamedTuple):
    """Generated spline state; poses are q_1..q_K then H_1..H_K."""

    pos: jnp.ndarray  # (n, 2K, 3)
    rot: jnp.ndarray  # (n, 2K, 3, 3)
    z: jnp.ndarray  # (n,)
    ell: jnp.ndarray  # (n, K)
    g: jnp.ndarray  # (n, N_GRIP)


class Tangent(NamedTuple):
    """Flow velocity of a State: world translation, body rotation, and Euclidean parts."""

    v: jnp.ndarray  # (n, 2K, 3)
    w: jnp.ndarray  # (n, 2K, 3)
    z: jnp.ndarray  # (n,)
    ell: jnp.ndarray  # (n, K)
    g: jnp.ndarray  # (n, N_GRIP)


class Cond(NamedTuple):
    """Conditioning inputs: normalized network inputs and the raw start state."""

    obs: jnp.ndarray  # (n, 2, D_obs), min-max normalized
    start: jnp.ndarray  # (n, 18), normalized start token
    q0_pos: jnp.ndarray  # (n, 3)
    q0_rot: jnp.ndarray  # (n, 3, 3)
    v0: jnp.ndarray  # (n, 6), physical units


class Stats(NamedTuple):
    """Prior, loss and input scales fitted on the training windows."""

    knot_pos: jnp.ndarray  # (2,): q0->q1, later steps
    knot_rot: jnp.ndarray  # (2,)
    handle_pos: jnp.ndarray  # ()
    handle_rot: jnp.ndarray  # ()
    z_mean: jnp.ndarray  # ()
    z_std: jnp.ndarray  # ()
    ell_scale: jnp.ndarray  # ()
    pos_scale: jnp.ndarray  # (): pooled offset from q_0
    rot_scale: jnp.ndarray  # ()
    obs_min: jnp.ndarray  # (D_obs,)
    obs_max: jnp.ndarray  # (D_obs,)
    start_min: jnp.ndarray  # (3,)
    start_max: jnp.ndarray  # (3,)
    v0_lin: jnp.ndarray  # ()
    v0_ang: jnp.ndarray  # ()


def _angle(r_a, r_b):
    # Angle of R_a^T R_b, float64 numpy.
    c = 0.5 * (np.trace(np.swapaxes(r_a, -1, -2) @ r_b, axis1=-2, axis2=-1) - 1.0)
    return np.arccos(np.clip(c, -1.0, 1.0))


def _rms_pos(dp):
    return np.sqrt(np.mean(np.sum(dp**2, axis=-1)) / 3.0)


def _rms_rot(theta):
    return np.sqrt(np.mean(theta**2) / 3.0)


def fit_stats(w):
    """Fit all scales on training windows.

    Args:
        w (dict): Window arrays q_pos (N,K+1,3), q_rot (N,K+1,3,3), H_pos, H_rot, logits (N,K), z (N,), v0 (N,6), obs (N,2,D).

    Returns:
        Stats: Scales as float32 arrays.
    """
    qp, qr, hp, hr = w["q_pos"], w["q_rot"], w["H_pos"], w["H_rot"]
    poses_p = np.concatenate([qp[:, 1:], hp[:, 1:]], axis=1)
    poses_r = np.concatenate([qr[:, 1:], hr[:, 1:]], axis=1)
    obs = w["obs"].reshape(-1, w["obs"].shape[-1])
    s = Stats(
        knot_pos=np.array([_rms_pos(qp[:, 1] - qp[:, 0]), _rms_pos(qp[:, 2:] - qp[:, 1:-1])]),
        knot_rot=np.array([_rms_rot(_angle(qr[:, 0], qr[:, 1])), _rms_rot(_angle(qr[:, 1:-1], qr[:, 2:]))]),
        handle_pos=_rms_pos(hp[:, 1:] - qp[:, 1:]),
        handle_rot=_rms_rot(_angle(qr[:, 1:], hr[:, 1:])),
        z_mean=w["z"].mean(),
        z_std=w["z"].std(),
        ell_scale=np.sqrt(np.mean(np.sum(w["logits"] ** 2, axis=-1)) / (K - 1)),
        pos_scale=_rms_pos(poses_p - qp[:, :1]),
        rot_scale=_rms_rot(_angle(qr[:, :1], poses_r)),
        obs_min=obs.min(0),
        obs_max=obs.max(0),
        start_min=qp[:, 0].min(0),
        start_max=qp[:, 0].max(0),
        v0_lin=_rms_pos(w["v0"][:, :3]),
        v0_ang=_rms_pos(w["v0"][:, 3:]),
    )
    return Stats(*[jnp.asarray(a, jnp.float32) for a in s])


def make_cond(w, stats):
    """Build the conditioning inputs of windows.

    Args:
        w (dict): Window arrays q_pos (n,K+1,3), q_rot (n,K+1,3,3), v0 (n,6), obs (n,2,D).
        stats (Stats): Scales.

    Returns:
        Cond: Conditioning inputs, float32.
    """
    return make_cond_from(jnp.asarray(w["obs"], jnp.float32), jnp.asarray(w["q_pos"][:, 0], jnp.float32),
                          jnp.asarray(w["q_rot"][:, 0], jnp.float32), jnp.asarray(w["v0"], jnp.float32), stats)


def make_cond_from(obs, q0_pos, q0_rot, v0, stats):
    """Build the conditioning inputs from raw observations and a start state.

    Args:
        obs (jnp.ndarray): Observations, shape (n, 2, D).
        q0_pos (jnp.ndarray): Start positions, shape (n, 3).
        q0_rot (jnp.ndarray): Start rotations, shape (n, 3, 3).
        v0 (jnp.ndarray): Start velocities, linear and body angular, shape (n, 6).
        stats (Stats): Scales.

    Returns:
        Cond: Conditioning inputs.
    """
    span = jnp.maximum(stats.obs_max - stats.obs_min, 1e-6)
    obs_n = 2.0 * (obs - stats.obs_min) / span - 1.0
    p_n = 2.0 * (q0_pos - stats.start_min) / (stats.start_max - stats.start_min) - 1.0
    start = jnp.concatenate([p_n, q0_rot.reshape(-1, 9), v0[:, :3] / stats.v0_lin, v0[:, 3:] / stats.v0_ang], axis=-1)
    return Cond(obs_n, start, q0_pos, q0_rot, v0)


def target_state(w):
    """Build the generated part of fitted windows.

    Args:
        w (dict): Window arrays q_pos, q_rot, H_pos, H_rot, z, logits, gripper.

    Returns:
        State: Targets, float32.
    """
    f = lambda a: jnp.asarray(a, jnp.float32)
    return State(
        pos=f(np.concatenate([w["q_pos"][:, 1:], w["H_pos"][:, 1:]], axis=1)),
        rot=f(np.concatenate([w["q_rot"][:, 1:], w["H_rot"][:, 1:]], axis=1)),
        z=f(w["z"]),
        ell=f(w["logits"]),
        g=f(w["gripper"]),
    )


def sample_prior(key, q0_pos, q0_rot, stats):
    """Draw prior states: random-walk knots from q_0, perturbed handles, Gaussian timing and gripper.

    Args:
        key (jax.Array): PRNG key.
        q0_pos (jnp.ndarray): Start positions, shape (n, 3).
        q0_rot (jnp.ndarray): Start rotations, shape (n, 3, 3).
        stats (Stats): Scales.

    Returns:
        State: Prior samples.
    """
    n = q0_pos.shape[0]
    kq, kh, kz, kl, kg = jax.random.split(key, 5)
    eq = jax.random.normal(kq, (n, K, 6))
    eh = jax.random.normal(kh, (n, K, 6))
    # Step 0 uses the first scale pair, steps 1..K-1 share the second.
    sp = jnp.concatenate([stats.knot_pos[:1], jnp.repeat(stats.knot_pos[1:], K - 1)])
    sr = jnp.concatenate([stats.knot_rot[:1], jnp.repeat(stats.knot_rot[1:], K - 1)])
    p, r, qp, qr = q0_pos, q0_rot, [], []
    for k in range(K):
        p, r = pose.exp(p, r, sp[k] * eq[:, k, :3], sr[k] * eq[:, k, 3:])
        qp.append(p)
        qr.append(r)
    qp, qr = jnp.stack(qp, 1), jnp.stack(qr, 1)
    hp, hr = pose.exp(qp, qr, stats.handle_pos * eh[..., :3], stats.handle_rot * eh[..., 3:])
    ell = jax.random.normal(kl, (n, K))
    ell = stats.ell_scale * (ell - ell.mean(-1, keepdims=True))
    return State(
        pos=jnp.concatenate([qp, hp], 1),
        rot=jnp.concatenate([qr, hr], 1),
        z=stats.z_mean + stats.z_std * jax.random.normal(kz, (n,)),
        ell=ell,
        g=jax.random.normal(kg, (n, N_GRIP)),
    )


def geodesic(x0, x1, t):
    """Return the point x_t on the geodesic from x0 to x1 and its constant velocity log_{x0}(x1).

    Args:
        x0 (State): Prior states.
        x1 (State): Target states.
        t (jnp.ndarray): Flow times in [0, 1], shape (n,).

    Returns:
        tuple: x_t (State) and the target velocity (Tangent).
    """
    v, w = pose.log(x0.pos, x0.rot, x1.pos, x1.rot)
    u = Tangent(v, w, x1.z - x0.z, x1.ell - x0.ell, x1.g - x0.g)
    tp = t[:, None, None]
    pos, rot = pose.exp(x0.pos, x0.rot, tp * v, tp * w)
    xt = State(pos, rot, x0.z + t * u.z, x0.ell + t[:, None] * u.ell, x0.g + t[:, None] * u.g)
    return xt, u


def features(x, cond, stats, start_frame=True, scaled_rot=True):
    """Build the network inputs of the action tokens.

    Args:
        x (State): Current states.
        cond (Cond): Conditioning inputs.
        stats (Stats): Scales.
        start_frame (bool): Give rotations as offsets R_s^T R from the start rotation; else as world R.
        scaled_rot (bool): Give rotation entries as (R - I) / rot_scale; else raw R (H28, H28.1).

    Returns:
        tuple: Segment tokens, shape (n, K, 29), and z token, shape (n, 1).
    """
    n = x.z.shape[0]
    dp = (x.pos - cond.q0_pos[:, None]) / stats.pos_scale
    rot = jnp.swapaxes(cond.q0_rot, -1, -2)[:, None] @ x.rot if start_frame else x.rot
    if scaled_rot:
        # Puts small rotation differences on the same O(1) scale as dp (H28.2).
        rot = (rot - jnp.eye(3)) / stats.rot_scale
    pose_f = jnp.concatenate([dp, rot.reshape(n, 2 * K, 9)], -1)
    seg = jnp.concatenate([pose_f[:, :K], pose_f[:, K:], (x.ell / stats.ell_scale)[..., None],
                           x.g.reshape(n, K, N_GRIP // K)], -1)
    return seg, ((x.z - stats.z_mean) / stats.z_std)[:, None]


def velocity(model, x, t, cond, stats, start_frame=True, scaled_rot=True):
    """Evaluate the flow velocity of the network in physical units, with zero-mean ell velocity.

    Args:
        model (callable): Network model(seg (n,K,29), ztok (n,1), obs, start, t) -> (seg_out (n,K,17), z_out (n,1)).
        x (State): Current states.
        t (jnp.ndarray): Flow times, shape (n,).
        cond (Cond): Conditioning inputs.
        stats (Stats): Scales.
        start_frame (bool): Network sees start-frame rotations and predicts start-frame omega.
        scaled_rot (bool): Rotation input scaling, as in features.

    Returns:
        Tangent: Velocities, omega in each pose's body frame.
    """
    seg, ztok = features(x, cond, stats, start_frame, scaled_rot)
    so, zo = model(seg, ztok, cond.obs, cond.start, t)
    n = t.shape[0]
    # The network predicts each part in units of its loss scale.
    v = jnp.concatenate([so[..., 0:3], so[..., 6:9]], 1) * stats.pos_scale
    w = jnp.concatenate([so[..., 3:6], so[..., 9:12]], 1) * stats.rot_scale
    if start_frame:
        # Change of coordinates: omega_body = R_t^T R_s omega_start.
        w = (jnp.swapaxes(x.rot, -1, -2) @ cond.q0_rot[:, None] @ w[..., None])[..., 0]
    ell = so[..., 12] * stats.ell_scale
    ell = ell - ell.mean(-1, keepdims=True)
    return Tangent(v, w, zo[:, 0] * stats.z_std, ell, so[..., 13:].reshape(n, N_GRIP))


def loss_terms(pred, target, stats):
    """Compute the five scale-normalized squared errors per sample.

    Args:
        pred (Tangent): Predicted velocities.
        target (Tangent): Target velocities.
        stats (Stats): Scales.

    Returns:
        jnp.ndarray: Terms pos, rot, z, ell, grip, shape (n, 5).
    """
    def ms(a, b, s):
        d = ((a - b) / s).reshape(a.shape[0], -1)
        return jnp.mean(d**2, -1)

    return jnp.stack([
        ms(pred.v, target.v, stats.pos_scale),
        ms(pred.w, target.w, stats.rot_scale),
        ms(pred.z, target.z, stats.z_std),
        ms(pred.ell, target.ell, stats.ell_scale),
        ms(pred.g, target.g, 1.0),
    ], -1)


def loss(model, x0, x1, t, cond, stats, start_frame=True, scaled_rot=True):
    """Compute the per-sample RFM loss terms on the closed-form geodesic target.

    Args:
        model (callable): Network, as in velocity.
        x0 (State): Prior states.
        x1 (State): Target states.
        t (jnp.ndarray): Flow times, shape (n,).
        cond (Cond): Conditioning inputs.
        stats (Stats): Scales.
        start_frame (bool): Rotation frame of the network, as in velocity.
        scaled_rot (bool): Rotation input scaling, as in features.

    Returns:
        jnp.ndarray: Loss terms, shape (n, 5); the loss is their sum.
    """
    xt, u = geodesic(x0, x1, t)
    return loss_terms(velocity(model, xt, t, cond, stats, start_frame, scaled_rot), u, stats)


def sample(model, x0, cond, stats, n_steps, start_frame=True, scaled_rot=True):
    """Integrate the flow from t = 0 to 1 with exp-Euler steps and rotation projection.

    Args:
        model (callable): Network, as in velocity.
        x0 (State): Prior states.
        cond (Cond): Conditioning inputs.
        stats (Stats): Scales.
        n_steps (int): Number of Euler steps.
        start_frame (bool): Rotation frame of the network, as in velocity.
        scaled_rot (bool): Rotation input scaling, as in features.

    Returns:
        State: Samples at t = 1.
    """
    dt = 1.0 / n_steps

    def step(x, k):
        u = velocity(model, x, jnp.full(x.z.shape, k * dt), cond, stats, start_frame, scaled_rot)
        pos, rot = pose.exp(x.pos, x.rot, dt * u.v, dt * u.w)
        # Projection removes float32 drift off SO(3) (H12.1).
        return State(pos, pose.project_rot(rot), x.z + dt * u.z, x.ell + dt * u.ell, x.g + dt * u.g), None

    x1, _ = jax.lax.scan(step, x0, jnp.arange(n_steps))
    return x1


def durations(x):
    """Return the segment durations h = T softmax(ell).

    Args:
        x (State): States.

    Returns:
        jnp.ndarray: Durations in s, shape (n, K).
    """
    return jnp.exp(x.z)[:, None] * jax.nn.softmax(x.ell, -1)


def controls(x, cond, i):
    """Build the Bézier controls of one sample, with H_0 = exp_{q0}((h_0 / 3) v_0).

    Args:
        x (State): States.
        cond (Cond): Conditioning inputs.
        i (int): Sample index.

    Returns:
        tuple: Controls of sample i and its durations, shape (K,).
    """
    h = durations(x)[i]
    v0 = cond.v0[i] * h[0] / 3.0
    h0p, h0r = pose.exp(cond.q0_pos[i], cond.q0_rot[i], v0[:3], v0[3:])
    c = bezier.Controls(
        q_pos=jnp.concatenate([cond.q0_pos[i][None], x.pos[i, :K]]),
        q_rot=jnp.concatenate([cond.q0_rot[i][None], x.rot[i, :K]]),
        h_pos=jnp.concatenate([h0p[None], x.pos[i, K:]]),
        h_rot=jnp.concatenate([h0r[None], x.rot[i, K:]]),
    )
    return c, h
