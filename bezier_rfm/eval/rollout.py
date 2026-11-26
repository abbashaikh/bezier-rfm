"""Lockstep RoboMimic Tool Hang rollouts of a window policy through the receding executor, one env per worker process."""

import json
import multiprocessing as mp

import numpy as np
from scipy.spatial.transform import Rotation

from bezier_rfm.datasets.windows import OBS_KEYS
from bezier_rfm.policy.executor import Executor


def env_worker(conn, env_args, source):
    """Serve reset and step requests for one environment until "close".

    Args:
        conn (Connection): Pipe end to the main process.
        env_args (dict): Environment metadata with the absolute OSC controller.
        source (str): Dataset with the demo start states and model files.

    Returns:
        None: Nothing.
    """
    import h5py

    from bezier_rfm.datasets.robomimic import arm_controller, load_demo_model, make_env, read_result

    data = h5py.File(source, "r")["data"]
    env = make_env(env_args, OBS_KEYS)
    base = env.env

    def observe():
        obs = env.get_observation()
        return np.concatenate([np.asarray(obs[key]).reshape(-1) for key in OBS_KEYS])

    while True:
        cmd, arg = conn.recv()
        if cmd == "reset":
            if isinstance(arg, int):
                # Test start: robosuite draws the object placement from the global NumPy RNG, as in DP's seeded envs.
                np.random.seed(arg)
                env.reset()
            else:
                load_demo_model(env, data[arg])
            # Refreshed start, as in scripts/screen_replay.py.
            arm_controller(env).new_update = True
            pose = read_result(env)
            conn.send((observe(), pose["pos"], pose["rot"]))
        elif cmd == "step":
            env.step(arg)
            # After mj_step, site poses describe the qpos before the last integration.
            base.sim.forward()
            pose = read_result(env)
            conn.send((observe(), bool(base._check_success()), pose["pos"], pose["rot"]))
        else:
            conn.close()
            return


def pad_requests(requests, size):
    """Pad a request list to a fixed size by repeating its last entry.

    Args:
        requests (list): Window requests, at most size entries.
        size (int): Fixed batch size.

    Returns:
        list: Requests of length size.
    """
    return requests + [requests[-1]] * (size - len(requests))


def run_group(conns, jobs, policy, batch_size, hold, max_steps, stop_on_success=False, m=1):
    """Run one group of episodes in lockstep, one per environment.

    Args:
        conns (list): Pipe ends of the environment workers, at least len(jobs).
        jobs (list): Episode dicts: demo (str name, or int env seed), optional v0 (6,), optional noise (pos (M, 3), rotvec (M, 3)).
        policy (callable): Maps a padded list of requests to one window dict per request.
        batch_size (int): Fixed policy batch size.
        hold (int): Steps that repeat the last command after the policy ends the episode.
        max_steps (int): Step cap per episode, or None.
        stop_on_success (bool): End an episode at its first success.
        m (int): Segments executed per window.

    Returns:
        list: Per episode: success, first success step, steps, executed actions (S, 7) and achieved grip-site poses (S, 3), (S, 3, 3).
    """
    eps = []
    for conn, job in zip(conns, jobs):
        conn.send(("reset", job["demo"]))
    for conn, job in zip(conns, jobs):
        obs, pos, rot = conn.recv()
        eps.append({"job": job, "ex": Executor(pos, rot, job.get("v0"), m=m), "obs": [obs], "queue": [], "done": False,
                    "ended": False, "hold": 0, "actions": [], "first": None, "calls": 0, "pos": [], "rot": []})
    while not all(e["done"] for e in eps):
        # The simulator is paused here: the windows are planned before any environment steps again.
        need = [e for e in eps if not e["done"] and not e["ended"] and not e["queue"]]
        for start in range(0, len(need), batch_size):
            part = need[start:start + batch_size]
            requests = []
            for e in part:
                frames, obs_dt = e["ex"].obs_frames()
                if frames[-1] >= len(e["obs"]):
                    raise RuntimeError("the start knot is ahead of the latest observation")
                requests.append({"demo": e["job"]["demo"], "call": e["calls"], "obs": np.stack([e["obs"][i] for i in frames]),
                                 "obs_dt": obs_dt, "q0_pos": e["ex"].q_pos, "q0_rot": e["ex"].q_rot, "v0": e["ex"].v})
            windows = policy(pad_requests(requests, batch_size))[:len(part)]
            for e, w in zip(part, windows):
                idx, cmds = e["ex"].step(w, last=bool(w.get("last", False)))
                if len(idx) and idx[0] != len(e["actions"]):
                    raise RuntimeError("the window commands do not start at the next step")
                e["queue"] = list(cmds)
                e["calls"] += 1
                e["ended"] = bool(w.get("last", False))
        active = [e for e in eps if not e["done"]]
        for e in active:
            if e["queue"]:
                action = e["queue"].pop(0).copy()
                i = len(e["actions"])
                noise = e["job"].get("noise")
                if noise is not None:
                    action[:3] += noise[0][i]
                    action[3:6] = (Rotation.from_rotvec(noise[1][i]) * Rotation.from_rotvec(action[3:6])).as_rotvec()
                e["actions"].append(action)
            else:
                # The policy ended the episode: repeat the last executed command.
                action = e["actions"][-1]
                e["hold"] += 1
            e["conn_action"] = action
        for e, conn in zip(eps, conns):
            if not e["done"]:
                conn.send(("step", e["conn_action"]))
        for e, conn in zip(eps, conns):
            if e["done"]:
                continue
            obs, success, pos, rot = conn.recv()
            e["obs"].append(obs)
            e["pos"].append(pos)
            e["rot"].append(rot)
            steps = len(e["obs"]) - 1
            if success and e["first"] is None:
                e["first"] = steps
            if (e["ended"] and not e["queue"] and e["hold"] >= hold) or (max_steps is not None and steps >= max_steps) \
                    or (stop_on_success and success):
                e["done"] = True
    return [{"demo": e["job"]["demo"], "success": e["first"] is not None, "first_success_step": e["first"],
             "steps": len(e["obs"]) - 1, "calls": e["calls"], "actions": np.array(e["actions"]),
             "achieved_pos": np.array(e["pos"]), "achieved_rot": np.array(e["rot"])} for e in eps]


def rollout(env_args, source, jobs, policy, n_envs=10, batch_size=16, hold=60, max_steps=None, stop_on_success=False,
            m=1):
    """Run all episodes in groups of n_envs lockstep environments.

    Args:
        env_args (dict or str): Environment metadata (absolute OSC controller), or its JSON string.
        source (str): Dataset with the demo start states and model files.
        jobs (list): Episode dicts (see run_group).
        policy (callable): Maps a padded list of requests to one window dict per request.
        n_envs (int): Environment worker processes.
        batch_size (int): Fixed policy batch size.
        hold (int): Steps that repeat the last command after the policy ends the episode.
        max_steps (int): Step cap per episode, or None.
        stop_on_success (bool): End an episode at its first success.
        m (int): Segments executed per window.

    Returns:
        list: One result dict per job, in job order.
    """
    if isinstance(env_args, str):
        env_args = json.loads(env_args)
    # Spawn, because the main process may already run JAX, which is not fork-safe.
    ctx = mp.get_context("spawn")
    n = min(n_envs, len(jobs))
    pipes = [ctx.Pipe() for _ in range(n)]
    procs = [ctx.Process(target=env_worker, args=(child, env_args, source), daemon=True) for _, child in pipes]
    for p in procs:
        p.start()
    conns = [parent for parent, _ in pipes]
    results = []
    try:
        for start in range(0, len(jobs), n):
            group = jobs[start:start + n]
            results += run_group(conns[:len(group)], group, policy, batch_size, hold, max_steps, stop_on_success, m)
            print(f"Finished {len(results)} of {len(jobs)} episodes", flush=True)
    finally:
        for conn in conns:
            conn.send(("close", None))
        for p in procs:
            p.join()
    return results
