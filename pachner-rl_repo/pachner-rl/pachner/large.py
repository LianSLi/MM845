"""100-tetrahedron stress test: scramble generator, manifold check, and stable Fitted-Q fine-tuning."""
import copy
import os
import pickle
import random
import time

import numpy as np

from pachner.data import build_mcomplex
from pachner.env import action_features, env_from_mcomplex, legal_actions

LARGE_TARGET_TETS = 100
SUCCESS_BONUS = 5.0


def grow_to_exact_tets(base_triangulation, target_tets=100, seed=0,
                       four_four_probability=0.15, max_attempts=20000):
    # Count-changing moves are exclusively 2-3; 4-4 moves are optional neutral
    # moves so the scramble is not a straight stack of 2-3s.
    rng = random.Random(seed)
    mc = build_mcomplex(base_triangulation.copy())
    if len(mc.Tetrahedra) > target_tets:
        raise ValueError(f"Base triangulation already has {len(mc.Tetrahedra)} tetrahedra.")
    history = []
    attempts = 0
    while len(mc.Tetrahedra) < target_tets:
        if attempts >= max_attempts:
            raise RuntimeError(f"Could not reach {target_tets} tetrahedra; at {len(mc.Tetrahedra)}.")
        attempts += 1
        acts = legal_actions(mc)
        grow = [a for a in acts if a[0] == "2-3"]
        neutral = [a for a in acts if a[0] == "4-4"]
        if neutral and rng.random() < four_four_probability:
            action = rng.choice(neutral)
        elif grow:
            action = rng.choice(grow)
        elif neutral:
            action = rng.choice(neutral)
        else:
            raise RuntimeError("No legal 2-3 or 4-4 move is available.")
        try:
            if action[0] == "2-3":
                ok = mc.two_to_three(action[2], action[1], must_succeed=False)
            else:
                ok = mc.four_to_four(action[1].get_arrow(), must_succeed=False)
        except Exception:
            ok = False
        if ok:
            mc.rebuild()
            history.append(action[0])
    return mc, history


def check_manifold(mc, ref_volume, ref_homology, rescue_tries=25):
    # snappy_manifold() is fine for invariants but NOT for counting tetrahedra.
    # Volume is only meaningful when SnapPy's solver converged (it sometimes
    # returns a degenerate solution for a perfectly valid triangulation), and
    # only for a hyperbolic filling. If it did not converge, re-derive a fresh
    # closed triangulation of the same manifold and re-solve, up to
    # rescue_tries times. Homology is the weaker fallback check.
    Mx = mc.snappy_manifold()
    if str(Mx.homology()) != ref_homology:
        return "MISMATCH (H1)"
    for _ in range(rescue_tries + 1):
        if Mx.solution_type() == "all tetrahedra positively oriented":
            ok = abs(float(Mx.volume()) - ref_volume) < 1e-6
            return "ok (H1 + volume)" if ok else "MISMATCH (volume)"
        Mx = build_mcomplex(Mx.filled_triangulation()).snappy_manifold()
    return "H1 only (volume solver failed)"


def start_signature(mc):
    return (len(mc.Tetrahedra), tuple(sorted(e.valence() for e in mc.Edges)))


def fast_train_batch(ag, batch_size=64, clip=(-10.0, 100.0), sync_every=150):
    # Vectorized Fitted-Q update with a frozen target network and clipped targets.
    if len(ag.replay) < 64:
        return
    if getattr(ag, "target_model", None) is None:
        ag.target_model = copy.deepcopy(ag.model)
        ag.n_updates = 0
    batch = ag.rng.sample(ag.replay, min(batch_size, len(ag.replay)))
    X = np.stack([np.concatenate([b[0], b[1]]) for b in batch])
    y = np.array([b[2] for b in batch], dtype=float)
    rows, owners = [], []
    for i, (s, a, r, s_next, nafs, done) in enumerate(batch):
        if not done and len(nafs):
            rows.append(np.hstack([np.repeat(s_next[None, :], len(nafs), axis=0), nafs]))
            owners.append((i, len(nafs)))
    if rows:
        q = ag.target_model.predict(np.vstack(rows))
        starts = np.cumsum([0] + [n for _, n in owners[:-1]])
        qmax = np.maximum.reduceat(q, starts)
        for (i, _), m in zip(owners, qmax):
            y[i] += ag.gamma * float(m)
    y = np.clip(y, clip[0], clip[1])
    ag.model.partial_fit(X, y)
    ag.fitted = True
    ag.n_updates += 1
    if ag.n_updates % sync_every == 0:
        ag.target_model = copy.deepcopy(ag.model)


def curriculum_size(ep, total, rng):
    ramp = int(0.6 * total)
    if ep < ramp:
        top = int(30 + 70 * ep / max(1, ramp))
        return rng.randint(20, max(21, top))
    return 100 if rng.random() < 0.7 else rng.randint(30, 100)


def mean_q(ag, n=500):
    rp = ag.replay[-n:]
    if not rp or not ag.fitted:
        return float("nan")
    return float(ag.model.predict(np.stack([np.concatenate([t[0], t[1]]) for t in rp])).mean())


def save_large_checkpoint(ag, episode, log, path):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(dict(model=ag.model, target_model=getattr(ag, "target_model", None),
                         n_updates=getattr(ag, "n_updates", 0), rng_state=ag.rng.getstate(),
                         replay=ag.replay[-4000:], episode=episode, log=log), f)
    os.replace(tmp, path)


def run_large_training(ag, base_T, target, ep_from, total, log, ckpt_path, time_limit=None, budgets=(180, 600)):
    t_start = time.time()
    for ep in range(ep_from, total):
        if time_limit is not None and time.time() - t_start > time_limit:
            return ep
        rng = random.Random(1_000_003 * ep + 17)
        size = curriculum_size(ep, total, rng)
        budget = rng.choice(budgets)
        eps = max(0.05, 0.5 - 0.45 * ep / max(1, int(0.6 * total)))
        mc, _ = grow_to_exact_tets(base_T, target_tets=size, seed=rng.randrange(10**9),
                                   four_four_probability=0.2)
        env = env_from_mcomplex(mc, budget)
        s, acts = env.obs()
        reached, n = False, 0
        t0 = time.time()
        for _ in range(budget):
            if not acts:
                break
            afs = [action_features(a) for a in acts]
            i = ag.select(s, acts, afs, eps)
            r, done, ok = env.step(acts[i])
            n += 1
            if env.tet_count() <= target:
                r += SUCCESS_BONUS
                done = True
                reached = True
            s2, acts2 = env.obs()
            nafs = (np.array([action_features(a) for a in acts2], dtype=np.float32)
                    if acts2 else np.zeros((0, 6), np.float32))
            ag.replay.append((s, afs[i], r, s2, nafs, done))
            if len(ag.replay) > 20000:
                ag.replay.pop(0)
            if n % 2 == 0:
                fast_train_batch(ag)
            s, acts = s2, acts2
            if done:
                break
        log.append(dict(ep=ep, size=size, budget=budget, eps=round(eps, 3), steps=n,
                        best=env.best_count, reached=bool(reached), secs=round(time.time() - t0, 1)))
        if (ep + 1) % 5 == 0:
            save_large_checkpoint(ag, ep + 1, log, ckpt_path)
        if (ep + 1) % 10 == 0:
            last = log[-10:]
            print(f"episodes {ep-8:3d}-{ep+1:3d} | sizes {min(r['size'] for r in last):3d}-{max(r['size'] for r in last):3d}"
                  f" | reached {sum(r['reached'] for r in last):2d}/10 | mean best {np.mean([r['best'] for r in last]):5.1f}"
                  f" | mean Q {mean_q(ag):5.1f}")
    return total


def print_training_summary(log, total):
    print("Training summary by phase (reached the original size during training, exploration still on):")
    for lo in range(0, total, 50):
        g = [r for r in log if lo <= r["ep"] < lo + 50]
        if not g:
            continue
        big = [r for r in g if r["size"] >= 90]
        print(f"  episodes {lo:3d}-{min(lo + 50, total):3d}: reached {sum(r['reached'] for r in g):2d}/{len(g)}"
              f" | starts of 90+ tetrahedra: {sum(r['reached'] for r in big)}/{len(big)}")
