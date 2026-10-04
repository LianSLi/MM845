"""Ver2 training: 900-episode mixed curriculum (synthetic scrambles of minimal manifolds + real Dehn fillings)."""
import os
import pickle
import random
import time

import snappy

from pachner.agent import PachnerQAgent, biased_choice
from pachner.data import best_known_for_instance, get_triangulation
from pachner.env import PachnerEnv, action_features, build_mcomplex, legal_actions

SEED = 0
TOTAL_EPISODES = 900
SYNTHETIC_MANIFOLDS = ["m003", "m004", "m015", "m032", "s912"]
REAL_TRAIN_INSTANCES = [
    ("4_1", (5, 1)), ("4_1", (8, 1)), ("4_1", (9, 1)),
    ("5_2", (9, 1)), ("5_2", (11, 1)),
    ("6_1", (5, 1)), ("6_1", (9, 1)),
    ("6_2", (7, 1)),
    ("6_3", (5, 1)),
    ("m006", (5, 1)), ("m006", (7, 1)),
]
SYNTH_FRAC_START, SYNTH_FRAC_END = 0.7, 0.3
SCRAMBLE_SHRINK_BIAS = -0.6
SUCCESS_BONUS = 5.0
MIN_SYN_DEPTH, MAX_SYN_DEPTH = 2, 30
MIN_REAL_EXTRA, MAX_REAL_EXTRA = 0, 10


def curriculum_frac(ep):
    return min(1.0, ep / (0.7 * TOTAL_EPISODES))


def synth_max_depth(ep):
    f = curriculum_frac(ep)
    return int(round(MIN_SYN_DEPTH + f * (MAX_SYN_DEPTH - MIN_SYN_DEPTH)))


def real_max_extra(ep):
    f = curriculum_frac(ep)
    return int(round(MIN_REAL_EXTRA + f * (MAX_REAL_EXTRA - MIN_REAL_EXTRA)))


def synth_episode_probability(ep):
    f = curriculum_frac(ep)
    return SYNTH_FRAC_START + f * (SYNTH_FRAC_END - SYNTH_FRAC_START)


def _apply_moves(mc, n_moves, rng, shrink_bias):
    for _ in range(n_moves):
        acts = legal_actions(mc)
        if not acts:
            break
        idx = biased_choice(acts, rng, shrink_bias=shrink_bias)
        kind, *rest = acts[idx]
        try:
            if kind == '2-3':
                tet, F = rest
                mc.two_to_three(F, tet, must_succeed=False)
            elif kind == '3-2':
                (edge,) = rest
                mc.three_to_two(edge.get_arrow(), must_succeed=False)
            else:
                (edge,) = rest
                mc.four_to_four(edge.get_arrow(), must_succeed=False)
        except Exception:
            pass
        mc.rebuild()


def make_synthetic_env(name, n_scramble, max_steps, rng):
    M0 = snappy.Manifold(name)
    mc = build_mcomplex(M0)
    target = len(mc.Tetrahedra)  # exact, verified true minimum
    _apply_moves(mc, n_scramble, rng, shrink_bias=SCRAMBLE_SHRINK_BIAS)
    env = PachnerEnv.__new__(PachnerEnv)
    env.mc = mc; env.start_count = len(mc.Tetrahedra); env.max_steps = max_steps
    env.steps = 0; env.best_count = env.start_count
    return env, target


def make_real_env(knot, pq, n_extra, max_steps, rng, best_known_cache):
    T, _, _ = get_triangulation(knot, pq)
    mc = build_mcomplex(T)
    target = best_known_cache[(knot, pq)]
    _apply_moves(mc, n_extra, rng, shrink_bias=SCRAMBLE_SHRINK_BIAS)
    env = PachnerEnv.__new__(PachnerEnv)
    env.mc = mc; env.start_count = len(mc.Tetrahedra); env.max_steps = max_steps
    env.steps = 0; env.best_count = env.start_count
    return env, target


def run_curriculum_episode(agent, env, target, epsilon, rng, train=True):
    trace = [env.tet_count()]
    s, actions = env.obs()
    action_feats = [action_features(a) for a in actions]
    succeeded = (env.tet_count() <= target)
    for _ in range(env.max_steps):
        if not actions or succeeded:
            break
        idx = agent.select(s, actions, action_feats, epsilon)
        chosen_feat = action_feats[idx]
        r, done, ok = env.step(actions[idx])
        new = env.tet_count()
        trace.append(new)
        if new <= target:
            r += SUCCESS_BONUS; done = True; succeeded = True
        s_next, actions_next = env.obs()
        next_feats = [action_features(a) for a in actions_next]
        if train:
            agent.remember(s, chosen_feat, r, s_next, next_feats, done)
            agent.train_batch()
        s, actions, action_feats = s_next, actions_next, next_feats
        if done:
            break
    return trace, succeeded


def save_checkpoint(state, path):
    tmp_path = path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(state, f)
    os.replace(tmp_path, path)


def load_checkpoint(path, n_restarts=60):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    print("Precomputing best-known targets for the real training pool...")
    cache = {}
    for knot, pq in REAL_TRAIN_INSTANCES:
        best, _ = best_known_for_instance(knot, pq, n_restarts=n_restarts)
        cache[(knot, pq)] = best
        print(f"  {knot}{pq}: best_known={best}")
    return {"agent": PachnerQAgent(hidden=(64, 64), gamma=0.97, seed=SEED),
            "learning_curve": [], "episode": 0, "rng_state": None,
            "py_rng_state": None, "best_known_cache": cache}


def train(state, n_episodes, rng, path, verbose_every=10, time_limit=None):
    agent = state["agent"]
    best_known_cache = state["best_known_cache"]
    if state["rng_state"] is not None:
        random.setstate(state["rng_state"])
    if state.get("py_rng_state") is not None:
        rng.setstate(state["py_rng_state"])
    start_ep = state["episode"]
    end_ep = min(TOTAL_EPISODES, start_ep + n_episodes)
    t0 = time.time()
    for ep in range(start_ep, end_ep):
        if time_limit is not None and time.time() - t0 > time_limit:
            break
        epsilon = max(0.05, 0.9 * (1 - ep / (0.7 * TOTAL_EPISODES)))
        is_synth = random.random() < synth_episode_probability(ep)
        if is_synth:
            name = random.choice(SYNTHETIC_MANIFOLDS)
            depth = random.randint(1, synth_max_depth(ep))
            max_steps = min(45, depth * 3 + 10)
            env, target = make_synthetic_env(name, depth, max_steps, rng)
            tag = f"SYN {name}"
        else:
            knot, pq = random.choice(REAL_TRAIN_INSTANCES)
            extra = random.randint(0, max(0, real_max_extra(ep)))
            max_steps = min(50, extra * 3 + 20)
            env, target = make_real_env(knot, pq, extra, max_steps, rng, best_known_cache)
            tag = f"REAL {knot}{pq}"
        trace, succeeded = run_curriculum_episode(agent, env, target, epsilon, rng, train=True)
        state["learning_curve"].append((ep, tag, env.start_count, env.tet_count(), target,
                                         int(succeeded), int(is_synth), epsilon))
        state["episode"] = ep + 1
        state["rng_state"] = random.getstate()
        state["py_rng_state"] = rng.getstate()
        save_checkpoint(state, path)
        if ep % verbose_every == 0 or ep == end_ep - 1:
            print(f"  ep {ep:4d}  {tag:14s}  start={env.start_count:3d} final={env.tet_count():3d} "
                  f"target={target:3d}  success={succeeded}  eps={epsilon:.2f}  [{time.time()-t0:.0f}s]")
    return state
