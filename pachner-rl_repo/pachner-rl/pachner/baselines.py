"""Reference methods (greedy variants, biased random walk) and the RL rollout, all acting on a PachnerEnv.

greedy "flat"     : 3-2 move if one exists, otherwise a uniformly random move from the flat action list
                    (this list is dominated by 2-3 moves).
greedy "balanced" : same, but the fallback first picks a move *type* uniformly (biased_choice, bias 0).
greedy "plateau"  : same, but the fallback prefers 4-4 (neutral) moves over 2-3 (growing) moves.
"""
import random

from pachner.agent import biased_choice
from pachner.env import PachnerEnv, action_features


def greedy_run(env, rng, fallback="flat", patience=6):
    """Greedy descent from `env`; stops after more than `patience` consecutive non-reducing steps."""
    trace = [env.tet_count()]
    no_improve = 0
    for _ in range(env.max_steps):
        acts = env.current_actions()
        if not acts:
            break
        reducing = [a for a in acts if a[0] == "3-2"]
        if reducing:
            a = rng.choice(reducing)
        elif fallback == "flat":
            a = rng.choice(acts)
        elif fallback == "balanced":
            a = acts[biased_choice(acts, rng, shrink_bias=0.0)]
        elif fallback == "plateau":
            a = rng.choice([x for x in acts if x[0] == "4-4"] or acts)
        else:
            raise ValueError(fallback)
        no_improve = 0 if reducing else no_improve + 1
        env.step(a)
        trace.append(env.tet_count())
        if no_improve > patience:
            break
    return trace


def random_walk_run(env, seed=None, patience=40, shrink_bias=0.3):
    """Biased random walk; stops after `patience` steps without a new best count."""
    rng = random.Random(seed)
    trace = [env.tet_count()]
    best_seen = env.tet_count()
    stall = 0
    for _ in range(env.max_steps):
        acts = env.current_actions()
        if not acts:
            break
        idx = biased_choice(acts, rng, shrink_bias=shrink_bias)
        _, done, ok = env.step(acts[idx])
        trace.append(env.tet_count())
        if ok and env.tet_count() < best_seen:
            best_seen = env.tet_count()
            stall = 0
        else:
            stall += 1
        if patience and stall > patience:
            break
        if done:
            break
    return trace


def rl_run(env, agent, epsilon=0.05, seed=0):
    """Roll out the learned Q-function (epsilon-greedy with a seeded RNG)."""
    agent.rng = random.Random(seed)
    trace = [env.tet_count()]
    s, acts = env.obs()
    for _ in range(env.max_steps):
        if not acts:
            break
        afs = [action_features(a) for a in acts]
        idx = agent.select(s, acts, afs, epsilon)
        _, done, ok = env.step(acts[idx])
        trace.append(env.tet_count())
        s, acts = env.obs()
        if done:
            break
    return trace


def greedy_simplify(triangulation, max_steps=60, rng=None):
    rng = rng or random.Random()
    env = PachnerEnv(triangulation, max_steps=max_steps)
    return env, greedy_run(env, rng, "flat", patience=6)


def random_walk_simplify(triangulation, max_steps=60, seed=None, patience=None, shrink_bias=0.0):
    env = PachnerEnv(triangulation, max_steps=max_steps)
    return env, random_walk_run(env, seed, patience, shrink_bias)


def snappy_builtin_best_of(triangulation, n_restarts=10):
    best = triangulation.num_tetrahedra()
    best_T = triangulation
    for _ in range(n_restarts):
        T2 = triangulation.copy()
        T2.randomize(); T2.simplify()
        if T2.num_tetrahedra() < best:
            best = T2.num_tetrahedra(); best_T = T2
    return best_T, best
