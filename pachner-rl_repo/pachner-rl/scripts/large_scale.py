"""100-tetrahedron stress test.

Start: a pinned minimal-ish closed triangulation of a Dehn filling (configs/instances.json -> large), grown to exactly
100 tetrahedra with random 2-3 (and some 4-4) moves. Target = the size of the original triangulation.
All methods start from identical scrambles (same seed). Final triangulations are checked to still be the same
manifold (homology + volume).

  python scripts/large_scale.py zeroshot            # agent trained on <=27 tets, applied unchanged
  python scripts/large_scale.py finetune            # warm-start fine-tuning on 20..100-tet scrambles (resumable)
  python scripts/large_scale.py compete             # RL (zero-shot / fine-tuned) vs greedy, two manifolds
"""
import argparse
import copy
import json
import pathlib
import pickle
import random
import sys
import time
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import snappy  # noqa: E402

from pachner.baselines import greedy_run, rl_run  # noqa: E402
from pachner.data import load_instances, triangulation_from_isosig  # noqa: E402
from pachner.env import env_from_mcomplex  # noqa: E402
from pachner.large import (check_manifold, grow_to_exact_tets, print_training_summary,  # noqa: E402
                           run_large_training, save_large_checkpoint, start_signature)

CKPT_VER2 = ROOT / "checkpoints" / "ver2_checkpoint.pkl"
CKPT_LARGE = ROOT / "checkpoints" / "large100_checkpoint.pkl"
N = 100


def load_agent(path=CKPT_VER2):
    return pickle.load(open(path, "rb"))["agent"]


def load_finetuned():
    ag = copy.deepcopy(load_agent())
    st = pickle.load(open(CKPT_LARGE, "rb"))
    ag.model, ag.target_model, ag.n_updates = st["model"], st["target_model"], st["n_updates"]
    ag.fitted = True
    return ag, st["episode"], st["log"]


def problem(key):
    e = load_instances()["large"][key]
    M = snappy.Manifold(e["knot"]); M.dehn_fill(tuple(e["pq"]))
    return triangulation_from_isosig(e["isosig"]), e["target"], float(M.volume()), str(M.homology())


def run_one(base_T, kind, arg, eps, seed, budget):
    mc, _ = grow_to_exact_tets(base_T, N, seed=seed)
    env = env_from_mcomplex(mc, budget)
    sig = start_signature(mc)
    if kind == "rl":
        rl_run(env, arg, epsilon=eps, seed=seed)
    else:
        greedy_run(env, random.Random(seed), "balanced" if arg else "flat")
    return env, sig


def cmd_zeroshot(a):
    agent = load_agent()
    base_T, target, vol, h1 = problem(a.manifold)
    names = {"rl": ("rl", agent, a.eps), "greedy": ("greedy", False, None), "greedy_balanced": ("greedy", True, None)}
    out = {"manifold": a.manifold, "target": target, "scrambles": a.scrambles, "eps": a.eps, "budgets": {}}
    for budget in (180, 600):
        rows = {n: [] for n in names}; secs = {n: 0.0 for n in names}; checks = {n: Counter() for n in names}
        for k in range(a.scrambles):
            sigs = []
            for n, (kind, arg, eps) in names.items():
                t0 = time.time()
                env, sig = run_one(base_T, kind, arg, eps, 12345 + k, budget)
                secs[n] += time.time() - t0
                sigs.append(sig); rows[n].append(env.best_count)
                if k < a.checked:
                    checks[n][check_manifold(env.mc, vol, h1)] += 1
            assert len(set(sigs)) == 1, "methods did not start from the same triangulation"
        out["budgets"][str(budget)] = {n: dict(best=rows[n], reached=int((np.array(rows[n]) <= target).sum()),
                                               mean_best=float(np.mean(rows[n])), median_best=float(np.median(rows[n])),
                                               secs_per_run=secs[n] / a.scrambles, checks=dict(checks[n])) for n in names}
        print(f"--- budget {budget}, {a.scrambles} scrambles of {N} tets, target {target} ---")
        for n in names:
            r = out["budgets"][str(budget)][n]
            print(f"{n:16s} reached {r['reached']:2d}/{a.scrambles}  mean best {r['mean_best']:5.1f}  median {r['median_best']:4.0f}  "
                  f"{r['secs_per_run']:.2f} s/run  checks {r['checks']}", flush=True)
    dest = pathlib.Path(a.out) if a.out else ROOT / "results" / f"large_zeroshot_{a.manifold}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(dest, "w"), indent=1)


def cmd_finetune(a):
    ag = copy.deepcopy(load_agent())
    ag.gamma, ag.replay, ag.target_model, ag.n_updates = 0.99, [], None, 0
    ep_done, log = 0, []
    if CKPT_LARGE.exists():
        st = pickle.load(open(CKPT_LARGE, "rb"))
        ag.model, ag.target_model, ag.n_updates = st["model"], st["target_model"], st["n_updates"]
        ag.fitted = True; ag.rng.setstate(st["rng_state"]); ag.replay = st["replay"]
        ep_done, log = st["episode"], st["log"]
        print(f"resuming at episode {ep_done}/{a.episodes}")
    base_T, target, _, _ = problem(a.manifold)
    if ep_done < a.episodes:
        ep_done = run_large_training(ag, base_T, target, ep_done, a.episodes, log, str(CKPT_LARGE), time_limit=a.time_limit)
        save_large_checkpoint(ag, ep_done, log, str(CKPT_LARGE))
    print(f"done {ep_done}/{a.episodes} episodes")
    print_training_summary(log, a.episodes)
    json.dump(log, open(ROOT / "results" / "large_finetune_log.json", "w"))


def cmd_compete(a):
    zero = load_agent(); tuned, ep, _ = load_finetuned()
    assert ep >= a.min_episodes, f"fine-tuning incomplete ({ep} episodes)"
    comp = {"RL zero-shot, eps 0": ("rl", zero, 0.0), "RL zero-shot, eps 0.05": ("rl", zero, 0.05),
            "RL fine-tuned, eps 0": ("rl", tuned, 0.0), "RL fine-tuned, eps 0.05": ("rl", tuned, 0.05),
            "greedy": ("greedy", False, None), "greedy balanced": ("greedy", True, None)}
    out = {"scrambles": a.scrambles, "seed0": a.seed0, "results": {}}
    for key in a.manifolds:
        base_T, target, vol, h1 = problem(key)
        res = {"target": target, "budgets": {}, "checks": {}}
        for budget in (180, 600):
            rows = {n: [] for n in comp}; checks = {n: Counter() for n in comp}
            for k in range(a.scrambles):
                for n, (kind, arg, eps) in comp.items():
                    env, _ = run_one(base_T, kind, arg, eps, a.seed0 + k, budget)
                    rows[n].append(env.best_count)
                    if k < a.checked:
                        checks[n][check_manifold(env.mc, vol, h1)] += 1
            res["budgets"][str(budget)] = rows
            res["checks"][str(budget)] = {n: dict(c) for n, c in checks.items()}
            print(f"{key} budget {budget}: target {target}", flush=True)
            for n in comp:
                b = np.array(rows[n]); print(f"   {n:24s} reached {int((b <= target).sum()):2d}/{a.scrambles}  mean best {b.mean():5.1f}", flush=True)
        out["results"][key] = res
        json.dump(out, open(ROOT / "results" / "large_compete.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    z = sub.add_parser("zeroshot"); z.add_argument("--manifold", default="4_1(5,1)"); z.add_argument("--scrambles", type=int, default=20)
    z.add_argument("--eps", type=float, default=0.05); z.add_argument("--checked", type=int, default=3)
    z.add_argument("--out", default=None, help="output json (default results/large_zeroshot_<manifold>.json)"); z.set_defaults(f=cmd_zeroshot)
    f = sub.add_parser("finetune"); f.add_argument("--manifold", default="4_1(5,1)"); f.add_argument("--episodes", type=int, default=200)
    f.add_argument("--time-limit", type=float, default=None); f.set_defaults(f=cmd_finetune)
    c = sub.add_parser("compete"); c.add_argument("--manifolds", nargs="+", default=["4_1(5,1)", "5_2(7,1)"])
    c.add_argument("--scrambles", type=int, default=15); c.add_argument("--seed0", type=int, default=777000)
    c.add_argument("--checked", type=int, default=3); c.add_argument("--min-episodes", type=int, default=200); c.set_defaults(f=cmd_compete)
    args = ap.parse_args(); args.f(args)
