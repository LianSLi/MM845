"""Paired evaluation: every method starts from an identical, seeded scrambled triangulation.

Real mode : Dehn-filled held-out manifolds (configs/instances.json -> held_out); start = pooled closed
            triangulation (isosig) + K growth-biased random Pachner moves; success = reach <= best known.
Synth mode: scrambles (depth d) of minimal census manifolds whose minimal size is known; success = reach the size.

Methods: rl (trained agent, eps=0.05), greedy (flat fallback), greedy_bal (type-balanced fallback),
greedy_plateau (4-4 fallback), random (biased random walk).
Results are appended to results/paired_<mode>.jsonl (resumable).

usage: python scripts/evaluate_paired.py real|synth [--seeds 10] [--checkpoint checkpoints/ver2_checkpoint.pkl]
"""
import argparse
import json
import os
import pathlib
import pickle
import random
import sys
import time
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import snappy  # noqa: E402

from pachner.baselines import greedy_run, random_walk_run, rl_run  # noqa: E402
from pachner.data import load_instances, triangulation_from_isosig  # noqa: E402
from pachner.env import build_mcomplex, env_from_mcomplex  # noqa: E402
from pachner.train import SCRAMBLE_SHRINK_BIAS, SYNTHETIC_MANIFOLDS, _apply_moves  # noqa: E402

REAL_K = [5, 10, 20]
SYNTH_DEPTHS = [5, 10, 20, 30]
METHODS = ["rl", "greedy", "random", "greedy_bal", "greedy_plateau"]


def signature(mc):
    return (len(mc.Tetrahedra), tuple(sorted(e.valence() for e in mc.Edges)))


def run_method(name, env, seed, agent):
    if name == "rl":
        trace = rl_run(env, agent, epsilon=0.05, seed=10_000 + seed)
    elif name == "greedy":
        trace = greedy_run(env, random.Random(20_000 + seed), "flat")
    elif name == "random":
        trace = random_walk_run(env, 30_000 + seed, patience=40, shrink_bias=0.3)
    elif name == "greedy_bal":
        trace = greedy_run(env, random.Random(40_000 + seed), "balanced")
    else:
        trace = greedy_run(env, random.Random(50_000 + seed), "plateau")
    return env.best_count, len(trace) - 1


def one_trial(base, K, seed, target, max_steps, agent):
    rec, sigs = {"target": target}, []
    for name in METHODS:
        mc = build_mcomplex(base)                     # fresh Mcomplex of the same pinned triangulation
        _apply_moves(mc, K, random.Random(seed), shrink_bias=SCRAMBLE_SHRINK_BIAS)
        sigs.append(signature(mc))
        env = env_from_mcomplex(mc, max_steps)
        env.move_counts = Counter()
        rec["start"] = env.start_count
        best, steps = run_method(name, env, seed, agent)
        rec[name], rec[name + "_steps"] = best, steps
        rec[name + "_moves"] = dict(env.move_counts)
    rec["same_start"] = all(s == sigs[0] for s in sigs)
    rec["trivial"] = rec["start"] <= target
    return rec


def jobs(mode, n_seeds, inst):
    if mode == "real":
        for seed in range(n_seeds):
            for key, e in inst["held_out"].items():
                for K in REAL_K:
                    yield dict(mode="real", instance=key, K=K, seed=seed, target=e["best_known"],
                               max_steps=40 + 3 * K, isosig=e["isosigs"][seed % len(e["isosigs"])])
    else:
        for seed in range(n_seeds):
            for name in SYNTHETIC_MANIFOLDS:
                for d in SYNTH_DEPTHS:
                    yield dict(mode="synth", instance=name, K=d, seed=seed, target=None,
                               max_steps=min(45, d * 3 + 10), isosig=None)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["real", "synth"])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "ver2_checkpoint.pkl"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--time-limit", type=float, default=None, help="stop after this many seconds (resumable)")
    a = ap.parse_args()
    out = a.out or str(ROOT / "results" / f"paired_{a.mode}.jsonl")
    agent = pickle.load(open(a.checkpoint, "rb"))["agent"]
    inst = load_instances()
    done = set()
    if os.path.exists(out):
        done = {(r["instance"], r["K"], r["seed"]) for r in map(json.loads, open(out))}
    t0, n_new = time.time(), 0
    for j in jobs(a.mode, a.seeds, inst):
        if (j["instance"], j["K"], j["seed"]) in done:
            continue
        if a.time_limit and time.time() - t0 > a.time_limit:
            break
        if j["mode"] == "real":
            base = triangulation_from_isosig(j["isosig"])
            target = j["target"]
        else:
            base = snappy.Manifold(j["instance"])
            target = len(build_mcomplex(base.copy()).Tetrahedra)   # known minimal size of the census manifold
        rec = one_trial(base, j["K"], j["seed"], target, j["max_steps"], agent)
        rec.update({k: j[k] for k in ("mode", "instance", "K", "seed", "max_steps")})
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        n_new += 1
        if n_new % 10 == 0:
            print(f"{n_new} trials  [{time.time()-t0:.0f}s]", flush=True)
    print(f"done: {n_new} new trials in {time.time()-t0:.0f}s -> {out}")
