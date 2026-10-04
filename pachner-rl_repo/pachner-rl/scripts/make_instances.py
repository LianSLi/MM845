"""Build configs/instances.json: pinned start triangulations (isomorphism signatures) and best-known sizes.

SnapPy's filled_triangulation() is randomized, so we draw a pool of closed triangulations once per
Dehn filling and store their isosigs; all evaluations then rebuild exactly these triangulations.
"best known" = the smallest closed triangulation found by `--restarts` rounds of randomize()+simplify()
on the correctly filled manifold. It is an upper bound on the true minimum, not a proof of minimality.

usage: python scripts/make_instances.py [--restarts 200] [--pool 20]
"""
import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from pachner.data import (INSTANCES_PATH, best_known_for_instance, get_triangulation,  # noqa: E402
                          isosig_of, key_of, triangulation_from_isosig)
from pachner.train import REAL_TRAIN_INSTANCES  # noqa: E402

HELD_OUT = [("4_1", (7, 2)), ("5_2", (7, 1)), ("6_1", (7, 1)), ("6_3", (7, 1)), ("m009", (5, 1))]
LARGE_BENCH = [("4_1", (5, 1)), ("5_2", (7, 1))]   # 100-tetrahedron benchmark manifolds


def build_entry(knot, pq, restarts, pool_size):
    sigs, sizes, vol, ident = [], {}, None, None
    for _ in range(3 * pool_size):
        T, vol, ident = get_triangulation(knot, pq)
        s = isosig_of(T)
        if s not in sizes:
            sizes[s] = triangulation_from_isosig(s).num_tetrahedra()
            sigs.append(s)
        if len(sigs) >= pool_size:
            break
    best, idstr = best_known_for_instance(knot, pq, n_restarts=restarts)
    best = min([best] + list(sizes.values()))
    import snappy
    N = snappy.Manifold(knot); N.dehn_fill(pq)
    return {"knot": knot, "pq": list(pq), "identify": idstr, "volume": float(N.volume()),
            "homology": str(N.homology()), "best_known": best,
            "pool_sizes": [sizes[s] for s in sigs], "isosigs": sigs}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--restarts", type=int, default=200)
    ap.add_argument("--pool", type=int, default=20)
    ap.add_argument("--out", default=str(INSTANCES_PATH))
    a = ap.parse_args()
    out = {"restarts": a.restarts, "held_out": {}, "train": {}}
    todo = [("held_out", k, p) for k, p in HELD_OUT] + [("train", k, p) for k, p in REAL_TRAIN_INSTANCES]
    seen = {}
    for group, knot, pq in todo:
        t0 = time.time()
        key = key_of(knot, pq)
        entry = seen.get(key) or build_entry(knot, pq, a.restarts, a.pool)
        seen[key] = entry
        out[group][key] = entry
        print(f"{group:8s} {key:12s} {entry['identify']:22s} best_known={entry['best_known']:3d} "
              f"pool sizes={sorted(set(entry['pool_sizes']))}  [{time.time()-t0:.0f}s]", flush=True)
        json.dump(out, open(a.out, "w"), indent=1)
    # base triangulations for the 100-tet benchmark: smallest member of each pool
    out["large"] = {}
    for knot, pq in LARGE_BENCH:
        key = key_of(knot, pq)
        e = seen.get(key) or build_entry(knot, pq, a.restarts, a.pool)
        i = min(range(len(e["isosigs"])), key=lambda j: e["pool_sizes"][j])
        out["large"][key] = {**{k: e[k] for k in ("knot", "pq", "volume", "homology", "best_known")},
                             "isosig": e["isosigs"][i], "target": e["pool_sizes"][i]}
        print("large", key, "target", e["pool_sizes"][i], "best_known", e["best_known"])
    json.dump(out, open(a.out, "w"), indent=1)
    print("wrote", a.out)
