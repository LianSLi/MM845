"""Summarize results/paired_{real,synth}.jsonl: success rates, paired differences, bootstrap CIs, sign tests.

Success = best tetrahedron count reached <= target (best-known size for real instances, known minimum for synthetic).
Trivial trials (start already <= target) are reported separately and excluded from the "non-trivial" numbers.
"""
import json
import pathlib
import sys
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pachner.stats import cluster_bootstrap, sign_test_p  # noqa: E402

METHODS = ["rl", "greedy", "greedy_bal", "greedy_plateau", "random"]
PAIRS = [("rl", "greedy"), ("rl", "greedy_bal"), ("rl", "greedy_plateau"), ("rl", "random")]


RES = ROOT / "results"


def load(mode):
    return [json.loads(l) for l in open(RES / f"paired_{mode}.jsonl")]


def succ(r, m):
    return int(r[m] <= r["target"])


def summarize(mode, rows):
    out = {"n_trials": len(rows), "n_trivial": sum(r["trivial"] for r in rows),
           "all_same_start": all(r["same_start"] for r in rows)}
    nt = [r for r in rows if not r["trivial"]]
    out["n_nontrivial"] = len(nt)
    out["success_all"] = {m: sum(succ(r, m) for r in rows) / len(rows) for m in METHODS}
    out["success_nontrivial"] = {m: cluster_bootstrap(nt, lambda r, m=m: succ(r, m), lambda r: (r["instance"], r["seed"]))
                                 for m in METHODS}
    out["below_target"] = {m: sum(r[m] < r["target"] for r in rows) for m in METHODS}
    out["paired_nontrivial"] = {}
    for a, b in PAIRS:
        d = lambda r, a=a, b=b: succ(r, a) - succ(r, b)  # noqa: E731
        mean, lo, hi = cluster_bootstrap(nt, d, lambda r: (r["instance"], r["seed"]))
        wins = sum(succ(r, a) > succ(r, b) for r in nt); losses = sum(succ(r, a) < succ(r, b) for r in nt)
        out["paired_nontrivial"][f"{a}-{b}"] = dict(diff_pts=100 * mean, lo=100 * lo, hi=100 * hi,
                                                   a_only=wins, b_only=losses, p_sign=sign_test_p(wins, losses))
    # per instance
    inst = defaultdict(list)
    for r in nt:
        inst[r["instance"]].append(r)
    out["per_instance"] = {k: {"n": len(v), **{m: sum(succ(r, m) for r in v) / len(v) for m in METHODS}} for k, v in sorted(inst.items())}
    # by scramble length K
    byk = defaultdict(list)
    for r in nt:
        byk[r["K"]].append(r)
    out["by_K"] = {str(k): {"n": len(v), **{m: sum(succ(r, m) for r in v) / len(v) for m in METHODS}} for k, v in sorted(byk.items())}
    # move mix (attempted moves) over all trials
    mix = {m: defaultdict(int) for m in METHODS}
    for r in rows:
        for m in METHODS:
            for k, v in r[m + "_moves"].items():
                mix[m][k] += v
    out["move_mix"] = {m: {k: v / sum(mix[m].values()) for k, v in d.items()} for m, d in mix.items()}
    return out


if __name__ == "__main__":
    if len(sys.argv) > 1:          # e.g. `python scripts/summarize_paired.py results/quick`
        RES = pathlib.Path(sys.argv[1]).resolve()
    res = {}
    for mode in ("real", "synth"):
        res[mode] = summarize(mode, load(mode))
        s = res[mode]
        print(f"\n===== {mode}: {s['n_trials']} trials ({s['n_trivial']} trivial, {s['n_nontrivial']} non-trivial); same start in all: {s['all_same_start']}")
        print("success (non-trivial, mean [95% CI]):")
        for m in METHODS:
            mean, lo, hi = s["success_nontrivial"][m]
            print(f"  {m:15s} {100*mean:5.1f}% [{100*lo:5.1f},{100*hi:5.1f}]   below-target runs: {s['below_target'][m]}")
        for k, v in s["paired_nontrivial"].items():
            print(f"  {k:22s} {v['diff_pts']:+6.1f} pts [{v['lo']:+5.1f},{v['hi']:+5.1f}]  only-first {v['a_only']:3d} only-second {v['b_only']:3d}  p={v['p_sign']:.3g}")
        print("per instance:", {k: {m: round(100 * x) if m != 'n' else x for m, x in v.items()} for k, v in s["per_instance"].items()})
        print("by K:", {k: {m: round(100 * x) if m != 'n' else x for m, x in v.items()} for k, v in s["by_K"].items()})
        print("move mix:", {m: {k: round(x, 2) for k, x in d.items()} for m, d in s["move_mix"].items()})
    json.dump(res, open(RES / "paired_summary.json", "w"), indent=1)
