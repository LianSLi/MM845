"""Figures for the report from results/*.json(l). usage: python scripts/make_figures.py"""
import json
import pathlib
import pickle
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FIG = ROOT / "report" / "figs"; FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42, "font.family": "serif",
                     "mathtext.fontset": "cm"})
INK, MUTED = "#0b0b0b", "#52514e"
COL = {"rl": "#2a78d6", "greedy": "#eb6834", "greedy_bal": "#1baf7a", "greedy_plateau": "#9a9a95", "random": "#c9c8c2"}
HATCH = {"rl": "", "greedy": "", "greedy_bal": "", "greedy_plateau": "//", "random": ".."}
NAME = {"rl": "RL", "greedy": "greedy", "greedy_bal": "greedy\nbalanced", "greedy_plateau": "greedy\n4-4", "random": "random\nwalk"}
M = ["rl", "greedy", "greedy_bal", "greedy_plateau", "random"]

# ---------- paired success ----------
S = json.load(open(ROOT / "results" / "paired_summary.json"))
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.05), sharey=True)
for ax, mode, title in zip(axes, ("real", "synth"), ("Real Dehn fillings (held out)", "Scrambled minimal triangulations")):
    s = S[mode]["success_nontrivial"]
    for i, m in enumerate(M):
        mean, lo, hi = (100 * x for x in s[m])
        ax.bar(i, mean, 0.72, color=COL[m], hatch=HATCH[m], edgecolor="white" if not HATCH[m] else MUTED, linewidth=0.4)
        ax.errorbar(i, mean, yerr=[[mean - lo], [hi - mean]], color=INK, lw=0.8, capsize=2)
        ax.text(i, hi + 1.5, f"{mean:.0f}", ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.set_xticks(range(len(M))); ax.set_xticklabels([NAME[m] for m in M], fontsize=7)
    ax.set_title(f"{title}  (n = {S[mode]['n_nontrivial']})", fontsize=8, color=INK, loc="left")
    ax.set_ylim(0, 105); ax.yaxis.grid(True, lw=0.3, color="#d8d8d4"); ax.set_axisbelow(True)
axes[0].set_ylabel("trials reaching target (%)")
fig.tight_layout(); fig.savefig(FIG / "fig_paired.pdf"); plt.close(fig)

# ---------- training curve ----------
T = json.load(open(ROOT / "results" / "training_stats.json"))
succ, real, eps = np.array(T["success"]), np.array(T["real_mask"]), np.array(T["eps"])
def roll(mask, w=60):
    idx = np.where(mask)[0]; v = succ[idx]
    y = np.array([v[max(0, i - w + 1):i + 1].mean() for i in range(len(v))])
    return idx[w // 2:], y[w // 2:]   # drop the start-up transient where the window is short
fig, ax = plt.subplots(figsize=(3.3, 1.95))
for mask, lab, c in ((real == 1, "real Dehn fillings", COL["rl"]), (real == 0, "synthetic scrambles", COL["greedy"])):
    x, y = roll(mask)
    ax.plot(x, 100 * y, color=c, lw=1.4, label=lab)
ax.set_xlabel("training episode"); ax.set_ylabel("success, rolling 60 (%)")
ax.set_ylim(0, 100); ax.yaxis.grid(True, lw=0.3, color="#d8d8d4"); ax.legend(frameon=False, fontsize=7, loc="lower right")
fig.tight_layout(); fig.savefig(FIG / "fig_training.pdf"); plt.close(fig)
print("figures: paired, training")



# ---------- data analysis: gap of SnapPy start triangulations, and excess after scrambling ----------
from collections import Counter  # noqa: E402
from pachner.stats import cluster_bootstrap  # noqa: E402
I = json.load(open(ROOT / "configs" / "instances.json"))
gaps = Counter(sz - e["best_known"] for g in ("held_out", "train") for e in I[g].values() for sz in e["pool_sizes"])
rows = [json.loads(l) for l in open(ROOT / "results" / "paired_real.jsonl")]
fig, axes = plt.subplots(1, 2, figsize=(6.6, 1.8))
ax = axes[0]
xs = sorted(gaps); tot = sum(gaps.values())
ax.bar(xs, [100 * gaps[x] / tot for x in xs], 0.6, color=COL["rl"])
for x in xs:
    ax.text(x, 100 * gaps[x] / tot + 1.5, f"{100 * gaps[x] / tot:.0f}%", ha="center", fontsize=7.5)
ax.set_xticks(xs); ax.set_xlabel("start size minus best-known size (tetrahedra)"); ax.set_ylabel("pooled triangulations (%)")
ax.set_title("SnapPy's filled triangulations", fontsize=8, loc="left"); ax.set_ylim(0, 80)
ax = axes[1]
for j, K in enumerate((5, 10, 20)):
    ex = [r["start"] - r["target"] for r in rows if r["K"] == K]
    c = Counter(ex); xs2 = list(range(0, max(max(c), 1) + 2))
    ax.step(xs2, [100 * c.get(x, 0) / len(ex) for x in xs2], where="mid", color=["#9fc3ee", COL["rl"], "#164b8a"][j], lw=1.6, label=f"$K$ = {K}")
ax.set_xlabel("start size minus best-known size after scrambling"); ax.set_ylabel("test trials (%)")
ax.set_title("Held-out test starts", fontsize=8, loc="left"); ax.legend(frameon=False, fontsize=7)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
for a in axes:
    a.yaxis.grid(True, lw=0.3, color="#d8d8d4"); a.set_axisbelow(True)
fig.tight_layout(); fig.savefig(FIG / "fig_data.pdf"); plt.close(fig)

# ---------- where does RL help? per-filling differences with cluster bootstrap over seeds ----------
srows = [json.loads(l) for l in open(ROOT / "results" / "paired_synth.jsonl")]
def forest(ax, rws, title):
    names = sorted({r["instance"] for r in rws}, key=lambda n: n)
    for i, n in enumerate(names):
        sub = [r for r in rws if r["instance"] == n and not r["trivial"]]
        for k, (m, c, mk, off) in enumerate((("greedy", COL["greedy"], "o", -0.12), ("greedy_bal", COL["greedy_bal"], "s", 0.12))):
            mean, lo, hi = cluster_bootstrap(sub, lambda r, m=m: int(r["rl"] <= r["target"]) - int(r[m] <= r["target"]), lambda r: r["seed"], B=2000)
            ax.errorbar(100 * mean, i + off, xerr=[[100 * (mean - lo)], [100 * (hi - mean)]], fmt=mk, color=c, ms=4, lw=1, capsize=1.5,
                        label=("RL minus greedy" if m == "greedy" else "RL minus balanced greedy") if i == 0 else None)
    ax.axvline(0, color=INK, lw=0.6)
    ax.set_yticks(range(len(names))); ax.set_yticklabels([n.replace("_", r"\_") if False else n for n in names], fontsize=7)
    ax.invert_yaxis(); ax.set_xlabel("difference in success rate (points)"); ax.set_title(title, fontsize=8, loc="left")
    ax.xaxis.grid(True, lw=0.3, color="#d8d8d4"); ax.set_axisbelow(True)
fig, axes = plt.subplots(1, 2, figsize=(6.6, 1.95), sharex=True)
forest(axes[0], rows, "Held-out fillings"); forest(axes[1], srows, "Scrambled minimal manifolds")
h, l = axes[1].get_legend_handles_labels()
fig.legend(h, l, frameon=False, fontsize=7, loc="lower center", ncol=2)
fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(FIG / "fig_perfilling.pdf"); plt.close(fig)
print("figures: data, perfilling")

# ---------- 100-tetrahedron trajectories (same scramble, three methods) ----------
if "--traj" in sys.argv or True:
    from pachner.baselines import greedy_run, rl_run
    from pachner.data import load_instances, triangulation_from_isosig
    from pachner.env import env_from_mcomplex
    from pachner.large import grow_to_exact_tets
    agent = pickle.load(open(ROOT / "checkpoints" / "ver2_checkpoint.pkl", "rb"))["agent"]
    e = load_instances()["large"]["4_1(5,1)"]; base = triangulation_from_isosig(e["isosig"])
    fig, ax = plt.subplots(figsize=(3.3, 1.95))
    for name, label in (("greedy", "greedy"), ("greedy_bal", "greedy balanced"), ("rl", "RL (zero-shot)")):
        mc, _ = grow_to_exact_tets(base, 100, seed=12345 + 0)
        env = env_from_mcomplex(mc, 600)
        if name == "rl":
            tr = rl_run(env, agent, 0.05, 12345)
        else:
            tr = greedy_run(env, random.Random(12345), "flat" if name == "greedy" else "balanced")
        ax.plot(range(len(tr)), tr, color=COL[name], lw=1.1, label=f"{label} (best {env.best_count})")
    ax.axhline(e["target"], color=INK, lw=0.7, ls="--"); ax.set_ylim(0, 105)
    ax.text(ax.get_xlim()[1] * 0.98, e["target"] - 2, "original size (9)", ha="right", va="top", fontsize=7)
    ax.set_xlabel("Pachner move"); ax.set_ylabel("tetrahedra"); ax.legend(frameon=False, fontsize=6.5)
    ax.yaxis.grid(True, lw=0.3, color="#d8d8d4"); fig.tight_layout(); fig.savefig(FIG / "fig_traj100.pdf"); plt.close(fig)
    print("figure: traj100")
