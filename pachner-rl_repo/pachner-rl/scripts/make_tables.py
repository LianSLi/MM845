"""LaTeX fragments and number macros for the report, generated from results/ and configs/ (never typed by hand).
usage: python scripts/make_tables.py  -> report/numbers.tex, report/tab_*.tex"""
import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
R = ROOT / "report"
S = json.load(open(ROOT / "results" / "paired_summary.json"))
T = json.load(open(ROOT / "results" / "training_stats.json"))
I = json.load(open(ROOT / "configs" / "instances.json"))
Z = json.load(open(ROOT / "results" / "large_zeroshot_4_1(5,1).json"))
C = json.load(open(ROOT / "results" / "large_compete.json"))

NAME = {"rl": "RL agent", "greedy": "greedy (flat fallback)", "greedy_bal": "greedy (balanced fallback)",
        "greedy_plateau": "greedy (4-4 fallback)", "random": "biased random walk"}
M = ["rl", "greedy", "greedy_bal", "greedy_plateau", "random"]
pct = lambda x: f"{100 * x:.1f}"  # noqa: E731
mac = {}


def wr(path, text):
    """Write table rows WITHOUT the final row terminator (the report adds `\\\\` after \\input; a file ending in `\\\\`
    makes booktabs rules fail after \\input)."""
    text = text.rstrip()
    if text.endswith("\\\\"):
        text = text[:-2].rstrip()
    pathlib.Path(path).write_text(text)



def fmt(name):
    """'4_1(7,2)' -> $4_1(7,2)$ ; 'm009(5,1)' -> $\\mathrm{m009}(5,1)$ (math mode avoids monospace comma gaps)."""
    import re
    m = re.match(r"^(\d+)_(\d+)\((-?\d+),(-?\d+)\)$", name)
    if m:
        return f"${m.group(1)}_{m.group(2)}({m.group(3)},{m.group(4)})$"
    m = re.match(r"^([A-Za-z]+\d+)\((-?\d+),(-?\d+)\)$", name)
    if m:
        return f"$\\mathrm{{{m.group(1)}}}({m.group(2)},{m.group(3)})$"
    return name

# ---------- paired tables ----------
for mode, tag in (("real", "Real"), ("synth", "Synth")):
    s = S[mode]
    mac[f"n{tag}"] = s["n_trials"]; mac[f"nNT{tag}"] = s["n_nontrivial"]; mac[f"nTriv{tag}"] = s["n_trivial"]
    rows = []
    for m in M:
        mean, lo, hi = s["success_nontrivial"][m]
        if m == "rl":
            d = r"\multicolumn{2}{c}{--}"
        else:
            p = s["paired_nontrivial"][f"rl-{m}"]
            d = f"${p['diff_pts']:+.1f}$ $[{p['lo']:+.1f},{p['hi']:+.1f}]$ & {p['a_only']}/{p['b_only']}"
        rows.append(f"{NAME[m]} & {pct(mean)} [{pct(lo)}, {pct(hi)}] & {d} \\\\")
    wr(R / f"tab_paired_{mode}.tex", "\n".join(rows) + "\n")
    for m in M:
        mac[f"succ{m.replace('_', '')}{tag}"] = pct(s["success_nontrivial"][m][0])
    for m in ("greedy", "greedy_bal", "greedy_plateau", "random"):
        p = s["paired_nontrivial"][f"rl-{m}"]
        k = m.replace("_", "")
        mac[f"diff{k}{tag}"] = f"{p['diff_pts']:+.1f}"; mac[f"lo{k}{tag}"] = f"{p['lo']:+.1f}"; mac[f"hi{k}{tag}"] = f"{p['hi']:+.1f}"
        mac[f"p{k}{tag}"] = f"{p['p_sign']:.2g}"
    mac[f"below{tag}"] = sum(s["below_target"].values())

# per-instance real
inst_rows = []
for k, v in S["real"]["per_instance"].items():
    inst_rows.append(f"{fmt(k)} & {v['n']} & " + " & ".join(f"{100 * v[m]:.0f}" for m in M) + r" \\")
wr(R / "tab_perinst.tex", "\n".join(inst_rows) + "\n")

# ---------- instances ----------
rows = []
for key, e in I["held_out"].items():
    ps = e["pool_sizes"]
    ident = fmt(e["identify"].strip("[]")) if e["identify"].strip("[]") else "--"
    rows.append(f"{fmt(key)} & {ident} & {e['homology']} & {e['best_known']} & {min(ps)}--{max(ps)} \\\\")
wr(R / "tab_instances.tex", "\n".join(rows) + "\n")
mac["trainList"] = ", ".join(fmt(k) for k in I["train"].keys())
gaps = [min(e["pool_sizes"]) - e["best_known"] for g in ("held_out", "train") for e in I[g].values()]
gaps_med = [int(np.median(e["pool_sizes"])) - e["best_known"] for g in ("held_out", "train") for e in I[g].values()]
mac["gapMax"] = max(max(e["pool_sizes"]) - e["best_known"] for g in ("held_out", "train") for e in I[g].values())
mac["gapMin"] = min(gaps)
mac["nTrainInst"] = len(I["train"]); mac["nHeldInst"] = len(I["held_out"])
mac["restarts"] = I["restarts"]

# ---------- training ----------
mac["trSynth"] = f"{100 * T['synthetic']['success']:.0f}"; mac["trSynthN"] = T["synthetic"]["n"]
mac["trReal"] = f"{100 * T['real']['success']:.0f}"; mac["trRealN"] = T["real"]["n"]
mac["trRealNT"] = f"{100 * T['real_nontrivial']['success']:.0f}"; mac["trMaxStart"] = T["max_start_overall"]

# ---------- 100 tets ----------
zs = []
for b in ("180", "600"):
    z = Z["budgets"][b]
    zs.append(f"{b} & " + " & ".join(f"{z[n]['reached']}/{Z['scrambles']} ({z[n]['mean_best']:.1f})" for n in ("rl", "greedy", "greedy_balanced")) + r" \\")
wr(R / "tab_zeroshot.tex", "\n".join(zs) + "\n")
mac["zsN"] = Z["scrambles"]; mac["zsTarget"] = Z["target"]
chk = {"ok": 0, "other": 0}
for b in Z["budgets"].values():
    for n in b.values():
        for k, v in n["checks"].items():
            chk["ok" if k.startswith("ok") else "other"] += v
for key, res in C["results"].items():
    for b, ch in res["checks"].items():
        for n, c in ch.items():
            for k, v in c.items():
                chk["ok" if k.startswith("ok") else "other"] += v
mac["chkOk"] = chk["ok"]; mac["chkOther"] = chk["other"]
cn = C["scrambles"]; mac["cN"] = cn
order = ["RL zero-shot, eps 0.05", "RL fine-tuned, eps 0.05", "RL zero-shot, eps 0", "RL fine-tuned, eps 0", "greedy", "greedy balanced"]
lab = {"RL zero-shot, eps 0.05": r"RL zero-shot ($\varepsilon{=}0.05$)", "RL fine-tuned, eps 0.05": r"RL fine-tuned ($\varepsilon{=}0.05$)",
       "RL zero-shot, eps 0": r"RL zero-shot ($\varepsilon{=}0$)", "RL fine-tuned, eps 0": r"RL fine-tuned ($\varepsilon{=}0$)",
       "greedy": "greedy", "greedy balanced": "greedy balanced"}
rows = []
for n in order:
    cells = []
    for key in C["results"]:
        res = C["results"][key]
        for b in ("180", "600"):
            v = np.array(res["budgets"][b][n]); cells.append(f"{int((v <= res['target']).sum())} ({v.mean():.1f})")
    rows.append(f"{lab[n]} & " + " & ".join(cells) + r" \\")
wr(R / "tab_compete.tex", "\n".join(rows) + "\n")
mac["cTargets"] = ", ".join(f"{k}: {v['target']}" for k, v in C["results"].items())

# ---------- fine-tuning log ----------
FL = json.load(open(ROOT / "results" / "large_finetune_log.json"))
last = [r for r in FL if r["ep"] >= len(FL) - 50]
big = [r for r in last if r["size"] >= 90]
mac["ftEpisodes"] = len(FL); mac["ftBigReached"] = sum(r["reached"] for r in big); mac["ftBigN"] = len(big)
mac["ftLastReached"] = sum(r["reached"] for r in last)


# ---------- success by scramble length ----------
rows = []
for mode, lab in (("real", "fillings"), ("synth", "minimal")):
    for k, v in S[mode]["by_K"].items():
        rows.append(f"{lab}, {'$K$' if mode == 'real' else 'depth'} {k} & {v['n']} & " + " & ".join(f"{100 * v[m]:.0f}" for m in ("rl", "greedy", "greedy_bal", "greedy_plateau", "random")) + r" \\")
wr(R / "tab_byK.tex", "\n".join(rows))

# ---------- zero-shot 100-tet run (20 scrambles) ----------
for b, tag in (("180", "a"), ("600", "b")):
    for n, short in (("rl", "RL"), ("greedy", "GR"), ("greedy_balanced", "GB")):
        mac[f"zs{short}{tag}"] = Z["budgets"][b][n]["reached"]
        mac[f"zsMean{short}{tag}"] = f"{Z['budgets'][b][n]['mean_best']:.1f}"
        mac[f"zsSecs{short}{tag}"] = f"{Z['budgets'][b][n]['secs_per_run']:.1f}"


# ---------- excess over target at the start of the real test trials ----------
RJ = [json.loads(l) for l in open(ROOT / "results" / "paired_real.jsonl")]
for K, tag in ((5, "a"), (10, "b"), (20, "c")):
    mac[f"medK{tag}"] = f"{np.median([r['start'] - r['target'] for r in RJ if r['K'] == K]):.0f}"
mac["gapWithinOne"] = f"{100 * np.mean([sz - e['best_known'] <= 1 for g in ('held_out', 'train') for e in I[g].values() for sz in e['pool_sizes']]):.0f}"

# ---------- macros ----------
def esc(v):
    return str(v)
with open(R / "numbers.tex", "w") as f:
    for k, v in mac.items():
        f.write(f"\\newcommand{{\\{k}}}{{{esc(v)}}}\n")
print(f"wrote {len(mac)} macros and tables to {R}")
