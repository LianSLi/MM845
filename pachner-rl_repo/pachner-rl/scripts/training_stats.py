"""Training-curve statistics from the Ver2 checkpoint's learning_curve log -> results/training_stats.json."""
import json
import pathlib
import pickle
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pachner_agent  # noqa: F401,E402  (module name stored inside the pickle)

st = pickle.load(open(ROOT / "checkpoints" / "ver2_checkpoint.pkl", "rb"))
lc = st["learning_curve"]  # (ep, tag, start, final, target, success, is_synth, eps)
rows = [dict(ep=r[0], tag=r[1], start=r[2], final=r[3], target=r[4], success=r[5], synth=r[6], eps=r[7]) for r in lc]
last = rows[-300:]
out = {"episodes": len(rows)}
for name, sel in (("synthetic", [r for r in last if r["synth"]]), ("real", [r for r in last if not r["synth"]])):
    out[name] = dict(n=len(sel), success=float(np.mean([r["success"] for r in sel])), max_start=max(r["start"] for r in sel))
real_nt = [r for r in last if not r["synth"] and r["start"] > r["target"]]
out["real_nontrivial"] = dict(n=len(real_nt), success=float(np.mean([r["success"] for r in real_nt])))
out["max_start_overall"] = max(r["start"] for r in rows)
out["rolling50"] = [float(np.mean([r["success"] for r in rows[max(0, i - 49):i + 1]])) for i in range(len(rows))]
out["rolling50_real"] = []
out["real_mask"] = [int(not r["synth"]) for r in rows]
out["success"] = [int(r["success"]) for r in rows]
out["eps"] = [r["eps"] for r in rows]
json.dump(out, open(ROOT / "results" / "training_stats.json", "w"))
print({k: v for k, v in out.items() if k not in ("rolling50", "rolling50_real", "real_mask", "success", "eps")})
