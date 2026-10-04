"""Sanity check of the environment (~10 s): imports, checkpoint loading, isosig round trip, one RL rollout."""
import pathlib
import pickle
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy, sklearn, snappy  # noqa: E401,E402

from pachner.baselines import greedy_run, rl_run  # noqa: E402
from pachner.data import load_instances, triangulation_from_isosig  # noqa: E402
from pachner.env import build_mcomplex, env_from_mcomplex  # noqa: E402
from pachner.train import SCRAMBLE_SHRINK_BIAS, _apply_moves  # noqa: E402

print(f"snappy {snappy.__version__}  numpy {numpy.__version__}  scikit-learn {sklearn.__version__}  python {sys.version.split()[0]}")
inst = load_instances()
e = inst["large"]["4_1(5,1)"]
T = triangulation_from_isosig(e["isosig"])
assert T.num_tetrahedra() == e["target"] and T.num_cusps() == 0, "isosig round trip failed"
agent = pickle.load(open(ROOT / "checkpoints" / "ver2_checkpoint.pkl", "rb"))["agent"]
res = {}
for name in ("rl", "greedy"):
    mc = build_mcomplex(T); _apply_moves(mc, 10, random.Random(1), SCRAMBLE_SHRINK_BIAS)
    env = env_from_mcomplex(mc, 70); start = env.start_count
    trace = rl_run(env, agent, 0.05, 1) if name == "rl" else greedy_run(env, random.Random(1))
    res[name] = (start, env.best_count)
print("scramble start -> best reached (target %d): " % e["target"], res)
print("OK")
