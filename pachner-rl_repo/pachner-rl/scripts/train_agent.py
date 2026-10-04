"""Train (or resume training of) the main agent: 900-episode mixed curriculum, one shared agent across all fillings.

usage: python scripts/train_agent.py [--checkpoint checkpoints/my_agent.pkl] [--episodes 900] [--time-limit 250]
The checkpoint is rewritten atomically after every episode, so the run can be interrupted and resumed.
Best-known targets for the training pool are read from configs/instances.json.
"""
import argparse
import os
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pachner.train as TR  # noqa: E402
from pachner.agent import PachnerQAgent  # noqa: E402
from pachner.data import load_instances  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "my_agent.pkl"))
    ap.add_argument("--episodes", type=int, default=TR.TOTAL_EPISODES)
    ap.add_argument("--time-limit", type=float, default=None, help="stop after this many seconds (then resume by re-running)")
    a = ap.parse_args()
    if pathlib.Path(a.checkpoint).resolve() == (ROOT / "checkpoints" / "ver2_checkpoint.pkl").resolve():
        sys.exit("refusing to overwrite the shipped checkpoint; choose another --checkpoint path")
    TR.TOTAL_EPISODES = a.episodes
    if os.path.exists(a.checkpoint):
        state = TR.load_checkpoint(a.checkpoint)
    else:
        inst = load_instances()["train"]
        cache = {(e["knot"], tuple(e["pq"])): e["best_known"] for e in inst.values()}
        assert set(cache) == set(map(tuple, [(k, tuple(p)) for k, p in TR.REAL_TRAIN_INSTANCES])), "training pool mismatch"
        random.seed(TR.SEED)
        state = {"agent": PachnerQAgent(hidden=(64, 64), gamma=0.97, seed=TR.SEED), "learning_curve": [], "episode": 0,
                 "rng_state": None, "py_rng_state": None, "best_known_cache": cache}
    rng = random.Random(TR.SEED)
    state = TR.train(state, a.episodes, rng, a.checkpoint, time_limit=a.time_limit)
    print(f"episodes done: {state['episode']}/{a.episodes}")
