"""Pachner-move environment built on SnapPy's t3mlite Mcomplex (moves 2-3, 3-2, 4-4)."""
from collections import Counter

import numpy as np
from snappy.snap import t3mlite as t3m
from snappy.snap.t3mlite import Mcomplex

STEP_COST = 0.02

MAX_VALENCE_BUCKET = 10


def build_mcomplex(snappy_triangulation):
    mc = Mcomplex(snappy_triangulation)
    mc.rebuild()
    return mc


def legal_actions(mc):
    actions = []
    seen_faces = set()
    for tet in mc.Tetrahedra:
        for F in t3m.TwoSubsimplices:
            face_obj = tet.Class[F]
            if face_obj.Index in seen_faces:
                continue
            seen_faces.add(face_obj.Index)
            if tet.Neighbor[F] is tet:
                continue
            actions.append(('2-3', tet, F))
    for e in mc.Edges:
        val = e.valence()
        if val == 3:
            actions.append(('3-2', e))
        elif val == 4:
            actions.append(('4-4', e))
    return actions


def _face_edge_valences(tet, F):
    verts_on_face = [v for v in (1, 2, 4, 8) if F & v]
    edge_masks = [F ^ v for v in verts_on_face]
    return [tet.Class[em].valence() for em in edge_masks]


def action_features(action):
    kind = action[0]
    is23 = float(kind == '2-3'); is32 = float(kind == '3-2'); is44 = float(kind == '4-4')
    if kind == '2-3':
        delta = 1.0; vals = _face_edge_valences(action[1], action[2])
    elif kind == '3-2':
        delta = -1.0; vals = [3]
    else:
        delta = 0.0; vals = [4]
    vmean = min(np.mean(vals), MAX_VALENCE_BUCKET) / MAX_VALENCE_BUCKET
    vmax = min(np.max(vals), MAX_VALENCE_BUCKET) / MAX_VALENCE_BUCKET
    return np.array([is23, is32, is44, delta / 3.0, vmean, vmax], dtype=np.float32)


def state_features(mc, start_count, steps_left, max_steps):
    n = len(mc.Tetrahedra)
    valences = [e.valence() for e in mc.Edges]
    m = max(len(valences), 1)
    frac_le3 = sum(v <= 3 for v in valences) / m
    frac_4 = sum(v == 4 for v in valences) / m
    frac_ge6 = sum(v >= 6 for v in valences) / m
    return np.array([n / start_count, steps_left / max_steps, frac_le3, frac_4,
                      frac_ge6, len(valences) / max(1, start_count)], dtype=np.float32)


class PachnerEnv:
    def __init__(self, snappy_triangulation, max_steps=40):
        self.mc = build_mcomplex(snappy_triangulation)
        self.start_count = len(self.mc.Tetrahedra)
        self.max_steps = max_steps
        self.steps = 0
        self.best_count = self.start_count
        self.move_counts = Counter()

    def current_actions(self):
        return legal_actions(self.mc)

    def obs(self, actions=None):
        if actions is None:
            actions = self.current_actions()
        s = state_features(self.mc, self.start_count, self.max_steps - self.steps, self.max_steps)
        return s, actions

    def step(self, action):
        prev = len(self.mc.Tetrahedra)
        kind = action[0]
        self.__dict__.setdefault('move_counts', Counter())[kind] += 1  # attempted moves, by type
        try:
            if kind == '2-3':
                ok = self.mc.two_to_three(action[2], action[1], must_succeed=False)
            elif kind == '3-2':
                ok = self.mc.three_to_two(action[1].get_arrow(), must_succeed=False)
            else:
                ok = self.mc.four_to_four(action[1].get_arrow(), must_succeed=False)
        except Exception:
            ok = False
        if ok:
            self.mc.rebuild()
        new = len(self.mc.Tetrahedra)
        self.best_count = min(self.best_count, new)
        self.steps += 1
        reward = (prev - new) - STEP_COST if ok else -1.0
        done = (self.steps >= self.max_steps) or (new > 3 * self.start_count + 10)
        return reward, done, ok

    def tet_count(self):
        return len(self.mc.Tetrahedra)

    def to_snappy(self):
        return self.mc.snappy_manifold()


def env_from_mcomplex(mc, max_steps=180):
    # Create a PachnerEnv from an already-built Mcomplex without changing it.
    env = PachnerEnv.__new__(PachnerEnv)
    env.mc = mc
    env.start_count = len(mc.Tetrahedra)
    env.max_steps = max_steps
    env.steps = 0
    env.best_count = env.start_count
    return env
