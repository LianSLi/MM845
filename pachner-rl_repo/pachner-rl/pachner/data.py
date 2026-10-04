"""Data: Dehn fillings of cusped census manifolds, reference ("best known") sizes, pinned start triangulations.

SnapPy's filled_triangulation() returns a *different* closed triangulation on every call, so every
start triangulation used in an evaluation is pinned in configs/instances.json by its isomorphism
signature and rebuilt with triangulation_from_isosig().
"""
import json
import pathlib
import re

import snappy

from pachner.env import build_mcomplex

INSTANCES_PATH = pathlib.Path(__file__).resolve().parents[1] / "configs" / "instances.json"


def parse_identify(s):
    m = re.search(r'([A-Za-z]+\d+)\((-?\d+)\s*,\s*(-?\d+)\)', s)
    if not m:
        return None
    return m.group(1), (int(m.group(2)), int(m.group(3)))


def best_known_minimum(parent_name, fill_coeffs, n_restarts=100):
    M = snappy.Manifold(parent_name)
    M.dehn_fill(fill_coeffs)
    best = M.filled_triangulation().num_tetrahedra()
    for _ in range(n_restarts):
        T = M.filled_triangulation()
        T.randomize()
        T.simplify()
        best = min(best, T.num_tetrahedra())
    return best


def best_known_for_instance(knot_name, pq, n_restarts=100):
    M = snappy.Manifold(knot_name)
    N = M.copy(); N.dehn_fill(pq)
    idstr = str(N.identify())
    parsed = parse_identify(idstr)
    if parsed is not None:
        parent_name, parent_fill = parsed
        best = best_known_minimum(parent_name, parent_fill, n_restarts=n_restarts)
    else:
        best = N.filled_triangulation().num_tetrahedra()
        for _ in range(n_restarts):
            T = N.filled_triangulation()
            T.randomize()
            T.simplify()
            best = min(best, T.num_tetrahedra())
    return best, idstr


def reference_invariants(snappy_manifold):
    return float(snappy_manifold.volume()), str(snappy_manifold.identify())


def get_triangulation(name, pq, max_attempts=6):
    # SnapPy's own filled_triangulation() is itself flaky ~30-60% of the
    # time even at zero Pachner moves (a solver-conditioning quirk, not a
    # topology bug) -- retry for a cleanly-solving start.
    M = snappy.Manifold(name)
    N = M.copy(); N.dehn_fill(pq)
    orig_vol, orig_id = reference_invariants(N)
    for _ in range(max_attempts):
        T = N.filled_triangulation()
        mc = build_mcomplex(T)
        M0 = mc.snappy_manifold()
        if M0.solution_type() == "all tetrahedra positively oriented":
            return T, orig_vol, orig_id
    return T, orig_vol, orig_id


def triangulation_from_isosig(sig):
    """Rebuild a closed triangulation from its isomorphism signature (deterministic).

    remove_finite_vertices=False is essential: by default SnapPy replaces the closed one-vertex
    triangulation with a small cusped triangulation plus a Dehn filling (wrong tetrahedra count).
    """
    return snappy.Triangulation(sig, remove_finite_vertices=False)


def isosig_of(T):
    return T.triangulation_isosig(decorated=False)


def key_of(knot, pq):
    return f"{knot}({pq[0]},{pq[1]})"


def load_instances(path=INSTANCES_PATH):
    with open(path) as f:
        return json.load(f)


def make_base_problem(manifold_name="4_1", filling=(5, 1), isosig=None):
    """Return (base_T, original size, reference volume, reference H1) for a Dehn filling.

    With isosig=None a fresh random closed triangulation is drawn; otherwise it is rebuilt from the signature.
    """
    M = snappy.Manifold(manifold_name)
    M.dehn_fill(filling)
    base_T = M.filled_triangulation() if isosig is None else triangulation_from_isosig(isosig)
    return base_T, base_T.num_tetrahedra(), float(M.volume()), str(M.homology())
