"""Fitted-Q agent (scikit-learn MLP) and the type-balanced action sampler.

Checkpoints pickled by the original training run refer to this code under the module name
`pachner_agent`; the top-level file pachner_agent.py re-exports it so they still load.
"""

import random
import numpy as np
from sklearn.neural_network import MLPRegressor


def biased_choice(actions, rng, shrink_bias=0.0):
    by_type = {}
    for i, a in enumerate(actions):
        by_type.setdefault(a[0], []).append(i)
    types = list(by_type.keys())
    weights = []
    for t in types:
        w = 1.0
        if t == '3-2':
            w *= max(0.02, 1.0 + shrink_bias)
        elif t == '2-3':
            w *= max(0.02, 1.0 - shrink_bias)
        weights.append(w)
    total = sum(weights)
    r = rng.random() * total
    cum = 0.0
    chosen_type = types[-1]
    for t, w in zip(types, weights):
        cum += w
        if r <= cum:
            chosen_type = t
            break
    return rng.choice(by_type[chosen_type])


class PachnerQAgent:
    def __init__(self, hidden=(64, 64), gamma=0.97, seed=0):
        self.gamma = gamma
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden, activation="relu", solver="adam",
            learning_rate_init=1e-3, warm_start=True, max_iter=1,
            random_state=seed,
        )
        self.fitted = False
        self.replay = []
        self.rng = random.Random(seed)

    def q_values(self, s_feat, action_feats):
        if not action_feats:
            return np.array([])
        X = np.stack([np.concatenate([s_feat, af]) for af in action_feats])
        if not self.fitted:
            return np.zeros(len(action_feats))
        return self.model.predict(X)

    def select(self, s_feat, actions, action_feats, epsilon, explore_shrink_bias=0.3):
        if not actions:
            return None
        if self.rng.random() < epsilon:
            return biased_choice(actions, self.rng, shrink_bias=explore_shrink_bias)
        qs = self.q_values(s_feat, action_feats)
        return int(np.argmax(qs))

    def remember(self, s, a_feat, r, s_next, next_action_feats, done):
        self.replay.append((s, a_feat, r, s_next, next_action_feats, done))
        if len(self.replay) > 40000:
            self.replay.pop(0)

    def train_batch(self, batch_size=256):
        if len(self.replay) < 32:
            return
        batch = self.rng.sample(self.replay, min(batch_size, len(self.replay)))
        X, y = [], []
        for s, a_feat, r, s_next, next_afs, done in batch:
            target = r
            if not done and next_afs:
                q_next = self.q_values(s_next, next_afs)
                target += self.gamma * float(np.max(q_next))
            X.append(np.concatenate([s, a_feat]))
            y.append(target)
        self.model.partial_fit(np.array(X), np.array(y))
        self.fitted = True
