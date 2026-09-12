"""

Parameterization: a small policy pi_theta over per-sample state features (neural
probs, confidence, fired content rules, fired x rule-precision, affect 6-d)
emits a 5-d adjustment Delta added to the neural logits; the final label is
argmax(neural_logits + Delta). The output head is zero-initialised, so training
starts at Delta=0 (exactly the neural path) and learns to use the rules/affect
in the state.

The policy (train_rlvr) is trained by REINFORCE against a verifiable programmatic
reward: class-balanced correctness, which needs no annotation beyond the labels.

"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nesymis import config
from nesymis.metrics import compute_metrics
from nesymis.base import set_seed

_dev = "cuda" if torch.cuda.is_available() else "cpu"

RL_CFG = {"hidden": 32, "epochs": 300, "lr": 3e-3, "batch_size": 256,
          "entropy_coef": 0.01, "override_penalty": 0.0}


def _softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def build_state(neural_logits, fired_mat, rprec_vec, affect):
    """Per-sample policy state: [probs(5), conf(1), fired(4), fired*prec(4), affect(6)] -> (N,20)."""
    probs = _softmax(neural_logits)
    conf = probs.max(1, keepdims=True)
    fired = fired_mat.astype(np.float32)
    fired_prec = fired * rprec_vec[None, :]
    return np.concatenate([probs, conf, fired, fired_prec, affect], axis=1).astype(np.float32)


class PolicyNet(nn.Module):
    def __init__(self, in_dim, n_classes=config.NUM_CLASSES, hidden=32):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU())
        self.head = nn.Linear(hidden, n_classes)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)   # warm-start: Delta = 0 => policy == neural

    def forward(self, s):
        return self.head(self.body(s))


class LinearEvidencePolicy(nn.Module):
    """Deployed decision layer: fully LINEAR over the state, with structural
    constraints that make every decision exactly decomposable:

      * fired-rule features may only ADD evidence, and only to their OWN class
        (ReLU-constrained weights, class-aligned scatter);
      * neural probabilities, confidence, and affect scores enter through an
        unconstrained linear map (exact per-feature attributions);
      * zero/near-zero init => warm start at the neural decision.

    State layout: [probs(K), conf(1), fired(R), fired*prec(R), emo(6)];
    deployed K=5 classes, R=4 rules; rule_classes maps rule j -> its class index."""

    def __init__(self, in_dim, n_classes=config.NUM_CLASSES, rule_classes=None):
        super().__init__()
        if rule_classes is None:
            rule_classes = [config.LABEL_TO_IDX[c] for c in config.STEREO_CLASSES]
        K, R = n_classes, len(rule_classes)
        assert in_dim == K + 1 + 2 * R + 6, f"unexpected state dim {in_dim} (K={K}, R={R})"
        self._k, self._r = K, R
        self.free = nn.Linear(K + 1 + 6, n_classes)          # probs+conf+emo -> all classes
        nn.init.zeros_(self.free.weight)
        nn.init.zeros_(self.free.bias)
        self.w_fired = nn.Parameter(torch.full((R,), 0.01))  # rule -> own class only
        self.w_fprec = nn.Parameter(torch.full((R,), 0.01))
        idx_free = torch.tensor(list(range(K + 1)) + list(range(K + 1 + 2 * R, in_dim)))
        self.register_buffer("idx_free", idx_free)
        self.register_buffer("stereo_idx", torch.tensor(list(rule_classes)))

    def forward(self, s):
        K, R = self._k, self._r
        delta = self.free(s[:, self.idx_free])
        contrib = (s[:, K + 1:K + 1 + R] * torch.relu(self.w_fired)
                   + s[:, K + 1 + R:K + 1 + 2 * R] * torch.relu(self.w_fprec))   # (N,R) >= 0
        return delta.index_add(1, self.stereo_idx, contrib)

    def rule_weights(self):
        return (torch.relu(self.w_fired).detach().cpu().numpy(),
                torch.relu(self.w_fprec).detach().cpu().numpy())


def balanced_weights(y, k=config.NUM_CLASSES):
    cnt = np.bincount(y, minlength=k).astype(np.float32)
    return cnt.sum() / (k * np.clip(cnt, 1.0, None))


@torch.no_grad()
def greedy_pred(policy, neural_logits, state):
    policy.eval()
    nl = torch.from_numpy(neural_logits).float().to(_dev)
    s = torch.from_numpy(state).float().to(_dev)
    return (nl + policy(s)).argmax(1).cpu().numpy()


def train_rlvr(neural_logits, state, y, train_mask, val_mask, cfg=None,
               class_weight=None, seed=config.SEED, verbose=False, policy_ctor=None):
    """REINFORCE on the one-step decision; reward = class-balanced correctness.
    policy_ctor(in_dim) -> nn.Module optionally overrides the PolicyNet architecture."""
    cfg = cfg or RL_CFG
    set_seed(seed)
    w = class_weight if class_weight is not None else balanced_weights(y[train_mask])
    wt = torch.tensor(w, dtype=torch.float32, device=_dev)
    NL = torch.from_numpy(neural_logits).float().to(_dev)
    S = torch.from_numpy(state).float().to(_dev)
    Y = torch.from_numpy(y).long().to(_dev)
    npred = NL.argmax(1)

    ctor = policy_ctor or (lambda d: PolicyNet(d, hidden=cfg["hidden"]))
    policy = ctor(state.shape[1]).to(_dev)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg["lr"])
    tr = np.where(train_mask)[0]
    best_f1, best_state = -1.0, None
    for ep in range(cfg["epochs"]):
        policy.train()
        np.random.shuffle(tr)
        for i in range(0, len(tr), cfg["batch_size"]):
            bi = torch.from_numpy(tr[i:i + cfg["batch_size"]]).long().to(_dev)
            logits = NL[bi] + policy(S[bi])
            logp = torch.log_softmax(logits, 1)
            p = logp.exp()
            a = torch.multinomial(p, 1).squeeze(1)
            logpa = logp.gather(1, a[:, None]).squeeze(1)
            r = wt[Y[bi]] * (a == Y[bi]).float()
            if cfg["override_penalty"] > 0:
                r = r - cfg["override_penalty"] * (a != npred[bi]).float()
            ent = -(p * logp).sum(1).mean()
            loss = -((r - r.mean()).detach() * logpa).mean() - cfg["entropy_coef"] * ent
            opt.zero_grad()
            loss.backward()
            opt.step()
        f1 = compute_metrics(y[val_mask], greedy_pred(policy, neural_logits[val_mask], state[val_mask]))["macro_f1"]
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
        if verbose and ep % 50 == 0:
            print(f"  ep {ep:>3}  val macroF1={f1:.3f}")
    policy.load_state_dict(best_state)
    return policy, best_f1


