"""
Reinforcement System — multi-algorithm policy over ITOps state-action space.

Supported algorithms
--------------------
q-learning      Off-policy TD; Q(s,a) ← Q + α[r + γ·maxQ(s′) − Q]
double-q        Two tables; one selects best_a, the other evaluates it.
                Eliminates maximisation bias in stochastic environments.
expected-sarsa  Expected value under ε-greedy policy instead of max.
                Lower variance; more stable in noisy-reward settings.
ppo             Proximal Policy Optimisation (tabular softmax policy + value fn).
                Clip objective prevents large destructive updates.
dpo             Direct Preference Optimisation adapted for sequential decisions.
                Treats resolved steps as preferred, unresolved as rejected.
actor-critic    Advantage Actor-Critic (A2C): policy-gradient actor +
                TD-error value critic. Stable, sample-efficient.
"""
import json
import logging
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from agent.environment import ACTIONS

logger = logging.getLogger("itops.rl")

CONCLUSIVE_MIN_ATTEMPTS = 5
CONCLUSIVE_MIN_RATE     = 0.80

ALGORITHMS = {
    "q-learning":     "Q-Learning        — off-policy TD, fast baseline",
    "double-q":       "Double Q-Learning — removes overestimation bias ⭐",
    "expected-sarsa": "Expected SARSA    — expected value, lower variance",
    "ppo":            "PPO               — clipped policy gradient",
    "dpo":            "DPO               — preference optimisation",
    "actor-critic":   "Actor-Critic (A2C)— policy gradient + value critic",
}

N_ACTIONS = len(ACTIONS)
ACTION_IDX = {a: i for i, a in enumerate(ACTIONS)}


# ── helpers ───────────────────────────────────────────────────────────────────
def _softmax(logits: dict) -> dict:
    vals = list(logits.values())
    m = max(vals)
    exp_v = {k: math.exp(v - m) for k, v in logits.items()}
    tot = sum(exp_v.values())
    return {k: v / tot for k, v in exp_v.items()}


def _init_logits():
    return {a: 0.0 for a in ACTIONS}


def _init_q():
    return {a: 0.0 for a in ACTIONS}


@dataclass
class PolicyImprovement:
    state_key: tuple
    old_best_action: str
    new_best_action: str
    q_delta: float
    description: str


class RLSystem:
    """
    Tabular reinforcement system — six interchangeable update algorithms.
    All algorithms expose the same public API so the evaluation harness
    requires no changes when switching algorithms.
    """

    def __init__(
        self,
        alpha: float = 0.15,
        gamma: float = 0.90,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        checkpoint_dir: str = "./rl_checkpoints",
        algorithm: str = "q-learning",
    ):
        self.alpha   = alpha
        self.gamma   = gamma
        self.epsilon = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.algorithm     = algorithm if algorithm in ALGORITHMS else "q-learning"
        self._checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Primary Q-table (used by all algorithms; policy-based algos sync here)
        self._q: dict[tuple, dict[str, float]] = defaultdict(_init_q)

        # ── Double Q-Learning: second table ───────────────────────────────────
        self._q2: dict[tuple, dict[str, float]] = defaultdict(_init_q)

        # ── Policy-based (PPO / DPO / Actor-Critic): logit & value tables ────
        self._logits: dict[tuple, dict[str, float]] = defaultdict(_init_logits)
        self._V:      dict[tuple, float]            = defaultdict(float)
        # PPO: store policy snapshot before each update for ratio computation
        self._old_pi: dict[tuple, dict[str, float]] = {}
        # DPO temperature
        self._dpo_beta = 0.5

        self._update_count = 0
        self._improvements: list[PolicyImprovement] = []

        # Success / MTTR trackers (shared — used for Override mechanism)
        self._success_tracker: dict[tuple, dict[str, dict]] = defaultdict(
            lambda: {a: {"attempts": 0, "successes": 0} for a in ACTIONS}
        )
        self._mttr_tracker: dict[tuple, dict[str, dict]] = defaultdict(
            lambda: {a: {"steps_sum": 0, "resolved_count": 0} for a in ACTIONS}
        )

    # ── Action selection ──────────────────────────────────────────────────────
    def select_action(self, state_key: tuple) -> str:
        if self.algorithm in ("ppo", "dpo", "actor-critic"):
            # Sample from learned policy (ε-greedy over softmax)
            if random.random() < self.epsilon:
                return random.choice(ACTIONS)
            probs = _softmax(self._logits[state_key])
            return max(probs, key=probs.__getitem__)
        if random.random() < self.epsilon:
            return random.choice(ACTIONS)
        return self.best_action(state_key)

    def best_action(self, state_key: tuple) -> str:
        """Greedy best action — used by the Override mechanism."""
        if self.algorithm in ("ppo", "dpo", "actor-critic"):
            return max(self._logits[state_key], key=self._logits[state_key].__getitem__)
        if self.algorithm == "double-q":
            combined = {
                a: (self._q[state_key][a] + self._q2[state_key][a]) / 2
                for a in ACTIONS
            }
            return max(combined, key=combined.__getitem__)
        return max(self._q[state_key], key=self._q[state_key].__getitem__)

    # ── Master update dispatcher ──────────────────────────────────────────────
    def update(
        self,
        state_key: tuple,
        action: str,
        reward: float,
        next_state_key: tuple,
        resolved: bool = False,
        resolution_steps: int = 1,
    ) -> float:
        dispatch = {
            "q-learning":     self._upd_q_learning,
            "double-q":       self._upd_double_q,
            "expected-sarsa": self._upd_expected_sarsa,
            "ppo":            self._upd_ppo,
            "dpo":            self._upd_dpo,
            "actor-critic":   self._upd_actor_critic,
        }
        td_error = dispatch[self.algorithm](
            state_key, action, reward, next_state_key, resolved
        )

        # Success / MTTR tracking (shared across all algorithms)
        self._success_tracker[state_key][action]["attempts"] += 1
        if resolved:
            self._success_tracker[state_key][action]["successes"] += 1
        self._mttr_tracker[state_key][action]["steps_sum"] += resolution_steps
        if resolved:
            self._mttr_tracker[state_key][action]["resolved_count"] += 1

        self._update_count += 1
        self._decay_epsilon()
        return td_error

    # ── Algorithm implementations ─────────────────────────────────────────────

    def _upd_q_learning(self, s, a, r, s2, _resolved) -> float:
        """Q(s,a) ← Q + α[r + γ·max Q(s′) − Q]"""
        old_q    = self._q[s][a]
        td_error = (r + self.gamma * max(self._q[s2].values())) - old_q
        self._q[s][a] = old_q + self.alpha * td_error
        return td_error

    def _upd_double_q(self, s, a, r, s2, _resolved) -> float:
        """
        Double Q-learning: alternate updates between Q1 and Q2.
        Eliminates overestimation bias — better for stochastic outcomes.
        """
        if random.random() < 0.5:
            best_a   = max(self._q[s2],  key=self._q[s2].__getitem__)
            target   = r + self.gamma * self._q2[s2][best_a]
            td_error = target - self._q[s][a]
            self._q[s][a] += self.alpha * td_error
        else:
            best_a   = max(self._q2[s2], key=self._q2[s2].__getitem__)
            target   = r + self.gamma * self._q[s2][best_a]
            td_error = target - self._q2[s][a]
            self._q2[s][a] += self.alpha * td_error
        return td_error

    def _upd_expected_sarsa(self, s, a, r, s2, _resolved) -> float:
        """
        Expected SARSA: (1−ε)·max Q(s′) + ε·mean Q(s′)
        Lower variance than Q-learning; more stable under stochastic rewards.
        """
        q2      = self._q[s2]
        max_q   = max(q2.values())
        mean_q  = sum(q2.values()) / N_ACTIONS
        expected = (1.0 - self.epsilon) * max_q + self.epsilon * mean_q
        old_q    = self._q[s][a]
        td_error = (r + self.gamma * expected) - old_q
        self._q[s][a] = old_q + self.alpha * td_error
        return td_error

    def _upd_ppo(self, s, a, r, s2, _resolved) -> float:
        """
        Proximal Policy Optimisation (tabular softmax policy + value function).
        Clip ratio prevents destructive large updates (clip_ε = 0.2).
        Policy gradient loss: L_clip = min(ρ·A, clip(ρ,1−ε,1+ε)·A)
        Value update: V(s) ← V(s) + α·(r + γ·V(s′) − V(s))
        """
        clip_eps = 0.20

        # Advantage estimate: A = r + γ·V(s′) − V(s)
        advantage = r + self.gamma * self._V[s2] - self._V[s]

        # Current policy
        curr_pi = _softmax(self._logits[s])
        curr_p  = curr_pi[a]

        # Old policy (uniform at first update for this state)
        old_p = self._old_pi.get(s, {}).get(a, 1.0 / N_ACTIONS)

        ratio = curr_p / (old_p + 1e-8)
        clipped = max(1 - clip_eps, min(1 + clip_eps, ratio))
        ppo_weight = min(ratio, clipped)   # effective gradient weight

        # Gradient of log π(a|s) w.r.t. logits (softmax jacobian)
        lr = self.alpha
        for act in ACTIONS:
            grad = (1.0 - curr_pi[act]) if act == a else (-curr_pi[act])
            self._logits[s][act] += lr * ppo_weight * advantage * grad

        # Update value function (TD)
        td_target = r + self.gamma * self._V[s2]
        td_error  = td_target - self._V[s]
        self._V[s] += self.alpha * td_error

        # Store current policy as "old" for next update
        self._old_pi[s] = curr_pi

        # Sync Q-table so Override mechanism still works
        for act in ACTIONS:
            self._q[s][act] = self._logits[s][act]

        return td_error

    def _upd_dpo(self, s, a, r, s2, resolved) -> float:
        """
        DPO adapted for sequential decisions.
        Reference policy π_ref = uniform (1/n_actions).
        Preferred  = actions that resolved the incident  (reward > 0, resolved)
        Rejected   = actions that failed                 (not resolved)
        Update: push logit of preferred action up relative to reference,
                push rejected action down.
        β controls the strength of the preference signal.
        """
        beta    = self._dpo_beta
        ref_log = math.log(1.0 / N_ACTIONS)   # log π_ref(a)
        curr_pi = _softmax(self._logits[s])
        curr_log = math.log(curr_pi[a] + 1e-10)

        if resolved and r > 0:
            # Preferred action: increase its logit
            log_ratio_w = curr_log - ref_log
            # Pick worst action in this state as the implicit rejected action
            rejected_a  = min(
                (x for x in ACTIONS if x != a),
                key=lambda x: self._logits[s][x],
            )
            rejected_pi  = curr_pi[rejected_a]
            log_ratio_l  = math.log(rejected_pi + 1e-10) - ref_log
            dpo_signal   = math.tanh(beta * (log_ratio_w - log_ratio_l))
            gradient     = 1.0 - dpo_signal    # push toward preferred

            lr = self.alpha * beta
            for act in ACTIONS:
                if act == a:
                    self._logits[s][act] += lr * gradient * (1 - curr_pi[act])
                else:
                    self._logits[s][act] -= lr * gradient * curr_pi[act]
        else:
            # Rejected action: decrease its logit, spread probability elsewhere
            log_ratio = curr_log - ref_log
            lr = self.alpha * beta
            for act in ACTIONS:
                if act == a:
                    self._logits[s][act] -= lr * curr_pi[act]
                else:
                    self._logits[s][act] += lr * curr_pi[act] / (N_ACTIONS - 1)

        # Sync Q-table for Override compatibility
        for act in ACTIONS:
            self._q[s][act] = self._logits[s][act]

        td_error = r - curr_pi[a]   # policy residual as proxy error
        return td_error

    def _upd_actor_critic(self, s, a, r, s2, _resolved) -> float:
        """
        Advantage Actor-Critic (A2C, tabular).
        Critic: V(s) ← V(s) + α·δ   where δ = r + γ·V(s′) − V(s)
        Actor:  logits updated via policy gradient weighted by advantage δ.
        More stable than pure policy gradient; no clipping needed.
        """
        # Critic: TD error (advantage)
        td_target = r + self.gamma * self._V[s2]
        delta     = td_target - self._V[s]
        self._V[s] += self.alpha * delta           # critic update

        # Actor: gradient ascent on log π(a|s) weighted by advantage
        curr_pi = _softmax(self._logits[s])
        lr_actor = self.alpha * 0.5                # separate actor learning rate
        for act in ACTIONS:
            grad = (1.0 - curr_pi[act]) if act == a else (-curr_pi[act])
            self._logits[s][act] += lr_actor * delta * grad

        # Sync Q-table for Override compatibility
        for act in ACTIONS:
            self._q[s][act] = self._logits[s][act] + self._V[s]

        return delta

    # ── Epsilon management ────────────────────────────────────────────────────
    def _decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def reset_epsilon(self, value: float = 0.20) -> None:
        self.epsilon = max(self.epsilon, value)

    # ── Evidence / override (shared) ──────────────────────────────────────────
    def success_evidence(self, state_key: tuple) -> list[dict]:
        rows = []
        for action, stats in self._success_tracker.get(state_key, {}).items():
            if stats["attempts"] > 0:
                mttr_data     = self._mttr_tracker.get(state_key, {}).get(action, {})
                resolved_cnt  = mttr_data.get("resolved_count", 0)
                steps_sum     = mttr_data.get("steps_sum", 0)
                avg_mttr      = round(steps_sum / resolved_cnt, 1) if resolved_cnt else None
                rows.append({
                    "action":    action,
                    "attempts":  stats["attempts"],
                    "successes": stats["successes"],
                    "rate":      round(stats["successes"] / stats["attempts"], 3),
                    "avg_mttr":  avg_mttr,
                })
        return sorted(rows, key=lambda x: (-x["rate"], -x["attempts"]))

    def is_conclusive(self, state_key: tuple) -> tuple[bool, str]:
        evidence = self.success_evidence(state_key)
        if not evidence:
            return False, ""
        best = evidence[0]
        if (best["attempts"] >= CONCLUSIVE_MIN_ATTEMPTS
                and best["rate"] >= CONCLUSIVE_MIN_RATE):
            return True, best["action"]
        return False, ""

    # ── Policy improvement (closed loop) ──────────────────────────────────────
    def compute_improvements(self, previous_policy: dict[tuple, str]) -> list[PolicyImprovement]:
        improvements = []
        for state_key, q_values in self._q.items():
            new_best = max(q_values, key=q_values.__getitem__)
            old_best = previous_policy.get(state_key)
            if old_best is not None and old_best != new_best:
                delta = q_values[new_best] - q_values.get(old_best, 0.0)
                imp = PolicyImprovement(
                    state_key=state_key,
                    old_best_action=old_best, new_best_action=new_best,
                    q_delta=round(delta, 4),
                    description=f"State {state_key}: '{new_best}' over '{old_best}' (ΔQ={delta:+.4f})",
                )
                improvements.append(imp)
                self._improvements.append(imp)
        return improvements

    def snapshot_policy(self) -> dict[tuple, str]:
        return {sk: max(qv, key=qv.__getitem__) for sk, qv in self._q.items()}

    # ── Checkpoint ────────────────────────────────────────────────────────────
    def save_checkpoint(self, tag: str = "latest") -> str:
        path = os.path.join(self._checkpoint_dir, f"qtable_{tag}.json")
        with open(path, "w") as f:
            json.dump({
                "q_table":      {str(k): v for k, v in self._q.items()},
                "epsilon":      self.epsilon,
                "update_count": self._update_count,
                "algorithm":    self.algorithm,
            }, f, indent=2)
        return path

    def load_checkpoint(self, tag: str = "latest") -> bool:
        path = os.path.join(self._checkpoint_dir, f"qtable_{tag}.json")
        if not os.path.exists(path):
            return False
        with open(path) as f:
            data = json.load(f)
        for k_str, v in data["q_table"].items():
            self._q[tuple(eval(k_str))] = v
        self.epsilon      = data.get("epsilon", self.epsilon)
        self._update_count = data.get("update_count", 0)
        return True

    def get_evidence(self, state_key) -> dict:
        """Return best-action evidence + trust score for a state key — used by LangFuse tracing."""
        key = tuple(state_key) if isinstance(state_key, list) else state_key
        ev  = self.success_evidence(key)
        if not ev:
            return {"attempts": 0, "successes": 0, "trust_score": 0.0}
        best = ev[0]
        n, n_plus = best["attempts"], best["successes"]
        try:
            from scipy.stats import beta as beta_dist
            trust = 1.0 - beta_dist.cdf(CONCLUSIVE_MIN_RATE, n_plus + 1, n - n_plus + 1)
        except Exception:
            trust = n_plus / max(n, 1)
        return {"attempts": n, "successes": n_plus, "trust_score": round(trust, 4)}

    def stats(self) -> dict[str, Any]:
        return {
            "known_states":       len(self._q),
            "update_count":       self._update_count,
            "epsilon":            round(self.epsilon, 4),
            "total_improvements": len(self._improvements),
            "algorithm":          self.algorithm,
        }
