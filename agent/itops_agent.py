"""
ITOps Agent — observe, decide, act.

Decision priority (v2 — evidence-aware):
  1. Q-table override  — when evidence is conclusive (≥5 attempts, ≥80% success)
                         LLM is skipped; Q-table's greedy best action is used directly.
  2. LLM (Ollama)      — natural language reasoning with evidence block injected.
  3. Q-table fallback  — epsilon-greedy; used when LLM unavailable or returns invalid.

All paths feed the same Q-table update so every decision teaches the RL system.
"""
import logging
from typing import TYPE_CHECKING, Optional

from agent.environment import ACTIONS, ITState

if TYPE_CHECKING:
    from agent.llm_client import OllamaClient
    from reinforcement.rl_system import RLSystem

logger = logging.getLogger("itops.agent")


class ITOpsAgent:
    """
    Observe → Decide → Act loop for IT operations.

    Pass an OllamaClient to enable LLM-driven decisions.
    When evidence is conclusive the Q-table overrides the LLM automatically.
    """

    def __init__(
        self,
        rl_system: "RLSystem",
        llm: Optional["OllamaClient"] = None,
        name: str = "ITOps-Agent-v2",
    ):
        self.name = name
        self.rl = rl_system
        self.llm = llm
        self._action_history: list[str] = []
        self._total_actions       = 0
        self._llm_decisions       = 0
        self._qtable_decisions    = 0
        self._override_decisions  = 0   # conclusive-evidence overrides
        self._last_source: str    = "qtable"

    # ------------------------------------------------------------------
    # 1. OBSERVE
    # ------------------------------------------------------------------
    def observe(self, state: ITState) -> dict:
        obs = {
            "state_key":      state.as_key(),
            "description":    state.describe(),
            "alert":          state.alert_type.name,
            "service_status": state.service_status.name,
            "load":           state.load_level.name,
            "cpu_pct":        round(state.cpu_usage * 100, 1),
            "mem_pct":        round(state.memory_usage * 100, 1),
            "episode":        state.episode,
            "step":           state.step,
        }
        logger.debug("Observed: %s", obs["description"])
        return obs

    # ------------------------------------------------------------------
    # 2. DECIDE
    # ------------------------------------------------------------------
    def decide(self, observation: dict) -> str:
        import random as _random
        state_key = observation["state_key"]
        action = None

        # ── Priority 1: conclusive evidence → skip LLM, use Q-table greedy ──
        conclusive, best_action = self.rl.is_conclusive(state_key)
        if conclusive:
            action = best_action
            self._override_decisions += 1
            self._last_source = "override"
            logger.info(
                "OVERRIDE: conclusive evidence for %s → %s  (ep=%s step=%s)",
                state_key, action, observation["episode"], observation["step"],
            )

        # ── Bootstrap exploration: if a state has very few attempts across all
        #    actions, explore randomly with a small fixed probability (0.15).
        #    This ensures actions the LLM never tries (e.g. restart_service for
        #    MEMORY_LEAK) get sampled early enough to build evidence.
        #    Once evidence accumulates the override fires and exploration stops.
        if action is None:
            evidence = self.rl.success_evidence(state_key)
            total_attempts = sum(e["attempts"] for e in evidence)
            if total_attempts < 8 and _random.random() < 0.15:
                action = self.rl.select_action(state_key)
                self._qtable_decisions += 1
                self._last_source = "qtable"
                logger.debug("BOOTSTRAP explore: %s  total_attempts=%d", action, total_attempts)

        # ── Priority 2: LLM with evidence hint ───────────────────────────────
        if action is None and self.llm is not None:
            hint = self._build_qtable_hint(state_key)
            action = self.llm.decide(observation, qtable_hint=hint)
            if action:
                self._llm_decisions += 1
                self._last_source = "llm"

        # ── Priority 3: Q-table epsilon-greedy fallback ───────────────────────
        if action is None:
            action = self.rl.select_action(state_key)
            self._qtable_decisions += 1
            self._last_source = "qtable"

        logger.debug(
            "Decided: %s  source=%s  ep=%s  step=%s  ε=%.3f",
            action, self._last_source,
            observation["episode"], observation["step"],
            self.rl.epsilon,
        )
        return action

    # ------------------------------------------------------------------
    # 3. ACT
    # ------------------------------------------------------------------
    def act(self, action: str) -> str:
        self._action_history.append(action)
        self._total_actions += 1
        logger.info("Acting: %s  (total=%d)", action, self._total_actions)
        return action

    # ------------------------------------------------------------------
    def _build_qtable_hint(self, state_key: tuple) -> dict:
        """
        Build evidence-based hint: real success/failure counts + MTTR per action.
        Empty if no updates yet — LLM uses domain knowledge only on first loop.
        """
        if self.rl._update_count == 0:
            return {}
        evidence = self.rl.success_evidence(state_key)
        if not evidence:
            return {}
        return {
            "evidence":  evidence,
            "n_updates": self.rl._update_count,
        }

    # ------------------------------------------------------------------
    def get_stats(self) -> dict:
        from collections import Counter
        counts = Counter(self._action_history[-200:])
        stats = {
            "agent":              self.name,
            "total_actions":      self._total_actions,
            "llm_decisions":      self._llm_decisions,
            "override_decisions": self._override_decisions,
            "qtable_decisions":   self._qtable_decisions,
            "llm_rate": round(
                self._llm_decisions / max(1, self._total_actions), 3
            ),
            "override_rate": round(
                self._override_decisions / max(1, self._total_actions), 3
            ),
            "recent_action_dist": dict(counts),
            "epsilon": round(self.rl.epsilon, 4),
        }
        if self.llm:
            stats["llm_stats"] = self.llm.stats()
        return stats
