"""
Feedback Evaluator — quality • success • efficiency.

Translates raw action outcomes into a scalar reward signal that the
reinforcement system can learn from. Mirrors the 'Feedback + metrics'
box in the architecture diagram.
"""
import logging
from dataclasses import dataclass

from agent.environment import OPTIMAL_ACTION, AlertType, ActionOutcome, ITState

logger = logging.getLogger("itops.feedback")


@dataclass
class FeedbackReport:
    quality_score: float    # 0–1: was the action a good fit for the alert?
    success_score: float    # 0–1: did the action resolve the issue?
    efficiency_score: float # 0–1: how quickly / cheaply?
    reward: float           # composite scalar for RL
    rationale: str


class FeedbackEvaluator:
    """
    Computes a reward for every (state, action, outcome) triple.

    Reward decomposition:
        R = w_q * quality + w_s * success + w_e * efficiency - penalties
    """

    WEIGHTS = {"quality": 0.30, "success": 0.50, "efficiency": 0.20}
    SIDE_EFFECT_PENALTY = 0.15

    def __init__(self):
        self._history: list[FeedbackReport] = []

    def evaluate(
        self,
        state: ITState,
        outcome: ActionOutcome,
    ) -> FeedbackReport:
        quality = self._quality(state, outcome.action)
        success = self._success(outcome)
        efficiency = self._efficiency(outcome)

        raw = (
            self.WEIGHTS["quality"]    * quality
            + self.WEIGHTS["success"]  * success
            + self.WEIGHTS["efficiency"] * efficiency
        )
        penalty = len(outcome.side_effects) * self.SIDE_EFFECT_PENALTY
        reward = round(max(-1.0, min(1.0, raw - penalty)), 4)

        rationale = (
            f"quality={quality:.2f}, success={success:.2f}, "
            f"efficiency={efficiency:.2f}, side_effects={outcome.side_effects}, "
            f"penalty={penalty:.2f} → reward={reward:.4f}"
        )

        report = FeedbackReport(
            quality_score=quality,
            success_score=success,
            efficiency_score=efficiency,
            reward=reward,
            rationale=rationale,
        )
        self._history.append(report)
        logger.debug("Feedback: %s", rationale)
        return report

    # ------------------------------------------------------------------
    def _quality(self, state: ITState, action: str) -> float:
        optimal = OPTIMAL_ACTION.get(state.alert_type, "no_op")
        if action == optimal:
            return 1.0
        # Partial credit for diagnostic actions
        if action in ("investigate", "escalate"):
            return 0.4
        if state.alert_type == AlertType.NONE and action == "no_op":
            return 1.0
        return 0.0

    def _success(self, outcome: ActionOutcome) -> float:
        return 1.0 if outcome.resolved else 0.0

    def _efficiency(self, outcome: ActionOutcome) -> float:
        if not outcome.resolved:
            return 0.0
        # resolution_steps 1 → 1.0, steps 8 → ~0.14
        return round(1.0 / max(1, outcome.resolution_steps), 3)

    # ------------------------------------------------------------------
    def aggregate(self, n: int = 100) -> dict:
        recent = self._history[-n:]
        if not recent:
            return {}
        return {
            "n": len(recent),
            "avg_reward": round(sum(r.reward for r in recent) / len(recent), 4),
            "avg_quality": round(sum(r.quality_score for r in recent) / len(recent), 4),
            "avg_success": round(sum(r.success_score for r in recent) / len(recent), 4),
            "avg_efficiency": round(sum(r.efficiency_score for r in recent) / len(recent), 4),
        }
