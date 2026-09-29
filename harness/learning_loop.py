"""
Continuous Learning Loop Harness.

Orchestrates the full cycle from the architecture diagram:

  ITOps Agent (observe→decide→act)
      ↓ operational telemetry
  Collect logs + traces
      ↓ evaluate outcomes
  Feedback + metrics
      ↓ reward signal
  Reinforcement system
      ↓ validated improvement changes
  ITOps Agent  ← (loop repeats)
"""
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from agent.environment import ITEnvironment
from agent.itops_agent import ITOpsAgent
from feedback.evaluator import FeedbackEvaluator
from reinforcement.rl_system import PolicyImprovement, RLSystem
from telemetry.collector import TelemetryCollector

logger = logging.getLogger("itops.harness")


@dataclass
class EpisodeResult:
    episode: int
    steps: int
    total_reward: float
    resolved_count: int
    improvements_applied: list[PolicyImprovement] = field(default_factory=list)


class LearningLoopHarness:
    """
    Drives the continuous learning loop.

    Parameters
    ----------
    episodes        : total learning episodes
    steps_per_ep    : max steps per episode
    improve_every   : apply RL improvements every N episodes
    checkpoint_every: save Q-table checkpoint every N episodes
    verbose         : print live episode summaries
    llm             : optional OllamaClient; when provided the agent uses
                      LLM for decide() with Q-table as fallback
    on_improvement  : optional callback fired when improvements are applied
    """

    def __init__(
        self,
        *,
        episodes: int = 200,
        steps_per_ep: int = 20,
        improve_every: int = 10,
        checkpoint_every: int = 50,
        log_dir: str = "./telemetry_logs",
        checkpoint_dir: str = "./rl_checkpoints",
        verbose: bool = True,
        llm=None,
        on_improvement: Optional[Callable[[list[PolicyImprovement]], None]] = None,
    ):
        self.episodes = episodes
        self.steps_per_ep = steps_per_ep
        self.improve_every = improve_every
        self.checkpoint_every = checkpoint_every
        self.verbose = verbose
        self.on_improvement = on_improvement

        # Wire up all components
        self.env = ITEnvironment()
        self.rl = RLSystem(checkpoint_dir=checkpoint_dir)
        self.agent = ITOpsAgent(self.rl, llm=llm)
        self.telemetry = TelemetryCollector(log_dir=log_dir)
        self.evaluator = FeedbackEvaluator()

        self._episode_results: list[EpisodeResult] = []
        self._policy_snapshot: dict[tuple, str] = {}

    # ------------------------------------------------------------------
    def run(self) -> list[EpisodeResult]:
        logger.info(
            "=== Learning Loop START  episodes=%d steps_per_ep=%d ===",
            self.episodes,
            self.steps_per_ep,
        )
        start = time.time()

        for ep in range(1, self.episodes + 1):
            result = self._run_episode(ep)
            self._episode_results.append(result)

            # ── Apply RL improvements every N episodes ──────────────────
            if ep % self.improve_every == 0:
                improvements = self._apply_improvements()
                result.improvements_applied = improvements
                if self.on_improvement and improvements:
                    self.on_improvement(improvements)

            # ── Checkpoint Q-table ──────────────────────────────────────
            if ep % self.checkpoint_every == 0:
                self.rl.save_checkpoint(tag=f"ep{ep:04d}")

            if self.verbose:
                self._print_episode(result, ep)

        elapsed = time.time() - start
        self.rl.save_checkpoint(tag="final")
        self._print_summary(elapsed)
        return self._episode_results

    # ------------------------------------------------------------------
    def _run_episode(self, episode: int) -> EpisodeResult:
        state = self.env.reset()
        total_reward = 0.0
        resolved_count = 0

        for step in range(1, self.steps_per_ep + 1):
            state.step = step

            # ── OBSERVE ─────────────────────────────────────────────────
            observation = self.agent.observe(state)

            # ── DECIDE ──────────────────────────────────────────────────
            action = self.agent.decide(observation)

            # ── ACT ─────────────────────────────────────────────────────
            self.agent.act(action)

            # ── Environment step ────────────────────────────────────────
            outcome = self.env.step(action)

            # ── COLLECT telemetry ────────────────────────────────────────
            feedback = self.evaluator.evaluate(state, outcome)

            trace = self.telemetry.record(
                episode=episode,
                step=step,
                observation=observation,
                action=action,
                resolved=outcome.resolved,
                resolution_steps=outcome.resolution_steps,
                side_effects=outcome.side_effects,
                reward=feedback.reward,
                extra={"rationale": feedback.rationale},
            )

            # ── Q-table update (reward signal → RL system) ───────────────
            next_obs_key = outcome.next_state.as_key()
            self.rl.update(
                state_key=observation["state_key"],
                action=action,
                reward=feedback.reward,
                next_state_key=next_obs_key,
            )

            total_reward += feedback.reward
            if outcome.resolved:
                resolved_count += 1
            state = outcome.next_state

        return EpisodeResult(
            episode=episode,
            steps=self.steps_per_ep,
            total_reward=round(total_reward, 4),
            resolved_count=resolved_count,
        )

    # ------------------------------------------------------------------
    def _apply_improvements(self) -> list[PolicyImprovement]:
        """
        Identify policy shifts since last snapshot → validated improvement changes
        pushed back to the agent environment (diagram: 'validated improvement changes').
        """
        improvements = self.rl.compute_improvements(self._policy_snapshot)
        self._policy_snapshot = self.rl.snapshot_policy()
        if improvements:
            logger.info(
                "Applied %d validated policy improvements from RL system.",
                len(improvements),
            )
        return improvements

    # ------------------------------------------------------------------
    def _print_episode(self, result: EpisodeResult, ep: int) -> None:
        marker = "★" if result.improvements_applied else " "
        print(
            f"[{marker}] Ep {ep:4d}/{self.episodes} | "
            f"Reward={result.total_reward:+7.3f} | "
            f"Resolved={result.resolved_count:2d}/{result.steps} | "
            f"ε={self.rl.epsilon:.3f} | "
            f"States={self.rl.stats()['known_states']:4d} | "
            f"Improvements={len(result.improvements_applied):3d}"
        )

    def _print_summary(self, elapsed: float) -> None:
        tsum = self.telemetry.summary()
        fb = self.evaluator.aggregate(n=500)
        rl = self.rl.stats()
        print("\n" + "=" * 65)
        print("CONTINUOUS LEARNING LOOP — FINAL SUMMARY")
        print("=" * 65)
        print(f"  Episodes run       : {self.episodes}")
        print(f"  Total agent actions: {rl['update_count']}")
        print(f"  Known Q-states     : {rl['known_states']}")
        print(f"  Final epsilon      : {rl['epsilon']:.4f}")
        print(f"  Total improvements : {rl['total_improvements']}")
        print(f"  Resolution rate    : {tsum.get('resolution_rate', 0):.1%}")
        print(f"  Avg reward         : {tsum.get('avg_reward', 0):+.4f}")
        print(f"  Avg quality        : {fb.get('avg_quality', 0):.4f}")
        print(f"  Avg success        : {fb.get('avg_success', 0):.4f}")
        print(f"  Avg efficiency     : {fb.get('avg_efficiency', 0):.4f}")
        print(f"  Elapsed            : {elapsed:.1f}s")
        print("=" * 65)

    # ------------------------------------------------------------------
    def episode_rewards(self) -> list[float]:
        return [r.total_reward for r in self._episode_results]

    def resolution_rates(self) -> list[float]:
        return [
            round(r.resolved_count / r.steps, 3)
            for r in self._episode_results
        ]
