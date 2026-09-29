"""
Telemetry Collector — logs + traces from every agent run.

Captures operational telemetry per the architecture diagram and
persists it so the feedback evaluator can process it.
"""
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger("itops.telemetry")


@dataclass
class RunTrace:
    run_id: str
    episode: int
    step: int
    timestamp: float
    state_description: str
    state_key: tuple
    action: str
    resolved: bool
    resolution_steps: int
    side_effects: list[str]
    reward: float = 0.0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["state_key"] = list(d["state_key"])  # JSON-serialisable
        return d


class TelemetryCollector:
    """
    Collects, stores, and exposes telemetry traces for each agent run.
    Writes JSON-lines to a log file for audit + replay.
    """

    def __init__(self, log_dir: str = "./telemetry_logs"):
        os.makedirs(log_dir, exist_ok=True)
        self._log_path = os.path.join(log_dir, "traces.jsonl")
        self._traces: list[RunTrace] = []
        self._run_counter = 0
        logger.info("Telemetry logging to %s", self._log_path)

    # ------------------------------------------------------------------
    def record(
        self,
        *,
        episode: int,
        step: int,
        observation: dict,
        action: str,
        resolved: bool,
        resolution_steps: int,
        side_effects: list[str],
        reward: float,
        extra: dict[str, Any] | None = None,
    ) -> RunTrace:
        self._run_counter += 1
        trace = RunTrace(
            run_id=f"ep{episode:04d}-st{step:04d}-r{self._run_counter:06d}",
            episode=episode,
            step=step,
            timestamp=time.time(),
            state_description=observation.get("description", ""),
            state_key=observation.get("state_key", ()),
            action=action,
            resolved=resolved,
            resolution_steps=resolution_steps,
            side_effects=side_effects,
            reward=round(reward, 4),
            metadata=extra or {},
        )
        self._traces.append(trace)
        self._flush(trace)
        logger.debug(
            "Telemetry recorded run_id=%s action=%s resolved=%s reward=%.2f",
            trace.run_id,
            action,
            resolved,
            reward,
        )
        return trace

    def recent(self, n: int = 50) -> list[RunTrace]:
        return self._traces[-n:]

    def all_traces(self) -> list[RunTrace]:
        return list(self._traces)

    def summary(self) -> dict:
        if not self._traces:
            return {"total_runs": 0}
        rewards = [t.reward for t in self._traces]
        resolved = [t for t in self._traces if t.resolved]
        return {
            "total_runs": len(self._traces),
            "resolved_count": len(resolved),
            "resolution_rate": round(len(resolved) / len(self._traces), 3),
            "avg_reward": round(sum(rewards) / len(rewards), 3),
            "max_reward": round(max(rewards), 3),
            "min_reward": round(min(rewards), 3),
        }

    def _flush(self, trace: RunTrace) -> None:
        try:
            with open(self._log_path, "a") as f:
                f.write(json.dumps(trace.to_dict()) + "\n")
        except OSError as exc:
            logger.warning("Telemetry flush failed: %s", exc)
