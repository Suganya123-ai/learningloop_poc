"""
Simulated IT environment — services, alerts, resource metrics.
The ITOps agent acts on this environment and receives outcomes.
"""
import random
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional


class ServiceStatus(IntEnum):
    RUNNING = 0
    DEGRADED = 1
    DOWN = 2


class LoadLevel(IntEnum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3


class AlertType(IntEnum):
    NONE = 0
    HIGH_CPU = 1
    MEMORY_LEAK = 2
    SERVICE_CRASH = 3
    NETWORK_LATENCY = 4
    DISK_FULL = 5
    DEPENDENCY_FAILURE = 6


# All actions the agent can take
ACTIONS = [
    "no_op",
    "restart_service",
    "scale_up",
    "scale_down",
    "rollback",
    "investigate",
    "clear_disk",
    "escalate",
]

# Correct action mapping per alert type (ground truth for reward calc)
OPTIMAL_ACTION = {
    AlertType.NONE:               "no_op",
    AlertType.HIGH_CPU:           "scale_up",
    AlertType.MEMORY_LEAK:        "restart_service",
    AlertType.SERVICE_CRASH:      "restart_service",
    AlertType.NETWORK_LATENCY:    "investigate",
    AlertType.DISK_FULL:          "clear_disk",
    AlertType.DEPENDENCY_FAILURE: "rollback",
}


@dataclass
class ITState:
    alert_type: AlertType
    service_status: ServiceStatus
    load_level: LoadLevel
    memory_usage: float        # 0.0 – 1.0
    cpu_usage: float           # 0.0 – 1.0
    previous_action: str = "no_op"
    episode: int = 0
    step: int = 0

    def as_key(self) -> tuple:
        return (
            int(self.alert_type),
            int(self.service_status),
            int(self.load_level),
            self.previous_action,
        )

    def describe(self) -> str:
        return (
            f"Alert={self.alert_type.name}, "
            f"Service={self.service_status.name}, "
            f"Load={self.load_level.name}, "
            f"CPU={self.cpu_usage:.0%}, "
            f"Mem={self.memory_usage:.0%}"
        )


@dataclass
class ActionOutcome:
    action: str
    resolved: bool
    resolution_steps: int       # lower = more efficient
    side_effects: list[str] = field(default_factory=list)
    next_state: Optional["ITState"] = None


class ITEnvironment:
    """Simulated IT environment with stochastic outcomes."""

    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)
        self._step = 0
        self._episode = 0
        self.state = self._random_state()

    # ------------------------------------------------------------------
    def reset(self) -> ITState:
        self._step = 0
        self._episode += 1
        self.state = self._random_state()
        self.state.episode = self._episode
        return self.state

    def step(self, action: str) -> ActionOutcome:
        """Apply action, compute outcome, transition to next state."""
        self._step += 1
        optimal = OPTIMAL_ACTION[self.state.alert_type]
        resolved, steps, side_effects = self._simulate(action, optimal)

        next_state = self._transition(action, resolved)
        next_state.previous_action = action
        next_state.episode = self._episode
        next_state.step = self._step
        self.state = next_state

        return ActionOutcome(
            action=action,
            resolved=resolved,
            resolution_steps=steps,
            side_effects=side_effects,
            next_state=next_state,
        )

    # ------------------------------------------------------------------
    def _simulate(
        self, action: str, optimal: str
    ) -> tuple[bool, int, list[str]]:
        side_effects: list[str] = []
        if action == optimal:
            resolved = self._rng.random() < 0.92
            steps = self._rng.randint(1, 3)
        elif action in ("investigate", "escalate"):
            resolved = self._rng.random() < 0.45
            steps = self._rng.randint(3, 6)
        elif action == "no_op":
            resolved = self._rng.random() < 0.05
            steps = 0
            if self.state.alert_type != AlertType.NONE:
                side_effects.append("issue_persisted")
        else:
            # Wrong but actionable — low chance of accidentally helping
            resolved = self._rng.random() < 0.20
            steps = self._rng.randint(4, 8)
            if action == "scale_down" and self.state.load_level >= LoadLevel.HIGH:
                side_effects.append("service_degraded_further")
        return resolved, steps, side_effects

    def _transition(self, action: str, resolved: bool) -> ITState:
        if resolved:
            alert = AlertType.NONE
            status = ServiceStatus.RUNNING
            load = LoadLevel(max(0, int(self.state.load_level) - 1))
        else:
            # Drift state toward worse without resolution
            alert = self._rng.choice(list(AlertType))
            status = ServiceStatus(
                min(2, int(self.state.service_status) + self._rng.randint(0, 1))
            )
            load = LoadLevel(
                min(3, int(self.state.load_level) + self._rng.randint(0, 1))
            )

        cpu = max(0.0, min(1.0, self._rng.gauss(0.4 + load * 0.15, 0.1)))
        mem = max(0.0, min(1.0, self._rng.gauss(0.5, 0.15)))
        return ITState(
            alert_type=alert,
            service_status=status,
            load_level=load,
            cpu_usage=cpu,
            memory_usage=mem,
        )

    def _random_state(self) -> ITState:
        alert = self._rng.choice(list(AlertType))
        load = self._rng.choice(list(LoadLevel))
        status = self._rng.choice(list(ServiceStatus))
        return ITState(
            alert_type=alert,
            service_status=status,
            load_level=load,
            cpu_usage=self._rng.uniform(0.1, 0.95),
            memory_usage=self._rng.uniform(0.2, 0.9),
        )
