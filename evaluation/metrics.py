"""
Performance metrics for ITOps agent evaluation.

Computes per-loop and cumulative metrics to track agent improvement
across recursive learning loop cycles.

Metrics:
  resolution_rate       — % of scenarios where the incident was resolved
  action_accuracy       — % of actions matching the optimal action
  avg_reward            — mean composite reward score
  avg_mttr              — mean steps-to-resolution (lower = faster response)
  first_attempt_rate    — % resolved on first action (best-case quality)
  side_effect_rate      — % actions that caused side effects (lower = better)
  escalation_rate       — % decisions that escalated instead of acting directly
  per_alert_accuracy    — action accuracy broken down by alert type
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class StepRecord:
    scenario_id: int
    scenario_name: str
    loop: int
    alert_type: str
    action_taken: str
    optimal_action: str
    resolved: bool
    resolution_steps: int
    side_effects: list[str]
    reward: float
    decision_source: str      # "llm" or "qtable"


@dataclass
class LoopMetrics:
    loop: int
    n_scenarios: int
    resolution_rate: float       # 0–1
    action_accuracy: float       # 0–1
    avg_reward: float
    avg_mttr: float              # mean steps when resolved; 0 if none resolved
    first_attempt_rate: float    # % resolved with resolution_steps == 1
    side_effect_rate: float      # % steps with any side effect
    escalation_rate: float       # % actions == "escalate"
    llm_rate: float              # % decisions made by LLM
    policy_improvements: int     # RL improvements applied this loop
    per_alert_accuracy: dict[str, float] = field(default_factory=dict)

    def delta(self, baseline: "LoopMetrics") -> dict[str, str]:
        """Return formatted delta strings vs a baseline loop."""
        def fmt(val: float, base: float, higher_better: bool = True) -> str:
            diff = val - base
            sign = "+" if diff >= 0 else ""
            arrow = "↑" if (diff > 0) == higher_better else ("↓" if diff != 0 else "→")
            return f"{sign}{diff:+.1%} {arrow}"

        return {
            "resolution_rate": fmt(self.resolution_rate, baseline.resolution_rate),
            "action_accuracy": fmt(self.action_accuracy, baseline.action_accuracy),
            "avg_reward":      fmt(self.avg_reward, baseline.avg_reward),
            "avg_mttr":        fmt(self.avg_mttr, baseline.avg_mttr, higher_better=False),
            "first_attempt":   fmt(self.first_attempt_rate, baseline.first_attempt_rate),
            "side_effects":    fmt(self.side_effect_rate, baseline.side_effect_rate, higher_better=False),
        }


class MetricsCollector:

    def __init__(self):
        self._records: list[StepRecord] = []
        self._loop_improvements: dict[int, int] = defaultdict(int)

    def record(self, rec: StepRecord) -> None:
        self._records.append(rec)

    def set_loop_improvements(self, loop: int, count: int) -> None:
        self._loop_improvements[loop] = count

    def compute_loop(self, loop: int) -> Optional[LoopMetrics]:
        recs = [r for r in self._records if r.loop == loop]
        if not recs:
            return None

        n = len(recs)
        resolved     = [r for r in recs if r.resolved]
        accurate     = [r for r in recs if r.action_taken == r.optimal_action]
        first_attempt= [r for r in resolved if r.resolution_steps == 1]
        with_fx      = [r for r in recs if r.side_effects]
        escalations  = [r for r in recs if r.action_taken == "escalate"]
        llm_decisions= [r for r in recs if r.decision_source == "llm"]

        # MTTR: average steps among resolved cases only
        mttr = (
            sum(r.resolution_steps for r in resolved) / len(resolved)
            if resolved else 0.0
        )

        # Per-alert accuracy
        per_alert: dict[str, list[bool]] = defaultdict(list)
        for r in recs:
            per_alert[r.alert_type].append(r.action_taken == r.optimal_action)
        per_alert_acc = {
            k: round(sum(v) / len(v), 3) for k, v in per_alert.items()
        }

        return LoopMetrics(
            loop=loop,
            n_scenarios=n,
            resolution_rate=round(len(resolved) / n, 3),
            action_accuracy=round(len(accurate) / n, 3),
            avg_reward=round(sum(r.reward for r in recs) / n, 4),
            avg_mttr=round(mttr, 2),
            first_attempt_rate=round(len(first_attempt) / max(len(resolved), 1), 3),
            side_effect_rate=round(len(with_fx) / n, 3),
            escalation_rate=round(len(escalations) / n, 3),
            llm_rate=round(len(llm_decisions) / n, 3),
            policy_improvements=self._loop_improvements.get(loop, 0),
            per_alert_accuracy=per_alert_acc,
        )

    def all_loops(self) -> list[LoopMetrics]:
        loops = sorted({r.loop for r in self._records})
        return [m for loop in loops if (m := self.compute_loop(loop))]

    def all_records(self) -> list[StepRecord]:
        return list(self._records)


# ── Pretty-print helpers ─────────────────────────────────────────────────────

def print_loop_table(loop_metrics: list[LoopMetrics]) -> None:
    """Print a side-by-side comparison table across all loops."""
    if not loop_metrics:
        return

    baseline = loop_metrics[0]
    col_w = 14

    header_row   = f"{'Metric':<28}" + "".join(f"{'Loop '+str(m.loop):>{col_w}}" for m in loop_metrics)
    divider      = "─" * (28 + col_w * len(loop_metrics))

    def row(label: str, vals: list[str]) -> str:
        return f"{label:<28}" + "".join(f"{v:>{col_w}}" for v in vals)

    def pct(v: float) -> str:
        return f"{v:.1%}"

    def num(v: float) -> str:
        return f"{v:.4f}"

    def steps(v: float) -> str:
        return f"{v:.2f}" if v else "—"

    print("\n" + "═" * len(divider))
    print("  AGENT PERFORMANCE — RECURSIVE LEARNING LOOP COMPARISON")
    print("═" * len(divider))
    print(header_row)
    print(divider)

    rows = [
        ("Scenarios / loop",    [str(m.n_scenarios)           for m in loop_metrics]),
        ("Resolution Rate  ↑",  [pct(m.resolution_rate)       for m in loop_metrics]),
        ("Action Accuracy  ↑",  [pct(m.action_accuracy)       for m in loop_metrics]),
        ("Avg Reward       ↑",  [num(m.avg_reward)            for m in loop_metrics]),
        ("Avg MTTR (steps) ↓",  [steps(m.avg_mttr)            for m in loop_metrics]),
        ("First-Attempt %  ↑",  [pct(m.first_attempt_rate)    for m in loop_metrics]),
        ("Side-Effect Rate ↓",  [pct(m.side_effect_rate)      for m in loop_metrics]),
        ("Escalation Rate  ↓",  [pct(m.escalation_rate)       for m in loop_metrics]),
        ("LLM Hit Rate",        [pct(m.llm_rate)              for m in loop_metrics]),
        ("Policy Improvements", [str(m.policy_improvements)   for m in loop_metrics]),
    ]

    for label, vals in rows:
        print(row(label, vals))

    # Delta rows vs Loop 1
    if len(loop_metrics) > 1:
        print(divider)
        print(row("  vs Loop 1", ["baseline"] + [""] * (len(loop_metrics) - 1)))
        delta_rows = [
            ("  Resolution Δ",   [m.delta(baseline)["resolution_rate"] for m in loop_metrics]),
            ("  Accuracy Δ",     [m.delta(baseline)["action_accuracy"] for m in loop_metrics]),
            ("  Reward Δ",       [m.delta(baseline)["avg_reward"]      for m in loop_metrics]),
            ("  MTTR Δ",         [m.delta(baseline)["avg_mttr"]        for m in loop_metrics]),
            ("  First-Attempt Δ",[m.delta(baseline)["first_attempt"]   for m in loop_metrics]),
        ]
        for label, vals in delta_rows:
            print(row(label, ["—"] + vals[1:]))

    print("═" * len(divider))


def print_per_alert_table(loop_metrics: list[LoopMetrics]) -> None:
    """Print per-alert-type action accuracy across loops."""
    all_alerts = sorted({
        alert
        for m in loop_metrics
        for alert in m.per_alert_accuracy
    })

    col_w = 12
    header = f"{'Alert Type':<26}" + "".join(f"{'L'+str(m.loop):>{col_w}}" for m in loop_metrics)
    divider = "─" * (26 + col_w * len(loop_metrics))

    print("\n  Action Accuracy by Alert Type")
    print(divider)
    print(header)
    print(divider)
    for alert in all_alerts:
        vals = [
            f"{m.per_alert_accuracy.get(alert, 0):.1%}"
            for m in loop_metrics
        ]
        print(f"  {alert:<24}" + "".join(f"{v:>{col_w}}" for v in vals))
    print(divider)
