"""
ITOps Agent — 50-Scenario Recursive Learning Loop Evaluation  (v2).

Improvements over v1:
  - Q-table override: when evidence is conclusive (≥5 attempts, ≥80% success)
    the LLM is bypassed; the Q-table's greedy best action is used directly.
  - MTTR tracked and shown in LLM evidence block.
  - Conclusive evidence injected into system prompt (highest LLM authority).
  - Mandatory prohibition language replaces soft recommendation.
  - Epsilon partially reset between loops to maintain exploration.
  - resolution_steps passed to rl.update() for MTTR tracker.

Usage:
    python run_evaluation.py                           # uses adept-it:latest
    python run_evaluation.py --model qwen2.5:7b-instruct
    python run_evaluation.py --loops 5 --model adept-it:latest
"""
import argparse
import logging
import sys
import time

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("evaluation.log", mode="w"),
    ],
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ITOps 50-Scenario Evaluation")
    p.add_argument("--model",   default="qwen2.5:7b-instruct")
    p.add_argument("--url",     default="http://localhost:11434")
    p.add_argument("--timeout", type=int,   default=60)
    p.add_argument("--temp",    type=float, default=0.1)
    p.add_argument("--loops",   type=int,   default=5,
                   help="Number of times to run all 50 scenarios")
    p.add_argument("--alpha",   type=float, default=0.15)
    p.add_argument("--gamma",   type=float, default=0.90)
    p.add_argument("--rl-algo", default="q-learning",
                   choices=["q-learning", "double-q", "expected-sarsa",
                            "ppo", "dpo", "actor-critic"],
                   help="RL update algorithm")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from agent.environment import ITEnvironment
    from agent.itops_agent import ITOpsAgent
    from agent.llm_client import OllamaClient
    from feedback.evaluator import FeedbackEvaluator
    from reinforcement.rl_system import RLSystem
    from telemetry.collector import TelemetryCollector
    from telemetry.langfuse_client import trace_ceo_override, flush as lf_flush
    from scenarios.itops_scenarios import SCENARIOS
    from evaluation.metrics import MetricsCollector, StepRecord, print_loop_table, print_per_alert_table

    # ── Component setup ──────────────────────────────────────────────────────
    llm = OllamaClient(
        model=args.model,
        base_url=args.url,
        timeout_s=args.timeout,
        temperature=args.temp,
    )
    rl        = RLSystem(alpha=args.alpha, gamma=args.gamma,
                         checkpoint_dir="./eval_checkpoints",
                         algorithm=args.rl_algo)
    agent     = ITOpsAgent(rl, llm=llm)
    env       = ITEnvironment(seed=42)
    evaluator = FeedbackEvaluator()
    telemetry = TelemetryCollector(log_dir="./eval_telemetry")
    metrics   = MetricsCollector()

    # ── Pre-flight check ─────────────────────────────────────────────────────
    if not llm.is_available():
        print("ERROR: Ollama not reachable. Start Ollama and retry.")
        sys.exit(1)

    scenarios_per_loop = len(SCENARIOS)   # 50
    total_steps        = args.loops * scenarios_per_loop

    print("\n" + "═" * 68)
    print("  ITOps Agent — Recursive Learning Loop Evaluation")
    print("═" * 68)
    print(f"  Model         : {args.model}")
    print(f"  Scenarios     : {scenarios_per_loop} per loop × {args.loops} loops = {total_steps} total")
    print(f"  RL params     : α={args.alpha}  γ={args.gamma}  algo={args.rl_algo}")
    print(f"  Decision mode : LLM ({args.model}) + Q-table recursive hint")
    print("═" * 68 + "\n")

    policy_snapshot: dict = {}
    step_counter   = 0
    start_time     = time.time()

    # ── Main evaluation loop ─────────────────────────────────────────────────
    for loop_num in range(1, args.loops + 1):
        print(f"\n{'─'*68}")
        print(f"  LOOP {loop_num} / {args.loops}  ({scenarios_per_loop} scenarios)")
        print(f"{'─'*68}")

        for idx, scenario in enumerate(SCENARIOS, start=1):
            step_counter += 1
            state = scenario.state
            state.episode = loop_num
            state.step    = idx

            # ── Observe ──────────────────────────────────────────────
            obs = agent.observe(state)
            obs["scenario_name"] = scenario.name
            obs["scenario_desc"] = scenario.description

            # ── Decide (LLM + Q-table hint) ──────────────────────────
            action = agent.decide(obs)
            agent.act(action)

            # ── Simulate outcome via environment ──────────────────────
            env.state = state
            outcome   = env.step(action)

            # ── Feedback ──────────────────────────────────────────────
            feedback = evaluator.evaluate(state, outcome)

            # ── Q-table update (resolved + resolution_steps for MTTR tracker) ──
            rl.update(
                state_key        = obs["state_key"],
                action           = action,
                reward           = feedback.reward,
                next_state_key   = outcome.next_state.as_key(),
                resolved         = outcome.resolved,
                resolution_steps = outcome.resolution_steps,
            )

            # ── LangFuse traces ───────────────────────────────────────
            src = agent._last_source
            if src == "llm":
                llm.record_outcome(
                    optimal_action = scenario.optimal_action,
                    resolved       = outcome.resolved,
                    reward         = feedback.reward,
                )
            elif src == "override":
                ev = rl.get_evidence(obs["state_key"])
                trace_ceo_override(
                    loop           = loop_num,
                    step           = idx,
                    scenario       = scenario.name,
                    alert_type     = state.alert_type.name,
                    state_key      = str(obs["state_key"]),
                    action         = action,
                    optimal_action = scenario.optimal_action,
                    trust_score    = ev.get("trust_score", 0.0),
                    n_attempts     = ev.get("attempts", 0),
                    n_success      = ev.get("successes", 0),
                    resolved       = outcome.resolved,
                    reward         = feedback.reward,
                )

            # ── Telemetry ─────────────────────────────────────────────
            telemetry.record(
                episode          = loop_num,
                step             = idx,
                observation      = obs,
                action           = action,
                resolved         = outcome.resolved,
                resolution_steps = outcome.resolution_steps,
                side_effects     = outcome.side_effects,
                reward           = feedback.reward,
                extra={"scenario": scenario.name, "optimal": scenario.optimal_action},
            )

            # ── Metrics record ────────────────────────────────────────
            metrics.record(StepRecord(
                scenario_id      = scenario.id,
                scenario_name    = scenario.name,
                loop             = loop_num,
                alert_type       = state.alert_type.name,
                action_taken     = action,
                optimal_action   = scenario.optimal_action,
                resolved         = outcome.resolved,
                resolution_steps = outcome.resolution_steps,
                side_effects     = outcome.side_effects,
                reward           = feedback.reward,
                decision_source  = agent._last_source,
            ))

            status  = "✓" if outcome.resolved else "✗"
            acc     = "★" if action == scenario.optimal_action else " "
            src_tag = {"llm": "L", "override": "O", "qtable": "Q"}.get(agent._last_source, "?")
            elapsed = time.time() - start_time
            print(
                f"  [{acc}{status}{src_tag}] S{scenario.id:02d} | {scenario.name[:28]:<28} | "
                f"Action: {action:<17} | Optimal: {scenario.optimal_action:<17} | "
                f"R={feedback.reward:+.3f} | {elapsed:.0f}s"
            )

        # ── End of loop: apply RL improvements ───────────────────────────────
        improvements    = rl.compute_improvements(policy_snapshot)
        policy_snapshot = rl.snapshot_policy()
        metrics.set_loop_improvements(loop_num, len(improvements))

        lm = metrics.compute_loop(loop_num)
        rl.save_checkpoint(tag=f"loop{loop_num:02d}")

        # Partial epsilon reset keeps exploration alive across loops
        rl.reset_epsilon(value=0.15)

        agent_stats = agent.get_stats()
        print(f"\n  Loop {loop_num} summary:")
        print(f"    Resolution rate : {lm.resolution_rate:.1%}")
        print(f"    Action accuracy : {lm.action_accuracy:.1%}")
        print(f"    Avg reward      : {lm.avg_reward:+.4f}")
        print(f"    Avg MTTR        : {lm.avg_mttr:.2f} steps")
        print(f"    Policy changes  : {len(improvements)} improvements")
        print(f"    Decision source : LLM={agent_stats['llm_decisions']}  "
              f"Override={agent_stats['override_decisions']}  "
              f"Q-table={agent_stats['qtable_decisions']}")

    # ── Final report ─────────────────────────────────────────────────────────
    total_elapsed = time.time() - start_time
    all_metrics   = metrics.all_loops()

    print_loop_table(all_metrics)
    print_per_alert_table(all_metrics)

    final_agent = agent.get_stats()
    print(f"\n  Total wall-clock : {total_elapsed:.1f}s")
    print(f"  Total LLM calls  : {llm.stats()['llm_calls']}")
    print(f"  Override fires   : {final_agent['override_decisions']}  (conclusive evidence bypassed LLM)")
    print(f"  Q-table states   : {rl.stats()['known_states']}")
    print(f"  Total Q-updates  : {rl.stats()['update_count']}")
    print(f"  Eval log         : ./eval_telemetry/traces.jsonl")
    print(f"  Checkpoints      : ./eval_checkpoints/")
    print(f"  LangFuse UI      : http://localhost:3000\n")
    lf_flush()   # ensure all traces are sent before process exits


if __name__ == "__main__":
    main()
