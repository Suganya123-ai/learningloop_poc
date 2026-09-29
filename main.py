"""
Entry point for the ITOps Recursive Learning Loop POC.

Run:
    python main.py                              # pure Q-learning
    python main.py --llm                        # LLM decide() via Ollama (llama3.1:8b)
    python main.py --llm --llm-model mistral    # different model
    python main.py --episodes 500 --steps 30 --plot
"""
import argparse
import logging
import sys
import os

# ── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,           # set to DEBUG for full trace
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("learningloop.log", mode="w"),
    ],
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ITOps Recursive Learning Loop POC"
    )
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--steps", type=int, default=20,
                   help="Steps per episode")
    p.add_argument("--improve-every", type=int, default=10,
                   help="Apply RL improvements every N episodes")
    p.add_argument("--alpha", type=float, default=0.15,
                   help="Q-learning rate")
    p.add_argument("--gamma", type=float, default=0.90,
                   help="Discount factor")
    p.add_argument("--epsilon", type=float, default=1.0,
                   help="Initial exploration rate")
    p.add_argument("--plot", action="store_true",
                   help="Plot reward/resolution curves after run")
    p.add_argument("--log-dir", default="./telemetry_logs")
    p.add_argument("--ckpt-dir", default="./rl_checkpoints")
    # LLM options
    p.add_argument("--llm", action="store_true",
                   help="Enable Ollama LLM for agent decide() with Q-table fallback")
    p.add_argument("--llm-model", default="llama3.1:8b",
                   help="Ollama model tag (default: llama3.1:8b)")
    p.add_argument("--llm-url", default="http://localhost:11434",
                   help="Ollama base URL")
    p.add_argument("--llm-timeout", type=int, default=30,
                   help="LLM call timeout in seconds")
    p.add_argument("--llm-temperature", type=float, default=0.1,
                   help="LLM sampling temperature")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Import here so logging is configured first
    from harness.learning_loop import LearningLoopHarness
    from reinforcement.rl_system import RLSystem

    # ── Optional LLM setup ───────────────────────────────────────────
    llm_client = None
    if args.llm:
        from agent.llm_client import OllamaClient
        llm_client = OllamaClient(
            model=args.llm_model,
            base_url=args.llm_url,
            timeout_s=args.llm_timeout,
            temperature=args.llm_temperature,
        )
        available = llm_client.is_available()
        llm_status = f"OK  model={args.llm_model}" if available else "UNREACHABLE — will use Q-table fallback"
        print(f"\n  Ollama LLM   : {llm_status}")

    print("\n" + "=" * 65)
    print("  ITOps Recursive Learning Loop — POC")
    mode = f"LLM ({args.llm_model}) + Q-table fallback" if args.llm else "Q-table only"
    print(f"  Decision mode: {mode}")
    print("  Architecture: Agent → Telemetry → Feedback → RL → Agent")
    print("=" * 65)
    print(f"  Episodes     : {args.episodes}")
    print(f"  Steps/episode: {args.steps}")
    print(f"  Improve every: {args.improve_every} episodes")
    print(f"  α={args.alpha}  γ={args.gamma}  ε₀={args.epsilon}")
    print("=" * 65 + "\n")

    harness = LearningLoopHarness(
        episodes=args.episodes,
        steps_per_ep=args.steps,
        improve_every=args.improve_every,
        log_dir=args.log_dir,
        checkpoint_dir=args.ckpt_dir,
        verbose=True,
        llm=llm_client,
        on_improvement=lambda imps: print(
            f"    → {len(imps)} policy change(s) pushed to agent"
        ),
    )

    # Override RL hyper-parameters from CLI
    harness.rl.alpha = args.alpha
    harness.rl.gamma = args.gamma
    harness.rl.epsilon = args.epsilon

    results = harness.run()

    # Print LLM usage summary if enabled
    if args.llm:
        stats = harness.agent.get_stats()
        print(f"\n  LLM decisions  : {stats['llm_decisions']}")
        print(f"  Q-table fallbacks: {stats['qtable_decisions']}")
        print(f"  LLM hit rate   : {stats['llm_rate']:.1%}")

    if args.plot:
        _plot(harness, results)


def _plot(harness, results) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed — skipping plot.")
        return

    rewards = harness.episode_rewards()
    rates = harness.resolution_rates()
    episodes = list(range(1, len(rewards) + 1))

    # Smooth with moving average
    window = max(1, len(rewards) // 20)
    smooth = lambda xs: np.convolve(xs, np.ones(window) / window, mode="valid")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle("ITOps Recursive Learning Loop — Training Curves", fontsize=14)

    ax1.plot(episodes, rewards, alpha=0.3, color="#4C9BE8", label="Raw")
    ax1.plot(
        episodes[window - 1:],
        smooth(rewards),
        color="#1A5FA8",
        linewidth=2,
        label=f"Moving avg (w={window})",
    )
    ax1.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax1.set_ylabel("Episode Total Reward")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    ax2.plot(episodes, rates, alpha=0.3, color="#E8884C", label="Raw")
    ax2.plot(
        episodes[window - 1:],
        smooth(rates),
        color="#A84E1A",
        linewidth=2,
        label=f"Moving avg (w={window})",
    )
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Resolution Rate")
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.3)

    plot_path = "learning_curves.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    print(f"\nPlot saved to {os.path.abspath(plot_path)}")


if __name__ == "__main__":
    main()
