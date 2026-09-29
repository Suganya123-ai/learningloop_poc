"""
Generate itops_results.html from eval_telemetry/traces.jsonl.

Reads the most-recent evaluation run, computes per-loop and per-alert metrics,
embeds matplotlib charts as base64 PNG, and writes a self-contained HTML file.

Usage:
    python generate_report.py                        # default paths
    python generate_report.py --traces my/traces.jsonl --out report.html
"""
import argparse
import base64
import io
import json
import os
import re
import shutil
from collections import defaultdict
from datetime import datetime


ALERT_NAMES = {
    0: "NONE", 1: "HIGH_CPU", 2: "MEMORY_LEAK", 3: "SERVICE_CRASH",
    4: "NETWORK_LATENCY", 5: "DISK_FULL", 6: "DEPENDENCY_FAILURE",
}
ALERT_COLORS = {
    "NONE": "#22c55e", "HIGH_CPU": "#f59e0b", "MEMORY_LEAK": "#a78bfa",
    "SERVICE_CRASH": "#ef4444", "NETWORK_LATENCY": "#38bdf8",
    "DISK_FULL": "#fb923c", "DEPENDENCY_FAILURE": "#f472b6",
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_traces(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def latest_run(rows: list[dict]) -> list[dict]:
    """
    Extract the most-recent run from a file that may contain many appended runs.

    Each run restarts the TelemetryCollector's internal counter (r######
    in run_id).  The last run starts at the index where the counter resets
    to a value lower than the previous record's counter.
    """
    if not rows:
        return []

    import re

    def _ctr(r):
        m = re.search(r'r(\d+)', r.get("run_id", "r0"))
        return int(m.group(1)) if m else 0

    cut = 0
    for i in range(1, len(rows)):
        if _ctr(rows[i]) < _ctr(rows[i - 1]):
            cut = i   # keep updating — we want the LAST reset
    return rows[cut:]


def compute_metrics(rows: list[dict]) -> dict:
    """Compute per-loop and per-alert metrics from a single run's records."""
    loops: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        loops[r["episode"]].append(r)

    loop_metrics = []
    alert_acc: dict[int, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))

    for loop_num in sorted(loops):
        recs = loops[loop_num]
        n = len(recs)
        accurate = sum(1 for r in recs if r["action"] == r["metadata"].get("optimal", ""))
        resolved = sum(1 for r in recs if r["resolved"])
        rewards  = [r["reward"] for r in recs]
        mttr_vals= [r["resolution_steps"] for r in recs if r["resolved"] and r["resolution_steps"] > 0]

        for r in recs:
            sk = r.get("state_key", [])
            alert_idx = sk[0] if sk else 0
            is_acc = r["action"] == r["metadata"].get("optimal", "")
            alert_acc[loop_num][ALERT_NAMES.get(alert_idx, str(alert_idx))].append(is_acc)

        per_alert = {
            a: round(sum(v) / len(v) * 100, 1)
            for a, v in alert_acc[loop_num].items() if v
        }

        loop_metrics.append({
            "loop": loop_num,
            "n": n,
            "accuracy": round(accurate / n * 100, 1) if n else 0,
            "resolution": round(resolved / n * 100, 1) if n else 0,
            "avg_reward": round(sum(rewards) / n, 4) if n else 0,
            "avg_mttr": round(sum(mttr_vals) / len(mttr_vals), 2) if mttr_vals else 0,
            "per_alert": per_alert,
        })

    return {"loops": loop_metrics, "total_records": len(rows)}


# ── Chart generation ──────────────────────────────────────────────────────────

def _to_base64(fig) -> str:
    import matplotlib
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def chart_accuracy_resolution(loop_metrics: list[dict]) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    loops  = [m["loop"] for m in loop_metrics]
    acc    = [m["accuracy"] for m in loop_metrics]
    res    = [m["resolution"] for m in loop_metrics]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    fig.patch.set_facecolor("#1e293b")
    ax.set_facecolor("#0f172a")

    ax.plot(loops, acc, "o-", color="#3b82f6", linewidth=2.5, markersize=7,
            label="Action Accuracy %")
    ax.plot(loops, res, "s--", color="#22c55e", linewidth=2.5, markersize=7,
            label="Resolution Rate %")

    for i, (x, y) in enumerate(zip(loops, acc)):
        ax.annotate(f"{y:.0f}%", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8, color="#93c5fd")
    for i, (x, y) in enumerate(zip(loops, res)):
        ax.annotate(f"{y:.0f}%", (x, y), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=8, color="#4ade80")

    ax.set_xlabel("Loop", color="#94a3b8")
    ax.set_ylabel("Percent (%)", color="#94a3b8")
    ax.set_title("Accuracy & Resolution Rate per Loop", color="#e2e8f0", pad=10)
    ax.tick_params(colors="#64748b")
    ax.set_xticks(loops)
    ax.set_ylim(0, 115)
    ax.grid(True, color="#1e293b", linestyle="--", alpha=0.5)
    ax.spines[:].set_color("#334155")
    legend = ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0",
                       fontsize=9)
    fig.tight_layout()
    b64 = _to_base64(fig)
    plt.close(fig)
    return b64


def chart_reward(loop_metrics: list[dict]) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    loops   = [m["loop"] for m in loop_metrics]
    rewards = [m["avg_reward"] for m in loop_metrics]
    colors  = ["#22c55e" if r > 0 else "#ef4444" for r in rewards]

    fig, ax = plt.subplots(figsize=(8, 3.2))
    fig.patch.set_facecolor("#1e293b")
    ax.set_facecolor("#0f172a")

    bars = ax.bar(loops, rewards, color=colors, alpha=0.85, edgecolor="#334155", width=0.6)
    for bar, val in zip(bars, rewards):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:+.3f}", ha="center", va="bottom", fontsize=8, color="#e2e8f0")

    ax.axhline(0, color="#475569", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Loop", color="#94a3b8")
    ax.set_ylabel("Avg Reward", color="#94a3b8")
    ax.set_title("Average Reward per Loop", color="#e2e8f0", pad=10)
    ax.tick_params(colors="#64748b")
    ax.set_xticks(loops)
    ax.grid(True, axis="y", color="#1e293b", linestyle="--", alpha=0.5)
    ax.spines[:].set_color("#334155")
    fig.tight_layout()
    b64 = _to_base64(fig)
    plt.close(fig)
    return b64


def chart_per_alert(loop_metrics: list[dict]) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    alert_order = ["HIGH_CPU", "MEMORY_LEAK", "SERVICE_CRASH",
                   "NETWORK_LATENCY", "DISK_FULL", "DEPENDENCY_FAILURE"]
    loops = [m["loop"] for m in loop_metrics]

    data = np.zeros((len(alert_order), len(loops)))
    for j, m in enumerate(loop_metrics):
        for i, alert in enumerate(alert_order):
            data[i, j] = m["per_alert"].get(alert, 0)

    fig, ax = plt.subplots(figsize=(9, 3.8))
    fig.patch.set_facecolor("#1e293b")
    ax.set_facecolor("#0f172a")

    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(loops)))
    ax.set_xticklabels([f"L{l}" for l in loops], color="#94a3b8", fontsize=9)
    ax.set_yticks(range(len(alert_order)))
    ax.set_yticklabels(alert_order, color="#e2e8f0", fontsize=9)
    ax.set_title("Per-Alert Action Accuracy % (green=100%, red=0%)",
                 color="#e2e8f0", pad=10)
    ax.tick_params(colors="#64748b")
    ax.spines[:].set_color("#334155")

    for i in range(len(alert_order)):
        for j in range(len(loops)):
            val = data[i, j]
            txt_color = "#0f172a" if 30 < val < 80 else "#ffffff"
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                    fontsize=8, color=txt_color, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.ax.tick_params(colors="#94a3b8", labelsize=8)
    cbar.outline.set_edgecolor("#334155")
    fig.tight_layout()
    b64 = _to_base64(fig)
    plt.close(fig)
    return b64


def chart_mttr(loop_metrics: list[dict]) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    loops = [m["loop"] for m in loop_metrics]
    mttr  = [m["avg_mttr"] for m in loop_metrics]

    fig, ax = plt.subplots(figsize=(8, 2.8))
    fig.patch.set_facecolor("#1e293b")
    ax.set_facecolor("#0f172a")

    ax.fill_between(loops, mttr, alpha=0.3, color="#f59e0b")
    ax.plot(loops, mttr, "o-", color="#f59e0b", linewidth=2.5, markersize=7)
    for x, y in zip(loops, mttr):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8, color="#fcd34d")

    ax.set_xlabel("Loop", color="#94a3b8")
    ax.set_ylabel("Steps", color="#94a3b8")
    ax.set_title("Avg MTTR (steps to resolve — lower is better)",
                 color="#e2e8f0", pad=10)
    ax.tick_params(colors="#64748b")
    ax.set_xticks(loops)
    ax.invert_yaxis()
    ax.grid(True, color="#1e293b", linestyle="--", alpha=0.5)
    ax.spines[:].set_color("#334155")
    fig.tight_layout()
    b64 = _to_base64(fig)
    plt.close(fig)
    return b64


# ── HTML builder ──────────────────────────────────────────────────────────────

def build_html(metrics: dict, charts: dict, args) -> str:
    loops     = metrics["loops"]
    total     = metrics["total_records"]
    n_loops   = len(loops)
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    first = loops[0]
    last  = loops[-1]
    acc_delta  = last["accuracy"] - first["accuracy"]
    res_delta  = last["resolution"] - first["resolution"]
    rew_delta  = last["avg_reward"] - first["avg_reward"]
    mttr_delta = last["avg_mttr"] - first["avg_mttr"]

    def arrow(d, higher_better=True):
        if abs(d) < 0.01:
            return "→"
        return ("▲" if d > 0 else "▼") if higher_better else ("▼" if d > 0 else "▲")

    def signed(d):
        return f"{d:+.1f}"

    # Per-loop table rows
    table_rows = ""
    for m in loops:
        table_rows += f"""
        <tr>
          <td>{m['loop']}</td>
          <td>{m['accuracy']:.1f}%</td>
          <td>{m['resolution']:.1f}%</td>
          <td>{m['avg_reward']:+.4f}</td>
          <td>{m['avg_mttr']:.2f}</td>
          <td>{m['n']}</td>
        </tr>"""

    # Per-alert table
    alert_order = ["HIGH_CPU", "MEMORY_LEAK", "SERVICE_CRASH",
                   "NETWORK_LATENCY", "DISK_FULL", "DEPENDENCY_FAILURE"]
    alert_header = "".join(f"<th>{a.replace('_','<br>')}</th>" for a in alert_order)
    alert_rows = ""
    for m in loops:
        cells = ""
        for a in alert_order:
            v = m["per_alert"].get(a)
            if v is None:
                cells += "<td>—</td>"
            else:
                color = "#22c55e" if v >= 80 else "#f59e0b" if v >= 50 else "#ef4444"
                cells += f'<td style="color:{color};font-weight:700">{v:.0f}%</td>'
        alert_rows += f"<tr><td>Loop {m['loop']}</td>{cells}</tr>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ITOps Agent — Evaluation Results</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #0f172a;
          color: #e2e8f0; line-height: 1.5; padding: 24px; }}
  .topbar {{ display:flex; justify-content:space-between; align-items:center;
             background:#1e293b; border:1px solid #334155; border-radius:10px;
             padding:12px 20px; margin-bottom:24px; }}
  .topbar a {{ color:#60a5fa; text-decoration:none; font-size:13px; font-weight:600; }}
  .topbar a:hover {{ color:#93c5fd; }}
  h1 {{ font-size:24px; font-weight:700; color:#60a5fa; margin-bottom:4px; }}
  h2 {{ font-size:16px; font-weight:600; color:#93c5fd; margin:28px 0 12px;
        border-bottom:1px solid #1e40af; padding-bottom:6px; }}
  .meta {{ font-size:12px; color:#64748b; }}
  .kpi-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr));
               gap:14px; margin:20px 0; }}
  .kpi {{ background:#1e293b; border:1px solid #334155; border-radius:10px;
          padding:16px; text-align:center; }}
  .kpi-label {{ font-size:11px; color:#64748b; text-transform:uppercase;
                letter-spacing:.8px; margin-bottom:6px; }}
  .kpi-value {{ font-size:26px; font-weight:700; }}
  .kpi-sub   {{ font-size:11px; color:#94a3b8; margin-top:3px; }}
  .chart-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; margin:20px 0; }}
  .chart-box {{ background:#1e293b; border:1px solid #334155; border-radius:10px;
                padding:16px; }}
  .chart-box img {{ width:100%; border-radius:6px; }}
  .chart-box.full {{ grid-column:1/-1; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  th {{ background:#1e3a5f; color:#93c5fd; padding:8px 12px;
        text-align:left; font-weight:600; font-size:11px;
        text-transform:uppercase; letter-spacing:.5px; }}
  td {{ padding:8px 12px; border-bottom:1px solid #1e293b; color:#e2e8f0; }}
  tr:hover td {{ background:#1e293b55; }}
  .tbl-wrap {{ background:#0f172a; border:1px solid #334155; border-radius:10px;
               overflow:hidden; margin:12px 0; }}
  .badge {{ display:inline-block; padding:2px 10px; border-radius:10px;
            font-size:11px; font-weight:700; }}
  footer {{ margin-top:40px; text-align:center; color:#334155; font-size:11px; }}
  @media(max-width:700px){{.chart-grid{{grid-template-columns:1fr}}}}
</style>
</head>
<body>

<div class="topbar">
  <div>
    <div style="font-size:18px;font-weight:700;color:#60a5fa">⚡ ITOps Agent — Evaluation Results</div>
    <div class="meta">Generated: {generated} &nbsp;|&nbsp; {total} decisions &nbsp;|&nbsp; {n_loops} loops</div>
  </div>
  <div style="display:flex;gap:16px;align-items:center">
    <a href="http://localhost:8503" target="_blank">⚡ Live Monitor →</a>
    <a href="http://localhost:8502" target="_blank">📊 Streamlit Dashboard →</a>
  </div>
</div>

<!-- KPI Cards -->
<h2>Summary — Loop {loops[0]['loop']} → Loop {loops[-1]['loop']}</h2>
<div class="kpi-grid">
  <div class="kpi">
    <div class="kpi-label">Action Accuracy</div>
    <div class="kpi-value" style="color:{'#22c55e' if last['accuracy']>=70 else '#f59e0b'}">{last['accuracy']:.1f}%</div>
    <div class="kpi-sub">{signed(acc_delta)}pp {arrow(acc_delta)} vs Loop 1</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Resolution Rate</div>
    <div class="kpi-value" style="color:{'#22c55e' if last['resolution']>=70 else '#f59e0b'}">{last['resolution']:.1f}%</div>
    <div class="kpi-sub">{signed(res_delta)}pp {arrow(res_delta)} vs Loop 1</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Avg Reward</div>
    <div class="kpi-value" style="color:#f59e0b">{last['avg_reward']:+.4f}</div>
    <div class="kpi-sub">{signed(rew_delta)} {arrow(rew_delta)} vs Loop 1</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Avg MTTR</div>
    <div class="kpi-value" style="color:#38bdf8">{last['avg_mttr']:.2f}</div>
    <div class="kpi-sub">{signed(mttr_delta)} steps {arrow(mttr_delta, higher_better=False)} vs Loop 1</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Loops Run</div>
    <div class="kpi-value" style="color:#94a3b8">{n_loops}</div>
    <div class="kpi-sub">{total} total decisions</div>
  </div>
</div>

<!-- Charts -->
<h2>Learning Curves</h2>
<div class="chart-grid">
  <div class="chart-box">
    <img src="data:image/png;base64,{charts['acc_res']}" alt="Accuracy & Resolution">
  </div>
  <div class="chart-box">
    <img src="data:image/png;base64,{charts['reward']}" alt="Reward per Loop">
  </div>
  <div class="chart-box full">
    <img src="data:image/png;base64,{charts['per_alert']}" alt="Per-Alert Accuracy Heatmap">
  </div>
  <div class="chart-box full">
    <img src="data:image/png;base64,{charts['mttr']}" alt="MTTR per Loop">
  </div>
</div>

<!-- Per-loop metrics table -->
<h2>Per-Loop Metrics</h2>
<div class="tbl-wrap">
  <table>
    <thead>
      <tr>
        <th>Loop</th><th>Accuracy</th><th>Resolution</th>
        <th>Avg Reward</th><th>MTTR</th><th>Scenarios</th>
      </tr>
    </thead>
    <tbody>{table_rows}</tbody>
  </table>
</div>

<!-- Per-alert accuracy table -->
<h2>Per-Alert Accuracy by Loop</h2>
<div class="tbl-wrap" style="overflow-x:auto">
  <table>
    <thead><tr><th>Loop</th>{alert_header}</tr></thead>
    <tbody>{alert_rows}</tbody>
  </table>
</div>

<footer>
  ITOps Agent Evaluation — generated {generated}<br>
  Data source: eval_telemetry/traces.jsonl &nbsp;|&nbsp;
  <a href="http://localhost:8503" style="color:#334155">⚡ Live Monitor</a>
</footer>
</body>
</html>"""


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--traces", default="./eval_telemetry/traces.jsonl")
    p.add_argument("--out",    default="./itops_results.html")
    p.add_argument("--static", default="./static/itops_results.html")
    p.add_argument("--all-runs", action="store_true",
                   help="Use all records in traces.jsonl (default: latest run only)")
    return p.parse_args()


def generate(traces_path: str, out_path: str, static_path: str = None,
             all_runs: bool = False) -> dict:
    rows = load_traces(traces_path)
    if not all_runs:
        rows = latest_run(rows)
    if not rows:
        raise ValueError("No records found in traces file")

    metrics = compute_metrics(rows)
    loops   = metrics["loops"]

    charts = {
        "acc_res":   chart_accuracy_resolution(loops),
        "reward":    chart_reward(loops),
        "per_alert": chart_per_alert(loops),
        "mttr":      chart_mttr(loops),
    }

    html = build_html(metrics, charts, None)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  Written: {out_path}  ({os.path.getsize(out_path)//1024} KB)")

    if static_path:
        os.makedirs(os.path.dirname(static_path), exist_ok=True)
        shutil.copy2(out_path, static_path)
        print(f"  Copied:  {static_path}")

    return metrics


if __name__ == "__main__":
    args = parse_args()
    print("Generating ITOps results report...")
    metrics = generate(
        traces_path=args.traces,
        out_path=args.out,
        static_path=args.static,
        all_runs=args.all_runs,
    )
    loops = metrics["loops"]
    print(f"  Loops processed : {len(loops)}")
    print(f"  Final accuracy  : {loops[-1]['accuracy']:.1f}%")
    print(f"  Final resolution: {loops[-1]['resolution']:.1f}%")
    print(f"  Report ready at : {args.out}")
