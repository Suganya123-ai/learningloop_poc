"""
ITOps Agent — Unified Dashboard
Combines live execution monitor, analytics dashboard, and system architecture.

Usage:
    streamlit run app.py --server.port 8502
"""
import json, os, re, time, signal, subprocess
from collections import defaultdict

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ITOps Agent Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
ALERT_NAMES = {
    0: "NONE", 1: "HIGH_CPU", 2: "MEMORY_LEAK", 3: "SERVICE_CRASH",
    4: "NETWORK_LATENCY", 5: "DISK_FULL", 6: "DEPENDENCY_FAILURE",
}
SRC_COLORS  = {"LLM": "#3b82f6", "Override": "#a78bfa", "Q-table": "#34d399"}
SRC_ICONS   = {"LLM": "🤖", "Override": "⭐", "Q-table": "📊"}
LOOP_COLORS = ["#6366f1", "#3b82f6", "#06b6d4", "#10b981", "#f59e0b",
               "#ef4444", "#ec4899", "#84cc16", "#14b8a6", "#f97316"]

LOG_PATH  = "/tmp/itops_live.log"
WORK_DIR  = "/home/suganya.selvaraj04/learningloop_agenticai/learningloop_poc"
PYTHON    = "/home/suganya.selvaraj04/.venv/bin/python"
TRACES    = os.path.join(WORK_DIR, "eval_telemetry/traces.jsonl")

# ── Regex patterns ────────────────────────────────────────────────────────────
STEP_PAT = re.compile(
    r'\[([★ ])([✓✗])([LOQ?])\]\s+S(\d+)\s+\|\s+(.+?)\s+\|\s+'
    r'Action:\s+(\S+)\s+.*?Optimal:\s+(\S+)\s+.*?R=([+\-][\d.]+)'
)
LOOP_PAT = re.compile(r'LOOP\s+(\d+)\s*/\s*(\d+)')
RES_PAT  = re.compile(r'Resolution rate\s*:\s*([\d.]+)')
ACC_PAT  = re.compile(r'Action accuracy\s*:\s*([\d.]+)')
REW_PAT  = re.compile(r'Avg reward\s*:\s*([+\-]?[\d.]+)')
MTTR_PAT = re.compile(r'Avg MTTR\s*:\s*([\d.]+)')
SRC_PAT  = re.compile(r'LLM=(\d+)\s+Override=(\d+)\s+Q-table=(\d+)')
OVER_PAT = re.compile(r'Override fires\s*:\s*(\d+)')
STATE_PAT= re.compile(r'Q-table states\s*:\s*(\d+)')

# ── Session state ─────────────────────────────────────────────────────────────
_LIVE_DEFAULTS = dict(
    running=False, pid=None, file_offset=0,
    steps=[], loop_metrics=[], current={},
    cur_loop=0, total_loops=5,
    llm_cum=0, override_cum=0, qtable_cum=0,
    qtable_states=0, done=False,
    report_generated=False, report_time='',
)
for _k, _v in _LIVE_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Live monitor helpers ───────────────────────────────────────────────────────
def _parse(line: str):
    m = STEP_PAT.search(line)
    if m:
        acc, res, src, sid, scen, act, opt, rwd = m.groups()
        return dict(
            type='step', loop=st.session_state.cur_loop,
            step=int(sid), scenario=scen.strip(),
            action=act.strip(), optimal=opt.strip(),
            reward=float(rwd), resolved=(res == '✓'),
            accurate=(acc == '★'),
            source={'L': 'LLM', 'O': 'Override', 'Q': 'Q-table'}.get(src, src),
        )
    if lm := LOOP_PAT.search(line):
        st.session_state.cur_loop    = int(lm.group(1))
        st.session_state.total_loops = int(lm.group(2))
    if rm := RES_PAT.search(line):  _upsert('res_rate',  float(rm.group(1)))
    if am := ACC_PAT.search(line):  _upsert('accuracy',  float(am.group(1)))
    if rew := REW_PAT.search(line): _upsert('reward',    float(rew.group(1)))
    if mt := MTTR_PAT.search(line): _upsert('mttr',      float(mt.group(1)))
    if sm := SRC_PAT.search(line):
        st.session_state.llm_cum      = int(sm.group(1))
        st.session_state.override_cum = int(sm.group(2))
        st.session_state.qtable_cum   = int(sm.group(3))
    if om := OVER_PAT.search(line): st.session_state.qtable_states = int(om.group(1))
    if qm := STATE_PAT.search(line): st.session_state.qtable_states = int(qm.group(1))
    return None

def _upsert(key, val):
    lm, loop = st.session_state.loop_metrics, st.session_state.cur_loop
    if lm and lm[-1].get('loop') == loop:
        lm[-1][key] = val
    else:
        lm.append({'loop': loop, key: val})

def _read_new():
    if not os.path.exists(LOG_PATH): return []
    new = []
    with open(LOG_PATH) as f:
        f.seek(st.session_state.file_offset)
        for line in f: new.append(line)
        st.session_state.file_offset = f.tell()
    return new

def _alive():
    pid = st.session_state.pid
    if not pid: return False
    try: os.kill(pid, 0); return True
    except (ProcessLookupError, PermissionError): return False

def _start(model, loops, alpha, gamma, temp, rl_algo):
    for k, v in _LIVE_DEFAULTS.items():
        st.session_state[k] = v if not isinstance(v, list) else []
    if os.path.exists(LOG_PATH): os.remove(LOG_PATH)
    cmd = [PYTHON, '-u', 'run_evaluation.py',
           '--model', model, '--loops', str(loops),
           '--alpha', str(alpha), '--gamma', str(gamma), '--temp', str(temp),
           '--rl-algo', rl_algo]
    proc = subprocess.Popen(
        cmd, stdout=open(LOG_PATH, 'w'), stderr=subprocess.STDOUT,
        cwd=WORK_DIR, env={**os.environ, 'PYTHONPATH': WORK_DIR},
    )
    st.session_state.pid         = proc.pid
    st.session_state.running     = True
    st.session_state.total_loops = loops
    st.session_state.active_algo = rl_algo

def _stop():
    if st.session_state.pid:
        try: os.kill(st.session_state.pid, signal.SIGTERM)
        except Exception: pass
    st.session_state.running = False
    st.session_state.pid     = None

# ── Analytics data loaders ────────────────────────────────────────────────────
@st.cache_data
def load_traces(path: str) -> pd.DataFrame:
    rows = []
    if not os.path.exists(path): return pd.DataFrame()
    with open(path) as f:
        for line in f:
            t = json.loads(line)
            sk = t.get("state_key", [0, 0, 0, "no_op"])
            rows.append({
                "loop":        t["episode"],
                "step":        t["step"],
                "scenario":    t["metadata"].get("scenario", ""),
                "alert_type":  ALERT_NAMES.get(sk[0] if isinstance(sk, list) else 0, "UNKNOWN"),
                "action":      t["action"],
                "optimal":     t["metadata"].get("optimal", ""),
                "resolved":    t["resolved"],
                "reward":      t["reward"],
                "steps_taken": t["resolution_steps"],
                "accurate":    t["action"] == t["metadata"].get("optimal", ""),
            })
    return pd.DataFrame(rows)

def per_loop_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for loop, g in df.groupby("loop"):
        resolved = g["resolved"].sum()
        accurate = g["accurate"].sum()
        n = len(g)
        mttr = g[g["resolved"]]["steps_taken"].mean() if resolved else 0
        first_attempt = ((g["resolved"]) & (g["steps_taken"] == 1)).sum()
        rows.append({
            "Loop": f"Loop {loop}", "loop_num": loop, "Scenarios": n,
            "Action Accuracy": round(accurate / n * 100, 1),
            "Resolution Rate": round(resolved / n * 100, 1),
            "Avg Reward":      round(g["reward"].mean(), 4),
            "Avg MTTR":        round(float(mttr), 2) if mttr else 0.0,
            "First-Attempt %": round(first_attempt / max(resolved, 1) * 100, 1),
        })
    return pd.DataFrame(rows).sort_values("loop_num")

def per_alert_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (loop, alert), g in df.groupby(["loop", "alert_type"]):
        rows.append({"loop_num": loop, "Loop": f"Loop {loop}",
                     "Alert": alert, "Accuracy %": round(g["accurate"].mean() * 100, 1)})
    return pd.DataFrame(rows)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
body, .stApp { background:#0f172a; color:#e2e8f0; }
[data-testid="stSidebar"] { background:#0f172a; border-right:1px solid #1e293b; }
[data-testid="stSidebar"] * { color:#e2e8f0 !important; }
.stSelectbox label,.stSlider label,.stNumberInput label { color:#94a3b8 !important; font-size:12px !important; }
.kpi { background:#1e293b; border:1px solid #334155; border-radius:10px;
       padding:16px 12px; text-align:center; }
.kpi-label { font-size:11px; color:#64748b; text-transform:uppercase;
             letter-spacing:.8px; margin-bottom:4px; }
.kpi-value { font-size:26px; font-weight:700; }
.kpi-sub   { font-size:11px; color:#64748b; margin-top:2px; }
.card { background:#1e293b; border:1px solid #334155; border-radius:10px; padding:18px 20px; margin-bottom:14px; }
.card h4 { font-size:13px; color:#94a3b8; margin-bottom:10px; font-weight:600; }
.badge { display:inline-block; padding:2px 10px; border-radius:10px; font-size:11px; font-weight:700; }
.pipe-row { display:flex; align-items:center; gap:0; flex-wrap:wrap; margin:12px 0; }
.pipe-node { border:2px solid #334155; border-radius:8px; padding:8px 12px;
             text-align:center; min-width:80px; font-size:11px; color:#94a3b8; }
.pipe-node.active          { border-color:#a78bfa; color:#e2e8f0; background:#2e1065; }
.pipe-node.active.llm      { border-color:#3b82f6; background:#1e3a5f; }
.pipe-node.active.qtable   { border-color:#34d399; background:#064e3b; }
.pipe-arrow { color:#334155; font-size:16px; padding:0 4px; }
.log-row { font-size:11px; padding:5px 8px; border-radius:5px; margin-bottom:3px;
           font-family:monospace; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.log-ok    { background:#14532d22; border-left:3px solid #22c55e; color:#86efac; }
.log-warn  { background:#78350f22; border-left:3px solid #f59e0b; color:#fcd34d; }
.log-err   { background:#450a0a22; border-left:3px solid #ef4444; color:#fca5a5; }
.sec-title { font-size:14px; font-weight:600; color:#93c5fd; margin:18px 0 10px;
             border-bottom:1px solid #1e40af; padding-bottom:6px; }
.status-dot { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px; }
.dot-run  { background:#22c55e; box-shadow:0 0 6px #22c55e; animation:pulse 1s infinite; }
.dot-idle { background:#334155; }
.dot-done { background:#3b82f6; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }
div[data-testid="metric-container"] { background:#1e293b; border:1px solid #334155; border-radius:10px; }
.stTabs [data-baseweb="tab-list"] { background:#0f172a; gap:4px; }
.stTabs [data-baseweb="tab"] { background:#1e293b; border-radius:8px 8px 0 0;
                                color:#94a3b8; border:1px solid #334155; padding:8px 20px; }
.stTabs [aria-selected="true"] { background:#1e3a5f !important; color:#60a5fa !important; border-color:#3b82f6 !important; }
</style>
""", unsafe_allow_html=True)

# ── Poll new output (runs before UI render so data is fresh) ──────────────────
if st.session_state.running:
    for line in _read_new():
        p = _parse(line)
        if p and p['type'] == 'step':
            st.session_state.steps.append(p)
            st.session_state.current = p
    if not _alive():
        st.session_state.running = False
        st.session_state.done    = True
        st.cache_data.clear()   # refresh analytics tab with new data

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ ITOps Agent")
    st.markdown("---")

    # Status
    if st.session_state.running:
        st.markdown('<span class="status-dot dot-run"></span>**Running…**', unsafe_allow_html=True)
        pct = st.session_state.cur_loop / max(st.session_state.total_loops, 1)
        st.progress(pct, text=f"Loop {st.session_state.cur_loop}/{st.session_state.total_loops}")
    elif st.session_state.done:
        st.markdown('<span class="status-dot dot-done"></span>**Completed**', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-dot dot-idle"></span>Idle — configure and start', unsafe_allow_html=True)

    st.markdown("#### Run Configuration")
    model = st.selectbox("LLM Model", [
        "qwen2.5:7b-instruct", "qwen2.5:3b", "adept-it:latest",
        "gemma4:e2b", "phi4-mini:latest",
    ], disabled=st.session_state.running)

    RL_ALGO_OPTIONS = {
        "q-learning":     "Q-Learning (baseline)",
        "double-q":       "Double Q-Learning ⭐",
        "expected-sarsa": "Expected SARSA",
        "ppo":            "PPO — Proximal Policy Optimisation",
        "dpo":            "DPO — Direct Preference Optimisation",
        "actor-critic":   "Actor-Critic (A2C)",
    }
    rl_algo_key = st.selectbox(
        "🧠 RL Algorithm",
        options=list(RL_ALGO_OPTIONS.keys()),
        format_func=lambda k: RL_ALGO_OPTIONS[k],
        disabled=st.session_state.running,
    )
    RL_ALGO_TIPS = {
        "q-learning":     "Off-policy TD. Fast, simple. Baseline for comparison.",
        "double-q":       "Two Q-tables eliminate overestimation bias. Best for stochastic outcomes. ⭐ Recommended.",
        "expected-sarsa": "Uses expected value under ε-greedy policy. Lower variance, more stable.",
        "ppo":            "Clipped policy gradient. Prevents destructive updates. Popular in deep RL.",
        "dpo":            "Preference optimisation. Treats resolved steps as preferred, failed as rejected.",
        "actor-critic":   "Policy gradient (actor) + value function (critic). Stable, sample-efficient.",
    }
    st.caption(RL_ALGO_TIPS[rl_algo_key])

    loops = st.number_input("Loops", 1, 20, 5, disabled=st.session_state.running)
    alpha = st.slider("α  Learning Rate",  0.05, 0.50, 0.15, 0.05, disabled=st.session_state.running)
    gamma = st.slider("γ  Discount Factor", 0.50, 0.99, 0.90, 0.01, disabled=st.session_state.running)
    temp  = st.slider("🌡 Temperature",      0.0,  1.0,  0.1, 0.05, disabled=st.session_state.running)

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("▶ Start", use_container_width=True,
                     disabled=st.session_state.running, type="primary"):
            _start(model, int(loops), alpha, gamma, temp, rl_algo_key)
            st.rerun()
    with c2:
        if st.button("■ Stop", use_container_width=True,
                     disabled=not st.session_state.running):
            _stop(); st.rerun()

    if st.button("🔄 Reload Analytics", use_container_width=True):
        st.cache_data.clear(); st.rerun()

    st.markdown("---")
    st.markdown("#### Live Counters")
    active_algo = st.session_state.get('active_algo', '—')
    st.markdown(f"🧠 Algorithm: `{active_algo}`")
    st.markdown(f"🤖 LLM: `{st.session_state.llm_cum}`")
    st.markdown(f"⭐ Override: `{st.session_state.override_cum}`")
    st.markdown(f"📊 Q-table: `{st.session_state.qtable_cum}`")
    st.markdown(f"🗂 Q-states: `{st.session_state.qtable_states}`")
    st.markdown(f"📝 Decisions: `{len(st.session_state.steps)}`")

# ── TABS ──────────────────────────────────────────────────────────────────────
tab_live, tab_analytics, tab_arch = st.tabs([
    "⚡ Live Monitor",
    "📊 Analytics",
    "🔁 Architecture",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE MONITOR
# ═══════════════════════════════════════════════════════════════════════════════
with tab_live:
    steps = st.session_state.steps
    cur   = st.session_state.current
    lm    = st.session_state.loop_metrics

    cur_steps = [s for s in steps if s['loop'] == st.session_state.cur_loop]
    n = max(len(cur_steps), 1)
    acc_now = round(sum(1 for s in cur_steps if s['accurate']) / n * 100, 1)
    res_now = round(sum(1 for s in cur_steps if s['resolved']) / n * 100, 1)
    rew_now = round(sum(s['reward'] for s in cur_steps) / n, 3) if cur_steps else 0.0

    # KPI cards
    k1, k2, k3, k4, k5 = st.columns(5)
    def _kpi(col, label, value, sub, color="#e2e8f0"):
        with col:
            st.markdown(
                f'<div class="kpi"><div class="kpi-label">{label}</div>'
                f'<div class="kpi-value" style="color:{color}">{value}</div>'
                f'<div class="kpi-sub">{sub}</div></div>', unsafe_allow_html=True)

    _kpi(k1, "Loop", f"{st.session_state.cur_loop}/{st.session_state.total_loops}", "current / total", "#60a5fa")
    _kpi(k2, "Step", f"{cur.get('step',0)}/50", "scenario in loop", "#94a3b8")
    _kpi(k3, "Accuracy",   f"{acc_now}%", "this loop so far",
         "#22c55e" if acc_now >= 70 else "#f59e0b" if acc_now >= 50 else "#ef4444")
    _kpi(k4, "Resolution", f"{res_now}%", "this loop so far",
         "#22c55e" if res_now >= 70 else "#f59e0b" if res_now >= 50 else "#ef4444")
    _kpi(k5, "Avg Reward", f"{rew_now:+.3f}", "this loop so far",
         "#f59e0b" if rew_now > 0 else "#ef4444")

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns([1, 1], gap="medium")

    # ── Left column: current decision + loop summaries ──────────────────────
    with left:
        st.markdown('<div class="sec-title">🎯 Current Decision</div>', unsafe_allow_html=True)
        if cur:
            src   = cur.get('source', '—')
            sc    = SRC_COLORS.get(src, '#94a3b8')
            si    = SRC_ICONS.get(src, '?')
            rc    = '#22c55e' if cur['resolved'] else '#ef4444'
            rt    = '✓ Resolved' if cur['resolved'] else '✗ Not Resolved'
            ab    = ('<span class="badge" style="background:#14532d;color:#4ade80">★ Optimal</span>'
                     if cur['accurate'] else
                     '<span class="badge" style="background:#450a0a;color:#f87171">✗ Suboptimal</span>')
            ov_c  = 'pipe-node active' if src == 'Override' else 'pipe-node'
            llm_c = 'pipe-node active llm' if src == 'LLM' else 'pipe-node'
            qt_c  = 'pipe-node active qtable' if src == 'Q-table' else 'pipe-node'
            st.markdown(f"""
<div class="card">
  <h4>Loop {cur['loop']} · S{cur['step']:02d} · {cur['scenario']}</h4>
  <div class="pipe-row">
    <div class="pipe-node">👁 Observe</div><div class="pipe-arrow">→</div>
    <div class="{ov_c}">⭐ Override?</div><div class="pipe-arrow">→</div>
    <div class="{llm_c}">🤖 LLM</div><div class="pipe-arrow">→</div>
    <div class="{qt_c}">📊 Q-table</div><div class="pipe-arrow">→</div>
    <div class="pipe-node">⚡ Act</div>
  </div>
  <table style="width:100%;font-size:12px;border-collapse:collapse;margin-top:8px">
    <tr><td style="color:#64748b;padding:4px 0;width:130px">Source</td>
        <td><span class="badge" style="background:{sc}22;color:{sc};border:1px solid {sc}">{si} {src}</span></td></tr>
    <tr><td style="color:#64748b;padding:4px 0">Action</td>
        <td><code style="color:#e2e8f0;background:#0f172a;padding:2px 8px;border-radius:4px">{cur['action']}</code> &nbsp;{ab}</td></tr>
    <tr><td style="color:#64748b;padding:4px 0">Optimal</td>
        <td><code style="color:#94a3b8;background:#0f172a;padding:2px 8px;border-radius:4px">{cur['optimal']}</code></td></tr>
    <tr><td style="color:#64748b;padding:4px 0">Outcome</td>
        <td><span style="color:{rc};font-weight:700">{rt}</span></td></tr>
    <tr><td style="color:#64748b;padding:4px 0">Reward</td>
        <td><span style="color:#f59e0b;font-weight:700;font-size:16px">{cur['reward']:+.4f}</span></td></tr>
  </table>
</div>""", unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="card" style="text-align:center;padding:40px;color:#334155">'
                '<div style="font-size:40px">⚡</div>'
                '<div style="margin-top:8px">Press ▶ Start to begin</div></div>',
                unsafe_allow_html=True)

        if lm:
            st.markdown('<div class="sec-title">📊 Loop Summaries</div>', unsafe_allow_html=True)
            df_lm = pd.DataFrame(lm).fillna(0).rename(columns={
                'loop': 'Loop', 'accuracy': 'Accuracy%', 'res_rate': 'Resolution%',
                'reward': 'Avg Reward', 'mttr': 'MTTR',
            })
            st.dataframe(
                df_lm.style.format({
                    'Accuracy%': '{:.1f}%', 'Resolution%': '{:.1f}%',
                    'Avg Reward': '{:+.4f}', 'MTTR': '{:.2f}',
                }),
                use_container_width=True, hide_index=True,
            )

    # ── Right column: live log + charts ─────────────────────────────────────
    with right:
        st.markdown('<div class="sec-title">📜 Decision Log</div>', unsafe_allow_html=True)
        recent = steps[-40:][::-1]
        if recent:
            html = ""
            for s in recent:
                ic = '★' if s['accurate'] else ' '
                ch = '✓' if s['resolved'] else '✗'
                sc_tag = {'LLM': 'L', 'Override': 'O', 'Q-table': 'Q'}.get(s['source'], '?')
                cls = ('log-ok' if s['accurate'] and s['resolved']
                       else 'log-warn' if s['resolved'] else 'log-err')
                html += (f'<div class="log-row {cls}">'
                         f'[{ic}{ch}{sc_tag}] L{s["loop"]}·S{s["step"]:02d} '
                         f'| {s["scenario"][:24]:<24} | {s["action"]:<14} | R={s["reward"]:+.3f}'
                         f'</div>')
            st.markdown(html, unsafe_allow_html=True)
        else:
            st.markdown(
                '<div style="color:#334155;text-align:center;padding:30px;'
                'border:1px dashed #334155;border-radius:8px">'
                'Decisions appear here as the agent runs</div>',
                unsafe_allow_html=True)

        if len(steps) >= 3:
            st.markdown('<div class="sec-title">📈 Reward — Live</div>', unsafe_allow_html=True)
            rewards   = [s['reward'] for s in steps]
            step_nums = list(range(1, len(rewards) + 1))
            bar_cols  = ['#22c55e' if s['accurate'] and s['resolved']
                         else '#f59e0b' if s['resolved'] else '#ef4444'
                         for s in steps]
            fig = go.Figure()
            fig.add_trace(go.Bar(x=step_nums, y=rewards, marker_color=bar_cols,
                                 marker_opacity=0.8, name='Reward',
                                 hovertemplate='Step %{x}<br>R=%{y:.4f}<extra></extra>'))
            if len(rewards) >= 5:
                window = 5
                avg = [sum(rewards[max(0,i-window):i+1])/min(i+1,window) for i in range(len(rewards))]
                fig.add_trace(go.Scatter(x=step_nums, y=avg, mode='lines',
                                         line=dict(color='#60a5fa', width=2), name='5-step avg'))
            fig.update_layout(
                paper_bgcolor='#1e293b', plot_bgcolor='#1e293b',
                font=dict(color='#94a3b8', size=10), height=200,
                margin=dict(t=10, r=10, b=30, l=40),
                xaxis=dict(gridcolor='#334155'), yaxis=dict(gridcolor='#334155', range=[-0.2, 1.1]),
                legend=dict(orientation='h', y=-0.3, bgcolor='rgba(0,0,0,0)'),
            )
            st.plotly_chart(fig, use_container_width=True)

        if len(lm) >= 1:
            st.markdown('<div class="sec-title">🎯 Accuracy per Loop</div>', unsafe_allow_html=True)
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=[m['loop'] for m in lm], y=[m.get('accuracy', 0) for m in lm],
                mode='lines+markers', line=dict(color='#3b82f6', width=2.5),
                marker=dict(size=8), name='Accuracy %'))
            fig2.add_trace(go.Scatter(
                x=[m['loop'] for m in lm], y=[m.get('res_rate', 0) for m in lm],
                mode='lines+markers', line=dict(color='#22c55e', width=2.5, dash='dot'),
                marker=dict(size=8), name='Resolution %'))
            fig2.update_layout(
                paper_bgcolor='#1e293b', plot_bgcolor='#1e293b',
                font=dict(color='#94a3b8', size=10), height=200,
                margin=dict(t=10, r=10, b=30, l=40),
                xaxis=dict(gridcolor='#334155', dtick=1),
                yaxis=dict(gridcolor='#334155', range=[0, 110]),
                legend=dict(orientation='h', y=-0.35, bgcolor='rgba(0,0,0,0)'),
            )
            st.plotly_chart(fig2, use_container_width=True)

    # Decision source split
    tot = st.session_state.llm_cum + st.session_state.override_cum + st.session_state.qtable_cum
    if tot > 0:
        st.markdown('<div class="sec-title">🔀 Decision Source Split (cumulative)</div>',
                    unsafe_allow_html=True)
        d1, d2, d3 = st.columns(3)
        td = max(tot, 1)
        with d1:
            p = round(st.session_state.llm_cum / td * 100, 1)
            st.markdown(f'<div class="kpi"><div class="kpi-label">🤖 LLM</div>'
                        f'<div class="kpi-value" style="color:#3b82f6">{p}%</div>'
                        f'<div class="kpi-sub">{st.session_state.llm_cum} decisions</div></div>',
                        unsafe_allow_html=True)
        with d2:
            p = round(st.session_state.override_cum / td * 100, 1)
            st.markdown(f'<div class="kpi"><div class="kpi-label">⭐ Q-table Override</div>'
                        f'<div class="kpi-value" style="color:#a78bfa">{p}%</div>'
                        f'<div class="kpi-sub">{st.session_state.override_cum} bypassed LLM</div></div>',
                        unsafe_allow_html=True)
        with d3:
            p = round(st.session_state.qtable_cum / td * 100, 1)
            st.markdown(f'<div class="kpi"><div class="kpi-label">📊 Q-table Fallback</div>'
                        f'<div class="kpi-value" style="color:#34d399">{p}%</div>'
                        f'<div class="kpi-sub">{st.session_state.qtable_cum} decisions</div></div>',
                        unsafe_allow_html=True)

    # Completion banner
    if st.session_state.done and steps:
        if not st.session_state.report_generated:
            try:
                subprocess.run([PYTHON, 'generate_report.py'], cwd=WORK_DIR,
                               timeout=60, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                st.session_state.report_generated = True
                st.session_state.report_time = time.strftime("%H:%M:%S")
            except Exception:
                pass
        rpt = st.session_state.report_time
        st.markdown(
            f'<div style="background:#14532d;border:1px solid #22c55e;border-radius:10px;'
            f'padding:16px 20px;margin-top:16px">'
            f'<span style="font-size:15px;font-weight:700;color:#4ade80">✅ Run complete — '
            f'{len(steps)} decisions · {st.session_state.cur_loop} loops · '
            f'Accuracy: {acc_now:.1f}%</span>'
            f'<div style="margin-top:10px">'
            f'Switch to the <strong>📊 Analytics</strong> tab to see full results.'
            f'</div>'
            + (f'<div style="color:#64748b;font-size:11px;margin-top:4px">'
               f'HTML report regenerated at {rpt}</div>' if rpt else '')
            + '</div>',
            unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ANALYTICS DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
with tab_analytics:
    df = load_traces(TRACES)

    if df.empty:
        st.warning("No telemetry data yet. Run an evaluation first (▶ Start in the sidebar).")
        st.stop()

    # Use only the latest run.
    # Each run resets the TelemetryCollector counter (r###### in run_id).
    # Find the last reset point and keep everything from there.
    import re as _re
    def _ctr(rid):
        m = _re.search(r'r(\d+)', str(rid))
        return int(m.group(1)) if m else 0
    if 'run_id' in df.columns:
        ctrs = df['run_id'].map(_ctr).values
        cut_idx = 0
        for i in range(1, len(ctrs)):
            if ctrs[i] < ctrs[i-1]:
                cut_idx = i
        df = df.iloc[cut_idx:].copy()
    # Re-number loops 1..N for clean display
    unique_loops = sorted(df['loop'].unique())
    ep_map = {old: new for new, old in enumerate(unique_loops, 1)}
    df['loop'] = df['loop'].map(ep_map)

    loop_df  = per_loop_metrics(df)
    alert_df = per_alert_accuracy(df)
    n_loops  = df["loop"].nunique()
    n_total  = len(df)

    l1 = loop_df[loop_df["loop_num"] == loop_df["loop_num"].min()].iloc[0]
    ln = loop_df[loop_df["loop_num"] == loop_df["loop_num"].max()].iloc[0]
    acc_delta = ln["Action Accuracy"] - l1["Action Accuracy"]
    res_delta = ln["Resolution Rate"] - l1["Resolution Rate"]

    # KPI row
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Loops", n_loops)
    m2.metric("Decisions", n_total)
    m3.metric("Accuracy L1", f"{l1['Action Accuracy']}%")
    m4.metric(f"Accuracy L{int(ln['loop_num'])}", f"{ln['Action Accuracy']}%",
              delta=f"{acc_delta:+.1f} pp")
    m5.metric("Peak Resolution", f"{loop_df['Resolution Rate'].max()}%")

    st.markdown("---")

    sub1, sub2, sub3, sub4 = st.tabs([
        "📈 Learning Curves", "🎯 Per-Alert", "📋 Metrics Table", "🔍 Scenario Detail"
    ])

    # ── Learning Curves ──────────────────────────────────────────────────────
    with sub1:
        ca, cb = st.columns(2)
        with ca:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=loop_df["Loop"], y=loop_df["Action Accuracy"],
                mode="lines+markers+text",
                text=[f"{v}%" for v in loop_df["Action Accuracy"]],
                textposition="top center",
                line=dict(color="#3b82f6", width=3), marker=dict(size=10),
                name="Action Accuracy"))
            fig.update_layout(title="Action Accuracy per Loop",
                              yaxis=dict(title="Accuracy %", range=[0, 110]),
                              xaxis_title="Loop", plot_bgcolor="white", height=360)
            st.plotly_chart(fig, use_container_width=True)

        with cb:
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                x=loop_df["Loop"], y=loop_df["Resolution Rate"],
                text=[f"{v}%" for v in loop_df["Resolution Rate"]],
                textposition="outside",
                marker_color=LOOP_COLORS[:n_loops]))
            fig2.update_layout(title="Resolution Rate per Loop",
                               yaxis=dict(title="Resolution %", range=[0, 115]),
                               xaxis_title="Loop", plot_bgcolor="white", height=360)
            st.plotly_chart(fig2, use_container_width=True)

        cc, cd = st.columns(2)
        with cc:
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=loop_df["Loop"], y=loop_df["Avg Reward"],
                mode="lines+markers+text",
                text=[f"{v:.3f}" for v in loop_df["Avg Reward"]],
                textposition="top center",
                line=dict(color="#10b981", width=3), marker=dict(size=10),
                fill="tozeroy", fillcolor="rgba(16,185,129,0.07)"))
            fig3.update_layout(title="Average Reward per Loop",
                               yaxis=dict(title="Reward", range=[0, 1.1]),
                               plot_bgcolor="white", height=320)
            st.plotly_chart(fig3, use_container_width=True)

        with cd:
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(
                x=loop_df["Loop"], y=loop_df["Avg MTTR"],
                mode="lines+markers+text",
                text=[f"{v:.2f}" for v in loop_df["Avg MTTR"]],
                textposition="top center",
                line=dict(color="#f59e0b", width=3), marker=dict(size=10)))
            fig4.update_layout(title="Avg MTTR — steps to resolve (lower = better)",
                               yaxis_title="Steps", plot_bgcolor="white", height=320)
            st.plotly_chart(fig4, use_container_width=True)

    # ── Per-Alert ────────────────────────────────────────────────────────────
    with sub2:
        pivot = alert_df.pivot(index="Alert", columns="Loop", values="Accuracy %").fillna(0)
        fig5 = px.imshow(pivot,
                         color_continuous_scale=["#fef2f2", "#fde68a", "#bbf7d0", "#166534"],
                         zmin=0, zmax=100, text_auto=".0f", aspect="auto",
                         title="Accuracy Heatmap — Alert Type × Loop")
        fig5.update_traces(texttemplate="%{text}%", textfont_size=13)
        fig5.update_layout(height=400)
        st.plotly_chart(fig5, use_container_width=True)

        alerts_ord = alert_df.groupby("Alert")["Accuracy %"].mean().sort_values().index.tolist()
        fig6 = px.bar(alert_df, x="Alert", y="Accuracy %", color="Loop",
                      barmode="group", category_orders={"Alert": alerts_ord},
                      color_discrete_sequence=LOOP_COLORS[:n_loops],
                      title="Per-Alert Accuracy by Loop", height=380)
        fig6.update_layout(plot_bgcolor="white", yaxis=dict(range=[0, 115]))
        st.plotly_chart(fig6, use_container_width=True)

        st.info("**Key finding:** SERVICE_CRASH improves from ~25% → 100% once the Q-table "
                "override accumulates conclusive evidence (≥5 attempts, ≥80% success). "
                "DEPENDENCY_FAILURE stays at 0% — the LLM never selects `rollback`.")

    # ── Metrics Table ─────────────────────────────────────────────────────────
    with sub3:
        disp = loop_df[["Loop","Scenarios","Action Accuracy","Resolution Rate",
                         "Avg Reward","Avg MTTR","First-Attempt %"]].copy()
        disp["Action Accuracy"] = disp["Action Accuracy"].map(lambda v: f"{v}%")
        disp["Resolution Rate"] = disp["Resolution Rate"].map(lambda v: f"{v}%")
        disp["First-Attempt %"] = disp["First-Attempt %"].map(lambda v: f"{v}%")
        disp["Avg Reward"]      = disp["Avg Reward"].map(lambda v: f"{v:+.4f}")
        disp["Avg MTTR"]        = disp["Avg MTTR"].map(lambda v: f"{v:.2f} steps")
        st.subheader("All Loops")
        st.dataframe(disp.set_index("Loop"), use_container_width=True, height=220)

        st.subheader("Δ vs Loop 1")
        baseline = loop_df[loop_df["loop_num"] == loop_df["loop_num"].min()].iloc[0]
        delta_rows = [{"Loop": r["Loop"],
                       "Accuracy Δ":   f"{r['Action Accuracy']-baseline['Action Accuracy']:+.1f} pp",
                       "Resolution Δ": f"{r['Resolution Rate']-baseline['Resolution Rate']:+.1f} pp",
                       "Reward Δ":     f"{r['Avg Reward']-baseline['Avg Reward']:+.4f}",
                       "MTTR Δ":       f"{r['Avg MTTR']-baseline['Avg MTTR']:+.2f} steps"}
                      for _, r in loop_df.iterrows()]
        st.dataframe(pd.DataFrame(delta_rows).set_index("Loop"), use_container_width=True, height=220)

        st.subheader("Action Distribution per Loop")
        ac = df.groupby(["loop","action"]).size().reset_index(name="count")
        ac["Loop"] = ac["loop"].map(lambda x: f"Loop {x}")
        fig7 = px.bar(ac, x="Loop", y="count", color="action", barmode="stack",
                      title="Actions taken per loop",
                      color_discrete_sequence=px.colors.qualitative.Set2, height=360)
        fig7.update_layout(plot_bgcolor="white")
        st.plotly_chart(fig7, use_container_width=True)

    # ── Scenario Detail ───────────────────────────────────────────────────────
    with sub4:
        sel = st.selectbox("Loop", sorted(df["loop"].unique()), format_func=lambda x: f"Loop {x}")
        ld = df[df["loop"] == sel].copy()
        ld["★"] = ld["accurate"].map(lambda v: "★" if v else " ")
        ld["✓"] = ld["resolved"].map(lambda v: "✓" if v else "✗")
        ld["Reward"] = ld["reward"].map(lambda v: f"{v:+.3f}")

        sa, sb, sc_ = st.columns(3)
        sa.metric("Accuracy",  f"{ld['accurate'].mean():.1%}")
        sb.metric("Resolution",f"{ld['resolved'].mean():.1%}")
        sc_.metric("Avg Reward",f"{ld['reward'].mean():+.4f}")

        disp2 = ld[["★","✓","scenario","alert_type","action","optimal","Reward"]].copy()
        disp2.columns = ["★","✓","Scenario","Alert","Action Taken","Optimal","Reward"]

        def _hl(row):
            if row["Action Taken"] == row["Optimal"] and row["✓"] == "✓":
                return ["background-color:#f0fdf4"]*len(row)
            elif row["Action Taken"] != row["Optimal"]:
                return ["background-color:#fef2f2"]*len(row)
            return ["background-color:#fffbeb"]*len(row)

        st.dataframe(disp2.style.apply(_hl, axis=1),
                     use_container_width=True, height=500)
        st.caption("🟢 correct + resolved  🔴 wrong action  🟡 correct action, unlucky outcome")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════════
with tab_arch:
    st.subheader("Recursive Learning Loop — How It Works")
    st.code("""
┌────────────────────────────────────────────────────────────────────┐
│                       AGENT ENVIRONMENT                            │
│                                                                    │
│   ┌─────────────────────────────┐                                  │
│   │        ITOps Agent          │──── operational ────►            │
│   │   observe · decide · act    │       telemetry                  │
│   │                             │                                  │
│   │  LLM: qwen2.5:7b-instruct   │   ┌──────────────────────────┐  │
│   │  Tier 1: Q-table Override   │   │  Collect logs + traces   │  │
│   │  Tier 2: LLM decision       │   │  (TelemetryCollector)    │  │
│   │  Tier 3: Q-table ε-greedy   │   └──────────┬───────────────┘  │
│   └─────────────────────────────┘              │ evaluate         │
│            ▲                                    ▼                  │
│            │ validated improvement changes      │                  │
│            │                        ┌──────────────────────────┐  │
│            │                        │   Feedback + metrics     │  │
│            │                        │  0.3×quality + 0.5×suc.  │  │
│            │                        │  + 0.2×efficiency        │  │
│            │                        └──────────┬───────────────┘  │
│            │                                   │ reward signal    │
│            │                                   ▼                  │
│            │                        ┌──────────────────────────┐  │
│            └────────────────────────│   Reinforcement System   │  │
│                                     │  Q-learning α=0.15 γ=0.9 │  │
│                                     │  Success tracker evidence │  │
│                                     └──────────────────────────┘  │
│                                                                    │
│                  ◄──── CONTINUOUS LEARNING LOOP ────►              │
└────────────────────────────────────────────────────────────────────┘
""", language="text")

    st.markdown("---")
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("### Components")
        st.markdown("""
| Component | Role |
|---|---|
| **ITOps Agent** | observe → decide (3-tier) → act |
| **ITEnvironment** | simulates 7 alert types, stochastic outcomes |
| **OllamaClient** | wraps qwen2.5:7b-instruct via Ollama REST API |
| **RLSystem** | tabular Q-learning + success-rate evidence tracker |
| **FeedbackEvaluator** | computes composite reward |
| **TelemetryCollector** | JSONL append log at `eval_telemetry/traces.jsonl` |
| **MetricsCollector** | per-loop accuracy, resolution, MTTR, source breakdown |
        """)

    with col_r:
        st.markdown("### Three-Tier Decision Pipeline")
        st.markdown("""
| Tier | Condition | Action |
|---|---|---|
| **1 — Override** | ≥5 attempts, ≥80% success | Use Q-table best action directly, skip LLM |
| **2 — LLM** | No conclusive evidence | Ask qwen2.5:7b-instruct with evidence block in prompt |
| **3 — Q-table ε-greedy** | LLM unavailable / invalid | ε-greedy Q-table selection |

**Q-learning formula:**
`Q(s,a) ← Q(s,a) + α[r + γ·max Q(s′,·) − Q(s,a)]`
α=0.15 · γ=0.90 · ε reset to 0.15 per loop

**State key:** `(alert_type[0-6], service_status[0-2], load_level[0-3], prev_action[str])`
= 336 possible states
        """)

    st.markdown("---")
    st.subheader("Evidence Block — What the LLM sees from Loop 2 onward")
    st.code("""
══ EVIDENCE FROM 8 PAST INCIDENTS WITH THIS ALERT TYPE ══
  restart_service  : 5/5 resolved (100%) ← BEST RESULT
  investigate      : 0/3 resolved (0%)
⚠ EVIDENCE-BACKED RECOMMENDATION: Use 'restart_service' (5/5 = 100%)
══════════════════════════════════════════════════════════
""", language="text")
    st.caption("Real data from Loop 1 outcomes injected into the system prompt — "
               "the LLM reads this and switches from `investigate` → `restart_service`.")

    st.markdown("---")
    st.markdown("### Frameworks & Libraries")
    st.markdown("""
| Category | Framework / Library | Version | Used For |
|---|---|---|---|
| **LLM Serving** | Ollama | local | Runs qwen2.5:7b-instruct locally |
| **LLM API** | `requests` (REST) | stdlib | HTTP calls to Ollama `/api/generate` |
| **RL / Math** | Pure Python `random`, `math` | stdlib | Tabular Q-table, ε-greedy |
| **Data** | `collections.defaultdict` | stdlib | Q-table storage, evidence tracking |
| **Simulation** | `dataclasses`, `enum` | stdlib | ITState, ITEnvironment |
| **Telemetry** | JSONL flat file | — | Append-only trace log |
| **Dashboard** | Streamlit | 1.x | This unified UI |
| **Charts** | Plotly | 5.x | Interactive charts |
| **Static charts** | Matplotlib | 3.x | `generate_report.py` base64 PNG |
| **Data frames** | Pandas | 2.x | Metrics aggregation |
    """)

# ── Auto-refresh while running ────────────────────────────────────────────────
if st.session_state.running:
    time.sleep(1.0)
    st.rerun()
