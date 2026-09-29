"""
ITOps Agent — Live Execution Monitor
Real-time view of the evaluation harness as it runs.

Usage:
    streamlit run live_monitor.py --server.port 8503
"""
import os, re, time, signal, subprocess
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ITOps Live Monitor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Regex patterns to parse run_evaluation.py stdout ─────────────────────────
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

LOG_PATH  = '/tmp/itops_live.log'
WORK_DIR  = '/home/suganya.selvaraj04/learningloop_agenticai/learningloop_poc'
PYTHON    = '/home/suganya.selvaraj04/.venv/bin/python'

ALERT_COLORS = {
    'NONE': '#22c55e', 'HIGH_CPU': '#f59e0b', 'MEMORY_LEAK': '#a78bfa',
    'SERVICE_CRASH': '#ef4444', 'NETWORK_LATENCY': '#38bdf8',
    'DISK_FULL': '#fb923c', 'DEPENDENCY_FAILURE': '#f472b6',
}
SRC_COLORS = {'LLM': '#3b82f6', 'Override': '#a78bfa', 'Q-table': '#34d399'}
SRC_ICONS  = {'LLM': '🤖', 'Override': '⭐', 'Q-table': '📊'}

ALERT_MAP = {
    '0': 'NONE', '1': 'HIGH_CPU', '2': 'MEMORY_LEAK', '3': 'SERVICE_CRASH',
    '4': 'NETWORK_LATENCY', '5': 'DISK_FULL', '6': 'DEPENDENCY_FAILURE',
}

# ── Session state init ────────────────────────────────────────────────────────
_DEFAULTS = dict(
    running=False, pid=None, file_offset=0,
    steps=[], loop_metrics=[], current={},
    cur_loop=0, total_loops=5,
    llm_cum=0, override_cum=0, qtable_cum=0,
    total_override=0, qtable_states=0, done=False,
)
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Helpers ───────────────────────────────────────────────────────────────────
def _parse(line: str):
    m = STEP_PAT.search(line)
    if m:
        acc, res, src, sid, scen, act, opt, rwd = m.groups()
        src_map = {'L': 'LLM', 'O': 'Override', 'Q': 'Q-table'}
        return dict(
            type='step', loop=st.session_state.cur_loop,
            step=int(sid), scenario=scen.strip(),
            action=act.strip(), optimal=opt.strip(),
            reward=float(rwd), resolved=(res == '✓'),
            accurate=(acc == '★'), source=src_map.get(src, src),
        )
    if lm := LOOP_PAT.search(line):
        st.session_state.cur_loop  = int(lm.group(1))
        st.session_state.total_loops = int(lm.group(2))
    if rm := RES_PAT.search(line):
        _upsert_loop_metric('res_rate', float(rm.group(1)))
    if am := ACC_PAT.search(line):
        _upsert_loop_metric('accuracy', float(am.group(1)))
    if rew := REW_PAT.search(line):
        _upsert_loop_metric('reward', float(rew.group(1)))
    if mt := MTTR_PAT.search(line):
        _upsert_loop_metric('mttr', float(mt.group(1)))
    if sm := SRC_PAT.search(line):
        st.session_state.llm_cum      = int(sm.group(1))
        st.session_state.override_cum = int(sm.group(2))
        st.session_state.qtable_cum   = int(sm.group(3))
    if om := OVER_PAT.search(line):
        st.session_state.total_override = int(om.group(1))
    if qm := STATE_PAT.search(line):
        st.session_state.qtable_states = int(qm.group(1))
    return None

def _upsert_loop_metric(key, val):
    lm = st.session_state.loop_metrics
    loop = st.session_state.cur_loop
    if lm and lm[-1].get('loop') == loop:
        lm[-1][key] = val
    else:
        lm.append({'loop': loop, key: val})

def _read_new():
    if not os.path.exists(LOG_PATH):
        return []
    new = []
    with open(LOG_PATH) as f:
        f.seek(st.session_state.file_offset)
        for line in f:
            new.append(line)
        st.session_state.file_offset = f.tell()
    return new

def _alive():
    pid = st.session_state.pid
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False

def _start(model, loops, alpha, gamma, temp):
    for k, v in _DEFAULTS.items():
        st.session_state[k] = v if not isinstance(v, list) else []
    st.session_state.report_generated = False
    st.session_state.report_time = ''
    if os.path.exists(LOG_PATH):
        os.remove(LOG_PATH)
    cmd = [
        PYTHON, '-u', 'run_evaluation.py',
        '--model', model, '--loops', str(loops),
        '--alpha', str(alpha), '--gamma', str(gamma), '--temp', str(temp),
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=open(LOG_PATH, 'w'), stderr=subprocess.STDOUT,
        cwd=WORK_DIR,
        env={**os.environ, 'PYTHONPATH': WORK_DIR},
    )
    st.session_state.pid     = proc.pid
    st.session_state.running = True
    st.session_state.total_loops = loops
    st.session_state.done    = False

def _stop():
    if st.session_state.pid:
        try:
            os.kill(st.session_state.pid, signal.SIGTERM)
        except Exception:
            pass
    st.session_state.running = False
    st.session_state.pid     = None

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
body, .stApp { background:#0f172a; color:#e2e8f0; }
[data-testid="stSidebar"] { background:#0f172a; border-right:1px solid #1e293b; }
[data-testid="stSidebar"] * { color:#e2e8f0 !important; }
.stSelectbox label, .stSlider label, .stNumberInput label { color:#94a3b8 !important; font-size:12px !important; }
.kpi-card { background:#1e293b; border:1px solid #334155; border-radius:10px;
            padding:16px 12px; text-align:center; }
.kpi-label { font-size:11px; color:#64748b; text-transform:uppercase;
             letter-spacing:.8px; margin-bottom:4px; }
.kpi-value { font-size:28px; font-weight:700; }
.kpi-sub   { font-size:11px; color:#64748b; margin-top:2px; }
.step-card { background:#1e293b; border:1px solid #334155; border-radius:10px;
             padding:18px 20px; margin-bottom:14px; }
.step-card h4 { font-size:13px; color:#94a3b8; margin-bottom:10px; font-weight:600; }
.badge { display:inline-block; padding:2px 10px; border-radius:10px;
         font-size:11px; font-weight:700; }
.pipe-row { display:flex; align-items:center; gap:0; flex-wrap:wrap; margin:12px 0; }
.pipe-node { border:2px solid #334155; border-radius:8px; padding:8px 12px;
             text-align:center; min-width:80px; font-size:11px; color:#94a3b8; }
.pipe-node.active { border-color:#a78bfa; color:#e2e8f0; background:#2e1065; }
.pipe-node.active.llm { border-color:#3b82f6; background:#1e3a5f; }
.pipe-node.active.qtable { border-color:#34d399; background:#064e3b; }
.pipe-arrow { color:#334155; font-size:16px; padding:0 4px; }
.log-row { font-size:11px; padding:5px 8px; border-radius:5px; margin-bottom:3px;
           font-family:monospace; white-space:nowrap; overflow:hidden;
           text-overflow:ellipsis; }
.log-hit-res  { background:#14532d22; border-left:3px solid #22c55e; color:#86efac; }
.log-hit-unres{ background:#78350f22; border-left:3px solid #f59e0b; color:#fcd34d; }
.log-miss     { background:#450a0a22; border-left:3px solid #ef4444; color:#fca5a5; }
.section-title { font-size:14px; font-weight:600; color:#93c5fd;
                 margin:18px 0 10px; border-bottom:1px solid #1e40af; padding-bottom:6px; }
div[data-testid="metric-container"] { background:#1e293b; border:1px solid #334155;
                                       border-radius:10px; padding:12px; }
.status-dot { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px; }
.dot-run { background:#22c55e; box-shadow:0 0 6px #22c55e; animation:pulse 1s infinite; }
.dot-idle { background:#334155; }
.dot-done { background:#3b82f6; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ ITOps Live Monitor")
    st.markdown("---")

    if st.session_state.running:
        st.markdown(
            '<span class="status-dot dot-run"></span>**Running...**',
            unsafe_allow_html=True,
        )
        loop_pct = (st.session_state.cur_loop / max(st.session_state.total_loops, 1))
        st.progress(loop_pct, text=f"Loop {st.session_state.cur_loop}/{st.session_state.total_loops}")
    elif st.session_state.done:
        st.markdown('<span class="status-dot dot-done"></span>**Completed**', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-dot dot-idle"></span>Idle', unsafe_allow_html=True)

    st.markdown("#### Configuration")
    model  = st.selectbox("LLM Model", [
        "qwen2.5:7b-instruct", "qwen2.5:3b", "adept-it:latest",
        "gemma4:e2b", "phi4-mini:latest",
    ], disabled=st.session_state.running)
    loops  = st.number_input("Loops", min_value=1, max_value=20,
                              value=5, disabled=st.session_state.running)
    alpha  = st.slider("α  Learning Rate", 0.05, 0.50, 0.15, 0.05,
                        disabled=st.session_state.running)
    gamma  = st.slider("γ  Discount Factor", 0.50, 0.99, 0.90, 0.01,
                        disabled=st.session_state.running)
    temp   = st.slider("🌡 LLM Temperature", 0.0, 1.0, 0.1, 0.05,
                        disabled=st.session_state.running)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶ Start", use_container_width=True,
                     disabled=st.session_state.running, type="primary"):
            _start(model, int(loops), alpha, gamma, temp)
            st.rerun()
    with col2:
        if st.button("■ Stop", use_container_width=True,
                     disabled=not st.session_state.running):
            _stop()
            st.rerun()

    st.markdown("---")
    st.markdown("#### Live Counters")
    total = max(len(st.session_state.steps), 1)
    st.markdown(f"🤖 **LLM** decisions: `{st.session_state.llm_cum}`")
    st.markdown(f"⭐ **Override** fires: `{st.session_state.override_cum}`")
    st.markdown(f"📊 **Q-table** fallback: `{st.session_state.qtable_cum}`")
    st.markdown(f"🗂 Q-table states: `{st.session_state.qtable_states}`")
    st.markdown(f"📝 Total decisions: `{len(st.session_state.steps)}`")

    st.markdown("---")
    st.caption("Results dashboard → port 8888")
    st.caption("Streamlit dashboard → port 8502")

# ── POLL & PROCESS new output lines ──────────────────────────────────────────
if st.session_state.running:
    new_lines = _read_new()
    for line in new_lines:
        parsed = _parse(line)
        if parsed and parsed['type'] == 'step':
            st.session_state.steps.append(parsed)
            st.session_state.current = parsed
    if not _alive():
        st.session_state.running = False
        st.session_state.done    = True

# ── KPI CARDS ─────────────────────────────────────────────────────────────────
steps = st.session_state.steps
cur   = st.session_state.current
lm    = st.session_state.loop_metrics

cur_loop_steps = [s for s in steps if s['loop'] == st.session_state.cur_loop]
n = max(len(cur_loop_steps), 1)
acc_now  = round(sum(1 for s in cur_loop_steps if s['accurate']) / n * 100, 1)
res_now  = round(sum(1 for s in cur_loop_steps if s['resolved']) / n * 100, 1)
rew_now  = round(sum(s['reward'] for s in cur_loop_steps) / n, 3) if cur_loop_steps else 0.0

st.markdown(
    '<p style="font-size:22px;font-weight:700;color:#60a5fa;margin-bottom:6px">'
    '⚡ ITOps Agent — Live Execution Monitor</p>',
    unsafe_allow_html=True,
)

k1, k2, k3, k4, k5 = st.columns(5)
def _kpi(col, label, value, sub, color="#e2e8f0"):
    with col:
        st.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-label">{label}</div>'
            f'<div class="kpi-value" style="color:{color}">{value}</div>'
            f'<div class="kpi-sub">{sub}</div></div>',
            unsafe_allow_html=True,
        )

loop_label = f"{st.session_state.cur_loop} / {st.session_state.total_loops}"
step_label = f"{cur.get('step', 0)} / 50"
_kpi(k1, "Loop", loop_label, "current / total", "#60a5fa")
_kpi(k2, "Step", step_label, "scenario in loop", "#94a3b8")
_kpi(k3, "Accuracy", f"{acc_now}%", "this loop so far",
     "#22c55e" if acc_now >= 70 else "#f59e0b" if acc_now >= 50 else "#ef4444")
_kpi(k4, "Resolution", f"{res_now}%", "this loop so far",
     "#22c55e" if res_now >= 70 else "#f59e0b" if res_now >= 50 else "#ef4444")
_kpi(k5, "Avg Reward", f"{rew_now:+.3f}", "this loop so far",
     "#f59e0b" if rew_now > 0 else "#ef4444")

st.markdown("<br>", unsafe_allow_html=True)

# ── MAIN TWO-COLUMN LAYOUT ────────────────────────────────────────────────────
left, right = st.columns([1, 1], gap="medium")

# ── LEFT: Current Decision Card ───────────────────────────────────────────────
with left:
    st.markdown('<div class="section-title">🎯 Current Decision</div>', unsafe_allow_html=True)

    if cur:
        alert_name = cur.get('action', '')   # we'll derive from scenario context
        action  = cur.get('action', '—')
        optimal = cur.get('optimal', '—')
        src     = cur.get('source', '—')
        resolved= cur.get('resolved', False)
        reward  = cur.get('reward', 0.0)
        scen    = cur.get('scenario', '—')
        accurate= cur.get('accurate', False)
        loop_n  = cur.get('loop', 0)
        step_n  = cur.get('step', 0)

        src_color = SRC_COLORS.get(src, '#94a3b8')
        src_icon  = SRC_ICONS.get(src, '?')
        res_color = '#22c55e' if resolved else '#ef4444'
        res_text  = '✓ Resolved' if resolved else '✗ Not Resolved'
        acc_badge = f'<span class="badge" style="background:#14532d;color:#4ade80">★ Optimal</span>' if accurate else \
                    f'<span class="badge" style="background:#450a0a;color:#f87171">✗ Suboptimal</span>'

        # Pipeline visualizer
        is_override = src == 'Override'
        is_llm      = src == 'LLM'
        is_qtable   = src == 'Q-table'

        ov_cls  = 'pipe-node active' if is_override else 'pipe-node'
        llm_cls = 'pipe-node active llm' if is_llm else 'pipe-node'
        qt_cls  = 'pipe-node active qtable' if is_qtable else 'pipe-node'

        st.markdown(f"""
<div class="step-card">
  <h4>Loop {loop_n} · Step S{step_n:02d} · {scen}</h4>

  <div class="pipe-row">
    <div class="pipe-node">👁 Observe</div>
    <div class="pipe-arrow">→</div>
    <div class="{ov_cls}">⭐ Override?</div>
    <div class="pipe-arrow">→</div>
    <div class="{llm_cls}">🤖 LLM</div>
    <div class="pipe-arrow">→</div>
    <div class="{qt_cls}">📊 Q-table</div>
    <div class="pipe-arrow">→</div>
    <div class="pipe-node">⚡ Act</div>
  </div>

  <table style="width:100%;font-size:12px;border-collapse:collapse;margin-top:8px">
    <tr>
      <td style="color:#64748b;padding:4px 0;width:130px">Decision source</td>
      <td><span class="badge" style="background:{src_color}22;color:{src_color};border:1px solid {src_color}">{src_icon} {src}</span></td>
    </tr>
    <tr>
      <td style="color:#64748b;padding:4px 0">Action taken</td>
      <td><code style="color:#e2e8f0;background:#0f172a;padding:2px 8px;border-radius:4px">{action}</code>
          &nbsp;{acc_badge}</td>
    </tr>
    <tr>
      <td style="color:#64748b;padding:4px 0">Optimal action</td>
      <td><code style="color:#94a3b8;background:#0f172a;padding:2px 8px;border-radius:4px">{optimal}</code></td>
    </tr>
    <tr>
      <td style="color:#64748b;padding:4px 0">Outcome</td>
      <td><span style="color:{res_color};font-weight:700">{res_text}</span></td>
    </tr>
    <tr>
      <td style="color:#64748b;padding:4px 0">Reward</td>
      <td><span style="color:#f59e0b;font-weight:700;font-size:16px">{reward:+.4f}</span></td>
    </tr>
  </table>
</div>
""", unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="step-card" style="text-align:center;padding:40px;color:#334155">'
            '<div style="font-size:40px">⚡</div>'
            '<div style="margin-top:8px">Waiting for first decision...</div>'
            '<div style="font-size:12px;margin-top:4px">Press ▶ Start to begin</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    # ── Loop Summary (appears after each loop completes) ─────────────────────
    if lm:
        st.markdown('<div class="section-title">📊 Loop Summaries</div>', unsafe_allow_html=True)
        df_lm = pd.DataFrame(lm).fillna(0)
        df_lm = df_lm.rename(columns={
            'loop': 'Loop', 'accuracy': 'Accuracy%',
            'res_rate': 'Resolution%', 'reward': 'Avg Reward', 'mttr': 'MTTR',
        })
        st.dataframe(
            df_lm.style.format({
                'Accuracy%': '{:.1f}%', 'Resolution%': '{:.1f}%',
                'Avg Reward': '{:+.4f}', 'MTTR': '{:.2f}',
            }).background_gradient(subset=['Accuracy%'], cmap='RdYlGn', vmin=0, vmax=100),
            use_container_width=True, hide_index=True,
        )

# ── RIGHT: Live Log + Charts ──────────────────────────────────────────────────
with right:
    st.markdown('<div class="section-title">📜 Live Decision Log</div>', unsafe_allow_html=True)

    recent = steps[-40:][::-1]  # last 40, newest first
    if recent:
        log_html = ""
        for s in recent:
            icon  = '★' if s['accurate'] else ' '
            check = '✓' if s['resolved'] else '✗'
            src_c = {'LLM': 'L', 'Override': 'O', 'Q-table': 'Q'}.get(s['source'], '?')
            rwd_s = f"R={s['reward']:+.3f}"
            scen_short = s['scenario'][:26]
            act_short  = s['action'][:14]
            loop_label = f"L{s['loop']}"

            if s['accurate'] and s['resolved']:
                cls = 'log-row log-hit-res'
            elif s['resolved']:
                cls = 'log-row log-hit-unres'
            else:
                cls = 'log-row log-miss'

            log_html += (
                f'<div class="{cls}">'
                f'[{icon}{check}{src_c}] {loop_label}·S{s["step"]:02d} '
                f'| {scen_short:<26} | {act_short:<14} | {rwd_s}'
                f'</div>'
            )
        st.markdown(log_html, unsafe_allow_html=True)
    else:
        st.markdown(
            '<div style="color:#334155;text-align:center;padding:30px;'
            'border:1px dashed #334155;border-radius:8px">'
            'Decision log will appear here as the agent runs</div>',
            unsafe_allow_html=True,
        )

    # ── Live Reward Chart ─────────────────────────────────────────────────────
    if len(steps) >= 3:
        st.markdown('<div class="section-title">📈 Reward — Live</div>', unsafe_allow_html=True)

        rewards = [s['reward'] for s in steps]
        step_nums = list(range(1, len(rewards) + 1))
        # Color each bar by resolved/accurate
        bar_colors = []
        for s in steps:
            if s['accurate'] and s['resolved']:
                bar_colors.append('#22c55e')
            elif s['resolved']:
                bar_colors.append('#f59e0b')
            else:
                bar_colors.append('#ef4444')

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=step_nums, y=rewards,
            marker_color=bar_colors, marker_opacity=0.8,
            name='Reward per step',
            hovertemplate='Step %{x}<br>R=%{y:.4f}<extra></extra>',
        ))
        # Running average line
        if len(rewards) >= 5:
            window = 5
            avg = [
                sum(rewards[max(0, i-window):i+1]) / min(i+1, window)
                for i in range(len(rewards))
            ]
            fig.add_trace(go.Scatter(
                x=step_nums, y=avg,
                mode='lines', line=dict(color='#60a5fa', width=2),
                name='5-step avg',
            ))
        fig.update_layout(
            paper_bgcolor='#1e293b', plot_bgcolor='#1e293b',
            font=dict(color='#94a3b8', size=10),
            height=200,
            margin=dict(t=10, r=10, b=30, l=40),
            xaxis=dict(gridcolor='#334155', title='Decision #'),
            yaxis=dict(gridcolor='#334155', title='Reward', range=[-0.2, 1.1]),
            legend=dict(orientation='h', y=-0.3, bgcolor='rgba(0,0,0,0)'),
            showlegend=True,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Per-loop accuracy chart ───────────────────────────────────────────────
    if len(lm) >= 1:
        st.markdown('<div class="section-title">🎯 Accuracy per Loop</div>', unsafe_allow_html=True)
        loop_nums = [m.get('loop', 0) for m in lm]
        acc_vals  = [m.get('accuracy', 0) for m in lm]
        res_vals  = [m.get('res_rate', 0) for m in lm]

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=loop_nums, y=acc_vals, mode='lines+markers',
            line=dict(color='#3b82f6', width=2.5), marker=dict(size=8),
            name='Action Accuracy %',
        ))
        fig2.add_trace(go.Scatter(
            x=loop_nums, y=res_vals, mode='lines+markers',
            line=dict(color='#22c55e', width=2.5, dash='dot'), marker=dict(size=8),
            name='Resolution Rate %',
        ))
        fig2.update_layout(
            paper_bgcolor='#1e293b', plot_bgcolor='#1e293b',
            font=dict(color='#94a3b8', size=10),
            height=200,
            margin=dict(t=10, r=10, b=30, l=40),
            xaxis=dict(gridcolor='#334155', title='Loop', dtick=1),
            yaxis=dict(gridcolor='#334155', title='%', range=[0, 110]),
            legend=dict(orientation='h', y=-0.35, bgcolor='rgba(0,0,0,0)'),
        )
        st.plotly_chart(fig2, use_container_width=True)

# ── Decision source donut (bottom bar) ───────────────────────────────────────
if st.session_state.llm_cum + st.session_state.override_cum + st.session_state.qtable_cum > 0:
    st.markdown('<div class="section-title">🔀 Decision Source Split (cumulative)</div>',
                unsafe_allow_html=True)
    d1, d2, d3 = st.columns(3)
    total_dec = max(st.session_state.llm_cum + st.session_state.override_cum + st.session_state.qtable_cum, 1)

    with d1:
        pct = round(st.session_state.llm_cum / total_dec * 100, 1)
        st.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-label">🤖 LLM</div>'
            f'<div class="kpi-value" style="color:#3b82f6">{pct}%</div>'
            f'<div class="kpi-sub">{st.session_state.llm_cum} decisions</div></div>',
            unsafe_allow_html=True,
        )
    with d2:
        pct = round(st.session_state.override_cum / total_dec * 100, 1)
        st.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-label">⭐ Q-table Override</div>'
            f'<div class="kpi-value" style="color:#a78bfa">{pct}%</div>'
            f'<div class="kpi-sub">{st.session_state.override_cum} bypassed LLM</div></div>',
            unsafe_allow_html=True,
        )
    with d3:
        pct = round(st.session_state.qtable_cum / total_dec * 100, 1)
        st.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-label">📊 Q-table Fallback</div>'
            f'<div class="kpi-value" style="color:#34d399">{pct}%</div>'
            f'<div class="kpi-sub">{st.session_state.qtable_cum} decisions</div></div>',
            unsafe_allow_html=True,
        )

# ── Auto-refresh while running ────────────────────────────────────────────────
if st.session_state.running:
    time.sleep(1.0)
    st.rerun()
elif st.session_state.done and st.session_state.steps:
    # Auto-regenerate the results HTML once per completed run
    if not st.session_state.get('report_generated'):
        try:
            import subprocess as _sp
            _sp.run(
                [PYTHON, 'generate_report.py'],
                cwd=WORK_DIR, timeout=60,
                stdout=_sp.DEVNULL, stderr=_sp.DEVNULL,
            )
            st.session_state.report_generated = True
            st.session_state.report_time = time.strftime("%H:%M:%S")
        except Exception:
            pass

    rpt_time = st.session_state.get('report_time', '')
    st.markdown(
        f'<div style="background:#14532d;border:1px solid #22c55e;border-radius:10px;'
        f'padding:16px 20px;margin-top:16px">'
        f'<span style="font-size:16px;font-weight:700;color:#4ade80">'
        f'✅ Evaluation complete</span>'
        f'&nbsp;&nbsp;<span style="color:#86efac;font-size:13px">'
        f'{len(steps)} decisions · {st.session_state.cur_loop} loops · '
        f'Final accuracy: {acc_now:.1f}% · Override fires: {st.session_state.override_cum}'
        f'</span>'
        f'<div style="margin-top:10px;display:flex;gap:16px;flex-wrap:wrap">'
        f'<a href="http://localhost:8888/itops_results.html" target="_blank" '
        f'style="background:#1e3a5f;color:#60a5fa;padding:6px 16px;border-radius:6px;'
        f'text-decoration:none;font-size:13px;font-weight:600;border:1px solid #3b82f6">'
        f'📊 View Full Results Report →</a>'
        f'<a href="http://localhost:8502" target="_blank" '
        f'style="background:#2e1065;color:#a78bfa;padding:6px 16px;border-radius:6px;'
        f'text-decoration:none;font-size:13px;font-weight:600;border:1px solid #7c3aed">'
        f'📈 Streamlit Dashboard →</a>'
        f'</div>'
        + (f'<div style="margin-top:6px;color:#64748b;font-size:11px">'
           f'Report regenerated at {rpt_time}</div>' if rpt_time else '')
        + '</div>',
        unsafe_allow_html=True,
    )
