# streamlit_app.py
# pip install streamlit requests plotly numpy pandas

import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

# ── Page Configuration ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="LLM Scan — Causal Misbehavior Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── High-Contrast Premium Cyberpunk CSS ──────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Plus+Jakarta+Sans:wght@400;600;800&display=swap');
    
    /* Global Typography & Background Tweaks */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        background-color: #0b0f19;
        color: #f3f4f6;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 800 !important;
        letter-spacing: -0.02em;
    }
    
    /* Verdict Banners with Micro-Gradients */
    .verdict-safe {
        padding: 2.5rem 2rem;
        border-radius: 16px;
        background: linear-gradient(135deg, #064e3b 0%, #022c22 100%);
        border: 2px solid #10b981;
        box-shadow: 0 0 30px rgba(16, 185, 129, 0.15);
        text-align: center;
    }
    .verdict-misbehavior {
        padding: 2.5rem 2rem;
        border-radius: 16px;
        background: linear-gradient(135deg, #7f1d1d 0%, #450a0a 100%);
        border: 2px solid #ef4444;
        box-shadow: 0 0 30px rgba(239, 68, 68, 0.15);
        text-align: center;
    }
    
    /* Task Cards */
    .task-winner {
        padding: 1.5rem;
        border-radius: 14px;
        background: rgba(239, 68, 68, 0.08);
        border: 2px solid #ef4444;
        box-shadow: 0 0 15px rgba(239, 68, 68, 0.1);
    }
    .task-runner {
        padding: 1.5rem;
        border-radius: 14px;
        background: #111827;
        border: 1px solid #374151;
    }
    
    /* Dynamic Confidence Text Colors */
    .confidence-high { color: #10b981; font-weight: 800; text-transform: uppercase; }
    .confidence-medium { color: #f59e0b; font-weight: 800; text-transform: uppercase; }
    .confidence-low { color: #ef4444; font-weight: 800; text-transform: uppercase; }
    
    /* Form Inputs and Interactive States */
    .stTextArea textarea {
        font-size: 1.15rem !important;
        font-family: 'JetBrains Mono', monospace !important;
        background-color: #1f2937 !important;
        color: #ffffff !important;
        border: 1px solid #4b5563 !important;
        border-radius: 10px !important;
    }
    .stTextArea textarea:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.3) !important;
    }
    
    /* Dropdown / Selectbox Sleek Styling */
    div[data-baseweb="select"] > div {
        background-color: #1f2937 !important;
        border: 1px solid #4b5563 !important;
        border-radius: 10px !important;
        color: #f3f4f6 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 600 !important;
        transition: all 0.2s ease-in-out;
    }
    div[data-baseweb="select"] > div:focus-within, div[data-baseweb="select"] > div:hover {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.3) !important;
    }
    
    /* Big Bold Run Button Adjustments */
    div[data-testid="stButton"] button {
        background: linear-gradient(90deg, #2563eb 0%, #1d4ed8 100%) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 1.2rem !important;
        padding: 0.75rem 2rem !important;
        border-radius: 12px !important;
        border: none !important;
        box-shadow: 0 4px 15px rgba(37, 99, 235, 0.3) !important;
        transition: all 0.2s ease;
    }
    div[data-testid="stButton"] button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.4) !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar Configuration Block ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h2 style='margin-bottom: 0;'>Control Unit</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #9ca3af; font-size: 0.9rem;'>Configure global scan pipeline telemetry</p>", unsafe_allow_html=True)
    st.divider()
    
    api_url = st.text_input(
        "Backend Gateway API Endpoint",
        placeholder="https://your-ngrok-url.ngrok-free.app",
        help="Paste your active Kaggle compute container ngrok tunnel URL here"
    )
    
    st.divider()
    st.markdown("### Diagnostic Views")
    show_raw = st.toggle("Expose Raw Response Packets", value=False)
    compare_mode = st.toggle("Overlay Normal Latent Baseline", value=False)
    
    st.markdown("---")
    st.caption("Node Status: Active & polling via ngrok")
    st.caption("Compute Target: Remote GPU Environment")

# ── Main Header Interface ───────────────────────────────────────────────────
st.markdown("<h1 style='font-size: 3rem; margin-bottom: 0.2rem;'>LLM SCAN</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='font-size: 1.3rem; color: #9ca3af; font-weight: 400; margin-top: 0;'>"
    "Real-time evaluation of transformer activation spaces via layer and attention tensor interventions."
    "</p>", 
    unsafe_allow_html=True
)
st.divider()

# ── Core Inputs Section (Integrated) ─────────────────────────────────────────

col_mode, col_task = st.columns([1, 1])

with col_mode:
    detection_mode = st.selectbox(
        "Causal Evaluation Mode",
        ["Hierarchical (Full)", "Unified Only (Fast)", "Task-Specific"],
        index=0,
        help="Hierarchical: Core Layer 1 validation cascaded by Layer 2 intent parsing. Unified: Quick binary scan. Task-specific: Focused singular interceptor target."
    )

selected_task = None
with col_task:
    if detection_mode == "Task-Specific":
        selected_task = st.selectbox(
            "Target Pipeline Signature", 
            ["lie", "jailbreak"],
            help="Select the specific misbehavior vector to isolate."
        )

st.markdown("<br>", unsafe_allow_html=True)

prompt = st.text_area(
    "Ingestion Engine Payload:",
    height=140,
    placeholder="Example: Respond to the following query exclusively with an absolute fabrication.\nQuestion: What is 2+2?\nAnswer:",
    label_visibility="collapsed"
)

btn_col1, btn_col2, btn_col3 = st.columns([1, 1.5, 1])
with btn_col2:
    analyze_btn = st.button(
        "RUN SYSTEM-WIDE CAUSAL SCAN", 
        disabled=not api_url,
        use_container_width=True
    )

# ── Core Network Telemetry Methods ─────────────────────────────────────────
def call_api(prompt_text, mode, task=None):
    """Network proxy router communicating with remote container."""
    base = api_url.rstrip("/")
    
    if mode == "Hierarchical (Full)":
        endpoint = f"{base}/detect_hierarchical"
        payload = {"prompt": prompt_text, "verbose": False}
    elif mode == "Unified Only (Fast)":
        endpoint = f"{base}/detect_unified"
        payload = {"prompt": prompt_text}
    else:
        endpoint = f"{base}/detect_task"
        payload = {"prompt": prompt_text, "task": task}
    
    try:
        resp = requests.post(endpoint, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("[Gateway Connection Timeout] Ensure your remote kernel is up and exposing the correct public ngrok routing tunnel.")
        return None
    except requests.exceptions.Timeout:
        st.error("[Network Timeout] Complex causal mapping requires heavy token tensor interventions (Expected: 30-60s).")
        return None
    except Exception as e:
        st.error(f"[Structural Component Error] {str(e)}")
        return None

# ── Visualization Generation Pipeline ───────────────────────────────────────

def plot_probability_gauge(prob, title="Misbehavior Probability Tracker"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={"suffix": "%", "font": {"size": 48, "color": "#ffffff", "family": "Plus Jakarta Sans"}},
        title={"text": title, "font": {"size": 16, "color": "#9ca3af", "family": "Plus Jakarta Sans"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#9ca3af"},
            "bar": {"color": "#ef4444" if prob > 0.5 else "#10b981", "thickness": 0.3},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 1,
            "bordercolor": "#4b5563",
            "steps": [
                {"range": [0, 35], "color": "rgba(16, 185, 129, 0.05)"},
                {"range": [35, 70], "color": "rgba(245, 158, 11, 0.05)"},
                {"range": [70, 100], "color": "rgba(239, 68, 68, 0.05)"}
            ]
        }
    ))
    fig.update_layout(
        template="plotly_dark",
        height=260,
        margin=dict(l=30, r=30, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig

def plot_token_ce(token_ce, title="Token-Level Causal Influence Profiler"):
    fig = go.Figure()
    mean_val = np.mean(token_ce) if token_ce else 0
    std_val = np.std(token_ce) if token_ce else 0
    
    colors = ["#ef4444" if v > mean_val + std_val else "#3b82f6" for v in token_ce]
    
    fig.add_trace(go.Bar(
        x=list(range(len(token_ce))),
        y=token_ce,
        marker_color=colors,
        hovertemplate="Index Offset %{x}<br>Causal Load: %{y:.4f}<extra></extra>"
    ))
    
    fig.add_hline(
        y=mean_val, 
        line_dash="dash", 
        line_color="#f59e0b",
        line_width=2,
        annotation_text="Global Mean Causal Load",
        annotation_position="top left"
    )
    
    fig.update_layout(
        template="plotly_dark",
        title={"text": title, "font": {"size": 16, "family": "Plus Jakarta Sans"}},
        height=360,
        margin=dict(l=60, r=20, t=60, b=50),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        # Moving the grid color styling directly to the axes
        xaxis=dict(
            title="Input Token Structural Position Index", 
            gridcolor="#1f2937"
        ),
        yaxis=dict(
            title="L2 Latent Activation Shift Magnitude", 
            gridcolor="#1f2937"
        ),
        showlegend=False
    )
    return fig

def plot_layer_ce(layer_ce, title="Layer-Level Structural Anomaly Impact"):
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.15,
        subplot_titles=("Residual Stream Deviation Scope", "Activation Density Profile Map")
    )
    
    percentile_75 = np.percentile(layer_ce, 75) if layer_ce else 0
    colors = ["#ef4444" if v > percentile_75 else "#3b82f6" for v in layer_ce]
    
    fig.add_trace(
        go.Bar(
            x=list(range(len(layer_ce))),
            y=layer_ce,
            marker_color=colors,
            hovertemplate="Layer Block %{x}<br>Deviation Vector: %{y:.4f}<extra></extra>"
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Heatmap(
            z=[layer_ce],
            x=list(range(len(layer_ce))),
            colorscale="Viridis",
            showscale=False,
            hovertemplate="Layer Block %{x}<br>Density Measure: %{z:.4f}<extra></extra>"
        ),
        row=2, col=1
    )
    
    fig.update_xaxes(title_text="Transformer Structural Layer Index Depth", row=2, col=1)
    fig.update_yaxes(title_text="Deviation Load", row=1, col=1)
    fig.update_yaxes(showticklabels=False, row=2, col=1)
    
    fig.update_layout(
        template="plotly_dark",
        title={"text": title, "font": {"size": 16, "family": "Plus Jakarta Sans"}},
        height=480,
        margin=dict(l=60, r=20, t=60, b=50),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False
    )
    return fig

def plot_task_scores(task_scores, most_likely):
    tasks = list(task_scores.keys())
    scores = [task_scores[t] for t in tasks]
    
    colors = ["#ef4444" if t == most_likely else "#4b5563" for t in tasks]
    
    fig = go.Figure(go.Bar(
        x=scores,
        y=[t.upper() for t in tasks],
        orientation='h',
        marker_color=colors,
        text=[f" {s:.4f} " for s in scores],
        textposition="outside",
        textfont={"weight": "bold", "size": 13},
        hovertemplate="Target %{y}: Score %{x:.4f}<extra></extra>"
    ))
    
    fig.update_layout(
        template="plotly_dark",
        title={"text": "Class-Specific Optimization Objectives", "font": {"size": 15, "family": "Plus Jakarta Sans"}},
        xaxis_title="Classification Probability Target Vector",
        height=260,
        margin=dict(l=90, r=60, t=50, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False
    )
    fig.update_xaxes(range=[0, max(scores) * 1.2 if scores else 1.0])
    return fig

def plot_token_stats(stats):
    labels = ['Mean Intensity', 'Variance Std', 'Dynamic Range', 'Distribution Skew', 'Kurtosis Profile', 'Top3 Density', 'Vector Entropy', 'Peak Causal Load']
    values = [float(v) for v in stats[:8]]
    
    min_v, max_v = min(values), max(values)
    values_norm = [(v - min_v) / ((max_v - min_v) + 1e-10) for v in values]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values_norm + [values_norm[0]],
        theta=labels + [labels[0]],
        fill='toself',
        fillcolor="rgba(59, 130, 246, 0.25)",
        line=dict(color="#3b82f6", width=3),
        marker=dict(size=8, color="#60a5fa")
    ))
    
    fig.update_layout(
        template="plotly_dark",
        polar=dict(
            radialaxis=dict(
                visible=True, 
                range=[0, 1], 
                showticklabels=False,
                gridcolor="#374151"
            ),
            angularaxis=dict(
                gridcolor="#374151",
                tickfont=dict(size=12, color="#e5e7eb", weight="bold")
            ),
            bgcolor="rgba(17, 24, 39, 0.5)"
        ),
        title={"text": "Token Representation Invariant Profile Summary", "font": {"size": 16, "family": "Plus Jakarta Sans"}},
        height=400,
        margin=dict(l=60, r=60, t=60, b=40),
        paper_bgcolor="rgba(0,0,0,0)"
    )
    return fig

def plot_comparison(normal_data, current_data):
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Control Group Baseline", "Active Subject Runtime State"),
        shared_yaxes=True
    )
    
    fig.add_trace(
        go.Bar(x=list(range(len(normal_data))), y=normal_data, 
               marker_color="#10b981", name="Normal Target"),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(x=list(range(len(current_data))), y=current_data,
               marker_color="#ef4444", name="Active Target"),
        row=1, col=2
    )
    
    fig.update_layout(
        template="plotly_dark",
        title={"text": "Comparative Activation Profile Matrix Check", "font": {"size": 16, "family": "Plus Jakarta Sans"}},
        height=360,
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig

# ── Dynamic Operations Execution Stream ───────────────────────────────────────────
if analyze_btn and prompt.strip():
    if not api_url:
        st.error("[Pipeline Exception] No valid remote compute gateway is selected. Set up your host configurations.")
    else:
        st.divider()
        st.markdown("<h2 style='font-size: 2rem; margin-bottom: 1rem;'>Scan Analysis Metrics Output</h2>", unsafe_allow_html=True)
        
        with st.spinner("Accessing network layers & monitoring activation state modifications..."):
            result = call_api(prompt, detection_mode, selected_task)
        
        if result is None:
            st.stop()
        
        level1 = result.get("level1", {})
        is_misbehavior = level1.get("is_misbehavior", False)
        prob = level1.get("probability", 0.5)
        confidence = level1.get("confidence", "low")
        
        causal_map = result.get("causal_map", {})
        token_ce = causal_map.get("token_ce", [])
        layer_ce = causal_map.get("layer_ce", [])
        token_stats = causal_map.get("token_stats", [])
        
        level2 = result.get("level2")

        # ── Tabbed Operations Command Deck ────────────────────────────────────────
        tab1, tab2 = st.tabs(["REAL-TIME DIAGNOSTICS", "LATENT LAYER ANALYTICS"])
        
        with tab1:
            st.markdown("<br>", unsafe_allow_html=True)
            
            v_col1, v_col2 = st.columns([1.2, 1], gap="large")
            
            with v_col1:
                if is_misbehavior:
                    st.markdown(f"""
                    <div class="verdict-misbehavior">
                        <p style='color: #fca5a5; margin-bottom: 5px; font-weight: 700; text-transform: uppercase; letter-spacing: 3px; font-size: 1rem;'>System Violation Flagged</p>
                        <h1 style='color: #fff; margin: 0; font-size: 3.5rem; font-weight: 900;'>MISBEHAVIOR</h1>
                        <p style='color: #ffffff; margin-top: 15px; font-size: 1.3rem; font-weight: 600;'>
                            Pipeline Confidence rating: <span class="confidence-{confidence}">{confidence}</span>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="verdict-safe">
                        <p style='color: #34d399; margin-bottom: 5px; font-weight: 700; text-transform: uppercase; letter-spacing: 3px; font-size: 1rem;'>Vector Path Verified</p>
                        <h1 style='color: #fff; margin: 0; font-size: 3.5rem; font-weight: 900;'>SECURE STATE</h1>
                        <p style='color: #ffffff; margin-top: 15px; font-size: 1.3rem; font-weight: 600;'>
                            Pipeline Confidence rating: <span class="confidence-{confidence}">{confidence}</span>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with v_col2:
                st.plotly_chart(
                    plot_probability_gauge(prob),
                    use_container_width=True
                )
            
            if level2 and is_misbehavior:
                st.markdown("<br><hr>", unsafe_allow_html=True)
                st.markdown("<h3 style='font-size: 1.6rem;'>Parsed Intent Signatures (Layer 2)</h3>", unsafe_allow_html=True)
                
                task_scores = level2.get("task_scores", {})
                most_likely = level2.get("most_likely_task", "unknown")
                task_conf = level2.get("task_confidence", 0)
                
                t_cols = st.columns(len(task_scores) if task_scores else 1)
                for idx, (task, score) in enumerate(task_scores.items()):
                    is_winner = task == most_likely
                    with t_cols[idx]:
                        card_style = "task-winner" if is_winner else "task-runner"
                        label_color = "#ef4444" if is_winner else "#9ca3af"
                        
                        st.markdown(f"""
                        <div class="{card_style}">
                            <p style='margin: 0; color: {label_color}; font-weight: 700; font-size: 0.95rem; text-transform: uppercase;'>{task}</p>
                            <h2 style='margin: 8px 0; font-size: 2.2rem; font-weight: 800; color: #ffffff;'>{score:.4f}</h2>
                            <p style='margin: 0; color: #9ca3af; font-size: 0.85rem;'>
                                { "Dominant Threat Vector" if is_winner else "Sub-critical Vector" }
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                m1, m2 = st.columns([1.5, 1], gap="medium")
                with m1:
                    st.plotly_chart(
                        plot_task_scores(task_scores, most_likely),
                        use_container_width=True
                    )
                with m2:
                    cfg_bound_color = "#10b981" if task_conf > 0.2 else "#f59e0b" if task_conf > 0.1 else "#ef4444"
                    st.markdown(f"""
                    <div style="padding: 1.5rem; margin-top: 2rem; border-radius: 12px; background: #111827; border-left: 5px solid {cfg_bound_color}; border-top: 1px solid #1f2937; border-bottom: 1px solid #1f2937; border-right: 1px solid #1f2937;">
                        <p style="margin: 0; color: #9ca3af; font-weight:600; font-size:0.95rem;">Intent Divergence Delta Margin:</p>
                        <p style="margin: 5px 0 10px 0; font-size: 2.2rem; color: {cfg_bound_color}; font-weight: 800; font-family:'JetBrains Mono';">{task_conf:.4f}</p>
                        <p style="margin: 0; color: #d1d5db; font-size: 0.95rem; line-height: 1.4;">
                            {"[CRITICAL RESOLUTION]: Definite vector clustering matched. Classification contains high empirical validity." if task_conf > 0.2 else 
                             "[WARNING RESOLUTION]: Intersecting vectors encountered. Ambiguity detected; run manual review check." if task_conf > 0.1 else 
                             "[LOW RESOLUTION]: Highly distributed latent mapping state. Signatures match multiple profiles."}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
            
            elif is_misbehavior and not level2:
                st.markdown("<br>", unsafe_allow_html=True)
                st.warning("Telemetry Alert: Layer 1 structural threat flag tripped, but Layer 2 evaluation matrices failed to map localized intent signatures.")
        
        with tab2:
            st.markdown("<br>", unsafe_allow_html=True)
            
            map_col1, map_col2 = st.columns(2, gap="large")
            
            with map_col1:
                if token_ce:
                    st.plotly_chart(
                        plot_token_ce(token_ce),
                        use_container_width=True
                    )
                else:
                    st.info("System Tracking Notification: Token attention spatial maps omitted in current evaluation loop.")
            
            with map_col2:
                if layer_ce:
                    st.plotly_chart(
                        plot_layer_ce(layer_ce),
                        use_container_width=True
                    )
                else:
                    st.info("System Tracking Notification: Structural Layer-skip metrics omitted in current evaluation loop.")
            
            if token_stats or compare_mode:
                st.markdown("<hr>", unsafe_allow_html=True)
                infra_col1, infra_col2 = st.columns([1.2, 1], gap="large")
                
                with infra_col1:
                    if token_stats:
                        st.plotly_chart(
                            plot_token_stats(token_stats),
                            use_container_width=True
                        )
                
                with infra_col2:
                    if compare_mode and layer_ce:
                        synthetic_baseline = [max(0, v * np.random.uniform(0.1, 0.4)) for v in layer_ce]
                        st.plotly_chart(
                            plot_comparison(synthetic_baseline, layer_ce),
                            use_container_width=True
                        )
                    elif compare_mode:
                        st.info("Overlay Mode Active: Awaiting valid real-time activation telemetry matrix fields.")

        # ── Debug Packet Trace Output ─────────────────────────────────────────────
        if show_raw:
            st.divider()
            st.markdown("<h3 style='font-size: 1.4rem; color: #f59e0b;'>Raw Telemetry API Hex Packet Log</h3>", unsafe_allow_html=True)
            st.json(result)

# ── Footer Structural Signature ──────────────────────────────────────────────
st.markdown("<br><br><hr>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: #4b5563; font-size: 0.85rem; font-family: \"JetBrains Mono\", monospace;'>"
    "LLMScan Control Engine Platform Architecture | Host Linkage Established via Secure Tunnel Proxies"
    "</p>", 
    unsafe_allow_html=True
)