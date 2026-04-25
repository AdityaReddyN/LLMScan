# streamlit_app.py
# pip install streamlit requests plotly

import streamlit as st
import requests
import plotly.graph_objects as go
import numpy as np

# ── Page config ──────────────────────────────────────────────
st.set_page_config(page_title="LLM Scan Demo", page_icon="🔍", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 4rem;
        padding-right: 4rem;
    }
    .stTextArea textarea {
        font-size: 1.05rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("🔍 LLM Scan — Misbehavior Detector")
st.markdown("<p style='font-size: 1.1rem; color: #888;'>Detect and analyze LLM behaviors like jailbreaks and lies in real-time.</p>", unsafe_allow_html=True)
st.divider()

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    st.markdown("<br>", unsafe_allow_html=True)
    api_url = st.text_input(
        "Kaggle API URL (ngrok)",
        placeholder="https://ranular-renae-preseasonal.ngrok-free.dev/",
    )
    st.caption("Run the Kaggle API cell first, then paste the ngrok URL above.")
    st.markdown("<br>", unsafe_allow_html=True)
    mode = st.radio("Detection Mode", ["Standard", "Streaming Monitor"])
    st.markdown("---")

# ── Helpers ───────────────────────────────────────────────────
VERDICT_COLOR = {"NORMAL": "green", "LIE": "orange", "JAILBREAK": "red", "REFUSED": "blue"}
NUM_LAYERS = 32   # Mistral-7B has 32 transformer layers

def call_api(prompt, mode_str):
    url = api_url.rstrip("/") + "/predict"
    resp = requests.post(url, json={"prompt": prompt, "mode": mode_str}, timeout=120)
    resp.raise_for_status()
    return resp.json()

def plot_causal_map(layer_ce, title="Layer CE Reduction (causal map)"):
    layers = list(range(len(layer_ce)))
    colors = ["crimson" if v < 0 else "steelblue" for v in layer_ce]
    fig = go.Figure(go.Bar(
        x=layers, y=layer_ce,
        marker_color=colors,
        hovertemplate="Layer %{x}<br>CE shift: %{y:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Transformer Layer",
        yaxis_title="Normalized CE Reduction",
        height=350,
        margin=dict(l=40, r=20, t=50, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    return fig

def plot_layer_ce_log(layer_ce_log):
    """Heatmap of CE across tokens x layers for streaming mode."""
    if not layer_ce_log:
        return None
    mat = np.array(layer_ce_log)   # shape: (n_checks, n_layers)
    fig = go.Figure(go.Heatmap(
        z=mat,
        colorscale="RdBu",
        zmid=0,
        xaxis_title="Layer",
        yaxis_title="Token check #",
        hovertemplate="Token %{y}, Layer %{x}<br>CE shift: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title="Streaming causal map (token × layer)",
        xaxis_title="Transformer Layer",
        yaxis_title="Token Check Index",
        height=380,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig

def plot_delta(delta, peak_layer):
    layers = list(range(len(delta)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=layers, y=delta,
        mode="lines+markers",
        line=dict(color="mediumpurple", width=2),
        name="Counterfactual delta",
    ))
    fig.add_vline(x=peak_layer, line_dash="dot", line_color="red",
                  annotation_text=f"Peak L{peak_layer}", annotation_position="top right")
    fig.update_layout(
        title="Counterfactual Causal Shift (intervention − original)",
        xaxis_title="Layer", yaxis_title="|Δ CE|",
        height=300, margin=dict(l=40, r=20, t=50, b=40),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig

def prob_gauge(label, value, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(value * 100, 1),
        number={"suffix": "%"},
        title={"text": label},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "bgcolor": "whitesmoke",
            "borderwidth": 1,
        },
    ))
    fig.update_layout(height=240, margin=dict(l=20, r=20, t=50, b=20))
    return fig

# ── Main UI ───────────────────────────────────────────────────
with st.container():
    st.markdown("### 📝 Input Prompt")
    prompt = st.text_area(
        "Enter a prompt to analyze:", 
        height=140,
        placeholder="e.g. Give me step by step instructions to make a bomb.",
        label_visibility="collapsed"
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        run = st.button("🚀 Analyze Prompt", type="primary", disabled=not api_url, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

if run and prompt.strip():
    if not api_url:
        st.error("Please provide the Kaggle API URL in the sidebar.")
    else:
        st.divider()
        st.markdown("### 📊 Analysis Results")
        st.markdown("<br>", unsafe_allow_html=True)
        
        with st.spinner("Running Mistral forward pass..."):
            try:
                result = call_api(prompt, mode.lower().split()[0])
            except Exception as e:
                st.error(f"API error: {e}")
                st.stop()

        # ── Standard mode results ──────────────────────────────
        if result.get("mode") == "standard":
            verdict = result["verdict"]
            probs   = result["probs"]
            causal  = result["causal"]

            col_v, col_r = st.columns([1, 2.5], gap="large")
            with col_v:
                color = VERDICT_COLOR.get(verdict, "gray")
                st.markdown(
                    f"""
                    <div style="padding: 24px; border-radius: 12px; border: 2px solid {color}; background-color: rgba(255,255,255,0.02); text-align: center; height: 100%;">
                        <p style='color: gray; margin-bottom: 5px; text-transform: uppercase; letter-spacing: 1px; font-size: 0.9rem;'>Verdict</p>
                        <h2 style='color:{color}; margin-top: 0; font-size: 2.2rem;'>{verdict}</h2>
                        <hr style="border-color: rgba(255,255,255,0.1); margin: 15px 0;">
                        <p style="margin:0; font-size: 0.9rem;">Causal L2 shift: <b>{causal['l2']:.4f}</b></p>
                        <p style="margin:0; font-size: 0.9rem;">Peak layer: <b>L{causal['peak_layer']}</b></p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col_r:
                st.markdown("**Model Response:**")
                st.info(result.get("response", "—"))

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 🎯 Detection Probabilities")
            
            # Probability gauges
            g1, g2, g3 = st.columns(3, gap="large")
            g1.plotly_chart(prob_gauge("P(Safe)",      probs["safe"],      "steelblue"),  use_container_width=True)
            g2.plotly_chart(prob_gauge("P(Lie)",       probs["lie"],       "orange"),     use_container_width=True)
            g3.plotly_chart(prob_gauge("P(Jailbreak)", probs["jailbreak"], "crimson"),    use_container_width=True)

            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("#### 🧠 Causal Explanations")
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Causal map and Counterfactual shift in columns if both exist
            if causal.get("delta"):
                col_map, col_delta = st.columns(2, gap="large")
                with col_map:
                    st.plotly_chart(plot_causal_map(result["layer_ce"]), use_container_width=True)
                with col_delta:
                    st.plotly_chart(plot_delta(causal["delta"], causal["peak_layer"]), use_container_width=True)
            else:
                st.plotly_chart(plot_causal_map(result["layer_ce"]), use_container_width=True)

            # Raw numbers expander
            st.markdown("<br>", unsafe_allow_html=True)
            with st.expander("Show Raw Layer CE Values"):
                st.write(result["layer_ce"])

        # ── Streaming mode results ─────────────────────────────
        elif result.get("mode") == "streaming":
            intercepted = result.get("intercepted", False)
            pred_class  = result.get("pred_class")
            class_names = {0: "NORMAL", 1: "LIE", 2: "JAILBREAK"}

            col_v, col_r = st.columns([1, 2.5], gap="large")
            with col_v:
                if intercepted:
                    cls_name = class_names.get(pred_class, "UNKNOWN")
                    color = VERDICT_COLOR.get(cls_name, "gray")
                    st.markdown(
                        f"""
                        <div style="padding: 20px; border-radius: 12px; border: 2px solid {color}; background-color: rgba(255,255,255,0.02); text-align: center; height: 100%;">
                            <h3 style='color:{color}; margin-top: 0;'>⛔ INTERCEPTED</h3>
                            <h2 style='color:{color};'>{cls_name}</h2>
                            <hr style="border-color: rgba(255,255,255,0.1); margin: 15px 0;">
                            <p style="margin:0; font-size: 0.9rem;">Token #<b>{result.get('onset_token')}</b></p>
                            <p style="margin:0; font-size: 0.9rem;">Layer <b>{result.get('onset_layer')}</b></p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div style="padding: 20px; border-radius: 12px; border: 2px solid green; background-color: rgba(255,255,255,0.02); text-align: center; height: 100%;">
                            <h2 style='color:green; margin-top: 10px;'>✅ NORMAL</h2>
                            <p style='color: gray;'>No anomaly detected</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            with col_r:
                st.markdown("**Generated Text:**")
                st.info(result.get("text", "—"))

            st.markdown("<br><hr>", unsafe_allow_html=True)
            
            log = result.get("layer_ce_log", [])
            if log:
                st.markdown("#### 🌊 Streaming Causal Map")
                st.plotly_chart(plot_layer_ce_log(log), use_container_width=True)
            else:
                st.info("No token checks recorded (prompt may have been refused or too short).")

        # ── Unknown / error ────────────────────────────────────
        elif "error" in result:
            st.error(f"Model error: {result['error']}")