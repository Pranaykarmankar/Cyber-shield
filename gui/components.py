"""
CyberShield UI Design System and Components
Includes custom Cyberpunk/Navy CSS, glassmorphism cards, KPI counters, and threat intelligence reports.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, List, Optional
from gui.config import SEVERITY_MAP, SEVERITY_COLOR, ACTIONS_MAP

def inject_cyber_theme():
    """Inject premium dark cyber-defense CSS stylesheet."""
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

        /* ── Core Theme Variables ── */
        :root {
            --bg-base: #080c14;
            --bg-surface: #0f172a;
            --bg-card: rgba(15, 23, 42, 0.75);
            --border-glass: rgba(56, 189, 248, 0.18);
            --border-glow: rgba(0, 240, 255, 0.35);
            --text-main: #f1f5f9;
            --text-muted: #94a3b8;
            --accent-cyan: #00f0ff;
            --accent-blue: #38bdf8;
            --accent-danger: #ef4444;
            --accent-warning: #f59e0b;
            --accent-success: #10b981;
            --accent-purple: #818cf8;
        }

        /* Streamlit App background */
        .stApp {
            background-color: var(--bg-base);
            color: var(--text-main);
            font-family: 'Inter', sans-serif;
        }

        /* ── Sidebar Styling ── */
        section[data-testid="stSidebar"] {
            background-color: #0c1220 !important;
            border-right: 1px solid rgba(56, 189, 248, 0.12);
        }

        .sidebar-brand-card {
            background: linear-gradient(135deg, rgba(14, 165, 233, 0.12), rgba(99, 102, 241, 0.08));
            border: 1px solid rgba(56, 189, 248, 0.25);
            border-radius: 12px;
            padding: 18px 16px;
            margin-bottom: 20px;
            text-align: center;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
        }

        .sidebar-brand-title {
            font-family: 'Inter', sans-serif;
            font-size: 1.35rem;
            font-weight: 800;
            letter-spacing: 0.5px;
            background: linear-gradient(135deg, #00f0ff, #38bdf8, #818cf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 6px 0 2px 0;
        }

        .sidebar-brand-sub {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            color: var(--accent-cyan);
            letter-spacing: 1.5px;
            text-transform: uppercase;
        }

        /* ── Page Hero Header ── */
        .page-hero-banner {
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.7) 100%);
            border: 1px solid var(--border-glass);
            border-left: 4px solid var(--accent-cyan);
            border-radius: 12px;
            padding: 22px 26px;
            margin-bottom: 24px;
            position: relative;
            overflow: hidden;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        }

        .page-hero-title {
            font-size: 1.65rem;
            font-weight: 800;
            color: #ffffff;
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 6px;
        }

        .page-hero-sub {
            color: var(--text-muted);
            font-size: 0.92rem;
            line-height: 1.5;
            max-width: 900px;
        }

        .pulse-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background-color: var(--accent-cyan);
            box-shadow: 0 0 10px var(--accent-cyan);
            animation: pulse-glow 2s infinite;
        }

        @keyframes pulse-glow {
            0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0, 240, 255, 0.7); }
            70% { transform: scale(1); box-shadow: 0 0 0 10px rgba(0, 240, 255, 0); }
            100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0, 240, 255, 0); }
        }

        /* ── Glass Cards & Containers ── */
        .cyber-card {
            background: var(--bg-card);
            border: 1px solid var(--border-glass);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            backdrop-filter: blur(8px);
            transition: border-color 0.2s ease, transform 0.2s ease;
        }

        .cyber-card:hover {
            border-color: rgba(56, 189, 248, 0.35);
        }

        /* ── Metric Display Cards ── */
        .metric-card-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }

        .kpi-metric-box {
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.85), rgba(30, 41, 59, 0.6));
            border: 1px solid rgba(56, 189, 248, 0.2);
            border-radius: 10px;
            padding: 16px 18px;
            text-align: left;
            position: relative;
        }

        .kpi-title {
            font-size: 0.78rem;
            font-family: 'JetBrains Mono', monospace;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--text-muted);
            margin-bottom: 6px;
        }

        .kpi-val {
            font-size: 1.8rem;
            font-weight: 800;
            color: #ffffff;
            font-family: 'Inter', sans-serif;
        }

        .kpi-val.cyan { color: var(--accent-cyan); }
        .kpi-val.red { color: var(--accent-danger); }
        .kpi-val.green { color: var(--accent-success); }
        .kpi-val.amber { color: var(--accent-warning); }

        .kpi-sub {
            font-size: 0.75rem;
            color: #64748b;
            margin-top: 4px;
        }

        /* ── Badges ── */
        .cyber-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 10px;
            border-radius: 6px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.74rem;
            font-weight: 600;
            text-transform: uppercase;
        }

        .badge-critical { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); }
        .badge-high { background: rgba(249, 115, 22, 0.2); color: #fb923c; border: 1px solid rgba(249, 115, 22, 0.4); }
        .badge-medium { background: rgba(245, 158, 11, 0.2); color: #fcd34d; border: 1px solid rgba(245, 158, 11, 0.4); }
        .badge-safe { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }

        /* ── Threat Intelligence Alert Box ── */
        .threat-alert-panel {
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.12), rgba(15, 23, 42, 0.85));
            border: 1px solid rgba(239, 68, 68, 0.35);
            border-left: 5px solid #ef4444;
            border-radius: 12px;
            padding: 22px;
            margin-top: 20px;
        }

        .threat-title {
            font-size: 1.25rem;
            font-weight: 700;
            color: #ffffff;
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 14px;
        }

        .action-list-item {
            display: flex;
            align-items: flex-start;
            gap: 10px;
            padding: 6px 0;
            color: #cbd5e1;
            font-size: 0.88rem;
        }

        /* ── Pipeline Architecture Flow Boxes ── */
        .pipeline-container {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 8px;
            overflow-x: auto;
            padding: 14px 6px;
            margin-bottom: 24px;
        }

        .pipeline-step {
            background: rgba(15, 23, 42, 0.8);
            border: 1px solid var(--border-glass);
            border-radius: 10px;
            padding: 14px 16px;
            flex: 1;
            min-width: 140px;
            text-align: center;
            transition: all 0.2s ease;
        }

        .pipeline-step.active {
            border-color: var(--accent-cyan);
            background: rgba(0, 240, 255, 0.08);
            box-shadow: 0 0 16px rgba(0, 240, 255, 0.15);
        }

        .pipeline-step-title {
            font-size: 0.88rem;
            font-weight: 700;
            color: #ffffff;
            margin-bottom: 4px;
        }

        .pipeline-step-desc {
            font-size: 0.72rem;
            color: var(--text-muted);
        }

        .pipeline-arrow {
            color: var(--accent-cyan);
            font-size: 1.2rem;
            font-weight: bold;
        }
    </style>
    """, unsafe_allow_html=True)

def render_header(title: str, subtitle: str, badge_text: str = "GAN MODULE"):
    """Render a dynamic hero header on each page."""
    st.markdown(f"""
    <div class="page-hero-banner">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
            <div>
                <div class="page-hero-title">
                    <span class="pulse-indicator"></span>
                    <span>{title}</span>
                </div>
                <div class="page-hero-sub">{subtitle}</div>
            </div>
            <div class="cyber-badge badge-critical" style="border-color: rgba(56, 189, 248, 0.4); color: #38bdf8; background: rgba(14, 165, 233, 0.15);">
                🛡️ {badge_text}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_kpi_metrics(metrics: List[Dict[str, str]]):
    """
    Render horizontal grid of KPI metric boxes.
    Each item: {'title': str, 'value': str, 'sub': str, 'color': 'cyan'|'red'|'green'|'amber'}
    """
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        c = m.get("color", "cyan")
        col.markdown(f"""
        <div class="kpi-metric-box">
            <div class="kpi-title">{m['title']}</div>
            <div class="kpi-val {c}">{m['value']}</div>
            <div class="kpi-sub">{m.get('sub', '')}</div>
        </div>
        """, unsafe_allow_html=True)

def render_threat_report(
    packet_id: int,
    attack_type: str,
    confidence: float,
    ae_error: float,
    ae_thresh: float,
    top_features: Dict[str, float]
):
    """Render a high-severity Threat Intelligence Alert Card for SOC analysts."""
    sev_info = SEVERITY_MAP.get(attack_type, ('HIGH', '🟠', 'high'))
    severity_label = sev_info[0]
    severity_badge_class = f"badge-{sev_info[2]}"
    actions = ACTIONS_MAP.get(attack_type, [
        "Quarantine host from subnet immediately.",
        "Inspect firewall flow telemetry and block anomalous ingress IPs."
    ])

    st.markdown(f"""
    <div class="threat-alert-panel">
        <div class="threat-title">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 1.5rem;">⚠️</span>
                <span>CRITICAL INCIDENT ALERT — PACKET #{packet_id}</span>
            </div>
            <span class="cyber-badge {severity_badge_class}">
                {severity_label} SEVERITY
            </span>
        </div>
        
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; margin-bottom: 16px;">
            <div style="background: rgba(15, 23, 42, 0.6); padding: 10px 14px; border-radius: 8px;">
                <div style="font-size: 0.74rem; color: #94a3b8; text-transform: uppercase;">Classified Threat</div>
                <div style="font-size: 1.15rem; font-weight: 700; color: #ef4444;">{attack_type}</div>
            </div>
            <div style="background: rgba(15, 23, 42, 0.6); padding: 10px 14px; border-radius: 8px;">
                <div style="font-size: 0.74rem; color: #94a3b8; text-transform: uppercase;">Classifier Confidence</div>
                <div style="font-size: 1.15rem; font-weight: 700; color: #38bdf8;">{confidence * 100:.1f}%</div>
            </div>
            <div style="background: rgba(15, 23, 42, 0.6); padding: 10px 14px; border-radius: 8px;">
                <div style="font-size: 0.74rem; color: #94a3b8; text-transform: uppercase;">Autoencoder Reconstruction MSE</div>
                <div style="font-size: 1.15rem; font-weight: 700; color: {'#ef4444' if ae_error > ae_thresh else '#34d399'};">
                    {ae_error:.6f} <span style="font-size: 0.75rem; color: #64748b;">(Limit: {ae_thresh:.5f})</span>
                </div>
            </div>
        </div>

        <div style="margin-top: 14px;">
            <div style="font-size: 0.84rem; font-weight: 700; color: #f87171; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px;">
                🚨 Automated Incident Response Playbook:
            </div>
    """, unsafe_allow_html=True)

    for act in actions:
        st.markdown(f"""
        <div class="action-list-item">
            <span style="color: #ef4444; font-weight: bold;">⚡</span>
            <span>{act}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div></div>", unsafe_allow_html=True)

def render_technical_limitations_expander():
    """Render technical notes regarding WGAN Tanh output compression and pipeline bridging."""
    with st.expander("ℹ️  Technical Notes, Mathematical Constraints & Pipeline Architecture"):
        st.markdown(r"""
        ### 🧠 WGAN-GP Generator Output Range & Tanh Compression
        - **Generator Activation:** The final layer of the `ConditionalGenerator` is a `nn.Tanh()`, enforcing the output features into $[-1.0, 1.0]$.
        - **Scaler Space:** The model was trained in `gan_scaler` space (StandardScaler fit on attack flows, clipped to $[-5.0, 5.0]$).
        - **Observation:** Because the real training flows span $[-5.0, 5.0]$, the synthetic points generated through the Tanh activation are naturally bounded to $[-1.0, 1.0]$. This results in slightly compressed distribution tails when mapped back to the raw flow domain.
        - **No Modification Policy:** In strict adherence to project evaluation standards, the checkpoint weights and architecture remain pristine.

        ---
        ### 🔄 Multi-Module Scaler Bridging Pipeline
        Because each defensive subsystem was trained independently with its own feature scaling distribution:
        1. **WGAN Generation:** $z \sim \mathcal{N}(0, I) \rightarrow G(z, c) \in [-1, 1]$ (GAN Scaler Space).
        2. **Inverse Transformation:** $X_{\\text{raw}} = \\text{gan\\_scaler}^{-1}(G(z, c))$.
        3. **Detector Transformation:** $X_{\\text{detector}} = \\text{clip}\\left(\\text{scaler}_{\\text{detector}}(X_{\\text{raw}}), -5, 5\\right)$.
        4. **Inference Execution:**
           - **Autoencoder:** $\\text{MSE}(X, \\hat{X}) > \\tau = 0.004557 \\implies \\text{ATTACK}$.
           - **Transformer:** $\\text{Softmax}(\\text{Transformer}(X)) \\implies \\text{Class Probability Distribution}$.
        """)
