"""
CyberShield GAN Multi-Page Streamlit Application
Entrypoint: streamlit run gui/app.py
"""

import os
import sys

# Ensure project root is in sys.path for robust imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
from gui.components import inject_cyber_theme, render_technical_limitations_expander
from gui.pages import (
    page1_overview,
    page2_generator,
    page3_real_vs_synthetic,
    page4_stress_test,
    page5_wgan_training,
    page6_diffusion
)

def main():
    # Page setup
    st.set_page_config(
        page_title="CyberShield — Generative AI Intrusion Defense",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Inject Cyberpunk / Navy CSS design system
    inject_cyber_theme()

    # ── Sidebar Navigation & Branding ───────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-brand-card">
            <div style="font-size: 2.2rem; filter: drop-shadow(0 0 10px #00f0ff);">🛡️</div>
            <div class="sidebar-brand-title">CyberShield</div>
            <div class="sidebar-brand-sub">Generative AI Defense Suite</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="font-size: 0.76rem; font-family: 'JetBrains Mono', monospace; text-transform: uppercase; color: #94a3b8; letter-spacing: 1px; margin-bottom: 8px;">
            Navigation Matrix
        </div>
        """, unsafe_allow_html=True)

        page_selection = st.radio(
            label="Select Module:",
            options=[
                "1. Overview & Pipeline",
                "2. GAN Generator",
                "3. Real vs Synthetic",
                "4. Stress Test (Key Demo)",
                "5. WGAN Training Dynamics",
                "6. Diffusion Text-to-Heatmap"
            ],
            index=0,
            label_visibility="collapsed"
        )

        st.markdown("<hr style='border: 1px solid rgba(56, 189, 248, 0.15); margin: 24px 0 16px 0;'>", unsafe_allow_html=True)

        # System Status Badges
        st.markdown("""
        <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(56, 189, 248, 0.2); border-radius: 8px; padding: 12px 14px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <span style="font-size: 0.75rem; color: #94a3b8;">Compute Device:</span>
                <span class="cyber-badge badge-safe">CPU Active</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <span style="font-size: 0.75rem; color: #94a3b8;">Dataset:</span>
                <span style="font-size: 0.75rem; color: #00f0ff; font-family: 'JetBrains Mono';">CICIDS 2017</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 0.75rem; color: #94a3b8;">Feature Space:</span>
                <span style="font-size: 0.75rem; color: #38bdf8; font-family: 'JetBrains Mono';">77 Flow Cols</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="text-align: center; margin-top: 24px; color: #64748b; font-size: 0.72rem;">
            CyberShield Final-Year Project<br>B.Tech CSE Cybersecurity
        </div>
        """, unsafe_allow_html=True)

    # ── Page Routing with Error Boundary ────────────────────────────────────
    try:
        if "1. Overview" in page_selection:
            page1_overview.render()
        elif "2. GAN Generator" in page_selection:
            page2_generator.render()
        elif "3. Real vs Synthetic" in page_selection:
            page3_real_vs_synthetic.render()
        elif "4. Stress Test" in page_selection:
            page4_stress_test.render()
        elif "5. WGAN Training" in page_selection:
            page5_wgan_training.render()
        elif "6. Diffusion" in page_selection:
            page6_diffusion.render()
    except Exception as e:
        st.error(f"❌ Unexpected error rendering page: {str(e)}")
        st.exception(e)

    # ── Global Technical Notes & Limitations Expander ───────────────────────
    st.markdown("<div style='margin-top: 40px;'></div>", unsafe_allow_html=True)
    render_technical_limitations_expander()

if __name__ == "__main__":
    main()
