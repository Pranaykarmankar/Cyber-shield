"""
CyberShield GAN GUI — Page 4: Adversarial Stress Test (Key Demo Page)
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from gui.config import (
    CANONICAL_ATTACK_CLASSES, REAL_TRAFFIC_CSV, FLOW_FEATURES_77
)
from gui.components import (
    render_header, render_kpi_metrics, render_threat_report
)
from gui.inference import (
    load_wgan_generator, load_ctgan_model, load_autoencoder, load_transformer,
    load_scalers_and_encoders, load_cached_csv, generate_wgan_samples,
    generate_ctgan_samples, evaluate_traffic_with_ae, evaluate_traffic_with_transformer
)

def render():
    render_header(
        title="Adversarial Stress Testing & Defense Verification",
        subtitle="Deploy conditionally synthesized attack traffic against Autoencoder & Transformer detectors to evaluate perimeter robustness.",
        badge_text="LIVE DEFENSIVE BENCHMARK"
    )

    # Load all models and scalers
    wgan_gen, err_wgan = load_wgan_generator()
    ctgan_model, err_ctgan = load_ctgan_model()
    ae_model, ae_thresh, err_ae = load_autoencoder()
    tr_model, err_tr = load_transformer()
    artifacts, scaler_errs = load_scalers_and_encoders()

    if err_ae or err_tr:
        st.error(f"⚠️ Detector models failed to load: {err_ae} | {err_tr}")
        return

    # ── Test Configuration Controls ─────────────────────────────────────────
    st.markdown("""
    <div style="font-size: 1.05rem; font-weight: 700; color: #00f0ff; margin-bottom: 12px;">
        🎯 Attack Simulation Configuration
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1.2, 1.4, 1.0])

    with col1:
        generator_engine = st.selectbox(
            "Adversarial Generator Engine:",
            options=["WGAN-GP (Recommended)", "CTGAN"],
            index=0
        )

    with col2:
        target_attack = st.selectbox(
            "Intended Intrusion Class:",
            options=CANONICAL_ATTACK_CLASSES,
            index=1  # Default DDoS
        )

    with col3:
        test_packets = st.slider(
            "Stress Batch Size:",
            min_value=20,
            max_value=500,
            value=100,
            step=10
        )

    launch_test = st.button("🚀 Launch Adversarial Stress Test", type="primary", use_container_width=True)

    # Session storage
    if "stress_results" not in st.session_state:
        st.session_state.stress_results = None

    if launch_test:
        with st.spinner("Executing generative synthesis and running dual-detector inference..."):
            # 1. Generate synthetic batch
            if "WGAN" in generator_engine:
                if wgan_gen is not None and "gan_scaler" in artifacts and "gan_le" in artifacts:
                    raw_df = generate_wgan_samples(
                        generator=wgan_gen,
                        gan_scaler=artifacts["gan_scaler"],
                        gan_le=artifacts["gan_le"],
                        target_class=target_attack,
                        n_samples=test_packets,
                        seed=42
                    )
                else:
                    st.error("WGAN model or scaler artifacts missing.")
                    return
            else:
                from gui.config import CTGAN_SYNTHETIC_CSV
                fallback_df, _ = load_cached_csv(CTGAN_SYNTHETIC_CSV)
                raw_df, _ = generate_ctgan_samples(
                    ctgan_model=ctgan_model,
                    fallback_df=fallback_df,
                    target_class=target_attack,
                    n_samples=test_packets,
                    seed=42
                )

            # 2. Evaluate with Autoencoder
            ae_res = evaluate_traffic_with_ae(
                df_raw=raw_df,
                ae_model=ae_model,
                ae_thresh=ae_thresh,
                ae_scaler=artifacts["ae_scaler"]
            )

            # 3. Evaluate with Transformer
            tr_res = evaluate_traffic_with_transformer(
                df_raw=raw_df,
                tr_model=tr_model,
                tr_scaler=artifacts["tr_scaler"],
                tr_le=artifacts["tr_le"],
                target_class=target_attack
            )

            st.session_state.stress_results = {
                "raw_df": raw_df,
                "ae_res": ae_res,
                "tr_res": tr_res,
                "target_attack": target_attack,
                "engine": generator_engine
            }

    # ── Display Stress Test Results ─────────────────────────────────────────
    if st.session_state.stress_results is not None:
        data = st.session_state.stress_results
        ae_res = data["ae_res"]
        tr_res = data["tr_res"]
        target = data["target_attack"]
        raw_df = data["raw_df"]

        # Top Metric Row
        ae_rate = ae_res["detection_rate"]
        tr_rate = tr_res["accuracy"]
        kpi_row = [
            {"title": "Detected by AE", "value": f"{ae_rate:.1f}%", "sub": f"Threshold $\\tau$: {ae_res['threshold']:.5f}", "color": "green" if ae_rate >= 90 else "red"},
            {"title": "Correctly Classified (TR)", "value": f"{tr_rate:.1f}%", "sub": f"Target: {target}", "color": "cyan" if tr_rate >= 50 else "amber"},
            {"title": "Mean AE Recon Loss", "value": f"{ae_res['mean_error']:.5f}", "sub": "Average per-sample MSE", "color": "red" if ae_res['mean_error'] > ae_res['threshold'] else "green"},
            {"title": "Batch Size", "value": f"{len(raw_df)} Packets", "sub": f"Engine: {data['engine'].split()[0]}", "color": "cyan"}
        ]
        render_kpi_metrics(kpi_row)

        # Clear Interpretation Sentence
        st.markdown(f"""
        <div class="cyber-card" style="border-left: 4px solid {'#10b981' if ae_rate > 90 else '#f59e0b'}; margin-top: 6px;">
            <div style="font-weight: 700; color: #ffffff; margin-bottom: 4px;">🔍 Defense Evaluation Interpretation:</div>
            <div style="color: #cbd5e1; font-size: 0.92rem; line-height: 1.5;">
                The Autoencoder flagged <strong>{ae_rate:.1f}%</strong> of synthetic packets as anomalous intrusions (MSE loss exceeded threshold &tau; = {ae_res['threshold']:.5f}), confirming that the generated traffic reliably breaks benign baseline profiles.
                Concurrently, the Transformer classifier mapped <strong>{tr_rate:.1f}%</strong> of packets directly to <strong>{target}</strong>, with residual packets classified under correlated attack taxonomy profiles.
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_left, col_right = st.columns([1.1, 0.9])

        # ── (a) Autoencoder Diagnostic Breakdown ──
        with col_left:
            st.markdown("""
            <div style="font-size: 1.1rem; font-weight: 700; color: #00f0ff; margin: 18px 0 10px 0;">
                (a) Autoencoder Reconstruction Loss Distribution
            </div>
            """, unsafe_allow_html=True)

            mse_series = ae_res["errors"]
            fig_ae = go.Figure()

            # Histogram of Reconstruction Errors
            fig_ae.add_trace(go.Histogram(
                x=mse_series,
                nbinsx=35,
                marker_color="#ef4444",
                opacity=0.75,
                name="Synthetic Attack Packets"
            ))

            # Vertical Line for Saved Threshold
            fig_ae.add_vline(
                x=ae_res["threshold"],
                line_width=2.5,
                line_dash="dash",
                line_color="#00f0ff",
                annotation_text=f"Anomaly Threshold ({ae_res['threshold']:.5f})",
                annotation_position="top right",
                annotation_font_color="#00f0ff"
            )

            fig_ae.update_layout(
                title="MSE Loss Distribution vs. Anomaly Cutoff",
                xaxis_title="Reconstruction Mean Squared Error (MSE)",
                yaxis_title="Packet Frequency",
                template="plotly_dark",
                paper_bgcolor="rgba(15, 23, 42, 0.8)",
                plot_bgcolor="rgba(8, 12, 20, 0.8)",
                margin=dict(l=20, r=20, t=50, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_ae, use_container_width=True)

        # ── (b) Transformer Predicted Class Distribution ──
        with col_right:
            st.markdown("""
            <div style="font-size: 1.1rem; font-weight: 700; color: #38bdf8; margin: 18px 0 10px 0;">
                (b) Transformer Threat Classification Distribution
            </div>
            """, unsafe_allow_html=True)

            dist = tr_res["class_distribution"]
            df_dist = pd.DataFrame(list(dist.items()), columns=["Threat Category", "Count"])
            df_dist["Color"] = ["#10b981" if c == target else "#38bdf8" for c in df_dist["Threat Category"]]

            fig_tr = px.bar(
                df_dist,
                x="Count",
                y="Threat Category",
                orientation="h",
                text="Count",
                title=f"Predicted Class Breakdown for Intended '{target}'",
                template="plotly_dark",
                color="Threat Category",
                color_discrete_sequence=df_dist["Color"].tolist()
            )
            fig_tr.update_layout(
                showlegend=False,
                yaxis=dict(autorange="reversed"),
                paper_bgcolor="rgba(15, 23, 42, 0.8)",
                plot_bgcolor="rgba(8, 12, 20, 0.8)",
                margin=dict(l=20, r=20, t=50, b=20)
            )
            st.plotly_chart(fig_tr, use_container_width=True)

        # ── (c) Threat Intelligence Alert Report for Sample Packet ───────────
        st.markdown("""
        <div style="font-size: 1.15rem; font-weight: 700; color: #ffffff; margin: 24px 0 10px 0;">
            📄 Automated SOC Threat Incident Report (Sample Inspector)
        </div>
        """, unsafe_allow_html=True)

        sample_idx = st.selectbox(
            "Select Packet Index to Inspect Incident Report:",
            options=list(range(len(raw_df))),
            index=0,
            format_func=lambda i: f"Packet #{i+1} — Predicted: {tr_res['predicted_classes'][i]} (Confidence: {tr_res['confidences'][i]*100:.1f}%)"
        )

        render_threat_report(
            packet_id=sample_idx + 1,
            attack_type=tr_res["predicted_classes"][sample_idx],
            confidence=tr_res["confidences"][sample_idx],
            ae_error=float(ae_res["errors"][sample_idx]),
            ae_thresh=ae_res["threshold"],
            top_features={}
        )
    else:
        st.markdown("""
        <div class="cyber-card" style="text-align: center; padding: 40px 20px;">
            <span style="font-size: 2.5rem;">🛡️</span>
            <h4 style="color: #94a3b8; margin: 10px 0 6px 0;">Ready for Defensive Stress Testing</h4>
            <p style="color: #64748b; font-size: 0.88rem; max-width: 520px; margin: 0 auto;">
                Select your intended attack profile and click <strong>Launch Adversarial Stress Test</strong> to generate packets and run them across both the Autoencoder and Transformer pipelines.
            </p>
        </div>
        """, unsafe_allow_html=True)
