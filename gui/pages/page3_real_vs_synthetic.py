"""
CyberShield GAN GUI — Page 3: Real vs Synthetic Comparative Fidelity
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from gui.config import (
    REAL_TRAFFIC_CSV, WGAN_SYNTHETIC_CSV, CTGAN_SYNTHETIC_CSV,
    GAN_DASHBOARD_PNG, GAN_COMPARISON_CSV, FLOW_FEATURES_77
)
from gui.components import render_header, render_kpi_metrics
from gui.inference import (
    load_cached_csv, compute_pca_projection, compute_feature_ks_scores
)

def render():
    render_header(
        title="Real vs. Synthetic Distribution Fidelity",
        subtitle="Empirical statistical comparison of generated network traffic against authentic attack flows.",
        badge_text="STATISTICAL VALIDATION"
    )

    # Load datasets safely
    df_real, err_real = load_cached_csv(REAL_TRAFFIC_CSV, nrows=2500)
    df_wgan, err_wgan = load_cached_csv(WGAN_SYNTHETIC_CSV, nrows=2500)
    df_ctgan, err_ctgan = load_cached_csv(CTGAN_SYNTHETIC_CSV, nrows=2500)
    df_comp, err_comp = load_cached_csv(GAN_COMPARISON_CSV)

    if df_wgan is None and df_ctgan is None:
        st.error("⚠️ Neither WGAN nor CTGAN synthetic datasets could be located.")
        return

    # Benchmark summary metrics
    if df_comp is not None:
        st.markdown("""
        <div style="font-size: 1.05rem; font-weight: 700; color: #00f0ff; margin-bottom: 12px;">
            📊 Benchmark Statistical Distance Summary (From Saved Evaluation)
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(df_comp, use_container_width=True, hide_index=True)

    # ── Section 1: 2D PCA Dimensionality Reduction ──────────────────────────
    st.markdown("""
    <div style="font-size: 1.15rem; font-weight: 700; color: #ffffff; margin: 26px 0 10px 0;">
        🌌 2D PCA Latent Space Manifold Alignment
    </div>
    <p style="color: #94a3b8; font-size: 0.88rem;">
        Linear dimensionality reduction across all 77 flow dimensions demonstrates how closely synthetic clusters overlap real attack manifolds.
    </p>
    """, unsafe_allow_html=True)

    pca_df, ev1, ev2 = compute_pca_projection(df_real, df_wgan, df_ctgan, max_samples=300)

    if not pca_df.empty:
        color_map = {
            "Real Attacks": "#10b981",
            "WGAN-GP": "#f59e0b",
            "CTGAN": "#00f0ff"
        }
        fig_pca = px.scatter(
            pca_df,
            x="PC1",
            y="PC2",
            color="Origin",
            color_discrete_map=color_map,
            opacity=0.75,
            title=f"PCA Flow Projection (PC1: {ev1:.1f}% var, PC2: {ev2:.1f}% var)",
            template="plotly_dark",
            hover_data={"Origin": True, "PC1": ":.2f", "PC2": ":.2f"}
        )
        fig_pca.update_layout(
            paper_bgcolor="rgba(15, 23, 42, 0.8)",
            plot_bgcolor="rgba(8, 12, 20, 0.8)",
            margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_pca, use_container_width=True)
    else:
        st.info("Insufficient data to compute PCA projection.")

    # ── Section 2: Univariate Feature Distribution Histogram ────────────────
    st.markdown("""
    <div style="font-size: 1.15rem; font-weight: 700; color: #ffffff; margin: 30px 0 10px 0;">
        📈 Feature-by-Feature Distribution Overlay
    </div>
    """, unsafe_allow_html=True)

    default_features = [
        "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
        "Total Length of Fwd Packets", "Packet Length Mean", "Flow Packets/s", "Fwd Header Length"
    ]
    selectable_features = [f for f in default_features if f in df_wgan.columns]
    remaining_features = [f for f in FLOW_FEATURES_77 if f in df_wgan.columns and f not in selectable_features]
    all_options = selectable_features + remaining_features

    feature_col = st.selectbox(
        "Select Feature to Compare:",
        options=all_options,
        index=0,
        help="Select any of the 77 flow features to inspect probability density alignment."
    )

    if feature_col:
        fig_hist = go.Figure()

        # Real distribution
        if df_real is not None and feature_col in df_real.columns:
            r_vals = pd.to_numeric(df_real[feature_col], errors='coerce').dropna()
            fig_hist.add_trace(go.Histogram(
                x=r_vals,
                name="Real Attacks",
                marker_color="#10b981",
                opacity=0.6,
                nbinsx=40,
                histnorm="probability density"
            ))

        # WGAN distribution
        if df_wgan is not None and feature_col in df_wgan.columns:
            w_vals = pd.to_numeric(df_wgan[feature_col], errors='coerce').dropna()
            fig_hist.add_trace(go.Histogram(
                x=w_vals,
                name="WGAN-GP",
                marker_color="#f59e0b",
                opacity=0.6,
                nbinsx=40,
                histnorm="probability density"
            ))

        # CTGAN distribution
        if df_ctgan is not None and feature_col in df_ctgan.columns:
            c_vals = pd.to_numeric(df_ctgan[feature_col], errors='coerce').dropna()
            fig_hist.add_trace(go.Histogram(
                x=c_vals,
                name="CTGAN",
                marker_color="#00f0ff",
                opacity=0.6,
                nbinsx=40,
                histnorm="probability density"
            ))

        fig_hist.update_layout(
            barmode="overlay",
            title=f"Probability Density Histogram for '{feature_col}'",
            xaxis_title=feature_col,
            yaxis_title="Probability Density",
            template="plotly_dark",
            paper_bgcolor="rgba(15, 23, 42, 0.8)",
            plot_bgcolor="rgba(8, 12, 20, 0.8)",
            margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # ── Section 3: Kolmogorov-Smirnov Statistical Bar Chart ──────────────────
    if df_real is not None and df_wgan is not None:
        st.markdown("""
        <div style="font-size: 1.15rem; font-weight: 700; color: #ffffff; margin: 30px 0 10px 0;">
            📐 Kolmogorov-Smirnov Two-Sample Divergence
        </div>
        <p style="color: #94a3b8; font-size: 0.88rem;">
            A lower KS score signifies closer agreement with genuine intrusion flow behavior (0.0 = perfect match).
        </p>
        """, unsafe_allow_html=True)

        ks_df = compute_feature_ks_scores(df_real, df_wgan)
        if not ks_df.empty:
            top_ks = ks_df.head(12)
            fig_ks = px.bar(
                top_ks,
                x="KS_Score",
                y="Feature",
                orientation="h",
                color="KS_Score",
                color_continuous_scale="Tealgrn",
                title="Top 12 Flow Features with Lowest Statistical Divergence (WGAN vs. Real)",
                template="plotly_dark"
            )
            fig_ks.update_layout(
                yaxis=dict(autorange="reversed"),
                paper_bgcolor="rgba(15, 23, 42, 0.8)",
                plot_bgcolor="rgba(8, 12, 20, 0.8)",
                margin=dict(l=20, r=20, t=50, b=20)
            )
            st.plotly_chart(fig_ks, use_container_width=True)

    # ── Section 4: Saved Pre-Trained GAN Evaluation Dashboard ────────────────
    if os.path.exists(GAN_DASHBOARD_PNG):
        st.markdown("""
        <div style="font-size: 1.15rem; font-weight: 700; color: #ffffff; margin: 32px 0 12px 0;">
            🖼️ Pre-Computed GAN Diagnostic Dashboard
        </div>
        """, unsafe_allow_html=True)
        st.image(
            GAN_DASHBOARD_PNG,
            caption="CyberShield GAN Comprehensive Training & Statistical Fidelity Dashboard",
            use_container_width=True
        )
