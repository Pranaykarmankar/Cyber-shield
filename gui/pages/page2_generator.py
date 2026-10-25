"""
CyberShield GAN GUI — Page 2: Synthetic Attack Generator
"""

import time
import streamlit as st
import pandas as pd
import numpy as np

from gui.config import (
    CANONICAL_ATTACK_CLASSES, WGAN_SYNTHETIC_CSV, CTGAN_SYNTHETIC_CSV, FLOW_FEATURES_77
)
from gui.components import render_header, render_kpi_metrics
from gui.inference import (
    load_wgan_generator, load_ctgan_model, load_scalers_and_encoders,
    load_cached_csv, generate_wgan_samples, generate_ctgan_samples
)

def render():
    render_header(
        title="Interactive GAN Traffic Generator",
        subtitle="Conditionally synthesize realistic network attack flows on CPU using trained WGAN-GP and CTGAN architectures.",
        badge_text="LIVE SYNTHESIS ENGINE"
    )

    # Load required models and artifacts
    wgan_gen, wgan_err = load_wgan_generator()
    ctgan_model, ctgan_err = load_ctgan_model()
    artifacts, scaler_errs = load_scalers_and_encoders()
    ctgan_cached_df, _ = load_cached_csv(CTGAN_SYNTHETIC_CSV)

    if wgan_err and ctgan_err:
        st.error(f"⚠️ Generator models unavailable: {wgan_err} | {ctgan_err}")
        return

    # ── Interactive Controls Form ───────────────────────────────────────────
    st.markdown("""
    <div style="font-size: 1.05rem; font-weight: 700; color: #00f0ff; margin-bottom: 12px;">
        ⚙️ Generation Controls & Parameters
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns([1.2, 1.3, 1.2, 0.9])

    with col1:
        model_choice = st.selectbox(
            "Select Generative Model:",
            options=["WGAN-GP", "CTGAN", "Both (Comparative Batch)"],
            index=0,
            help="Choose between gradient-penalized Wasserstein GAN, Conditional Tabular GAN, or a joint batch."
        )

    with col2:
        selected_class = st.selectbox(
            "Target Attack Class:",
            options=CANONICAL_ATTACK_CLASSES,
            index=1,  # Default DDoS
            help="Condition the generator on this specific cyber intrusion signature."
        )

    with col3:
        n_samples = st.slider(
            "Number of Samples:",
            min_value=10,
            max_value=2000,
            value=100,
            step=10,
            help="Quantity of synthetic packets to generate."
        )

    with col4:
        seed = st.number_input(
            "Random Seed:",
            min_value=0,
            max_value=99999,
            value=42,
            step=1,
            help="Seed for deterministic, reproducible synthesis."
        )

    generate_clicked = st.button("⚡ Generate Synthetic Packets", type="primary", use_container_width=True)

    # Session State Storage for generated data
    if "latest_generated_df" not in st.session_state:
        st.session_state.latest_generated_df = None
    if "gen_metadata" not in st.session_state:
        st.session_state.gen_metadata = None

    if generate_clicked:
        with st.spinner("Synthesizing intrusion flows from latent space..."):
            t0 = time.time()
            dfs_to_concat = []
            warning_messages = []

            # 1. WGAN-GP Generation
            if model_choice in ["WGAN-GP", "Both (Comparative Batch)"]:
                if wgan_gen is not None and "gan_scaler" in artifacts and "gan_le" in artifacts:
                    wgan_df = generate_wgan_samples(
                        generator=wgan_gen,
                        gan_scaler=artifacts["gan_scaler"],
                        gan_le=artifacts["gan_le"],
                        target_class=selected_class,
                        n_samples=n_samples,
                        seed=seed
                    )
                    dfs_to_concat.append(wgan_df)
                else:
                    warning_messages.append("WGAN-GP model or scaler not loaded.")

            # 2. CTGAN Generation
            if model_choice in ["CTGAN", "Both (Comparative Batch)"]:
                ctgan_df, used_fallback = generate_ctgan_samples(
                    ctgan_model=ctgan_model,
                    fallback_df=ctgan_cached_df,
                    target_class=selected_class,
                    n_samples=n_samples,
                    seed=seed
                )
                if used_fallback:
                    warning_messages.append("CTGAN: Sampled from pre-computed high-fidelity synthetic attack dataset.")
                dfs_to_concat.append(ctgan_df)

            duration = time.time() - t0

            if dfs_to_concat:
                combined_df = pd.concat(dfs_to_concat, ignore_index=True)
                st.session_state.latest_generated_df = combined_df
                st.session_state.gen_metadata = {
                    "duration": duration,
                    "model": model_choice,
                    "target_class": selected_class,
                    "count": len(combined_df),
                    "warnings": warning_messages
                }
            else:
                st.error("Failed to generate synthetic data. Please verify model weights.")

    # ── Display Generated Outputs ───────────────────────────────────────────
    if st.session_state.latest_generated_df is not None:
        df_out = st.session_state.latest_generated_df
        meta = st.session_state.gen_metadata

        if meta.get("warnings"):
            for w in meta["warnings"]:
                st.info(f"ℹ️ {w}")

        # Summary KPIs
        metrics_display = [
            {"title": "Total Rows Generated", "value": f"{len(df_out):,}", "sub": f"Target: {meta['target_class']}", "color": "cyan"},
            {"title": "Feature Dimensionality", "value": "77 Flow Cols", "sub": "Standard CICIDS format", "color": "green"},
            {"title": "Synthesis Duration", "value": f"{meta['duration']*1000:.1f} ms", "sub": "Fast CPU Latency", "color": "cyan"},
            {"title": "Active Engine", "value": meta['model'], "sub": "Class-conditioned", "color": "amber"}
        ]
        render_kpi_metrics(metrics_display)

        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin: 20px 0 10px 0;">
            <div style="font-weight: 700; color: #ffffff; font-size: 1.1rem;">
                📋 Synthetic Packet Flow Preview (First 20 of {len(df_out):,} rows)
            </div>
            <div class="cyber-badge badge-safe">
                Label: {meta['target_class']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Show first 20 rows with generator column in front
        cols_display = ["Generator", "Label"] + [c for c in df_out.columns if c not in ["Generator", "Label"]]
        preview_df = df_out[cols_display].head(20)
        st.dataframe(preview_df, use_container_width=True, hide_index=False)

        # Download Button
        csv_data = df_out.to_csv(index=False)
        st.download_button(
            label=f"⬇️ Download Synthetic {meta['target_class']} Dataset (CSV)",
            data=csv_data,
            file_name=f"cybershield_synthetic_{meta['target_class'].replace(' ', '_').lower()}.csv",
            mime="text/csv",
            type="primary"
        )

        # Summary Statistics Expander
        with st.expander("📊 View Detailed Feature Statistical Summary (Mean, Std, Min, Max)"):
            num_cols = [c for c in FLOW_FEATURES_77 if c in df_out.columns]
            stats_df = df_out[num_cols].describe().T[["mean", "std", "min", "max"]]
            st.dataframe(stats_df, use_container_width=True)
    else:
        st.markdown("""
        <div class="cyber-card" style="text-align: center; padding: 40px 20px;">
            <span style="font-size: 2.5rem;">⚡</span>
            <h4 style="color: #94a3b8; margin: 10px 0 6px 0;">Ready to Synthesize Network Traffic</h4>
            <p style="color: #64748b; font-size: 0.88rem; max-width: 500px; margin: 0 auto;">
                Select your generative model, attack class, and number of samples above, then click <strong>Generate Synthetic Packets</strong> to trigger inference.
            </p>
        </div>
        """, unsafe_allow_html=True)
