# =========================================================================
# CyberShield — STREAMLIT GUI  (v3.0 — Ember Shield Dashboard)
# Network Intrusion Detection Dashboard
# Integrates: Autoencoder (AE) + Transformer (Attack Classifier)
#
# Run locally:
#   pip install streamlit torch scikit-learn pandas numpy joblib plotly
#   streamlit run cybershield_app.py
#
# Required files in same folder:
#   cybershield_ae.pth
#   ae_scaler.pkl
#   cybershield_transformer.pth
#   transformer_scaler.pkl
#   transformer_label_encoder.pkl
# =========================================================================

import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
import time
import os
import math

# ── Page Configuration ────────────────────────────────────────────────────
st.set_page_config(
    page_title="CyberShield — AI Intrusion Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Ember Shield Theme CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Import Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* ── Root Variables ── */
    :root {
        --bg-primary: #0b0a10;
        --bg-secondary: #12101a;
        --bg-card: rgba(18, 16, 26, 0.7);
        --bg-glass: rgba(18, 14, 24, 0.6);
        --border-glass: rgba(230, 57, 70, 0.15);
        --text-primary: #e2e8f0;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        --accent-crimson: #e63946;
        --accent-ember: #ff6b35;
        --accent-gold: #ffd166;
        --accent-green: #2ec4b6;
        --accent-red: #ef4444;
        --accent-orange: #f97316;
        --accent-yellow: #eab308;
        --gradient-primary: linear-gradient(135deg, #e63946, #ff6b35, #ffd166);
        --gradient-danger: linear-gradient(135deg, #ef4444, #f97316);
        --glow-crimson: 0 0 20px rgba(230, 57, 70, 0.3);
        --glow-ember: 0 0 20px rgba(255, 107, 53, 0.3);
        --glow-red: 0 0 20px rgba(239, 68, 68, 0.3);
    }

    /* ── Immersive Animations ── */
    @keyframes grid-drift {
        0%, 100% { opacity: 0.3; }
        50% { opacity: 0.6; }
    }
    @keyframes radar-pulse {
        0% { transform: scale(0.5); opacity: 0.8; }
        100% { transform: scale(2.5); opacity: 0; }
    }
    @keyframes glitch-flicker {
        0%, 92%, 100% { opacity: 0; }
        93% { opacity: 0.7; clip-path: inset(15% 0 65% 0); transform: translate(-3px); }
        95% { opacity: 0.5; clip-path: inset(55% 0 10% 0); transform: translate(3px); }
        97% { opacity: 0.6; clip-path: inset(30% 0 40% 0); transform: translate(-2px); }
    }
    @keyframes threat-pulse {
        0%, 100% { box-shadow: 0 0 5px rgba(239, 68, 68, 0.2); }
        50% { box-shadow: 0 0 20px rgba(239, 68, 68, 0.5), 0 0 40px rgba(239, 68, 68, 0.2); }
    }

    /* Radar Pulse (Hero) */
    .hero-radar-wrap {
        position: relative;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 120px; height: 120px;
    }
    .hero-radar-ring {
        position: absolute;
        top: 50%; left: 50%;
        margin: -30px 0 0 -30px;
        border-radius: 50%;
        border: 1px solid rgba(230, 57, 70, 0.3);
        width: 60px; height: 60px;
        animation: radar-pulse 3s ease-out infinite;
    }
    .hero-radar-ring.ring-2 { animation-delay: 1s; }
    .hero-radar-ring.ring-3 { animation-delay: 2s; }
    .hero-radar-wrap .hero-shield { position: relative; z-index: 1; }

    /* Glitch Effect (Title) */
    .hero-title { position: relative; }
    .hero-title::after {
        content: attr(data-text);
        position: absolute;
        top: 0; left: 0;
        width: 100%; height: 100%;
        background: linear-gradient(135deg, #ff6b35, #ffd166);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: glitch-flicker 6s ease-in-out infinite;
        pointer-events: none;
    }

    /* Threat Pulse (Critical) */
    .report-card.severity-critical { animation: threat-pulse 2s ease-in-out infinite; }
    .alert-banner.critical { animation: threat-pulse 2s ease-in-out infinite; }

    /* ── Global ── */
    .stApp {
        background: var(--bg-primary) !important;
        font-family: 'Inter', sans-serif !important;
    }
    .stApp::before {
        content: '';
        position: fixed;
        top: 0; left: 0;
        width: 100%; height: 100%;
        background-image: radial-gradient(rgba(230, 57, 70, 0.06) 1px, transparent 1px);
        background-size: 32px 32px;
        animation: grid-drift 8s ease-in-out infinite;
        pointer-events: none;
        z-index: 0;
    }

    .main .block-container {
        padding-top: 1rem !important;
        max-width: 1400px;
    }

    /* ── Hide default streamlit elements ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg-primary); }
    ::-webkit-scrollbar-thumb { background: var(--accent-crimson); border-radius: 3px; }

    /* ── Hero Header ── */
    .hero-container {
        text-align: center;
        padding: 2rem 1rem 1.5rem 1rem;
        position: relative;
    }

    .hero-shield {
        font-size: 3.2rem;
        display: inline-block;
        animation: float 3s ease-in-out infinite;
        filter: drop-shadow(0 0 25px rgba(230, 57, 70, 0.5));
    }

    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-8px); }
    }

    .hero-title {
        font-family: 'Inter', sans-serif;
        font-size: 2.8rem;
        font-weight: 800;
        background: var(--gradient-primary);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.5px;
        margin: 0.3rem 0 0.2rem 0;
        line-height: 1.2;
    }

    .hero-subtitle {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        color: var(--text-muted);
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }

    .hero-tagline {
        font-size: 0.95rem;
        color: var(--text-secondary);
        margin-top: 0.3rem;
    }

    .hero-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, var(--accent-crimson), var(--accent-ember), var(--accent-gold), transparent);
        margin: 1rem auto;
        max-width: 600px;
        border-radius: 1px;
    }

    /* ── Glass Card ── */
    .glass-card {
        background: var(--bg-glass);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid var(--border-glass);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    .glass-card:hover {
        border-color: rgba(230, 57, 70, 0.3);
        box-shadow: var(--glow-crimson);
    }

    /* ── Metric Cards ── */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin: 1rem 0;
    }
    @media (max-width: 768px) {
        .metric-grid { grid-template-columns: repeat(2, 1fr); }
    }

    .metric-card {
        background: var(--bg-glass);
        backdrop-filter: blur(12px);
        border: 1px solid var(--border-glass);
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: var(--glow-crimson);
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        border-radius: 14px 14px 0 0;
    }
    .metric-card.cyan::before { background: var(--gradient-primary); }
    .metric-card.green::before { background: linear-gradient(90deg, #2ec4b6, #5eead4); }
    .metric-card.red::before { background: var(--gradient-danger); }
    .metric-card.purple::before { background: linear-gradient(90deg, #ffd166, #ffe5a0); }

    .metric-icon { font-size: 1.6rem; margin-bottom: 0.4rem; }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: var(--text-primary);
        line-height: 1.2;
    }
    .metric-label {
        font-size: 0.78rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 0.3rem;
        font-weight: 500;
    }
    .metric-delta {
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 0.2rem;
        padding: 2px 8px;
        border-radius: 20px;
        display: inline-block;
    }
    .metric-delta.green { background: rgba(46, 196, 182,0.15); color: #5eead4; }
    .metric-delta.red { background: rgba(239,68,68,0.15); color: #f87171; }
    .metric-delta.yellow { background: rgba(234,179,8,0.15); color: #facc15; }
    .metric-delta.cyan { background: rgba(230, 57, 70,0.15); color: #ff8c61; }

    /* ── Section Headers ── */
    .section-header {
        font-family: 'Inter', sans-serif;
        font-size: 1.3rem;
        font-weight: 700;
        color: var(--text-primary);
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin: 1.5rem 0 0.8rem 0;
    }
    .section-header .icon {
        font-size: 1.2rem;
    }
    .section-header .line {
        flex: 1;
        height: 1px;
        background: linear-gradient(90deg, var(--border-glass), transparent);
    }

    /* ── Alert Banners ── */
    .alert-banner {
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin: 0.5rem 0;
        display: flex;
        align-items: center;
        gap: 0.8rem;
        font-weight: 500;
        backdrop-filter: blur(8px);
        border: 1px solid;
    }
    .alert-banner.critical {
        background: rgba(239, 68, 68, 0.1);
        border-color: rgba(239, 68, 68, 0.3);
        color: #fca5a5;
    }
    .alert-banner.high {
        background: rgba(249, 115, 22, 0.1);
        border-color: rgba(249, 115, 22, 0.3);
        color: #fdba74;
    }
    .alert-banner.medium {
        background: rgba(234, 179, 8, 0.1);
        border-color: rgba(234, 179, 8, 0.3);
        color: #fde047;
    }
    .alert-banner.safe {
        background: rgba(46, 196, 182, 0.1);
        border-color: rgba(46, 196, 182, 0.3);
        color: #99f6e4;
    }
    .alert-banner .alert-icon { font-size: 1.3rem; }

    /* ── Threat Report Card ── */
    .report-card {
        background: rgba(12, 10, 18, 0.9);
        border: 1px solid rgba(230, 57, 70, 0.15);
        border-radius: 14px;
        padding: 1.5rem;
        margin: 0.8rem 0;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        color: #ffd6a5;
        line-height: 1.7;
        position: relative;
        overflow: hidden;
    }
    .report-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
    }
    .report-card.severity-critical::before { background: var(--gradient-danger); }
    .report-card.severity-high::before { background: linear-gradient(90deg, #f97316, #fb923c); }
    .report-card.severity-medium::before { background: linear-gradient(90deg, #eab308, #facc15); }
    .report-card.severity-safe::before { background: linear-gradient(90deg, #2ec4b6, #5eead4); }

    .report-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-bottom: 0.8rem;
        border-bottom: 1px solid rgba(230, 57, 70, 0.1);
        margin-bottom: 0.8rem;
    }
    .report-title {
        font-size: 1rem;
        font-weight: 700;
        font-family: 'Inter', sans-serif;
    }
    .report-severity-badge {
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    .badge-critical { background: rgba(239,68,68,0.2); color: #fca5a5; border: 1px solid rgba(239,68,68,0.4); }
    .badge-high { background: rgba(249,115,22,0.2); color: #fdba74; border: 1px solid rgba(249,115,22,0.4); }
    .badge-medium { background: rgba(234,179,8,0.2); color: #fde047; border: 1px solid rgba(234,179,8,0.4); }
    .badge-safe { background: rgba(46, 196, 182,0.2); color: #99f6e4; border: 1px solid rgba(46, 196, 182,0.4); }

    .report-row {
        display: flex;
        gap: 0.5rem;
        padding: 0.15rem 0;
    }
    .report-label { color: #64748b; min-width: 130px; }
    .report-value { color: #e2e8f0; }
    .report-actions {
        margin-top: 0.8rem;
        padding-top: 0.8rem;
        border-top: 1px solid rgba(230, 57, 70, 0.1);
    }
    .report-action-item {
        padding: 0.2rem 0;
        color: #fbbf24;
    }

    /* ── Welcome Card ── */
    .welcome-card {
        background: var(--bg-glass);
        backdrop-filter: blur(16px);
        border: 1px solid var(--border-glass);
        border-radius: 20px;
        padding: 2.5rem;
        text-align: center;
        margin: 1rem 0;
    }
    .welcome-title {
        font-size: 1.6rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 1rem;
    }
    .welcome-desc {
        color: var(--text-secondary);
        font-size: 0.95rem;
        line-height: 1.7;
        max-width: 700px;
        margin: 0 auto 1.5rem auto;
    }

    .pipeline-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
        max-width: 800px;
        margin-left: auto;
        margin-right: auto;
    }
    .pipeline-step {
        background: rgba(18, 14, 24, 0.8);
        border: 1px solid var(--border-glass);
        border-radius: 14px;
        padding: 1.2rem 1rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    .pipeline-step:hover {
        border-color: var(--accent-crimson);
        transform: translateY(-2px);
        box-shadow: var(--glow-crimson);
    }
    .pipeline-step .step-num {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: var(--accent-crimson);
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }
    .pipeline-step .step-icon { font-size: 1.6rem; margin-bottom: 0.3rem; }
    .pipeline-step .step-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--text-primary);
    }
    .pipeline-step .step-desc {
        font-size: 0.75rem;
        color: var(--text-muted);
        margin-top: 0.2rem;
    }

    /* ── Model Badge ── */
    .model-badges {
        display: flex;
        justify-content: center;
        gap: 1rem;
        margin-top: 1.5rem;
        flex-wrap: wrap;
    }
    .model-badge {
        background: rgba(18, 14, 24, 0.8);
        border: 1px solid var(--border-glass);
        border-radius: 10px;
        padding: 0.6rem 1.2rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .model-badge .badge-dot {
        width: 8px; height: 8px;
        border-radius: 50%;
        display: inline-block;
    }
    .model-badge .badge-text {
        font-size: 0.8rem;
        color: var(--text-secondary);
    }
    .badge-dot.ae { background: var(--accent-crimson); box-shadow: 0 0 8px rgba(230, 57, 70,0.5); }
    .badge-dot.tf { background: var(--accent-gold); box-shadow: 0 0 8px rgba(255, 209, 102,0.5); }
    .badge-dot.ds { background: var(--accent-green); box-shadow: 0 0 8px rgba(46, 196, 182,0.5); }

    /* ── Scanning animation ── */
    .scan-overlay {
        text-align: center;
        padding: 3rem 1rem;
    }
    .scan-ring {
        width: 120px; height: 120px;
        margin: 0 auto 1.5rem auto;
        border-radius: 50%;
        border: 3px solid rgba(230, 57, 70, 0.15);
        border-top: 3px solid var(--accent-crimson);
        animation: spin 1s linear infinite;
        position: relative;
    }
    .scan-ring::after {
        content: '🛡️';
        position: absolute;
        top: 50%; left: 50%;
        transform: translate(-50%, -50%);
        font-size: 2.2rem;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    .scan-text {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
        color: var(--accent-crimson);
        letter-spacing: 1px;
    }
    .scan-sub {
        font-size: 0.8rem;
        color: var(--text-muted);
        margin-top: 0.3rem;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #12101a 0%, #0b0a10 100%) !important;
        border-right: 1px solid rgba(230, 57, 70, 0.1) !important;
    }
    section[data-testid="stSidebar"] .stRadio > label {
        color: var(--text-secondary) !important;
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown span {
        color: var(--text-secondary) !important;
    }

    .sidebar-brand {
        text-align: center;
        padding: 1.2rem 0 0.5rem 0;
    }
    .sidebar-logo {
        font-size: 2.5rem;
        filter: drop-shadow(0 0 15px rgba(230, 57, 70,0.5));
        animation: pulse-glow 2s ease-in-out infinite;
    }
    @keyframes pulse-glow {
        0%, 100% { filter: drop-shadow(0 0 15px rgba(230, 57, 70,0.5)); }
        50% { filter: drop-shadow(0 0 25px rgba(230, 57, 70,0.8)); }
    }
    .sidebar-title {
        font-family: 'Inter', sans-serif;
        font-size: 1.5rem;
        font-weight: 800;
        background: var(--gradient-primary);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-top: 0.3rem;
    }
    .sidebar-version {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: var(--text-muted);
        letter-spacing: 2px;
    }
    .sidebar-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--border-glass), transparent);
        margin: 1rem 0;
    }
    .sidebar-section-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: var(--accent-crimson);
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }

    /* ── Status indicator ── */
    .status-bar {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 0.8rem;
        background: rgba(46, 196, 182, 0.08);
        border: 1px solid rgba(46, 196, 182, 0.2);
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .status-dot {
        width: 8px; height: 8px;
        background: #2ec4b6;
        border-radius: 50%;
        animation: blink 1.5s ease-in-out infinite;
    }
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.3; }
    }
    .status-text {
        font-size: 0.78rem;
        color: #99f6e4;
        font-weight: 500;
    }

    /* ── Streamlit Tab Styling ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background: var(--bg-glass) !important;
        border: 1px solid var(--border-glass) !important;
        border-radius: 10px !important;
        color: var(--text-secondary) !important;
        font-weight: 500 !important;
        padding: 0.5rem 1.2rem !important;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(230, 57, 70, 0.12) !important;
        border-color: rgba(230, 57, 70, 0.4) !important;
        color: var(--accent-crimson) !important;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        display: none;
    }
    .stTabs [data-baseweb="tab-border"] {
        display: none;
    }

    /* ── Streamlit Button Override ── */
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="stBaseButton-primary"] {
        background: var(--gradient-primary) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.6rem 2rem !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.5px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(230, 57, 70, 0.3) !important;
    }
    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="stBaseButton-primary"]:hover {
        box-shadow: 0 6px 25px rgba(230, 57, 70, 0.5) !important;
        transform: translateY(-1px) !important;
    }

    .stDownloadButton > button {
        background: rgba(18, 14, 24, 0.8) !important;
        border: 1px solid var(--border-glass) !important;
        color: var(--text-secondary) !important;
        border-radius: 10px !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
    }
    .stDownloadButton > button:hover {
        border-color: var(--accent-crimson) !important;
        color: var(--accent-crimson) !important;
    }

    /* ── Dataframe ── */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }

    /* ── Slider override ── */
    .stSlider [data-testid="stThumbValue"] {
        color: var(--accent-crimson) !important;
    }

    /* ── Footer ── */
    .app-footer {
        text-align: center;
        padding: 2rem 0 1rem 0;
        border-top: 1px solid var(--border-glass);
        margin-top: 2rem;
    }
    .footer-text {
        font-size: 0.75rem;
        color: var(--text-muted);
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 1px;
    }
    .footer-tech {
        display: flex;
        justify-content: center;
        gap: 1.5rem;
        margin-top: 0.5rem;
        flex-wrap: wrap;
    }
    .footer-tech span {
        font-size: 0.7rem;
        color: var(--text-muted);
        padding: 3px 10px;
        border: 1px solid rgba(230, 57, 70,0.1);
        border-radius: 20px;
    }
</style>
""", unsafe_allow_html=True)


# ── Model Definitions ─────────────────────────────────────────────────────
# IMPORTANT: These architectures MUST match the trained model checkpoints
class TabularAutoencoder(nn.Module):
    def __init__(self, input_dim: int, bottleneck: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.LeakyReLU(0.2),
            nn.Linear(128, 64),        nn.BatchNorm1d(64),  nn.LeakyReLU(0.2),
            nn.Linear(64, 32),         nn.BatchNorm1d(32),  nn.LeakyReLU(0.2),
            nn.Linear(32, bottleneck),                       nn.LeakyReLU(0.2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 32), nn.LeakyReLU(0.2),
            nn.Linear(32, 64),  nn.BatchNorm1d(64),  nn.LeakyReLU(0.2),
            nn.Linear(64, 128), nn.BatchNorm1d(128), nn.LeakyReLU(0.2),
            nn.Linear(128, input_dim),
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))


class CyberTransformer(nn.Module):
    def __init__(self, input_dim, num_classes,
                 d_model=64, nhead=4, num_layers=3, dropout=0.1):
        super().__init__()
        self.input_dim       = input_dim
        self.feature_embedding = nn.Sequential(
            nn.Linear(1, d_model), nn.LayerNorm(d_model), nn.ReLU()
        )
        self.pos_embedding = nn.Parameter(
            torch.randn(1, input_dim, d_model) * 0.1
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model*4, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.classifier  = nn.Sequential(
            nn.Linear(d_model * input_dim, 512), nn.LayerNorm(512), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        B = x.size(0)
        x = x.unsqueeze(-1)
        x = self.feature_embedding(x) + self.pos_embedding
        x = self.transformer(x)
        return self.classifier(x.reshape(B, -1))


# ── Load Models ───────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    device = torch.device("cpu")

    # Resolve paths relative to this script
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Load AE
    ae_ckpt    = torch.load(os.path.join(base_dir, "cybershield_ae.pth"), map_location=device, weights_only=False)
    ae_scaler  = joblib.load(os.path.join(base_dir, "ae_scaler.pkl"))
    ae_model   = TabularAutoencoder(
        input_dim  = ae_ckpt['input_dim'],
        bottleneck = ae_ckpt.get('bottleneck', 16)
    )
    ae_model.load_state_dict(ae_ckpt['model_state_dict'])
    ae_model.eval()
    ae_threshold = ae_ckpt['threshold']

    # Load Transformer
    tr_ckpt    = torch.load(os.path.join(base_dir, "cybershield_transformer.pth"), map_location=device, weights_only=False)
    tr_scaler  = joblib.load(os.path.join(base_dir, "transformer_scaler.pkl"))
    tr_le      = joblib.load(os.path.join(base_dir, "transformer_label_encoder.pkl"))
    tr_model   = CyberTransformer(
        input_dim   = tr_ckpt['input_dim'],
        num_classes = tr_ckpt['num_classes'],
        d_model     = tr_ckpt.get('d_model', 64),
        nhead       = tr_ckpt.get('nhead', 4),
        num_layers  = tr_ckpt.get('num_layers', 3),
    )
    tr_model.load_state_dict(tr_ckpt['model_state_dict'])
    tr_model.eval()
    class_names = tr_ckpt['class_names']

    return (ae_model, ae_scaler, ae_threshold,
            tr_model, tr_scaler, tr_le, class_names, device)


@st.cache_resource
def load_vae_models():
    """Load the VAE pipeline: StandardScaler + LabelEncoder + Random Forest classifier."""
    base_dir = os.path.dirname(os.path.abspath(__file__))

    vae_scaler = joblib.load(os.path.join(base_dir, "scaler.pkl"))
    vae_le     = joblib.load(os.path.join(base_dir, "label_encoder.pkl"))
    rf_model   = joblib.load(os.path.join(base_dir, "rf_model.pkl"))
    vae_class_names = list(vae_le.classes_)

    return rf_model, vae_scaler, vae_le, vae_class_names


# ── Severity & Actions Map ────────────────────────────────────────────────
SEVERITY_MAP = {
    'BENIGN'                    : ('NONE',     '✅', 'safe'),
    'DDoS'                      : ('CRITICAL', '🔴', 'critical'),
    'DoS Hulk'                  : ('HIGH',     '🟠', 'high'),
    'DoS GoldenEye'             : ('HIGH',     '🟠', 'high'),
    'DoS slowloris'             : ('MEDIUM',   '🟡', 'medium'),
    'DoS Slowhttptest'          : ('MEDIUM',   '🟡', 'medium'),
    'PortScan'                  : ('MEDIUM',   '🟡', 'medium'),
    'Bot'                       : ('HIGH',     '🟠', 'high'),
    'FTP-Patator'               : ('HIGH',     '🟠', 'high'),
    'SSH-Patator'               : ('HIGH',     '🟠', 'high'),
    'Web Attack \u2013 Brute Force'  : ('HIGH',     '🟠', 'high'),
    'Web Attack \u2013 XSS'          : ('MEDIUM',   '🟡', 'medium'),
    'Web Attack \u2013 Sql Injection': ('CRITICAL', '🔴', 'critical'),
    'Infiltration'              : ('CRITICAL', '🔴', 'critical'),
    'Heartbleed'                : ('CRITICAL', '🔴', 'critical'),
}

SEVERITY_COLOR = {
    'CRITICAL': '#ef4444',
    'HIGH':     '#f97316',
    'MEDIUM':   '#eab308',
    'NONE':     '#2ec4b6',
}

ACTIONS_MAP = {
    'BENIGN'                    : ['No action required — traffic is normal.'],
    'DDoS'                      : ['Block source IPs immediately', 'Enable DDoS protection', 'Alert security team', 'Activate rate limiting'],
    'DoS Hulk'                  : ['Throttle HTTP requests', 'Enable connection limits', 'Monitor server resources'],
    'DoS GoldenEye'             : ['Block keep-alive connections', 'Apply IP rate limiting', 'Restart affected services'],
    'DoS slowloris'             : ['Increase min data rate threshold', 'Set connection timeout values', 'Use reverse proxy'],
    'DoS Slowhttptest'          : ['Enforce transfer speed limits', 'Enable HTTP request timeout', 'Filter at load balancer'],
    'PortScan'                  : ['Block scanning IP in firewall', 'Close unnecessary open ports', 'Enable IDS port sweep alerts'],
    'Bot'                       : ['Isolate infected host', 'Run malware scan', 'Block C&C server IPs', 'Reset credentials'],
    'FTP-Patator'               : ['Lock FTP accounts after 3 failures', 'Switch to SFTP', 'Block attacking IP'],
    'SSH-Patator'               : ['Block attacking IP', 'Enable key-based auth only', 'Disable SSH password login'],
    'Web Attack \u2013 Brute Force'  : ['Enable CAPTCHA on login', 'Lock after failed attempts', 'Block originating IP'],
    'Web Attack \u2013 XSS'          : ['Sanitize all user inputs', 'Implement CSP headers', 'Patch affected web pages'],
    'Web Attack \u2013 Sql Injection': ['Patch vulnerable queries immediately', 'Use parameterized queries', 'Audit database access logs', 'Notify compliance team'],
    'Infiltration'              : ['Isolate affected hosts', 'Conduct forensic investigation', 'Reset all credentials', 'Notify incident response'],
    'Heartbleed'                : ['Patch OpenSSL immediately', 'Revoke and reissue SSL certs', 'Reset all passwords', 'Audit for leaked data'],
}

# ── Attack Intelligence Knowledge Base ────────────────────────────────────
ATTACK_KNOWLEDGE_BASE = {
    'DDoS': {
        'full_name': 'Distributed Denial of Service',
        'category': 'Volumetric / Network Layer Attack',
        'cvss_score': 9.1,
        'mitre_id': 'T1498',
        'description': (
            'A Distributed Denial of Service (DDoS) attack overwhelms a target server, network, or service '
            'by flooding it with massive volumes of traffic from multiple compromised systems (botnets). '
            'The goal is to exhaust server resources—CPU, memory, bandwidth—rendering the service unavailable '
            'to legitimate users. DDoS attacks can generate traffic volumes exceeding hundreds of Gbps, '
            'making them one of the most disruptive cyber threats.'
        ),
        'how_it_works': [
            'The attacker assembles a botnet of thousands of compromised machines (zombies).',
            'A command-and-control (C&C) server instructs all bots to send requests simultaneously.',
            'The target is flooded with SYN packets, UDP datagrams, HTTP requests, or amplified DNS responses.',
            'Server resources are exhausted, causing legitimate traffic to be dropped or severely delayed.',
            'The attack may use amplification techniques (DNS, NTP, SSDP) to multiply traffic volume by 50-100x.',
        ],
        'indicators_of_compromise': [
            'Sudden spike in inbound traffic volume (10x-1000x normal baseline)',
            'Unusually high number of connections from geographically dispersed IPs',
            'Server CPU and memory utilization sustained at 95-100%',
            'Increased packet drop rate and connection timeouts',
            'Network bandwidth saturation on upstream links',
            'Unusual traffic patterns: identical packet sizes, fixed intervals, spoofed source IPs',
        ],
        'impact': [
            'Complete service unavailability for customers and internal users',
            'Revenue loss estimated at $20,000-$100,000+ per hour for e-commerce',
            'Reputational damage and customer trust erosion',
            'Potential SLA violations and contractual penalties',
            'Cascading failures in dependent microservices and databases',
            'Diversion tactic: DDoS may mask simultaneous data exfiltration attempts',
        ],
        'immediate_actions': [
            'Activate DDoS mitigation service (Cloudflare, AWS Shield, Akamai)',
            'Enable rate limiting on edge routers and load balancers',
            'Block top offending source IP ranges using firewall ACLs',
            'Scale up infrastructure: add server instances behind load balancer',
            'Engage ISP upstream filtering to drop malicious traffic before it reaches your network',
            'Alert the security operations center (SOC) and initiate incident response',
        ],
        'prevention_measures': [
            'Deploy a Web Application Firewall (WAF) with DDoS rulesets',
            'Use a Content Delivery Network (CDN) to absorb and distribute traffic',
            'Implement network-level rate limiting and connection throttling',
            'Configure SYN cookies and TCP connection timeouts on servers',
            'Deploy Intrusion Prevention Systems (IPS) with DDoS signatures',
            'Establish traffic baselines and anomaly detection alerts',
            'Maintain an incident response playbook specific to DDoS scenarios',
            'Conduct regular DDoS simulation exercises (red team / tabletop)',
            'Use Anycast routing to distribute attack traffic across multiple PoPs',
            'Ensure DNS infrastructure is resilient (multiple providers, long TTLs)',
        ],
        'long_term_strategy': [
            'Invest in cloud-based DDoS protection with auto-scaling capabilities',
            'Implement network segmentation to limit blast radius',
            'Establish relationships with ISP and law enforcement for rapid response',
            'Review and update DDoS response plan quarterly',
            'Monitor threat intelligence feeds for emerging DDoS techniques',
        ],
        'references': [
            'MITRE ATT&CK T1498 — Network Denial of Service',
            'NIST SP 800-61 Rev. 2 — Computer Security Incident Handling Guide',
            'OWASP — Denial of Service Prevention Cheat Sheet',
            'US-CERT — Understanding Denial-of-Service Attacks',
        ],
    },
    'DoS Hulk': {
        'full_name': 'HTTP Unbearable Load King (HULK) DoS',
        'category': 'Application Layer (L7) Denial of Service',
        'cvss_score': 7.5,
        'mitre_id': 'T1499.002',
        'description': (
            'HULK (HTTP Unbearable Load King) is an application-layer denial-of-service tool that generates '
            'unique, obfuscated HTTP GET/POST requests to bypass caching mechanisms and CDN protections. '
            'Unlike volumetric DDoS, HULK targets the web application layer directly, generating requests '
            'that appear legitimate but are designed to consume maximum server-side processing resources.'
        ),
        'how_it_works': [
            'HULK generates HTTP requests with randomized URL parameters, headers, and user-agent strings.',
            'Each request is unique, defeating signature-based detection and caching layers.',
            'The tool uses keep-alive connections to maintain persistent sessions, draining server thread pools.',
            'Randomized referrers and query strings ensure requests bypass CDN caches and hit the origin server.',
            'The attack can be launched from a single machine but generates thousands of requests per second.',
        ],
        'indicators_of_compromise': [
            'Abnormally high HTTP request rate from limited source IPs',
            'Requests with highly randomized URL parameters and query strings',
            'Diverse User-Agent strings that cycle through a known HULK dictionary',
            'Server thread pool exhaustion and increasing response times',
            'Web server access logs showing unusual parameter patterns',
        ],
        'impact': [
            'Web application becomes slow or completely unresponsive',
            'Backend database may be overwhelmed by excessive query load',
            'Server thread/worker pool exhaustion blocks all legitimate requests',
            'Potential memory leaks from unclosed connections',
        ],
        'immediate_actions': [
            'Enable HTTP request rate limiting per source IP (e.g., 100 req/min)',
            'Block source IPs exhibiting HULK patterns in WAF',
            'Increase web server worker/thread pool limits temporarily',
            'Enable connection timeouts and keep-alive limits',
            'Monitor server resource utilization and scale if needed',
        ],
        'prevention_measures': [
            'Deploy a WAF with behavioral analysis (not just signature-based)',
            'Implement CAPTCHA or JavaScript challenges for suspicious traffic',
            'Use reverse proxy (Nginx/HAProxy) with rate limiting and connection caps',
            'Configure web server to limit concurrent connections per IP',
            'Enable request body size limits and URL length restrictions',
            'Implement application-level request throttling and queuing',
            'Monitor and alert on unusual request patterns using SIEM',
            'Use CDN with bot mitigation capabilities',
        ],
        'long_term_strategy': [
            'Implement auto-scaling to handle traffic spikes',
            'Deploy machine learning-based anomaly detection for L7 attacks',
            'Regular load testing to understand application breaking points',
        ],
        'references': [
            'MITRE ATT&CK T1499.002 — Service Exhaustion Flood',
            'OWASP — HTTP Flood Attack',
        ],
    },
    'DoS GoldenEye': {
        'full_name': 'GoldenEye HTTP DoS',
        'category': 'Application Layer (L7) Denial of Service',
        'cvss_score': 7.5,
        'mitre_id': 'T1499.002',
        'description': (
            'GoldenEye is a Python-based HTTP denial-of-service tool that uses HTTP Keep-Alive with '
            'no-cache directives to force the server to process every request freshly. It opens multiple '
            'persistent connections and sends partial HTTP requests, keeping server resources occupied '
            'indefinitely without completing the transaction.'
        ),
        'how_it_works': [
            'Opens hundreds of concurrent HTTP connections using Keep-Alive headers.',
            'Sends requests with Cache-Control: no-cache to bypass server-side caching.',
            'Uses randomized URLs and user-agents to evade pattern detection.',
            'Maintains connections open by sending partial headers at slow intervals.',
            'Gradually exhausts the web server connection pool and worker threads.',
        ],
        'indicators_of_compromise': [
            'High number of concurrent Keep-Alive connections from few IPs',
            'Requests consistently containing Cache-Control: no-cache headers',
            'Slow connection completion rates with many half-open connections',
            'Server connection pool nearing capacity with legitimate traffic dropping',
        ],
        'impact': [
            'Web server connection pool exhaustion',
            'All legitimate users unable to establish new connections',
            'Backend services starved of resources',
        ],
        'immediate_actions': [
            'Kill long-lived idle Keep-Alive connections',
            'Reduce Keep-Alive timeout to 5-10 seconds',
            'Apply IP-based rate limiting on concurrent connections',
            'Restart web server processes if connection pool is deadlocked',
        ],
        'prevention_measures': [
            'Set aggressive Keep-Alive timeouts (5-15 seconds)',
            'Limit maximum concurrent connections per source IP',
            'Use reverse proxy to terminate and manage HTTP connections',
            'Implement connection queuing with priority for returning users',
            'Deploy WAF with slow-HTTP attack detection rules',
            'Monitor connection pool utilization with alerts at 80% threshold',
        ],
        'long_term_strategy': [
            'Architect applications for graceful degradation under load',
            'Implement circuit breakers and bulkhead patterns',
        ],
        'references': ['MITRE ATT&CK T1499.002', 'CWE-400 — Uncontrolled Resource Consumption'],
    },
    'DoS slowloris': {
        'full_name': 'Slowloris Low-Bandwidth DoS',
        'category': 'Slow-Rate Application Layer (L7) Attack',
        'cvss_score': 6.8,
        'mitre_id': 'T1499.001',
        'description': (
            'Slowloris is a low-bandwidth denial-of-service attack that holds connections open by sending '
            'partial HTTP headers at very slow rates. It exploits the web server behavior of waiting for '
            'complete headers before processing a request, tying up all available connection slots with '
            'minimal bandwidth usage. A single attacker machine can take down an Apache server.'
        ),
        'how_it_works': [
            'Opens many connections to the target web server.',
            'Sends a partial HTTP request header (e.g., "GET / HTTP/1.1\r\n").',
            'Periodically sends additional header lines to keep the connection alive.',
            'Never sends the final "\r\n\r\n" to complete the header, so the server keeps waiting.',
            'Server connection slots fill up, blocking all new legitimate connections.',
        ],
        'indicators_of_compromise': [
            'Many connections in ESTABLISHED state but with no data exchange',
            'Connections with incomplete HTTP headers persisting for minutes',
            'Server max-connections limit reached with low bandwidth utilization',
            'Increasing connection queue length with timeouts',
        ],
        'impact': ['Web server becomes unreachable despite low network bandwidth usage',
                   'Difficult to detect since traffic volume is minimal',
                   'Affects only the targeted service, not the entire network'],
        'immediate_actions': [
            'Set minimum data rate thresholds (e.g., RequestReadTimeout in Apache)',
            'Reduce header timeout to 10-20 seconds',
            'Increase max connections temporarily while applying mitigations',
        ],
        'prevention_measures': [
            'Use web servers resistant to Slowloris (Nginx, IIS) instead of Apache',
            'Configure RequestReadTimeout directive in Apache: header=10-20 body=10-20',
            'Deploy reverse proxy (Nginx/HAProxy) in front of application servers',
            'Set minimum transfer rate requirements for all connections',
            'Implement connection timeout policies for incomplete requests',
            'Use mod_qos or mod_evasive Apache modules',
        ],
        'long_term_strategy': ['Migrate to event-driven servers (Nginx) that handle Slowloris natively'],
        'references': ['MITRE ATT&CK T1499.001 — OS Exhaustion Flood', 'CVE-2007-6750'],
    },
    'DoS Slowhttptest': {
        'full_name': 'Slow HTTP Test (Slow POST/READ)',
        'category': 'Slow-Rate Application Layer (L7) Attack',
        'cvss_score': 6.8,
        'mitre_id': 'T1499.001',
        'description': (
            'SlowHTTPTest implements multiple slow HTTP attack variants: Slow POST (sending body data '
            'one byte at a time), Slow READ (reading response very slowly), and Apache Range header attack. '
            'These attacks exploit server patience for slow clients, consuming connection resources.'
        ),
        'how_it_works': [
            'Slow POST: Sends Content-Length header with large value, then transmits body data at ~1 byte/sec.',
            'Slow READ: Sends legitimate request but reads the response at an extremely slow rate.',
            'Server keeps connection and resources allocated waiting for the slow client to finish.',
            'Combines multiple techniques for maximum effectiveness.',
        ],
        'indicators_of_compromise': [
            'Extremely low data transfer rates on many simultaneous connections',
            'Large POST requests that take minutes to complete',
            'Connection durations far exceeding normal session times',
        ],
        'impact': ['Same as Slowloris — connection pool exhaustion with minimal bandwidth'],
        'immediate_actions': [
            'Enforce minimum transfer speed limits (e.g., 500 bytes/sec minimum)',
            'Enable HTTP request body timeout (10-30 seconds)',
            'Configure load balancer to close slow connections',
        ],
        'prevention_measures': [
            'Set request body timeout and minimum data rate on web server',
            'Use reverse proxy with buffering to absorb slow connections',
            'Deploy WAF with slow-HTTP detection capabilities',
            'Limit POST request body size to expected maximums',
            'Implement server-side request queuing with deadlines',
        ],
        'long_term_strategy': ['Adopt timeout-aware architectures'],
        'references': ['MITRE ATT&CK T1499.001', 'OWASP Slow HTTP Attack'],
    },
    'PortScan': {
        'full_name': 'Network Port Scanning / Reconnaissance',
        'category': 'Reconnaissance / Discovery',
        'cvss_score': 5.3,
        'mitre_id': 'T1046',
        'description': (
            'Port scanning is a reconnaissance technique used to discover open ports and running services '
            'on target hosts. Attackers use tools like Nmap to probe TCP/UDP ports, identify service versions, '
            'and map the network topology. While not directly harmful, port scanning is almost always the '
            'precursor to a more targeted attack.'
        ),
        'how_it_works': [
            'Attacker sends SYN packets (TCP) or UDP datagrams to a range of ports on the target.',
            'Open ports respond with SYN-ACK (TCP) or application data (UDP).',
            'Closed ports respond with RST (TCP) or ICMP unreachable (UDP).',
            'Service version detection sends specific probes to identify software and versions.',
            'OS fingerprinting analyzes response characteristics to determine the operating system.',
        ],
        'indicators_of_compromise': [
            'Single source IP connecting to many different ports in rapid succession',
            'Sequential or randomized port probing patterns',
            'SYN packets without completing the TCP handshake (SYN scan)',
            'Unusual connection attempts to uncommon ports (e.g., 4444, 5555, 31337)',
            'IDS/IPS alerts for port sweep or network scan signatures',
        ],
        'impact': [
            'Reveals network topology and attack surface to adversaries',
            'Identifies vulnerable services and unpatched software versions',
            'Precursor to exploitation — 85% of successful breaches start with reconnaissance',
        ],
        'immediate_actions': [
            'Block the scanning IP in firewall immediately',
            'Review and close all unnecessary open ports',
            'Check for any exploitation attempts following the scan',
            'Enable IDS port sweep detection alerts',
        ],
        'prevention_measures': [
            'Close all unnecessary ports — minimize the attack surface',
            'Use host-based firewalls (iptables, Windows Firewall) in addition to network firewalls',
            'Implement port knocking for administrative services',
            'Deploy honeypots to detect and divert scanning activity',
            'Use network segmentation to limit scan visibility',
            'Configure IDS/IPS with port scan detection rules',
            'Hide service banners and version information',
            'Regularly audit open ports with internal scanning tools',
        ],
        'long_term_strategy': [
            'Implement zero-trust network architecture',
            'Regular vulnerability assessments and penetration testing',
            'Network micro-segmentation',
        ],
        'references': ['MITRE ATT&CK T1046 — Network Service Discovery', 'NIST SP 800-115'],
    },
    'Bot': {
        'full_name': 'Botnet Command & Control Traffic',
        'category': 'Command and Control / Malware',
        'cvss_score': 8.5,
        'mitre_id': 'T1071',
        'description': (
            'Bot traffic indicates that a host on the network has been compromised and is communicating '
            'with a Command & Control (C&C) server. The infected machine (bot/zombie) receives instructions '
            'to perform malicious activities: DDoS attacks, spam distribution, cryptocurrency mining, '
            'data exfiltration, or lateral movement within the network.'
        ),
        'how_it_works': [
            'Malware infects a host via phishing, drive-by download, or exploitation of vulnerabilities.',
            'The malware establishes a persistent connection to a C&C server (HTTP, IRC, P2P, or DNS tunneling).',
            'The bot receives commands: scan the network, exfiltrate data, launch attacks, download payloads.',
            'Communication is often encrypted and uses domain generation algorithms (DGA) to evade blocking.',
            'The bot may propagate laterally to other hosts on the network.',
        ],
        'indicators_of_compromise': [
            'Periodic beaconing to external IPs at regular intervals (e.g., every 60 seconds)',
            'DNS queries to algorithmically generated domain names (DGA patterns)',
            'Outbound connections to known C&C IP addresses or domains',
            'Unusual outbound traffic at odd hours (nights, weekends)',
            'Host running unknown processes with network connections',
            'Registry modifications and persistence mechanisms on the host',
        ],
        'impact': [
            'Complete compromise of the infected host — full attacker control',
            'Data exfiltration of sensitive information (credentials, PII, financial data)',
            'Lateral movement risk — infection can spread to other network hosts',
            'Legal liability if bot is used for attacking external targets',
            'Cryptocurrency mining consuming compute resources and electricity',
        ],
        'immediate_actions': [
            'Isolate the infected host from the network immediately (quarantine VLAN)',
            'Block C&C server IPs and domains at firewall and DNS level',
            'Run full malware scan with updated signatures (EDR preferred)',
            'Capture memory dump and disk image for forensic analysis',
            'Reset all credentials that were accessible from the compromised host',
            'Notify the incident response team',
        ],
        'prevention_measures': [
            'Deploy Endpoint Detection and Response (EDR) on all hosts',
            'Implement DNS sinkholing for known malicious domains',
            'Use threat intelligence feeds to block known C&C infrastructure',
            'Enable application whitelisting to prevent unauthorized executables',
            'Segment the network to limit lateral movement potential',
            'Conduct regular phishing awareness training for employees',
            'Keep all software patched and up to date',
            'Monitor outbound traffic for beaconing patterns using SIEM',
            'Implement email filtering with attachment sandboxing',
        ],
        'long_term_strategy': [
            'Adopt zero-trust security model with continuous verification',
            'Implement network detection and response (NDR) for encrypted traffic analysis',
            'Regular threat hunting exercises',
        ],
        'references': ['MITRE ATT&CK T1071 — Application Layer Protocol', 'NIST Cybersecurity Framework'],
    },
    'FTP-Patator': {
        'full_name': 'FTP Brute Force (Patator)',
        'category': 'Credential Access / Brute Force',
        'cvss_score': 7.3,
        'mitre_id': 'T1110.001',
        'description': (
            'FTP-Patator is a brute-force attack against FTP (File Transfer Protocol) services using the '
            'Patator tool. The attacker systematically tries username/password combinations from wordlists '
            'to gain unauthorized file system access. FTP is particularly vulnerable because it transmits '
            'credentials in plaintext and many deployments use weak or default passwords.'
        ),
        'how_it_works': [
            'Attacker identifies an open FTP service (port 21) via port scanning.',
            'Patator tool is configured with username and password wordlists.',
            'The tool attempts thousands of login combinations per minute.',
            'Successful login grants read/write access to the FTP directory.',
            'Attacker may upload malware, download sensitive files, or pivot further.',
        ],
        'indicators_of_compromise': [
            'Extremely high rate of FTP authentication failures from a single IP',
            'Sequential or dictionary-pattern usernames in login attempts',
            'FTP service logs showing thousands of failed logins in minutes',
            'Successful login following a burst of failures (compromise indicator)',
        ],
        'impact': [
            'Unauthorized access to files on the FTP server',
            'Potential upload of malware or web shells',
            'Data theft of sensitive documents',
            'Credential reuse — compromised FTP passwords may work on other services',
        ],
        'immediate_actions': [
            'Lock FTP accounts after 3-5 consecutive failures',
            'Block the attacking IP address immediately',
            'Review FTP logs for any successful unauthorized logins',
            'If compromised: change all FTP passwords, audit uploaded files',
        ],
        'prevention_measures': [
            'Replace FTP with SFTP or FTPS (encrypted alternatives)',
            'Implement account lockout after failed login attempts',
            'Use strong, unique passwords and enforce password complexity policies',
            'Restrict FTP access to specific IP ranges using firewall rules',
            'Enable multi-factor authentication where possible',
            'Disable anonymous FTP access unless explicitly required',
            'Use fail2ban or similar tools to auto-block brute force IPs',
            'Monitor FTP authentication logs for anomalous patterns',
        ],
        'long_term_strategy': [
            'Migrate all file transfers to secure protocols (SFTP, SCP)',
            'Implement certificate-based authentication',
            'Use a centralized identity management system',
        ],
        'references': ['MITRE ATT&CK T1110.001 — Password Guessing', 'CWE-307 — Improper Restriction of Excessive Authentication Attempts'],
    },
    'SSH-Patator': {
        'full_name': 'SSH Brute Force (Patator)',
        'category': 'Credential Access / Brute Force',
        'cvss_score': 8.1,
        'mitre_id': 'T1110.001',
        'description': (
            'SSH-Patator is a brute-force attack against SSH (Secure Shell) services. The attacker uses '
            'automated tools to try thousands of username/password combinations against the SSH daemon. '
            'A successful SSH compromise is particularly dangerous as it grants full shell access to the server, '
            'often with elevated privileges.'
        ),
        'how_it_works': [
            'Attacker identifies SSH service (port 22) on the target via scanning.',
            'Patator or Hydra tool is loaded with credential wordlists.',
            'Automated login attempts are made at high speed (100-1000+ attempts/min).',
            'If successful, attacker gains interactive shell access to the server.',
            'From there: install rootkits, pivot to other hosts, exfiltrate data.',
        ],
        'indicators_of_compromise': [
            'Flood of SSH authentication failures in /var/log/auth.log',
            'Login attempts with common usernames (root, admin, ubuntu, test)',
            'Source IPs from known brute-force networks or Tor exit nodes',
            'Successful login from an unexpected IP following failed attempts',
        ],
        'impact': [
            'Full server compromise with shell access',
            'Rootkit installation and persistent backdoor access',
            'Lateral movement to other servers using stolen SSH keys',
            'Complete data exfiltration capability',
        ],
        'immediate_actions': [
            'Block the attacking IP at the firewall',
            'Disable SSH password authentication — switch to key-based only',
            'Review /var/log/auth.log for successful unauthorized logins',
            'If compromised: rotate all SSH keys, audit for backdoors',
        ],
        'prevention_measures': [
            'Use SSH key-based authentication exclusively — disable password login',
            'Change SSH port from default 22 to a non-standard port',
            'Implement fail2ban with aggressive thresholds (ban after 3 failures)',
            'Use AllowUsers/AllowGroups directives to restrict SSH access',
            'Disable root login via SSH (PermitRootLogin no)',
            'Deploy port knocking or VPN-only SSH access',
            'Enable two-factor authentication for SSH (Google Authenticator)',
            'Use SSH certificates instead of static keys for large environments',
        ],
        'long_term_strategy': [
            'Implement a bastion/jump host architecture',
            'Use SSH certificate authorities for centralized key management',
            'Regular rotation of SSH keys',
        ],
        'references': ['MITRE ATT&CK T1110.001', 'CIS Benchmark for SSH'],
    },
    'Web Attack \u2013 Brute Force': {
        'full_name': 'Web Application Brute Force Attack',
        'category': 'Credential Access / Web Application Attack',
        'cvss_score': 7.3,
        'mitre_id': 'T1110',
        'description': (
            'A web application brute force attack targets login forms, API endpoints, or authentication '
            'mechanisms by systematically trying username/password combinations. Attackers use tools like '
            'Burp Suite, Hydra, or custom scripts to automate credential stuffing and password spraying.'
        ),
        'how_it_works': [
            'Attacker identifies login pages or authentication API endpoints.',
            'Automated tool submits login requests with credentials from breach databases.',
            'Credential stuffing uses known email/password pairs from other data breaches.',
            'Password spraying tries common passwords against many usernames.',
            'Rate limiting and CAPTCHA evasion techniques may be employed.',
        ],
        'indicators_of_compromise': [
            'High volume of POST requests to login endpoints from single/few IPs',
            'Many HTTP 401/403 responses followed by a 200 (successful compromise)',
            'Login attempts using emails from known data breach lists',
            'Requests from headless browsers or automated tools (unusual User-Agents)',
        ],
        'impact': ['Account takeover', 'Data breach via compromised accounts', 'Privilege escalation'],
        'immediate_actions': [
            'Enable CAPTCHA on login pages immediately',
            'Lock accounts after 5 failed login attempts',
            'Block the attacking IP range',
            'Force password reset for any potentially compromised accounts',
        ],
        'prevention_measures': [
            'Implement CAPTCHA or reCAPTCHA on all login forms',
            'Enforce account lockout policies (5 attempts, 15-minute lockout)',
            'Deploy rate limiting on authentication endpoints',
            'Implement multi-factor authentication (MFA) for all users',
            'Use credential breach monitoring services (Have I Been Pwned API)',
            'Enforce strong password policies (12+ chars, complexity requirements)',
            'Implement login anomaly detection (new device, location, time)',
            'Use progressive delays between failed login attempts',
        ],
        'long_term_strategy': ['Adopt passwordless authentication (FIDO2/WebAuthn)', 'Implement adaptive MFA'],
        'references': ['MITRE ATT&CK T1110', 'OWASP Authentication Cheat Sheet'],
    },
    'Web Attack \u2013 XSS': {
        'full_name': 'Cross-Site Scripting (XSS)',
        'category': 'Web Application Attack / Injection',
        'cvss_score': 6.5,
        'mitre_id': 'T1059.007',
        'description': (
            'Cross-Site Scripting (XSS) injects malicious JavaScript code into web pages viewed by other users. '
            'The injected script executes in the victim browser context, allowing the attacker to steal session '
            'cookies, redirect users to phishing sites, deface web pages, or perform actions on behalf of the victim. '
            'XSS is consistently ranked in the OWASP Top 10 web application vulnerabilities.'
        ),
        'how_it_works': [
            'Stored XSS: Malicious script is permanently stored on the server (e.g., in a comment, profile).',
            'Reflected XSS: Script is embedded in a URL and reflected back in the server response.',
            'DOM-based XSS: Script manipulates the DOM directly without server involvement.',
            'The injected JavaScript executes in the victim browser with full page access.',
            'Attacker can steal cookies (document.cookie), capture keystrokes, or modify page content.',
        ],
        'indicators_of_compromise': [
            'HTTP requests containing script tags or JavaScript event handlers in parameters',
            'URL parameters with encoded <script> tags or javascript: URIs',
            'Unusual characters in form submissions: <, >, ", \', (, )',
            'WAF alerts for XSS pattern matches',
        ],
        'impact': [
            'Session hijacking via stolen cookies',
            'Credential theft through fake login forms',
            'Malware distribution to site visitors',
            'Defacement of web pages',
            'Compliance violations (PCI DSS, GDPR)',
        ],
        'immediate_actions': [
            'Sanitize and validate all user inputs on affected pages',
            'Implement Content Security Policy (CSP) headers',
            'Review and patch the vulnerable code path',
            'Invalidate all active sessions if stored XSS is confirmed',
        ],
        'prevention_measures': [
            'Implement input validation and output encoding on all user-supplied data',
            'Deploy Content Security Policy (CSP) headers to prevent inline script execution',
            'Use HTTP-only and Secure flags on all session cookies',
            'Implement a Web Application Firewall (WAF) with XSS rulesets',
            'Use modern frameworks with built-in XSS protection (React, Angular)',
            'Conduct regular code reviews focusing on output encoding',
            'Perform automated DAST scanning (OWASP ZAP, Burp Suite)',
            'Implement Subresource Integrity (SRI) for external scripts',
        ],
        'long_term_strategy': ['Adopt secure coding standards', 'Regular security training for developers'],
        'references': ['OWASP XSS Prevention Cheat Sheet', 'CWE-79', 'MITRE ATT&CK T1059.007'],
    },
    'Web Attack \u2013 Sql Injection': {
        'full_name': 'SQL Injection (SQLi)',
        'category': 'Web Application Attack / Injection',
        'cvss_score': 9.8,
        'mitre_id': 'T1190',
        'description': (
            'SQL Injection is a critical vulnerability that allows attackers to inject malicious SQL code into '
            'application queries through user input fields. Successful exploitation can lead to complete database '
            'compromise, including unauthorized data access, modification, deletion, and in some cases, '
            'remote code execution on the database server. SQLi remains the #1 web application vulnerability.'
        ),
        'how_it_works': [
            'Attacker identifies input fields that are incorporated into SQL queries without proper sanitization.',
            'Malicious SQL fragments are injected: \' OR 1=1 --, UNION SELECT, etc.',
            'The database engine executes the injected SQL as part of the legitimate query.',
            'Union-based SQLi extracts data from other tables.',
            'Blind SQLi infers data through true/false responses or time delays.',
            'In severe cases, xp_cmdshell or similar functions enable OS command execution.',
        ],
        'indicators_of_compromise': [
            'HTTP requests containing SQL keywords: UNION, SELECT, DROP, INSERT, --, \', OR 1=1',
            'Unusual database errors returned in HTTP responses (error-based SQLi)',
            'Abnormally long request processing times (time-based blind SQLi)',
            'WAF alerts for SQL injection pattern matches',
            'Database audit logs showing unauthorized queries or schema enumeration',
        ],
        'impact': [
            'Complete database compromise — read, modify, delete any data',
            'Theft of all user credentials, PII, financial records',
            'Authentication bypass — login as any user including admin',
            'Remote code execution on the database server',
            'Regulatory penalties (GDPR: up to 4% of annual revenue, PCI DSS fines)',
            'Massive reputational damage and customer notification requirements',
        ],
        'immediate_actions': [
            'Patch the vulnerable query IMMEDIATELY — use parameterized queries',
            'Take the affected application endpoint offline if necessary',
            'Audit database access logs for unauthorized data retrieval',
            'Notify the compliance/legal team — potential data breach',
            'Check for database backdoors (new users, stored procedures)',
            'Initiate incident response procedure',
        ],
        'prevention_measures': [
            'Use parameterized queries / prepared statements for ALL database interactions',
            'Implement ORM (Object-Relational Mapping) frameworks instead of raw SQL',
            'Deploy a WAF with SQL injection detection and blocking rules',
            'Apply principle of least privilege to database accounts',
            'Disable unnecessary database features (xp_cmdshell, LOAD_FILE)',
            'Implement input validation with whitelisting (not blacklisting)',
            'Conduct regular static code analysis (SAST) and dynamic testing (DAST)',
            'Use stored procedures with parameterized inputs',
            'Enable database audit logging for all queries',
            'Regular penetration testing with focus on injection vulnerabilities',
        ],
        'long_term_strategy': [
            'Adopt secure SDLC practices with mandatory code review for data access',
            'Implement runtime application self-protection (RASP)',
            'Database activity monitoring (DAM) for real-time query analysis',
        ],
        'references': ['OWASP SQL Injection Prevention', 'CWE-89', 'MITRE ATT&CK T1190', 'PCI DSS Requirement 6'],
    },
    'Infiltration': {
        'full_name': 'Network Infiltration / Advanced Persistent Threat',
        'category': 'Lateral Movement / Exfiltration',
        'cvss_score': 9.5,
        'mitre_id': 'T1071 / T1041',
        'description': (
            'Infiltration attacks represent advanced, multi-stage intrusions where an attacker has already gained '
            'initial access and is moving laterally through the network, escalating privileges, and exfiltrating '
            'data. This is characteristic of Advanced Persistent Threats (APTs) and indicates a serious, '
            'ongoing compromise that requires immediate incident response.'
        ),
        'how_it_works': [
            'Initial access gained through phishing, vulnerability exploitation, or supply chain compromise.',
            'Attacker establishes persistence (scheduled tasks, registry keys, rootkits).',
            'Lateral movement using stolen credentials, pass-the-hash, or exploitation.',
            'Privilege escalation to domain admin or root level.',
            'Data discovery and staging for exfiltration.',
            'Data exfiltrated via encrypted channels, DNS tunneling, or cloud storage.',
        ],
        'indicators_of_compromise': [
            'Unusual internal network traffic patterns (east-west traffic spikes)',
            'Authentication from service accounts at unusual times',
            'Large data transfers to external destinations',
            'New scheduled tasks or services on multiple hosts',
            'PowerShell execution with encoded commands',
            'Lateral movement tools detected (Mimikatz, PsExec, Cobalt Strike)',
        ],
        'impact': [
            'Complete network compromise with persistent attacker access',
            'Massive data exfiltration of intellectual property and sensitive data',
            'Supply chain contamination if attacker modifies software builds',
            'Extended dwell time (average 200+ days before detection)',
            'Recovery costs averaging $4.45 million per breach (IBM 2023)',
        ],
        'immediate_actions': [
            'Isolate ALL affected hosts immediately — do not alert the attacker',
            'Engage professional incident response team (internal or external DFIR)',
            'Preserve forensic evidence — memory dumps, disk images, network captures',
            'Reset ALL domain credentials — assume complete credential compromise',
            'Review all outbound connections for data exfiltration indicators',
            'Notify executive leadership and legal counsel',
        ],
        'prevention_measures': [
            'Implement network micro-segmentation to limit lateral movement',
            'Deploy EDR on all endpoints with behavioral detection',
            'Enable comprehensive logging (DNS, DHCP, authentication, file access)',
            'Implement privileged access management (PAM) with just-in-time access',
            'Use network detection and response (NDR) for encrypted traffic analysis',
            'Conduct regular threat hunting exercises',
            'Implement data loss prevention (DLP) controls at network boundaries',
            'Deploy deception technology (honeypots, honey tokens, honey credentials)',
        ],
        'long_term_strategy': [
            'Adopt zero-trust architecture across the entire organization',
            'Implement a Security Operations Center (SOC) with 24/7 monitoring',
            'Regular red team exercises simulating APT scenarios',
            'Threat intelligence integration for proactive defense',
        ],
        'references': ['MITRE ATT&CK Framework — Full Kill Chain', 'NIST SP 800-61', 'IBM Cost of a Data Breach Report'],
    },
    'Heartbleed': {
        'full_name': 'Heartbleed (CVE-2014-0160)',
        'category': 'Cryptographic Vulnerability Exploitation',
        'cvss_score': 9.4,
        'mitre_id': 'T1190',
        'description': (
            'Heartbleed is a critical vulnerability in the OpenSSL cryptographic library (CVE-2014-0160) that '
            'allows attackers to read up to 64KB of server memory per request by exploiting the TLS heartbeat '
            'extension. Leaked memory can contain private keys, session tokens, passwords, and other sensitive '
            'data. This is one of the most impactful vulnerabilities ever discovered, affecting ~17% of all '
            'SSL-enabled servers at the time of disclosure.'
        ),
        'how_it_works': [
            'The TLS heartbeat extension allows clients to send a payload and request an echo.',
            'The client specifies a payload length in the heartbeat request.',
            'Vulnerable OpenSSL versions do not validate the actual payload length.',
            'Attacker sends a small payload but claims a length of 64KB.',
            'Server responds with 64KB of its process memory, leaking sensitive data.',
            'Attack is repeatable and leaves no trace in standard server logs.',
        ],
        'indicators_of_compromise': [
            'Heartbeat requests with mismatched payload length fields',
            'Unusual TLS heartbeat frequency from external sources',
            'IDS/IPS alerts for Heartbleed signature matches',
            'Detection of leaked private key material in network traffic',
        ],
        'impact': [
            'Private SSL/TLS key compromise — enables decryption of all traffic',
            'Session token theft — attacker can hijack any active user session',
            'Password exposure from server memory',
            'No logging — exploitation is undetectable by standard monitoring',
            'Requires full SSL certificate revocation and reissuance',
        ],
        'immediate_actions': [
            'Patch OpenSSL to version 1.0.1g or later IMMEDIATELY',
            'Revoke and reissue ALL SSL/TLS certificates on affected servers',
            'Invalidate all active sessions and force re-authentication',
            'Reset ALL user passwords as memory may have been leaked',
            'Audit for any signs of private key misuse',
        ],
        'prevention_measures': [
            'Maintain OpenSSL at the latest patched version at all times',
            'Implement automated vulnerability scanning for known CVEs',
            'Use Perfect Forward Secrecy (PFS) cipher suites to limit key compromise impact',
            'Deploy IDS/IPS with Heartbleed detection signatures',
            'Implement certificate transparency monitoring',
            'Use hardware security modules (HSMs) to protect private keys',
            'Enable automatic security updates for critical packages',
            'Conduct regular SSL/TLS configuration audits (SSL Labs, testssl.sh)',
        ],
        'long_term_strategy': [
            'Adopt a formal vulnerability management program with SLA for critical patches',
            'Implement certificate automation (Let\'s Encrypt, ACME protocol)',
            'Consider memory-safe TLS implementations (BoringSSL, LibreSSL)',
        ],
        'references': ['CVE-2014-0160', 'MITRE ATT&CK T1190', 'heartbleed.com', 'RFC 6520 — TLS Heartbeat Extension'],
    },
    'BENIGN': {
        'full_name': 'Normal / Benign Traffic',
        'category': 'No Threat',
        'cvss_score': 0.0,
        'mitre_id': 'N/A',
        'description': 'Traffic classified as normal, legitimate network activity with no indicators of malicious behavior.',
        'how_it_works': [],
        'indicators_of_compromise': [],
        'impact': [],
        'immediate_actions': ['No action required — traffic is normal.'],
        'prevention_measures': ['Continue monitoring with CyberShield.'],
        'long_term_strategy': [],
        'references': [],
    },
}

# Default entry for unknown attacks
_DEFAULT_ATTACK_INTEL = {
    'full_name': 'Unknown Attack Type',
    'category': 'Uncategorized',
    'cvss_score': 5.0,
    'mitre_id': 'N/A',
    'description': 'An attack type not yet profiled in the CyberShield knowledge base. Investigate manually.',
    'how_it_works': ['Unknown attack vector — manual analysis required.'],
    'indicators_of_compromise': ['Anomalous network traffic patterns detected by AI models.'],
    'impact': ['Potential security compromise — severity unknown.'],
    'immediate_actions': ['Investigate traffic patterns', 'Monitor for further anomalies', 'Consult threat intelligence'],
    'prevention_measures': ['Enable comprehensive network monitoring', 'Keep IDS/IPS signatures updated'],
    'long_term_strategy': ['Regular security audits'],
    'references': ['MITRE ATT&CK Framework'],
}


# ── Prediction Functions ──────────────────────────────────────────────────
def run_ae(model, scaler, threshold, features: np.ndarray, device):
    x_scaled = np.clip(scaler.transform(features), -5, 5)
    tensor   = torch.FloatTensor(x_scaled)
    with torch.no_grad():
        recon  = model(tensor)
        errors = torch.mean((tensor - recon) ** 2, dim=1).numpy()
    preds = (errors > threshold).astype(int)
    return errors, preds


def run_transformer(model, scaler, le, class_names, features: np.ndarray, device):
    x_scaled = np.clip(scaler.transform(features), -5, 5)
    tensor   = torch.FloatTensor(x_scaled)
    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1).numpy()
        preds  = logits.argmax(dim=1).numpy()
    pred_classes = [class_names[p] for p in preds]
    confidences  = [probs[i, preds[i]] for i in range(len(preds))]
    return pred_classes, confidences, probs


def run_vae_pipeline(rf_model, scaler, le, class_names, features: np.ndarray):
    """Run the VAE pipeline: scale features → Random Forest classification."""
    x_scaled = scaler.transform(features)
    probs    = rf_model.predict_proba(x_scaled)
    preds    = rf_model.predict(x_scaled)
    pred_classes = [class_names[int(p)] for p in preds]
    confidences  = [float(probs[i, int(preds[i])]) for i in range(len(preds))]
    # Use a simple anomaly heuristic: BENIGN = normal, everything else = attack
    ae_preds = np.array([0 if c == 'BENIGN' else 1 for c in pred_classes])
    # Compute a pseudo reconstruction error based on max non-benign probability
    benign_idx = class_names.index('BENIGN') if 'BENIGN' in class_names else 0
    recon_errors = np.array([1.0 - float(probs[i, benign_idx]) for i in range(len(preds))])
    return pred_classes, confidences, probs, ae_preds, recon_errors


# ── Report Generator ──────────────────────────────────────────────────────
def generate_report_text(attack_class, confidence, packet_idx=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")
    severity, icon, _ = SEVERITY_MAP.get(attack_class, ('UNKNOWN', '⚠️', 'medium'))
    actions = ACTIONS_MAP.get(attack_class, ['Investigate and monitor'])
    action_text = "\n".join(f"  [{i+1}] {a}" for i, a in enumerate(actions))

    if attack_class == 'BENIGN':
        return (
            f"CYBERSHIELD — TRAFFIC REPORT\n"
            f"{'='*48}\n"
            f"Timestamp   : {timestamp}\n"
            f"Status      : ✅ NORMAL TRAFFIC — No threat detected\n"
            f"Confidence  : {confidence*100:.2f}%\n"
            f"Action      : No action required.\n"
            f"{'='*48}"
        )

    return (
        f"CYBERSHIELD — THREAT ALERT REPORT\n"
        f"{'='*48}\n"
        f"⚠️  ATTACK DETECTED\n\n"
        f"Timestamp   : {timestamp}\n"
        f"Attack Type : {attack_class}\n"
        f"Severity    : {icon} {severity}\n"
        f"Confidence  : {confidence*100:.2f}%\n"
        f"Packet      : {packet_idx if packet_idx is not None else 'N/A'}\n\n"
        f"RECOMMENDED ACTIONS:\n{action_text}\n\n"
        f"Generated by CyberShield AI v3.0 Ember Shield\n"
        f"{'='*48}"
    )

# ── Detailed HTML Report Generator ────────────────────────────────────────
def generate_detailed_html_report(attack_class, confidence, packet_idx=None,
                                   recon_error=None, all_probs=None, class_names=None):
    """Generate a comprehensive, stylish HTML threat report with embedded charts."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")
    date_str = datetime.now().strftime("%B %d, %Y")
    severity, icon, css_cls = SEVERITY_MAP.get(attack_class, ('UNKNOWN', '⚠️', 'medium'))
    actions = ACTIONS_MAP.get(attack_class, ['Investigate and monitor'])
    intel = ATTACK_KNOWLEDGE_BASE.get(attack_class, _DEFAULT_ATTACK_INTEL)

    # Severity colors
    sev_colors = {
        'CRITICAL': ('#ef4444', '#fca5a5', '#1a0505'),
        'HIGH': ('#f97316', '#fdba74', '#1a0d05'),
        'MEDIUM': ('#eab308', '#fde047', '#1a1505'),
        'NONE': ('#2ec4b6', '#99f6e4', '#051a17'),
    }
    sev_main, sev_light, sev_bg = sev_colors.get(severity, ('#64748b', '#94a3b8', '#111'))
    conf_pct = confidence * 100
    risk_level = "CRITICAL" if conf_pct > 90 else "HIGH" if conf_pct > 70 else "MODERATE" if conf_pct > 50 else "LOW"

    # Build probability chart SVG if we have class probabilities
    prob_chart_html = ""
    if all_probs is not None and class_names is not None and packet_idx is not None:
        idx = packet_idx - 1 if packet_idx else 0
        if idx < len(all_probs):
            probs = all_probs[idx]
            sorted_indices = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:8]
            bar_items = ""
            for rank, i in enumerate(sorted_indices):
                pct = probs[i] * 100
                name = class_names[i] if i < len(class_names) else f"Class {i}"
                bar_color = sev_main if name == attack_class else '#334155'
                bar_items += f"""
                <div style="display:flex;align-items:center;gap:10px;margin:6px 0;">
                    <div style="width:140px;font-size:11px;color:#94a3b8;text-align:right;flex-shrink:0;
                                overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{name}</div>
                    <div style="flex:1;background:#1e1b2e;border-radius:4px;height:22px;position:relative;overflow:hidden;">
                        <div style="width:{pct}%;height:100%;background:{bar_color};border-radius:4px;
                                    transition:width 0.5s;"></div>
                    </div>
                    <div style="width:50px;font-size:11px;color:#e2e8f0;font-weight:600;">{pct:.1f}%</div>
                </div>"""
            prob_chart_html = f"""
            <div style="background:#12101a;border:1px solid rgba(230,57,70,0.15);border-radius:12px;padding:20px;margin:20px 0;">
                <h3 style="color:#e2e8f0;font-size:16px;margin:0 0 15px 0;">📊 Classification Probability Distribution</h3>
                <p style="color:#64748b;font-size:12px;margin:0 0 12px 0;">Top 8 class probabilities from the AI model</p>
                {bar_items}
            </div>"""

    # Build reconstruction error gauge
    recon_gauge_html = ""
    if recon_error is not None:
        gauge_pct = min(recon_error * 1000, 100)  # Scale for visual
        recon_gauge_html = f"""
        <div style="background:#12101a;border:1px solid rgba(230,57,70,0.15);border-radius:12px;padding:20px;margin:20px 0;">
            <h3 style="color:#e2e8f0;font-size:16px;margin:0 0 10px 0;">🔬 Anomaly Detection Score</h3>
            <div style="display:flex;align-items:center;gap:15px;">
                <div style="flex:1;">
                    <div style="background:#1e1b2e;border-radius:8px;height:14px;overflow:hidden;">
                        <div style="width:{gauge_pct}%;height:100%;background:linear-gradient(90deg,#2ec4b6,#eab308,#ef4444);
                                    border-radius:8px;"></div>
                    </div>
                </div>
                <div style="font-size:20px;font-weight:800;color:{sev_main};font-family:'JetBrains Mono',monospace;">
                    {recon_error:.6f}
                </div>
            </div>
            <p style="color:#64748b;font-size:11px;margin:8px 0 0 0;">
                Reconstruction error — higher values indicate stronger anomaly signals
            </p>
        </div>"""

    # Risk score gauge SVG
    cvss = intel.get('cvss_score', 5.0)
    cvss_pct = cvss * 10
    cvss_color = '#ef4444' if cvss >= 9 else '#f97316' if cvss >= 7 else '#eab308' if cvss >= 4 else '#2ec4b6'

    # Build sections
    how_it_works_html = ""
    if intel.get('how_it_works'):
        items = "".join(f'<li style="margin:6px 0;color:#cbd5e1;line-height:1.6;">{h}</li>' for h in intel['how_it_works'])
        how_it_works_html = f"""
        <div style="margin:25px 0;">
            <h3 style="color:#e2e8f0;font-size:18px;border-bottom:1px solid rgba(230,57,70,0.15);padding-bottom:8px;">
                ⚙️ How This Attack Works
            </h3>
            <ol style="padding-left:20px;margin:12px 0;">{items}</ol>
        </div>"""

    ioc_html = ""
    if intel.get('indicators_of_compromise'):
        items = "".join(f"""
        <div style="display:flex;gap:8px;padding:8px 12px;margin:4px 0;background:rgba(230,57,70,0.05);
                    border-left:3px solid {sev_main};border-radius:0 6px 6px 0;">
            <span style="color:{sev_main};">⚠</span>
            <span style="color:#cbd5e1;font-size:13px;line-height:1.5;">{ioc}</span>
        </div>""" for ioc in intel['indicators_of_compromise'])
        ioc_html = f"""
        <div style="margin:25px 0;">
            <h3 style="color:#e2e8f0;font-size:18px;border-bottom:1px solid rgba(230,57,70,0.15);padding-bottom:8px;">
                🔍 Indicators of Compromise (IoCs)
            </h3>
            {items}
        </div>"""

    impact_html = ""
    if intel.get('impact'):
        items = "".join(f'<li style="margin:6px 0;color:#fca5a5;line-height:1.6;">{imp}</li>' for imp in intel['impact'])
        impact_html = f"""
        <div style="margin:25px 0;background:rgba(239,68,68,0.05);border:1px solid rgba(239,68,68,0.15);
                    border-radius:12px;padding:20px;">
            <h3 style="color:#fca5a5;font-size:18px;margin:0 0 10px 0;">💥 Potential Impact</h3>
            <ul style="padding-left:20px;margin:8px 0;">{items}</ul>
        </div>"""

    immediate_html = ""
    if intel.get('immediate_actions'):
        items = "".join(f"""
        <div style="display:flex;gap:10px;padding:10px 14px;margin:5px 0;background:rgba(239,68,68,0.08);
                    border:1px solid rgba(239,68,68,0.15);border-radius:8px;">
            <span style="color:#ef4444;font-weight:700;font-size:14px;">⚡</span>
            <span style="color:#e2e8f0;font-size:13px;line-height:1.5;">{act}</span>
        </div>""" for act in intel['immediate_actions'])
        immediate_html = f"""
        <div style="margin:25px 0;">
            <h3 style="color:#ef4444;font-size:18px;border-bottom:1px solid rgba(239,68,68,0.15);padding-bottom:8px;">
                🚨 Immediate Response Actions
            </h3>
            <p style="color:#94a3b8;font-size:12px;margin:8px 0;">Execute these actions within the first 15 minutes of detection</p>
            {items}
        </div>"""

    prevention_html = ""
    if intel.get('prevention_measures'):
        items = "".join(f"""
        <div style="display:flex;gap:10px;padding:10px 14px;margin:5px 0;background:rgba(46,196,182,0.05);
                    border:1px solid rgba(46,196,182,0.12);border-radius:8px;">
            <span style="color:#2ec4b6;font-size:14px;">🛡</span>
            <span style="color:#cbd5e1;font-size:13px;line-height:1.5;">{p}</span>
        </div>""" for p in intel['prevention_measures'])
        prevention_html = f"""
        <div style="margin:25px 0;">
            <h3 style="color:#2ec4b6;font-size:18px;border-bottom:1px solid rgba(46,196,182,0.15);padding-bottom:8px;">
                🛡️ Prevention & Hardening Measures
            </h3>
            <p style="color:#94a3b8;font-size:12px;margin:8px 0;">Implement these controls to prevent future occurrences</p>
            {items}
        </div>"""

    strategy_html = ""
    if intel.get('long_term_strategy'):
        items = "".join(f'<li style="margin:8px 0;color:#ffd166;line-height:1.6;">{s}</li>' for s in intel['long_term_strategy'])
        strategy_html = f"""
        <div style="margin:25px 0;background:rgba(255,209,102,0.05);border:1px solid rgba(255,209,102,0.12);
                    border-radius:12px;padding:20px;">
            <h3 style="color:#ffd166;font-size:18px;margin:0 0 10px 0;">🗺️ Long-Term Strategic Recommendations</h3>
            <ul style="padding-left:20px;margin:8px 0;">{items}</ul>
        </div>"""

    references_html = ""
    if intel.get('references'):
        items = "".join(f'<li style="margin:4px 0;color:#94a3b8;font-size:12px;">{r}</li>' for r in intel['references'])
        references_html = f"""
        <div style="margin:25px 0;border-top:1px solid rgba(230,57,70,0.1);padding-top:15px;">
            <h3 style="color:#64748b;font-size:14px;">📚 References & Further Reading</h3>
            <ul style="padding-left:20px;margin:8px 0;">{items}</ul>
        </div>"""

    # ─── Assemble the full HTML report ───
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CyberShield Threat Report — {attack_class}</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Inter', -apple-system, sans-serif;
            background: #0b0a10;
            color: #e2e8f0;
            line-height: 1.6;
            min-height: 100vh;
        }}
        .report-container {{
            max-width: 900px;
            margin: 0 auto;
            padding: 40px 30px;
        }}
        @media print {{
            body {{ background: #fff; color: #1a1a1a; }}
            .report-container {{ padding: 20px; }}
        }}
    </style>
</head>
<body>
<div class="report-container">

    <!-- ═══ HEADER ═══ -->
    <div style="text-align:center;padding:30px 0;border-bottom:2px solid rgba(230,57,70,0.2);margin-bottom:30px;">
        <div style="font-size:48px;margin-bottom:10px;">🛡️</div>
        <h1 style="font-size:32px;font-weight:800;
                   background:linear-gradient(135deg,#e63946,#ff6b35,#ffd166);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                   background-clip:text;">CyberShield</h1>
        <div style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#64748b;
                    letter-spacing:3px;text-transform:uppercase;margin-top:5px;">
            AI-Powered Threat Intelligence Report · v3.0 Ember Shield
        </div>
    </div>

    <!-- ═══ CLASSIFICATION BANNER ═══ -->
    <div style="background:{sev_bg};border:2px solid {sev_main};border-radius:16px;padding:25px;margin:20px 0;
                position:relative;overflow:hidden;">
        <div style="position:absolute;top:0;left:0;right:0;height:4px;
                    background:linear-gradient(90deg,{sev_main},{sev_light});"></div>
        <div style="display:flex;justify-content:space-between;align-items:start;flex-wrap:wrap;gap:15px;">
            <div>
                <div style="font-size:12px;color:{sev_light};font-family:'JetBrains Mono',monospace;
                            letter-spacing:2px;text-transform:uppercase;">Threat Classification</div>
                <h2 style="font-size:28px;font-weight:800;color:#e2e8f0;margin:8px 0;">{icon} {attack_class}</h2>
                <div style="font-size:14px;color:#94a3b8;">{intel.get('full_name', attack_class)}</div>
            </div>
            <div style="text-align:right;">
                <div style="display:inline-block;background:{sev_main};color:#fff;padding:6px 18px;
                            border-radius:20px;font-size:13px;font-weight:700;letter-spacing:1px;">
                    {severity}
                </div>
                <div style="font-size:11px;color:#64748b;margin-top:8px;">
                    MITRE ATT&CK: {intel.get('mitre_id', 'N/A')}
                </div>
            </div>
        </div>
    </div>

    <!-- ═══ EXECUTIVE SUMMARY GRID ═══ -->
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:20px 0;">
        <div style="background:#12101a;border:1px solid rgba(230,57,70,0.15);border-radius:12px;padding:16px;text-align:center;">
            <div style="font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:1.5px;">Confidence</div>
            <div style="font-size:24px;font-weight:800;color:{sev_main};margin:6px 0;">{conf_pct:.1f}%</div>
        </div>
        <div style="background:#12101a;border:1px solid rgba(230,57,70,0.15);border-radius:12px;padding:16px;text-align:center;">
            <div style="font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:1.5px;">CVSS Score</div>
            <div style="font-size:24px;font-weight:800;color:{cvss_color};margin:6px 0;">{cvss}/10</div>
        </div>
        <div style="background:#12101a;border:1px solid rgba(230,57,70,0.15);border-radius:12px;padding:16px;text-align:center;">
            <div style="font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:1.5px;">Risk Level</div>
            <div style="font-size:24px;font-weight:800;color:{sev_main};margin:6px 0;">{risk_level}</div>
        </div>
        <div style="background:#12101a;border:1px solid rgba(230,57,70,0.15);border-radius:12px;padding:16px;text-align:center;">
            <div style="font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:1.5px;">Category</div>
            <div style="font-size:13px;font-weight:600;color:#94a3b8;margin:6px 0;">{intel.get('category', 'N/A')}</div>
        </div>
    </div>

    <!-- ═══ CVSS GAUGE ═══ -->
    <div style="background:#12101a;border:1px solid rgba(230,57,70,0.15);border-radius:12px;padding:20px;margin:20px 0;">
        <h3 style="color:#e2e8f0;font-size:16px;margin:0 0 12px 0;">🎯 CVSS Risk Score</h3>
        <div style="display:flex;align-items:center;gap:15px;">
            <div style="flex:1;">
                <div style="background:#1e1b2e;border-radius:8px;height:18px;overflow:hidden;position:relative;">
                    <div style="position:absolute;top:0;left:0;width:25%;height:100%;background:#2ec4b6;opacity:0.15;"></div>
                    <div style="position:absolute;top:0;left:25%;width:25%;height:100%;background:#eab308;opacity:0.15;"></div>
                    <div style="position:absolute;top:0;left:50%;width:25%;height:100%;background:#f97316;opacity:0.15;"></div>
                    <div style="position:absolute;top:0;left:75%;width:25%;height:100%;background:#ef4444;opacity:0.15;"></div>
                    <div style="width:{cvss_pct}%;height:100%;background:{cvss_color};border-radius:8px;
                                position:relative;z-index:1;"></div>
                </div>
                <div style="display:flex;justify-content:space-between;margin-top:4px;">
                    <span style="font-size:10px;color:#2ec4b6;">Low (0-3.9)</span>
                    <span style="font-size:10px;color:#eab308;">Medium (4-6.9)</span>
                    <span style="font-size:10px;color:#f97316;">High (7-8.9)</span>
                    <span style="font-size:10px;color:#ef4444;">Critical (9-10)</span>
                </div>
            </div>
            <div style="font-size:32px;font-weight:900;color:{cvss_color};font-family:'JetBrains Mono',monospace;
                        min-width:60px;text-align:center;">{cvss}</div>
        </div>
    </div>

    {recon_gauge_html}

    <!-- ═══ DESCRIPTION ═══ -->
    <div style="margin:25px 0;">
        <h3 style="color:#e2e8f0;font-size:18px;border-bottom:1px solid rgba(230,57,70,0.15);padding-bottom:8px;">
            📋 Threat Description
        </h3>
        <p style="color:#cbd5e1;font-size:14px;line-height:1.8;margin:12px 0;text-align:justify;">
            {intel.get('description', 'No description available.')}
        </p>
    </div>

    {how_it_works_html}
    {ioc_html}
    {prob_chart_html}
    {impact_html}
    {immediate_html}
    {prevention_html}
    {strategy_html}

    <!-- ═══ METADATA ═══ -->
    <div style="margin:30px 0;background:#12101a;border:1px solid rgba(230,57,70,0.1);border-radius:12px;padding:20px;">
        <h3 style="color:#64748b;font-size:14px;margin:0 0 12px 0;">📋 Report Metadata</h3>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:12px;
                    font-family:'JetBrains Mono',monospace;">
            <div><span style="color:#64748b;">Report ID:</span> <span style="color:#94a3b8;">CS-{datetime.now().strftime('%Y%m%d%H%M%S')}-P{packet_idx or 0:03d}</span></div>
            <div><span style="color:#64748b;">Generated:</span> <span style="color:#94a3b8;">{timestamp}</span></div>
            <div><span style="color:#64748b;">Packet Index:</span> <span style="color:#94a3b8;">#{packet_idx or 'N/A'}</span></div>
            <div><span style="color:#64748b;">AI Confidence:</span> <span style="color:#94a3b8;">{conf_pct:.2f}%</span></div>
            <div><span style="color:#64748b;">Detection Model:</span> <span style="color:#94a3b8;">CyberShield AI v3.0</span></div>
            <div><span style="color:#64748b;">Dataset:</span> <span style="color:#94a3b8;">CICIDS 2017</span></div>
        </div>
    </div>

    {references_html}

    <!-- ═══ FOOTER ═══ -->
    <div style="text-align:center;padding:25px 0;border-top:1px solid rgba(230,57,70,0.1);margin-top:30px;">
        <div style="font-size:11px;color:#64748b;font-family:'JetBrains Mono',monospace;letter-spacing:2px;">
            CYBERSHIELD v3.0 · EMBER SHIELD · AI-POWERED THREAT INTELLIGENCE
        </div>
        <div style="font-size:10px;color:#475569;margin-top:5px;">
            Generated on {date_str} · PyTorch · Scikit-learn · CICIDS 2017
        </div>
        <div style="font-size:10px;color:#475569;margin-top:3px;">
            This report is auto-generated by AI models. Always verify findings with manual analysis.
        </div>
    </div>

</div>
</body>
</html>"""
    return html




# ── Plotly Chart Helpers ──────────────────────────────────────────────────
PLOTLY_DARK_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Inter, sans-serif', color='#94a3b8', size=12),
    margin=dict(l=50, r=30, t=50, b=50),
    xaxis=dict(gridcolor='rgba(230, 57, 70,0.08)', zerolinecolor='rgba(230, 57, 70,0.08)'),
    yaxis=dict(gridcolor='rgba(230, 57, 70,0.08)', zerolinecolor='rgba(230, 57, 70,0.08)'),
    legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='rgba(230, 57, 70,0.15)', borderwidth=1,
                font=dict(size=11)),
)


def make_recon_error_chart(recon_errors, ae_preds, threshold):
    """Interactive bar chart of reconstruction errors per packet."""
    colors = ['#ef4444' if p == 1 else '#2ec4b6' for p in ae_preds]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(range(1, len(recon_errors)+1)),
        y=recon_errors,
        marker_color=colors,
        marker_line_width=0,
        opacity=0.85,
        hovertemplate='Packet #%{x}<br>Error: %{y:.6f}<extra></extra>',
    ))
    fig.add_hline(y=threshold, line_dash='dash', line_color='#f59e0b', line_width=2,
                  annotation_text=f'Threshold = {threshold:.4f}',
                  annotation_font_color='#f59e0b',
                  annotation_font_size=11)
    fig.update_layout(
        title=dict(text='🔬 Autoencoder Reconstruction Error', font=dict(size=16, color='#e2e8f0')),
        xaxis_title='Packet Index',
        yaxis_title='Reconstruction Error (MSE)',
        height=380,
        **PLOTLY_DARK_LAYOUT,
    )
    return fig


def make_attack_distribution_chart(pred_classes):
    """Horizontal bar chart of attack type distribution."""
    counts = pd.Series(pred_classes).value_counts()
    colors = []
    for c in counts.index:
        sev = SEVERITY_MAP.get(c, ('NONE','',''))[0]
        colors.append(SEVERITY_COLOR.get(sev, '#64748b'))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=counts.index,
        x=counts.values,
        orientation='h',
        marker_color=colors,
        marker_line_width=0,
        opacity=0.85,
        hovertemplate='%{y}: %{x} packets<extra></extra>',
    ))
    fig.update_layout(
        title=dict(text='📊 Attack Type Distribution', font=dict(size=16, color='#e2e8f0')),
        xaxis_title='Count',
        yaxis_autorange='reversed',
        height=max(280, len(counts)*50 + 120),
        **PLOTLY_DARK_LAYOUT,
    )
    return fig


def make_confidence_chart(confidences, ae_preds):
    """Bar chart of transformer confidence per packet."""
    colors = ['#ef4444' if p == 1 else '#ffd166' for p in ae_preds]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(range(1, len(confidences)+1)),
        y=[c*100 for c in confidences],
        marker_color=colors,
        marker_line_width=0,
        opacity=0.85,
        hovertemplate='Packet #%{x}<br>Confidence: %{y:.1f}%<extra></extra>',
    ))
    fig.add_hline(y=50, line_dash='dot', line_color='#475569', line_width=1)
    fig.update_layout(
        title=dict(text='🤖 Transformer Classification Confidence', font=dict(size=16, color='#e2e8f0')),
        xaxis_title='Packet Index',
        yaxis_title='Confidence (%)',
        yaxis_range=[0, 105],
        height=320,
        **PLOTLY_DARK_LAYOUT,
    )
    return fig


def make_severity_donut(pred_classes):
    """Donut chart of severity distribution."""
    sev_counts = {}
    for c in pred_classes:
        sev = SEVERITY_MAP.get(c, ('UNKNOWN','',''))[0]
        sev_counts[sev] = sev_counts.get(sev, 0) + 1

    labels = list(sev_counts.keys())
    values = list(sev_counts.values())
    colors = [SEVERITY_COLOR.get(l, '#64748b') for l in labels]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(colors=colors, line=dict(color='#0b0a10', width=2)),
        textinfo='label+percent',
        textfont=dict(size=12, color='#e2e8f0'),
        hovertemplate='%{label}: %{value} packets (%{percent})<extra></extra>',
    )])
    fig.update_layout(
        title=dict(text='🎯 Severity Breakdown', font=dict(size=16, color='#e2e8f0')),
        showlegend=False,
        height=350,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter, sans-serif', color='#94a3b8'),
        margin=dict(l=20, r=20, t=50, b=20),
        annotations=[dict(text='SEVERITY', x=0.5, y=0.5, font_size=13,
                          font_color='#64748b', showarrow=False)],
    )
    return fig


def make_threat_gauge(n_attacks, total):
    """Gauge chart showing threat level."""
    pct = (n_attacks / total * 100) if total > 0 else 0
    if pct >= 50:
        bar_color = '#ef4444'
    elif pct >= 25:
        bar_color = '#f97316'
    elif pct > 0:
        bar_color = '#eab308'
    else:
        bar_color = '#2ec4b6'

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=pct,
        number=dict(suffix='%', font=dict(size=36, color='#e2e8f0')),
        delta=dict(reference=0, valueformat='.1f'),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor='#475569', tickfont=dict(color='#64748b')),
            bar=dict(color=bar_color, thickness=0.7),
            bgcolor='rgba(18, 14, 24,0.8)',
            borderwidth=1,
            bordercolor='rgba(230, 57, 70,0.15)',
            steps=[
                dict(range=[0, 25],  color='rgba(46, 196, 182,0.08)'),
                dict(range=[25, 50], color='rgba(234,179,8,0.08)'),
                dict(range=[50, 75], color='rgba(249,115,22,0.08)'),
                dict(range=[75, 100], color='rgba(239,68,68,0.08)'),
            ],
            threshold=dict(line=dict(color='#ef4444', width=2), thickness=0.75, value=pct),
        ),
    ))
    fig.update_layout(
        title=dict(text='⚡ Network Threat Level', font=dict(size=16, color='#e2e8f0')),
        height=280,
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter, sans-serif', color='#94a3b8'),
        margin=dict(l=30, r=30, t=60, b=20),
    )
    return fig


# ── HTML Helpers ──────────────────────────────────────────────────────────
def render_hero():
    st.markdown("""
    <div class="hero-container">
        <div class="hero-radar-wrap">
            <div class="hero-radar-ring ring-1"></div>
            <div class="hero-radar-ring ring-2"></div>
            <div class="hero-radar-ring ring-3"></div>
            <div class="hero-shield">🛡️</div>
        </div>
        <div class="hero-title" data-text="CyberShield">CyberShield</div>
        <div class="hero-subtitle">AI-Powered Network Intrusion Detection</div>
        <div class="hero-tagline">Ember Shield · Deep Learning Anomaly Detection &amp; Attack Classification</div>
        <div class="hero-divider"></div>
    </div>
    """, unsafe_allow_html=True)


def render_metrics(total, normal, attacks, critical):
    normal_pct = f"{normal/total*100:.0f}%" if total > 0 else "0%"
    attack_pct = f"{attacks/total*100:.0f}%" if total > 0 else "0%"
    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-card cyan">
            <div class="metric-icon">📡</div>
            <div class="metric-value">{total:,}</div>
            <div class="metric-label">Total Packets</div>
            <div class="metric-delta cyan">Analyzed</div>
        </div>
        <div class="metric-card green">
            <div class="metric-icon">✅</div>
            <div class="metric-value">{normal:,}</div>
            <div class="metric-label">Normal Traffic</div>
            <div class="metric-delta green">{normal_pct}</div>
        </div>
        <div class="metric-card red">
            <div class="metric-icon">🚨</div>
            <div class="metric-value">{attacks:,}</div>
            <div class="metric-label">Attacks Detected</div>
            <div class="metric-delta red">{attack_pct}</div>
        </div>
        <div class="metric-card purple">
            <div class="metric-icon">💀</div>
            <div class="metric-value">{critical}</div>
            <div class="metric-label">Critical Alerts</div>
            <div class="metric-delta {'red' if critical > 0 else 'green'}">{'DANGER' if critical > 0 else 'CLEAR'}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_report_card(packet_idx, attack_class, confidence, severity, icon, css_class):
    """Render a premium threat report card."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")
    actions = ACTIONS_MAP.get(attack_class, ['Investigate and monitor'])

    badge_class = f"badge-{css_class}"
    actions_html = "".join(
        f'<div class="report-action-item">⚡ [{i+1}] {a}</div>'
        for i, a in enumerate(actions)
    )

    st.markdown(f"""
    <div class="report-card severity-{css_class}">
        <div class="report-header">
            <div class="report-title">{icon} Packet #{packet_idx} — {attack_class}</div>
            <div class="report-severity-badge {badge_class}">{severity}</div>
        </div>
        <div class="report-row"><span class="report-label">⏱ Timestamp</span><span class="report-value">{timestamp}</span></div>
        <div class="report-row"><span class="report-label">🎯 Attack Type</span><span class="report-value">{attack_class}</span></div>
        <div class="report-row"><span class="report-label">📊 Confidence</span><span class="report-value">{confidence*100:.2f}%</span></div>
        <div class="report-row"><span class="report-label">⚠️ Severity</span><span class="report-value">{icon} {severity}</span></div>
        <div class="report-actions">
            <div class="report-label" style="margin-bottom: 0.3rem;">🔧 Recommended Actions:</div>
            {actions_html}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_welcome():
    st.markdown("""
    <div class="welcome-card">
        <div class="welcome-title">👋 Welcome to CyberShield</div>
        <div class="welcome-desc">
            CyberShield uses a dual-AI pipeline to detect and classify network intrusions in real-time.
            Upload network traffic data or use our demo samples to see the system in action.
        </div>
        <div class="pipeline-grid">
            <div class="pipeline-step">
                <div class="step-num">Step 01</div>
                <div class="step-icon">📂</div>
                <div class="step-title">Upload Data</div>
                <div class="step-desc">CSV or demo samples</div>
            </div>
            <div class="pipeline-step">
                <div class="step-num">Step 02</div>
                <div class="step-icon">🔬</div>
                <div class="step-title">AE Scan</div>
                <div class="step-desc">Anomaly detection</div>
            </div>
            <div class="pipeline-step">
                <div class="step-num">Step 03</div>
                <div class="step-icon">🤖</div>
                <div class="step-title">Classify</div>
                <div class="step-desc">Attack identification</div>
            </div>
            <div class="pipeline-step">
                <div class="step-num">Step 04</div>
                <div class="step-icon">📄</div>
                <div class="step-title">Report</div>
                <div class="step-desc">Threat intelligence</div>
            </div>
        </div>
        <div class="model-badges">
            <div class="model-badge">
                <span class="badge-dot ae"></span>
                <span class="badge-text">Autoencoder — Anomaly Detection</span>
            </div>
            <div class="model-badge">
                <span class="badge-dot tf"></span>
                <span class="badge-text">Transformer — Attack Classification</span>
            </div>
            <div class="model-badge">
                <span class="badge-dot ds"></span>
                <span class="badge-text">CICIDS 2017 Dataset</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── MAIN APP ──────────────────────────────────────────────────────────────
def main():
    # ── Hero Header ──
    render_hero()

    # ── Load Models ──
    models_loaded = False
    vae_loaded = False
    ae_threshold = 0.005  # default fallback
    with st.spinner("Initializing AI models..."):
        try:
            (ae_model, ae_scaler, ae_threshold,
             tr_model, tr_scaler, tr_le,
             class_names, device) = load_models()
            models_loaded = True
        except Exception as e:
            st.error(f"❌ Could not load AE/Transformer models: {e}")
            st.info("Make sure all `.pth` and `.pkl` files are in the same folder as this app.")

        try:
            (rf_model, vae_scaler, vae_le, vae_class_names) = load_vae_models()
            vae_loaded = True
        except Exception as e:
            st.warning(f"⚠️ VAE/RF models not available: {e}")

    if not models_loaded and not vae_loaded:
        st.error("No models could be loaded. Check that model files are present.")
        return

    # ── Sidebar ──
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-brand">
            <div class="sidebar-logo">🛡️</div>
            <div class="sidebar-title">CyberShield</div>
            <div class="sidebar-version">v3.0 · EMBER SHIELD</div>
        </div>
        """, unsafe_allow_html=True)

        if models_loaded:
            st.markdown("""
            <div class="status-bar">
                <div class="status-dot"></div>
                <div class="status-text">Models Online</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-section-title">🧠 Model Selection</div>', unsafe_allow_html=True)

        model_choice = st.radio(
            "Choose detection model",
            ["🔬 AE → Transformer", "🧬 VAE → Random Forest"],
            index=0,
            label_visibility="collapsed",
            help="AE: Autoencoder anomaly detection + Transformer classification.\nVAE: CVAE-augmented Random Forest direct classification."
        )
        use_vae = model_choice.startswith("🧬")

        # Show model description
        if use_vae:
            st.markdown("""
            <div style="font-size: 0.72rem; color: #ffe5a0; background: rgba(255, 209, 102,0.08);
                        border: 1px solid rgba(255, 209, 102,0.2); border-radius: 8px;
                        padding: 0.6rem 0.8rem; margin: 0.3rem 0 0.5rem 0; line-height: 1.5;">
                <strong>VAE Pipeline</strong><br>
                Conditional VAE augmented<br>
                Random Forest classifier<br>
                <span style="color: #ffd166;">15-class direct classification</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="font-size: 0.72rem; color: #ff8c61; background: rgba(230, 57, 70,0.08);
                        border: 1px solid rgba(230, 57, 70,0.2); border-radius: 8px;
                        padding: 0.6rem 0.8rem; margin: 0.3rem 0 0.5rem 0; line-height: 1.5;">
                <strong>AE Pipeline</strong><br>
                Autoencoder anomaly detection<br>
                + Transformer classification<br>
                <span style="color: #e63946;">Dual-stage detection</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-section-title">📂 Input Source</div>', unsafe_allow_html=True)

        input_mode = st.radio(
            "Choose input",
            ["Upload CSV File", "Use Sample Data"],
            index=1,
            label_visibility="collapsed"
        )

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-section-title">⚙️ Detection Settings</div>', unsafe_allow_html=True)

        if not use_vae:
            ae_threshold_override = st.slider(
                "AE Detection Threshold",
                min_value=0.0001,
                max_value=0.05,
                value=float(ae_threshold),
                step=0.0001,
                format="%.4f",
                help="Lower = more sensitive (more alerts). Default from training."
            )
        else:
            ae_threshold_override = float(ae_threshold)  # not used for VAE
            st.markdown("""
            <div style="font-size: 0.72rem; color: #64748b; padding: 0.3rem 0;">
                VAE pipeline uses Random Forest confidence for detection.
                No threshold tuning required.
            </div>
            """, unsafe_allow_html=True)

        show_all = st.checkbox("Show all packets (not just attacks)", value=False)

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-section-title">ℹ️ About</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-size: 0.75rem; color: #64748b; line-height: 1.6;">
            <strong style="color: #94a3b8;">Domain:</strong> Cybersecurity · Gen AI<br>
            <strong style="color: #94a3b8;">Dataset:</strong> CICIDS 2017<br>
            <strong style="color: #94a3b8;">Models:</strong> {'VAE · Random Forest' if use_vae else 'AE · Transformer'}<br>
            <strong style="color: #94a3b8;">Stack:</strong> PyTorch · Scikit-learn · Streamlit
        </div>
        """, unsafe_allow_html=True)

    # ── Get Features ──
    features = None
    df_display = None

    if input_mode == "Upload CSV File":
        uploaded = st.file_uploader(
            "Upload network traffic CSV (CICIDS 2017 format)",
            type=["csv"],
            help="Upload a CSV file with numeric network flow features."
        )
        if uploaded:
            df_raw = pd.read_csv(uploaded)
            df_raw.columns = df_raw.columns.str.strip()

            drop_cols = [c for c in [
                "Flow ID", "Source IP", "Source Port",
                "Destination IP", "Destination Port",
                "Timestamp", "Label", "Unnamed: 0"
            ] if c in df_raw.columns]
            df_feat = df_raw.drop(columns=drop_cols)
            df_feat = df_feat.select_dtypes(include=[np.number])
            df_feat = df_feat.replace([np.inf, -np.inf], np.nan).dropna()

            features   = df_feat.values
            df_display = df_raw.head(len(features))

            st.markdown(f"""
            <div class="alert-banner safe">
                <span class="alert-icon">✅</span>
                <span>Loaded <strong>{len(features):,}</strong> packets from uploaded file — ready to scan.</span>
            </div>
            """, unsafe_allow_html=True)

    else:
        # Generate realistic sample data
        np.random.seed(42)
        n_input = ae_model.encoder[0].in_features
        n_samples = 20

        normal_data = np.random.randn(12, n_input) * 0.3
        attack_data = np.random.randn(8, n_input) * 3.0 + 2.0

        features = np.vstack([normal_data, attack_data])
        labels   = ['BENIGN']*12 + ['Simulated Attack']*8
        df_display = pd.DataFrame({
            'Packet #': range(1, n_samples+1),
            'True Label (Demo)': labels
        })

        st.markdown("""
        <div class="alert-banner medium">
            <span class="alert-icon">⚡</span>
            <span>Using <strong>simulated demo data</strong> (20 packets). Upload a real CSV for actual results.</span>
        </div>
        """, unsafe_allow_html=True)

    # ── Run Detection ──
    scan_label = "🔍  Launch CyberShield Scan"
    if features is not None and st.button(scan_label, type="primary", use_container_width=True):

        # Scanning animation
        scan_placeholder = st.empty()

        if use_vae and vae_loaded:
            # ── VAE Pipeline ──
            scan_placeholder.markdown("""
            <div class="scan-overlay">
                <div class="scan-ring"></div>
                <div class="scan-text">SCANNING NETWORK TRAFFIC...</div>
                <div class="scan-sub">VAE-augmented Random Forest classification in progress</div>
            </div>
            """, unsafe_allow_html=True)
            time.sleep(1.2)

            pred_classes, confidences, all_probs, ae_preds, recon_errors = run_vae_pipeline(
                rf_model, vae_scaler, vae_le, vae_class_names, features
            )

            scan_placeholder.markdown("""
            <div class="scan-overlay">
                <div class="scan-ring"></div>
                <div class="scan-text">GENERATING REPORTS...</div>
                <div class="scan-sub">Building threat intelligence reports</div>
            </div>
            """, unsafe_allow_html=True)
            time.sleep(0.6)
            scan_placeholder.empty()

        elif not use_vae and models_loaded:
            # ── AE Pipeline ──
            scan_placeholder.markdown("""
            <div class="scan-overlay">
                <div class="scan-ring"></div>
                <div class="scan-text">SCANNING NETWORK TRAFFIC...</div>
                <div class="scan-sub">Autoencoder anomaly detection in progress</div>
            </div>
            """, unsafe_allow_html=True)
            time.sleep(1.2)

            # AE — Anomaly Detection
            recon_errors, ae_preds = run_ae(
                ae_model, ae_scaler, ae_threshold_override, features, device
            )

            scan_placeholder.markdown("""
            <div class="scan-overlay">
                <div class="scan-ring"></div>
                <div class="scan-text">CLASSIFYING THREATS...</div>
                <div class="scan-sub">Transformer attack classification running</div>
            </div>
            """, unsafe_allow_html=True)
            time.sleep(1.0)

            # Transformer — Classification
            pred_classes, confidences, all_probs = run_transformer(
                tr_model, tr_scaler, tr_le, class_names, features, device
            )

            scan_placeholder.markdown("""
            <div class="scan-overlay">
                <div class="scan-ring"></div>
                <div class="scan-text">GENERATING REPORTS...</div>
                <div class="scan-sub">Building threat intelligence reports</div>
            </div>
            """, unsafe_allow_html=True)
            time.sleep(0.6)
            scan_placeholder.empty()

        else:
            scan_placeholder.empty()
            st.error("❌ Selected model is not available. Please choose another model.")
            st.stop()

        # Build results dataframe
        pipeline_label = "VAE/RF" if use_vae else "AE"
        results = pd.DataFrame({
            'Packet #'    : range(1, len(features)+1),
            'AE Error'    : recon_errors.round(6),
            'AE Result'   : ['🔴 ATTACK' if p == 1 else '🟢 NORMAL' for p in ae_preds],
            'Attack Type' : pred_classes,
            'Confidence'  : [f"{c*100:.1f}%" for c in confidences],
            'Severity'    : [SEVERITY_MAP.get(c, ('?', '?', '?'))[0] for c in pred_classes],
        })

        # ── Summary Metrics ──
        n_attacks  = int(ae_preds.sum())
        n_normal   = len(ae_preds) - n_attacks
        n_critical = sum(
            1 for c in pred_classes
            if SEVERITY_MAP.get(c, ('',))[0] == 'CRITICAL'
        )

        st.markdown("""
        <div class="section-header">
            <span class="icon">📊</span>
            <span>Detection Summary</span>
            <span class="line"></span>
        </div>
        """, unsafe_allow_html=True)

        render_metrics(len(features), n_normal, n_attacks, n_critical)

        # ── Tabs ──
        tab1, tab2, tab3 = st.tabs([
            "📋  Packet Results",
            "📈  Visualizations",
            "📄  Threat Reports"
        ])

        # ── Tab 1: Packet Results ──
        with tab1:
            st.markdown("""
            <div class="section-header">
                <span class="icon">📋</span>
                <span>Packet-by-Packet Analysis</span>
                <span class="line"></span>
            </div>
            """, unsafe_allow_html=True)

            filter_df = results if show_all else results[results['AE Result'] == '🔴 ATTACK']

            if len(filter_df) == 0:
                st.markdown("""
                <div class="alert-banner safe">
                    <span class="alert-icon">✅</span>
                    <span>All clear — <strong>no attacks detected</strong> in this traffic!</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.dataframe(
                    filter_df,
                    use_container_width=True,
                    hide_index=True,
                    height=min(400, len(filter_df)*38 + 40),
                )

                csv_out = filter_df.to_csv(index=False)
                st.download_button(
                    "⬇️  Download Results CSV",
                    data      = csv_out,
                    file_name = "cybershield_results.csv",
                    mime      = "text/csv"
                )

        # ── Tab 2: Visualizations ──
        with tab2:
            st.markdown("""
            <div class="section-header">
                <span class="icon">📈</span>
                <span>Visual Analytics Dashboard</span>
                <span class="line"></span>
            </div>
            """, unsafe_allow_html=True)

            # Row 1: Gauge + Donut
            col_g1, col_g2 = st.columns(2)
            with col_g1:
                st.plotly_chart(
                    make_threat_gauge(n_attacks, len(features)),
                    use_container_width=True, config={'displayModeBar': False}
                )
            with col_g2:
                st.plotly_chart(
                    make_severity_donut(pred_classes),
                    use_container_width=True, config={'displayModeBar': False}
                )

            # Row 2: Recon Error + Distribution
            col_a, col_b = st.columns(2)
            with col_a:
                st.plotly_chart(
                    make_recon_error_chart(recon_errors, ae_preds, ae_threshold_override),
                    use_container_width=True, config={'displayModeBar': False}
                )
            with col_b:
                st.plotly_chart(
                    make_attack_distribution_chart(pred_classes),
                    use_container_width=True, config={'displayModeBar': False}
                )

            # Row 3: Full-width confidence
            st.plotly_chart(
                make_confidence_chart(confidences, ae_preds),
                use_container_width=True, config={'displayModeBar': False}
            )

        # ── Tab 3: Threat Reports ──
        with tab3:
            st.markdown("""
            <div class="section-header">
                <span class="icon">📄</span>
                <span>Auto-Generated Threat Intelligence</span>
                <span class="line"></span>
            </div>
            """, unsafe_allow_html=True)

            attack_indices = [i for i, p in enumerate(ae_preds) if p == 1]

            if len(attack_indices) == 0:
                st.markdown("""
                <div class="alert-banner safe">
                    <span class="alert-icon">✅</span>
                    <span>No threats detected — <strong>no reports generated</strong>.</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Summary banner
                if n_critical > 0:
                    st.markdown(f"""
                    <div class="alert-banner critical">
                        <span class="alert-icon">🚨</span>
                        <span><strong>{n_critical} CRITICAL</strong> threat(s) detected — immediate action required!</span>
                    </div>
                    """, unsafe_allow_html=True)

                shown = attack_indices[:8]
                for idx in shown:
                    attack  = pred_classes[idx]
                    conf    = confidences[idx]
                    severity, icon, css_class = SEVERITY_MAP.get(
                        attack, ('UNKNOWN', '⚠️', 'medium')
                    )

                    render_report_card(idx + 1, attack, conf, severity, icon, css_class)

                    # Download detailed HTML report
                    pkt_recon = recon_errors[idx] if idx < len(recon_errors) else None
                    pkt_probs = all_probs if all_probs is not None else None
                    pkt_cnames = (vae_class_names if use_vae else class_names) if 'class_names' in dir() or 'vae_class_names' in dir() else None
                    report_html = generate_detailed_html_report(
                        attack, conf, idx + 1,
                        recon_error=pkt_recon,
                        all_probs=pkt_probs,
                        class_names=pkt_cnames,
                    )
                    st.download_button(
                        f"📄  Download Full Report — Packet #{idx+1}",
                        data      = report_html,
                        file_name = f"CyberShield_Threat_Report_Packet_{idx+1}.html",
                        mime      = "text/html",
                        key       = f"dl_{idx}"
                    )

                if len(attack_indices) > 8:
                    remaining = len(attack_indices) - 8
                    st.markdown(f"""
                    <div class="alert-banner medium">
                        <span class="alert-icon">ℹ️</span>
                        <span>Showing 8 of {len(attack_indices)} reports. <strong>{remaining} more</strong> available in exported CSV.</span>
                    </div>
                    """, unsafe_allow_html=True)

                # Combined full analysis report
                st.markdown("""
                <div class="section-header" style="margin-top:2rem;">
                    <span class="icon">📦</span>
                    <span>Combined Threat Analysis</span>
                    <span class="line"></span>
                </div>
                """, unsafe_allow_html=True)

                combined_reports = []
                for cidx in attack_indices:
                    c_attack = pred_classes[cidx]
                    c_conf = confidences[cidx]
                    c_recon = recon_errors[cidx] if cidx < len(recon_errors) else None
                    c_cnames = (vae_class_names if use_vae else class_names) if 'class_names' in dir() or 'vae_class_names' in dir() else None
                    combined_reports.append(generate_detailed_html_report(
                        c_attack, c_conf, cidx + 1,
                        recon_error=c_recon,
                        all_probs=all_probs,
                        class_names=c_cnames,
                    ))
                full_report = '<hr style="border:none;border-top:3px solid rgba(230,57,70,0.3);margin:40px 0;">'.join(combined_reports)
                st.download_button(
                    "📦  Download Complete Threat Analysis (All Attacks)",
                    data=full_report,
                    file_name="CyberShield_Full_Threat_Analysis.html",
                    mime="text/html",
                    key="dl_full_analysis",
                    type="primary",
                    use_container_width=True,
                )

    elif features is None and input_mode == "Upload CSV File":
        render_welcome()

    elif features is not None and input_mode == "Use Sample Data":
        # Show welcome if button not clicked yet
        render_welcome()

    else:
        render_welcome()

    # ── Footer ──
    st.markdown("""
    <div class="app-footer">
        <div class="footer-text">CYBERSHIELD · AI-POWERED NETWORK DEFENSE</div>
        <div class="footer-tech">
            <span>PyTorch</span>
            <span>Streamlit</span>
            <span>Scikit-learn</span>
            <span>CICIDS 2017</span>
            <span>Plotly</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Run App ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
