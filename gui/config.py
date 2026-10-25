"""
CyberShield GAN GUI Configuration
Centralized configuration for paths, labels, severity, and system constants.
"""

import os
import glob

# Project root directory (parent of gui/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def resolve_path(*relative_candidates):
    """
    Find the first existing file from a list of relative paths candidates.
    If none exist, return the primary candidate under BASE_DIR.
    """
    for rel in relative_candidates:
        candidate = os.path.join(BASE_DIR, rel)
        if os.path.exists(candidate):
            return candidate
    return os.path.join(BASE_DIR, relative_candidates[0])

# Model Checkpoint and Artifact Paths
WGAN_CKPT_PATH = resolve_path("models/cybershield_wgan.pth", "cybershield_wgan.pth")
CTGAN_MODEL_PATH = resolve_path("models/cybershield_ctgan.pkl", "cybershield_ctgan.pkl")
GAN_SCALER_PATH = resolve_path("models/gan_scaler.pkl", "gan_scaler.pkl")
GAN_LE_PATH = resolve_path("models/gan_label_encoder.pkl", "gan_label_encoder.pkl")
GAN_COMPARISON_CSV = resolve_path("evaluations/metrics/gan_comparison.csv", "gan_comparison.csv")
GAN_DASHBOARD_PNG = resolve_path("evaluations/figures/gan_dashboard.png", "gan_dashboard.png")

# Diffusion Model and Artifact Paths
DIFFUSION_CKPT_PATH = resolve_path("models/cybershield_diffusion.pth", "cybershield_diffusion.pth", "cybershield_diffusion (1).pth")
DIFFUSION_CLASS_NAMES_PATH = resolve_path("diffusion_class_names.pkl", "models/diffusion_class_names.pkl")
DIFFUSION_COMPARISON_CSV = resolve_path("diffusion_comparison.csv", "evaluations/metrics/diffusion_comparison.csv")
DIFFUSION_LOSS_HISTORY_CSV = resolve_path("diffusion_loss_history.csv", "evaluations/metrics/diffusion_loss_history.csv")
DIFFUSION_HEATMAPS_PATH = resolve_path("diffusion_generated_heatmaps.npy", "data/diffusion_generated_heatmaps.npy")
DIFFUSION_LABELS_PATH = resolve_path("diffusion_generated_labels.npy", "data/diffusion_generated_labels.npy")
DIFFUSION_DASHBOARD_PNG = resolve_path("diffusion_dashboard.png", "evaluations/figures/diffusion_dashboard.png")
TEXT_TO_HEATMAP_DEMO_PNG = resolve_path("text_to_heatmap_demo.png", "evaluations/figures/text_to_heatmap_demo.png")

WGAN_SYNTHETIC_CSV = resolve_path("data/wgan_synthetic_attacks.csv", "wgan_synthetic_attacks.csv")
CTGAN_SYNTHETIC_CSV = resolve_path("data/ctgan_synthetic_attacks.csv", "ctgan_synthetic_attacks.csv")
REAL_TRAFFIC_CSV = resolve_path("data/cybershield_mixed_traffic.csv", "data/test_traffic.csv")

# Detector Models (AE & Transformer)
AE_CKPT_PATH = resolve_path("models/cybershield_ae.pth", "cybershield_ae.pth")
AE_SCALER_PATH = resolve_path("models/ae_scaler.pkl", "ae_scaler.pkl", "models/scaler.pkl")

TR_CKPT_PATH = resolve_path("models/cybershield_transformer.pth", "cybershield_transformer.pth", "models/best_transformer.pth")
TR_SCALER_PATH = resolve_path("models/transformer_scaler.pkl", "transformer_scaler.pkl")
TR_LE_PATH = resolve_path("models/transformer_label_encoder.pkl", "transformer_label_encoder.pkl")

# Evaluation Figures & Metrics
AE_DASHBOARD_PNG = resolve_path("evaluations/figures/ae_diagnostics_dashboard.png")
CVAE_DASHBOARD_PNG = resolve_path("evaluations/figures/cvae_diagnostics_dashboard.png")
CONFUSION_MATRIX_PNG = resolve_path("evaluations/figures/intrusion_classifier_confusion_matrix.png")
ATTENTION_PNG = resolve_path("evaluations/figures/transformer_attention.png")

AE_PERFORMANCE_CSV = resolve_path("evaluations/metrics/ae_detection_performance.csv")
SYSTEM_METRICS_CSV = resolve_path("evaluations/metrics/overall_system_metrics.csv")
PER_CLASS_METRICS_CSV = resolve_path("evaluations/metrics/per_class_evaluation_matrix.csv")

# 77 Flow Features from CICIDS 2017 Dataset
FLOW_FEATURES_77 = [
    "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean", "Bwd Packet Length Std",
    "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "Fwd Header Length", "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s",
    "Min Packet Length", "Max Packet Length", "Packet Length Mean", "Packet Length Std", "Packet Length Variance",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count", "URG Flag Count",
    "CWE Flag Count", "ECE Flag Count", "Down/Up Ratio", "Average Packet Size",
    "Avg Fwd Segment Size", "Avg Bwd Segment Size", "Fwd Header Length.1",
    "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets", "Subflow Fwd Bytes", "Subflow Bwd Packets", "Subflow Bwd Bytes",
    "Init_Win_bytes_forward", "Init_Win_bytes_backward", "act_data_pkt_fwd", "min_seg_size_forward",
    "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min"
]

# Canonical Attack Classes (Clean representation without unicode replacement chars)
CANONICAL_ATTACK_CLASSES = [
    "Bot",
    "DDoS",
    "DoS GoldenEye",
    "DoS Hulk",
    "DoS Slowhttptest",
    "DoS slowloris",
    "FTP-Patator",
    "PortScan",
    "SSH-Patator",
    "Web Attack – Brute Force",
    "Web Attack – XSS"
]

# Severity Mapping for Threat Alerts
SEVERITY_MAP = {
    'BENIGN': ('NONE', '✅', 'safe'),
    'DDoS': ('CRITICAL', '🔴', 'critical'),
    'DoS Hulk': ('HIGH', '🟠', 'high'),
    'DoS GoldenEye': ('HIGH', '🟠', 'high'),
    'DoS slowloris': ('MEDIUM', '🟡', 'medium'),
    'DoS Slowhttptest': ('MEDIUM', '🟡', 'medium'),
    'PortScan': ('MEDIUM', '🟡', 'medium'),
    'Bot': ('HIGH', '🟠', 'high'),
    'FTP-Patator': ('HIGH', '🟠', 'high'),
    'SSH-Patator': ('HIGH', '🟠', 'high'),
    'Web Attack – Brute Force': ('HIGH', '🟠', 'high'),
    'Web Attack – XSS': ('MEDIUM', '🟡', 'medium'),
    'Web Attack \ufffd Brute Force': ('HIGH', '🟠', 'high'),
    'Web Attack \ufffd XSS': ('MEDIUM', '🟡', 'medium'),
    'Infiltration': ('CRITICAL', '🔴', 'critical'),
    'Heartbleed': ('CRITICAL', '🔴', 'critical'),
}

SEVERITY_COLOR = {
    'CRITICAL': '#ef4444',
    'HIGH': '#f97316',
    'MEDIUM': '#eab308',
    'NONE': '#2ec4b6',
}

# Incident Response Recommended Actions
ACTIONS_MAP = {
    'BENIGN': [
        'No mitigation required — packet matches standard baseline traffic.',
        'Log telemetry to historical flow registry for continuous baseline calibration.'
    ],
    'DDoS': [
        'Trigger upstream Anycast BGP blackholing for high-rate flood signatures.',
        'Activate Cloudflare / AWS Shield DDoS mitigation rate limits.',
        'Isolate target service subnet and deploy SYN cookie challenges.',
        'Alert SOC Incident Commander immediately for coordinated defense.'
    ],
    'DoS Hulk': [
        'Enforce per-IP request throttling on reverse proxy ingress.',
        'Inspect User-Agent headers and reject randomized keep-alive floods.',
        'Scale stateless web frontends horizontally behind load balancer.'
    ],
    'DoS GoldenEye': [
        'Limit maximum concurrent HTTP keep-alive connections per host.',
        'Drop persistent uncompleted HTTP requests exceeding 3000ms idle threshold.',
        'Apply connection concurrency caps in NGINX / Envoy proxy.'
    ],
    'DoS slowloris': [
        'Enforce minimum transfer speed requirements for client request headers.',
        'Lower client header timeout to 5 seconds on load balancer.',
        'Deploy reverse proxy buffer to prevent backend thread exhaustion.'
    ],
    'DoS Slowhttptest': [
        'Throttle slow HTTP POST body submissions with strict chunk timeouts.',
        'Enforce absolute socket lifetime caps for inbound client streams.'
    ],
    'PortScan': [
        'Drop source IP packets at border firewall for 1 hour via automated fail2ban.',
        'Verify target port exposure and close unused management ports (SSH/RDP).',
        'Check network deception honeypot triggers for lateral reconnaissance.'
    ],
    'Bot': [
        'Quarantine affected host from local VLAN to restrict C2 beaconing.',
        'Block Command-and-Control (C2) destination IP & domain on perimeter DNS firewall.',
        'Capture complete PCAP stream and initiate host endpoint forensic sweep.'
    ],
    'FTP-Patator': [
        'Temporarily block brute-forcing IP after 5 failed authentication attempts.',
        'Disable unencrypted FTP and migrate services to SFTP / FTPS with key-based auth.',
        'Verify credential leak dumps for compromised service accounts.'
    ],
    'SSH-Patator': [
        'Trigger fail2ban automated SSH iptables drop on repeated password authentication failures.',
        'Enforce SSH key-only authentication and disable remote root login.',
        'Restrict SSH port access to trusted management bastion VPN CIDRs.'
    ],
    'Web Attack – Brute Force': [
        'Enforce adaptive CAPTCHA challenge on authentication endpoints.',
        'Enable account lockout policies and rate-limit credential validation APIs.',
        'Audit web application firewall (WAF) rule efficacy against automated POST floods.'
    ],
    'Web Attack \ufffd Brute Force': [
        'Enforce adaptive CAPTCHA challenge on authentication endpoints.',
        'Enable account lockout policies and rate-limit credential validation APIs.',
        'Audit web application firewall (WAF) rule efficacy against automated POST floods.'
    ],
    'Web Attack – XSS': [
        'Inspect and sanitize HTTP GET/POST query parameters via WAF rules.',
        'Enforce Content Security Policy (CSP) headers across frontend web assets.',
        'Audit affected web routes for reflected or stored script injection vulnerabilities.'
    ],
    'Web Attack \ufffd XSS': [
        'Inspect and sanitize HTTP GET/POST query parameters via WAF rules.',
        'Enforce Content Security Policy (CSP) headers across frontend web assets.',
        'Audit affected web routes for reflected or stored script injection vulnerabilities.'
    ],
}
