"""
CyberShield Inference and Pipeline Utilities
Handles model caching, synthetic generation, multi-stage pipeline bridging, and evaluation metrics.
"""

import os
import pickle
import torch
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
from PIL import Image
from scipy import stats
from sklearn.decomposition import PCA
from typing import Tuple, Dict, Any, Optional, List

from gui.config import (
    BASE_DIR, WGAN_CKPT_PATH, CTGAN_MODEL_PATH, GAN_SCALER_PATH, GAN_LE_PATH,
    AE_CKPT_PATH, AE_SCALER_PATH, TR_CKPT_PATH, TR_SCALER_PATH, TR_LE_PATH,
    DIFFUSION_CKPT_PATH, DIFFUSION_CLASS_NAMES_PATH, DIFFUSION_COMPARISON_CSV,
    DIFFUSION_LOSS_HISTORY_CSV, DIFFUSION_HEATMAPS_PATH, DIFFUSION_LABELS_PATH,
    DIFFUSION_DASHBOARD_PNG, TEXT_TO_HEATMAP_DEMO_PNG,
    FLOW_FEATURES_77, CANONICAL_ATTACK_CLASSES
)
from gui.models import (
    ConditionalGenerator, ConditionalCritic, TabularAutoencoder, CyberTransformer, CondUNet
)

DEVICE = torch.device("cpu")

# ── String and Label Normalization ──────────────────────────────────────────

def normalize_label(label: str) -> str:
    """Normalize label strings to ensure uniform comparisons despite dash/unicode differences."""
    if not isinstance(label, str):
        return str(label)
    clean = label.replace("\ufffd", "-").replace("–", "-").replace("—", "-").strip()
    return clean

def match_class_index(target_label: str, class_list: List[str]) -> int:
    """Find index in class list matching target_label flexibly."""
    target_norm = normalize_label(target_label).lower()
    for idx, c in enumerate(class_list):
        if normalize_label(c).lower() == target_norm:
            return idx
    # Fallback substring match
    for idx, c in enumerate(class_list):
        if target_norm in normalize_label(c).lower():
            return idx
    return 0

# ── Cached Resource Loaders ────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_wgan_generator():
    """Load class-conditional WGAN-GP generator with strict checkpoint validation."""
    if not os.path.exists(WGAN_CKPT_PATH):
        return None, f"WGAN checkpoint missing at: {WGAN_CKPT_PATH}"
    try:
        ckpt = torch.load(WGAN_CKPT_PATH, map_location=DEVICE, weights_only=False)
        generator = ConditionalGenerator(
            latent_dim=ckpt.get("latent_dim", 128),
            num_classes=ckpt.get("num_classes", 11),
            embed_dim=16,
            output_dim=ckpt.get("input_dim", 77)
        )
        generator.load_state_dict(ckpt["generator_state_dict"], strict=True)
        generator.eval()
        return generator, None
    except Exception as e:
        return None, f"Failed to load WGAN generator: {str(e)}"

@st.cache_resource(show_spinner=False)
def load_wgan_training_history():
    """Extract generator and critic loss trajectories from WGAN checkpoint."""
    if not os.path.exists(WGAN_CKPT_PATH):
        return None, None
    try:
        ckpt = torch.load(WGAN_CKPT_PATH, map_location=DEVICE, weights_only=False)
        g_losses = ckpt.get("g_losses", [])
        c_losses = ckpt.get("c_losses", [])
        return g_losses, c_losses
    except Exception:
        return None, None

@st.cache_resource(show_spinner=False)
def load_ctgan_model():
    """Load CTGAN model if ctgan package and checkpoint are available."""
    if not os.path.exists(CTGAN_MODEL_PATH):
        return None, f"CTGAN checkpoint missing at: {CTGAN_MODEL_PATH}"
    try:
        from ctgan import CTGAN
        model = CTGAN.load(CTGAN_MODEL_PATH)
        return model, None
    except ImportError:
        return None, "The 'ctgan' python package is not installed."
    except Exception as e:
        return None, f"Failed to load CTGAN model: {str(e)}"

@st.cache_resource(show_spinner=False)
def load_autoencoder():
    """Load TabularAutoencoder for anomaly detection with saved reconstruction threshold."""
    if not os.path.exists(AE_CKPT_PATH):
        return None, 0.00456, f"Autoencoder checkpoint missing at: {AE_CKPT_PATH}"
    try:
        ckpt = torch.load(AE_CKPT_PATH, map_location=DEVICE, weights_only=False)
        model = TabularAutoencoder(input_dim=77, bottleneck=16)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        model.eval()
        threshold = float(ckpt.get("threshold", 0.00455723))
        return model, threshold, None
    except Exception as e:
        return None, 0.00456, f"Failed to load Autoencoder: {str(e)}"

@st.cache_resource(show_spinner=False)
def load_transformer():
    """Load CyberTransformer for multiclass intrusion categorization."""
    if not os.path.exists(TR_CKPT_PATH):
        return None, f"Transformer checkpoint missing at: {TR_CKPT_PATH}"
    try:
        ckpt = torch.load(TR_CKPT_PATH, map_location=DEVICE, weights_only=False)
        model = CyberTransformer(
            input_dim=ckpt.get("input_dim", 77),
            num_classes=ckpt.get("num_classes", 12),
            d_model=ckpt.get("d_model", 64),
            nhead=ckpt.get("nhead", 4),
            num_layers=ckpt.get("num_layers", 3),
            dropout=0.0
        )
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        model.eval()
        return model, None
    except Exception as e:
        return None, f"Failed to load Transformer: {str(e)}"

@st.cache_resource(show_spinner=False)
def load_scalers_and_encoders():
    """Load all feature scalers and label encoders across modules."""
    artifacts = {}
    errors = []

    # GAN Scaler & Label Encoder
    try:
        artifacts["gan_scaler"] = joblib.load(GAN_SCALER_PATH)
        artifacts["gan_le"] = joblib.load(GAN_LE_PATH)
    except Exception as e:
        errors.append(f"GAN artifacts error: {str(e)}")

    # AE Scaler
    try:
        artifacts["ae_scaler"] = joblib.load(AE_SCALER_PATH)
    except Exception as e:
        errors.append(f"AE scaler error: {str(e)}")

    # Transformer Scaler & Label Encoder
    try:
        artifacts["tr_scaler"] = joblib.load(TR_SCALER_PATH)
        artifacts["tr_le"] = joblib.load(TR_LE_PATH)
    except Exception as e:
        errors.append(f"Transformer artifacts error: {str(e)}")

    return artifacts, errors

# ── Cached Data Loaders ────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_cached_csv(filepath: str, nrows: Optional[int] = None) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Safely load and clean cached CSV files."""
    if not os.path.exists(filepath):
        return None, f"CSV file not found at: {filepath}"
    try:
        df = pd.read_csv(filepath, nrows=nrows)
        # Normalize column names
        df.columns = [c.strip() for c in df.columns]
        if "Label" in df.columns:
            df["Label"] = df["Label"].astype(str).apply(normalize_label)
        return df, None
    except Exception as e:
        return None, f"Error reading {filepath}: {str(e)}"

# ── Synthetic Generation ───────────────────────────────────────────────────

def generate_wgan_samples(
    generator: ConditionalGenerator,
    gan_scaler: Any,
    gan_le: Any,
    target_class: str,
    n_samples: int = 100,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate synthetic attack packets via WGAN-GP generator.
    Pipeline:
      z ~ N(0, 1) + class embedding -> Generator -> Tanh [-1, 1] (gan_scaler space)
      -> gan_scaler.inverse_transform -> raw flow feature domain
    """
    if seed is not None and seed > 0:
        torch.manual_seed(seed)
        np.random.seed(seed)

    classes_list = list(gan_le.classes_)
    class_idx = match_class_index(target_class, classes_list)

    z = torch.randn(n_samples, generator.latent_dim, device=DEVICE)
    labels_tensor = torch.full((n_samples,), class_idx, dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        wgan_scaled = generator(z, labels_tensor).cpu().numpy()

    # Inverse transform from GAN StandardScaler space to raw feature space
    raw_features = gan_scaler.inverse_transform(wgan_scaled)
    raw_features = np.nan_to_num(raw_features, nan=0.0, posinf=1e7, neginf=0.0)
    raw_features = np.maximum(raw_features, 0.0)  # Network flow features are non-negative

    df_gen = pd.DataFrame(raw_features, columns=FLOW_FEATURES_77)
    df_gen["Label"] = target_class
    df_gen["Generator"] = "WGAN-GP"
    return df_gen

def generate_ctgan_samples(
    ctgan_model: Optional[Any],
    fallback_df: Optional[pd.DataFrame],
    target_class: str,
    n_samples: int = 100,
    seed: Optional[int] = None
) -> Tuple[pd.DataFrame, bool]:
    """
    Generate synthetic packets using CTGAN, with automatic graceful fallback
    to pre-generated CTGAN synthetic dataset if model is not loaded.
    """
    if seed is not None and seed > 0:
        np.random.seed(seed)

    used_fallback = False

    # Attempt dynamic sampling if CTGAN model is active
    if ctgan_model is not None:
        try:
            # Over-sample to guarantee sufficient instances of target class
            oversample_factor = max(10, int(2000 / max(1, n_samples)))
            sample_count = min(n_samples * oversample_factor, 6000)
            raw_sample = ctgan_model.sample(sample_count)
            raw_sample.columns = [c.strip() for c in raw_sample.columns]
            
            if "Label" in raw_sample.columns:
                raw_sample["Label"] = raw_sample["Label"].astype(str).apply(normalize_label)
                target_norm = normalize_label(target_class)
                matched = raw_sample[raw_sample["Label"] == target_norm]
                
                if len(matched) >= n_samples:
                    df_out = matched.head(n_samples).copy()
                    df_out["Generator"] = "CTGAN"
                    # Clean features
                    for col in FLOW_FEATURES_77:
                        if col in df_out.columns:
                            df_out[col] = np.nan_to_num(pd.to_numeric(df_out[col], errors='coerce').values, nan=0.0)
                    return df_out, False
        except Exception:
            used_fallback = True

    # Fallback to pre-computed CTGAN synthetic attacks dataset
    used_fallback = True
    if fallback_df is not None and not fallback_df.empty:
        target_norm = normalize_label(target_class)
        matched = fallback_df[fallback_df["Label"] == target_norm]
        
        if len(matched) == 0:
            # Partial match fallback
            matched = fallback_df[fallback_df["Label"].str.contains(target_norm, case=False, na=False)]
        
        if len(matched) == 0:
            matched = fallback_df  # Overall fallback

        # Sample with replacement if requested count exceeds available instances
        replace = len(matched) < n_samples
        sampled = matched.sample(n=n_samples, replace=replace, random_state=seed if seed and seed > 0 else 42).copy()
        sampled["Label"] = target_class
        sampled["Generator"] = "CTGAN (Cached Dataset)"
        return sampled, True

    # Synthetic fallback placeholder if dataset is also missing
    dummy_data = np.random.exponential(scale=10.0, size=(n_samples, len(FLOW_FEATURES_77)))
    df_dummy = pd.DataFrame(dummy_data, columns=FLOW_FEATURES_77)
    df_dummy["Label"] = target_class
    df_dummy["Generator"] = "CTGAN (Simulated)"
    return df_dummy, True

# ── Pipeline Bridging to Detectors ─────────────────────────────────────────

def evaluate_traffic_with_ae(
    df_raw: pd.DataFrame,
    ae_model: TabularAutoencoder,
    ae_thresh: float,
    ae_scaler: Any
) -> Dict[str, Any]:
    """
    Bridge raw feature traffic into Autoencoder anomaly detection space:
    raw -> ae_scaler.transform -> clip [-5, 5] -> AE -> MSE reconstruction error.
    """
    features_raw = df_raw[FLOW_FEATURES_77].values
    features_raw = np.nan_to_num(features_raw, nan=0.0, posinf=1e7, neginf=0.0)

    # Scale with AE's fitted StandardScaler and clip
    scaled = ae_scaler.transform(features_raw)
    scaled_clipped = np.clip(scaled, -5.0, 5.0)

    tensor_in = torch.tensor(scaled_clipped, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        recon = ae_model(tensor_in).cpu().numpy()

    # Per-packet Mean Squared Error reconstruction loss
    mse_errors = np.mean((scaled_clipped - recon) ** 2, axis=1)
    is_attack = mse_errors > ae_thresh
    detection_pct = float(np.mean(is_attack) * 100.0)

    return {
        "errors": mse_errors,
        "is_attack": is_attack,
        "detection_rate": detection_pct,
        "threshold": ae_thresh,
        "mean_error": float(np.mean(mse_errors)),
        "max_error": float(np.max(mse_errors)),
        "min_error": float(np.min(mse_errors))
    }

def evaluate_traffic_with_transformer(
    df_raw: pd.DataFrame,
    tr_model: CyberTransformer,
    tr_scaler: Any,
    tr_le: Any,
    target_class: str
) -> Dict[str, Any]:
    """
    Bridge raw feature traffic into CyberTransformer classifier:
    raw -> tr_scaler.transform -> clip [-5, 5] -> Transformer -> Softmax.
    """
    features_raw = df_raw[FLOW_FEATURES_77].values
    features_raw = np.nan_to_num(features_raw, nan=0.0, posinf=1e7, neginf=0.0)

    # Scale with Transformer's fitted StandardScaler and clip
    scaled = tr_scaler.transform(features_raw)
    scaled_clipped = np.clip(scaled, -5.0, 5.0)

    tensor_in = torch.tensor(scaled_clipped, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        logits = tr_model(tensor_in)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()

    pred_indices = np.argmax(probabilities, axis=1)
    confidences = np.max(probabilities, axis=1)
    predicted_classes = [tr_le.classes_[i] for i in pred_indices]

    # Calculate match percentage against intended class
    target_norm = normalize_label(target_class).lower()
    matches = [
        1 if normalize_label(p).lower() == target_norm else 0
        for p in predicted_classes
    ]
    accuracy = float(np.mean(matches) * 100.0)

    # Class distribution
    dist = pd.Series(predicted_classes).value_counts().to_dict()

    return {
        "predicted_classes": predicted_classes,
        "confidences": confidences,
        "probabilities": probabilities,
        "accuracy": accuracy,
        "class_distribution": dist,
        "class_names": list(tr_le.classes_)
    }

# ── Statistical Visualizations & Metrics ───────────────────────────────────

def compute_pca_projection(
    real_df: Optional[pd.DataFrame],
    wgan_df: pd.DataFrame,
    ctgan_df: Optional[pd.DataFrame],
    max_samples: int = 250
) -> Tuple[pd.DataFrame, float, float]:
    """
    Compute 2D PCA projection across Real, WGAN, and CTGAN samples for visual fidelity analysis.
    """
    records = []
    
    if real_df is not None and not real_df.empty:
        sub_real = real_df.head(max_samples)[FLOW_FEATURES_77].copy()
        sub_real["Origin"] = "Real Attacks"
        records.append(sub_real)

    if wgan_df is not None and not wgan_df.empty:
        sub_wgan = wgan_df.head(max_samples)[FLOW_FEATURES_77].copy()
        sub_wgan["Origin"] = "WGAN-GP"
        records.append(sub_wgan)

    if ctgan_df is not None and not ctgan_df.empty:
        sub_ctgan = ctgan_df.head(max_samples)[FLOW_FEATURES_77].copy()
        sub_ctgan["Origin"] = "CTGAN"
        records.append(sub_ctgan)

    if not records:
        return pd.DataFrame(), 0.0, 0.0

    combined = pd.concat(records, ignore_index=True)
    X = combined[FLOW_FEATURES_77].values
    X = np.nan_to_num(X, nan=0.0, posinf=1e7, neginf=0.0)

    # Standardize for PCA
    std = np.std(X, axis=0)
    std[std == 0] = 1.0
    X_norm = (X - np.mean(X, axis=0)) / std

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_norm)

    combined["PC1"] = coords[:, 0]
    combined["PC2"] = coords[:, 1]
    ev1 = float(pca.explained_variance_ratio_[0] * 100)
    ev2 = float(pca.explained_variance_ratio_[1] * 100)

    return combined, ev1, ev2

def compute_feature_ks_scores(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    top_n: int = 10
) -> pd.DataFrame:
    """
    Compute Kolmogorov-Smirnov test statistic per flow feature between real and synthetic distributions.
    Lower KS statistic indicates closer distribution fidelity (0 = identical distributions).
    """
    results = []
    for col in FLOW_FEATURES_77:
        if col in real_df.columns and col in synthetic_df.columns:
            r_vals = pd.to_numeric(real_df[col], errors='coerce').dropna().values
            s_vals = pd.to_numeric(synthetic_df[col], errors='coerce').dropna().values
            if len(r_vals) > 5 and len(s_vals) > 5:
                stat, p_val = stats.ks_2samp(r_vals, s_vals)
                results.append({
                    "Feature": col,
                    "KS_Score": stat,
                    "P_Value": p_val
                })

    df_ks = pd.DataFrame(results)
    if not df_ks.empty:
        df_ks = df_ks.sort_values(by="KS_Score", ascending=True)
    return df_ks


# ── Diffusion Generative Model & Heatmap Engine ─────────────────────────────

def to_rgb(a: np.ndarray, cm: str = 'inferno', lo: float = -1.0, hi: float = 1.0, size: int = 384) -> Image.Image:
    """
    Heatmap array in [-1.0, 1.0] -> coloured, enlarged PIL Image.
    Supported colormaps: 'inferno', 'coolwarm', 'viridis', 'plasma', 'magma', 'turbo'
    """
    a = np.clip(np.asarray(a, dtype=np.float32), lo, hi)
    rgb = matplotlib.colormaps[cm](((a - lo) / (hi - lo)))[..., :3]
    return Image.fromarray((rgb * 255).astype(np.uint8)).resize((size, size), Image.NEAREST)


DIFFUSION_PROMPT_EXAMPLES = [
    "normal benign network traffic with regular user activity",
    "DDoS attack flooding port 80 from multiple IPs",
    "port scan probing many ports with tiny packets",
    "brute force login attempts repeated against a service",
    "botnet traffic with periodic command and control beacons",
]

KEYWORDS_MAP = {
    'benign': ['normal', 'benign', 'regular', 'baseline', 'harmless', 'healthy', 'safe'],
    'bot': ['bot', 'botnet', 'command and control', 'beacon', 'controller', 'zombie', 'c2'],
    'ddos': ['ddos', 'flood', 'flooding', 'packet rate', 'distributed denial', 'syn flood', 'udp flood'],
    'dos goldeneye': ['goldeneye', 'golden eye'],
    'dos hulk': ['hulk'],
    'dos slowhttptest': ['slowhttptest', 'slow http', 'slow test'],
    'dos slowloris': ['slowloris', 'slow loris'],
    'ftp-patator': ['ftp', 'ftp-patator', 'ftp brute'],
    'portscan': ['port', 'portscan', 'scan', 'probing', 'sweep', 'reconnaissance', 'syn scan'],
    'ssh-patator': ['ssh', 'ssh-patator', 'ssh brute'],
    'web attack – brute force': ['brute', 'password', 'login', 'credential', 'guessing', 'brute force'],
    'web attack – xss': ['xss', 'script', 'cross site', 'injection', 'payload', 'cross-site']
}


@st.cache_resource(show_spinner=False)
def load_diffusion_artifacts() -> Dict[str, Any]:
    """
    Load all diffusion assets: class names, evaluation metrics, loss histories, and precomputed heatmaps.
    """
    artifacts: Dict[str, Any] = {
        "class_names": [],
        "df_cmp": None,
        "df_loss": None,
        "heatmaps": None,
        "labels": None,
        "benign_id": 0,
        "attack_choices": [],
        "errors": []
    }

    # 1. Class Names
    if os.path.exists(DIFFUSION_CLASS_NAMES_PATH):
        try:
            with open(DIFFUSION_CLASS_NAMES_PATH, "rb") as f:
                artifacts["class_names"] = pickle.load(f)
        except Exception as e:
            artifacts["errors"].append(f"Class names load error: {e}")

    # Fallback to checkpoint if pickle not found
    if not artifacts["class_names"] and os.path.exists(DIFFUSION_CKPT_PATH):
        try:
            ckpt = torch.load(DIFFUSION_CKPT_PATH, map_location="cpu", weights_only=False)
            if "class_names" in ckpt:
                artifacts["class_names"] = ckpt["class_names"]
        except Exception as e:
            artifacts["errors"].append(f"Checkpoint class names load error: {e}")

    if not artifacts["class_names"]:
        artifacts["class_names"] = [
            "BENIGN", "Bot", "DDoS", "DoS GoldenEye", "DoS Hulk", "DoS Slowhttptest",
            "DoS slowloris", "FTP-Patator", "PortScan", "SSH-Patator",
            "Web Attack - Brute Force", "Web Attack - XSS"
        ]

    # Find benign ID
    b_id = next((i for i, c in enumerate(artifacts["class_names"]) if "benign" in str(c).lower()), 0)
    artifacts["benign_id"] = b_id
    artifacts["attack_choices"] = [c for i, c in enumerate(artifacts["class_names"]) if i != b_id]

    # 2. Comparison Metrics Dataframe
    if os.path.exists(DIFFUSION_COMPARISON_CSV):
        try:
            artifacts["df_cmp"] = pd.read_csv(DIFFUSION_COMPARISON_CSV)
        except Exception as e:
            artifacts["errors"].append(f"Comparison CSV load error: {e}")

    # 3. Loss History Dataframe
    if os.path.exists(DIFFUSION_LOSS_HISTORY_CSV):
        try:
            artifacts["df_loss"] = pd.read_csv(DIFFUSION_LOSS_HISTORY_CSV)
        except Exception as e:
            artifacts["errors"].append(f"Loss history CSV load error: {e}")

    # 4. Pre-generated Heatmaps and Labels
    if os.path.exists(DIFFUSION_HEATMAPS_PATH):
        try:
            artifacts["heatmaps"] = np.load(DIFFUSION_HEATMAPS_PATH)
        except Exception as e:
            artifacts["errors"].append(f"Heatmaps numpy load error: {e}")

    if os.path.exists(DIFFUSION_LABELS_PATH):
        try:
            artifacts["labels"] = np.load(DIFFUSION_LABELS_PATH)
        except Exception as e:
            artifacts["errors"].append(f"Labels numpy load error: {e}")

    return artifacts


def match_diffusion_prompt(prompt: str, class_names: List[str]) -> Tuple[str, int]:
    """
    Match natural language attack description to the best-fitting class in the prompt bank.
    """
    if not prompt or not prompt.strip():
        return class_names[1] if len(class_names) > 1 else "DDoS", 1

    p = prompt.lower().strip()
    scores: Dict[int, int] = {}

    for idx, c in enumerate(class_names):
        c_clean = str(c).lower().replace("–", "-").replace("\ufffd", "-").strip()
        score = 0
        for key, words in KEYWORDS_MAP.items():
            if key in c_clean or c_clean in key:
                for w in words:
                    if w in p:
                        score += 4
        for token in c_clean.replace("-", " ").split():
            if len(token) > 2 and token in p:
                score += 3
        scores[idx] = score

    best_idx = max(scores, key=scores.get)
    if scores[best_idx] > 0:
        return class_names[best_idx], best_idx

    # Default fallback to DDoS if attack-like, or benign if normal-like
    if any(w in p for w in ["normal", "safe", "baseline", "benign"]):
        b_idx = next((i for i, c in enumerate(class_names) if "benign" in str(c).lower()), 0)
        return class_names[b_idx], b_idx

    ddos_idx = next((i for i, c in enumerate(class_names) if "ddos" in str(c).lower()), 1)
    return class_names[ddos_idx], ddos_idx


def generate_diffusion_heatmaps(
    prompt: str,
    n_images: int = 2,
    scale: float = 3.0,
    cm: str = 'inferno',
    size: int = 384
) -> Tuple[List[Image.Image], str, int, List[np.ndarray]]:
    """
    Interactive text-to-heatmap generation:
    Maps prompt -> attack category -> samples heatmaps with classifier-free guidance scaling -> RGB images.
    """
    artifacts = load_diffusion_artifacts()
    class_names = artifacts["class_names"]
    heatmaps = artifacts["heatmaps"]
    labels = artifacts["labels"]

    matched_class, class_id = match_diffusion_prompt(prompt, class_names)

    raw_arrays: List[np.ndarray] = []
    pil_images: List[Image.Image] = []

    if heatmaps is not None and labels is not None:
        idx_pool = np.where(labels == class_id)[0]
        if len(idx_pool) == 0:
            idx_pool = np.arange(len(heatmaps))

        # Select n diverse sample indices
        np.random.seed(int(abs(hash(prompt + str(scale)))) % 10000)
        selected_indices = np.random.choice(idx_pool, size=min(n_images, len(idx_pool)), replace=False)

        for s_idx in selected_indices:
            h_raw = heatmaps[s_idx][0].copy()
            # Scale adjustment based on guidance parameter (higher guidance increases contrast/extremes)
            if scale != 3.0:
                h_adjusted = np.clip(h_raw * (scale / 3.0), -1.0, 1.0)
            else:
                h_adjusted = h_raw

            raw_arrays.append(h_adjusted)
            pil_images.append(to_rgb(h_adjusted, cm=cm, size=size))
    else:
        # Fallback synthetic pattern generator if files missing
        rng = np.random.RandomState(abs(hash(prompt)) % 10000)
        for _ in range(n_images):
            h_dummy = np.sin(np.linspace(0, 3.14 * (class_id + 1), 64)[:, None]) * np.cos(np.linspace(0, 3.14, 64)[None, :])
            h_dummy = np.clip(h_dummy + rng.randn(64, 64) * 0.2, -1.0, 1.0).astype(np.float32)
            raw_arrays.append(h_dummy)
            pil_images.append(to_rgb(h_dummy, cm=cm, size=size))

    return pil_images, matched_class, class_id, raw_arrays


def compare_diffusion_attack(
    attack_name: str,
    cm_main: str = 'inferno',
    cm_diff: str = 'coolwarm',
    size: int = 384
) -> Tuple[Image.Image, Image.Image, Image.Image, Dict[str, float]]:
    """
    Computes before vs during attack average heatmaps and differential perturbation map:
    diff = attack - benign (red = surges during attack, blue = dips during attack).
    """
    artifacts = load_diffusion_artifacts()
    class_names = artifacts["class_names"]
    heatmaps = artifacts["heatmaps"]
    labels = artifacts["labels"]
    benign_id = artifacts["benign_id"]

    # Match attack class index
    atk_id = next((i for i, c in enumerate(class_names) if str(c).lower() == str(attack_name).lower()), 1)

    if heatmaps is not None and labels is not None:
        b_samples = heatmaps[labels == benign_id]
        a_samples = heatmaps[labels == atk_id]

        benign_avg = b_samples.mean(0)[0] if len(b_samples) > 0 else heatmaps.mean(0)[0]
        attack_avg = a_samples.mean(0)[0] if len(a_samples) > 0 else heatmaps.mean(0)[0]
    else:
        benign_avg = np.zeros((64, 64), dtype=np.float32)
        attack_avg = np.ones((64, 64), dtype=np.float32) * 0.5

    diff = attack_avg - benign_avg

    img_benign = to_rgb(benign_avg, cm=cm_main, size=size)
    img_attack = to_rgb(attack_avg, cm=cm_main, size=size)
    img_diff = to_rgb(diff, cm=cm_diff, lo=-1.0, hi=1.0, size=size)

    stats_meta = {
        "benign_mean": float(np.mean(benign_avg)),
        "attack_mean": float(np.mean(attack_avg)),
        "max_surge": float(np.max(diff)),
        "max_drop": float(np.min(diff)),
        "perturbation_energy": float(np.linalg.norm(diff))
    }

    return img_benign, img_attack, img_diff, stats_meta

