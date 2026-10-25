# 🛡️ CyberShield GAN GUI Module

**CyberShield** is an advanced generative AI cyber defense framework on the **CICIDS 2017** network intrusion dataset (77 numeric flow features, BENIGN + multi-class attack categories).

This graphical dashboard provides a comprehensive live demo suite specifically designed for project guide presentation, highlighting the role of Generative Adversarial Networks (WGAN-GP and CTGAN) in synthesizing realistic attack traffic and stress-testing defense mechanisms.

---

## 🚀 How to Run the Application

### 1. Prerequisites & Environment Setup
Ensure Python 3.10+ is installed with the required dependencies:
```bash
pip install -r gui/requirements.txt
```

### 2. Launch the Streamlit GUI
Run the Streamlit application from the root project folder:
```bash
streamlit run gui/app.py
```
*Alternatively, you can also launch it via PowerShell / CMD:*
```bash
python -m streamlit run gui/app.py
```

The application runs entirely on **CPU**, starts within seconds, and opens automatically in your web browser at `http://localhost:8501`.

---

## 📂 Architecture & File Structure

```
Cyber Shield/
├── gui/
│   ├── app.py                      # Streamlit application main entrypoint & navigation
│   ├── config.py                   # Centralized paths, canonical labels, severity & action maps
│   ├── models.py                   # Strict PyTorch definitions (Generator, Critic, AE, Transformer)
│   ├── inference.py                # Cached loaders, WGAN generation, CTGAN sampling, pipeline bridges
│   ├── components.py               # Cyberpunk dark theme CSS, hero headers, metric cards, threat reports
│   ├── requirements.txt            # Python package specifications
│   ├── README.md                   # This documentation guide
│   └── pages/
│       ├── __init__.py
│       ├── page1_overview.py       # Pipeline walkthrough, counterfeiter analogy, saved metrics
│       ├── page2_generator.py      # Interactive WGAN-GP / CTGAN packet generation & CSV download
│       ├── page3_real_vs_synthetic.py # 2D PCA, feature histogram overlay, KS statistics
│       ├── page4_stress_test.py    # Key Demo: Live AE anomaly detection & Transformer classification
│       └── page5_wgan_training.py  # 100-Epoch dual loss curves & Wasserstein distance explanation
├── cybershield_wgan.pth            # Trained WGAN-GP generator & critic checkpoint
├── cybershield_ctgan.pkl           # Trained CTGAN tabular model
├── gan_scaler.pkl                  # GAN StandardScaler fitted on attack flows
├── gan_label_encoder.pkl           # 11 Attack classes label encoder
├── wgan_synthetic_attacks.csv      # Pre-generated 4,994 WGAN synthetic samples
├── ctgan_synthetic_attacks.csv     # Pre-generated 5,000 CTGAN synthetic samples
├── gan_comparison.csv              # KS scores and statistical mean/std differences
├── gan_dashboard.png               # Pre-computed GAN evaluation dashboard
└── models/
    ├── cybershield_ae.pth          # Trained Autoencoder checkpoint (threshold: 0.004557)
    ├── ae_scaler.pkl               # Autoencoder feature scaler
    ├── cybershield_transformer.pth # Trained 12-class CyberTransformer classifier
    ├── transformer_scaler.pkl      # Transformer feature scaler
    └── transformer_label_encoder.pkl # Transformer 12-class label encoder (BENIGN + 11 attacks)
```

---

## 🖥️ The 5 Pages of the CyberShield GUI

### Page 1: Overview & Generative Defense Pipeline
- **Pipeline Architecture:** Visualizes the full multi-stage journey: **AE → VAE → GAN → Transformer → Diffusion (Planned)**.
- **The Counterfeiter Analogy:** Clearly explains the adversarial game between the **Generator (Counterfeiter)** minting fake network traffic and the **Critic (Detective)** inspecting statistical flow signatures.
- **Saved Diagnostics:** Displays historical diagnostic dashboards for the Autoencoder, CVAE latent space, and Transformer confusion matrix / self-attention heatmaps.

### Page 2: Interactive GAN Generator
- **Flexible Controls:** Choose between **WGAN-GP**, **CTGAN**, or **Both (Comparative Batch)**.
- **Conditioning:** Dropdown across 11 attack classes (`Bot`, `DDoS`, `DoS GoldenEye`, `DoS Hulk`, `FTP-Patator`, `PortScan`, `SSH-Patator`, etc.).
- **Live Output:** Interactive preview of the first 20 packets, full summary statistics (mean, std, min, max across 77 flow dimensions), and a **Download CSV** button.

### Page 3: Real vs. Synthetic Statistical Fidelity
- **2D PCA Projection:** Interactive Plotly scatter plot projecting Real Attacks, WGAN-GP, and CTGAN into 2D principal component space to verify cluster manifold alignment.
- **Feature Distribution Overlays:** Select any of the 77 flow features to compare probability density histograms between real and generated packets.
- **Kolmogorov-Smirnov (KS) Scores:** Quantifies statistical two-sample divergence across flow metrics.
- **Pre-computed Dashboard:** Embedded high-resolution `gan_dashboard.png` and `gan_comparison.csv`.

### Page 4: Adversarial Stress Test (Key Guide Demo)
- **Live Dual-Detector Benchmark:** Generate $N$ packets of an attack class on the fly, then route them through both defensive checkpoints:
  1. **Autoencoder (Anomaly Detector):** Measures per-packet Mean Squared Error (MSE) reconstruction loss against the threshold $\tau = 0.004557$. Displays detection percentage and error distribution histogram.
  2. **CyberTransformer (Intrusion Classifier):** Measures multiclass prediction distribution and classification confidence across 12 flow categories.
- **Automated SOC Incident Report:** Detailed threat report card for any selected packet, including severity badge (`CRITICAL`, `HIGH`, `MEDIUM`), confidence gauge, and recommended incident response playbook actions.

### Page 5: WGAN-GP Training Dynamics
- **100-Epoch Dual Loss Curves:** Plots Generator loss and Critic loss (Wasserstein estimate) directly from checkpoint data.
- **Theoretical Explanation:** Explains Earth Mover's Distance, absence of vanishing gradients, and how the Gradient Penalty ($\lambda = 10$) prevents mode collapse.

---

## 🔬 Technical Details & Multi-Module Pipeline Bridging

### 1. Generator Tanh Output Range
The `ConditionalGenerator` ends with a `nn.Tanh()` activation, bounding raw output values in `gan_scaler` space to $[-1.0, 1.0]$. Because real training flows in `gan_scaler` space span $[-5.0, 5.0]$, synthetic features exhibit compressed distribution tails. As per project guidelines, model weights remain untouched, and this behavior is documented in the UI's Technical Notes.

### 2. Scaler Space Bridging Formula
Each defense subsystem in CyberShield was trained independently with its own feature scaler. To route synthetic traffic from the GAN to the detectors without distortion:
$$\mathbf{X}_{\text{raw}} = \text{gan\_scaler}^{-1}\left(G(z, c)\right)$$
$$\mathbf{X}_{\text{detector}} = \text{clip}\left(\text{scaler}_{\text{detector}}\left(\mathbf{X}_{\text{raw}}\right), -5.0, 5.0\right)$$

This mathematical bridging ensures that both the Autoencoder and Transformer evaluate realistic physical network flow magnitudes.
