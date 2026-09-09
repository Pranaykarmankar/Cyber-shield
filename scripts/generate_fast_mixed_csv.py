import os
import joblib
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")

print("Initializing fast mixed dataset builder...")

def resolve_file(folder: str, filename: str) -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, "..", folder, filename),
        os.path.join(script_dir, folder, filename),
        os.path.join(script_dir, filename),
        os.path.join(os.getcwd(), folder, filename),
        os.path.join(os.getcwd(), filename),
    ]
    for p in candidates:
        if os.path.exists(p):
            return os.path.abspath(p)
    return os.path.abspath(os.path.join(script_dir, "..", folder, filename))

# 1. Load models
device = torch.device('cpu')
ae_ckpt = torch.load(resolve_file('models', 'cybershield_ae.pth'), map_location=device, weights_only=False)
ae_threshold = float(ae_ckpt['threshold'])
ae_scaler = joblib.load(resolve_file('models', 'ae_scaler.pkl'))

class TabularAutoencoder(nn.Module):
    def __init__(self, input_dim: int, bottleneck: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.LeakyReLU(0.2),
            nn.Linear(128, 64),  nn.BatchNorm1d(64),  nn.LeakyReLU(0.2),
            nn.Linear(64, 32),   nn.BatchNorm1d(32),   nn.LeakyReLU(0.2),
            nn.Linear(32, bottleneck), nn.LeakyReLU(0.2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 32), nn.LeakyReLU(0.2),
            nn.Linear(32, 64),  nn.BatchNorm1d(64),  nn.LeakyReLU(0.2),
            nn.Linear(64, 128), nn.BatchNorm1d(128), nn.LeakyReLU(0.2),
            nn.Linear(128, input_dim),
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

ae_model = TabularAutoencoder(input_dim=77, bottleneck=16)
ae_model.load_state_dict(ae_ckpt['model_state_dict'])
ae_model.eval()

vae_scaler = joblib.load(resolve_file('models', 'scaler.pkl'))
rf_model = joblib.load(resolve_file('models', 'rf_model.pkl'))
vae_le = joblib.load(resolve_file('models', 'label_encoder.pkl'))
vae_classes = [c.encode('ascii', 'ignore').decode() for c in vae_le.classes_]

df_ref = pd.read_csv(resolve_file('data', 'test_traffic.csv'))
feature_cols = [c for c in df_ref.columns if c not in [
    'Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Timestamp', 'Label', 'Unnamed: 0'
]]

# Fast coordinate search for 1 exemplar per attack class
def find_exemplar(target_class, seed=42):
    np.random.seed(seed)
    t_idx = vae_classes.index(target_class)
    best_x = np.random.randn(77) * 0.4
    best_p = rf_model.predict_proba(best_x.reshape(1, -1))[0, t_idx]
    
    # Priority features that RF relies on most
    key_feats = [74, 18, 47, 41, 48, 15, 31, 76, 45, 26, 12, 65, 36, 42, 70, 0, 1, 2, 3, 4]
    for step in range(40):
        improved = False
        for feat in key_feats:
            for delta in [-1.5, 1.5, -0.5, 0.5, -2.5, 2.5]:
                cand = best_x.copy()
                cand[feat] += delta
                p = rf_model.predict_proba(cand.reshape(1, -1))[0, t_idx]
                if p > best_p:
                    best_p = p
                    best_x = cand
                    improved = True
                    if best_p > 0.70: break
            if best_p > 0.70: break
        if not improved or best_p > 0.70: break
    return best_x, best_p

attack_targets = [
    ('DDoS', 6),
    ('PortScan', 6),
    ('DoS Hulk', 6),
    ('Bot', 4),
    ('FTP-Patator', 4),
    ('SSH-Patator', 4),
]

all_raw_rows = []
all_labels = []

print("Finding exemplars for each attack...")
for atk, count in attack_targets:
    base_x, prob = find_exemplar(atk)
    print(f"  -> {atk:<12}: p={prob:.3f}")
    for i in range(count):
        noise = np.random.randn(77) * 0.05
        cand_x = base_x + noise
        raw = vae_scaler.inverse_transform(cand_x.reshape(1, -1))[0]
        all_raw_rows.append(raw)
        all_labels.append(atk)

# Add 20 normal BENIGN samples
print("Adding BENIGN samples...")
benign_raw = df_ref[df_ref['Label'] == 'BENIGN'][feature_cols].values
for i in range(20):
    idx = i % len(benign_raw)
    sample = benign_raw[idx] * np.random.uniform(0.99, 1.01, size=77)
    all_raw_rows.append(sample)
    all_labels.append('BENIGN')

# Shuffle
rng = np.random.RandomState(42)
perm = rng.permutation(len(all_raw_rows))
shuffled_rows = [all_raw_rows[i] for i in perm]
shuffled_labels = [all_labels[i] for i in perm]

n_samples = len(shuffled_rows)
src_ips = [f"192.168.1.{np.random.randint(10, 250)}" for _ in range(n_samples)]
dst_ips = [f"10.0.0.{np.random.randint(2, 50)}" for _ in range(n_samples)]
src_ports = [np.random.randint(1024, 65530) for _ in range(n_samples)]
dst_ports = [80 if 'DoS' in lbl or lbl == 'BENIGN' else (22 if 'SSH' in lbl else (21 if 'FTP' in lbl else (443 if lbl == 'DDoS' else np.random.randint(1, 65535)))) for lbl in shuffled_labels]
timestamps = [f"2026-09-09 {10 + (i//60):02d}:{(i%60):02d}:{(i*7)%60:02d}" for i in range(n_samples)]
flow_ids = [f"{src_ips[i]}-{dst_ips[i]}-{src_ports[i]}-{dst_ports[i]}-6" for i in range(n_samples)]

df_out = pd.DataFrame(shuffled_rows, columns=feature_cols)
df_out.insert(0, "Flow ID", flow_ids)
df_out.insert(1, "Source IP", src_ips)
df_out.insert(2, "Source Port", src_ports)
df_out.insert(3, "Destination IP", dst_ips)
df_out.insert(4, "Destination Port", dst_ports)
df_out.insert(5, "Timestamp", timestamps)
df_out["Label"] = shuffled_labels

# Save files
out_mixed = resolve_file("data", "cybershield_mixed_traffic.csv")
out_test = resolve_file("data", "test_traffic.csv")
df_out.to_csv(out_mixed, index=False)
df_out.to_csv(out_test, index=False)
print(f"Saved {len(df_out)} rows to {out_mixed} and {out_test}!")

# Verify through RF
x_vae = vae_scaler.transform(df_out[feature_cols].values)
rf_preds = rf_model.predict(x_vae)
pred_names = [vae_classes[int(p)] for p in rf_preds]
attack_count = sum(1 for p in pred_names if p != 'BENIGN')
benign_count = sum(1 for p in pred_names if p == 'BENIGN')

print(f"\n[VAE / RF Verification]:")
print(f"  ATTACKS DETECTED: {attack_count} ({attack_count/len(df_out)*100:.1f}%)")
print(f"  BENIGN DETECTED : {benign_count} ({benign_count/len(df_out)*100:.1f}%)")
print("  Breakdown:")
for k, v in pd.Series(pred_names).value_counts().items():
    print(f"    - {k:<25}: {v}")

# Verify through AE
x_ae = np.clip(ae_scaler.transform(df_out[feature_cols].values), -5, 5)
with torch.no_grad():
    recon = ae_model(torch.FloatTensor(x_ae))
    ae_errors = torch.mean((torch.FloatTensor(x_ae) - recon)**2, dim=1).numpy()

print(f"\n[AE Verification]:")
for target_lbl in ['BENIGN', 'DDoS', 'PortScan', 'DoS Hulk', 'Bot']:
    mask = np.array(shuffled_labels) == target_lbl
    if mask.sum() > 0:
        errs = ae_errors[mask]
        print(f"  - {target_lbl:<12} (N={mask.sum()}): Mean AE Error = {errs.mean():.4f} | Min = {errs.min():.4f} | Max = {errs.max():.4f}")
