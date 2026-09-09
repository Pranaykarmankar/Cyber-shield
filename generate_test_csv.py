"""
Generate a realistic test CSV for CyberShield testing.
Creates 50 rows with CICIDS 2017 features:
  - Normal (BENIGN) traffic
  - Multiple attack types: DDoS, DoS Hulk, PortScan, Bot,
    SSH-Patator, FTP-Patator, Web Attack – Brute Force, etc.

The CSV includes metadata columns (Flow ID, Source IP, etc.)
that the app will automatically drop before inference.
"""

import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

# ── CICIDS 2017 Feature Names (77 features, in standard order) ──────────
FEATURE_NAMES = [
    "Destination Port", "Flow Duration", "Total Fwd Packets",
    "Total Backward Packets", "Total Length of Fwd Packets",
    "Total Length of Bwd Packets", "Fwd Packet Length Max",
    "Fwd Packet Length Min", "Fwd Packet Length Mean",
    "Fwd Packet Length Std", "Bwd Packet Length Max",
    "Bwd Packet Length Min", "Bwd Packet Length Mean",
    "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max",
    "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std",
    "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags", "Bwd PSH Flags",
    "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Length",
    "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s",
    "Min Packet Length", "Max Packet Length", "Packet Length Mean",
    "Packet Length Std", "Packet Length Variance", "FIN Flag Count",
    "SYN Flag Count", "RST Flag Count", "PSH Flag Count",
    "ACK Flag Count", "URG Flag Count", "CWE Flag Count",
    "ECE Flag Count", "Down/Up Ratio", "Average Packet Size",
    "Avg Fwd Segment Size", "Avg Bwd Segment Size",
    "Fwd Header Length.1", "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate", "Subflow Fwd Packets",
    "Subflow Fwd Bytes", "Subflow Bwd Packets", "Subflow Bwd Bytes",
    "Init_Win_bytes_forward", "Init_Win_bytes_backward",
    "act_data_pkt_fwd", "min_seg_size_forward", "Active Mean",
    "Active Std", "Active Max", "Active Min", "Idle Mean", "Idle Std",
    "Idle Max", "Idle Min",
]

assert len(FEATURE_NAMES) == 77, f"Expected 77 features, got {len(FEATURE_NAMES)}"

np.random.seed(2024)

# ── Traffic Profiles ────────────────────────────────────────────────────
# Each profile defines realistic value ranges for key features.
# Unspecified features use small random noise (normal traffic baseline).

def base_row():
    """Generate a baseline normal-traffic row."""
    return {
        "Destination Port": np.random.choice([80, 443, 8080, 53, 22, 21]),
        "Flow Duration": np.random.uniform(1000, 120_000_000),
        "Total Fwd Packets": np.random.randint(1, 30),
        "Total Backward Packets": np.random.randint(1, 25),
        "Total Length of Fwd Packets": np.random.uniform(0, 5000),
        "Total Length of Bwd Packets": np.random.uniform(0, 8000),
        "Fwd Packet Length Max": np.random.uniform(0, 1500),
        "Fwd Packet Length Min": np.random.uniform(0, 100),
        "Fwd Packet Length Mean": np.random.uniform(0, 800),
        "Fwd Packet Length Std": np.random.uniform(0, 500),
        "Bwd Packet Length Max": np.random.uniform(0, 1500),
        "Bwd Packet Length Min": np.random.uniform(0, 100),
        "Bwd Packet Length Mean": np.random.uniform(0, 800),
        "Bwd Packet Length Std": np.random.uniform(0, 500),
        "Flow Bytes/s": np.random.uniform(0, 500_000),
        "Flow Packets/s": np.random.uniform(0, 5000),
        "Flow IAT Mean": np.random.uniform(0, 30_000_000),
        "Flow IAT Std": np.random.uniform(0, 20_000_000),
        "Flow IAT Max": np.random.uniform(0, 120_000_000),
        "Flow IAT Min": np.random.uniform(0, 1000),
        "Fwd IAT Total": np.random.uniform(0, 120_000_000),
        "Fwd IAT Mean": np.random.uniform(0, 30_000_000),
        "Fwd IAT Std": np.random.uniform(0, 20_000_000),
        "Fwd IAT Max": np.random.uniform(0, 120_000_000),
        "Fwd IAT Min": np.random.uniform(0, 10000),
        "Bwd IAT Total": np.random.uniform(0, 120_000_000),
        "Bwd IAT Mean": np.random.uniform(0, 30_000_000),
        "Bwd IAT Std": np.random.uniform(0, 20_000_000),
        "Bwd IAT Max": np.random.uniform(0, 120_000_000),
        "Bwd IAT Min": np.random.uniform(0, 10000),
        "Fwd PSH Flags": 0,
        "Bwd PSH Flags": 0,
        "Fwd URG Flags": 0,
        "Bwd URG Flags": 0,
        "Fwd Header Length": np.random.randint(20, 200),
        "Bwd Header Length": np.random.randint(20, 200),
        "Fwd Packets/s": np.random.uniform(0, 2000),
        "Bwd Packets/s": np.random.uniform(0, 2000),
        "Min Packet Length": np.random.uniform(0, 100),
        "Max Packet Length": np.random.uniform(100, 1500),
        "Packet Length Mean": np.random.uniform(50, 800),
        "Packet Length Std": np.random.uniform(0, 500),
        "Packet Length Variance": np.random.uniform(0, 250_000),
        "FIN Flag Count": np.random.choice([0, 1]),
        "SYN Flag Count": np.random.choice([0, 1]),
        "RST Flag Count": 0,
        "PSH Flag Count": np.random.choice([0, 1]),
        "ACK Flag Count": np.random.choice([0, 1]),
        "URG Flag Count": 0,
        "CWE Flag Count": 0,
        "ECE Flag Count": 0,
        "Down/Up Ratio": np.random.uniform(0, 5),
        "Average Packet Size": np.random.uniform(50, 1000),
        "Avg Fwd Segment Size": np.random.uniform(0, 800),
        "Avg Bwd Segment Size": np.random.uniform(0, 800),
        "Fwd Header Length.1": np.random.randint(20, 200),
        "Fwd Avg Bytes/Bulk": 0,
        "Fwd Avg Packets/Bulk": 0,
        "Fwd Avg Bulk Rate": 0,
        "Bwd Avg Bytes/Bulk": 0,
        "Bwd Avg Packets/Bulk": 0,
        "Bwd Avg Bulk Rate": 0,
        "Subflow Fwd Packets": np.random.randint(1, 30),
        "Subflow Fwd Bytes": np.random.uniform(0, 5000),
        "Subflow Bwd Packets": np.random.randint(1, 25),
        "Subflow Bwd Bytes": np.random.uniform(0, 8000),
        "Init_Win_bytes_forward": np.random.choice([8192, 16384, 29200, 65535]),
        "Init_Win_bytes_backward": np.random.choice([8192, 16384, 29200, 65535]),
        "act_data_pkt_fwd": np.random.randint(0, 10),
        "min_seg_size_forward": np.random.choice([20, 32, 44]),
        "Active Mean": np.random.uniform(0, 1_000_000),
        "Active Std": np.random.uniform(0, 500_000),
        "Active Max": np.random.uniform(0, 1_000_000),
        "Active Min": np.random.uniform(0, 500_000),
        "Idle Mean": np.random.uniform(0, 50_000_000),
        "Idle Std": np.random.uniform(0, 30_000_000),
        "Idle Max": np.random.uniform(0, 120_000_000),
        "Idle Min": np.random.uniform(0, 50_000_000),
    }


def make_benign():
    """Normal HTTPS / web-browsing traffic."""
    r = base_row()
    r["Destination Port"] = np.random.choice([80, 443, 8080])
    r["Flow Duration"] = np.random.uniform(50_000, 30_000_000)
    r["Total Fwd Packets"] = np.random.randint(2, 15)
    r["Total Backward Packets"] = np.random.randint(2, 12)
    r["Flow Bytes/s"] = np.random.uniform(500, 100_000)
    r["Flow Packets/s"] = np.random.uniform(1, 500)
    r["SYN Flag Count"] = 1
    r["ACK Flag Count"] = 1
    r["FIN Flag Count"] = np.random.choice([0, 1])
    r["RST Flag Count"] = 0
    r["Init_Win_bytes_forward"] = np.random.choice([29200, 65535])
    r["Init_Win_bytes_backward"] = np.random.choice([29200, 65535])
    return r, "BENIGN"


def make_ddos():
    """DDoS — massive packet flood, high bytes/s, very short IATs."""
    r = base_row()
    r["Destination Port"] = 80
    r["Flow Duration"] = np.random.uniform(1, 5000)
    r["Total Fwd Packets"] = np.random.randint(500, 50000)
    r["Total Backward Packets"] = np.random.randint(0, 5)
    r["Total Length of Fwd Packets"] = np.random.uniform(50000, 5_000_000)
    r["Total Length of Bwd Packets"] = np.random.uniform(0, 100)
    r["Fwd Packet Length Max"] = np.random.uniform(40, 100)
    r["Fwd Packet Length Mean"] = np.random.uniform(30, 80)
    r["Flow Bytes/s"] = np.random.uniform(5_000_000, 500_000_000)
    r["Flow Packets/s"] = np.random.uniform(50_000, 1_000_000)
    r["Flow IAT Mean"] = np.random.uniform(0, 10)
    r["Flow IAT Min"] = 0
    r["Fwd IAT Mean"] = np.random.uniform(0, 5)
    r["SYN Flag Count"] = np.random.choice([0, 1])
    r["ACK Flag Count"] = 1
    r["Down/Up Ratio"] = 0
    r["Init_Win_bytes_forward"] = np.random.choice([8192, 512, 1024])
    r["Init_Win_bytes_backward"] = 0
    return r, "DDoS"


def make_dos_hulk():
    """DoS Hulk — HTTP flood, many short-lived connections."""
    r = base_row()
    r["Destination Port"] = 80
    r["Flow Duration"] = np.random.uniform(100, 50000)
    r["Total Fwd Packets"] = np.random.randint(10, 500)
    r["Total Backward Packets"] = np.random.randint(0, 10)
    r["Total Length of Fwd Packets"] = np.random.uniform(5000, 500_000)
    r["Flow Bytes/s"] = np.random.uniform(500_000, 50_000_000)
    r["Flow Packets/s"] = np.random.uniform(5000, 200_000)
    r["Flow IAT Mean"] = np.random.uniform(0, 100)
    r["Fwd Packet Length Mean"] = np.random.uniform(200, 600)
    r["SYN Flag Count"] = 1
    r["PSH Flag Count"] = 1
    r["ACK Flag Count"] = 1
    return r, "DoS Hulk"


def make_portscan():
    """PortScan — many short flows to different ports, SYN flag."""
    r = base_row()
    r["Destination Port"] = np.random.randint(1, 65535)
    r["Flow Duration"] = np.random.uniform(0, 5000)
    r["Total Fwd Packets"] = np.random.randint(1, 3)
    r["Total Backward Packets"] = np.random.randint(0, 2)
    r["Total Length of Fwd Packets"] = np.random.uniform(0, 100)
    r["Total Length of Bwd Packets"] = np.random.uniform(0, 50)
    r["Fwd Packet Length Max"] = np.random.uniform(40, 60)
    r["Fwd Packet Length Mean"] = np.random.uniform(40, 60)
    r["Flow Bytes/s"] = np.random.uniform(10_000, 1_000_000)
    r["Flow Packets/s"] = np.random.uniform(1000, 100_000)
    r["SYN Flag Count"] = 1
    r["ACK Flag Count"] = 0
    r["RST Flag Count"] = np.random.choice([0, 1])
    r["FIN Flag Count"] = 0
    r["Init_Win_bytes_forward"] = np.random.choice([1024, 2048, 8192])
    r["Init_Win_bytes_backward"] = 0
    r["Down/Up Ratio"] = 0
    return r, "PortScan"


def make_bot():
    """Bot — periodic beaconing, moderate traffic, C&C communication."""
    r = base_row()
    r["Destination Port"] = np.random.choice([443, 8443, 6667, 9001])
    r["Flow Duration"] = np.random.uniform(1_000_000, 60_000_000)
    r["Total Fwd Packets"] = np.random.randint(5, 50)
    r["Total Backward Packets"] = np.random.randint(5, 50)
    r["Flow Bytes/s"] = np.random.uniform(100, 5000)
    r["Flow Packets/s"] = np.random.uniform(0.5, 50)
    r["Flow IAT Mean"] = np.random.uniform(500_000, 5_000_000)
    r["Flow IAT Std"] = np.random.uniform(10_000, 500_000)
    r["Fwd Packet Length Mean"] = np.random.uniform(50, 300)
    r["Bwd Packet Length Mean"] = np.random.uniform(50, 300)
    r["Init_Win_bytes_forward"] = np.random.choice([8192, 16384])
    return r, "Bot"


def make_ssh_patator():
    """SSH-Patator — brute-force SSH login attempts."""
    r = base_row()
    r["Destination Port"] = 22
    r["Flow Duration"] = np.random.uniform(10_000, 5_000_000)
    r["Total Fwd Packets"] = np.random.randint(5, 30)
    r["Total Backward Packets"] = np.random.randint(5, 25)
    r["Total Length of Fwd Packets"] = np.random.uniform(200, 5000)
    r["Total Length of Bwd Packets"] = np.random.uniform(200, 5000)
    r["Fwd Packet Length Mean"] = np.random.uniform(20, 100)
    r["Flow Bytes/s"] = np.random.uniform(1000, 50_000)
    r["Flow Packets/s"] = np.random.uniform(10, 500)
    r["SYN Flag Count"] = 1
    r["ACK Flag Count"] = 1
    r["PSH Flag Count"] = 1
    return r, "SSH-Patator"


def make_ftp_patator():
    """FTP-Patator — brute-force FTP login attempts."""
    r = base_row()
    r["Destination Port"] = 21
    r["Flow Duration"] = np.random.uniform(10_000, 2_000_000)
    r["Total Fwd Packets"] = np.random.randint(3, 20)
    r["Total Backward Packets"] = np.random.randint(3, 15)
    r["Total Length of Fwd Packets"] = np.random.uniform(100, 2000)
    r["Total Length of Bwd Packets"] = np.random.uniform(100, 2000)
    r["Fwd Packet Length Mean"] = np.random.uniform(10, 60)
    r["Flow Bytes/s"] = np.random.uniform(500, 20_000)
    r["Flow Packets/s"] = np.random.uniform(5, 200)
    r["SYN Flag Count"] = 1
    r["ACK Flag Count"] = 1
    r["PSH Flag Count"] = 1
    return r, "FTP-Patator"


def make_web_bruteforce():
    """Web Attack – Brute Force — HTTP login brute-forcing."""
    r = base_row()
    r["Destination Port"] = np.random.choice([80, 443, 8080])
    r["Flow Duration"] = np.random.uniform(5_000, 1_000_000)
    r["Total Fwd Packets"] = np.random.randint(5, 50)
    r["Total Backward Packets"] = np.random.randint(3, 30)
    r["Total Length of Fwd Packets"] = np.random.uniform(500, 20_000)
    r["Total Length of Bwd Packets"] = np.random.uniform(200, 10_000)
    r["Fwd Packet Length Mean"] = np.random.uniform(100, 500)
    r["Flow Bytes/s"] = np.random.uniform(5000, 500_000)
    r["Flow Packets/s"] = np.random.uniform(50, 5000)
    r["PSH Flag Count"] = 1
    r["ACK Flag Count"] = 1
    r["SYN Flag Count"] = 1
    return r, "Web Attack \u2013 Brute Force"


# ── Generate Rows ───────────────────────────────────────────────────────
generators = [
    (make_benign, 20),         # 20 normal traffic rows
    (make_ddos, 5),            # 5 DDoS
    (make_dos_hulk, 5),        # 5 DoS Hulk
    (make_portscan, 5),        # 5 PortScan
    (make_bot, 4),             # 4 Bot
    (make_ssh_patator, 4),     # 4 SSH brute-force
    (make_ftp_patator, 4),     # 4 FTP brute-force
    (make_web_bruteforce, 3),  # 3 Web brute-force
]

rows = []
labels = []
for gen_fn, count in generators:
    for _ in range(count):
        row_dict, label = gen_fn()
        row = [row_dict.get(f, np.random.uniform(0, 10)) for f in FEATURE_NAMES]
        rows.append(row)
        labels.append(label)

# Shuffle to mix normal and attack traffic
idx = np.random.permutation(len(rows))
rows = [rows[i] for i in idx]
labels = [labels[i] for i in idx]

# ── Build DataFrame ─────────────────────────────────────────────────────
# Add metadata columns (the app will drop these automatically)
source_ips = [f"192.168.{np.random.randint(1,10)}.{np.random.randint(1,254)}" for _ in range(len(rows))]
dest_ips = [f"10.0.0.{np.random.randint(1,254)}" for _ in range(len(rows))]
src_ports = [np.random.randint(1024, 65535) for _ in range(len(rows))]
timestamps = [f"2024-07-04 {np.random.randint(8,22):02d}:{np.random.randint(0,59):02d}:{np.random.randint(0,59):02d}" for _ in range(len(rows))]
flow_ids = [f"{src}-{d}-{sp}-{int(rows[i][0])}-6" for i, (src, d, sp) in enumerate(zip(source_ips, dest_ips, src_ports))]

df = pd.DataFrame(rows, columns=FEATURE_NAMES)

# Insert metadata columns at the front
df.insert(0, "Flow ID", flow_ids)
df.insert(1, "Source IP", source_ips)
df.insert(2, "Source Port", src_ports)
df.insert(3, "Destination IP", dest_ips)
df.insert(4, "Timestamp", timestamps)

# Add Label column at the end (the app drops this too)
df["Label"] = labels

# ── Save ────────────────────────────────────────────────────────────────
output_path = "test_traffic.csv"
df.to_csv(output_path, index=False)

# ── Summary ─────────────────────────────────────────────────────────────
print(f"\n✅ Generated: {output_path}")
print(f"   Total rows  : {len(df)}")
print(f"   Total cols  : {len(df.columns)} ({len(FEATURE_NAMES)} features + 6 metadata)")
print(f"\n📊 Label distribution:")
for lbl, cnt in pd.Series(labels).value_counts().items():
    print(f"   {lbl:30s} → {cnt} rows")
print(f"\n🛡️  Upload this file to CyberShield to test detection!")
