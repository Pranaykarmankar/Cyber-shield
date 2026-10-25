"""
CyberShield Dashboard Launcher
Usage:
    python run_app.py
    or
    streamlit run app/cybershield_app.py
"""

import os
import sys
import subprocess

# Ensure UTF-8 output handling on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(root_dir, "app", "cybershield_app.py")
    
    if not os.path.exists(app_path):
        print(f"Error: Could not find application entrypoint at {app_path}")
        sys.exit(1)
        
    print("[CyberShield] Launching CyberShield AI Intrusion Detection Dashboard...")
    print(f"  Target: {app_path}")
    print("  Starting Streamlit web server on CPU...\n")
    
    cmd = [sys.executable, "-m", "streamlit", "run", app_path]
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
