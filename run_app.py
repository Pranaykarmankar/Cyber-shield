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

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(root_dir, "app", "cybershield_app.py")
    
    if not os.path.exists(app_path):
        print(f"Error: Could not find application entrypoint at {app_path}")
        sys.exit(1)
        
    print("🛡️  Launching CyberShield Ember Shield Dashboard...")
    cmd = [sys.executable, "-m", "streamlit", "run", app_path]
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
