"""
CyberShield Diffusion GUI Launcher (Gradio)
Usage:
    python run_diffusion_gui.py
"""
import os
import sys

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(root_dir, "gui", "diffusion_gradio_app.py")
    if not os.path.exists(app_path):
        print(f"Error: {app_path} not found.")
        sys.exit(1)
    
    print("[CyberShield] Starting Gradio Text-to-Heatmap Diffusion Studio...")
    import subprocess
    cmd = [sys.executable, app_path]
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
