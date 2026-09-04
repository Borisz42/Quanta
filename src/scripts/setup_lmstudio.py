"""Helper script to verify, download, start, and load Qwen3.5-4B-MTP-GGUF in LM Studio.

Model Source:
https://huggingface.co/unsloth/Qwen3.5-4B-MTP-GGUF
"""

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

MODEL_HUGGINGFACE_ID = "unsloth/Qwen3.5-4B-MTP-GGUF"
MODEL_IDENTIFIER = "qwen3.5-4b-mtp"
DEFAULT_BASE_URL = "http://127.0.0.1:1234/v1"


def find_lms_executable() -> str:
    """Locate lms.exe on PATH or in standard user installation paths."""
    lms_path = shutil.which("lms")
    if lms_path:
        return lms_path

    user_home = os.path.expanduser("~")
    candidates = [
        os.path.join(user_home, ".lmstudio", "bin", "lms.exe"),
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "LM Studio", "resources", "app", "bin", "lms.exe"),
        r"C:\Users\PC\.lmstudio\bin\lms.exe",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c

    return "lms"


def is_server_running(base_url: str = DEFAULT_BASE_URL) -> bool:
    """Check if the LM Studio local server is responding."""
    try:
        url = f"{base_url.rstrip('/')}/models"
        req = urllib.request.Request(url, headers={"User-Agent": "QUANTA-Setup"})
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            return resp.status == 200
    except Exception:
        return False


def get_loaded_models(base_url: str = DEFAULT_BASE_URL) -> list:
    """Query currently loaded model IDs from the server."""
    try:
        url = f"{base_url.rstrip('/')}/models"
        req = urllib.request.Request(url, headers={"User-Agent": "QUANTA-Setup"})
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return [m.get("id") for m in data.get("data", [])]
    except Exception:
        return []


def run_cmd(cmd_list):
    """Run a CLI command and print output."""
    print(f"-> Executing: {' '.join(cmd_list)}")
    res = subprocess.run(cmd_list, capture_output=True, text=True)
    if res.stdout:
        print(res.stdout.strip())
    if res.stderr and res.returncode != 0:
        print(f"Error: {res.stderr.strip()}")
    return res.returncode == 0


def main():
    print("================================================================================")
    print("QUANTA LM STUDIO Qwen3.5-4B-MTP-GGUF SETUP & VERIFIER")
    print(f"Target Hugging Face Model: {MODEL_HUGGINGFACE_ID}")
    print(f"Target Model Identifier:   {MODEL_IDENTIFIER}")
    print(f"Target Server URL:         {DEFAULT_BASE_URL}")
    print("================================================================================\n")

    lms_bin = find_lms_executable()
    print(f"[1/4] Checking LM Studio CLI: {lms_bin}")

    # Check if server is running
    print(f"[2/4] Checking server status on {DEFAULT_BASE_URL}...")
    running = is_server_running()
    if running:
        print("  [OK] LM Studio server is online.")
    else:
        print("  ! LM Studio server is not responding. Starting via CLI...")
        run_cmd([lms_bin, "server", "start"])
        time.sleep(2.0)
        if is_server_running():
            print("  [OK] Server started successfully.")
        else:
            print("  ! Could not start server automatically. Please launch LM Studio and start the server.")

    # Check downloaded models
    print(f"[3/4] Verifying local model download ({MODEL_HUGGINGFACE_ID})...")
    res = subprocess.run([lms_bin, "ls"], capture_output=True, text=True)
    if MODEL_IDENTIFIER in res.stdout or "qwen3.5-4b" in res.stdout.lower():
        print(f"  [OK] Model {MODEL_HUGGINGFACE_ID} is already downloaded.")
    else:
        print(f"  ! Model not found locally. Downloading from Hugging Face via lms get...")
        run_cmd([lms_bin, "get", MODEL_HUGGINGFACE_ID, "-y"])

    # Check loaded model in memory
    print(f"[4/4] Checking loaded models in memory...")
    loaded = get_loaded_models()
    print(f"  Currently loaded models: {loaded}")
    if any(MODEL_IDENTIFIER in str(m).lower() for m in loaded):
        print(f"  [OK] {MODEL_IDENTIFIER} is currently LOADED and active in memory.")
    else:
        print(f"  ! Loading {MODEL_IDENTIFIER} into memory...")
        run_cmd([lms_bin, "load", MODEL_IDENTIFIER, "--identifier", MODEL_IDENTIFIER, "-y"])
        time.sleep(2.0)
        loaded = get_loaded_models()
        print(f"  Updated loaded models: {loaded}")

    print("\n================================================================================")
    print("Setup verified! You can now run:")
    print("  python src/scripts/generate_complex_translation_examples.py --backend lmstudio")
    print("================================================================================")


if __name__ == "__main__":
    main()
