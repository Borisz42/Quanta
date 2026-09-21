#!/usr/bin/env python3
"""QUANTA Environment & Artifact Initialization Script.

Performs first-time setup for QUANTA:
1. Validates local environment and Hugging Face API credentials.
2. Ensures all project directory structures exist.
3. Audits all required offline databases and runtime artifacts.
4. Automatically downloads any missing assets from Hugging Face: Borisz42/QUANTA.
5. Runs verification checks to confirm the neuro-symbolic engine is ready for execution.

Usage:
    python scripts/init_quanta.py
    python scripts/init_quanta.py --skip-tests
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

# Ensure Windows stdout handles UTF-8 cleanly
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
SCRIPTS_DIR = REPO_ROOT / "scripts"

for p in (str(SRC_DIR), str(REPO_ROOT), str(SCRIPTS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

from core.artifacts import (
    ARTIFACT_REGISTRY,
    ArtifactCriticality,
    audit_artifacts,
    resolve_artifact_path,
)

try:
    from sync_hf import QuantaHFSynchronizer
    SYNC_AVAILABLE = True
except ImportError:
    SYNC_AVAILABLE = False


def ensure_directories():
    """Ensure standard runtime directories exist."""
    required_dirs = [
        REPO_ROOT / "data" / "raw",
        REPO_ROOT / "data" / "grammar",
        REPO_ROOT / "data" / "validation_corpus",
        REPO_ROOT / "data" / "virtual_page_table",
        REPO_ROOT / "data" / "training",
        REPO_ROOT / "output",
        REPO_ROOT / "models",
    ]
    for d in required_dirs:
        d.mkdir(parents=True, exist_ok=True)


def check_credentials():
    print("[1/4] Checking environment credentials...", flush=True)
    env_file = REPO_ROOT / ".env"
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    if env_file.exists():
        print(f"  [OK] Local .env detected at {env_file}")
    else:
        print("  [!] .env file not found. Copy .env.example to .env and configure HF_TOKEN.")

    if hf_token:
        masked = hf_token[:6] + "..." + hf_token[-4:] if len(hf_token) > 10 else "***"
        print(f"  [OK] Active Hugging Face Token: {masked}")
    else:
        print("  [!] Warning: No HF_TOKEN detected. Remote Hugging Face download may be rate-limited.")


def download_missing_from_hf() -> bool:
    print("[2/4] Auditing runtime artifacts and checking Hugging Face (Borisz42/QUANTA)...", flush=True)
    if not SYNC_AVAILABLE:
        print("  [!] sync_hf module unavailable. Skipping remote download.", file=sys.stderr)
        return False

    try:
        syncer = QuantaHFSynchronizer()
        diffs = syncer.inspect_diffs()

        missing_or_mod = [d for d in diffs if d.status.value in ("REMOTE_ONLY", "MODIFIED")]
        if not missing_or_mod:
            print("  [OK] All local runtime artifacts are present and synchronized.")
            return True

        print(f"  [*] Downloading {len(missing_or_mod)} missing/updated artifacts from Hugging Face...")
        res = syncer.download_diffs(diffs)
        return res == 0
    except Exception as e:
        print(f"  [!] Hugging Face check/download encountered an error: {e}", file=sys.stderr)
        return False


def verify_critical_artifacts() -> bool:
    print("[3/4] Verifying local critical runtime artifacts...", flush=True)
    audit = audit_artifacts(critical_only=True)

    if audit["all_critical_present"]:
        print(f"  [OK] All {audit['total_artifacts']} critical runtime artifacts are present locally.")
        return True
    else:
        print("  [!] Missing critical artifacts:", file=sys.stderr)
        for missing in audit["missing_critical"]:
            print(f"      - {missing}", file=sys.stderr)
        return False


def run_sanity_tests() -> bool:
    print("[4/4] Running pre-flight sanity test suite...", flush=True)
    cmd = [sys.executable, "-m", "pytest", "tests/test_slots.py", "tests/test_artifact_guards.py", "-q"]
    res = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if res.returncode == 0:
        print("  [OK] Pre-flight sanity tests passed successfully.")
        return True
    else:
        print("  [!] Pre-flight sanity tests failed.", file=sys.stderr)
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Initialize QUANTA environment and download runtime artifacts.")
    parser.add_argument("--skip-tests", action="store_true", help="Skip running pre-flight pytest checks")
    args = parser.parse_args()

    print("=" * 80)
    print("QUANTA Neuro-Symbolic Architecture Initialization (Borisz42/QUANTA)")
    print("=" * 80)

    t0 = time.perf_counter()
    ensure_directories()
    check_credentials()
    download_missing_from_hf()

    is_ok = verify_critical_artifacts()
    if not is_ok:
        print("\n[!] QUANTA initialization incomplete: some critical artifacts are still missing.", file=sys.stderr)
        print("    Run 'python scripts/gather_artifacts.py' or 'python scripts/sync_hf.py --download'.\n", file=sys.stderr)
        return 1

    if not args.skip_tests:
        test_ok = run_sanity_tests()
        if not test_ok:
            return 1

    elapsed = time.perf_counter() - t0
    print("\n" + "=" * 80)
    print(f"QUANTA initialization complete! (took {elapsed:.2f}s)")
    print("All neuro-symbolic vector engines, solvers, and transducers are ready.")
    print("=" * 80 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
