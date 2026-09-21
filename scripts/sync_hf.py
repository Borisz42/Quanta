#!/usr/bin/env python3
"""QUANTA Hugging Face Synchronization CLI.

Manages bidirectional synchronization of all QUANTA offline databases, semantic registries,
grammars, validation corpora, training datasets, and raw benchmarks with the
Hugging Face model repository: Borisz42/QUANTA.

Usage:
    # Check differences between local artifacts and Hugging Face:
    python scripts/sync_hf.py --check

    # Upload all new and modified artifacts to Hugging Face:
    python scripts/sync_hf.py --upload

    # Download any missing or updated artifacts from Hugging Face:
    python scripts/sync_hf.py --download

    # Simulate without performing network writes:
    python scripts/sync_hf.py --upload --dry-run
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import Enum
import hashlib
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Ensure Windows stdout handles UTF-8 cleanly
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

try:
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

from core.artifacts import (
    ARTIFACT_REGISTRY,
    ArtifactSpec,
    ArtifactCriticality,
    normalize_rel_path,
    resolve_artifact_path,
)

DEFAULT_REPO_ID = "Borisz42/QUANTA"
DEFAULT_REPO_TYPE = "model"


class SyncStatus(Enum):
    IDENTICAL = "IDENTICAL"    # Local and remote match
    MODIFIED = "MODIFIED"      # File exists on both, but size or hash differ
    LOCAL_ONLY = "LOCAL_ONLY"  # Exists locally, missing on remote
    REMOTE_ONLY = "REMOTE_ONLY"# Exists on remote, missing locally
    MISSING_ALL = "MISSING"    # Missing on both sides


@dataclass
class FileDiff:
    rel_path: str
    category: str
    status: SyncStatus
    local_size: Optional[int]
    remote_size: Optional[int]
    local_sha256: Optional[str] = None
    remote_sha256: Optional[str] = None
    local_path: Optional[Path] = None


def compute_sha256(file_path: Path, chunk_size: int = 1024 * 1024 * 8) -> str:
    """Computes SHA-256 hash in 8MB chunks to efficiently process multi-hundred MB databases."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def format_size(size_bytes: Optional[int]) -> str:
    if size_bytes is None:
        return "-"
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


class QuantaHFSynchronizer:
    def __init__(
        self,
        repo_id: str = DEFAULT_REPO_ID,
        repo_type: str = DEFAULT_REPO_TYPE,
        token: Optional[str] = None,
    ):
        if not HF_HUB_AVAILABLE:
            raise RuntimeError("huggingface_hub is not installed. Install via: pip install huggingface_hub")

        self.repo_id = repo_id
        self.repo_type = repo_type
        self.token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        self.api = HfApi(token=self.token)

    def fetch_remote_tree(self) -> Dict[str, Dict[str, Any]]:
        """Queries Hugging Face for all files currently in the repository."""
        remote_files: Dict[str, Dict[str, Any]] = {}
        print(f"[*] Querying remote repository {self.repo_id} ({self.repo_type})...", flush=True)
        try:
            items = list(self.api.list_repo_tree(self.repo_id, repo_type=self.repo_type, recursive=True))
            for it in items:
                if type(it).__name__ == "RepoFile":
                    lfs_info = getattr(it, "lfs", None)
                    remote_sha = None
                    if lfs_info and isinstance(lfs_info, dict):
                        remote_sha = lfs_info.get("sha256")
                    elif hasattr(lfs_info, "sha256"):
                        remote_sha = getattr(lfs_info, "sha256")

                    remote_files[normalize_rel_path(it.path)] = {
                        "path": it.path,
                        "size": getattr(it, "size", 0),
                        "sha256": remote_sha,
                        "blob_id": getattr(it, "blob_id", None),
                    }
            print(f"    Found {len(remote_files)} remote files on {self.repo_id}.", flush=True)
        except Exception as e:
            print(f"[!] Warning: Could not fetch remote tree from {self.repo_id}: {e}", file=sys.stderr)
        return remote_files

    def inspect_diffs(self, target_file: Optional[str] = None) -> List[FileDiff]:
        """Compares local repository artifacts against remote files."""
        remote_tree = self.fetch_remote_tree()
        diffs: List[FileDiff] = []

        # Collect tracked local specs
        specs_to_check = list(ARTIFACT_REGISTRY)

        # Filter if targeted file requested
        if target_file:
            norm_target = normalize_rel_path(target_file)
            specs_to_check = [s for s in specs_to_check if normalize_rel_path(s.rel_path) == norm_target]

        print(f"[*] Comparing {len(specs_to_check)} tracked artifacts against remote tree...", flush=True)
        checked_paths = set()

        for spec in specs_to_check:
            norm_path = normalize_rel_path(spec.rel_path)
            checked_paths.add(norm_path)
            local_resolved = resolve_artifact_path(norm_path)
            local_exists = local_resolved is not None and local_resolved.is_file()
            local_size = local_resolved.stat().st_size if local_exists else None

            remote_meta = remote_tree.get(norm_path)
            remote_exists = remote_meta is not None
            remote_size = remote_meta["size"] if remote_exists else None

            if local_exists and remote_exists:
                # Compare sizes first for performance
                if local_size == remote_size:
                    # Sizes match; check hash if remote has LFS sha256
                    rem_sha = remote_meta.get("sha256")
                    if rem_sha:
                        loc_sha = compute_sha256(local_resolved)
                        if loc_sha.lower() == rem_sha.lower():
                            status = SyncStatus.IDENTICAL
                        else:
                            status = SyncStatus.MODIFIED
                    else:
                        # Non-LFS exact size match
                        status = SyncStatus.IDENTICAL
                else:
                    status = SyncStatus.MODIFIED
            elif local_exists and not remote_exists:
                status = SyncStatus.LOCAL_ONLY
            elif not local_exists and remote_exists:
                status = SyncStatus.REMOTE_ONLY
            else:
                status = SyncStatus.MISSING_ALL

            diffs.append(
                FileDiff(
                    rel_path=norm_path,
                    category=spec.category,
                    status=status,
                    local_size=local_size,
                    remote_size=remote_size,
                    local_path=local_resolved if local_exists else None,
                )
            )

        # Also check if there are remote files not in the local registry
        for rem_path, rem_meta in remote_tree.items():
            if rem_path not in checked_paths and not rem_path.startswith(".git"):
                local_cand = REPO_ROOT / rem_path
                local_exists = local_cand.exists()
                local_size = local_cand.stat().st_size if local_exists else None

                if not local_exists:
                    diffs.append(
                        FileDiff(
                            rel_path=rem_path,
                            category="Remote Asset",
                            status=SyncStatus.REMOTE_ONLY,
                            local_size=None,
                            remote_size=rem_meta.get("size"),
                        )
                    )

        return diffs

    def print_diff_table(self, diffs: List[FileDiff]):
        print("\n" + "=" * 105)
        print(f" {'Hugging Face Synchronization Status: ' + self.repo_id:^103} ")
        print("=" * 105)
        print(f" {'Status':<13} | {'Artifact Path':<50} | {'Local Size':<12} | {'Remote Size':<12}")
        print("-" * 105)

        category_order = [
            "Slot Registries",
            "Offline Databases",
            "Grammars",
            "Raw Benchmarks",
            "Validation Corpus",
            "Training Datasets",
            "Profiler Artifacts",
            "Translation Examples",
            "Raw Source Data",
        ]

        # Group by category
        by_cat: Dict[str, List[FileDiff]] = {}
        for d in diffs:
            by_cat.setdefault(d.category, []).append(d)

        for cat in category_order + [k for k in by_cat if k not in category_order]:
            items = by_cat.get(cat, [])
            if not items:
                continue
            print(f"--- {cat} " + "-" * (99 - len(cat)))
            for d in items:
                status_color = d.status.value
                print(
                    f" {status_color:<13} | {d.rel_path:<50} | {format_size(d.local_size):<12} | {format_size(d.remote_size):<12}"
                )

        print("=" * 105)
        total_local = sum(d.local_size or 0 for d in diffs if d.local_size)
        total_remote = sum(d.remote_size or 0 for d in diffs if d.remote_size)

        identical_count = sum(1 for d in diffs if d.status == SyncStatus.IDENTICAL)
        modified_count = sum(1 for d in diffs if d.status == SyncStatus.MODIFIED)
        local_only_count = sum(1 for d in diffs if d.status == SyncStatus.LOCAL_ONLY)
        remote_only_count = sum(1 for d in diffs if d.status == SyncStatus.REMOTE_ONLY)

        print(f" Summary: {identical_count} identical, {modified_count} modified, {local_only_count} local-only, {remote_only_count} remote-only")
        print(f" Local total:  {format_size(total_local)}")
        print(f" Remote total: {format_size(total_remote)}")
        print("=" * 105 + "\n")

    def upload_diffs(self, diffs: List[FileDiff], dry_run: bool = False) -> int:
        """Uploads files that are LOCAL_ONLY or MODIFIED to Hugging Face."""
        upload_candidates = [
            d for d in diffs if d.status in (SyncStatus.LOCAL_ONLY, SyncStatus.MODIFIED) and d.local_path
        ]

        if not upload_candidates:
            print("[*] No files require upload. All remote files are identical.")
            return 0

        print(f"\n[*] Preparing to upload {len(upload_candidates)} artifacts to {self.repo_id}...")
        total_bytes = sum(d.local_size or 0 for d in upload_candidates)
        print(f"    Total upload volume: {format_size(total_bytes)}")

        if dry_run:
            print("\n[DRY-RUN] Files that would be uploaded:")
            for d in upload_candidates:
                print(f"  - [{d.status.value}] {d.rel_path} ({format_size(d.local_size)})")
            return 0

        uploaded_count = 0
        t0 = time.perf_counter()

        for idx, d in enumerate(upload_candidates, 1):
            rel_p = d.rel_path
            loc_p = d.local_path
            size_s = format_size(d.local_size)

            print(f"[{idx}/{len(upload_candidates)}] Uploading {rel_p} ({size_s})...", flush=True)
            try:
                commit_msg = f"QUANTA: upload {rel_p} ({size_s})"
                self.api.upload_file(
                    path_or_fileobj=str(loc_p),
                    path_in_repo=rel_p,
                    repo_id=self.repo_id,
                    repo_type=self.repo_type,
                    commit_message=commit_msg,
                )
                print(f"  [OK] Successfully uploaded {rel_p}", flush=True)
                uploaded_count += 1
            except Exception as e:
                print(f"  [!] Failed to upload {rel_p}: {e}", file=sys.stderr, flush=True)

        elapsed = time.perf_counter() - t0
        print(f"\n[*] Upload completed: {uploaded_count}/{len(upload_candidates)} files in {elapsed:.1f}s.")
        return 0 if uploaded_count == len(upload_candidates) else 1

    def download_diffs(self, diffs: List[FileDiff], dry_run: bool = False) -> int:
        """Downloads files that are REMOTE_ONLY or MODIFIED from Hugging Face."""
        download_candidates = [
            d for d in diffs if d.status in (SyncStatus.REMOTE_ONLY, SyncStatus.MODIFIED)
        ]

        if not download_candidates:
            print("[*] No files require download. All local files are up-to-date.")
            return 0

        print(f"\n[*] Preparing to download {len(download_candidates)} artifacts from {self.repo_id}...")

        if dry_run:
            print("\n[DRY-RUN] Files that would be downloaded:")
            for d in download_candidates:
                print(f"  - [{d.status.value}] {d.rel_path} ({format_size(d.remote_size)})")
            return 0

        downloaded_count = 0
        t0 = time.perf_counter()

        for idx, d in enumerate(download_candidates, 1):
            rel_p = d.rel_path
            target_local = REPO_ROOT / rel_p
            target_local.parent.mkdir(parents=True, exist_ok=True)

            print(f"[{idx}/{len(download_candidates)}] Downloading {rel_p} ({format_size(d.remote_size)})...", flush=True)
            try:
                hf_hub_download(
                    repo_id=self.repo_id,
                    repo_type=self.repo_type,
                    filename=rel_p,
                    local_dir=str(REPO_ROOT),
                    token=self.token,
                )
                print(f"  [OK] Successfully downloaded {rel_p}", flush=True)
                downloaded_count += 1
            except Exception as e:
                print(f"  [!] Failed to download {rel_p}: {e}", file=sys.stderr, flush=True)

        elapsed = time.perf_counter() - t0
        print(f"\n[*] Download completed: {downloaded_count}/{len(download_candidates)} files in {elapsed:.1f}s.")
        return 0 if downloaded_count == len(download_candidates) else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Synchronize QUANTA runtime artifacts and offline databases with Hugging Face."
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=DEFAULT_REPO_ID,
        help=f"Hugging Face repository ID (default: {DEFAULT_REPO_ID})",
    )
    parser.add_argument(
        "--repo-type",
        type=str,
        default=DEFAULT_REPO_TYPE,
        choices=["model", "dataset", "space"],
        help=f"Hugging Face repository type (default: {DEFAULT_REPO_TYPE})",
    )
    parser.add_argument(
        "--check", "--status",
        action="store_true",
        help="Check differences between local files and remote repository (default mode)",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload new and modified local artifacts to Hugging Face",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download missing or updated artifacts from Hugging Face",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate synchronization without performing uploads or downloads",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Target a single specific artifact path (e.g. data/conceptnet_offline.db)",
    )

    args = parser.parse_args()

    # Default to check if neither upload nor download is explicitly specified
    if not args.upload and not args.download:
        args.check = True

    syncer = QuantaHFSynchronizer(repo_id=args.repo_id, repo_type=args.repo_type)
    diffs = syncer.inspect_diffs(target_file=args.file)
    syncer.print_diff_table(diffs)

    exit_code = 0
    if args.upload:
        exit_code = syncer.upload_diffs(diffs, dry_run=args.dry_run)
    elif args.download:
        exit_code = syncer.download_diffs(diffs, dry_run=args.dry_run)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
