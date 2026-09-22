"""Unsloth GPU Server Lifecycle Manager & Strict Hardware Guard for QUANTA.

Provides:
1. Automated Server Detection & Background Wakeup:
   - Polls http://127.0.0.1:8888/v1/models.
   - Automatically spawns Unsloth Studio (`unsloth studio --api-only -H 127.0.0.1 -p 8888`)
     or standalone `llama-server.exe` with GPU acceleration if offline.
2. Automated Model Loading:
   - Queries loaded status and sends POST /api/inference/load if dormant.
3. Strict CPU Offload Guard:
   - Strictly prohibits CPU execution if GPU is unavailable or VRAM offload fails,
     raising RuntimeError unless QUANTA_ALLOW_CPU_OFFLOAD=1 is set.
4. GPU Telemetry:
   - Queries nvidia-smi for VRAM utilization, total VRAM, and GPU compute load.
5. Direct Chat Completions:
   - Performs inference calls with latency and throughput instrumentation.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger("quanta.server.unsloth_manager")

DEFAULT_BACKEND_HOST = "127.0.0.1"
DEFAULT_BACKEND_PORT = 8888
DEFAULT_MODEL_ID = "unsloth/Qwen3.5-4B-MTP-GGUF"


class ChatResponse(dict):
    """Chat completion result supporting both dictionary and 4-tuple unpacking."""
    def __iter__(self):
        yield self["content"]
        yield self["latency_s"]
        yield self["completion_tokens"]
        yield self["tokens_per_sec"]

    @property
    def content(self) -> str:
        return self["content"]


class UnslothServerManager:
    """Manages Unsloth Studio / llama-server lifecycle, GPU telemetry, and hardware policy."""

    def __init__(
        self,
        host: str = DEFAULT_BACKEND_HOST,
        port: int = DEFAULT_BACKEND_PORT,
        target_model: str = DEFAULT_MODEL_ID,
        allow_cpu_offload: Optional[bool] = None,
    ):
        self.host = host
        self.port = port
        self.target_model = target_model
        self.base_url = f"http://{self.host}:{self.port}"
        self.api_url = f"{self.base_url}/v1"
        self._allow_cpu_offload = allow_cpu_offload
        self._server_process: Optional[subprocess.Popen] = None

    @property
    def is_cpu_offload_allowed(self) -> bool:
        """Determines whether CPU execution is permitted via config or environment."""
        if self._allow_cpu_offload is not None:
            return bool(self._allow_cpu_offload)
        env_val = os.environ.get("QUANTA_ALLOW_CPU_OFFLOAD", "").strip().lower()
        return env_val in ("1", "true", "yes")

    def enforce_gpu_policy(self, allow_cpu_flag: Optional[bool] = None) -> bool:
        """Validates that GPU acceleration is active and enforces strict policy.

        Raises:
            RuntimeError: If GPU is unavailable or simulated as missing, and
                          CPU offload is not explicitly allowed.
        Returns:
            True if GPU is available and verified, False if running on CPU with explicit permission.
        """
        allow_cpu = allow_cpu_flag if allow_cpu_flag is not None else self.is_cpu_offload_allowed

        # 1. Check for simulated missing GPU
        if os.environ.get("QUANTA_SIMULATE_NO_GPU", "").strip().lower() in ("1", "true", "yes"):
            if not allow_cpu:
                raise RuntimeError(
                    "Strict GPU Execution Policy Enforced: GPU is simulated as unavailable, "
                    "and CPU offload is strictly prohibited. Set QUANTA_ALLOW_CPU_OFFLOAD=1 to override."
                )
            logger.warning("GPU simulated unavailable; CPU offload permitted via QUANTA_ALLOW_CPU_OFFLOAD=1.")
            return False

        # 2. Check for physical NVIDIA GPU via nvidia-smi
        telemetry = self.get_gpu_telemetry()
        if not telemetry or telemetry.get("status") != "active":
            if not allow_cpu:
                raise RuntimeError(
                    "Strict GPU Execution Policy Enforced: No NVIDIA GPU detected or nvidia-smi unreachable, "
                    "and CPU offload is strictly prohibited. Set QUANTA_ALLOW_CPU_OFFLOAD=1 to override."
                )
            logger.warning("No active NVIDIA GPU detected; proceeding on CPU per explicit override.")
            return False

        return True

    def get_gpu_telemetry(self) -> Dict[str, Any]:
        """Queries nvidia-smi for real-time VRAM and GPU utilization metrics."""
        # Simulated no GPU check
        if os.environ.get("QUANTA_SIMULATE_NO_GPU", "").strip().lower() in ("1", "true", "yes"):
            return {
                "status": "unavailable",
                "error": "QUANTA_SIMULATE_NO_GPU flag is active",
            }

        try:
            cmd = [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,memory.free,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=5.0)
            line = res.stdout.strip().splitlines()[0]
            parts = [p.strip() for p in line.split(",")]
            return {
                "status": "active",
                "name": parts[0],
                "used_mb": float(parts[1]),
                "total_mb": float(parts[2]),
                "free_mb": float(parts[3]),
                "util_pct": float(parts[4]),
                "temp_c": float(parts[5]) if len(parts) > 5 else 0.0,
            }
        except Exception as exc:
            logger.debug("nvidia-smi query failed: %s", exc)
            return {
                "status": "unavailable",
                "error": str(exc),
            }

    def is_service_responsive(self, timeout: float = 3.0) -> bool:
        """Checks if the server responds on /v1/models."""
        try:
            with httpx.Client(timeout=timeout) as client:
                r = client.get(f"{self.api_url}/models")
                return r.status_code == 200
        except Exception:
            return False

    def ensure_unsloth_service_running(self, timeout: float = 30.0) -> bool:
        """Ensures that the Unsloth server is responsive, launching it in background if down."""
        if self.is_service_responsive(timeout=3.0):
            logger.info("Unsloth service is already running on %s", self.base_url)
            return True

        logger.info("Unsloth service not responsive on %s. Initiating auto-wakeup...", self.base_url)

        # Enforce GPU policy prior to starting server
        self.enforce_gpu_policy()

        # Locate startup candidates
        launched = False

        # Candidate 1: unsloth studio CLI via python
        py_exe = sys.executable
        try:
            cmd = [
                py_exe,
                "-m",
                "unsloth_cli",
                "studio",
                "--api-only",
                "-H",
                str(self.host),
                "-p",
                str(self.port),
            ]
            # Use detached process creation on Windows
            creationflags = 0
            if sys.platform == "win32":
                creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS

            self._server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creationflags,
            )
            launched = True
            logger.info("Spawned Unsloth Studio process (PID: %s)", self._server_process.pid)
        except Exception as e:
            logger.warning("Failed to spawn unsloth_cli studio: %s", e)

        # Candidate 2: direct llama-server.exe if candidate 1 fails
        if not launched:
            llama_candidates = [
                Path(r"C:\Users\PC\.unsloth\llama.cpp\build\bin\Release\llama-server.exe"),
                Path(shutil.which("llama-server") or ""),
            ]
            for candidate in llama_candidates:
                if candidate and candidate.exists():
                    try:
                        cmd = [
                            str(candidate),
                            "--port",
                            str(self.port),
                            "--host",
                            str(self.host),
                            "-ngl",
                            "-1",  # Offload all layers to GPU
                        ]
                        creationflags = 0
                        if sys.platform == "win32":
                            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS

                        self._server_process = subprocess.Popen(
                            cmd,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            creationflags=creationflags,
                        )
                        launched = True
                        logger.info("Spawned llama-server fallback (PID: %s)", self._server_process.pid)
                        break
                    except Exception as e:
                        logger.warning("Failed to spawn llama-server: %s", e)

        if not launched:
            logger.error("Could not find suitable server executable to launch Unsloth service.")
            return False

        # Poll until service responds
        t0 = time.time()
        while time.time() - t0 < timeout:
            time.sleep(1.0)
            if self.is_service_responsive(timeout=2.0):
                logger.info("Unsloth service successfully awakened and verified on %s", self.base_url)
                return True

        logger.error("Unsloth service failed to respond within %.1f seconds.", timeout)
        return False

    def ensure_model_loaded(
        self,
        model_id: Optional[str] = None,
        gpu_layers: int = -1,
        timeout: float = 30.0,
        gguf_variant: str = "Q5_K_M",
        speculative_type: str = "mtp",
    ) -> bool:
        """Verifies that the target model is loaded in GPU VRAM, requesting load if dormant."""
        target = model_id or self.target_model

        # Ensure service is running
        if not self.ensure_unsloth_service_running(timeout=timeout):
            return False

        # Enforce GPU policy
        self.enforce_gpu_policy()

        # Check model loaded state
        try:
            with httpx.Client(timeout=5.0) as client:
                r = client.get(f"{self.api_url}/models")
                if r.status_code == 200:
                    models = r.json().get("data", [])
                    entry = next((m for m in models if m.get("id") == target), None)
                    if entry and entry.get("loaded"):
                        logger.info("Model '%s' is verified loaded in VRAM.", target)
                        return True

                # Not loaded or not present: trigger load request
                logger.info("Model '%s' is dormant. Sending POST /api/inference/load...", target)
                load_payload = {
                    "model_path": target,
                    "gpu_memory_mode": "auto",
                    "gpu_layers": gpu_layers,
                    "gguf_variant": gguf_variant,
                    "speculative_type": speculative_type,
                }
                load_resp = client.post(f"{self.base_url}/api/inference/load", json=load_payload, timeout=30.0)
                if load_resp.status_code in (200, 202):
                    t0 = time.time()
                    while time.time() - t0 < timeout:
                        time.sleep(1.0)
                        chk = client.get(f"{self.api_url}/models", timeout=5.0)
                        if chk.status_code == 200:
                            models = chk.json().get("data", [])
                            m = next((item for item in models if item.get("id") == target), None)
                            if m and m.get("loaded"):
                                logger.info("Model '%s' successfully loaded into VRAM.", target)
                                return True
        except Exception as e:
            logger.warning("Error verifying/loading model '%s': %s", target, e)

        return False

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: int = 100,
        temperature: float = 0.1,
        timeout: float = 45.0,
    ) -> ChatResponse:
        """Executes a chat completion call to the Unsloth GPU server.

        Returns:
            ChatResponse containing:
            - content: Generated text
            - latency_s: Total round-trip latency in seconds
            - prompt_tokens: Prompt token count
            - completion_tokens: Completion token count
            - tokens_per_sec: Generation throughput
            - gpu_telemetry: Real-time GPU telemetry snapshot
        """
        target = model or self.target_model
        payload = {
            "model": target,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        t0 = time.perf_counter()
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(f"{self.api_url}/chat/completions", json=payload)
        dt_s = time.perf_counter() - t0

        if resp.status_code != 200:
            raise RuntimeError(f"Unsloth server returned status {resp.status_code}: {resp.text}")

        data = resp.json()
        content = data["choices"][0]["message"]["content"].strip()
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", len(content.split()))
        tps = completion_tokens / dt_s if dt_s > 0 else 0.0

        telemetry = self.get_gpu_telemetry()

        return ChatResponse({
            "content": content,
            "latency_s": dt_s,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "tokens_per_sec": tps,
            "gpu_telemetry": telemetry,
            "raw_response": data,
        })
