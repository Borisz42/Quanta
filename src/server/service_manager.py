"""QUANTA Service Manager & Unified Orchestrator.

Provides automated detection, auto-spinup, health auditing, and testing for:
1. Base Downstream LLM (Unsloth Studio / llama-server on :8888)
2. QUANTA Neuro-Symbolic Reverse Proxy (FastAPI/uvicorn on :8000)
3. QUANTA Model Context Protocol (MCP) Server (stdio / programmatic)

Usage via Python:
    from server.service_manager import QuantaServiceManager
    mgr = QuantaServiceManager()
    mgr.ensure_live_backend()

Usage via CLI:
    python -m server.service_manager status
    python -m server.service_manager start --all
    python -m server.service_manager test-mcp
    python -m server.service_manager test-proxy
"""

from __future__ import annotations

import sys

# Ensure stdout and stderr support UTF-8 on Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import argparse
import atexit
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import httpx

from server.unsloth_manager import UnslothServerManager

logger = logging.getLogger("quanta.server.service_manager")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def format_ansi(text: str, color_code: str) -> str:
    """Formats text with standard ANSI escape color sequences."""
    return f"\033[{color_code}m{text}\033[0m"


class QuantaServiceManager:
    """Orchestrates QUANTA service discovery, automated spinup, and diagnostic auditing."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8888/v1",
        quanta_url: str = "http://127.0.0.1:8000/v1",
        target_model: str = "unsloth/Qwen3.5-4B-MTP-GGUF",
        allow_cpu_offload: Optional[bool] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.quanta_url = quanta_url.rstrip("/")
        self.target_model = target_model
        self.allow_cpu_offload = allow_cpu_offload
        self.unsloth_mgr = UnslothServerManager(
            host=self._extract_host(self.base_url),
            port=self._extract_port(self.base_url, default=8888),
            target_model=self.target_model,
            allow_cpu_offload=self.allow_cpu_offload,
        )
        self._spawned_processes: List[subprocess.Popen] = []

    @staticmethod
    def _extract_host(url: str) -> str:
        parsed = urlparse(url if "://" in url else f"http://{url}")
        return parsed.hostname or "127.0.0.1"

    @staticmethod
    def _extract_port(url: str, default: int = 8000) -> int:
        parsed = urlparse(url if "://" in url else f"http://{url}")
        return parsed.port or default

    # -------------------------------------------------------------------------
    # Health & Readiness Checks
    # -------------------------------------------------------------------------

    def is_base_llm_running(self, timeout: float = 10.0) -> bool:
        """Checks if the Base LLM server is responsive."""
        try:
            with httpx.Client(timeout=timeout) as client:
                r = client.get(f"{self.base_url}/models")
                return r.status_code == 200
        except Exception:
            return False

    def is_model_loaded(self, model_id: Optional[str] = None) -> bool:
        """Checks if target model is loaded in VRAM on the Base LLM server."""
        return self.unsloth_mgr.is_model_loaded(model_id or self.target_model)

    def is_proxy_running(self, timeout: float = 3.0) -> bool:
        """Checks if the QUANTA Reverse Proxy is responsive on /health or /v1/health."""
        proxy_root = self.quanta_url.replace("/v1", "")
        for path in ("/health", "/v1/health"):
            try:
                with httpx.Client(timeout=timeout) as client:
                    r = client.get(f"{proxy_root}{path}")
                    if r.status_code == 200:
                        return True
            except Exception:
                pass
        return False

    def get_service_status(self) -> Dict[str, Any]:
        """Gathers a comprehensive status report across all components."""
        base_running = self.is_base_llm_running()
        model_loaded = self.is_model_loaded() if base_running else False
        proxy_running = self.is_proxy_running()
        gpu_telemetry = self.unsloth_mgr.get_gpu_telemetry()

        proxy_details: Dict[str, Any] = {}
        if proxy_running:
            try:
                proxy_root = self.quanta_url.replace("/v1", "")
                with httpx.Client(timeout=2.0) as client:
                    resp = client.get(f"{proxy_root}/health")
                    if resp.status_code == 200:
                        proxy_details = resp.json()
            except Exception:
                pass

        return {
            "base_llm": {
                "url": self.base_url,
                "running": base_running,
                "model_id": self.target_model,
                "model_loaded": model_loaded,
            },
            "quanta_proxy": {
                "url": self.quanta_url,
                "running": proxy_running,
                "details": proxy_details,
            },
            "gpu": gpu_telemetry,
        }

    # -------------------------------------------------------------------------
    # Auto-Spinup Orchestration
    # -------------------------------------------------------------------------

    def ensure_base_llm(self, timeout: float = 60.0) -> bool:
        """Ensures the Base LLM server is running and the target model is loaded."""
        if self.is_base_llm_running() and self.is_model_loaded():
            logger.info("Base LLM is already running and model '%s' is loaded.", self.target_model)
            return True

        print(format_ansi(f"[*] Checking Base LLM service on {self.base_url}...", "1;33"))

        if not self.is_base_llm_running():
            print(format_ansi("    -> Service is offline. Spawning Base LLM (Unsloth Studio / llama-server)...", "1;33"))
            launched = self.unsloth_mgr.ensure_unsloth_service_running(timeout=timeout)
            if not launched:
                print(format_ansi("    [!] Failed to spin up Base LLM automatically.", "1;31"))
                return False
            if self.unsloth_mgr._server_process:
                self._spawned_processes.append(self.unsloth_mgr._server_process)
            print(format_ansi(f"    [+] Base LLM server awakened on {self.base_url}", "1;32"))

        # Verify model loaded state
        if not self.is_model_loaded():
            print(format_ansi(f"[*] Model '{self.target_model}' is dormant. Requesting load into VRAM...", "1;33"))
            loaded = self.unsloth_mgr.ensure_model_loaded(self.target_model, timeout=timeout)
            if not loaded:
                print(format_ansi(f"    [!] Failed to verify/load model '{self.target_model}'.", "1;31"))
                return False
            print(format_ansi(f"    [+] Model '{self.target_model}' loaded in VRAM.", "1;32"))

        return True

    def ensure_proxy(
        self,
        backend_url: Optional[str] = None,
        timeout: float = 30.0,
        threshold: int = 2000,
        db_path: Optional[str] = None,
    ) -> bool:
        """Ensures the QUANTA Reverse Proxy is running on quanta_url, launching if down."""
        if self.is_proxy_running():
            logger.info("QUANTA Reverse Proxy is already running on %s", self.quanta_url)
            return True

        target_backend = backend_url or self.base_url
        host = self._extract_host(self.quanta_url)
        port = self._extract_port(self.quanta_url, default=8000)

        print(format_ansi(f"[*] Checking QUANTA Reverse Proxy on {self.quanta_url}...", "1;33"))
        print(format_ansi(f"    -> Reverse Proxy is offline. Spawning on {host}:{port} -> Backend: {target_backend}...", "1;33"))

        env = os.environ.copy()
        pythonpath = env.get("PYTHONPATH", "")
        src_path = str(REPO_ROOT / "src")
        root_path = str(REPO_ROOT)
        env["PYTHONPATH"] = f"{src_path};{root_path};{pythonpath}" if pythonpath else f"{src_path};{root_path}"
        env["QUANTA_BACKEND_URL"] = target_backend
        env["QUANTA_COMPRESSION_THRESHOLD"] = str(threshold)
        if db_path:
            env["QUANTA_PAGE_TABLE_PATH"] = str(db_path)

        cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "server.proxy:app",
            "--host",
            str(host),
            "--port",
            str(port),
            "--log-level",
            "warning",
        ]

        creationflags = 0
        if sys.platform == "win32":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS

        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(REPO_ROOT),
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creationflags,
            )
            self._spawned_processes.append(proc)
            logger.info("Spawned QUANTA Reverse Proxy subprocess (PID: %d)", proc.pid)
        except Exception as exc:
            logger.error("Failed to spawn QUANTA Reverse Proxy: %s", exc)
            print(format_ansi(f"    [!] Error spawning reverse proxy: {exc}", "1;31"))
            return False

        # Wait for service readiness
        t0 = time.time()
        while time.time() - t0 < timeout:
            time.sleep(0.5)
            if self.is_proxy_running(timeout=1.0):
                print(format_ansi(f"    [+] QUANTA Reverse Proxy successfully running on http://{host}:{port}/v1", "1;32"))
                return True

        print(format_ansi(f"    [!] QUANTA Reverse Proxy failed to become ready within {timeout:.1f}s.", "1;31"))
        return False

    def ensure_live_backend(self, timeout: float = 60.0) -> bool:
        """Coordinates full readiness check and auto-spinup for both Base LLM and Reverse Proxy."""
        print(format_ansi("\n================================================================================", "1;36"))
        print(format_ansi("       QUANTA LIVE BACKEND READINESS & AUTO-SPINUP VERIFICATION", "1;36"))
        print(format_ansi("================================================================================", "1;36"))

        # Step 1: Ensure Base LLM
        base_ok = self.ensure_base_llm(timeout=timeout)
        if not base_ok:
            print(format_ansi("\n[!] WARNING: Downstream Base LLM could not be verified.", "1;31"))
            print(format_ansi("    Inference requests through QUANTA will use local neuro-symbolic fallback.", "1;33"))

        # Step 2: Ensure QUANTA Reverse Proxy
        proxy_ok = self.ensure_proxy(backend_url=self.base_url, timeout=timeout)
        if not proxy_ok:
            print(format_ansi("\n[!] ERROR: QUANTA Reverse Proxy failed to initialize.", "1;31"))
            return False

        print(format_ansi("\n[+] All Required Services are Ready for Live Execution:", "1;32"))
        print(f"    * Base LLM (Unsloth Studio) : {self.base_url} [ONLINE - Model Loaded]")
        print(f"    * QUANTA Reverse Proxy     : {self.quanta_url} [ONLINE - Middleware Active]")
        print(format_ansi("================================================================================\n", "1;36"))
        return True

    def stop_spawned_services(self):
        """Cleanly terminates any subprocess that was launched by this manager instance."""
        for proc in self._spawned_processes:
            try:
                if proc.poll() is None:
                    proc.terminate()
                    proc.wait(timeout=2.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        self._spawned_processes.clear()

    # -------------------------------------------------------------------------
    # Diagnostics & Testing Utilities
    # -------------------------------------------------------------------------

    def test_proxy_completion(
        self,
        prompt: str = "Explain the difference between parametric LLMs and neuro-symbolic memory.",
        max_tokens: int = 100,
    ) -> Dict[str, Any]:
        """Sends a test chat completion request through the QUANTA Reverse Proxy."""
        if not self.is_proxy_running():
            self.ensure_proxy()

        payload = {
            "model": "quanta-context-expander",
            "messages": [
                {"role": "system", "content": "You are a concise expert technical assistant."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.1,
        }

        t0 = time.perf_counter()
        with httpx.Client(timeout=45.0) as client:
            resp = client.post(f"{self.quanta_url}/chat/completions", json=payload)
        dt = time.perf_counter() - t0

        if resp.status_code != 200:
            return {
                "success": False,
                "status_code": resp.status_code,
                "error": resp.text,
                "latency_s": dt,
            }

        data = resp.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = data.get("usage", {})
        meta = data.get("quanta_metadata", {})

        return {
            "success": True,
            "status_code": 200,
            "content": content,
            "latency_s": dt,
            "tokens": usage,
            "quanta_metadata": meta,
        }

    def run_mcp_test(self) -> Dict[str, Any]:
        """Runs an end-to-end self-test of the QUANTA Model Context Protocol (MCP) server."""
        from server.mcp_server import MCPServer
        from memory.global_kb import GlobalKnowledgeBase

        print(format_ansi("\n--- RUNNING QUANTA MCP SERVER TOOL SELF-TEST ---", "1;33"))
        db_path = REPO_ROOT / "data" / "wikipedia_quanta.db"
        global_kb = GlobalKnowledgeBase(db_path) if db_path.exists() else None

        server = MCPServer(transducer_backend="mock", global_kb=global_kb)
        results: Dict[str, Any] = {}

        # 1. tools/list
        req_list = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
        resp_list = server.handle_message(req_list) or {}
        tools = resp_list.get("result", {}).get("tools", [])
        tool_names = [t.get("name") for t in tools]
        results["tools_list"] = {"count": len(tools), "tools": tool_names, "status": "PASS" if len(tools) >= 5 else "FAIL"}
        print(f"  [1/5] tools/list: {len(tools)} tools discovered {tool_names}")

        # 2. quanta_reset_memory
        req_reset = {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "quanta_reset_memory", "arguments": {}}}
        resp_reset = server.handle_message(req_reset) or {}
        results["reset"] = resp_reset.get("result", {})
        print(f"  [2/5] quanta_reset_memory: {resp_reset.get('result', {}).get('content', [{}])[0].get('text', '')}")

        # 3. quanta_ingest_document
        sample_doc = (
            "Ada Lovelace was an English mathematician and writer, chiefly known for her work on "
            "Charles Babbage's mechanical general-purpose computer, the Analytical Engine. She wrote "
            "the first algorithm intended to be carried out by such a machine."
        )
        req_ingest = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "quanta_ingest_document", "arguments": {"doc_id": "test_ada", "content": sample_doc}},
        }
        t0 = time.perf_counter()
        resp_ingest = server.handle_message(req_ingest) or {}
        dt_ingest = (time.perf_counter() - t0) * 1000.0
        results["ingest"] = {"response": resp_ingest.get("result", {}), "latency_ms": round(dt_ingest, 2)}
        print(f"  [3/5] quanta_ingest_document: Ingested in {dt_ingest:.1f}ms")

        # 4. quanta_query_memory
        req_query = {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": "quanta_query_memory", "arguments": {"query": "Who worked on the Analytical Engine?", "max_tokens": 200}},
        }
        t0 = time.perf_counter()
        resp_query = server.handle_message(req_query) or {}
        dt_query = (time.perf_counter() - t0) * 1000.0
        q_text = resp_query.get("result", {}).get("content", [{}])[0].get("text", "")
        results["query"] = {"text": q_text, "latency_ms": round(dt_query, 2)}
        print(f"  [4/5] quanta_query_memory: Retrieved context in {dt_query:.1f}ms")

        # 5. quanta_get_memory_stats
        req_stats = {"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {"name": "quanta_get_memory_stats", "arguments": {}}}
        resp_stats = server.handle_message(req_stats) or {}
        stats_text = resp_stats.get("result", {}).get("content", [{}])[0].get("text", "")
        results["stats"] = stats_text
        print(f"  [5/5] quanta_get_memory_stats: {stats_text}")

        print(format_ansi("[+] All 5 MCP tools verified successfully!\n", "1;32"))
        return results

    def run_mcp_interactive(self):
        """Interactive terminal loop for manually testing MCP server queries."""
        from server.mcp_server import MCPServer
        from memory.global_kb import GlobalKnowledgeBase

        print(format_ansi("\n================================================================================", "1;36"))
        print(format_ansi("         QUANTA MCP SERVER INTERACTIVE TEST CONSOLE", "1;36"))
        print(format_ansi("================================================================================", "1;36"))
        print("Commands:")
        print("  ingest <text>       - Ingest document content into Merkle memory")
        print("  query <question>    - Retrieve compact sub-graph context via spreading activation")
        print("  entity <name>       - Inspect 1024-D vector bands and graph edges for an entity")
        print("  stats               - Show PageTable and ActiveCanvas memory statistics")
        print("  reset               - Clear working memory")
        print("  exit / quit         - Exit console\n")

        db_path = REPO_ROOT / "data" / "wikipedia_quanta.db"
        global_kb = GlobalKnowledgeBase(db_path) if db_path.exists() else None
        server = MCPServer(transducer_backend="mock", global_kb=global_kb)

        while True:
            try:
                raw = input(format_ansi("mcp> ", "1;32")).strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting MCP interactive console.")
                break

            if not raw:
                continue
            if raw.lower() in ("exit", "quit", "q"):
                print("Exiting MCP interactive console.")
                break

            parts = raw.split(" ", 1)
            cmd = parts[0].lower()
            arg = parts[1].strip() if len(parts) > 1 else ""

            if cmd == "ingest":
                if not arg:
                    print("Usage: ingest <document text>")
                    continue
                req = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {"name": "quanta_ingest_document", "arguments": {"doc_id": "manual", "content": arg}},
                }
                resp = server.handle_message(req) or {}
                txt = resp.get("result", {}).get("content", [{}])[0].get("text", "")
                print(format_ansi(txt, "1;36"))

            elif cmd == "query":
                if not arg:
                    print("Usage: query <question or topic>")
                    continue
                req = {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {"name": "quanta_query_memory", "arguments": {"query": arg, "max_tokens": 500}},
                }
                resp = server.handle_message(req) or {}
                txt = resp.get("result", {}).get("content", [{}])[0].get("text", "")
                print(format_ansi(txt, "1;37"))

            elif cmd == "entity":
                if not arg:
                    print("Usage: entity <entity name or QID>")
                    continue
                req = {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {"name": "quanta_get_entity_details", "arguments": {"name_or_cid": arg}},
                }
                resp = server.handle_message(req) or {}
                txt = resp.get("result", {}).get("content", [{}])[0].get("text", "")
                print(format_ansi(txt, "1;37"))

            elif cmd == "stats":
                req = {
                    "jsonrpc": "2.0",
                    "id": 4,
                    "method": "tools/call",
                    "params": {"name": "quanta_get_memory_stats", "arguments": {}},
                }
                resp = server.handle_message(req) or {}
                txt = resp.get("result", {}).get("content", [{}])[0].get("text", "")
                print(format_ansi(txt, "1;33"))

            elif cmd == "reset":
                req = {
                    "jsonrpc": "2.0",
                    "id": 5,
                    "method": "tools/call",
                    "params": {"name": "quanta_reset_memory", "arguments": {}},
                }
                resp = server.handle_message(req) or {}
                txt = resp.get("result", {}).get("content", [{}])[0].get("text", "")
                print(format_ansi(txt, "1;32"))
            else:
                print(f"Unknown command: '{cmd}'. Available: ingest, query, entity, stats, reset, exit")


# Global manager instance
_default_manager: Optional[QuantaServiceManager] = None


def get_service_manager() -> QuantaServiceManager:
    """Returns or initializes the singleton QuantaServiceManager instance."""
    global _default_manager
    if _default_manager is None:
        _default_manager = QuantaServiceManager()
        atexit.register(_default_manager.stop_spawned_services)
    return _default_manager


def main():
    parser = argparse.ArgumentParser(description="QUANTA Unified Service Manager & Test CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Status
    subparsers.add_parser("status", help="Print status of Base LLM, Reverse Proxy, MCP, and GPU")

    # Start
    start_p = subparsers.add_parser("start", help="Spin up backend services")
    start_p.add_argument("--all", action="store_true", help="Spin up both Base LLM and Reverse Proxy")
    start_p.add_argument("--proxy", action="store_true", help="Spin up QUANTA Reverse Proxy only")
    start_p.add_argument("--llm", action="store_true", help="Spin up Base LLM only")
    start_p.add_argument("--base-url", type=str, default="http://127.0.0.1:8888/v1", help="Base LLM URL")
    start_p.add_argument("--quanta-url", type=str, default="http://127.0.0.1:8000/v1", help="QUANTA Proxy URL")

    # Test MCP
    test_mcp_p = subparsers.add_parser("test-mcp", help="Run automated MCP tools self-test")
    test_mcp_p.add_argument("--interactive", action="store_true", help="Launch interactive MCP console")

    # Test Proxy
    test_proxy_p = subparsers.add_parser("test-proxy", help="Send test query through QUANTA Reverse Proxy")
    test_proxy_p.add_argument("--prompt", type=str, default="Explain how QUANTA compresses long dialogue contexts.")

    args = parser.parse_args()
    mgr = get_service_manager()

    if args.command == "status" or not args.command:
        status = mgr.get_service_status()
        print("\n" + format_ansi("================ QUANTA SERVICE STATUS DASHBOARD ================", "1;36"))
        base = status["base_llm"]
        base_st = format_ansi("ONLINE", "1;32") if base["running"] else format_ansi("OFFLINE", "1;31")
        model_st = format_ansi("LOADED (VRAM)", "1;32") if base["model_loaded"] else format_ansi("DORMANT/OFFLINE", "1;33")
        print(f"  Base LLM Server (:8888)  : {base_st} ({base['url']})")
        print(f"  Target Model In VRAM     : {model_st} ({base['model_id']})")

        proxy = status["quanta_proxy"]
        proxy_st = format_ansi("ONLINE", "1;32") if proxy["running"] else format_ansi("OFFLINE", "1;31")
        print(f"  QUANTA Reverse Proxy (:8000): {proxy_st} ({proxy['url']})")
        if proxy.get("details"):
            d = proxy["details"]
            print(f"    -> Nodes Stored: {d.get('stored_nodes', 0)} | Requests: {d.get('total_requests', 0)} | Tokens Saved: {d.get('tokens_saved_estimate', 0)}")

        gpu = status["gpu"]
        gpu_st = format_ansi(gpu.get("status", "unknown").upper(), "1;32" if gpu.get("status") == "active" else "1;33")
        print(f"  NVIDIA GPU Telemetry     : {gpu_st}")
        if gpu.get("status") == "active":
            print(f"    -> Device: {gpu.get('name')} | VRAM: {gpu.get('used_mb', 0):.0f}/{gpu.get('total_mb', 0):.0f} MB ({gpu.get('util_pct', 0)}% GPU Compute)")
        print(format_ansi("=================================================================\n", "1;36"))

    elif args.command == "start":
        if args.base_url:
            mgr.base_url = args.base_url
        if args.quanta_url:
            mgr.quanta_url = args.quanta_url

        if args.all or (not args.proxy and not args.llm):
            mgr.ensure_live_backend()
        elif args.llm:
            mgr.ensure_base_llm()
        elif args.proxy:
            mgr.ensure_proxy()

    elif args.command == "test-mcp":
        if args.interactive:
            mgr.run_mcp_interactive()
        else:
            mgr.run_mcp_test()

    elif args.command == "test-proxy":
        print(format_ansi(f"\n[*] Sending test query to {mgr.quanta_url}/chat/completions...", "1;33"))
        res = mgr.test_proxy_completion(prompt=args.prompt)
        if res["success"]:
            print(format_ansi("\n[+] Response Received Successfully:", "1;32"))
            print(f"Content: {res['content']}\n")
            print(f"Latency: {res['latency_s']:.3f}s | Tokens: {res['tokens']}")
            print(f"QUANTA Metadata: {res['quanta_metadata']}\n")
        else:
            print(format_ansi(f"\n[!] Request failed with HTTP {res['status_code']}: {res['error']}", "1;31"))


if __name__ == "__main__":
    main()
