"""Unit tests for UnslothServerManager and Strict CPU Offload Guard."""

import os
import pytest
from server.unsloth_manager import UnslothServerManager, ChatResponse


class TestUnslothServerManager:
    """Test suite for server lifecycle and hardware guard policies."""

    def test_chat_response_unpacking(self):
        """ChatResponse must support both dict lookup and 4-tuple unpacking."""
        resp = ChatResponse({
            "content": "Hello World",
            "latency_s": 0.5,
            "completion_tokens": 10,
            "tokens_per_sec": 20.0,
            "gpu_telemetry": {"used_mb": 4096},
        })

        # Dict access
        assert resp["content"] == "Hello World"
        assert resp.content == "Hello World"
        assert resp["latency_s"] == 0.5

        # Tuple unpack
        content, latency, tokens, tps = resp
        assert content == "Hello World"
        assert latency == 0.5
        assert tokens == 10
        assert tps == 20.0

    def test_strict_cpu_guard_raises_runtime_error_when_no_gpu(self, monkeypatch):
        """Simulating missing GPU must raise RuntimeError if CPU offload not permitted."""
        monkeypatch.setenv("QUANTA_SIMULATE_NO_GPU", "1")
        monkeypatch.delenv("QUANTA_ALLOW_CPU_OFFLOAD", raising=False)

        mgr = UnslothServerManager(allow_cpu_offload=False)
        with pytest.raises(RuntimeError) as exc_info:
            mgr.enforce_gpu_policy()

        assert "Strict GPU Execution Policy Enforced" in str(exc_info.value)
        assert "QUANTA_ALLOW_CPU_OFFLOAD=1" in str(exc_info.value)

    def test_strict_cpu_guard_allows_execution_when_flag_set(self, monkeypatch):
        """Setting QUANTA_ALLOW_CPU_OFFLOAD=1 allows fallback execution with warning."""
        monkeypatch.setenv("QUANTA_SIMULATE_NO_GPU", "1")
        monkeypatch.setenv("QUANTA_ALLOW_CPU_OFFLOAD", "1")

        mgr = UnslothServerManager()
        result = mgr.enforce_gpu_policy()
        # Returns False indicating running on CPU
        assert result is False

    def test_gpu_telemetry_simulated_unavailable(self, monkeypatch):
        """Telemetry must report unavailable when simulated flag is on."""
        monkeypatch.setenv("QUANTA_SIMULATE_NO_GPU", "1")
        mgr = UnslothServerManager()
        telem = mgr.get_gpu_telemetry()
        assert telem["status"] == "unavailable"

    def test_gpu_telemetry_fields_on_active_hardware(self):
        """On physical GPU hardware, telemetry dictionary has standard expected keys."""
        mgr = UnslothServerManager()
        telem = mgr.get_gpu_telemetry()
        assert "status" in telem
        if telem["status"] == "active":
            assert "name" in telem
            assert "used_mb" in telem
            assert "total_mb" in telem
            assert "util_pct" in telem
