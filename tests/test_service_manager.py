"""Unit tests for QuantaServiceManager and unified server lifecycle."""

import pytest
from server.service_manager import QuantaServiceManager, get_service_manager


class TestQuantaServiceManager:
    """Tests for QuantaServiceManager discovery and diagnostics."""

    def test_manager_url_parsing(self):
        mgr = QuantaServiceManager(
            base_url="http://127.0.0.1:8888/v1",
            quanta_url="http://127.0.0.1:8000/v1",
        )
        assert mgr._extract_host(mgr.base_url) == "127.0.0.1"
        assert mgr._extract_port(mgr.base_url, 8888) == 8888
        assert mgr._extract_host(mgr.quanta_url) == "127.0.0.1"
        assert mgr._extract_port(mgr.quanta_url, 8000) == 8000

    def test_singleton_accessor(self):
        m1 = get_service_manager()
        m2 = get_service_manager()
        assert m1 is m2

    def test_service_status_structure(self):
        mgr = QuantaServiceManager()
        st = mgr.get_service_status()
        assert "base_llm" in st
        assert "quanta_proxy" in st
        assert "gpu" in st
        assert "running" in st["base_llm"]
        assert "running" in st["quanta_proxy"]

    def test_mcp_self_test(self):
        mgr = QuantaServiceManager()
        res = mgr.run_mcp_test()
        assert "tools_list" in res
        assert res["tools_list"]["count"] >= 5
        assert "reset" in res
        assert "ingest" in res
        assert "query" in res
        assert "stats" in res
