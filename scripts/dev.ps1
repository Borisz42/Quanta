<#
.SYNOPSIS
    QUANTA Development Task Runner for Windows PowerShell.
.DESCRIPTION
    Provides shortcuts for running benchmarks, servers, tests, linting, formatting, and generating artifact exports.
.EXAMPLE
    .\scripts\dev.ps1 benchmark
    .\scripts\dev.ps1 serve
    .\scripts\dev.ps1 test-mcp
    .\scripts\dev.ps1 status
#>

param (
    [Parameter(Position = 0, Mandatory = $false)]
    [ValidateSet("serve", "serve-mcp", "test-mcp", "test-proxy", "interactive-mcp", "status", "benchmark", "benchmark-quick", "benchmark-publish", "demo", "test", "test-verbose", "lint", "export", "gather-artifacts", "init", "sync-hf", "help")]
    [string]$Target = "help"
)

$ErrorActionPreference = "Stop"

# Ensure repository root and src are in PYTHONPATH
$env:PYTHONPATH = ".;src;$env:PYTHONPATH"

switch ($Target) {
    "serve" {
        Write-Host "Starting QUANTA Reverse Proxy with automated backend verification..." -ForegroundColor Cyan
        .\scripts\serve.ps1 -Mode proxy
    }
    "serve-mcp" {
        Write-Host "Starting QUANTA Model Context Protocol (MCP) server..." -ForegroundColor Cyan
        .\scripts\serve.ps1 -Mode mcp
    }
    "test-mcp" {
        Write-Host "Running QUANTA Model Context Protocol (MCP) tools self-test..." -ForegroundColor Cyan
        .\scripts\serve.ps1 -Mode mcp -Test
    }
    "test-proxy" {
        Write-Host "Testing QUANTA Reverse Proxy end-to-end..." -ForegroundColor Cyan
        .\scripts\serve.ps1 -Mode proxy -Test
    }
    "interactive-mcp" {
        Write-Host "Launching QUANTA MCP interactive test console..." -ForegroundColor Cyan
        .\scripts\serve.ps1 -Mode mcp -Interactive
    }
    "status" {
        python -m server.service_manager status
    }
    "benchmark" {
        Write-Host "Launching QUANTA Interactive Paired Benchmark Suite..." -ForegroundColor Cyan
        python scripts/run_paired_benchmarks.py --interactive
    }
    "benchmark-quick" {
        Write-Host "Running Quick Mock Paired Benchmark Smoke Test..." -ForegroundColor Cyan
        python scripts/run_paired_benchmarks.py --suite humaneval:2,arc_science:2,musique:2,squad_overhead:2 --mode mock --export-submissions --export-latex
    }
    "benchmark-publish" {
        Write-Host "Running Publication-Grade 3-Way Ablation Benchmark Suite..." -ForegroundColor Cyan
        python scripts/run_paired_benchmarks.py --ablation-mode 3way --export-submissions --export-latex --interactive
    }
    "demo" {
        Write-Host "Running QUANTA Context Expansion System Demonstration..." -ForegroundColor Cyan
        python scripts/demonstrate_context_expansion.py
    }
    "test" {
        Write-Host "Running QUANTA test suite..." -ForegroundColor Cyan
        pytest
    }
    "test-verbose" {
        Write-Host "Running verbose QUANTA test suite..." -ForegroundColor Cyan
        pytest -v -s
    }
    "lint" {
        Write-Host "Running ruff check..." -ForegroundColor Cyan
        if (Get-Command ruff -ErrorAction SilentlyContinue) {
            ruff check src/ tests/
        } else {
            Write-Host "ruff is not installed. Skipping lint." -ForegroundColor Yellow
        }
    }
    "export" {
        Write-Host "Generating translation examples and exports..." -ForegroundColor Cyan
        python src/scripts/generate_translation_examples.py
    }
    "gather-artifacts" {
        Write-Host "Auditing and gathering all required QUANTA runtime artifacts..." -ForegroundColor Cyan
        python scripts/gather_artifacts.py
    }
    "init" {
        Write-Host "Initializing QUANTA environment and downloading runtime artifacts..." -ForegroundColor Cyan
        python scripts/init_quanta.py
    }
    "sync-hf" {
        Write-Host "Checking synchronization with Hugging Face (Borisz42/QUANTA)..." -ForegroundColor Cyan
        python scripts/sync_hf.py --check
    }
    "help" {
        Write-Host "Available targets:" -ForegroundColor Green
        Write-Host "  Servers & Services:" -ForegroundColor Cyan
        Write-Host "    serve           - Spin up Reverse Proxy on :8000 with auto-backend detection"
        Write-Host "    serve-mcp       - Start MCP JSON-RPC stdio server"
        Write-Host "    test-proxy      - Run end-to-end chat completion test through proxy"
        Write-Host "    test-mcp        - Run self-test across all 5 MCP tools"
        Write-Host "    interactive-mcp - Open interactive terminal shell to test MCP queries"
        Write-Host "    status          - View live service status dashboard"
        Write-Host "  Benchmarks & Tests:" -ForegroundColor Cyan
        Write-Host "    benchmark       - Launch interactive paired benchmark suite"
        Write-Host "    benchmark-quick - Run quick mock benchmark smoke test"
        Write-Host "    benchmark-publish - Run full 3-way ablation benchmark suite"
        Write-Host "    test            - Run pytest suite"
        Write-Host "    test-verbose    - Run verbose pytest suite"
        Write-Host "    demo            - Run context expansion demonstration"
    }
}
