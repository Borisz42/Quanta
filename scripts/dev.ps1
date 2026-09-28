<#
.SYNOPSIS
    QUANTA Development Task Runner for Windows PowerShell.
.DESCRIPTION
    Provides shortcuts for running tests, linting, formatting, and generating artifact exports.
.EXAMPLE
    .\scripts\dev.ps1 test
    .\scripts\dev.ps1 export
#>

param (
    [Parameter(Position = 0, Mandatory = $false)]
    [ValidateSet("test", "test-verbose", "demo", "benchmark", "benchmark-quick", "benchmark-publish", "lint", "export", "gather-artifacts", "init", "sync-hf", "help")]
    [string]$Target = "test"
)

$ErrorActionPreference = "Stop"

switch ($Target) {
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
        Write-Host "Available targets: benchmark, benchmark-quick, benchmark-publish, test, test-verbose, demo, lint, export, gather-artifacts, init, sync-hf, help" -ForegroundColor Green
    }
}
