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
    [ValidateSet("test", "test-verbose", "lint", "export", "help")]
    [string]$Target = "test"
)

$ErrorActionPreference = "Stop"

switch ($Target) {
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
    "help" {
        Write-Host "Available targets: test, test-verbose, lint, export, help" -ForegroundColor Green
    }
}
