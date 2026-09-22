<#
.SYNOPSIS
    QUANTA Context Expansion Server Startup Script for Windows PowerShell.
.DESCRIPTION
    Starts either the OpenAI-compatible reverse proxy (FastAPI/Uvicorn) or
    the Model Context Protocol (MCP) JSON-RPC stdio server.
.PARAMETER Port
    HTTP port for the reverse proxy (default: 8000).
.PARAMETER Host
    Bind address for the reverse proxy (default: 127.0.0.1).
.PARAMETER Backend
    Target downstream LLM / SLM backend URL (default: http://localhost:8888/v1).
.PARAMETER Mode
    Server operating mode: 'proxy' or 'mcp' (default: proxy).
.PARAMETER Threshold
    Dialogue compression token threshold (default: 2000).
.PARAMETER DbPath
    Optional path to persistent SQLite PageTable database.
.EXAMPLE
    .\scripts\serve.ps1 -Port 8000 -Backend "http://localhost:8888/v1"
    .\scripts\serve.ps1 -Mode mcp
#>

param (
    [Parameter(Mandatory = $false)]
    [int]$Port = 8000,

    [Parameter(Mandatory = $false)]
    [string]$Host = "127.0.0.1",

    [Parameter(Mandatory = $false)]
    [string]$Backend = "http://localhost:8888/v1",

    [Parameter(Mandatory = $false)]
    [ValidateSet("proxy", "mcp")]
    [string]$Mode = "proxy",

    [Parameter(Mandatory = $false)]
    [int]$Threshold = 2000,

    [Parameter(Mandatory = $false)]
    [string]$DbPath = ""
)

$ErrorActionPreference = "Stop"

# Set environment variables for server subprocess
$env:PYTHONPATH = ".;src;$env:PYTHONPATH"
$env:QUANTA_BACKEND_URL = $Backend
$env:QUANTA_COMPRESSION_THRESHOLD = "$Threshold"
if ($DbPath -ne "") {
    $env:QUANTA_PAGE_TABLE_PATH = $DbPath
}

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  QUANTA Context Expansion Middleware & Server Runner    " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  Mode      : $Mode" -ForegroundColor Green
Write-Host "  Host      : $Host" -ForegroundColor Green
Write-Host "  Port      : $Port" -ForegroundColor Green
Write-Host "  Backend   : $Backend" -ForegroundColor Green
Write-Host "  Threshold : $Threshold tokens" -ForegroundColor Green
if ($DbPath -ne "") {
    Write-Host "  Database  : $DbPath" -ForegroundColor Green
}
Write-Host "==========================================================" -ForegroundColor Cyan

switch ($Mode) {
    "proxy" {
        Write-Host "Starting OpenAI-compatible Reverse Proxy on http://${Host}:${Port}/v1 ..." -ForegroundColor Cyan
        python -m uvicorn server.proxy:app --host $Host --port $Port --log-level info
    }
    "mcp" {
        Write-Host "Starting Model Context Protocol (MCP) stdio server..." -ForegroundColor Cyan
        python -m server.mcp_server
    }
}
