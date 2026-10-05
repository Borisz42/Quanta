<#
.SYNOPSIS
    QUANTA Context Expansion Server & MCP Runner for Windows PowerShell.
.DESCRIPTION
    Starts and manages either the OpenAI-compatible reverse proxy (FastAPI/Uvicorn) or
    the Model Context Protocol (MCP) JSON-RPC stdio server, with automated backend readiness checks.
.PARAMETER Port
    HTTP port for the reverse proxy (default: 8000).
.PARAMETER Host
    Bind address for the reverse proxy (default: 127.0.0.1).
.PARAMETER Backend
    Target downstream LLM backend URL (default: http://127.0.0.1:8888/v1).
.PARAMETER Mode
    Server operating mode: 'proxy' or 'mcp' (default: proxy).
.PARAMETER Threshold
    Dialogue compression token threshold (default: 2000).
.PARAMETER DbPath
    Optional path to persistent SQLite PageTable database.
.PARAMETER NoBackend
    Skip automated verification and spinup of downstream Base LLM (port 8888).
.PARAMETER Test
    Run self-test diagnostics (proxy completion or MCP tools test).
.PARAMETER Interactive
    Open interactive test console (for MCP mode).
.PARAMETER Status
    Display service status dashboard across Base LLM, Reverse Proxy, MCP, and GPU.
.EXAMPLE
    # Start Reverse Proxy with automated backend verification and custom port:
    .\scripts\serve.ps1 -Port 8000 -Backend "http://127.0.0.1:8888/v1" -Mode proxy

    # Start MCP server over stdio:
    .\scripts\serve.ps1 -Mode mcp

    # Test Reverse Proxy end-to-end:
    .\scripts\serve.ps1 -Mode proxy -Test

    # Test MCP server tools:
    .\scripts\serve.ps1 -Mode mcp -Test

    # Open interactive MCP test console:
    .\scripts\serve.ps1 -Mode mcp -Interactive

    # View service status dashboard:
    .\scripts\serve.ps1 -Status
#>

param (
    [Parameter(Mandatory = $false)]
    [int]$Port = 8000,

    [Parameter(Mandatory = $false)]
    [Alias("Host")]
    [string]$BindHost = "127.0.0.1",

    [Parameter(Mandatory = $false)]
    [string]$Backend = "http://127.0.0.1:8888/v1",

    [Parameter(Mandatory = $false)]
    [ValidateSet("proxy", "mcp")]
    [string]$Mode = "proxy",

    [Parameter(Mandatory = $false)]
    [int]$Threshold = 2000,

    [Parameter(Mandatory = $false)]
    [string]$DbPath = "",

    [Parameter(Mandatory = $false)]
    [switch]$NoBackend,

    [Parameter(Mandatory = $false)]
    [switch]$Test,

    [Parameter(Mandatory = $false)]
    [switch]$Interactive,

    [Parameter(Mandatory = $false)]
    [switch]$Status
)

$ErrorActionPreference = "Stop"

# Set environment variables for server subprocess
$env:PYTHONPATH = ".;src;$env:PYTHONPATH"
$env:QUANTA_BACKEND_URL = $Backend
$env:QUANTA_COMPRESSION_THRESHOLD = "$Threshold"
if ($DbPath -ne "") {
    $env:QUANTA_PAGE_TABLE_PATH = $DbPath
}

# 1. Status Dashboard
if ($Status) {
    python -m server.service_manager status
    exit 0
}

# 2. Interactive Testing
if ($Interactive) {
    if ($Mode -eq "mcp") {
        python -m server.service_manager test-mcp --interactive
    } else {
        Write-Host "Interactive mode is designed for MCP server testing (-Mode mcp -Interactive)." -ForegroundColor Yellow
        python -m server.service_manager test-mcp --interactive
    }
    exit 0
}

# 3. Diagnostic Testing
if ($Test) {
    if ($Mode -eq "proxy") {
        Write-Host "Running End-to-End Reverse Proxy Diagnostics..." -ForegroundColor Cyan
        python -m server.service_manager test-proxy
    } else {
        Write-Host "Running QUANTA MCP Server Tools Diagnostics..." -ForegroundColor Cyan
        python -m server.service_manager test-mcp
    }
    exit 0
}

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  QUANTA Context Expansion Middleware & Server Runner    " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  Mode      : $Mode" -ForegroundColor Green
Write-Host "  Host      : $BindHost" -ForegroundColor Green
Write-Host "  Port      : $Port" -ForegroundColor Green
Write-Host "  Backend   : $Backend" -ForegroundColor Green
Write-Host "  Threshold : $Threshold tokens" -ForegroundColor Green
if ($DbPath -ne "") {
    Write-Host "  Database  : $DbPath" -ForegroundColor Green
}
Write-Host "==========================================================" -ForegroundColor Cyan

switch ($Mode) {
    "proxy" {
        # Check and ensure downstream Base LLM if requested
        if (-not $NoBackend) {
            Write-Host "Verifying downstream Base LLM on $Backend..." -ForegroundColor Cyan
            python -c "from server.service_manager import get_service_manager; mgr = get_service_manager(); mgr.base_url = '$Backend'; mgr.ensure_base_llm()"
        }

        Write-Host "Starting OpenAI-compatible Reverse Proxy on http://${BindHost}:${Port}/v1 ..." -ForegroundColor Cyan
        python -m uvicorn server.proxy:app --host $BindHost --port $Port --log-level info
    }
    "mcp" {
        Write-Host "Starting Model Context Protocol (MCP) stdio server..." -ForegroundColor Cyan
        python -m server.mcp_server
    }
}
