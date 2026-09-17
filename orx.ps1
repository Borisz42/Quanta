<#
.SYNOPSIS
    OpenResearch (orx) PowerShell runner for the QUANTA repository.
.DESCRIPTION
    Wraps the local openresearch-cli/orx.exe binary and forwards all arguments,
    preserving exit codes and streaming output cleanly.
.EXAMPLE
    .\orx.ps1 up
    .\orx.ps1 skill
    .\orx.ps1 projects
#>

[CmdletBinding()]
param (
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Arguments
)

$orxPath = Join-Path $PSScriptRoot "openresearch-cli\orx.exe"

if (-not (Test-Path $orxPath)) {
    Write-Error "OpenResearch CLI binary not found at '$orxPath'. Please download and place orx.exe in the openresearch-cli/ directory."
    exit 1
}

& $orxPath @Arguments
exit $LASTEXITCODE
