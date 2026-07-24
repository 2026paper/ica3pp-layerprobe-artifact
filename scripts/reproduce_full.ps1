[CmdletBinding()]
param(
    [string]$PythonExe = "python",
    [string]$ResultTag = "reproduction",
    [switch]$ValidateOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path

if ($ResultTag -notmatch "^[A-Za-z0-9_-]+$") {
    throw "ResultTag may contain only letters, digits, underscore, and hyphen."
}

Push-Location $root
try {
    if ($ValidateOnly) {
        $queueScript = Join-Path $PSScriptRoot "queue_formal_singlehost.ps1"
        if (-not (Test-Path -LiteralPath $queueScript -PathType Leaf)) {
            throw "Missing full-reproduction queue script: $queueScript"
        }
        & $PythonExe --version
        if ($LASTEXITCODE -ne 0) {
            throw "Python validation failed with exit code $LASTEXITCODE."
        }
        Write-Output "Full reproduction entry-point validation: PASS"
        return
    }
    & (Join-Path $PSScriptRoot "queue_formal_singlehost.ps1") `
        -PythonExe $PythonExe `
        -ResultTag $ResultTag
    if ($LASTEXITCODE -ne 0) {
        throw "Full reproduction queue failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}

Write-Output "Full reproduction queue: PASS"
