[CmdletBinding()]
param(
    [string]$PythonExe = "python",
    [int]$Workers = 2,
    [string]$OutputDir
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssfffZ")
$output = if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    Join-Path $root "reproduction_runs\minimal_$stamp"
}
elseif ([IO.Path]::IsPathRooted($OutputDir)) {
    [IO.Path]::GetFullPath($OutputDir)
}
else {
    [IO.Path]::GetFullPath((Join-Path $root $OutputDir))
}

Push-Location $root
try {
    & $PythonExe -m pytest -p no:cacheprovider
    if ($LASTEXITCODE -ne 0) {
        throw "Test suite failed with exit code $LASTEXITCODE."
    }
    & (Join-Path $PSScriptRoot "01_reproduce_smoke.ps1") `
        -PythonExe $PythonExe `
        -Workers $Workers `
        -OutputDir $output
    if ($LASTEXITCODE -ne 0) {
        throw "Semantic smoke reproduction failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}

Write-Output "Minimal reproduction: PASS"
Write-Output "Output: $output"
