[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$recovered = @(
    Get-ChildItem (Join-Path $root "results") -Directory |
        Where-Object {
            Test-Path (Join-Path $_.FullName "RECOVERY_OVERRIDE.json")
        }
)

if ($recovered.Count -ne 2) {
    throw "Expected exactly two recovered formal-result directories."
}

$checked = 0
foreach ($directory in $recovered) {
    $formal = Get-Content `
        -LiteralPath (Join-Path $directory.FullName "FORMAL_PROVENANCE.json") `
        -Encoding utf8 `
        -Raw |
        ConvertFrom-Json
    $recoveryPath = Join-Path $directory.FullName "RECOVERY_OVERRIDE.json"
    $recovery = Get-Content -LiteralPath $recoveryPath -Encoding utf8 -Raw |
        ConvertFrom-Json

    if ([string]$formal.status -cne "recovered_formal_result") {
        throw "Unexpected formal status in $($directory.Name)."
    }
    $recoveryHash = (
        Get-FileHash -Algorithm SHA256 -LiteralPath $recoveryPath
    ).Hash.ToLowerInvariant()
    if ($recoveryHash -cne [string]$formal.recovery_record_sha256) {
        throw "Recovery-record hash mismatch in $($directory.Name)."
    }

    foreach ($evidence in @($recovery.machine_evidence)) {
        $relative = [string]$evidence.path_relative_to_repo
        if ([IO.Path]::IsPathRooted($relative) -or $relative -match "(^|/)\.\.(/|$)") {
            throw "Unsafe recovery evidence path: $relative"
        }
        $path = Join-Path $root ($relative.Replace("/", "\"))
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "Missing recovery evidence: $relative"
        }
        $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash.ToLowerInvariant()
        if ($hash -cne [string]$evidence.sha256) {
            throw "Recovery evidence hash mismatch: $relative"
        }
        if ((Get-Item -LiteralPath $path).Length -ne [long]$evidence.bytes) {
            throw "Recovery evidence byte-count mismatch: $relative"
        }
        $checked += 1
    }

    if ([string]$formal.output_label -ceq "method_ladder_n10") {
        if ($formal.native_clean_run_present -ne $false) {
            throw "Method recovery must declare no native CLEAN_RUN."
        }
    }
    elseif ([string]$formal.output_label -ceq "parallel_scaling_n10") {
        if ($formal.native_clean_run_present -ne $true) {
            throw "Scaling recovery must declare a native CLEAN_RUN."
        }
        $roles = @($recovery.machine_evidence | ForEach-Object { [string]$_.role })
        foreach ($required in @(
            "native_clean_run",
            "wrapper_failed_record",
            "target_stdout",
            "target_stderr"
        )) {
            if ($roles -notcontains $required) {
                throw "Scaling recovery lacks required role: $required"
            }
        }
    }
}

Write-Output "Recovery-evidence verification: PASS ($checked machine-evidence files)"
