[CmdletBinding()]
param(
    [string]$PythonExe,
    [ValidateRange(1, 16)]
    [int]$Workers = 2,
    [string]$OutputDir
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path -LiteralPath (Join-Path $ScriptDir "..")).Path
$SrcDir = Join-Path $RepoRoot "src"

function Resolve-OutputDirectory {
    param([string]$Requested, [string]$DefaultName)
    if ([string]::IsNullOrWhiteSpace($Requested)) {
        return [IO.Path]::GetFullPath(
            (Join-Path $RepoRoot ("artifact_runs\" + $DefaultName))
        )
    }
    if ([IO.Path]::IsPathRooted($Requested)) {
        return [IO.Path]::GetFullPath($Requested)
    }
    return [IO.Path]::GetFullPath((Join-Path $RepoRoot $Requested))
}

function Resolve-Python {
    param([string]$Requested)
    $candidate = $Requested
    if ([string]::IsNullOrWhiteSpace($candidate)) {
        $candidate = $env:LAYERPROBE_PYTHON
    }
    if ([string]::IsNullOrWhiteSpace($candidate)) {
        $candidate = "python"
    }
    if (Test-Path -LiteralPath $candidate -PathType Leaf) {
        return (Resolve-Path -LiteralPath $candidate).Path
    }
    $command = Get-Command -Name $candidate -CommandType Application -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($null -eq $command) {
        throw "Python executable was not found: $candidate"
    }
    return $command.Source
}

function Invoke-LoggedNative {
    param(
        [string]$Executable,
        [string[]]$ArgumentList,
        [string]$LogPath
    )
    $savedPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $Executable @ArgumentList 2>&1 | Tee-Object -FilePath $LogPath
        $nativeExit = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $savedPreference
    }
    if ($nativeExit -ne 0) {
        throw "Command failed with exit code ${nativeExit}: $Executable"
    }
}

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$runName = "01_reproduce_smoke_${stamp}_pid$PID"
$RunDir = Resolve-OutputDirectory -Requested $OutputDir -DefaultName $runName
$transcriptStarted = $false
$exitCode = 1

try {
    if (Test-Path -LiteralPath $RunDir) {
        throw "Refusing to overwrite an existing output directory: $RunDir"
    }
    New-Item -ItemType Directory -Path $RunDir | Out-Null
    Start-Transcript -LiteralPath (Join-Path $RunDir "run.log") -NoClobber | Out-Null
    $transcriptStarted = $true

    $resolvedPython = Resolve-Python -Requested $PythonExe
    $env:PYTHONPATH = "$SrcDir$([IO.Path]::PathSeparator)$RepoRoot"
    $env:PYTHONHASHSEED = "0"
    $env:PYTHONDONTWRITEBYTECODE = "1"

    $deadlineOutput = Join-Path $RunDir "deadline_smoke"
    $oracleOutput = Join-Path $RunDir "independent_oracle_smoke"
    $deadlineConfig = Join-Path $RepoRoot "experiments\deadline_profile_8c32g.json"
    $oracleConfig = Join-Path $RepoRoot "experiments\independent_trace_oracle_config.json"

    Invoke-LoggedNative `
        -Executable $resolvedPython `
        -ArgumentList @(
            "-m", "pytest",
            "-c", (Join-Path $RepoRoot "pyproject.toml"),
            "-p", "no:cacheprovider",
            (Join-Path $RepoRoot "tests")
        ) `
        -LogPath (Join-Path $RunDir "pytest.log")

    Invoke-LoggedNative `
        -Executable $resolvedPython `
        -ArgumentList @(
            (Join-Path $RepoRoot "experiments\deadline_runner.py"),
            "--config", $deadlineConfig,
            "--output", $deadlineOutput,
            "--mode", "smoke",
            "--only", "correctness_gate",
            "--ignore-freeze"
        ) `
        -LogPath (Join-Path $RunDir "deadline_smoke.log")

    Invoke-LoggedNative `
        -Executable $resolvedPython `
        -ArgumentList @(
            (Join-Path $RepoRoot "experiments\independent_trace_oracle.py"),
            "--config", $oracleConfig,
            "--output", $oracleOutput,
            "--smoke",
            "--workers", [string]$Workers
        ) `
        -LogPath (Join-Path $RunDir "independent_oracle_smoke.log")

    $semanticPath = Join-Path $deadlineOutput "semantic_checks.json"
    $oracleSummaryPath = Join-Path $oracleOutput "summary.json"
    if (-not (Test-Path -LiteralPath $semanticPath -PathType Leaf)) {
        throw "Smoke semantic checks were not produced: $semanticPath"
    }
    if (-not (Test-Path -LiteralPath $oracleSummaryPath -PathType Leaf)) {
        throw "Independent-oracle summary was not produced: $oracleSummaryPath"
    }

    $semanticChecks = @(Get-Content -Raw -LiteralPath $semanticPath | ConvertFrom-Json)
    if ($semanticChecks.Count -eq 0) {
        throw "Smoke semantic-check list is empty."
    }
    $failedChecks = @($semanticChecks | Where-Object { $_.status -ne "PASS" })
    if ($failedChecks.Count -ne 0) {
        throw "At least one deadline smoke semantic check failed."
    }

    $oracleSummary = Get-Content -Raw -LiteralPath $oracleSummaryPath | ConvertFrom-Json
    if (-not ([string]$oracleSummary.status).StartsWith("PASS_")) {
        throw "Independent-oracle smoke did not report PASS: $($oracleSummary.status)"
    }
    if (
        [int]$oracleSummary.comparison.counts.validity_mismatch_count -ne 0 -or
        [int]$oracleSummary.comparison.counts.factorized_validity_mismatch_count -ne 0 -or
        [int]$oracleSummary.comparison.counts.flat_trace_mismatch_count -ne 0 -or
        [int]$oracleSummary.comparison.counts.factorized_trace_mismatch_count -ne 0 -or
        [int]$oracleSummary.comparison.counts.direct_candidate_mismatch_count -ne 0 -or
        [int]$oracleSummary.comparison.counts.factorized_candidate_mismatch_count -ne 0
    ) {
        throw "Independent-oracle smoke reported a semantic mismatch."
    }

    @(
        "# LayerProbe artifact smoke result",
        "",
        "- Status: **PASS**",
        ('- Completed: `{0}`' -f (Get-Date).ToString("o")),
        ('- Python: `{0}`' -f $resolvedPython),
        ('- Oracle workers: `{0}`' -f $Workers),
        "- Unit/integration tests: PASS",
        "- Matched semantic digest gate: PASS",
        "- Independent trace-oracle smoke: PASS",
        "",
        "This is a quick implementation check, not a replacement for the frozen full-domain results."
    ) | Set-Content -LiteralPath (Join-Path $RunDir "SMOKE_RESULT.md") -Encoding UTF8

    Write-Output "Artifact smoke: PASS"
    Write-Output "Output: $RunDir"
    $exitCode = 0
}
catch {
    Write-Error $_
    $exitCode = 1
}
finally {
    if ($transcriptStarted) {
        Stop-Transcript | Out-Null
    }
}

exit $exitCode
