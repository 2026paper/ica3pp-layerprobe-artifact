[CmdletBinding()]
param(
    [string]$PythonExe,
    [string]$PrimaryRunDir,
    [string]$CommunicationDir,
    [string]$OracleDir,
    [string]$CacheAblationDir,
    [string]$AgentSensitivityDir,
    [string]$StressDir,
    [string]$FrozenSourceRoot,
    [string]$OutputDir
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path -LiteralPath (Join-Path $ScriptDir "..")).Path
$SrcDir = Join-Path $RepoRoot "src"

function Resolve-RepoPath {
    param([string]$Requested, [string]$DefaultRelative)
    $candidate = if ([string]::IsNullOrWhiteSpace($Requested)) {
        $DefaultRelative
    } else {
        $Requested
    }
    if ([IO.Path]::IsPathRooted($candidate)) {
        return [IO.Path]::GetFullPath($candidate)
    }
    return [IO.Path]::GetFullPath((Join-Path $RepoRoot $candidate))
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
$defaultOutput = "artifact_runs\02_verify_frozen_outputs_${stamp}_pid$PID"
$RunDir = Resolve-RepoPath -Requested $OutputDir -DefaultRelative $defaultOutput
$resolvedPrimary = Resolve-RepoPath `
    -Requested $PrimaryRunDir `
    -DefaultRelative "results\deadline_paper_distancefix_20260723_xeon"
$resolvedCommunication = Resolve-RepoPath `
    -Requested $CommunicationDir `
    -DefaultRelative "results\communication_full_24624_distancefix_provenance_v2_20260723_xeon"
$resolvedOracle = Resolve-RepoPath `
    -Requested $OracleDir `
    -DefaultRelative "results\independent_trace_oracle_full_24624_distancefix_20260723_xeon"
$resolvedCache = Resolve-RepoPath `
    -Requested $CacheAblationDir `
    -DefaultRelative "results\cache_key_ablation_full_24624_distancefix_20260723_xeon"
$resolvedAgent = Resolve-RepoPath `
    -Requested $AgentSensitivityDir `
    -DefaultRelative "results\agent_sensitivity_full_24624_distancefix_provenance_v2_20260723_xeon"
$resolvedStress = Resolve-RepoPath `
    -Requested $StressDir `
    -DefaultRelative "results\range_extension_stress_distancefix_provenance_v2_20260723_xeon"
$resolvedFrozenSource = Resolve-RepoPath `
    -Requested $FrozenSourceRoot `
    -DefaultRelative "frozen_source_snapshots\distancefix_pre_flatp8\src"

$transcriptStarted = $false
$exitCode = 1

try {
    if (Test-Path -LiteralPath $RunDir) {
        throw "Refusing to overwrite an existing output directory: $RunDir"
    }
    foreach ($inputDirectory in @(
        $resolvedPrimary,
        $resolvedCommunication,
        $resolvedOracle,
        $resolvedCache,
        $resolvedAgent,
        $resolvedStress,
        $resolvedFrozenSource
    )) {
        if (-not (Test-Path -LiteralPath $inputDirectory -PathType Container)) {
            throw "Frozen input directory is missing: $inputDirectory"
        }
    }

    New-Item -ItemType Directory -Path $RunDir | Out-Null
    Start-Transcript -LiteralPath (Join-Path $RunDir "run.log") -NoClobber | Out-Null
    $transcriptStarted = $true

    $resolvedPython = Resolve-Python -Requested $PythonExe
    $env:PYTHONPATH = "$SrcDir$([IO.Path]::PathSeparator)$RepoRoot"
    $env:PYTHONDONTWRITEBYTECODE = "1"

    $deadlineVerification = Join-Path $RunDir "deadline_verification"
    $enhancedVerification = Join-Path $RunDir "enhanced_verification"

    Invoke-LoggedNative `
        -Executable $resolvedPython `
        -ArgumentList @(
            (Join-Path $RepoRoot "experiments\verify_deadline_outputs.py"),
            "--run-dir", $resolvedPrimary,
            "--communication-dir", $resolvedCommunication,
            "--frozen-snapshot-root", (Split-Path -Parent $resolvedFrozenSource),
            "--output", $deadlineVerification
        ) `
        -LogPath (Join-Path $RunDir "verify_deadline_outputs.log")

    Invoke-LoggedNative `
        -Executable $resolvedPython `
        -ArgumentList @(
            (Join-Path $RepoRoot "experiments\verify_enhanced_outputs.py"),
            "--oracle-dir", $resolvedOracle,
            "--cache-dir", $resolvedCache,
            "--agent-dir", $resolvedAgent,
            "--stress-dir", $resolvedStress,
            "--communication-dir", $resolvedCommunication,
            "--source-root", $resolvedFrozenSource,
            "--output", $enhancedVerification
        ) `
        -LogPath (Join-Path $RunDir "verify_enhanced_outputs.log")

    $deadlineReport = Get-Content -Raw -LiteralPath (
        Join-Path $deadlineVerification "verification_report.json"
    ) | ConvertFrom-Json
    $enhancedReport = Get-Content -Raw -LiteralPath (
        Join-Path $enhancedVerification "verification_report.json"
    ) | ConvertFrom-Json
    if ($deadlineReport.overall -ne "PASS" -or $enhancedReport.overall -ne "PASS") {
        throw "At least one frozen-output verifier did not report PASS."
    }

    @(
        "# Frozen-output verification index",
        "",
        "- Overall status: **PASS**",
        ('- Completed: `{0}`' -f (Get-Date).ToString("o")),
        ('- Primary runner ledger: `{0}`' -f $resolvedPrimary),
        ('- Communication census: `{0}`' -f $resolvedCommunication),
        ('- Independent oracle: `{0}`' -f $resolvedOracle),
        ('- Cache-key ablation: `{0}`' -f $resolvedCache),
        ('- Agent sensitivity: `{0}`' -f $resolvedAgent),
        ('- Same-family stress audit: `{0}`' -f $resolvedStress),
        ('- Paired frozen source tree: `{0}`' -f $resolvedFrozenSource),
        "",
        "The verifiers read saved files and do not execute the simulator. This script does not create or update a release manifest."
    ) | Set-Content -LiteralPath (Join-Path $RunDir "VERIFICATION_INDEX.md") -Encoding UTF8

    Write-Output "Frozen-output verification: PASS"
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
