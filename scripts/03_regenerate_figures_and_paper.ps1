[CmdletBinding()]
param(
    [string]$PythonExe,
    [string]$TectonicExe,
    [string]$PaperSourceDir,
    [string]$PrimaryRunDir,
    [string]$CommunicationDir,
    [string]$MethodLadderDir,
    [string]$DeliberativeDir,
    [string]$SchedulerDir,
    [string]$StyleReference,
    [string]$OutputDir,
    [switch]$FiguresOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path -LiteralPath (Join-Path $ScriptDir "..")).Path
$SrcDir = Join-Path $RepoRoot "src"
$script:RecoveryCertificatesVerified = $false

function Get-PortableRelativePath {
    param(
        [Parameter(Mandatory = $true)][string]$BasePath,
        [Parameter(Mandatory = $true)][string]$TargetPath
    )

    $baseFull = [IO.Path]::GetFullPath($BasePath)
    if (
        -not $baseFull.EndsWith([IO.Path]::DirectorySeparatorChar) -and
        -not $baseFull.EndsWith([IO.Path]::AltDirectorySeparatorChar)
    ) {
        $baseFull += [IO.Path]::DirectorySeparatorChar
    }
    $baseUri = [Uri]::new($baseFull)
    $targetUri = [Uri]::new([IO.Path]::GetFullPath($TargetPath))
    if ($baseUri.Scheme -ne $targetUri.Scheme) {
        throw "Cannot make a relative path across URI schemes."
    }
    $relative = [Uri]::UnescapeDataString(
        $baseUri.MakeRelativeUri($targetUri).ToString()
    ).Replace(
        [IO.Path]::AltDirectorySeparatorChar,
        [IO.Path]::DirectorySeparatorChar
    )
    if ([string]::IsNullOrEmpty($relative)) {
        return "."
    }
    return $relative
}

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

function Resolve-Executable {
    param([string]$Requested, [string]$EnvironmentVariable, [string]$Fallback)
    $candidate = $Requested
    if ([string]::IsNullOrWhiteSpace($candidate)) {
        $candidate = [Environment]::GetEnvironmentVariable($EnvironmentVariable)
    }
    if ([string]::IsNullOrWhiteSpace($candidate)) {
        $candidate = $Fallback
    }
    if (Test-Path -LiteralPath $candidate -PathType Leaf) {
        return (Resolve-Path -LiteralPath $candidate).Path
    }
    $command = Get-Command -Name $candidate -CommandType Application -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($null -eq $command) {
        throw "Required executable was not found: $candidate"
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

function Read-JsonUtf8 {
    param([Parameter(Mandatory = $true)][string]$Path)
    return (
        [IO.File]::ReadAllText($Path, [Text.Encoding]::UTF8) |
            ConvertFrom-Json
    )
}

function Get-ResultTreeSha256 {
    param([Parameter(Mandatory = $true)][string]$Directory)

    $entries = @(
        Get-ChildItem -LiteralPath $Directory -File -Recurse |
            Where-Object {
                $_.Name -ne "FORMAL_PROVENANCE.json" -and
                $_.Name -ne "RECOVERY_OVERRIDE.json" -and
                $_.Name -notlike "DO_NOT_USE_*.json"
            } |
            Sort-Object -Property FullName |
            ForEach-Object {
                $relative = (
                    Get-PortableRelativePath `
                        -BasePath $Directory `
                        -TargetPath $_.FullName
                ).Replace([char]0x5C, [char]0x2F)
                $hash = (
                    Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256
                ).Hash.ToLowerInvariant()
                "$hash $([long]$_.Length) $relative"
            }
    )
    if ($entries.Count -eq 0) {
        throw "Formal result directory is empty: $Directory"
    }
    $sha = [Security.Cryptography.SHA256]::Create()
    try {
        $bytes = $sha.ComputeHash(
            [Text.Encoding]::UTF8.GetBytes(($entries -join "`n"))
        )
    } finally {
        $sha.Dispose()
    }
    return [BitConverter]::ToString(
        $bytes
    ).Replace("-", "").ToLowerInvariant()
}

function Assert-AdmissibleFormalResult {
    param(
        [Parameter(Mandatory = $true)][string]$Directory,
        [Parameter(Mandatory = $true)][string]$ExpectedLabel
    )

    foreach ($required in @("runs.csv", "summary.json", "FORMAL_PROVENANCE.json")) {
        if (-not (Test-Path -LiteralPath (Join-Path $Directory $required) -PathType Leaf)) {
            throw "Formal result lacks ${required}: $Directory"
        }
    }
    $sidecarPath = Join-Path $Directory "FORMAL_PROVENANCE.json"
    $sidecar = Read-JsonUtf8 -Path $sidecarPath
    if ([string]$sidecar.status -eq "recovered_formal_result") {
        $recoveryScript = Join-Path $ScriptDir (
            "recover_completed_formal_results.ps1"
        )
        if (-not (Test-Path -LiteralPath $recoveryScript -PathType Leaf)) {
            throw "Recovery verifier is missing: $recoveryScript"
        }
        if (-not $script:RecoveryCertificatesVerified) {
            $packagedRecoveryVerifier = Join-Path $ScriptDir (
                "verify_recovery_evidence.ps1"
            )
            if (Test-Path -LiteralPath $packagedRecoveryVerifier -PathType Leaf) {
                & $packagedRecoveryVerifier | Out-Host
            }
            else {
                & $recoveryScript -VerifyOnly | Out-Host
            }
            $script:RecoveryCertificatesVerified = $true
        }
        $expectedRelative = (
            Get-PortableRelativePath `
                -BasePath $RepoRoot `
                -TargetPath $Directory
        ).Replace([char]0x5C, [char]0x2F)
        $expectedClass = switch ($ExpectedLabel) {
            "method_ladder_n10" {
                "wrapper_tail_monitor_interruption_after_target_success"
            }
            "parallel_scaling_n10" {
                "wrapper_only_post_success_sidecar_failure"
            }
            default {
                throw (
                    "No recovery class is authorized for formal label: " +
                    $ExpectedLabel
                )
            }
        }
        if (
            [int]$sidecar.schema_version -ne 2 -or
            [string]$sidecar.provenance_mode -ne "recovery_override" -or
            [string]$sidecar.authorization -ne
                "explicit_local_wrapper_recovery_2026-07-25" -or
            [string]$sidecar.output_label -ne $ExpectedLabel -or
            [string]$sidecar.expected_output_dir -ne $expectedRelative -or
            [string]$sidecar.result_tree_relative_to_repo -ne
                $expectedRelative -or
            [string]$sidecar.recovery_class -ne $expectedClass
        ) {
            throw "Recovered formal provenance gate failed: $sidecarPath"
        }
        $recoveryPath = [IO.Path]::GetFullPath(
            (Join-Path $RepoRoot $sidecar.recovery_record_relative_to_repo)
        )
        if (
            -not $recoveryPath.Equals(
                (Join-Path $Directory "RECOVERY_OVERRIDE.json"),
                [StringComparison]::OrdinalIgnoreCase
            ) -or
            -not (Test-Path -LiteralPath $recoveryPath -PathType Leaf)
        ) {
            throw "Recovery override path is unavailable or mismatched."
        }
        $recoveryHash = (
            Get-FileHash -LiteralPath $recoveryPath -Algorithm SHA256
        ).Hash.ToLowerInvariant()
        if ($recoveryHash -ne [string]$sidecar.recovery_record_sha256) {
            throw "Recovery override hash mismatch: $recoveryPath"
        }
        $recovery = Read-JsonUtf8 -Path $recoveryPath
        if (
            [int]$recovery.schema_version -ne 2 -or
            [string]$recovery.status -ne "recovery_override" -or
            [string]$recovery.output_label -ne $ExpectedLabel -or
            [string]$recovery.recovery_class -ne $expectedClass -or
            -not [bool]$recovery.claim_boundary.target_completed -or
            -not [bool]$recovery.claim_boundary.raw_rows_and_semantics_verified
        ) {
            throw "Recovery override content is not authorized."
        }
        $treeHash = Get-ResultTreeSha256 -Directory $Directory
        if (
            $treeHash -ne [string]$sidecar.result_tree_sha256 -or
            $treeHash -ne [string]$recovery.result_tree.tree_sha256
        ) {
            throw "Recovered formal result-tree hash mismatch: $Directory"
        }
        return
    }

    $forbidden = @(
        Get-ChildItem -LiteralPath $Directory -File -Filter "DO_NOT_USE_*.json"
    )
    if ($forbidden.Count -gt 0) {
        throw "Formal result has a DO_NOT_USE marker: $Directory"
    }
    if (
        $sidecar.status -ne "admissible_formal_result" -or
        [int]$sidecar.runner_exit_code -ne 0 -or
        [int]$sidecar.foreign_python_count -ne 0 -or
        [int]$sidecar.ignored_control_probe_count -ne 0 -or
        [int]$sidecar.preflight_stable_seconds -lt 30 -or
        -not ($sidecar.source_inputs_unchanged -is [bool]) -or
        -not [bool]$sidecar.source_inputs_unchanged -or
        $sidecar.output_label -ne $ExpectedLabel
    ) {
        throw "Formal provenance gate failed: $sidecarPath"
    }
    $cleanPath = [IO.Path]::GetFullPath(
        (Join-Path $RepoRoot $sidecar.clean_record_relative_to_repo)
    )
    if (-not (Test-Path -LiteralPath $cleanPath -PathType Leaf)) {
        throw "Formal CLEAN_RUN record is missing: $cleanPath"
    }
    $cleanHash = (
        Get-FileHash -LiteralPath $cleanPath -Algorithm SHA256
    ).Hash.ToLowerInvariant()
    if ($cleanHash -ne [string]$sidecar.clean_record_sha256) {
        throw "Formal CLEAN_RUN hash mismatch: $cleanPath"
    }
    $clean = Read-JsonUtf8 -Path $cleanPath
    if (
        $clean.status -ne "clean_run" -or
        [int]$clean.runner_exit_code -ne 0 -or
        [int]$clean.foreign_python_count -ne 0 -or
        [int]$clean.ignored_control_probe_count -ne 0 -or
        [int]$clean.preflight_stable_seconds -lt 30 -or
        -not [bool]$clean.source_input_integrity.complete -or
        -not [bool]$clean.source_input_integrity.unchanged -or
        [string]$clean.output_label -ne $ExpectedLabel -or
        [string]$sidecar.output_label -ne [string]$clean.output_label
    ) {
        throw "Referenced isolation record is not an admissible CLEAN_RUN: $cleanPath"
    }
    $treeHash = Get-ResultTreeSha256 -Directory $Directory
    if (
        $treeHash -ne [string]$sidecar.result_tree_sha256 -or
        $treeHash -ne [string]$clean.result_tree.tree_sha256
    ) {
        throw "Formal result-tree hash mismatch: $Directory"
    }
}

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$defaultOutput = "artifact_runs\03_regenerate_figures_and_paper_${stamp}_pid$PID"
$RunDir = Resolve-RepoPath -Requested $OutputDir -DefaultRelative $defaultOutput
$defaultPaperSource = if (
    Test-Path -LiteralPath (Join-Path $RepoRoot "paper") -PathType Container
) {
    "paper"
} else {
    "..\..\07_论文\manuscript\draft"
}
$resolvedPaperSource = if ($FiguresOnly) {
    $null
}
else {
    Resolve-RepoPath `
        -Requested $PaperSourceDir `
        -DefaultRelative $defaultPaperSource
}
$resolvedPrimary = Resolve-RepoPath `
    -Requested $PrimaryRunDir `
    -DefaultRelative "results\parallel_scaling_n10_clean_20260724"
$resolvedCommunication = Resolve-RepoPath `
    -Requested $CommunicationDir `
    -DefaultRelative "results\communication_full_24624_distancefix_provenance_v2_20260723_xeon"
$resolvedMethodLadder = Resolve-RepoPath `
    -Requested $MethodLadderDir `
    -DefaultRelative "results\review_method_ladder_n10_clean_20260724"
$resolvedDeliberative = Resolve-RepoPath `
    -Requested $DeliberativeDir `
    -DefaultRelative "results\deliberative_policy_n10_clean_20260724"
$resolvedScheduler = Resolve-RepoPath `
    -Requested $SchedulerDir `
    -DefaultRelative "results\scheduler_sensitivity_n10_clean_20260724"
$resolvedStyleReference = if ([string]::IsNullOrWhiteSpace($StyleReference)) {
    $null
} else {
    Resolve-RepoPath -Requested $StyleReference -DefaultRelative ""
}

$transcriptStarted = $false
$exitCode = 1

try {
    if (Test-Path -LiteralPath $RunDir) {
        throw "Refusing to overwrite an existing output directory: $RunDir"
    }
    $inputDirectories = @(
        $resolvedPrimary,
        $resolvedCommunication,
        $resolvedMethodLadder,
        $resolvedDeliberative,
        $resolvedScheduler
    )
    if (-not $FiguresOnly) {
        $inputDirectories = @($resolvedPaperSource) + $inputDirectories
    }
    foreach ($inputDirectory in $inputDirectories) {
        if (-not (Test-Path -LiteralPath $inputDirectory -PathType Container)) {
            throw "Required input directory is missing: $inputDirectory"
        }
    }
    Assert-AdmissibleFormalResult `
        -Directory $resolvedMethodLadder `
        -ExpectedLabel "method_ladder_n10"
    Assert-AdmissibleFormalResult `
        -Directory $resolvedPrimary `
        -ExpectedLabel "parallel_scaling_n10"
    Assert-AdmissibleFormalResult `
        -Directory $resolvedDeliberative `
        -ExpectedLabel "deliberative_policy_n10"
    Assert-AdmissibleFormalResult `
        -Directory $resolvedScheduler `
        -ExpectedLabel "scheduler_sensitivity_n10"
    if (
        $null -ne $resolvedStyleReference -and
        -not (Test-Path -LiteralPath $resolvedStyleReference -PathType Leaf)
    ) {
        throw "Optional style reference is missing: $resolvedStyleReference"
    }

    $paperInputs = @("paper.tex", "references.bib", "llncs.cls", "splncs04.bst")
    if (-not $FiguresOnly) {
        foreach ($paperInput in $paperInputs) {
            $sourcePath = Join-Path $resolvedPaperSource $paperInput
            if (-not (Test-Path -LiteralPath $sourcePath -PathType Leaf)) {
                throw "Required paper source is missing: $sourcePath"
            }
        }
    }

    New-Item -ItemType Directory -Path $RunDir | Out-Null
    $logsDir = Join-Path $RunDir "logs"
    $stagedPaper = if ($FiguresOnly) { $null } else { Join-Path $RunDir "paper" }
    $figuresDir = if ($FiguresOnly) {
        Join-Path $RunDir "figures"
    }
    else {
        Join-Path $stagedPaper "figures"
    }
    New-Item -ItemType Directory -Path $logsDir | Out-Null
    New-Item -ItemType Directory -Path $figuresDir | Out-Null
    Start-Transcript -LiteralPath (Join-Path $RunDir "run.log") -NoClobber | Out-Null
    $transcriptStarted = $true

    $resolvedPython = Resolve-Executable `
        -Requested $PythonExe `
        -EnvironmentVariable "LAYERPROBE_PYTHON" `
        -Fallback "python"
    $localTectonic = [IO.Path]::GetFullPath(
        (Join-Path $RepoRoot "..\..\..\_tools\tectonic-0.16.9\unpacked\tectonic.exe")
    )
    $tectonicFallback = if (Test-Path -LiteralPath $localTectonic -PathType Leaf) {
        $localTectonic
    } else {
        "tectonic"
    }
    $resolvedTectonic = if ($FiguresOnly) {
        $null
    }
    else {
        Resolve-Executable `
            -Requested $TectonicExe `
            -EnvironmentVariable "LAYERPROBE_TECTONIC" `
            -Fallback $tectonicFallback
    }

    if (-not $FiguresOnly) {
        foreach ($paperInput in $paperInputs) {
            Copy-Item `
                -LiteralPath (Join-Path $resolvedPaperSource $paperInput) `
                -Destination (Join-Path $stagedPaper $paperInput)
        }
    }

    $env:PYTHONPATH = "$SrcDir$([IO.Path]::PathSeparator)$RepoRoot"
    $env:PYTHONHASHSEED = "0"
    $env:PYTHONDONTWRITEBYTECODE = "1"

    $figure1Arguments = @(
        (Join-Path $RepoRoot "experiments\build_figure1_layerprobe_method.py"),
        "--output-dir", $figuresDir,
        "--stem", "fig0_layerprobe_overview"
    )
    if ($null -ne $resolvedStyleReference) {
        $figure1Arguments += @("--style-reference", $resolvedStyleReference)
    }
    Invoke-LoggedNative `
        -Executable $resolvedPython `
        -ArgumentList $figure1Arguments `
        -LogPath (Join-Path $logsDir "figure1.log")

    Invoke-LoggedNative `
        -Executable $resolvedPython `
        -ArgumentList @(
            (Join-Path $RepoRoot "experiments\build_review_response_figure.py"),
            "--method-runs", (Join-Path $resolvedMethodLadder "runs.csv"),
            "--method-summary", (Join-Path $resolvedMethodLadder "summary.json"),
            "--deliberative-runs", (Join-Path $resolvedDeliberative "runs.csv"),
            "--deliberative-summary", (Join-Path $resolvedDeliberative "summary.json"),
            "--scaling-runs", (Join-Path $resolvedPrimary "runs.csv"),
            "--scaling-summary", (Join-Path $resolvedPrimary "summary.json"),
            "--scheduler-runs", (Join-Path $resolvedScheduler "runs.csv"),
            "--scheduler-summary", (Join-Path $resolvedScheduler "summary.json"),
            "--output-dir", $figuresDir,
            "--stem", "fig1_combined_results",
            "--required-new-repeats", "10"
        ) `
        -LogPath (Join-Path $logsDir "review_response_figure.log")

    Invoke-LoggedNative `
        -Executable $resolvedPython `
        -ArgumentList @(
            (Join-Path $RepoRoot "experiments\build_figure2_semantic_evidence.py"),
            "--output-dir", $figuresDir,
            "--visual-review-pass"
        ) `
        -LogPath (Join-Path $logsDir "figure2_semantic_evidence.log")

    Invoke-LoggedNative `
        -Executable $resolvedPython `
        -ArgumentList @(
            (Join-Path $RepoRoot "experiments\build_figure4_allview_evidence.py"),
            "--output-dir", $figuresDir,
            "--visual-review-status", "PASS"
        ) `
        -LogPath (Join-Path $logsDir "figure4_allview_evidence.log")

    $paperPdf = $null
    if (-not $FiguresOnly) {
        Push-Location $stagedPaper
        try {
            Invoke-LoggedNative `
                -Executable $resolvedTectonic `
                -ArgumentList @("paper.tex", "--keep-logs", "--keep-intermediates") `
                -LogPath (Join-Path $logsDir "tectonic.log")
        }
        finally {
            Pop-Location
        }

        $paperPdf = Join-Path $stagedPaper "paper.pdf"
        $paperLog = Join-Path $stagedPaper "paper.log"
        if (
            -not (Test-Path -LiteralPath $paperPdf -PathType Leaf) -or
            (Get-Item -LiteralPath $paperPdf).Length -le 0
        ) {
            throw "Compiled paper PDF is missing or empty: $paperPdf"
        }
        if (-not (Test-Path -LiteralPath $paperLog -PathType Leaf)) {
            throw "Tectonic did not leave paper.log for QA."
        }

        $fatalLogPatterns = @(
            "LaTeX Error",
            "Emergency stop",
            "Undefined control sequence",
            "There were undefined",
            "Citation .* undefined",
            "Reference .* undefined",
            "Overfull \\[hv]box",
            "Missing character"
        )
        $fatalMatches = @(
            Select-String -LiteralPath $paperLog -Pattern $fatalLogPatterns -CaseSensitive:$false
        )
        if ($fatalMatches.Count -ne 0) {
            $fatalMatches | ForEach-Object { Write-Error $_.Line }
            throw "Paper log contains a fatal publication-QA pattern."
        }

        $pageCheck = @'
import sys
from pypdf import PdfReader

path = sys.argv[1]
pages = len(PdfReader(path).pages)
print(f"paper_pages={pages}")
if pages > 20:
    raise SystemExit(f"paper exceeds the 20-page regular-paper limit: {pages}")
'@
        $pageCheckPath = Join-Path $logsDir "paper_page_check.py"
        $pageCheck | Set-Content -LiteralPath $pageCheckPath -Encoding UTF8
        Invoke-LoggedNative `
            -Executable $resolvedPython `
            -ArgumentList @($pageCheckPath, $paperPdf) `
            -LogPath (Join-Path $logsDir "paper_page_check.log")
    }

    $expectedFigures = @(
        "fig0_layerprobe_overview.pdf",
        "fig1_combined_results.pdf",
        "fig2_semantic_evidence.pdf",
        "fig4_allview_evidence.pdf"
    )
    foreach ($expectedFigure in $expectedFigures) {
        $figurePath = Join-Path $figuresDir $expectedFigure
        if (
            -not (Test-Path -LiteralPath $figurePath -PathType Leaf) -or
            (Get-Item -LiteralPath $figurePath).Length -le 0
        ) {
            throw "Expected generated figure is missing or empty: $figurePath"
        }
    }

    @(
        "# Figure and paper regeneration result",
        "",
        "- Status: **PASS**",
        ('- Completed: `{0}`' -f (Get-Date).ToString("o")),
        ('- Staged paper: `{0}`' -f $(if ($FiguresOnly) { "not included (FiguresOnly)" } else { $paperPdf })),
        "- Original paper source was not modified.",
        "- Figure 1 was generated without an external image by default.",
        "- All four active vector figures were regenerated.",
        $(if ($FiguresOnly) {
            "- Paper compilation was intentionally skipped."
        } else {
            "- Tectonic compilation and the 20-page limit gate passed."
        }),
        "",
        "This run directory is a disposable reproduction product, not a release manifest."
    ) | Set-Content -LiteralPath (Join-Path $RunDir "REGENERATION_RESULT.md") -Encoding UTF8

    Write-Output "Final four-figure regeneration: PASS"
    if (-not $FiguresOnly) {
        Write-Output "Paper: $paperPdf"
    }
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
