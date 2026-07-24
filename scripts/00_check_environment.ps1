[CmdletBinding()]
param(
    [string]$PythonExe,
    [string]$TectonicExe,
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

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$runName = "00_check_environment_${stamp}_pid$PID"
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
    $resolvedTectonic = Resolve-Executable `
        -Requested $TectonicExe `
        -EnvironmentVariable "LAYERPROBE_TECTONIC" `
        -Fallback $tectonicFallback

    $requiredFiles = @(
        (Join-Path $RepoRoot "pyproject.toml"),
        (Join-Path $RepoRoot "src\layerprobe\evaluator.py"),
        (Join-Path $RepoRoot "experiments\deadline_runner.py"),
        (Join-Path $RepoRoot "experiments\independent_trace_oracle.py"),
        (Join-Path $RepoRoot "experiments\grid_transfer_audit.py"),
        (Join-Path $RepoRoot "experiments\randomized_mutation_audit.py"),
        (Join-Path $RepoRoot "experiments\scheduler_sensitivity.py"),
        (Join-Path $RepoRoot "experiments\deliberative_policy_benchmark.py"),
        (Join-Path $RepoRoot "experiments\cost_profile.py"),
        (Join-Path $RepoRoot "experiments\build_figure1_layerprobe_method.py"),
        (Join-Path $RepoRoot "experiments\build_review_response_figure.py"),
        (Join-Path $RepoRoot "scripts\run_isolated_formal.ps1")
    )
    foreach ($requiredFile in $requiredFiles) {
        if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) {
            throw "Required project file is missing: $requiredFile"
        }
    }

    $os = Get-CimInstance -ClassName Win32_OperatingSystem
    $processors = @(Get-CimInstance -ClassName Win32_Processor)
    $physicalCores = ($processors | Measure-Object -Property NumberOfCores -Sum).Sum
    $logicalProcessors = ($processors | Measure-Object -Property NumberOfLogicalProcessors -Sum).Sum
    $memoryGiB = [math]::Round(([double]$os.TotalVisibleMemorySize * 1KB / 1GB), 2)
    $repoDrive = (Get-Item -LiteralPath $RepoRoot).PSDrive
    $freeGiB = [math]::Round(([double]$repoDrive.Free / 1GB), 2)
    if ($freeGiB -lt 5) {
        throw "Less than 5 GiB is free on the artifact drive: $freeGiB GiB"
    }

    $fontCandidates = @(
        (Join-Path $env:WINDIR "Fonts\times.ttf"),
        (Join-Path $env:WINDIR "Fonts\timesbd.ttf"),
        (Join-Path $env:WINDIR "Fonts\timesi.ttf"),
        (Join-Path $env:WINDIR "Fonts\timesbi.ttf")
    )
    $presentFonts = @($fontCandidates | Where-Object {
        Test-Path -LiteralPath $_ -PathType Leaf
    })
    if ($presentFonts.Count -ne $fontCandidates.Count) {
        throw "The complete Times New Roman regular/bold/italic/bold-italic family is required."
    }

    $env:PYTHONPATH = "$SrcDir$([IO.Path]::PathSeparator)$RepoRoot"
    $pythonProbe = @'
import platform
import sys
import matplotlib
import numpy
import PIL
import psutil
import pypdf
import pytest
import layerprobe

expected = {
    "python": "3.12.7",
    "numpy": "1.26.4",
    "matplotlib": "3.9.2",
    "Pillow": "10.4.0",
    "pypdf": "6.14.2",
    "psutil": "5.9.0",
    "pytest": "7.4.4",
}
observed = {
    "python": platform.python_version(),
    "numpy": numpy.__version__,
    "matplotlib": matplotlib.__version__,
    "Pillow": PIL.__version__,
    "pypdf": pypdf.__version__,
    "psutil": psutil.__version__,
    "pytest": pytest.__version__,
}
if observed != expected:
    raise SystemExit(
        "reference environment mismatch: "
        + repr({"expected": expected, "observed": observed})
    )

print("python=" + sys.version.replace("\n", " "))
print("executable=" + sys.executable)
print("platform=" + platform.platform())
print("numpy=" + numpy.__version__)
print("matplotlib=" + matplotlib.__version__)
print("Pillow=" + PIL.__version__)
print("pypdf=" + pypdf.__version__)
print("psutil=" + psutil.__version__)
print("pytest=" + pytest.__version__)
print("layerprobe_import=PASS")
'@
    $pythonProbePath = Join-Path $RunDir "environment_probe.py"
    $pythonProbe | Set-Content -LiteralPath $pythonProbePath -Encoding UTF8
    Invoke-LoggedNative `
        -Executable $resolvedPython `
        -ArgumentList @($pythonProbePath) `
        -LogPath (Join-Path $RunDir "python_environment.log")
    Invoke-LoggedNative `
        -Executable $resolvedTectonic `
        -ArgumentList @("--version") `
        -LogPath (Join-Path $RunDir "tectonic_environment.log")
    $tectonicVersionText = (
        Get-Content -LiteralPath (Join-Path $RunDir "tectonic_environment.log") -Raw
    ).Trim()
    if ($tectonicVersionText -ne "Tectonic 0.16.9") {
        throw "Reference environment requires Tectonic 0.16.9; found: $tectonicVersionText"
    }

    $snapshot = @(
        "status=PASS",
        "checked_at=$((Get-Date).ToString('o'))",
        "repo_root=$RepoRoot",
        "os=$($os.Caption) $($os.Version)",
        "cpu=$((($processors | ForEach-Object { $_.Name }) -join '; '))",
        "physical_cores=$physicalCores",
        "logical_processors=$logicalProcessors",
        "memory_gib=$memoryGiB",
        "artifact_drive_free_gib=$freeGiB",
        "python=$resolvedPython",
        "tectonic=$resolvedTectonic",
        "times_new_roman_files=$($presentFonts.Count)"
    )
    $snapshot | Set-Content -LiteralPath (Join-Path $RunDir "environment_snapshot.txt") -Encoding UTF8

    Write-Output "Environment check: PASS"
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
