[CmdletBinding()]
param(
    [string]$PythonExe = "python",

    [string]$ResultTag = "",

    [switch]$SkipMethodLadder,

    [string]$AcceptedMethodResult = "",

    [switch]$SkipParallelScaling,

    [string]$AcceptedScalingResult = "",

    [ValidateRange(1, 60)]
    [int]$QueuePollSeconds = 5,

    [ValidateRange(1, 1440)]
    [int]$MaximumWaitMinutes = 240
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path -LiteralPath (Join-Path $ScriptDir "..")).Path
$Wrapper = Join-Path $ScriptDir "run_isolated_formal.ps1"
$QueueRoot = Join-Path $RepoRoot "artifact_runs\formal_queue"
$stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssfffZ")
$resolvedResultTag = if ([string]::IsNullOrWhiteSpace($ResultTag)) {
    "reproduction_$stamp"
} else {
    $ResultTag
}
if ($resolvedResultTag -notmatch "^[A-Za-z0-9][A-Za-z0-9_.-]*$") {
    throw "ResultTag must contain only letters, digits, dot, underscore, and hyphen."
}
$methodResult = "results\review_method_ladder_$resolvedResultTag"
$scalingResult = "results\parallel_scaling_$resolvedResultTag"
$schedulerResult = "results\scheduler_sensitivity_$resolvedResultTag"
$deliberativeResult = "results\deliberative_policy_$resolvedResultTag"
$profileResult = "results\cost_profile_full_24624_$resolvedResultTag"
$QueueDir = Join-Path $QueueRoot "singlehost_${stamp}_pid$PID"
$QueueLog = Join-Path $QueueDir "queue.log"
$CompletedOutputs = [ordered]@{}

function Write-QueueLog {
    param([Parameter(Mandatory = $true)][string]$Message)

    $line = "{0} {1}" -f (
        (Get-Date).ToUniversalTime().ToString("o")
    ), $Message
    Add-Content -LiteralPath $QueueLog -Value $line -Encoding UTF8
}

function Get-ForeignPython {
    return @(
        Get-CimInstance -ClassName Win32_Process |
            Where-Object {
                $isPython = ([string]$_.Name) -imatch (
                    '^pythonw?(?:\d+(?:\.\d+)*)?\.exe$'
                )
                $isPython
            }
    )
}

function Wait-ForPythonFreeInstant {
    $deadline = (Get-Date).AddMinutes($MaximumWaitMinutes)
    while ((Get-Date) -lt $deadline) {
        $foreign = @(Get-ForeignPython)
        if ($foreign.Count -eq 0) {
            Write-QueueLog "No Python-family process detected; handing control to the 30-second isolation gate."
            return
        }
        $ids = ($foreign | ForEach-Object { [string]$_.ProcessId }) -join ","
        Write-QueueLog "Waiting for foreign Python PIDs: $ids"
        Start-Sleep -Seconds $QueuePollSeconds
    }
    throw "Timed out waiting for a Python-free launch opportunity."
}

function Invoke-Formal {
    param(
        [Parameter(Mandatory = $true)][string]$ScriptPath,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$OutputLabel,
        [string]$ExpectedOutputDir,
        [string[]]$InputFiles = @(),
        [int]$StableSeconds = 30
    )

    $currentArguments = @($Arguments)
    $currentExpectedOutput = $ExpectedOutputDir
    $contaminationAttempt = 0
    while ($true) {
        if (
            -not [string]::IsNullOrWhiteSpace($currentExpectedOutput) -and
            (Test-Path -LiteralPath (
                Join-Path $RepoRoot $currentExpectedOutput
            ))
        ) {
            throw "Expected formal output already exists: $currentExpectedOutput"
        }
        Wait-ForPythonFreeInstant
        $wrapperArgs = @{
            PythonExe = $PythonExe
            ScriptPath = $ScriptPath
            ArgumentList = $currentArguments
            OutputLabel = $OutputLabel
            InputFiles = $InputFiles
            PreflightStableSeconds = $StableSeconds
            PollSeconds = 1
        }
        if (-not [string]::IsNullOrWhiteSpace($currentExpectedOutput)) {
            $wrapperArgs["ExpectedOutputDir"] = $currentExpectedOutput
        }
        Write-QueueLog "Starting isolated target: $OutputLabel"
        & $Wrapper @wrapperArgs
        $exitCode = $LASTEXITCODE
        Write-QueueLog "Isolation wrapper exit for ${OutputLabel}: $exitCode"
        if ($exitCode -eq 20) {
            continue
        }
        if ($exitCode -eq 21) {
            if ([string]::IsNullOrWhiteSpace($ExpectedOutputDir)) {
                Write-QueueLog (
                    "Contaminated target has no output directory; retrying " +
                    "the same command."
                )
                continue
            }
            $contaminationAttempt += 1
            do {
                $retryStamp = (
                    Get-Date
                ).ToUniversalTime().ToString("yyyyMMddTHHmmssfffZ")
                $retryNonce = [Guid]::NewGuid().ToString("N").Substring(0, 8)
                $currentExpectedOutput = (
                    "${ExpectedOutputDir}_retry${contaminationAttempt}_" +
                    "${retryStamp}_${retryNonce}"
                )
            } while (
                Test-Path -LiteralPath (
                    Join-Path $RepoRoot $currentExpectedOutput
                )
            )
            $outputArgumentIndex = [Array]::IndexOf(
                $currentArguments,
                "--output"
            )
            if (
                $outputArgumentIndex -lt 0 -or
                $outputArgumentIndex + 1 -ge $currentArguments.Count
            ) {
                throw (
                    "Cannot retarget contaminated output for ${OutputLabel}: " +
                    "--output argument is missing."
                )
            }
            $currentArguments[$outputArgumentIndex + 1] = (
                $currentExpectedOutput
            )
            Write-QueueLog (
                "Contaminated output retained as inadmissible evidence; " +
                "retrying $OutputLabel at $currentExpectedOutput."
            )
            continue
        }
        if ($exitCode -eq 22) {
            throw (
                "Formal target $OutputLabel observed a source/input change; " +
                "its output is inadmissible and the queue must be restarted " +
                "from a stable source tree."
            )
        }
        if ($exitCode -ne 0) {
            throw "Formal target $OutputLabel failed or was contaminated (exit $exitCode)."
        }
        if (-not [string]::IsNullOrWhiteSpace($currentExpectedOutput)) {
            $script:CompletedOutputs[$OutputLabel] = $currentExpectedOutput
        }
        return
    }
}

try {
    if (-not (Test-Path -LiteralPath $QueueRoot -PathType Container)) {
        New-Item -ItemType Directory -Path $QueueRoot | Out-Null
    }
    New-Item -ItemType Directory -Path $QueueDir | Out-Null
    Write-QueueLog "Formal single-host queue created."

    $formalControlInputs = @(
        "scripts\run_isolated_formal.ps1",
        "scripts\queue_formal_singlehost.ps1"
    )

    Invoke-Formal `
        -ScriptPath "experiments\deadline_runner.py" `
        -Arguments @("--help") `
        -OutputLabel "wrapper_runtime_smoke" `
        -InputFiles $formalControlInputs `
        -StableSeconds 5

    $commonInputs = @(
        "src\layerprobe\__init__.py",
        "src\layerprobe\cli.py",
        "src\layerprobe\evaluator.py",
        "src\layerprobe\mechanics.py",
        "src\layerprobe\model.py",
        "src\layerprobe\workloads.py",
        "scripts\run_isolated_formal.ps1",
        "scripts\queue_formal_singlehost.ps1"
    )

    if ($SkipMethodLadder) {
        if ([string]::IsNullOrWhiteSpace($AcceptedMethodResult)) {
            throw (
                "AcceptedMethodResult is required when SkipMethodLadder " +
                "is set."
            )
        }
        $acceptedMethodPath = Join-Path $RepoRoot $AcceptedMethodResult
        if (-not (Test-Path -LiteralPath $acceptedMethodPath -PathType Container)) {
            throw "Accepted method result does not exist: $AcceptedMethodResult"
        }
        $CompletedOutputs["method_ladder_n10"] = $AcceptedMethodResult
        Write-QueueLog (
            "Reusing completed method ladder after wrapper-only recovery: " +
            $AcceptedMethodResult
        )
    } else {
        Invoke-Formal `
            -ScriptPath "experiments\deadline_runner.py" `
            -Arguments @(
                "--config", "experiments\deadline_profile_review_8c32g.json",
                "--mode", "paper",
                "--only", "method_ladder",
                "--output", $methodResult,
                "--ignore-freeze"
            ) `
            -OutputLabel "method_ladder_n10" `
            -ExpectedOutputDir $methodResult `
            -InputFiles (
                @("experiments\deadline_profile_review_8c32g.json") +
                $commonInputs
            )
    }

    if ($SkipParallelScaling) {
        if ([string]::IsNullOrWhiteSpace($AcceptedScalingResult)) {
            throw (
                "AcceptedScalingResult is required when " +
                "SkipParallelScaling is set."
            )
        }
        $acceptedScalingPath = Join-Path $RepoRoot $AcceptedScalingResult
        if (-not (Test-Path -LiteralPath $acceptedScalingPath -PathType Container)) {
            throw "Accepted scaling result does not exist: $AcceptedScalingResult"
        }
        $CompletedOutputs["parallel_scaling_n10"] = $AcceptedScalingResult
        Write-QueueLog (
            "Reusing completed parallel scaling after wrapper-only recovery: " +
            $AcceptedScalingResult
        )
    } else {
        Invoke-Formal `
            -ScriptPath "experiments\deadline_runner.py" `
            -Arguments @(
                "--config", "experiments\deadline_profile_review_8c32g.json",
                "--mode", "paper",
                "--only", "parallel_scaling",
                "--output", $scalingResult,
                "--ignore-freeze"
            ) `
            -OutputLabel "parallel_scaling_n10" `
            -ExpectedOutputDir $scalingResult `
            -InputFiles (
                @("experiments\deadline_profile_review_8c32g.json") +
                $commonInputs
            )
    }

    Invoke-Formal `
        -ScriptPath "experiments\scheduler_sensitivity.py" `
        -Arguments @(
            "--output", $schedulerResult,
            "--workers", "8",
            "--repeats", "10",
            "--kernels", "24624",
            "--presentations", "18"
        ) `
        -OutputLabel "scheduler_sensitivity_n10" `
        -ExpectedOutputDir $schedulerResult `
        -InputFiles $commonInputs

    Invoke-Formal `
        -ScriptPath "experiments\deliberative_policy_benchmark.py" `
        -Arguments @(
            "--output", $deliberativeResult,
            "--workers", "8",
            "--repeats", "10",
            "--depths", "0", "2", "4", "6",
            "--population-kernels", "24624",
            "--sample-size", "512",
            "--presentations", "18",
            "--bootstrap-samples", "10000"
        ) `
        -OutputLabel "deliberative_policy_n10" `
        -ExpectedOutputDir $deliberativeResult `
        -InputFiles $commonInputs

    Invoke-Formal `
        -ScriptPath "experiments\cost_profile.py" `
        -Arguments @(
            "--output", $profileResult,
            "--kernels", "24624",
            "--presentations", "18"
        ) `
        -OutputLabel "cost_profile_full_24624" `
        -ExpectedOutputDir $profileResult `
        -InputFiles (
            @("pyproject.toml") +
            $commonInputs
        )

    if ($CompletedOutputs.Count -ne 5) {
        throw (
            "Formal queue reached completion with " +
            "$($CompletedOutputs.Count) recorded outputs; expected 5."
        )
    }
    Write-QueueLog "FORMAL_QUEUE_COMPLETE"
    [ordered]@{
        status = "complete"
        completed_at = (Get-Date).ToUniversalTime().ToString("o")
        result_tag = $resolvedResultTag
        output_count = $CompletedOutputs.Count
        outputs = $CompletedOutputs
    } | ConvertTo-Json | Set-Content `
        -LiteralPath (Join-Path $QueueDir "COMPLETE.json") `
        -Encoding UTF8
    exit 0
} catch {
    Write-QueueLog ("FORMAL_QUEUE_FAILED: " + $_.Exception.Message)
    [ordered]@{
        status = "failed"
        failed_at = (Get-Date).ToUniversalTime().ToString("o")
        error = $_.Exception.Message
    } | ConvertTo-Json | Set-Content `
        -LiteralPath (Join-Path $QueueDir "FAILED.json") `
        -Encoding UTF8
    exit 1
}
