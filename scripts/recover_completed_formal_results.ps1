[CmdletBinding()]
param(
    [switch]$VerifyOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RecoveryScriptPath = [IO.Path]::GetFullPath($MyInvocation.MyCommand.Path)
$RepoRoot = (Resolve-Path -LiteralPath (Join-Path $ScriptDir "..")).Path
$Utf8NoBom = [Text.UTF8Encoding]::new($false)

function Get-PortableRelativePath {
    param(
        [Parameter(Mandatory = $true)][string]$BasePath,
        [Parameter(Mandatory = $true)][string]$TargetPath
    )

    $baseFull = [IO.Path]::GetFullPath($BasePath).TrimEnd(
        [IO.Path]::DirectorySeparatorChar,
        [IO.Path]::AltDirectorySeparatorChar
    ) + [IO.Path]::DirectorySeparatorChar
    $targetFull = [IO.Path]::GetFullPath($TargetPath)
    $baseUri = [Uri]::new($baseFull)
    $targetUri = [Uri]::new($targetFull)
    return [Uri]::UnescapeDataString(
        $baseUri.MakeRelativeUri($targetUri).ToString()
    ).Replace("/", [string][IO.Path]::DirectorySeparatorChar)
}

function Convert-ToManifestPath {
    param([Parameter(Mandatory = $true)][string]$Path)
    return $Path.Replace(
        [string][IO.Path]::DirectorySeparatorChar,
        "/"
    )
}

function Resolve-RepoFile {
    param([Parameter(Mandatory = $true)][string]$RelativePath)

    if (
        [string]::IsNullOrWhiteSpace($RelativePath) -or
        [IO.Path]::IsPathRooted($RelativePath) -or
        $RelativePath.Contains(":") -or
        @($RelativePath -split "[\\/]" | Where-Object {
            $_ -eq ".."
        }).Count -gt 0
    ) {
        throw "Unsafe repository-relative path: $RelativePath"
    }
    $candidate = [IO.Path]::GetFullPath((Join-Path $RepoRoot $RelativePath))
    $rootPrefix = $RepoRoot.TrimEnd(
        [IO.Path]::DirectorySeparatorChar
    ) + [IO.Path]::DirectorySeparatorChar
    if (-not $candidate.StartsWith(
        $rootPrefix,
        [StringComparison]::OrdinalIgnoreCase
    )) {
        throw "Path escapes the repository: $RelativePath"
    }
    if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) {
        throw "Required evidence file is missing: $candidate"
    }
    return $candidate
}

function Get-Sha256 {
    param([Parameter(Mandatory = $true)][string]$Path)
    return (
        Get-FileHash -LiteralPath $Path -Algorithm SHA256
    ).Hash.ToLowerInvariant()
}

function Read-Json {
    param([Parameter(Mandatory = $true)][string]$Path)
    return (
        [IO.File]::ReadAllText($Path, [Text.Encoding]::UTF8) |
            ConvertFrom-Json
    )
}

function Get-FileRecord {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Role
    )

    $resolved = (Resolve-Path -LiteralPath $Path).Path
    $item = Get-Item -LiteralPath $resolved
    return [ordered]@{
        role = $Role
        path_relative_to_repo = Convert-ToManifestPath -Path (
            Get-PortableRelativePath `
                -BasePath $RepoRoot `
                -TargetPath $resolved
        )
        bytes = [long]$item.Length
        sha256 = Get-Sha256 -Path $resolved
    }
}

function Get-ResultTreeRecord {
    param([Parameter(Mandatory = $true)][string]$Directory)

    $entries = @(
        Get-ChildItem -LiteralPath $Directory -File -Recurse -Force |
            Where-Object {
                $_.Name -ne "FORMAL_PROVENANCE.json" -and
                $_.Name -ne "RECOVERY_OVERRIDE.json" -and
                $_.Name -notlike "DO_NOT_USE_*.json"
            } |
            Sort-Object -Property FullName |
            ForEach-Object {
                [ordered]@{
                    path = Convert-ToManifestPath -Path (
                        Get-PortableRelativePath `
                            -BasePath $Directory `
                            -TargetPath $_.FullName
                    )
                    bytes = [long]$_.Length
                    sha256 = Get-Sha256 -Path $_.FullName
                }
            }
    )
    if ($entries.Count -eq 0) {
        throw "Result payload is empty: $Directory"
    }
    $canonical = @(
        $entries | ForEach-Object {
            "$($_.sha256) $($_.bytes) $($_.path)"
        }
    ) -join "`n"
    $sha = [Security.Cryptography.SHA256]::Create()
    try {
        $digest = $sha.ComputeHash(
            [Text.Encoding]::UTF8.GetBytes($canonical)
        )
    }
    finally {
        $sha.Dispose()
    }
    return [ordered]@{
        path_relative_to_repo = Convert-ToManifestPath -Path (
            Get-PortableRelativePath `
                -BasePath $RepoRoot `
                -TargetPath $Directory
        )
        file_count = $entries.Count
        tree_sha256 = [BitConverter]::ToString(
            $digest
        ).Replace("-", "").ToLowerInvariant()
        files = $entries
    }
}

function Get-DeadlineFingerprint {
    $sourceFiles = @(
        Get-ChildItem -LiteralPath (Join-Path $RepoRoot "src") `
            -File -Recurse -Filter "*.py" |
            Sort-Object -Property FullName
    )
    $sourceFiles += Get-Item -LiteralPath (
        Join-Path $RepoRoot "experiments\deadline_runner.py"
    )
    $sourceFiles += Get-Item -LiteralPath (
        Join-Path $RepoRoot "experiments\deadline_profile_review_8c32g.json"
    )
    $stream = [IO.MemoryStream]::new()
    try {
        foreach ($file in $sourceFiles) {
            $label = Get-PortableRelativePath `
                -BasePath $RepoRoot `
                -TargetPath $file.FullName
            $labelBytes = [Text.Encoding]::UTF8.GetBytes($label)
            $stream.Write($labelBytes, 0, $labelBytes.Length)
            $stream.WriteByte(0)
            $fileBytes = [IO.File]::ReadAllBytes($file.FullName)
            $stream.Write($fileBytes, 0, $fileBytes.Length)
            $stream.WriteByte(0)
        }
        $sha = [Security.Cryptography.SHA256]::Create()
        try {
            $digest = $sha.ComputeHash($stream.ToArray())
        }
        finally {
            $sha.Dispose()
        }
    }
    finally {
        $stream.Dispose()
    }
    return [BitConverter]::ToString(
        $digest
    ).Replace("-", "").ToLowerInvariant()
}

function Assert-ExactValues {
    param(
        [AllowEmptyCollection()][object[]]$Actual,
        [AllowEmptyCollection()][object[]]$Expected,
        [Parameter(Mandatory = $true)][string]$Label
    )

    $actualValues = @($Actual | ForEach-Object { [string]$_ } | Sort-Object)
    $expectedValues = @($Expected | ForEach-Object { [string]$_ } | Sort-Object)
    if (
        $actualValues.Count -ne $expectedValues.Count -or
        ($actualValues -join "`n") -cne ($expectedValues -join "`n")
    ) {
        throw (
            "$Label mismatch. Expected [{0}], observed [{1}]." -f
            ($expectedValues -join ", "),
            ($actualValues -join ", ")
        )
    }
}

function Assert-ExactOrderedValues {
    param(
        [AllowEmptyCollection()][object[]]$Actual,
        [AllowEmptyCollection()][object[]]$Expected,
        [Parameter(Mandatory = $true)][string]$Label
    )

    $actualValues = @($Actual | ForEach-Object { [string]$_ })
    $expectedValues = @($Expected | ForEach-Object { [string]$_ })
    if (
        $actualValues.Count -ne $expectedValues.Count -or
        ($actualValues -join "`n") -cne ($expectedValues -join "`n")
    ) {
        throw "$Label has unexpected ordered values."
    }
}

function Assert-ResultPayload {
    param(
        [Parameter(Mandatory = $true)][hashtable]$Specification
    )

    $directory = Join-Path $RepoRoot $Specification.result_relative
    if (-not (Test-Path -LiteralPath $directory -PathType Container)) {
        throw "Recovered result directory is missing: $directory"
    }
    foreach ($required in @(
        "frozen_config.json",
        "metadata.json",
        "progress.json",
        "runs.csv",
        "semantic_checks.json",
        "summary.json",
        "SUMMARY.md"
    )) {
        if (-not (Test-Path -LiteralPath (
            Join-Path $directory $required
        ) -PathType Leaf)) {
            throw "Recovered result lacks ${required}: $directory"
        }
    }

    $rows = @(Import-Csv -LiteralPath (Join-Path $directory "runs.csv"))
    $progress = Read-Json -Path (Join-Path $directory "progress.json")
    $summary = Read-Json -Path (Join-Path $directory "summary.json")
    $metadata = Read-Json -Path (Join-Path $directory "metadata.json")
    $semanticFileObject = Read-Json -Path (
        Join-Path $directory "semantic_checks.json"
    )
    $semanticFile = @($semanticFileObject)

    if (
        $rows.Count -ne $Specification.expected_rows -or
        [string]$progress.status -ne "completed" -or
        [int]$progress.planned_jobs -ne $Specification.expected_rows -or
        [int]$progress.completed_jobs -ne $Specification.expected_rows -or
        $null -ne $progress.current_job -or
        [string]$summary.status -ne "selected_results_semantics_checked" -or
        [int]$summary.planned_job_count -ne $Specification.expected_rows -or
        [int]$summary.completed_job_count -ne $Specification.expected_rows -or
        [int]$summary.run_count -ne $Specification.expected_rows -or
        [string]$metadata.mode -ne "paper"
    ) {
        throw "Completion ledger is inconsistent: $directory"
    }
    Assert-ExactValues `
        -Actual @($summary.selected_studies) `
        -Expected @($Specification.study) `
        -Label "$($Specification.output_label) selected studies"
    if (@($rows | Where-Object {
        [string]$_.study -cne $Specification.study -or
        [string]$_.case -cne "24624k_18p" -or
        [int]$_.kernel_count -ne 24624 -or
        [int]$_.presentation_count -ne 18
    }).Count -ne 0) {
        throw "A raw row is outside the declared study/case: $directory"
    }
    if (@($rows.job_id | Sort-Object -Unique).Count -ne $rows.Count) {
        throw "Raw job identifiers are not unique: $directory"
    }
    Assert-ExactValues `
        -Actual @($rows.repeat | Sort-Object -Unique) `
        -Expected @(0, 1, 2, 3, 4, 5, 6, 7, 8, 9) `
        -Label "$($Specification.output_label) repeats"
    foreach ($repeat in 0..9) {
        if (@($rows | Where-Object {
            [int]$_.repeat -eq $repeat
        }).Count -ne $Specification.rows_per_repeat) {
            throw "Repeat $repeat has the wrong row count: $directory"
        }
    }
    if (@($rows.digest | Sort-Object -Unique).Count -ne 1) {
        throw "Raw runs do not share one semantic digest: $directory"
    }

    $summaryChecks = @($summary.semantic_checks)
    if (
        $semanticFile.Count -ne 10 -or
        $summaryChecks.Count -ne 10 -or
        @($semanticFile | Where-Object {
            [string]$_.study -cne $Specification.study -or
            [string]$_.case -cne "24624k_18p" -or
            [string]$_.status -cne "PASS" -or
            [int]$_.runs -ne $Specification.rows_per_repeat -or
            [int]$_.digests -ne 1
        }).Count -ne 0 -or
        @($summaryChecks | Where-Object {
            [string]$_.study -cne $Specification.study -or
            [string]$_.case -cne "24624k_18p" -or
            [string]$_.status -cne "PASS" -or
            [int]$_.runs -ne $Specification.rows_per_repeat -or
            [int]$_.digests -ne 1
        }).Count -ne 0
    ) {
        throw "Semantic checks are incomplete or failed: $directory"
    }

    if ($Specification.study -eq "method_ladder") {
        $groups = @(
            $rows |
                Group-Object -Property method, workers |
                ForEach-Object {
                    if ($_.Count -ne 10) {
                        throw "Method/worker group is not n=10: $($_.Name)"
                    }
                    ([string]$_.Name).Replace(", ", "|")
                }
        )
        Assert-ExactValues `
            -Actual $groups `
            -Expected @(
                "factorized|1",
                "factorized|8",
                "flat|1",
                "flat_parallel|8",
                "kernel_memo|1",
                "kernel_memo_parallel|8"
            ) `
            -Label "method-ladder groups"
    }
    else {
        $workers = @(
            $rows |
                Group-Object -Property workers |
                ForEach-Object {
                    if ($_.Count -ne 10) {
                        throw "Worker group is not n=10: $($_.Name)"
                    }
                    [string][int]$_.Name
                }
        )
        Assert-ExactValues `
            -Actual $workers `
            -Expected @("1", "2", "4", "6", "8", "12", "16") `
            -Label "parallel-scaling workers"
    }

    $currentFingerprint = Get-DeadlineFingerprint
    if (
        [string]$metadata.code_fingerprint_sha256 -cne
        $currentFingerprint
    ) {
        throw "Current target code/config no longer matches result metadata."
    }

    return [ordered]@{
        directory = $directory
        result_tree = Get-ResultTreeRecord -Directory $directory
        validation = [ordered]@{
            expected_rows = $Specification.expected_rows
            observed_rows = $rows.Count
            progress_status = [string]$progress.status
            progress_planned_jobs = [int]$progress.planned_jobs
            progress_completed_jobs = [int]$progress.completed_jobs
            summary_status = [string]$summary.status
            summary_run_count = [int]$summary.run_count
            semantic_check_count = $semanticFile.Count
            semantic_checks_all_pass = $true
            repeats = 10
            rows_per_repeat = $Specification.rows_per_repeat
            unique_semantic_digests = 1
        }
        source_fingerprint = [ordered]@{
            metadata_sha256 = [string]$metadata.code_fingerprint_sha256
            recovery_time_sha256 = $currentFingerprint
            recovery_time_match = $true
            limitation = (
                "For a recovered result, this match proves the presently " +
                "retained target code/config matches the fingerprint saved " +
                "by the runner; only a native CLEAN_RUN can prove wrapper-" +
                "observed pre/post identity during the timed interval."
            )
        }
    }
}

function Get-TargetBinding {
    $source = Get-FileRecord `
        -Path (Join-Path $RepoRoot "experiments\deadline_runner.py") `
        -Role "target_source_script"
    $inputs = @(
        "experiments\deadline_profile_review_8c32g.json",
        "src\layerprobe\__init__.py",
        "src\layerprobe\cli.py",
        "src\layerprobe\evaluator.py",
        "src\layerprobe\mechanics.py",
        "src\layerprobe\model.py",
        "src\layerprobe\workloads.py"
    ) | ForEach-Object {
        Get-FileRecord `
            -Path (Join-Path $RepoRoot $_) `
            -Role "target_declared_input"
    }
    return [ordered]@{
        source_script = $source
        input_files = @($inputs)
        scope = "target_code_and_config_at_recovery_time"
    }
}

function Assert-MethodRecoveryEvidence {
    param(
        [Parameter(Mandatory = $true)][hashtable]$Specification,
        [Parameter(Mandatory = $true)][hashtable]$Payload
    )

    $wrapperDirectory = Join-Path $RepoRoot $Specification.wrapper_relative
    $stdoutPath = Join-Path $wrapperDirectory "stdout.log"
    $stderrPath = Join-Path $wrapperDirectory "stderr.log"
    $queuePath = Resolve-RepoFile -RelativePath $Specification.queue_log_relative
    foreach ($path in @($stdoutPath, $stderrPath)) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "Method wrapper log is missing: $path"
        }
    }
    $stdout = Get-Content -LiteralPath $stdoutPath -Raw
    if (
        $stdout -notmatch '"status"\s*:\s*"selected_results_semantics_checked"' -or
        $stdout -notmatch '"runs"\s*:\s*60'
    ) {
        throw "Method stdout lacks the runner's terminal success record."
    }
    if ((Get-Item -LiteralPath $stderrPath).Length -ne 0) {
        throw "Method stderr is not empty."
    }
    $queueLines = @(Get-Content -LiteralPath $queuePath)
    $starts = @(
        $queueLines | Where-Object {
            $_ -match 'Starting isolated target: method_ladder_n10$'
        }
    )
    $exits = @(
        $queueLines | Where-Object {
            $_ -match 'Isolation wrapper exit for method_ladder_n10:'
        }
    )
    if ($starts.Count -lt 1 -or $exits.Count -ge $starts.Count) {
        throw "Queue log does not show the unterminated final wrapper attempt."
    }
    if (@(
        Get-ChildItem -LiteralPath $wrapperDirectory -File -Force |
            Where-Object { $_.Name -notin @("stdout.log", "stderr.log") }
    ).Count -ne 0) {
        throw "Unexpected terminal wrapper record exists for method recovery."
    }
    return [ordered]@{
        native_clean_run_present = $false
        recovery_class = (
            "wrapper_tail_monitor_interruption_after_target_success"
        )
        machine_evidence = @(
            Get-FileRecord -Path $stdoutPath -Role "target_stdout"
            Get-FileRecord -Path $stderrPath -Role "target_stderr"
            Get-FileRecord -Path $queuePath -Role "original_queue_log"
        )
        known_wrapper_issue = [ordered]@{
            phase = "after_target_payload_finalization_before_wrapper_terminal_record"
            description = (
                "The target wrote 60/60 rows, final summary/progress files, " +
                "ten passing semantic groups, and a terminal success object " +
                "to stdout. The final queue attempt has a start line but no " +
                "wrapper-exit line or native terminal wrapper record."
            )
            causal_attribution = (
                "Operator-reported stale runner-process state in the old " +
                "wrapper tail monitor; the retained machine evidence proves " +
                "the incomplete wrapper tail, not the causal implementation " +
                "detail by itself."
            )
        }
        claim_boundary = [ordered]@{
            target_completed = $true
            raw_rows_and_semantics_verified = $true
            native_clean_run = $false
            runner_exit_code_from_native_record = $null
            full_interval_foreign_python_count = $null
            wrapper_observed_source_inputs_unchanged = $null
            statement = (
                "This is a recovered completed target, not a native clean " +
                "run. Do not use this certificate to claim wrapper-certified " +
                "zero interference or wrapper-certified pre/post source " +
                "identity for the method-ladder timing interval."
            )
        }
    }
}

function Assert-ScalingRecoveryEvidence {
    param(
        [Parameter(Mandatory = $true)][hashtable]$Specification,
        [Parameter(Mandatory = $true)][hashtable]$Payload
    )

    $wrapperDirectory = Join-Path $RepoRoot $Specification.wrapper_relative
    $cleanPath = Join-Path $wrapperDirectory "CLEAN_RUN.json"
    $failedPath = Join-Path $wrapperDirectory "FAILED.json"
    $stdoutPath = Join-Path $wrapperDirectory "stdout.log"
    $stderrPath = Join-Path $wrapperDirectory "stderr.log"
    $queuePath = Resolve-RepoFile -RelativePath $Specification.queue_log_relative
    $dnuFiles = @(
        Get-ChildItem -LiteralPath $Payload.directory -File -Force |
            Where-Object { $_.Name -like "DO_NOT_USE_*.json" }
    )
    if ($dnuFiles.Count -ne 1) {
        throw "Scaling recovery requires exactly one preserved DO_NOT_USE marker."
    }
    foreach ($path in @(
        $cleanPath,
        $failedPath,
        $stdoutPath,
        $stderrPath
    )) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "Scaling wrapper evidence is missing: $path"
        }
    }
    $clean = Read-Json -Path $cleanPath
    $failed = Read-Json -Path $failedPath
    $dnu = Read-Json -Path $dnuFiles[0].FullName
    if (
        [string]$clean.status -cne "clean_run" -or
        [int]$clean.runner_exit_code -ne 0 -or
        [int]$clean.foreign_python_count -ne 0 -or
        [int]$clean.ignored_control_probe_count -ne 0 -or
        [int]$clean.preflight_stable_seconds -lt 30 -or
        -not [bool]$clean.source_input_integrity.complete -or
        -not [bool]$clean.source_input_integrity.unchanged -or
        [string]$clean.output_label -cne $Specification.output_label
    ) {
        throw "Scaling native CLEAN_RUN does not pass its isolation fields."
    }
    if (
        [string]$clean.result_tree.path_relative_to_repo -cne
            [string]$Payload.result_tree.path_relative_to_repo -or
        [string]$clean.result_tree.tree_sha256 -cne
            [string]$Payload.result_tree.tree_sha256 -or
        [int]$clean.result_tree.file_count -ne
            [int]$Payload.result_tree.file_count
    ) {
        throw (
            "Scaling native CLEAN_RUN result tree does not bind payload: " +
            "clean=$($clean.result_tree.tree_sha256)/" +
            "$($clean.result_tree.file_count), actual=" +
            "$($Payload.result_tree.tree_sha256)/" +
            "$($Payload.result_tree.file_count)."
        )
    }
    if (
        [string]$failed.status -cne "wrapper_failed" -or
        [int]$failed.runner_pid -ne [int]$clean.runner_pid -or
        [string]$failed.error -notmatch "expected_output_dir" -or
        [string]$failed.logs.stdout_sha256 -cne
            [string]$clean.logs.hashes.stdout_sha256 -or
        [string]$failed.logs.stderr_sha256 -cne
            [string]$clean.logs.hashes.stderr_sha256
    ) {
        throw "Scaling FAILED record is not the known post-success sidecar bug."
    }
    if (
        [string]$dnu.status -cne "inadmissible_formal_result" -or
        [string]$dnu.reason -notmatch "expected_output_dir" -or
        [string]$dnu.isolation_record_relative_to_repo -cne (
            Convert-ToManifestPath -Path (
                Get-PortableRelativePath `
                    -BasePath $RepoRoot `
                    -TargetPath $failedPath
            )
        )
    ) {
        throw "Preserved scaling DO_NOT_USE marker does not bind FAILED.json."
    }
    if ((Get-Item -LiteralPath $stderrPath).Length -ne 0) {
        throw "Scaling stderr is not empty."
    }
    $stdout = Get-Content -LiteralPath $stdoutPath -Raw
    if (
        $stdout -notmatch '"status"\s*:\s*"selected_results_semantics_checked"' -or
        $stdout -notmatch '"runs"\s*:\s*70'
    ) {
        throw "Scaling stdout lacks the runner's terminal success record."
    }

    return [ordered]@{
        native_clean_run_present = $true
        recovery_class = "wrapper_only_post_success_sidecar_failure"
        machine_evidence = @(
            Get-FileRecord -Path $cleanPath -Role "native_clean_run"
            Get-FileRecord -Path $failedPath -Role "wrapper_failed_record"
            Get-FileRecord -Path $stdoutPath -Role "target_stdout"
            Get-FileRecord -Path $stderrPath -Role "target_stderr"
            Get-FileRecord -Path $queuePath -Role "original_queue_log"
            Get-FileRecord `
                -Path $dnuFiles[0].FullName `
                -Role "preserved_do_not_use_marker"
        )
        known_wrapper_issue = [ordered]@{
            phase = "after_clean_record_write_during_sidecar_creation"
            missing_property = "expected_output_dir"
            description = (
                "The old wrapper wrote a valid native CLEAN_RUN after the " +
                "runner exited 0, then tried to read a property absent from " +
                "that record while creating FORMAL_PROVENANCE. Its outer " +
                "catch wrote FAILED.json and the preserved DO_NOT_USE marker."
            )
        }
        claim_boundary = [ordered]@{
            target_completed = $true
            raw_rows_and_semantics_verified = $true
            native_clean_run = $true
            runner_exit_code_from_native_record = 0
            full_interval_foreign_python_count = 0
            wrapper_observed_source_inputs_unchanged = $true
            statement = (
                "The recovery overrides only the wrapper's post-success " +
                "sidecar failure. It does not erase or reinterpret the " +
                "preserved DO_NOT_USE marker."
            )
        }
    }
}

function Write-Json {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)]$Value
    )
    [IO.File]::WriteAllText(
        $Path,
        ($Value | ConvertTo-Json -Depth 20) + "`n",
        $Utf8NoBom
    )
}

function Assert-GeneratedCertificates {
    param(
        [Parameter(Mandatory = $true)][hashtable]$Specification,
        [Parameter(Mandatory = $true)][hashtable]$Payload
    )

    $overridePath = Join-Path $Payload.directory "RECOVERY_OVERRIDE.json"
    $provenancePath = Join-Path $Payload.directory "FORMAL_PROVENANCE.json"
    foreach ($path in @($overridePath, $provenancePath)) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "Generated recovery certificate is missing: $path"
        }
    }
    $override = Read-Json -Path $overridePath
    $provenance = Read-Json -Path $provenancePath
    $actualTree = Get-ResultTreeRecord -Directory $Payload.directory
    $expectedRecoveryRelative = Convert-ToManifestPath -Path (
        Get-PortableRelativePath `
            -BasePath $RepoRoot `
            -TargetPath $overridePath
    )
    if (
        [int]$override.schema_version -ne 2 -or
        [string]$override.status -cne "recovery_override" -or
        [string]$override.authorization -cne
            "explicit_local_wrapper_recovery_2026-07-25" -or
        [string]$override.output_label -cne $Specification.output_label -or
        [string]$override.expected_output_dir -cne
            $actualTree.path_relative_to_repo -or
        [string]$override.study -cne $Specification.study -or
        [string]$override.recovery_class -cne
            $Specification.recovery_class -or
        [int]$override.validation.expected_rows -ne
            $Specification.expected_rows -or
        [int]$override.validation.observed_rows -ne
            $Specification.expected_rows -or
        -not [bool]$override.validation.semantic_checks_all_pass -or
        [int]$override.validation.semantic_check_count -ne 10 -or
        [int]$override.validation.repeats -ne 10 -or
        [int]$override.validation.rows_per_repeat -ne
            $Specification.rows_per_repeat -or
        [int]$override.validation.unique_semantic_digests -ne 1 -or
        -not [bool]$override.source_fingerprint.recovery_time_match -or
        [string]$override.source_fingerprint.metadata_sha256 -cne
            (Get-DeadlineFingerprint) -or
        [string]$override.target_binding.scope -cne
            "target_code_and_config_at_recovery_time" -or
        [string]$override.result_tree.tree_sha256 -cne
            $actualTree.tree_sha256 -or
        [int]$override.result_tree.file_count -ne
            [int]$actualTree.file_count -or
        [int]$provenance.schema_version -ne 2 -or
        [string]$provenance.status -cne "recovered_formal_result" -or
        [string]$provenance.provenance_mode -cne "recovery_override" -or
        [string]$provenance.authorization -cne
            "explicit_local_wrapper_recovery_2026-07-25" -or
        [string]$provenance.output_label -cne
            $Specification.output_label -or
        [string]$provenance.expected_output_dir -cne
            $actualTree.path_relative_to_repo -or
        [string]$provenance.study -cne $Specification.study -or
        [string]$provenance.recovery_class -cne
            $Specification.recovery_class -or
        [string]$provenance.binding_scope -cne
            "target_code_and_config_at_recovery_time" -or
        [string]$provenance.result_tree_sha256 -cne
            $actualTree.tree_sha256 -or
        [int]$provenance.result_file_count -ne
            [int]$actualTree.file_count -or
        [string]$provenance.recovery_record_relative_to_repo -cne
            $expectedRecoveryRelative -or
        [string]$provenance.recovery_record_sha256 -cne
            (Get-Sha256 -Path $overridePath)
    ) {
        throw "Recovery certificates do not match their payload: $($Payload.directory)"
    }
    Assert-ExactOrderedValues `
        -Actual @($override.argument_list) `
        -Expected @($Specification.arguments) `
        -Label "$($Specification.output_label) override arguments"
    Assert-ExactOrderedValues `
        -Actual @($provenance.argument_list) `
        -Expected @($Specification.arguments) `
        -Label "$($Specification.output_label) provenance arguments"

    $generatorPath = Resolve-RepoFile `
        -RelativePath ([string]$override.generator.path_relative_to_repo)
    if (
        [string]$override.generator.role -cne
            "recovery_certificate_generator" -or
        [long](Get-Item -LiteralPath $generatorPath).Length -ne
            [long]$override.generator.bytes -or
        (Get-Sha256 -Path $generatorPath) -cne
            [string]$override.generator.sha256 -or
        -not $generatorPath.Equals(
            $RecoveryScriptPath,
            [StringComparison]::OrdinalIgnoreCase
        )
    ) {
        throw "Recovery certificate generator binding is invalid."
    }

    $bindingRecords = @(
        $override.target_binding.source_script
        $override.target_binding.input_files
    )
    foreach ($record in $bindingRecords) {
        $path = Resolve-RepoFile `
            -RelativePath ([string]$record.path_relative_to_repo)
        if (
            [long](Get-Item -LiteralPath $path).Length -ne
                [long]$record.bytes -or
            (Get-Sha256 -Path $path) -cne [string]$record.sha256
        ) {
            throw "Recovered target binding mismatch: $path"
        }
    }
    $evidence = @($override.machine_evidence)
    foreach ($record in $evidence) {
        $path = Resolve-RepoFile `
            -RelativePath ([string]$record.path_relative_to_repo)
        if (
            [long](Get-Item -LiteralPath $path).Length -ne
                [long]$record.bytes -or
            (Get-Sha256 -Path $path) -cne [string]$record.sha256
        ) {
            throw "Recovery evidence hash mismatch: $path"
        }
    }
}

$specifications = @(
    @{
        output_label = "method_ladder_n10"
        study = "method_ladder"
        expected_rows = 60
        rows_per_repeat = 6
        result_relative = (
            "results\review_method_ladder_n10_local_20260724c_" +
            "retry17_20260724T163927256Z_e393119e"
        )
        wrapper_relative = (
            "artifact_runs\formal_isolation\method_ladder_n10_" +
            "20260724T163927451Z_wrapper28368_776a53de"
        )
        queue_log_relative = (
            "artifact_runs\formal_queue\singlehost_" +
            "20260724T155844591Z_pid28368\queue.log"
        )
        recovery_class = (
            "wrapper_tail_monitor_interruption_after_target_success"
        )
        arguments = @(
            "--config",
            "experiments\deadline_profile_review_8c32g.json",
            "--mode",
            "paper",
            "--only",
            "method_ladder",
            "--output",
            (
                "results\review_method_ladder_n10_local_20260724c_" +
                "retry17_20260724T163927256Z_e393119e"
            ),
            "--ignore-freeze"
        )
    },
    @{
        output_label = "parallel_scaling_n10"
        study = "parallel_scaling"
        expected_rows = 70
        rows_per_repeat = 7
        result_relative = "results\parallel_scaling_n10_local_20260724c"
        wrapper_relative = (
            "artifact_runs\formal_isolation\parallel_scaling_n10_" +
            "20260724T175308263Z_wrapper32044_d24c7c15"
        )
        queue_log_relative = (
            "artifact_runs\formal_queue\singlehost_" +
            "20260724T175256300Z_pid32044\queue.log"
        )
        recovery_class = "wrapper_only_post_success_sidecar_failure"
        arguments = @(
            "--config",
            "experiments\deadline_profile_review_8c32g.json",
            "--mode",
            "paper",
            "--only",
            "parallel_scaling",
            "--output",
            "results\parallel_scaling_n10_local_20260724c",
            "--ignore-freeze"
        )
    }
)

$scriptRecord = Get-FileRecord `
    -Path $RecoveryScriptPath `
    -Role "recovery_certificate_generator"
$targetBinding = Get-TargetBinding

foreach ($specification in $specifications) {
    $payload = Assert-ResultPayload -Specification $specification
    $evidence = if ($specification.study -eq "method_ladder") {
        Assert-MethodRecoveryEvidence `
            -Specification $specification `
            -Payload $payload
    }
    else {
        Assert-ScalingRecoveryEvidence `
            -Specification $specification `
            -Payload $payload
    }
    if (
        [string]$evidence.recovery_class -cne
        [string]$specification.recovery_class
    ) {
        throw "Recovery class mismatch."
    }

    $overridePath = Join-Path $payload.directory "RECOVERY_OVERRIDE.json"
    $provenancePath = Join-Path $payload.directory "FORMAL_PROVENANCE.json"
    if (-not $VerifyOnly) {
        if (
            (Test-Path -LiteralPath $overridePath -PathType Leaf) -or
            (Test-Path -LiteralPath $provenancePath -PathType Leaf)
        ) {
            throw (
                "Recovery certificates already exist; use -VerifyOnly " +
                "instead of overwriting them: $($payload.directory)"
            )
        }
        $issuedAt = (Get-Date).ToUniversalTime().ToString("o")
        $override = [ordered]@{
            schema_version = 2
            status = "recovery_override"
            authorization = "explicit_local_wrapper_recovery_2026-07-25"
            issued_at_utc = $issuedAt
            output_label = $specification.output_label
            expected_output_dir = $payload.result_tree.path_relative_to_repo
            study = $specification.study
            recovery_class = $evidence.recovery_class
            native_clean_run_present = $evidence.native_clean_run_present
            validation = $payload.validation
            source_fingerprint = $payload.source_fingerprint
            target_binding = $targetBinding
            argument_list = @($specification.arguments)
            result_tree = $payload.result_tree
            machine_evidence = @($evidence.machine_evidence)
            known_wrapper_issue = $evidence.known_wrapper_issue
            claim_boundary = $evidence.claim_boundary
            generator = $scriptRecord
        }
        Write-Json -Path $overridePath -Value $override
        $provenance = [ordered]@{
            schema_version = 2
            status = "recovered_formal_result"
            provenance_mode = "recovery_override"
            authorization = "explicit_local_wrapper_recovery_2026-07-25"
            output_label = $specification.output_label
            expected_output_dir = $payload.result_tree.path_relative_to_repo
            study = $specification.study
            recovery_class = $evidence.recovery_class
            native_clean_run_present = $evidence.native_clean_run_present
            result_tree_relative_to_repo = (
                $payload.result_tree.path_relative_to_repo
            )
            result_tree_sha256 = $payload.result_tree.tree_sha256
            result_file_count = $payload.result_tree.file_count
            source_script = $targetBinding.source_script
            input_files = @($targetBinding.input_files)
            binding_scope = $targetBinding.scope
            source_fingerprint = $payload.source_fingerprint
            argument_list = @($specification.arguments)
            recovery_record_relative_to_repo = Convert-ToManifestPath -Path (
                Get-PortableRelativePath `
                    -BasePath $RepoRoot `
                    -TargetPath $overridePath
            )
            recovery_record_sha256 = Get-Sha256 -Path $overridePath
            claim_boundary = $evidence.claim_boundary
        }
        Write-Json -Path $provenancePath -Value $provenance
    }
    Assert-GeneratedCertificates `
        -Specification $specification `
        -Payload $payload
    Write-Output (
        "RECOVERY_CERTIFICATE_OK {0} {1} rows={2}" -f
        $specification.output_label,
        $payload.result_tree.tree_sha256,
        $specification.expected_rows
    )
}
