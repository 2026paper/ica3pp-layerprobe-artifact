[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$PythonExe,

    [Parameter(Mandatory = $true)]
    [string]$ScriptPath,

    [string[]]$ArgumentList = @(),

    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$OutputLabel,

    [string]$ExpectedOutputDir,

    [string[]]$InputFiles = @(),

    [ValidateRange(0, 300)]
    [int]$PreflightStableSeconds = 30,

    [ValidateRange(1, 60)]
    [int]$PollSeconds = 2
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path -LiteralPath (Join-Path $ScriptDir "..")).Path
$Runner = $null
$KnownTree = @{}
$RunDir = $null
$StdoutPath = $null
$StderrPath = $null
$ResolvedPython = $null
$ResolvedScript = $null
$RenderedCommand = $null
$RunStartUtc = $null
$WrapperExitCode = 99
$ResolvedExpectedOutput = $null
$ResolvedInputFiles = @()

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
    $targetFull = [IO.Path]::GetFullPath($TargetPath)
    $baseUri = [Uri]::new($baseFull)
    $targetUri = [Uri]::new($targetFull)
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

function Resolve-Executable {
    param([Parameter(Mandatory = $true)][string]$Requested)

    if (Test-Path -LiteralPath $Requested -PathType Leaf) {
        return (Resolve-Path -LiteralPath $Requested).Path
    }
    $command = Get-Command -Name $Requested -CommandType Application -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($null -eq $command) {
        throw "Python executable was not found: $Requested"
    }
    return $command.Source
}

function Resolve-RepositoryFile {
    param([Parameter(Mandatory = $true)][string]$Requested)

    $candidate = if ([IO.Path]::IsPathRooted($Requested)) {
        [IO.Path]::GetFullPath($Requested)
    } else {
        [IO.Path]::GetFullPath((Join-Path $RepoRoot $Requested))
    }
    if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) {
        throw "Python script was not found: $candidate"
    }
    return (Resolve-Path -LiteralPath $candidate).Path
}

function Resolve-RepositoryPath {
    param([Parameter(Mandatory = $true)][string]$Requested)

    $candidate = if ([IO.Path]::IsPathRooted($Requested)) {
        [IO.Path]::GetFullPath($Requested)
    } else {
        [IO.Path]::GetFullPath((Join-Path $RepoRoot $Requested))
    }
    $relative = Get-PortableRelativePath `
        -BasePath $RepoRoot `
        -TargetPath $candidate
    if (
        [IO.Path]::IsPathRooted($relative) -or
        $relative -eq ".." -or
        $relative.StartsWith("..$([IO.Path]::DirectorySeparatorChar)")
    ) {
        throw "Expected output must remain inside the repository: $candidate"
    }
    if (Test-Path -LiteralPath $candidate -PathType Leaf) {
        throw "Expected output path is a file: $candidate"
    }
    return $candidate
}

function Get-HashedFileRecord {
    param([Parameter(Mandatory = $true)][string]$Path)

    $resolved = (Resolve-Path -LiteralPath $Path).Path
    $item = Get-Item -LiteralPath $resolved
    return [ordered]@{
        path_relative_to_repo = (
            Get-PortableRelativePath `
                -BasePath $RepoRoot `
                -TargetPath $resolved
        ).Replace([char]0x5C, [char]0x2F)
        bytes = [long]$item.Length
        sha256 = (
            Get-FileHash -LiteralPath $resolved -Algorithm SHA256
        ).Hash.ToLowerInvariant()
    }
}

function Test-HashedFileRecordEqual {
    param(
        [Parameter(Mandatory = $true)]$Left,
        [Parameter(Mandatory = $true)]$Right
    )

    return (
        [string]$Left.path_relative_to_repo -ceq
            [string]$Right.path_relative_to_repo -and
        [long]$Left.bytes -eq [long]$Right.bytes -and
        [string]$Left.sha256 -ceq [string]$Right.sha256
    )
}

function Get-PostRunBindingAudit {
    param(
        [Parameter(Mandatory = $true)][string]$Script,
        [Parameter(Mandatory = $true)]
        [AllowEmptyCollection()]
        [string[]]$Inputs,
        [Parameter(Mandatory = $true)]$PreRunScript,
        [Parameter(Mandatory = $true)]
        [AllowEmptyCollection()]
        [object[]]$PreRunInputs
    )

    try {
        $postRunScript = Get-HashedFileRecord -Path $Script
        $postRunInputs = @(
            $Inputs | ForEach-Object {
                Get-HashedFileRecord -Path $_
            }
        )
        $unchanged = Test-HashedFileRecordEqual `
            -Left $PreRunScript `
            -Right $postRunScript
        if ($postRunInputs.Count -ne $PreRunInputs.Count) {
            $unchanged = $false
        } else {
            for ($index = 0; $index -lt $PreRunInputs.Count; $index += 1) {
                if (-not (Test-HashedFileRecordEqual `
                    -Left $PreRunInputs[$index] `
                    -Right $postRunInputs[$index])) {
                    $unchanged = $false
                }
            }
        }
        return [ordered]@{
            complete      = $true
            unchanged     = [bool]$unchanged
            source_script = $postRunScript
            input_files   = $postRunInputs
            error         = $null
        }
    } catch {
        return [ordered]@{
            complete      = $false
            unchanged     = $false
            source_script = $null
            input_files   = @()
            error         = $_.Exception.Message
        }
    }
}

function Get-DirectoryHashRecord {
    param([Parameter(Mandatory = $true)][string]$Directory)

    if (-not (Test-Path -LiteralPath $Directory -PathType Container)) {
        throw "Expected result directory was not created: $Directory"
    }
    $entries = @(
        Get-ChildItem -LiteralPath $Directory -File -Recurse |
            Where-Object {
                $_.Name -ne "FORMAL_PROVENANCE.json" -and
                $_.Name -notlike "DO_NOT_USE_*.json"
            } |
            Sort-Object -Property FullName |
            ForEach-Object {
                $relative = (
                    Get-PortableRelativePath `
                        -BasePath $Directory `
                        -TargetPath $_.FullName
                ).Replace([char]0x5C, [char]0x2F)
                [ordered]@{
                    path = $relative
                    bytes = [long]$_.Length
                    sha256 = (
                        Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256
                    ).Hash.ToLowerInvariant()
                }
            }
    )
    if ($entries.Count -eq 0) {
        throw "Expected result directory contains no result files: $Directory"
    }
    $canonicalLines = @(
        $entries | ForEach-Object {
            "$($_.sha256) $($_.bytes) $($_.path)"
        }
    )
    $canonical = $canonicalLines -join "`n"
    $sha = [Security.Cryptography.SHA256]::Create()
    try {
        $digestBytes = $sha.ComputeHash(
            [Text.Encoding]::UTF8.GetBytes($canonical)
        )
    } finally {
        $sha.Dispose()
    }
    return [ordered]@{
        path_relative_to_repo = (
            Get-PortableRelativePath `
                -BasePath $RepoRoot `
                -TargetPath $Directory
        ).Replace([char]0x5C, [char]0x2F)
        file_count = $entries.Count
        tree_sha256 = [BitConverter]::ToString(
            $digestBytes
        ).Replace("-", "").ToLowerInvariant()
        files = $entries
    }
}

function ConvertTo-WindowsCommandLineArgument {
    param([AllowEmptyString()][string]$Value)

    if ($Value.Length -gt 0 -and $Value -notmatch '[\s"]') {
        return $Value
    }

    # Quote according to the CommandLineToArgvW backslash-before-quote rules.
    $builder = [Text.StringBuilder]::new()
    [void]$builder.Append([char]0x22)
    $backslashes = 0
    foreach ($character in $Value.ToCharArray()) {
        if ($character -eq [char]0x5C) {
            $backslashes += 1
            continue
        }
        if ($character -eq [char]0x22) {
            [void]$builder.Append([char]0x5C, (2 * $backslashes) + 1)
            [void]$builder.Append([char]0x22)
            $backslashes = 0
            continue
        }
        if ($backslashes -gt 0) {
            [void]$builder.Append([char]0x5C, $backslashes)
            $backslashes = 0
        }
        [void]$builder.Append($character)
    }
    if ($backslashes -gt 0) {
        [void]$builder.Append([char]0x5C, 2 * $backslashes)
    }
    [void]$builder.Append([char]0x22)
    return $builder.ToString()
}

function Get-AllProcessSnapshot {
    return @(Get-CimInstance -ClassName Win32_Process)
}

function Test-IsPythonProcess {
    param([Parameter(Mandatory = $true)]$Process)

    $name = [string]$Process.Name
    # Cover the common Windows launcher names (python.exe, pythonw.exe,
    # python3.exe, python312.exe, python3.12.exe, and analogous versions)
    # without matching unrelated executables whose names merely contain
    # "python".
    return $name -imatch '^pythonw?(?:\d+(?:\.\d+)*)?\.exe$'
}

function Test-IsIgnoredControlProbe {
    param([Parameter(Mandatory = $true)]$Process)

    # Formal evidence uses the conservative rule that every Python-family
    # process outside the launched process tree is competing work.  This
    # function remains as a single audit point for the provenance schema, but
    # deliberately has an empty whitelist.
    return $false
}

function Get-CreationTicks {
    param([Parameter(Mandatory = $true)]$Process)

    if ($null -eq $Process.CreationDate) {
        return [long]0
    }
    try {
        return ([datetime]$Process.CreationDate).ToUniversalTime().Ticks
    } catch {
        return [long]0
    }
}

function Get-ProcessRecord {
    param([Parameter(Mandatory = $true)]$Process)

    $createdUtc = $null
    if ($null -ne $Process.CreationDate) {
        try {
            $createdUtc = ([datetime]$Process.CreationDate).ToUniversalTime().ToString("o")
        } catch {
            $createdUtc = [string]$Process.CreationDate
        }
    }
    return [ordered]@{
        pid               = [int]$Process.ProcessId
        parent_pid        = [int]$Process.ParentProcessId
        name              = [string]$Process.Name
        creation_time_utc = $createdUtc
        command_line      = [string]$Process.CommandLine
    }
}

function Expand-RunnerTree {
    param(
        [Parameter(Mandatory = $true)][object[]]$Processes,
        [Parameter(Mandatory = $true)][hashtable]$Known,
        [Parameter(Mandatory = $true)][int]$RootPid
    )

    $currentTreeIds = [Collections.Generic.HashSet[int]]::new()
    $changed = $true
    while ($changed) {
        $changed = $false
        foreach ($process in $Processes) {
            $processId = [int]$process.ProcessId
            $parentId = [int]$process.ParentProcessId
            $creationTicks = Get-CreationTicks -Process $process
            $belongs = $false

            if ($Known.ContainsKey($processId)) {
                $knownTicks = [long]$Known[$processId]
                $belongs = ($knownTicks -eq 0 -or $knownTicks -eq $creationTicks)
            }
            # A historical PID in $Known is not sufficient evidence that a
            # newly observed process is still part of the live runner tree:
            # Windows can reuse the PID after the original worker exits.
            # Expand only from a parent whose exact live instance was already
            # admitted to the tree in this snapshot.
            if (-not $belongs -and $currentTreeIds.Contains($parentId)) {
                $belongs = $true
            }

            if ($belongs) {
                if (-not $Known.ContainsKey($processId) -or [long]$Known[$processId] -eq 0) {
                    $Known[$processId] = $creationTicks
                }
                if ($currentTreeIds.Add($processId)) {
                    $changed = $true
                }
            }
        }
    }

    return @(
        $Processes | Where-Object {
            $currentTreeIds.Contains([int]$_.ProcessId)
        }
    )
}

function Get-ForeignPythonProcesses {
    param(
        [Parameter(Mandatory = $true)][object[]]$Processes,
        [Parameter(Mandatory = $true)]
        [AllowEmptyCollection()]
        [object[]]$TreeProcesses
    )

    $treeIds = [Collections.Generic.HashSet[int]]::new()
    foreach ($treeProcess in $TreeProcesses) {
        [void]$treeIds.Add([int]$treeProcess.ProcessId)
    }
    return @(
        $Processes | Where-Object {
            (Test-IsPythonProcess -Process $_) -and
            -not (Test-IsIgnoredControlProbe -Process $_) -and
            -not $treeIds.Contains([int]$_.ProcessId)
        }
    )
}

function Get-IgnoredControlProbeProcesses {
    param(
        [Parameter(Mandatory = $true)][object[]]$Processes,
        [Parameter(Mandatory = $true)]
        [AllowEmptyCollection()]
        [object[]]$TreeProcesses
    )

    $treeIds = [Collections.Generic.HashSet[int]]::new()
    foreach ($treeProcess in $TreeProcesses) {
        [void]$treeIds.Add([int]$treeProcess.ProcessId)
    }
    return @(
        $Processes | Where-Object {
            (Test-IsIgnoredControlProbe -Process $_) -and
            -not $treeIds.Contains([int]$_.ProcessId)
        }
    )
}

function Add-IgnoredControlProbeRecords {
    param(
        [Parameter(Mandatory = $true)][hashtable]$Index,
        [Parameter(Mandatory = $true)]
        [AllowEmptyCollection()]
        [object[]]$Processes
    )

    foreach ($process in $Processes) {
        $key = "{0}:{1}" -f (
            [int]$process.ProcessId
        ), (
            Get-CreationTicks -Process $process
        )
        if (-not $Index.ContainsKey($key)) {
            $Index[$key] = Get-ProcessRecord -Process $process
        }
    }
}

function Get-IgnoredControlProbeRecords {
    param([Parameter(Mandatory = $true)][hashtable]$Index)

    return @(
        $Index.Values |
            Sort-Object -Property creation_time_utc, pid
    )
}

function Test-ExactProcessIdentity {
    param(
        [Parameter(Mandatory = $true)]$Expected,
        [Parameter(Mandatory = $true)]$Observed
    )

    if ([int]$Expected.ProcessId -ne [int]$Observed.ProcessId) {
        return $false
    }
    $expectedTicks = Get-CreationTicks -Process $Expected
    $observedTicks = Get-CreationTicks -Process $Observed
    return ($expectedTicks -eq 0 -or $observedTicks -eq $expectedTicks)
}

function Stop-RunnerTreeExact {
    param(
        [Parameter(Mandatory = $true)][int]$RootPid,
        [Parameter(Mandatory = $true)][hashtable]$Known
    )

    $stopped = [Collections.Generic.List[object]]::new()
    # Multiple fresh snapshots catch children that were being created while the
    # first exact process list was captured. Every Stop-Process target is
    # identity-checked immediately before use; no name-wide kill is performed.
    for ($pass = 0; $pass -lt 3; $pass += 1) {
        $snapshot = Get-AllProcessSnapshot
        $tree = @(Expand-RunnerTree -Processes $snapshot -Known $Known -RootPid $RootPid)
        if ($tree.Count -eq 0) {
            break
        }

        $rootFirst = @($tree | Where-Object { [int]$_.ProcessId -eq $RootPid })
        $otherProcesses = @(
            $tree |
                Where-Object { [int]$_.ProcessId -ne $RootPid } |
                Sort-Object -Property CreationDate -Descending
        )
        foreach ($expected in @($rootFirst + $otherProcesses)) {
            $processId = [int]$expected.ProcessId
            $observed = Get-CimInstance -ClassName Win32_Process `
                -Filter "ProcessId = $processId" `
                -ErrorAction SilentlyContinue
            if ($null -eq $observed) {
                continue
            }
            if (-not (Test-ExactProcessIdentity -Expected $expected -Observed $observed)) {
                continue
            }
            Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
            $stopped.Add((Get-ProcessRecord -Process $expected))
        }
        Start-Sleep -Milliseconds 100
    }
    return @($stopped)
}

function Get-HostRecord {
    $os = Get-CimInstance -ClassName Win32_OperatingSystem
    $computer = Get-CimInstance -ClassName Win32_ComputerSystem
    $processors = @(Get-CimInstance -ClassName Win32_Processor)
    return [ordered]@{
        os_caption         = [string]$os.Caption
        os_version         = [string]$os.Version
        cpu_models          = @($processors | ForEach-Object {
            ([string]$_.Name).Trim()
        })
        physical_cores      = [int](
            ($processors | Measure-Object -Property NumberOfCores -Sum).Sum
        )
        logical_processors = [int]$computer.NumberOfLogicalProcessors
        total_memory_bytes = [long]$computer.TotalPhysicalMemory
    }
}

function Get-SystemLoadRecord {
    $os = Get-CimInstance -ClassName Win32_OperatingSystem
    $processors = @(Get-CimInstance -ClassName Win32_Processor)
    return [ordered]@{
        time_utc = (Get-Date).ToUniversalTime().ToString("o")
        free_physical_memory_bytes = [long]$os.FreePhysicalMemory * 1024
        cpu_load_percent = @($processors | ForEach-Object {
            [int]$_.LoadPercentage
        })
    }
}

function Write-DoNotUseMarker {
    param(
        [string]$Directory,
        [Parameter(Mandatory = $true)][string]$Reason,
        [Parameter(Mandatory = $true)][string]$IsolationRecord
    )

    if (
        [string]::IsNullOrWhiteSpace($Directory) -or
        -not (Test-Path -LiteralPath $Directory -PathType Container)
    ) {
        return
    }
    $stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssfffZ")
    $markerPath = Join-Path $Directory "DO_NOT_USE_${stamp}.json"
    $marker = [ordered]@{
        status = "inadmissible_formal_result"
        reason = $Reason
        isolation_record_relative_to_repo = (
            Get-PortableRelativePath `
                -BasePath $RepoRoot `
                -TargetPath $IsolationRecord
        ).Replace([char]0x5C, [char]0x2F)
    }
    Write-JsonCreateNew -Path $markerPath -Value $marker
}

function Get-LogHashRecord {
    param(
        [string]$StandardOutput,
        [string]$StandardError
    )

    $record = [ordered]@{}
    foreach ($item in @(
        [pscustomobject]@{ Name = "stdout_sha256"; Path = $StandardOutput },
        [pscustomobject]@{ Name = "stderr_sha256"; Path = $StandardError }
    )) {
        $hash = $null
        if (-not [string]::IsNullOrWhiteSpace($item.Path) -and
            (Test-Path -LiteralPath $item.Path -PathType Leaf)) {
            for ($attempt = 0; $attempt -lt 10; $attempt += 1) {
                try {
                    $hash = (Get-FileHash -LiteralPath $item.Path -Algorithm SHA256).Hash.ToLowerInvariant()
                    break
                } catch {
                    Start-Sleep -Milliseconds 100
                }
            }
        }
        $record[$item.Name] = $hash
    }
    return $record
}

function Write-JsonCreateNew {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)]$Value
    )

    $json = $Value | ConvertTo-Json -Depth 10
    $encoding = [Text.UTF8Encoding]::new($false)
    $stream = [IO.File]::Open(
        $Path,
        [IO.FileMode]::CreateNew,
        [IO.FileAccess]::Write,
        [IO.FileShare]::Read
    )
    try {
        $writer = [IO.StreamWriter]::new($stream, $encoding)
        try {
            $writer.WriteLine($json)
            $writer.Flush()
        } finally {
            $writer.Dispose()
        }
    } finally {
        $stream.Dispose()
    }
}

try {
    $ResolvedPython = Resolve-Executable -Requested $PythonExe
    $ResolvedScript = Resolve-RepositoryFile -Requested $ScriptPath
    if (-not [string]::IsNullOrWhiteSpace($ExpectedOutputDir)) {
        $ResolvedExpectedOutput = Resolve-RepositoryPath `
            -Requested $ExpectedOutputDir
        if (Test-Path -LiteralPath $ResolvedExpectedOutput) {
            throw (
                "Refusing to start a formal run with a pre-existing " +
                "ExpectedOutputDir: $ResolvedExpectedOutput"
            )
        }
    }
    $ResolvedInputFiles = @(
        foreach ($inputFile in $InputFiles) {
            Resolve-RepositoryFile -Requested $inputFile
        }
    )
    $resolvedInputIndex = @{}
    foreach ($resolvedInputFile in $ResolvedInputFiles) {
        $inputKey = $resolvedInputFile.ToLowerInvariant()
        if ($resolvedInputIndex.ContainsKey($inputKey)) {
            throw "Declared input file was supplied more than once: $resolvedInputFile"
        }
        $resolvedInputIndex[$inputKey] = $true
    }

    $safeLabel = ($OutputLabel -replace '[^A-Za-z0-9._-]+', '_').Trim([char[]]"_.")
    if ([string]::IsNullOrWhiteSpace($safeLabel)) {
        throw "OutputLabel contains no usable filename characters."
    }
    $outputRoot = Join-Path $RepoRoot "artifact_runs\formal_isolation"
    if (Test-Path -LiteralPath $outputRoot -PathType Leaf) {
        throw "Formal-isolation output root is a file: $outputRoot"
    }
    if (-not (Test-Path -LiteralPath $outputRoot -PathType Container)) {
        New-Item -ItemType Directory -Path $outputRoot | Out-Null
    }

    $stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssfffZ")
    $uniqueSuffix = [Guid]::NewGuid().ToString("N").Substring(0, 8)
    $runName = "${safeLabel}_${stamp}_wrapper${PID}_${uniqueSuffix}"
    $RunDir = Join-Path $outputRoot $runName
    if (Test-Path -LiteralPath $RunDir) {
        throw "Refusing to overwrite an existing formal-run directory: $RunDir"
    }
    New-Item -ItemType Directory -Path $RunDir | Out-Null

    $StdoutPath = Join-Path $RunDir "stdout.log"
    $StderrPath = Join-Path $RunDir "stderr.log"
    $hostRecord = Get-HostRecord
    $wrapperStartUtc = (Get-Date).ToUniversalTime()
    $IgnoredControlProbeIndex = @{}

    # The target command does not exist yet, so every Python-family process is
    # foreign. Require a continuous Python-free interval instead of a single
    # instantaneous snapshot.
    $preflightPython = @()
    $preflightStart = Get-Date
    do {
        $preflightSnapshot = Get-AllProcessSnapshot
        $preflightIgnored = @(
            Get-IgnoredControlProbeProcesses `
                -Processes $preflightSnapshot `
                -TreeProcesses @()
        )
        Add-IgnoredControlProbeRecords `
            -Index $IgnoredControlProbeIndex `
            -Processes $preflightIgnored
        $preflightPython = @(
            $preflightSnapshot |
                Where-Object {
                    (Test-IsPythonProcess -Process $_) -and
                    -not (Test-IsIgnoredControlProbe -Process $_)
                }
        )
        if ($preflightPython.Count -gt 0) {
            break
        }
        $elapsedStable = ((Get-Date) - $preflightStart).TotalSeconds
        if ($elapsedStable -ge $PreflightStableSeconds) {
            break
        }
        $remaining = $PreflightStableSeconds - $elapsedStable
        $sleepMilliseconds = [int](
            [math]::Max(
                100,
                [math]::Min($PollSeconds * 1000, $remaining * 1000)
            )
        )
        Start-Sleep -Milliseconds $sleepMilliseconds
    } while ($true)
    if ($preflightPython.Count -gt 0) {
        $refusal = [ordered]@{
            schema_version          = 1
            status                  = "refused_preflight"
            reason                  = "Python-family processes existed before the isolated command was started."
            time_utc                = (Get-Date).ToUniversalTime().ToString("o")
            wrapper_pid             = $PID
            output_label            = $OutputLabel
            poll_interval_seconds   = $PollSeconds
            requested_stable_seconds = $PreflightStableSeconds
            host                    = $hostRecord
            foreign_python_count    = $preflightPython.Count
            foreign_python_processes = @($preflightPython | ForEach-Object {
                Get-ProcessRecord -Process $_
            })
            ignored_control_probe_count = $IgnoredControlProbeIndex.Count
            ignored_control_probe_processes = @(
                Get-IgnoredControlProbeRecords `
                    -Index $IgnoredControlProbeIndex
            )
        }
        Write-JsonCreateNew -Path (Join-Path $RunDir "REFUSED.json") -Value $refusal
        $WrapperExitCode = 20
    } else {
        $preRunLoad = Get-SystemLoadRecord
        $scriptHashRecord = Get-HashedFileRecord -Path $ResolvedScript
        $inputHashRecords = @(
            $ResolvedInputFiles | ForEach-Object {
                Get-HashedFileRecord -Path $_
            }
        )
        $processArguments = @($ResolvedScript) + @($ArgumentList)
        $renderedArguments = ($processArguments | ForEach-Object {
            ConvertTo-WindowsCommandLineArgument -Value $_
        }) -join " "
        $RenderedCommand = (
            (ConvertTo-WindowsCommandLineArgument -Value $ResolvedPython) +
            " " +
            $renderedArguments
        )

        $RunStartUtc = (Get-Date).ToUniversalTime()
        $Runner = Start-Process `
            -FilePath $ResolvedPython `
            -ArgumentList $renderedArguments `
            -WorkingDirectory $RepoRoot `
            -WindowStyle Hidden `
            -RedirectStandardOutput $StdoutPath `
            -RedirectStandardError $StderrPath `
            -PassThru

        $runnerCreationTicks = [long]0
        try {
            $runnerCim = Get-CimInstance -ClassName Win32_Process `
                -Filter "ProcessId = $($Runner.Id)" `
                -ErrorAction SilentlyContinue
            if ($null -ne $runnerCim) {
                $runnerCreationTicks = Get-CreationTicks -Process $runnerCim
            }
        } catch {
            $runnerCreationTicks = [long]0
        }
        $KnownTree[[int]$Runner.Id] = $runnerCreationTicks

        $contamination = $null
        while ($true) {
            $snapshot = Get-AllProcessSnapshot
            $tree = @(Expand-RunnerTree `
                -Processes $snapshot `
                -Known $KnownTree `
                -RootPid ([int]$Runner.Id))
            $ignoredControlProbes = @(
                Get-IgnoredControlProbeProcesses `
                    -Processes $snapshot `
                    -TreeProcesses $tree
            )
            Add-IgnoredControlProbeRecords `
                -Index $IgnoredControlProbeIndex `
                -Processes $ignoredControlProbes
            $foreign = @(Get-ForeignPythonProcesses -Processes $snapshot -TreeProcesses $tree)
            if ($foreign.Count -gt 0) {
                $contamination = [ordered]@{
                    detected_time_utc = (Get-Date).ToUniversalTime().ToString("o")
                    processes = @($foreign | ForEach-Object {
                        Get-ProcessRecord -Process $_
                    })
                }
                break
            }

            $Runner.Refresh()
            if ($Runner.HasExited -and $tree.Count -eq 0) {
                break
            }
            Start-Sleep -Seconds $PollSeconds
        }

        if ($null -ne $contamination) {
            $terminated = @(Stop-RunnerTreeExact `
                -RootPid ([int]$Runner.Id) `
                -Known $KnownTree)
            try {
                [void]$Runner.WaitForExit(10000)
            } catch {
                # The exact process-tree stop record below remains authoritative.
            }
            $runnerExitCode = $null
            try {
                $Runner.Refresh()
                if ($Runner.HasExited) {
                    $runnerExitCode = [int]$Runner.ExitCode
                }
            } catch {
                $runnerExitCode = $null
            }
            $endUtc = (Get-Date).ToUniversalTime()
            $postRunBindingAudit = Get-PostRunBindingAudit `
                -Script $ResolvedScript `
                -Inputs $ResolvedInputFiles `
                -PreRunScript $scriptHashRecord `
                -PreRunInputs $inputHashRecords
            $contaminatedRecord = [ordered]@{
                schema_version        = 1
                status                = "contaminated"
                start_time_utc        = $RunStartUtc.ToString("o")
                end_time_utc          = $endUtc.ToString("o")
                wrapper_pid           = $PID
                runner_pid            = [int]$Runner.Id
                runner_exit_code      = $runnerExitCode
                output_label          = $OutputLabel
                host                  = $hostRecord
                preflight_stable_seconds = $PreflightStableSeconds
                pre_run_system_load   = $preRunLoad
                post_run_system_load  = Get-SystemLoadRecord
                command               = [ordered]@{
                    python_executable = $ResolvedPython
                    script_path       = $ResolvedScript
                    script            = $scriptHashRecord
                    input_files       = $inputHashRecords
                    argument_list     = @($ArgumentList)
                    rendered          = $RenderedCommand
                }
                expected_output_dir   = if ($null -eq $ResolvedExpectedOutput) {
                    $null
                } else {
                    (
                        Get-PortableRelativePath `
                            -BasePath $RepoRoot `
                            -TargetPath $ResolvedExpectedOutput
                    ).Replace([char]0x5C, [char]0x2F)
                }
                poll_interval_seconds = $PollSeconds
                source_input_integrity = $postRunBindingAudit
                foreign_python_count  = @($contamination.processes).Count
                foreign_python_processes = @($contamination.processes)
                ignored_control_probe_count = $IgnoredControlProbeIndex.Count
                ignored_control_probe_processes = @(
                    Get-IgnoredControlProbeRecords `
                        -Index $IgnoredControlProbeIndex
                )
                terminated_runner_tree_processes = $terminated
                logs                  = [ordered]@{
                    stdout_path = $StdoutPath
                    stderr_path = $StderrPath
                    hashes      = Get-LogHashRecord `
                        -StandardOutput $StdoutPath `
                        -StandardError $StderrPath
                }
            }
            $contaminatedPath = Join-Path $RunDir "CONTAMINATED.json"
            Write-JsonCreateNew -Path $contaminatedPath -Value $contaminatedRecord
            Write-DoNotUseMarker `
                -Directory $ResolvedExpectedOutput `
                -Reason "Foreign Python-family process detected during formal timing." `
                -IsolationRecord $contaminatedPath
            $WrapperExitCode = 21
        } else {
            $Runner.WaitForExit()
            $runnerExitCode = [int]$Runner.ExitCode
            $endUtc = (Get-Date).ToUniversalTime()
            $postRunBindingAudit = Get-PostRunBindingAudit `
                -Script $ResolvedScript `
                -Inputs $ResolvedInputFiles `
                -PreRunScript $scriptHashRecord `
                -PreRunInputs $inputHashRecords
            $resultRecord = $null
            if ($null -ne $ResolvedExpectedOutput -and $runnerExitCode -eq 0) {
                $resultRecord = Get-DirectoryHashRecord `
                    -Directory $ResolvedExpectedOutput
            }
            $baseRecord = [ordered]@{
                schema_version        = 1
                start_time_utc        = $RunStartUtc.ToString("o")
                end_time_utc          = $endUtc.ToString("o")
                duration_seconds      = [math]::Round(
                    ($endUtc - $RunStartUtc).TotalSeconds,
                    6
                )
                wrapper_pid           = $PID
                runner_pid            = [int]$Runner.Id
                runner_exit_code      = $runnerExitCode
                output_label          = $OutputLabel
                host                  = $hostRecord
                preflight_stable_seconds = $PreflightStableSeconds
                pre_run_system_load   = $preRunLoad
                post_run_system_load  = Get-SystemLoadRecord
                command               = [ordered]@{
                    python_executable = $ResolvedPython
                    script_path       = $ResolvedScript
                    script            = $scriptHashRecord
                    input_files       = $inputHashRecords
                    argument_list     = @($ArgumentList)
                    rendered          = $RenderedCommand
                }
                poll_interval_seconds = $PollSeconds
                source_input_integrity = $postRunBindingAudit
                foreign_python_count  = 0
                expected_output_dir   = if ($null -eq $ResolvedExpectedOutput) {
                    $null
                } else {
                    (
                        Get-PortableRelativePath `
                            -BasePath $RepoRoot `
                            -TargetPath $ResolvedExpectedOutput
                    ).Replace([char]0x5C, [char]0x2F)
                }
                ignored_control_probe_count = $IgnoredControlProbeIndex.Count
                ignored_control_probe_processes = @(
                    Get-IgnoredControlProbeRecords `
                        -Index $IgnoredControlProbeIndex
                )
                result_tree           = $resultRecord
                logs                  = [ordered]@{
                    stdout_path = $StdoutPath
                    stderr_path = $StderrPath
                    hashes      = Get-LogHashRecord `
                        -StandardOutput $StdoutPath `
                        -StandardError $StderrPath
                }
            }
            if (-not [bool]$postRunBindingAudit.unchanged) {
                $changedRecord = [ordered]@{}
                foreach ($entry in $baseRecord.GetEnumerator()) {
                    $changedRecord[$entry.Key] = $entry.Value
                }
                $changedRecord["status"] = "source_or_declared_input_changed"
                $changedPath = Join-Path $RunDir "SOURCE_INPUTS_CHANGED.json"
                Write-JsonCreateNew `
                    -Path $changedPath `
                    -Value $changedRecord
                Write-DoNotUseMarker `
                    -Directory $ResolvedExpectedOutput `
                    -Reason (
                        "The source script or a declared input changed " +
                        "during the formal run."
                    ) `
                    -IsolationRecord $changedPath
                $WrapperExitCode = 22
            } elseif ($runnerExitCode -eq 0) {
                $cleanRecord = [ordered]@{}
                foreach ($entry in $baseRecord.GetEnumerator()) {
                    $cleanRecord[$entry.Key] = $entry.Value
                }
                $cleanRecord["status"] = "clean_run"
                $cleanPath = Join-Path $RunDir "CLEAN_RUN.json"
                Write-JsonCreateNew -Path $cleanPath -Value $cleanRecord
                if ($null -ne $resultRecord) {
                    $cleanHash = (
                        Get-FileHash -LiteralPath $cleanPath -Algorithm SHA256
                    ).Hash.ToLowerInvariant()
                    $sidecar = [ordered]@{
                        schema_version = 1
                        status = "admissible_formal_result"
                        output_label = $OutputLabel
                        runner_exit_code = 0
                        foreign_python_count = 0
                        ignored_control_probe_count = $IgnoredControlProbeIndex.Count
                        preflight_stable_seconds = $PreflightStableSeconds
                        expected_output_dir = $baseRecord.expected_output_dir
                        argument_list = @($ArgumentList)
                        result_tree_relative_to_repo = (
                            $resultRecord.path_relative_to_repo
                        )
                        result_tree_sha256 = $resultRecord.tree_sha256
                        result_file_count = $resultRecord.file_count
                        source_script = $scriptHashRecord
                        input_files = $inputHashRecords
                        source_inputs_unchanged = $true
                        post_run_source_script = (
                            $postRunBindingAudit.source_script
                        )
                        post_run_input_files = @(
                            $postRunBindingAudit.input_files
                        )
                        clean_record_relative_to_repo = (
                            Get-PortableRelativePath `
                                -BasePath $RepoRoot `
                                -TargetPath $cleanPath
                        ).Replace([char]0x5C, [char]0x2F)
                        clean_record_sha256 = $cleanHash
                    }
                    Write-JsonCreateNew `
                        -Path (Join-Path $ResolvedExpectedOutput "FORMAL_PROVENANCE.json") `
                        -Value $sidecar
                }
                $WrapperExitCode = 0
            } else {
                $failedRecord = [ordered]@{}
                foreach ($entry in $baseRecord.GetEnumerator()) {
                    $failedRecord[$entry.Key] = $entry.Value
                }
                $failedRecord["status"] = "clean_interval_failed_program"
                $programFailedPath = Join-Path $RunDir "PROGRAM_FAILED.json"
                Write-JsonCreateNew `
                    -Path $programFailedPath `
                    -Value $failedRecord
                Write-DoNotUseMarker `
                    -Directory $ResolvedExpectedOutput `
                    -Reason "Target program exited with code $runnerExitCode." `
                    -IsolationRecord $programFailedPath
                $WrapperExitCode = $runnerExitCode
            }
        }
    }
} catch {
    $failureTime = (Get-Date).ToUniversalTime()
    $terminatedOnFailure = @()
    if ($null -ne $Runner -and $KnownTree.Count -gt 0) {
        try {
            $terminatedOnFailure = @(Stop-RunnerTreeExact `
                -RootPid ([int]$Runner.Id) `
                -Known $KnownTree)
        } catch {
            $terminatedOnFailure = @()
        }
    }
    if ($null -ne $RunDir -and (Test-Path -LiteralPath $RunDir -PathType Container)) {
        $failurePath = Join-Path $RunDir "FAILED.json"
        if (-not (Test-Path -LiteralPath $failurePath)) {
            $failureRecord = [ordered]@{
                schema_version = 1
                status = "wrapper_failed"
                time_utc = $failureTime.ToString("o")
                wrapper_pid = $PID
                runner_pid = if ($null -eq $Runner) { $null } else { [int]$Runner.Id }
                command = $RenderedCommand
                error = $_.Exception.Message
                terminated_runner_tree_processes = $terminatedOnFailure
                logs = Get-LogHashRecord `
                    -StandardOutput $StdoutPath `
                    -StandardError $StderrPath
            }
            Write-JsonCreateNew -Path $failurePath -Value $failureRecord
            Write-DoNotUseMarker `
                -Directory $ResolvedExpectedOutput `
                -Reason ("Isolation wrapper failed: " + $_.Exception.Message) `
                -IsolationRecord $failurePath
        }
    }
    Write-Error -Message ("Formal isolation wrapper failed: " + $_.Exception.Message) `
        -ErrorAction Continue
    $WrapperExitCode = 99
}

if ($null -ne $RunDir) {
    Write-Host "Formal isolation record: $RunDir"
}
exit $WrapperExitCode
