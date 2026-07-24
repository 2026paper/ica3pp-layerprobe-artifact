[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$checksumPath = Join-Path $root "SHA256SUMS.txt"
$manifestPath = Join-Path $root "MANIFEST.json"

if (-not (Test-Path -LiteralPath $checksumPath -PathType Leaf)) {
    throw "Missing SHA256SUMS.txt."
}
if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
    throw "Missing MANIFEST.json."
}

$checked = 0
foreach ($line in [IO.File]::ReadAllLines($checksumPath)) {
    if ([string]::IsNullOrWhiteSpace($line)) {
        continue
    }
    if ($line -notmatch "^([0-9a-f]{64})  (.+)$") {
        throw "Malformed checksum line: $line"
    }
    $expected = $Matches[1]
    $relative = $Matches[2]
    if ([IO.Path]::IsPathRooted($relative) -or $relative -match "(^|/)\.\.(/|$)") {
        throw "Unsafe checksum path: $relative"
    }
    $path = Join-Path $root ($relative.Replace("/", "\"))
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Missing payload file: $relative"
    }
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash.ToLowerInvariant()
    if ($actual -cne $expected) {
        throw "Checksum mismatch: $relative"
    }
    $checked += 1
}

Write-Output "Package checksum verification: PASS ($checked files)"
