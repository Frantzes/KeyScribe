param(
    [string]$Target = "x86_64-pc-windows-msvc",
    [switch]$SkipCargoBuild,
    [string]$OutputRoot = "build/windows"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir "..")

Push-Location $repoRoot
try {
    if (-not $SkipCargoBuild) {
        Write-Host "Building release binary for $Target..."
        cargo build --release --target $Target
        if ($LASTEXITCODE -ne 0) {
            throw "cargo build failed with exit code $LASTEXITCODE"
        }
    }

    $binaryName = "transcriber.exe"
    $binaryCandidates = @(
        (Join-Path $repoRoot "target/$Target/release/$binaryName"),
        (Join-Path $repoRoot "target/release/$binaryName")
    )

    $binaryPath = $null
    foreach ($candidate in $binaryCandidates) {
        if (Test-Path $candidate) {
            $binaryPath = $candidate
            break
        }
    }

    if (-not $binaryPath) {
        throw "Could not find $binaryName in target/$Target/release or target/release"
    }

    $bundleName = "transcriber-windows-x64"
    $bundleDir = Join-Path $repoRoot "$OutputRoot/$bundleName"
    $modelsDir = Join-Path $bundleDir "models"

    if (Test-Path $bundleDir) {
        Remove-Item -Path $bundleDir -Recurse -Force
    }
    New-Item -ItemType Directory -Path $modelsDir -Force | Out-Null

    Copy-Item -Path $binaryPath -Destination (Join-Path $bundleDir $binaryName) -Force

    $modelPath = Join-Path $repoRoot "models/basic-pitch.onnx"
    if (-not (Test-Path $modelPath)) {
        throw "Missing model file: models/basic-pitch.onnx"
    }
    Copy-Item -Path $modelPath -Destination (Join-Path $modelsDir "basic-pitch.onnx") -Force

    $directMlCandidates = @(
        (Join-Path $repoRoot "target/$Target/release/DirectML.dll"),
        (Join-Path $repoRoot "target/release/DirectML.dll"),
        (Join-Path $repoRoot "target/debug/DirectML.dll")
    )

    $directMlPath = $null
    foreach ($candidate in $directMlCandidates) {
        if (Test-Path $candidate) {
            $directMlPath = $candidate
            break
        }
    }

    if (-not $directMlPath) {
        $directMlPath = Get-ChildItem -Path (Join-Path $repoRoot "target") -Filter "DirectML.dll" -Recurse -File -ErrorAction SilentlyContinue |
            Select-Object -First 1 -ExpandProperty FullName
    }

    if ($directMlPath) {
        Copy-Item -Path $directMlPath -Destination (Join-Path $bundleDir "DirectML.dll") -Force
        Write-Host "Included DirectML.dll from: $directMlPath"
    } else {
        Write-Warning "DirectML.dll not found. If the app fails to start on another machine, include DirectML.dll next to transcriber.exe."
    }

    $bundleReadmePath = Join-Path $bundleDir "README-portable.txt"
    Set-Content -Path $bundleReadmePath -Encoding UTF8 -Value @"
Audio Transcriber portable Windows bundle

Contents:
- transcriber.exe
- models/basic-pitch.onnx
- DirectML.dll (if discovered)

Run transcriber.exe from this folder so the relative model path works.
"@

    $zipPath = Join-Path $repoRoot "$OutputRoot/$bundleName.zip"
    if (Test-Path $zipPath) {
        Remove-Item -Path $zipPath -Force
    }

    Compress-Archive -Path (Join-Path $bundleDir "*") -DestinationPath $zipPath -CompressionLevel Optimal

    Write-Host "Portable bundle directory: $bundleDir"
    Write-Host "Portable bundle zip:       $zipPath"
}
finally {
    Pop-Location
}
