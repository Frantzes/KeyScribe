param(
    [string]$Target = "x86_64-pc-windows-msvc",
    [switch]$SkipCargoBuild,
    [string]$OutputRoot = "build/windows",
    [switch]$PersonalUpdate,
    [switch]$SkipZip
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

    $bundleBinaryName = "keyscribe.exe"
    $binaryCandidates = @(
        (Join-Path $repoRoot "target/$Target/release/$bundleBinaryName"),
        (Join-Path $repoRoot "target/release/$bundleBinaryName")
    )

    $binaryPath = $null
    foreach ($candidate in $binaryCandidates) {
        if (Test-Path $candidate) {
            $binaryPath = $candidate
            break
        }
    }

    if (-not $binaryPath) {
        throw "Could not find $bundleBinaryName in target/$Target/release or target/release"
    }

    $bundleName = "keyscribe-windows-x64"
    $bundleDir = Join-Path $repoRoot "$OutputRoot/$bundleName"
    $modelsDir = Join-Path $bundleDir "models"

    if ($PersonalUpdate) {
        Write-Host "Personal update mode enabled: preserving existing files in $bundleDir"
    } elseif (Test-Path $bundleDir) {
        Remove-Item -Path $bundleDir -Recurse -Force
    }
    New-Item -ItemType Directory -Path $modelsDir -Force | Out-Null

    $bundleBinaryPath = Join-Path $bundleDir $bundleBinaryName
    try {
        Copy-Item -Path $binaryPath -Destination $bundleBinaryPath -Force -ErrorAction Stop
    } catch {
        if (-not $PersonalUpdate) {
            throw
        }

        $stagedBinaryPath = Join-Path $bundleDir "keyscribe.update.exe"
        Copy-Item -Path $binaryPath -Destination $stagedBinaryPath -Force

        $swapScriptPath = Join-Path $bundleDir "apply-update.cmd"
        Set-Content -Path $swapScriptPath -Encoding ASCII -Value @"
@echo off
setlocal
echo Applying Keyscribe update...
copy /Y "keyscribe.update.exe" "keyscribe.exe" >nul
if errorlevel 1 (
  echo Update failed. Make sure Keyscribe is fully closed, then run this again.
  exit /b 1
)
del "keyscribe.update.exe" >nul 2>nul
echo Update applied successfully.
exit /b 0
"@

        Write-Warning "Could not overwrite running keyscribe.exe. Staged update as keyscribe.update.exe."
        Write-Warning "Close Keyscribe, then run apply-update.cmd in $bundleDir to finish replacing the executable."
    }

    $modelSourceDir = Join-Path $repoRoot "models"
    $modelFiles = Get-ChildItem -Path $modelSourceDir -Filter "*.onnx" -File -ErrorAction Stop
    if (-not $modelFiles -or $modelFiles.Count -eq 0) {
        throw "Missing model files in models/"
    }
    foreach ($modelFile in $modelFiles) {
        Copy-Item -Path $modelFile.FullName -Destination (Join-Path $modelsDir $modelFile.Name) -Force
    }

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
        Write-Warning "DirectML.dll not found. If the app fails to start on another machine, include DirectML.dll next to keyscribe.exe."
    }

    $bundleReadmePath = Join-Path $bundleDir "README-portable.txt"
    Set-Content -Path $bundleReadmePath -Encoding UTF8 -Value @"
KeyScribe portable Windows bundle

Contents:
- keyscribe.exe
- models/*.onnx
- DirectML.dll (if discovered)

Run keyscribe.exe from this folder so the relative model path works.
"@

    $shouldZip = -not $SkipZip -and -not $PersonalUpdate
    if ($shouldZip) {
        $zipPath = Join-Path $repoRoot "$OutputRoot/$bundleName.zip"
        if (Test-Path $zipPath) {
            Remove-Item -Path $zipPath -Force
        }

        Compress-Archive -Path (Join-Path $bundleDir "*") -DestinationPath $zipPath -CompressionLevel Optimal
        Write-Host "Portable bundle zip:       $zipPath"
    } else {
        Write-Host "Portable zip generation skipped."
    }

    Write-Host "Portable bundle directory: $bundleDir"
}
finally {
    Pop-Location
}
