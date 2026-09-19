param(
    [string]$Target = "x86_64-pc-windows-msvc",
    [switch]$SkipCargoBuild,
    [string]$OutputRoot = "build/windows",
    [switch]$PersonalUpdate,
    [switch]$SkipZip
)

$ErrorActionPreference = "Continue"

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

    # --- Core ONNX models (small, required for transcription) ---
    # htdemucs_6s.onnx is NOT bundled: it is downloaded on first local Demucs
    # use (default separation is MVSep cloud). FFmpeg and the GPU pack
    # (CUDA/cuDNN/ORT CUDA provider) are downloaded on demand too. See
    # src/assets.rs and README.md ("Runtime downloads").
    $modelSourceDir = Join-Path $repoRoot "models"
    New-Item -ItemType Directory -Path $modelSourceDir -Force | Out-Null

    $assetBase = "https://github.com/Frantzes/KeyScribe/releases/download/assets-v1"
    $requiredModels = @("beat_this_small.onnx", "mel_spectrogram.onnx", "basic-pitch.onnx")

    foreach ($modelName in $requiredModels) {
        $modelPath = Join-Path $modelSourceDir $modelName
        if (-not (Test-Path $modelPath)) {
            Write-Host "Downloading $modelName from GitHub Releases..."
            Invoke-WebRequest -Uri "$assetBase/$modelName" -OutFile $modelPath -UseBasicParsing
        }
    }

    $bundledModels = @(
        "basic-pitch.onnx",
        "beat_this_small.onnx",
        "mel_spectrogram.onnx",
        "melody_quantizer.onnx",
        "melody_quantizer.onnx.data",
        "melody_quantizer_v2_seq.onnx"
    )
    foreach ($modelName in $bundledModels) {
        $src = Join-Path $modelSourceDir $modelName
        if (Test-Path $src) {
            Copy-Item -Path $src -Destination (Join-Path $modelsDir $modelName) -Force
        } elseif ($modelName -in $requiredModels) {
            throw "Missing required model $modelName in models/"
        }
    }

    # --- ONNX Runtime core (pinned Microsoft PyPI wheel) ---
    # Provides onnxruntime.dll (CPU inference + host for the CUDA provider) and
    # the shared provider loader. The CUDA provider itself ships in the
    # on-demand GPU pack, pinned to the same 1.24.4 wheel in src/assets.rs.
    $ortWheelUrl = "https://files.pythonhosted.org/packages/fa/bc/35f3a37226d7a28c84b8b456f52237ccd39eb7111114bcf9ac340178e1ec/onnxruntime_gpu-1.24.4-cp313-cp313-win_amd64.whl"
    $ortWheelSha = "6be8bf2048777c517fca33eb61e114969fa326619feaa789d8c75f24337ea762"
    $ortVendorDir = Join-Path $repoRoot "vendor\ort-core"
    $ortWheelPath = Join-Path $ortVendorDir "onnxruntime_gpu-1.24.4-cp313-cp313-win_amd64.whl"
    $ortCapiDir = Join-Path $ortVendorDir "capi"

    if (-not (Test-Path (Join-Path $ortCapiDir "onnxruntime.dll"))) {
        New-Item -ItemType Directory -Path $ortVendorDir -Force | Out-Null
        if (-not (Test-Path $ortWheelPath)) {
            Write-Host "Downloading ONNX Runtime 1.24.4 wheel (pinned, verified)..."
            Invoke-WebRequest -Uri $ortWheelUrl -OutFile $ortWheelPath -UseBasicParsing
        }
        $actualSha = (Get-FileHash $ortWheelPath -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actualSha -ne $ortWheelSha) {
            Remove-Item $ortWheelPath -Force -ErrorAction SilentlyContinue
            throw "ONNX Runtime wheel checksum mismatch: expected $ortWheelSha, got $actualSha"
        }
        $zipCopy = "$ortWheelPath.zip"
        Copy-Item $ortWheelPath $zipCopy -Force
        $extractDir = Join-Path $ortVendorDir "extract"
        Remove-Item $extractDir -Recurse -Force -ErrorAction SilentlyContinue
        Expand-Archive -Path $zipCopy -DestinationPath $extractDir -Force
        New-Item -ItemType Directory -Path $ortCapiDir -Force | Out-Null
        foreach ($dll in @("onnxruntime.dll", "onnxruntime_providers_shared.dll")) {
            $src = Join-Path $extractDir "onnxruntime\capi\$dll"
            if (Test-Path $src) {
                Copy-Item -Path $src -Destination (Join-Path $ortCapiDir $dll) -Force
            }
        }
        Remove-Item $extractDir -Recurse -Force -ErrorAction SilentlyContinue
        Remove-Item $zipCopy -Force -ErrorAction SilentlyContinue
    } else {
        Write-Host "ONNX Runtime core already cached in vendor/ort-core"
    }

    foreach ($dll in @("onnxruntime.dll", "onnxruntime_providers_shared.dll")) {
        $src = Join-Path $ortCapiDir $dll
        if (-not (Test-Path $src)) {
            throw "Missing $dll in vendor/ort-core (delete the folder and rebuild)"
        }
        Copy-Item -Path $src -Destination (Join-Path $bundleDir $dll) -Force
    }

    $bundleReadmePath = Join-Path $bundleDir "README-portable.txt"
    Set-Content -Path $bundleReadmePath -Encoding UTF8 -Value @"
KeyScribe portable Windows bundle

Contents:
- keyscribe.exe
- onnxruntime.dll + onnxruntime_providers_shared.dll (ONNX Runtime core)
- models/*.onnx (basic-pitch, beat_this_small, mel_spectrogram)

Downloaded automatically on first use (no action needed):
- Demucs htdemucs_6s model - only when you run stem separation with a local
  Demucs model (the default MVSep separation runs in the cloud)
- FFmpeg - only when a file needs it (unsupported audio format or video)
- GPU pack (CUDA 12 + cuDNN 9 + ONNX Runtime CUDA provider) - only when you
  run local Demucs separation on a machine with an NVIDIA GPU

All AI inference (note detection, stem separation, beat tracking) runs
in-process via ONNX Runtime - no Python or external runtime required.
GPU acceleration requires an NVIDIA GPU with CUDA-capable drivers; without
it stem separation falls back to CPU.

Run keyscribe.exe from this folder so relative model and DLL paths work.
"@

    $shouldZip = -not $SkipZip -and -not $PersonalUpdate
    if ($shouldZip) {
        $zipPath = Join-Path $repoRoot "$OutputRoot/$bundleName.zip"
        if (Test-Path $zipPath) {
            Remove-Item -Path $zipPath -Force
        }

        try {
            Compress-Archive -Path (Join-Path $bundleDir "*") -DestinationPath $zipPath -CompressionLevel Optimal -ErrorAction Stop
        } catch {
            Write-Warning "Failed to create zip: $_"
        }
        Write-Host "Portable bundle zip:       $zipPath"
    } else {
        Write-Host "Portable zip generation skipped."
    }

    Write-Host "Portable bundle directory: $bundleDir"
}
finally {
    Pop-Location
}

exit 0
