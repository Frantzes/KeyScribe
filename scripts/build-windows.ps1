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

    # --- Download ONNX models if missing ---
    $modelSourceDir = Join-Path $repoRoot "models"
    New-Item -ItemType Directory -Path $modelSourceDir -Force | Out-Null

    $assetBase = "https://github.com/Frantzes/KeyScribe/releases/download/assets-v1"
    $requiredModels = @("htdemucs_6s.onnx", "beat_this_small.onnx", "mel_spectrogram.onnx")

    foreach ($modelName in $requiredModels) {
        $modelPath = Join-Path $modelSourceDir $modelName
        if (-not (Test-Path $modelPath)) {
            Write-Host "Downloading $modelName from GitHub Releases..."
            Invoke-WebRequest -Uri "$assetBase/$modelName" -OutFile $modelPath -UseBasicParsing
        }
    }

    $modelFiles = @(Get-ChildItem -Path $modelSourceDir -Filter "*.onnx" -File -ErrorAction Stop)
    if ($modelFiles.Count -eq 0) {
        throw "Missing model files in models/"
    }
    foreach ($modelFile in $modelFiles) {
        Copy-Item -Path $modelFile.FullName -Destination (Join-Path $modelsDir $modelFile.Name) -Force
    }

    # --- ONNX Runtime CPU DLLs (for development / CPU fallback) ---
    $ortCpuVendorDir = Join-Path $repoRoot "vendor\onnxruntime"
    $ortCpuBase = "https://github.com/Frantzes/KeyScribe/releases/download/assets-v1"
    $ortCpuDlls = @("onnxruntime.dll", "DirectML.dll")

    foreach ($dll in $ortCpuDlls) {
        $dllPath = Join-Path $ortCpuVendorDir $dll
        if (-not (Test-Path $dllPath)) {
            Write-Host "Downloading $dll from GitHub Releases..."
            New-Item -ItemType Directory -Path $ortCpuVendorDir -Force | Out-Null
            Invoke-WebRequest -Uri "$ortCpuBase/$dll" -OutFile $dllPath -UseBasicParsing
        }
    }

    # --- CUDA / cuDNN Bundling ---
    # The CUDA execution provider (onnxruntime_providers_cuda.dll) is downloaded
    # by ort-sys as part of the cu12 prebuilt archive and linked statically into
    # keyscribe.exe. At runtime it needs the CUDA 12 runtime DLLs and cuDNN 9
    # DLLs on the search path. We bundle them next to the executable and preload
    # them at startup (see src/demucs.rs preload_cuda_dylibs) so users get GPU
    # acceleration without installing the CUDA toolkit themselves.
    $cudaDllNames = @(
        "cudart64_12.dll",
        "cublas64_12.dll",
        "cublasLt64_12.dll",
        "cufft64_11.dll",
        "curand64_10.dll",
        "nvrtc64_120_0.dll"
    )
    $cudnnDllNames = @(
        "cudnn64_9.dll",
        "cudnn_graph64_9.dll",
        "cudnn_ops64_9.dll",
        "cudnn_heuristic64_9.dll",
        "cudnn_adv64_9.dll",
        "cudnn_cnn64_9.dll",
        "cudnn_engines_precompiled64_9.dll",
        "cudnn_engines_runtime_compiled64_9.dll"
    )

    # 1) CUDA runtime DLLs: copy from a local CUDA toolkit install if present.
    $cudaToolkitDirs = @(
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\bin",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin"
    )
    $cudaToolkitBin = $null
    foreach ($dir in $cudaToolkitDirs) {
        if (Test-Path (Join-Path $dir "cudart64_12.dll")) {
            $cudaToolkitBin = $dir
            break
        }
    }
    if ($cudaToolkitBin) {
        Write-Host "CUDA runtime found in: $cudaToolkitBin"
        foreach ($dll in $cudaDllNames) {
            $src = Join-Path $cudaToolkitBin $dll
            if (Test-Path $src) {
                Copy-Item -Path $src -Destination (Join-Path $bundleDir $dll) -Force
            }
        }
        Write-Host "Bundled CUDA runtime DLLs"
    } else {
        Write-Warning "CUDA 12 toolkit not found in default install paths. CUDA GPU acceleration will be unavailable; ensure cudart64_12.dll etc. are on PATH at runtime."
    }

    # 2) cuDNN 9 DLLs: download from NVIDIA if not already cached in vendor/cudnn.
    $cudnnVendorDir = Join-Path $repoRoot "vendor\cudnn"
    $cudnnReady = $false
    foreach ($dll in $cudnnDllNames) {
        if (Test-Path (Join-Path $cudnnVendorDir $dll)) { $cudnnReady = $true; break }
    }
    if (-not $cudnnReady) {
        Write-Host "cuDNN 9 DLLs not found in vendor/cudnn. Downloading..."
        New-Item -ItemType Directory -Path $cudnnVendorDir -Force | Out-Null
        # NVIDIA cuDNN 9.3 for CUDA 12 (Windows x86_64) local installer.
        # This is the public redist mirror; the archive is a self-extracting zip.
        $cudnnUrl = "https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/windows-x86_64/cudnn-windows-x86_64-9.3.0.75_cuda12-archive.zip"
        $cudnnZip = Join-Path $cudnnVendorDir "cudnn.zip"
        try {
            Invoke-WebRequest -Uri $cudnnUrl -OutFile $cudnnZip -UseBasicParsing
            Write-Host "Extracting cuDNN..."
            $extractTmp = Join-Path $cudnnVendorDir "extract"
            Expand-Archive -Path $cudnnZip -DestinationPath $extractTmp -Force
            # The archive layout is cudnn-windows-x86_64-9.x.x_cuda12-archive\bin\*.dll
            $cudnnBin = Get-ChildItem -Path $extractTmp -Recurse -Directory -Filter "bin" |
                Where-Object { (Get-ChildItem $_.FullName -Filter "cudnn64_9.dll").Count -gt 0 } |
                Select-Object -First 1
            if ($cudnnBin) {
                foreach ($dll in $cudnnDllNames) {
                    $src = Join-Path $cudnnBin.FullName $dll
                    if (Test-Path $src) {
                        Copy-Item -Path $src -Destination (Join-Path $cudnnVendorDir $dll) -Force
                    }
                }
                Write-Host "Cached cuDNN 9 DLLs in vendor/cudnn"
            } else {
                Write-Warning "Could not locate cuDNN bin directory in archive. GPU acceleration may be unavailable."
            }
            Remove-Item -Path $extractTmp -Recurse -Force -ErrorAction SilentlyContinue
        } catch {
            Write-Warning "Failed to download cuDNN: $_. GPU acceleration will be unavailable unless cuDNN 9 DLLs are on PATH."
        } finally {
            Remove-Item -Path $cudnnZip -Force -ErrorAction SilentlyContinue
        }
    } else {
        Write-Host "cuDNN 9 DLLs already cached in vendor/cudnn"
    }

    # Copy cuDNN DLLs into the bundle.
    foreach ($dll in $cudnnDllNames) {
        $src = Join-Path $cudnnVendorDir $dll
        if (Test-Path $src) {
            Copy-Item -Path $src -Destination (Join-Path $bundleDir $dll) -Force
        }
    }

    # 3) Download the official ONNX Runtime GPU build (onnxruntime.dll +
    #    onnxruntime_providers_cuda.dll) from the onnxruntime-gpu pip wheel.
    #    We use ORT 1.24.2 which requires CUDA 12 + cuDNN 9 (matching our
    #    bundled DLLs). The ort crate loads these dynamically at runtime via
    #    the `load-dynamic` feature.
    $ortVendorDir = Join-Path $repoRoot "vendor\ort-gpu"
    $ortDllNames = @(
        "onnxruntime.dll",
        "onnxruntime_providers_cuda.dll",
        "onnxruntime_providers_shared.dll"
    )
    $ortReady = $true
    foreach ($dll in $ortDllNames) {
        if (-not (Test-Path (Join-Path $ortVendorDir $dll))) { $ortReady = $false; break }
    }
    if (-not $ortReady) {
        Write-Host "ONNX Runtime GPU DLLs not found in vendor/ort-gpu. Downloading ORT 1.24.2 GPU wheel..."
        New-Item -ItemType Directory -Path $ortVendorDir -Force | Out-Null
        $downloadSuccess = $false

        # Method 1: Try pip download from the repo's venv (has correct Python version).
        $venvPip = Join-Path $repoRoot ".venv\Scripts\python.exe"
        if (Test-Path $venvPip) {
            Write-Host "  Trying pip download via venv..."
            try {
                & { $ErrorActionPreference = "Continue"; & $venvPip -m pip download onnxruntime-gpu==1.24.2 --no-deps -d $ortVendorDir 2>&1 | Out-Null }
                $whlFile = Get-ChildItem -Path $ortVendorDir -Filter "*.whl" -File | Select-Object -First 1
                if ($whlFile) {
                    $downloadSuccess = $true
                }
            } catch {
                Write-Warning "  pip download via venv failed: $_"
            }
        }

        # Method 2: Try system pip/python.
        if (-not $downloadSuccess) {
            $sysPip = (Get-Command pip -ErrorAction SilentlyContinue).Source
            if (-not $sysPip) { $sysPip = (Get-Command python -ErrorAction SilentlyContinue).Source }
            if ($sysPip) {
                Write-Host "  Trying pip download via system Python..."
                try {
                    & { $ErrorActionPreference = "Continue"; & $sysPip -m pip download onnxruntime-gpu==1.24.2 --no-deps -d $ortVendorDir 2>&1 | Out-Null }
                    $whlFile = Get-ChildItem -Path $ortVendorDir -Filter "*.whl" -File | Select-Object -First 1
                    if ($whlFile) {
                        $downloadSuccess = $true
                    }
                } catch {
                    Write-Warning "  pip download via system Python failed: $_"
                }
            }
        }

        # Method 3: Direct download from PyPI file URLs.
        if (-not $downloadSuccess) {
            Write-Host "  Trying direct download from PyPI (via JSON API)..."
            try {
                $pypiApi = "https://pypi.org/pypi/onnxruntime-gpu/1.24.2/json"
                $pkg = Invoke-RestMethod -Uri $pypiApi -UseBasicParsing
                $wheels = $pkg.urls | Where-Object {
                    $_.url -like "*win_amd64*" -and $_.packagetype -eq "bdist_wheel"
                }
                if ($wheels) {
                    $dl = $wheels[0]
                    $whlPath = Join-Path $ortVendorDir $dl.filename
                    Invoke-WebRequest -Uri $dl.url -OutFile $whlPath -UseBasicParsing
                    $whlFile = Get-Item $whlPath
                    $downloadSuccess = $true
                    Write-Host "  Downloaded $($dl.filename)"
                }
            } catch {
                Write-Warning "  PyPI JSON API failed: $_"
            }
        }

        if ($downloadSuccess -and $whlFile) {
            # .whl is a zip — copy to .zip and extract
            $zipFile = $whlFile.FullName + ".zip"
            Copy-Item -Path $whlFile.FullName -Destination $zipFile
            $extractDir = Join-Path $ortVendorDir "extract"
            Expand-Archive -Path $zipFile -DestinationPath $extractDir -Force
            $capiDir = Join-Path $extractDir "onnxruntime\capi"
            if (Test-Path $capiDir) {
                foreach ($dll in $ortDllNames) {
                    $src = Join-Path $capiDir $dll
                    if (Test-Path $src) {
                        Copy-Item -Path $src -Destination (Join-Path $ortVendorDir $dll) -Force
                    }
                }
                Write-Host "  Cached ORT 1.24.2 GPU DLLs in vendor/ort-gpu"
            } else {
                Write-Warning "  Could not find onnxruntime/capi in wheel. GPU acceleration will be unavailable."
            }
            Remove-Item -Path $extractDir -Recurse -Force -ErrorAction SilentlyContinue
            Remove-Item -Path $zipFile -Force -ErrorAction SilentlyContinue
            Remove-Item -Path $whlFile.FullName -Force -ErrorAction SilentlyContinue
        } else {
            Write-Warning "  All download methods failed. GPU acceleration will be unavailable."
            Write-Warning "  To fix manually: pip download onnxruntime-gpu==1.24.2 --no-deps, extract the .whl, and copy the DLLs from onnxruntime/capi/ to vendor/ort-gpu/"
        }
    } else {
        Write-Host "ORT GPU DLLs already cached in vendor/ort-gpu"
    }

    # Copy ORT DLLs into the bundle.
    foreach ($dll in $ortDllNames) {
        $src = Join-Path $ortVendorDir $dll
        if (Test-Path $src) {
            Copy-Item -Path $src -Destination (Join-Path $bundleDir $dll) -Force
        }
    }

    # --- FFmpeg Bundling ---
    $ffmpegDir = Join-Path $repoRoot "vendor/ffmpeg"
    $ffmpegExePath = Join-Path $ffmpegDir "ffmpeg.exe"
    
    if (-not (Test-Path $ffmpegExePath)) {
        Write-Host "FFmpeg not found in vendor/ffmpeg. Downloading static build..."
        New-Item -ItemType Directory -Path $ffmpegDir -Force | Out-Null
        
        $ffmpegZip = Join-Path $ffmpegDir "ffmpeg.zip"
        # Using the BtbN GPL static build from the aggregator site (points to github releases)
        $ffmpegUrl = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
        
        Invoke-WebRequest -Uri $ffmpegUrl -OutFile $ffmpegZip
        
        Write-Host "Extracting FFmpeg..."
        Expand-Archive -Path $ffmpegZip -DestinationPath $ffmpegDir -Force
        
        # The zip contains a subfolder like ffmpeg-7.1-essentials_build/bin/ffmpeg.exe
        $extractedExe = Get-ChildItem -Path $ffmpegDir -Filter "ffmpeg.exe" -Recurse | Select-Object -First 1
        if ($extractedExe) {
            Move-Item -Path $extractedExe.FullName -Destination $ffmpegExePath -Force
        }
        
        Remove-Item -Path $ffmpegZip -Force
        # Clean up the extra folders from the zip
        Get-ChildItem -Path $ffmpegDir -Directory | Remove-Item -Recurse -Force
    }

    if (Test-Path $ffmpegExePath) {
        Copy-Item -Path $ffmpegExePath -Destination (Join-Path $bundleDir "ffmpeg.exe") -Force
        Write-Host "Included ffmpeg.exe from: $ffmpegExePath"
    } else {
        Write-Warning "Failed to prepare ffmpeg.exe. Video features may not work."
    }

    $bundleReadmePath = Join-Path $bundleDir "README-portable.txt"
    Set-Content -Path $bundleReadmePath -Encoding UTF8 -Value @"
KeyScribe portable Windows bundle

Contents:
- keyscribe.exe
- ffmpeg.exe
- models/*.onnx (basic-pitch, htdemucs_6s, mel_spectrogram, beat_this_small)
- CUDA 12 runtime + cuDNN 9 DLLs + onnxruntime_providers_cuda.dll (GPU accel)

All AI inference (note detection, stem separation, beat tracking) runs
in-process via ONNX Runtime - no Python or external runtime required.
GPU acceleration requires an NVIDIA GPU with CUDA-capable drivers.

Run keyscribe.exe from this folder so the relative model, ffmpeg, and
CUDA/cuDNN DLL paths work.
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
