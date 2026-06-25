param(
    [switch]$SkipModels,
    [switch]$SkipOrt
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir "..")

Push-Location $repoRoot
try {
    # --- Models ---
    if (-not $SkipModels) {
        $modelsDir = Join-Path $repoRoot "models"
        New-Item -ItemType Directory -Path $modelsDir -Force | Out-Null

        # mel_spectrogram.onnx and beat_this_small.onnx are committed to beat-this-rs repo
        # Download them from the raw GitHub URLs.
        $beatThisTag = "0.3.0"
        $beatThisBase = "https://raw.githubusercontent.com/danigb/beat-this-rs/refs/tags/v$beatThisTag/models"

        $modelFiles = @(
            @{Name="mel_spectrogram.onnx"; Url="$beatThisBase/mel_spectrogram.onnx"},
            @{Name="beat_this_small.onnx"; Url="$beatThisBase/beat_this_small.onnx"}
        )

        foreach ($m in $modelFiles) {
            $outPath = Join-Path $modelsDir $m.Name
            if (-not (Test-Path $outPath)) {
                Write-Host "Downloading $($m.Name)..."
                Invoke-WebRequest -Uri $m.Url -OutFile $outPath -UseBasicParsing
            } else {
                Write-Host "$($m.Name) already exists, skipping"
            }
        }

        Write-Host ""
        Write-Host "=== htdemucs_6s ONNX models ==="
        Write-Host "The htdemucs_6s ONNX models are too large to distribute via git."
        Write-Host "To use stem separation, export the model from Python:"
        Write-Host "  1. Install demucs: pip install demucs"
        Write-Host "  2. Export to ONNX: python scripts/export_demucs_onnx.py"
        Write-Host "  Or download pre-exported models from the project's Releases page."
        Write-Host ""
        Write-Host "Place the following files in $modelsDir :"
        Write-Host "  - htdemucs_6s.onnx  (fp32, ~270 MB, best quality)"
        Write-Host "  - htdemucs_6s_faster_(fp16weights).onnx  (fp16, ~136 MB, faster)"
    }

    # --- ONNX Runtime CPU DLLs (Windows) ---
    if (-not $SkipOrt -and $IsWindows) {
        $ortVendorDir = Join-Path $repoRoot "vendor\onnxruntime"
        $ortDlls = @("onnxruntime.dll", "DirectML.dll")

        $allExist = $true
        foreach ($dll in $ortDlls) {
            if (-not (Test-Path (Join-Path $ortVendorDir $dll))) { $allExist = $false; break }
        }

        if (-not $allExist) {
            Write-Host "Downloading ONNX Runtime CPU + DirectML DLLs..."
            New-Item -ItemType Directory -Path $ortVendorDir -Force | Out-Null

            # Download ONNX Runtime CPU NuGet package (includes DirectML for Windows)
            $ortVersion = "1.23.1"
            $nugetUrl = "https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime/$ortVersion"
            $nupkgPath = Join-Path $ortVendorDir "onnxruntime.nupkg"

            Invoke-WebRequest -Uri $nugetUrl -OutFile $nupkgPath -UseBasicParsing

            # Rename .nupkg to .zip and extract
            $zipPath = $nupkgPath + ".zip"
            Rename-Item -Path $nupkgPath -NewName (Split-Path $zipPath -Leaf) -Force
            $extractDir = Join-Path $ortVendorDir "extract"
            Expand-Archive -Path $zipPath -DestinationPath $extractDir -Force

            # Copy the DLLs from the native NuGet folder
            $nativeDir = Join-Path $extractDir "runtimes\win-x64\native"
            if (Test-Path $nativeDir) {
                foreach ($dll in $ortDlls) {
                    $src = Join-Path $nativeDir $dll
                    if (Test-Path $src) {
                        Copy-Item -Path $src -Destination (Join-Path $ortVendorDir $dll) -Force
                        Write-Host "  Copied $dll"
                    }
                }
            }

            # Cleanup
            Remove-Item -Path $extractDir -Recurse -Force -ErrorAction SilentlyContinue
            Remove-Item -Path $zipPath -Force -ErrorAction SilentlyContinue
        } else {
            Write-Host "ONNX Runtime DLLs already exist in vendor/onnxruntime, skipping"
        }
    }

    Write-Host ""
    Write-Host "Setup complete."
}
finally {
    Pop-Location
}
