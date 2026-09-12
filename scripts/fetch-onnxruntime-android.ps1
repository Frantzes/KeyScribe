# Download the official ONNX Runtime Android AAR and extract the arm64-v8a
# shared library into android/libs/, where cargo-apk packages it into the APK.
#
# The version must match the ONNX Runtime API version the `ort` crate is built
# against. `ort 2.0.0-rc.12` uses `api-24`, i.e. ONNX Runtime 1.24.x.
param(
    [string]$Version = "1.24.2",
    [string]$Abi = "arm64-v8a"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot

$vendor = Join-Path $repo "vendor\onnxruntime-android"
New-Item -ItemType Directory -Force -Path $vendor | Out-Null

$aarName = "onnxruntime-android-$Version.aar"
$aar = Join-Path $vendor $aarName
$url = "https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime-android/$Version/$aarName"

if (-not (Test-Path $aar)) {
    Write-Host "Downloading $url"
    $ProgressPreference = "SilentlyContinue"
    Invoke-WebRequest -UseBasicParsing $url -OutFile $aar
}

$outDir = Join-Path $repo "android\libs\$Abi"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip = [System.IO.Compression.ZipFile]::OpenRead($aar)
try {
    $entry = $zip.Entries |
        Where-Object { $_.FullName -eq "jni/$Abi/libonnxruntime.so" } |
        Select-Object -First 1
    if (-not $entry) {
        throw "jni/$Abi/libonnxruntime.so not found in $aarName"
    }
    $dest = Join-Path $outDir "libonnxruntime.so"
    [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $dest, $true)
    Write-Host "Extracted $dest"
}
finally {
    $zip.Dispose()
}
