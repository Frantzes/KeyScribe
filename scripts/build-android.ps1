# Build and (optionally) install the Android APK.
#
# cargo-apk cannot package Java code (it hardcodes android:hasCode="false"),
# but the native document picker needs a small Java MainActivity to receive
# `onActivityResult` (android-activity 0.5 has no activity-result API). This
# script therefore builds the Rust cdylib with cargo and assembles the APK
# itself with aapt2 / d8 / zipalign / apksigner.
param(
    [switch]$Install,
    [string]$Abi = "arm64-v8a",
    [string]$RustTarget = "aarch64-linux-android",
    [int]$MinSdk = 26,
    [int]$TargetSdk = 35
)

# Native tools (cargo, javac, d8) write progress to stderr; with EAP=Stop
# PowerShell turns that into a terminating NativeCommandError. Rely on
# $LASTEXITCODE instead.
$ErrorActionPreference = "Continue"
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
$repo = Split-Path -Parent $PSScriptRoot

$sdk = if ($env:ANDROID_HOME) { $env:ANDROID_HOME } else { "$env:LOCALAPPDATA\Android\Sdk" }
$ndk = if ($env:ANDROID_NDK_ROOT) {
    $env:ANDROID_NDK_ROOT
} else {
    (Get-ChildItem "$sdk\ndk" | Sort-Object Name -Descending | Select-Object -First 1).FullName
}
$buildTools = (Get-ChildItem "$sdk\build-tools" | Sort-Object Name -Descending | Select-Object -First 1).FullName
$platform = (Get-ChildItem "$sdk\platforms" | Sort-Object Name -Descending | Select-Object -First 1).FullName
$androidJar = Join-Path $platform "android.jar"
$jdk = if ($env:JAVA_HOME) { $env:JAVA_HOME } else { "C:\Program Files\Android\Android Studio\jbr" }
$toolchain = "$ndk\toolchains\llvm\prebuilt\windows-x86_64"

$env:ANDROID_HOME = $sdk
$env:ANDROID_NDK_ROOT = $ndk
$env:JAVA_HOME = $jdk
$env:PATH = "$jdk\bin;$env:PATH"
$env:CC_aarch64_linux_android = "$toolchain\bin\aarch64-linux-android$MinSdk-clang.cmd"
$env:CXX_aarch64_linux_android = "$toolchain\bin\aarch64-linux-android$MinSdk-clang++.cmd"
$env:AR_aarch64_linux_android = "$toolchain\bin\llvm-ar.exe"
$env:CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER = "$toolchain\bin\aarch64-linux-android$MinSdk-clang.cmd"

$work = "$repo\target\android"

Write-Host "[1/6] Building Rust cdylib ($RustTarget)..."
cargo build --release --lib --target $RustTarget --no-default-features --features android
if ($LASTEXITCODE -ne 0) { throw "cargo build failed" }
$rustSo = "$repo\target\$RustTarget\release\libkeyscribe_lib.so"

Write-Host "[2/6] Compiling Java..."
$classesDir = "$work\classes"
Remove-Item $classesDir -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $classesDir | Out-Null
$javaFiles = (Get-ChildItem -Recurse -Filter *.java "$repo\android\java").FullName
& "$jdk\bin\javac.exe" --release 8 -nowarn -classpath $androidJar -d $classesDir $javaFiles
if ($LASTEXITCODE -ne 0) { throw "javac failed" }

Write-Host "[3/6] Dexing..."
$dexDir = "$work\dex"
Remove-Item $dexDir -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $dexDir | Out-Null
$classFiles = (Get-ChildItem -Recurse -Filter *.class $classesDir).FullName
& "$buildTools\d8.bat" --lib $androidJar --min-api $MinSdk --output $dexDir $classFiles
if ($LASTEXITCODE -ne 0) { throw "d8 failed" }

Write-Host "[4/6] Compiling + linking resources/manifest..."
$compiledRes = "$work\res.zip"
Remove-Item $compiledRes -Force -ErrorAction SilentlyContinue
& "$buildTools\aapt2.exe" compile --dir "$repo\android\res" -o $compiledRes
if ($LASTEXITCODE -ne 0) { throw "aapt2 compile failed" }

$unsigned = "$work\keyscribe-unsigned.apk"
Remove-Item $unsigned -Force -ErrorAction SilentlyContinue
& "$buildTools\aapt2.exe" link `
    -o $unsigned `
    --manifest "$repo\android\AndroidManifest.xml" `
    -I $androidJar `
    -A "$repo\android\assets" `
    -R $compiledRes `
    --min-sdk-version $MinSdk `
    --target-sdk-version $TargetSdk
if ($LASTEXITCODE -ne 0) { throw "aapt2 link failed" }

Write-Host "[5/6] Packaging native libraries and dex..."
$zip = [System.IO.Compression.ZipFile]::Open($unsigned, [System.IO.Compression.ZipArchiveMode]::Update)
function Add-Entry($archive, [string]$src, [string]$entry, [bool]$store) {
    $zipEntry = $archive.CreateEntry(
        $entry,
        $(if ($store) { [System.IO.Compression.CompressionLevel]::NoCompression }
          else { [System.IO.Compression.CompressionLevel]::Optimal })
    )
    $stream = $zipEntry.Open()
    $bytes = [System.IO.File]::ReadAllBytes($src)
    $stream.Write($bytes, 0, $bytes.Length)
    $stream.Dispose()
}
# Native libs are compressed: extractNativeLibs="true" lets Android unpack
# them at install time, which roughly halves the APK size.
Add-Entry $zip "$dexDir\classes.dex" "classes.dex" $false
Add-Entry $zip $rustSo "lib/$Abi/libkeyscribe_lib.so" $false
Add-Entry $zip "$repo\android\libs\$Abi\libonnxruntime.so" "lib/$Abi/libonnxruntime.so" $false
Add-Entry $zip "$toolchain\sysroot\usr\lib\$RustTarget\libc++_shared.so" "lib/$Abi/libc++_shared.so" $false
$zip.Dispose()

$aligned = "$work\keyscribe-aligned.apk"
Remove-Item $aligned -Force -ErrorAction SilentlyContinue
& "$buildTools\zipalign.exe" -p -f 4 $unsigned $aligned
if ($LASTEXITCODE -ne 0) { throw "zipalign failed" }

Write-Host "[6/6] Signing..."
$apk = "$work\keyscribe.apk"
Remove-Item $apk -Force -ErrorAction SilentlyContinue
$keystore = "$env:USERPROFILE\.android\debug.keystore"
& "$buildTools\apksigner.bat" sign `
    --ks $keystore `
    --ks-key-alias androiddebugkey `
    --ks-pass pass:android `
    --key-pass pass:android `
    --out $apk `
    $aligned
if ($LASTEXITCODE -ne 0) { throw "apksigner failed" }

Write-Host "APK: $apk ($([math]::Round((Get-Item $apk).Length / 1MB, 1)) MB)"

if ($Install) {
    $adb = "$sdk\platform-tools\adb.exe"
    # MIUI/some OEM ROMs reject `adb install` with INSTALL_FAILED_USER_RESTRICTED
    # unless the package verifier is disabled.
    & $adb shell settings put global verifier_verify_adb_installs 0 | Out-Null
    & $adb shell settings put global package_verifier_enable 0 | Out-Null
    & $adb install --no-streaming -r $apk
    if ($LASTEXITCODE -ne 0) {
        throw "adb install failed (exit $LASTEXITCODE). On MIUI enable 'Install via USB' / confirm on-device, then re-run."
    }
}
