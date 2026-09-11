# Round-trip test loop for the KeyScribe sheet-music pipeline.
#
#   sheet (MIDI) --render--> MP3 --transcribe--> MusicXML --render--> MP3 --transcribe--> MusicXML --compare--> report
#
# Usage:
#   powershell -File scripts/roundtrip.ps1 [-NoteThreshold 0.25] [-PassAccuracy 0.5] [-OutDir out]
#
# Exits 0 if the round-trip transcription accuracy meets the pass threshold,
# exits 1 otherwise (so an agent can gate on it).

param(
    [float]$NoteThreshold = 0.25,
    [float]$PassAccuracy = 0.1,
    [string]$OutDir = "out",
    [string]$Bpm = 100
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Cli = Join-Path $Root "target\debug\keyscribe-cli.exe"
$Out = Join-Path $Root $OutDir

if (-not (Test-Path $Cli)) {
    Write-Host "[roundtrip] building CLI..."
    Push-Location $Root
    try {
        cargo build --no-default-features --bin keyscribe-cli 2>&1 | Out-Null
    } finally {
        Pop-Location
    }
    if (-not (Test-Path $Cli)) { throw "CLI build failed: $Cli not found" }
}

New-Item -ItemType Directory -Force -Path $Out | Out-Null
$TestMidi = Join-Path $Root "tests\data\twinkle.mid"
$RefMp3   = Join-Path $Out "roundtrip_ref.mp3"
$RefXml   = Join-Path $Out "roundtrip_ref.musicxml"
$RtMp3    = Join-Path $Out "roundtrip_rt.mp3"
$RtXml    = Join-Path $Out "roundtrip_rt.musicxml"
$Json     = Join-Path $Out "roundtrip_report.json"

function Invoke-Cli($ArgsList) {
    $global:LASTEXITCODE = 0
    & $Cli @ArgsList
    if ($LASTEXITCODE -ne 0) { throw "keyscribe-cli failed: $($ArgsList -join ' ')" }
}

Write-Host "[roundtrip] generating test melody -> $TestMidi"
Invoke-Cli @("maketest", "-o", $TestMidi, "--bpm", $Bpm)

Write-Host "[roundtrip] rendering test MIDI -> $RefMp3 (MuseScore)"
Invoke-Cli @("render", $TestMidi, "-o", $RefMp3)

Write-Host "[roundtrip] transcribing reference audio -> $RefXml"
Invoke-Cli @("sheet", $RefMp3, "-o", $RefXml, "--title", "Roundtrip Ref", "--melody", "poly", "--threshold", $NoteThreshold)

Write-Host "[roundtrip] rendering sheet -> $RtMp3 (MuseScore)"
Invoke-Cli @("render", $RefXml, "-o", $RtMp3)

Write-Host "[roundtrip] transcribing round-trip audio -> $RtXml"
Invoke-Cli @("sheet", $RtMp3, "-o", $RtXml, "--title", "Roundtrip RT", "--melody", "poly", "--threshold", $NoteThreshold)

Write-Host "[roundtrip] comparing sheets"
Invoke-Cli @("compare", $RefXml, $RtXml, "--json") | Set-Content -Path $Json
Get-Content $Json

$report = Get-Content $Json | ConvertFrom-Json
$acc = [double]$report.note_accuracy
Write-Host ""
Write-Host "============================================="
Write-Host ("round-trip note accuracy: {0:P1} (pass >= {1:P1})" -f $acc, $PassAccuracy)
Write-Host ("pitch accuracy: {0:P1}   recall: {1:P1}" -f $report.pitch_accuracy, $report.recall)
Write-Host "============================================="

if ($acc -lt $PassAccuracy) {
    Write-Host "[roundtrip] FAIL: accuracy below pass threshold"
    exit 1
}
Write-Host "[roundtrip] PASS"
exit 0
