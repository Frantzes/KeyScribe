use std::path::Path;
use std::process::Command;

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeatTrackResult {
    pub beats: Vec<f32>,
    pub downbeats: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct BeatTrackConfig {
    pub model: String,
    pub device: BeatTrackDevice,
    pub dbn: bool,
}

impl Default for BeatTrackConfig {
    fn default() -> Self {
        Self {
            model: "final0".to_string(),
            device: BeatTrackDevice::Auto,
            dbn: false,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum BeatTrackDevice {
    Auto,
    Cpu,
    Cuda,
}

impl BeatTrackDevice {
    fn as_str(self) -> &'static str {
        match self {
            BeatTrackDevice::Auto => "auto",
            BeatTrackDevice::Cpu => "cpu",
            BeatTrackDevice::Cuda => "cuda",
        }
    }
}

pub fn run_beat_this(audio_path: &Path, config: &BeatTrackConfig) -> Result<BeatTrackResult> {
    if !audio_path.is_file() {
        return Err(anyhow!(
            "Beat tracking input is not a file: {}",
            audio_path.display()
        ));
    }

    let python_exe = if cfg!(windows) {
        "python/src/.venv/Scripts/python.exe"
    } else {
        "python/src/.venv/bin/python"
    };
    let runner_script = "python/src/beat_this_runner.py";

    let mut cmd = Command::new(python_exe);
    cmd.arg(runner_script)
        .arg(audio_path)
        .arg("--model")
        .arg(&config.model)
        .arg("--device")
        .arg(config.device.as_str());

    if config.dbn {
        cmd.arg("--dbn");
    }

    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x08000000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }

    let output = cmd.output().context("BeatThis process failed to start")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow!(
            "BeatThis process failed with status {:?}: {}",
            output.status.code(),
            stderr.trim()
        ));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let parsed: BeatTrackResult =
        serde_json::from_str(stdout.trim()).context("Failed to parse BeatThis JSON output")?;
    Ok(parsed)
}
