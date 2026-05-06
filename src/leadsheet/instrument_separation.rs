use anyhow::{anyhow, Result};
use std::borrow::Cow;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct SeparationConfig {
    pub model_name: String,
    pub song_hash: Option<String>,
    pub source_path: Option<PathBuf>,
    pub cache_dir: Option<PathBuf>,
}

impl Default for SeparationConfig {
    fn default() -> Self {
        Self {
            model_name: "htdemucs_6s".to_string(),
            song_hash: None,
            source_path: None,
            cache_dir: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum StemType {
    Vocals,
    Drums,
    Bass,
    Piano,
    Guitar,
    Other,
    Custom(String),
}

impl StemType {
    pub fn from_label(label: impl AsRef<str>) -> Self {
        let label = label.as_ref().trim();
        if label.is_empty() {
            return StemType::Other;
        }

        let lower = label.to_ascii_lowercase();
        if lower.contains("vocal") || lower.contains("voice") || lower.contains("lead") {
            StemType::Vocals
        } else if lower.contains("drum") || lower.contains("perc") || lower.contains("beat") {
            StemType::Drums
        } else if lower.contains("bass") {
            StemType::Bass
        } else if lower.contains("piano") {
            StemType::Piano
        } else if lower.contains("guitar") {
            StemType::Guitar
        } else if lower.contains("other")
            || lower.contains("backing")
            || lower.contains("accomp")
            || lower.contains("instrumental")
            || lower.contains("pad")
            || lower.contains("strings")
            || lower.contains("synth")
            || lower.contains("melody")
        {
            StemType::Other
        } else {
            StemType::Custom(label.to_string())
        }
    }

    pub fn display_name(&self) -> Cow<'_, str> {
        match self {
            StemType::Vocals => Cow::Borrowed("Vocals"),
            StemType::Drums => Cow::Borrowed("Drums"),
            StemType::Bass => Cow::Borrowed("Bass"),
            StemType::Piano => Cow::Borrowed("Piano"),
            StemType::Guitar => Cow::Borrowed("Guitar"),
            StemType::Other => Cow::Borrowed("Other"),
            StemType::Custom(label) => Cow::Borrowed(label.as_str()),
        }
    }

    pub fn is_melodic(&self) -> bool {
        match self {
            StemType::Vocals | StemType::Other | StemType::Piano | StemType::Guitar => true,
            StemType::Bass | StemType::Drums => false,
            StemType::Custom(label) => {
                let lower = label.to_ascii_lowercase();
                !(lower.contains("bass")
                    || lower.contains("drum")
                    || lower.contains("perc")
                    || lower.contains("beat"))
            }
        }
    }
}

pub struct InstrumentSeparator {
    config: SeparationConfig,
}

impl InstrumentSeparator {
    pub fn new(config: SeparationConfig) -> Result<Self> {
        Ok(Self { config })
    }

    pub fn separate(
        &mut self,
        _audio: &[f32],
        _channels: usize,
        _sample_rate: u32,
        progress_callback: Option<Box<dyn Fn(f32) + Send + Sync>>,
    ) -> Result<Vec<SeparatedStem>> {
        let song_hash = self.config.song_hash.as_ref().ok_or_else(|| anyhow!("Song hash required for separation"))?;
        let source_path = self.config.source_path.as_ref().ok_or_else(|| anyhow!("Source path required for separation"))?;
        let cache_dir = self.config.cache_dir.as_ref().ok_or_else(|| anyhow!("Cache directory required for separation"))?;

        // Construct a unique stem cache directory for this song and model
        let stem_cache_root = cache_dir.join("stems").join(song_hash).join(&self.config.model_name);
        
        // Check if we already have stems in the cache
        if stem_cache_root.exists() {
            let stems = self.load_stems_from_dir(&stem_cache_root)?;
            if !stems.is_empty() {
                println!("[DEBUG] Loaded stems from cache: {:?}", stem_cache_root);
                if let Some(ref cb) = progress_callback {
                    cb(1.0);
                }
                return Ok(stems);
            }
        }

        std::fs::create_dir_all(&stem_cache_root)?;

        let python_exe = if cfg!(windows) {
            "python/src/.venv/Scripts/python.exe"
        } else {
            "python/src/.venv/bin/python"
        };

        let runner_script = "python/src/demucs_runner.py";

        println!("[DEBUG] Running demucs via python: {} {} ...", python_exe, runner_script);

        if let Some(ref cb) = progress_callback {
            cb(0.1); // Indication that we started
        }

        let mut cmd = std::process::Command::new(python_exe);
        cmd.arg(runner_script)
            .arg(source_path)
            .arg("-o")
            .arg(&stem_cache_root)
            .arg("-m")
            .arg(&self.config.model_name)
            .arg("-f")
            .arg("mp3")
            .arg("-b")
            .arg("320")
            .arg("--device")
            .arg("auto");

        // Hide console window on Windows
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            const CREATE_NO_WINDOW: u32 = 0x08000000;
            cmd.creation_flags(CREATE_NO_WINDOW);
        }

        let status = cmd.status()?;
        if !status.success() {
            return Err(anyhow!("Demucs process failed with status: {:?}", status));
        }

        // Demucs outputs to: <output_dir>/<model_name>/<filename_no_ext>/<stem>.mp3
        let filename = source_path.file_stem().and_then(|s| s.to_str()).unwrap_or("input");
        let output_stems_dir = stem_cache_root.join(&self.config.model_name).join(filename);

        let stems = self.load_stems_from_dir(&output_stems_dir)?;

        if stems.is_empty() {
            return Err(anyhow!("No stems were generated by Demucs"));
        }

        // Move stems from the demucs-created subfolders to our root stem_cache_root for easier subsequent loading
        for stem in &stems {
            let stem_filename = format!("{}.mp3", stem.stem_type.display_name().to_lowercase());
            let src = output_stems_dir.join(&stem_filename);
            let dst = stem_cache_root.join(&stem_filename);
            if src.exists() {
                let _ = std::fs::rename(src, dst);
            }
        }
        
        // Clean up demucs nested folders
        let _ = std::fs::remove_dir_all(stem_cache_root.join(&self.config.model_name));

        if let Some(ref cb) = progress_callback {
            cb(1.0);
        }

        Ok(stems)
    }

    fn load_stems_from_dir(&self, dir: &Path) -> Result<Vec<SeparatedStem>> {
        let mut stems = Vec::new();
        if !dir.exists() {
            return Ok(stems);
        }

        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("mp3") {
                let stem_name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("unknown");
                let audio = crate::audio_io::load_audio_file(&path)?;
                
                stems.push(SeparatedStem {
                    stem_type: StemType::from_label(stem_name),
                    samples_mono: Arc::new(audio.samples_mono.to_vec()),
                    samples_interleaved: Arc::new(audio.samples_interleaved.to_vec()),
                    channels: audio.channels,
                    sample_rate: audio.sample_rate,
                    confidence: 0.9,
                });
            }
        }

        // Sort stems for consistent UI order
        stems.sort_by_key(|s| match s.stem_type {
            StemType::Drums => 0,
            StemType::Bass => 1,
            StemType::Other => 2,
            StemType::Vocals => 3,
            StemType::Piano => 4,
            StemType::Guitar => 5,
            _ => 6,
        });

        Ok(stems)
    }
}

use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct SeparatedStem {
    pub stem_type: StemType,
    pub samples_mono: Arc<Vec<f32>>,
    pub samples_interleaved: Arc<Vec<f32>>,
    pub channels: u16,
    pub sample_rate: u32,
    pub confidence: f32,
}

pub fn extract_melodic_audio(stems: &[SeparatedStem], melodic_stem: &StemType) -> Option<Arc<Vec<f32>>> {
    stems
        .iter()
        .find(|stem| &stem.stem_type == melodic_stem)
        .map(|stem| Arc::clone(&stem.samples_mono))
}

pub fn blend_for_chords(stems: &[SeparatedStem]) -> Arc<Vec<f32>> {
    if stems.is_empty() {
        return Arc::new(Vec::new());
    }
    
    if stems.len() == 1 {
        return Arc::clone(&stems[0].samples_mono);
    }

    let total_mono_len = stems.iter().map(|stem| stem.samples_mono.len()).max().unwrap_or(0);
    if total_mono_len == 0 {
        return Arc::new(Vec::new());
    }

    let mut blend = vec![0.0f32; total_mono_len];
    for stem in stems {
        let audio = &stem.samples_mono;
        let len = audio.len().min(total_mono_len);
        for i in 0..len {
            blend[i] += audio[i];
        }
    }

    let stem_count = stems.len() as f32;
    if stem_count > 1.0 {
        let inv_count = 1.0 / stem_count;
        for sample in blend.iter_mut() {
            *sample *= inv_count;
        }
    }
    Arc::new(blend)
}

pub fn blend_interleaved_stems(stems: &[SeparatedStem]) -> (Arc<Vec<f32>>, u16) {
    if stems.is_empty() {
        return (Arc::new(Vec::new()), 1);
    }

    if stems.len() == 1 {
        return (Arc::clone(&stems[0].samples_interleaved), stems[0].channels);
    }

    let channels = stems.iter().map(|s| s.channels).max().unwrap_or(1);
    let total_frames = stems.iter().map(|s| s.samples_interleaved.len() / s.channels as usize).max().unwrap_or(0);
    
    let mut blend = vec![0.0f32; total_frames * channels as usize];
    for stem in stems {
        let stem_frames = stem.samples_interleaved.len() / stem.channels as usize;
        let frames_to_copy = stem_frames.min(total_frames);
        
        if stem.channels == channels {
            for i in 0..frames_to_copy * channels as usize {
                blend[i] += stem.samples_interleaved[i];
            }
        } else if stem.channels == 1 && channels == 2 {
            for f in 0..frames_to_copy {
                let s = stem.samples_interleaved[f];
                blend[f * 2] += s;
                blend[f * 2 + 1] += s;
            }
        } else {
            for f in 0..frames_to_copy {
                let s = stem.samples_mono[f];
                for ch in 0..channels as usize {
                    blend[f * channels as usize + ch] += s;
                }
            }
        }
    }

    let stem_count = stems.len() as f32;
    if stem_count > 1.0 {
        let inv_count = 1.0 / stem_count;
        for sample in blend.iter_mut() {
            *sample *= inv_count;
        }
    }
    (Arc::new(blend), channels)
}
