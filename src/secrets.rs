//! Secure-at-rest storage for small secrets (the MVSep API key).
//!
//! The key is never written to disk in plaintext:
//!
//! - **Android**: AES-256-GCM using a key generated inside the hardware-backed
//!   Android Keystore (see `MainActivity.storeSecret`). The key material never
//!   leaves the Keystore, and the ciphertext lives in the app-private files dir.
//! - **Windows**: DPAPI (`CryptProtectData`), which encrypts with a key derived
//!   from the current user's login credentials — no key file is stored.
//! - **Other platforms**: no OS keystore integration yet; callers fall back to
//!   the legacy user `.env` file.
//!
//! The API intentionally only exposes `store`/`load`/`clear`; callers should
//! treat failures as "no secure store available" and fall back, never as a
//! reason to write the secret in plaintext unconditionally.

use anyhow::Result;

/// Persist `key` using the platform keystore. An empty key clears it.
pub fn store(key: &str) -> Result<()> {
    let key = key.trim();
    if key.is_empty() {
        return clear();
    }
    platform::store(key)
}

/// Remove the stored secret, if any.
pub fn clear() -> Result<()> {
    platform::clear()
}

/// Load the stored secret, or `None` when absent/unavailable.
pub fn load() -> Option<String> {
    platform::load()
}

#[cfg(target_os = "android")]
mod platform {
    use anyhow::{anyhow, Result};

    pub fn store(key: &str) -> Result<()> {
        if crate::android::store_secret(key) {
            Ok(())
        } else {
            Err(anyhow!("Android Keystore store failed"))
        }
    }

    pub fn clear() -> Result<()> {
        if crate::android::clear_secret() {
            Ok(())
        } else {
            Err(anyhow!("Android Keystore clear failed"))
        }
    }

    pub fn load() -> Option<String> {
        crate::android::load_secret()
    }
}

#[cfg(all(windows, not(target_os = "android")))]
mod platform {
    use anyhow::{anyhow, Result};
    use std::path::PathBuf;
    use windows_sys::Win32::Foundation::LocalFree;
    use windows_sys::Win32::Security::Cryptography::{
        CryptProtectData, CryptUnprotectData, CRYPT_INTEGER_BLOB,
    };

    fn secret_path() -> Result<PathBuf> {
        let dir = directories::ProjectDirs::from("com", "Frantzes", "KeyScribe")
            .map(|d| d.data_local_dir().to_path_buf())
            .ok_or_else(|| anyhow!("could not locate the user data directory"))?;
        std::fs::create_dir_all(&dir)?;
        Ok(dir.join("mvsep.key.dpapi"))
    }

    /// Run DPAPI over `data`. `protect = true` encrypts, `false` decrypts.
    fn dpapi(data: &[u8], protect: bool) -> Result<Vec<u8>> {
        unsafe {
            let mut input = CRYPT_INTEGER_BLOB {
                cbData: data.len() as u32,
                pbData: data.as_ptr() as *mut u8,
            };
            let mut output = CRYPT_INTEGER_BLOB {
                cbData: 0,
                pbData: std::ptr::null_mut(),
            };

            let ok = if protect {
                CryptProtectData(
                    &mut input,
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    0,
                    &mut output,
                )
            } else {
                CryptUnprotectData(
                    &mut input,
                    std::ptr::null_mut(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    0,
                    &mut output,
                )
            };

            if ok == 0 {
                return Err(anyhow!("DPAPI call failed"));
            }

            let result = std::slice::from_raw_parts(output.pbData, output.cbData as usize).to_vec();
            LocalFree(output.pbData as *mut _);
            Ok(result)
        }
    }

    pub fn store(key: &str) -> Result<()> {
        let blob = dpapi(key.as_bytes(), true)?;
        std::fs::write(secret_path()?, blob)?;
        Ok(())
    }

    pub fn clear() -> Result<()> {
        let path = secret_path()?;
        if path.exists() {
            std::fs::remove_file(path)?;
        }
        Ok(())
    }

    pub fn load() -> Option<String> {
        let blob = std::fs::read(secret_path().ok()?).ok()?;
        let data = dpapi(&blob, false).ok()?;
        String::from_utf8(data).ok().filter(|s| !s.is_empty())
    }

    #[cfg(test)]
    mod tests {
        use super::dpapi;

        #[test]
        fn dpapi_roundtrip() {
            let plaintext = b"hello-secret-123";
            let encrypted = dpapi(plaintext, true).expect("encrypt");
            assert_ne!(&encrypted[..], &plaintext[..], "ciphertext must differ");
            let decrypted = dpapi(&encrypted, false).expect("decrypt");
            assert_eq!(&decrypted[..], &plaintext[..]);
        }
    }
}

#[cfg(not(any(windows, target_os = "android")))]
mod platform {
    use anyhow::{anyhow, Result};

    pub fn store(_key: &str) -> Result<()> {
        Err(anyhow!("no OS keystore integration on this platform"))
    }

    pub fn clear() -> Result<()> {
        Ok(())
    }

    pub fn load() -> Option<String> {
        None
    }
}
