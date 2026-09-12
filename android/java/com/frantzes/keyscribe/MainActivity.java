package com.frantzes.keyscribe;

import android.app.NativeActivity;
import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.provider.OpenableColumns;
import android.security.keystore.KeyGenParameterSpec;
import android.security.keystore.KeyProperties;
import android.util.Log;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.security.KeyStore;
import javax.crypto.Cipher;
import javax.crypto.KeyGenerator;
import javax.crypto.SecretKey;
import javax.crypto.spec.GCMParameterSpec;

/**
 * Thin NativeActivity subclass that exists solely to receive the result of the
 * system document picker.
 *
 * android-activity 0.5 (pinned by winit 0.29 / eframe 0.27) has no
 * activity-result API, and the NDK ANativeActivity callbacks have no
 * onActivityResult hook, so a small Java class is the only way to get the URI
 * back. The native glue still loads `libkeyscribe_lib.so` and calls
 * `android_main`, exactly as with a plain `android.app.NativeActivity`.
 */
public class MainActivity extends NativeActivity {
    private static final int REQUEST_PICK_AUDIO = 0x4b53;

    private static final String TAG = "KeyScribe";

    static {
        // NativeActivity loads libkeyscribe_lib.so under the boot class loader,
        // which does not expose its native methods to this (app) class. Loading
        // it here first associates it with MainActivity's class loader so
        // nativeOnAudioPicked resolves. The later NativeActivity load is a
        // no-op by name.
        System.loadLibrary("keyscribe_lib");
    }

    private static String pendingClipboard = null;

    /** Implemented in Rust (`src/android.rs`). */
    private static native void nativeOnAudioPicked(String path);

    // --- Secure secret storage (Android Keystore AES-256-GCM) -------------

    private static final String SECRET_KEY_ALIAS = "keyscribe_mvsep_key";
    private static final String SECRET_FILE = "mvsep.key";
    private static final int GCM_IV_BYTES = 12;
    private static final int GCM_TAG_BITS = 128;

    /** Returns the Keystore AES key, generating it on first use. */
    private SecretKey secretKey() throws Exception {
        KeyStore keyStore = KeyStore.getInstance("AndroidKeyStore");
        keyStore.load(null);
        java.security.Key existing = keyStore.getKey(SECRET_KEY_ALIAS, null);
        if (existing instanceof SecretKey) {
            return (SecretKey) existing;
        }
        KeyGenerator generator =
            KeyGenerator.getInstance(KeyProperties.KEY_ALGORITHM_AES, "AndroidKeyStore");
        KeyGenParameterSpec spec = new KeyGenParameterSpec.Builder(
                SECRET_KEY_ALIAS,
                KeyProperties.PURPOSE_ENCRYPT | KeyProperties.PURPOSE_DECRYPT)
            .setBlockModes(KeyProperties.BLOCK_MODE_GCM)
            .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE)
            .setKeySize(256)
            .build();
        generator.init(spec);
        return generator.generateKey();
    }

    /** Encrypts {@code value} with the Keystore key (IV || ciphertext). */
    public boolean storeSecret(String value) {
        try {
            Cipher cipher = Cipher.getInstance("AES/GCM/NoPadding");
            cipher.init(Cipher.ENCRYPT_MODE, secretKey());
            byte[] iv = cipher.getIV();
            byte[] ciphertext = cipher.doFinal(value.getBytes(StandardCharsets.UTF_8));
            byte[] blob = new byte[iv.length + ciphertext.length];
            System.arraycopy(iv, 0, blob, 0, iv.length);
            System.arraycopy(ciphertext, 0, blob, iv.length, ciphertext.length);
            try (FileOutputStream out =
                     new FileOutputStream(new File(getFilesDir(), SECRET_FILE))) {
                out.write(blob);
            }
            return true;
        } catch (Throwable t) {
            Log.w(TAG, "storeSecret failed: " + t);
            return false;
        }
    }

    /** Decrypts and returns the stored secret, or an empty string. */
    public String loadSecret() {
        try {
            File file = new File(getFilesDir(), SECRET_FILE);
            if (!file.isFile()) {
                return "";
            }
            byte[] blob = Files.readAllBytes(file.toPath());
            if (blob.length <= GCM_IV_BYTES) {
                return "";
            }
            byte[] iv = java.util.Arrays.copyOfRange(blob, 0, GCM_IV_BYTES);
            byte[] ciphertext =
                java.util.Arrays.copyOfRange(blob, GCM_IV_BYTES, blob.length);
            Cipher cipher = Cipher.getInstance("AES/GCM/NoPadding");
            cipher.init(
                Cipher.DECRYPT_MODE, secretKey(), new GCMParameterSpec(GCM_TAG_BITS, iv));
            byte[] plaintext = cipher.doFinal(ciphertext);
            return new String(plaintext, StandardCharsets.UTF_8);
        } catch (Throwable t) {
            Log.w(TAG, "loadSecret failed: " + t);
            return "";
        }
    }

    /** Deletes the stored secret and its Keystore key. */
    public boolean clearSecret() {
        boolean ok = true;
        try {
            new File(getFilesDir(), SECRET_FILE).delete();
        } catch (Throwable t) {
            ok = false;
        }
        try {
            KeyStore keyStore = KeyStore.getInstance("AndroidKeyStore");
            keyStore.load(null);
            if (keyStore.containsAlias(SECRET_KEY_ALIAS)) {
                keyStore.deleteEntry(SECRET_KEY_ALIAS);
            }
        } catch (Throwable t) {
            ok = false;
        }
        return ok;
    }

    /** Returns the current clipboard text, or an empty string. */
    public String getClipboardText() {
        try {
            ClipboardManager manager = (ClipboardManager) getSystemService(CLIPBOARD_SERVICE);
            if (manager != null && manager.hasPrimaryClip()) {
                ClipData clip = manager.getPrimaryClip();
                if (clip != null && clip.getItemCount() > 0) {
                    CharSequence text = clip.getItemAt(0).coerceToText(this);
                    if (text != null) {
                        return text.toString();
                    }
                }
            }
        } catch (Throwable ignored) {
        }
        return "";
    }

    /**
     * System bar inset in physical pixels (top or bottom). Used instead of the
     * native content rect, which some OEM ROMs report with an inflated top.
     */
    public int systemBarTopPx() {
        return systemBarInsetPx(true);
    }

    public int systemBarBottomPx() {
        return systemBarInsetPx(false);
    }

    private int systemBarInsetPx(boolean top) {
        try {
            android.view.WindowInsets insets =
                getWindow().getDecorView().getRootWindowInsets();
            if (insets != null) {
                if (android.os.Build.VERSION.SDK_INT >= 30) {
                    android.graphics.Insets bars =
                        insets.getInsets(android.view.WindowInsets.Type.systemBars());
                    return top ? bars.top : bars.bottom;
                }
                return top
                    ? insets.getSystemWindowInsetTop()
                    : insets.getSystemWindowInsetBottom();
            }
        } catch (Throwable ignored) {
        }
        return 0;
    }

    /**
     * Copies {@code text} to the system clipboard. Android 10+ only allows
     * clipboard writes from the focused app, so if the window does not have
     * focus yet the value is applied from {@link #onWindowFocusChanged}.
     */
    public void setClipboard(String text) {
        pendingClipboard = text;
        applyPendingClipboard();
    }

    @Override
    public void onWindowFocusChanged(boolean hasFocus) {
        super.onWindowFocusChanged(hasFocus);
        if (hasFocus) {
            applyPendingClipboard();
        }
    }

    private void applyPendingClipboard() {
        if (pendingClipboard == null) {
            return;
        }
        try {
            ClipboardManager manager =
                (ClipboardManager) getSystemService(CLIPBOARD_SERVICE);
            manager.setPrimaryClip(ClipData.newPlainText("KeyScribe", pendingClipboard));
            ClipData clip = manager.getPrimaryClip();
            if (clip != null && clip.getItemCount() > 0) {
                CharSequence current = clip.getItemAt(0).coerceToText(this);
                if (current != null && current.toString().equals(pendingClipboard)) {
                    Log.i(TAG, "clipboard set ok");
                    pendingClipboard = null;
                }
            }
        } catch (Throwable t) {
            Log.w(TAG, "clipboard set failed: " + t);
        }
    }

    /**
     * Launches the Storage Access Framework audio picker. Instance method so
     * native code can invoke it on the Activity object directly (FindClass
     * from the native main thread cannot see app classes).
     */
    public void pickAudio() {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("audio/*");
        intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
        // Some providers report audio files as generic octet-streams.
        intent.putExtra(Intent.EXTRA_MIME_TYPES, new String[]{"audio/*"});
        startActivityForResult(intent, REQUEST_PICK_AUDIO);
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode != REQUEST_PICK_AUDIO) {
            return;
        }
        String path = "";
        if (resultCode == RESULT_OK && data != null && data.getData() != null) {
            String copied = copyToCache(data.getData());
            if (copied != null) {
                path = copied;
            }
        }
        nativeOnAudioPicked(path);
    }

    /** Copies the selected content URI into the app cache and returns its absolute path. */
    private String copyToCache(Uri uri) {
        try {
            String name = displayName(uri);
            if (name == null || name.isEmpty()) {
                name = "imported_audio";
            }
            File dir = new File(getCacheDir(), "imports");
            if (!dir.exists() && !dir.mkdirs()) {
                return null;
            }
            File dest = new File(dir, name);
            try (InputStream in = getContentResolver().openInputStream(uri);
                 OutputStream out = new FileOutputStream(dest)) {
                if (in == null) {
                    return null;
                }
                byte[] buffer = new byte[1 << 16];
                int read;
                while ((read = in.read(buffer)) > 0) {
                    out.write(buffer, 0, read);
                }
            }
            return dest.getAbsolutePath();
        } catch (Exception e) {
            return null;
        }
    }

    private String displayName(Uri uri) {
        try (android.database.Cursor cursor =
                 getContentResolver().query(uri, null, null, null, null)) {
            if (cursor != null && cursor.moveToFirst()) {
                int index = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME);
                if (index >= 0) {
                    return cursor.getString(index);
                }
            }
        } catch (Exception ignored) {
        }
        return null;
    }
}
