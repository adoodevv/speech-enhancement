"""
Step 5: Objective metrics -- required because "do not rely on listening alone."

Two metrics:
  1. Segmental SNR improvement (dB) -- the KPM benchmark metric itself.
  2. Log-Spectral Distance (LSD) -- a spectral distortion measure, used as the
     "distortion ceiling" check (excessive distortion should cap your score
     even if SegSNR improvement looks great -- e.g. spectral subtraction with
     too-aggressive oversubtraction removes noise but mangles speech).

NOTE: both metrics ideally compare against a clean reference. Since you have
no true clean signal for the blind file, use these two ways:
  - On your OWN recordings: compare against clean_ref_*.wav (near-field takes)
    as an approximate reference.
  - On the BLIND file: you can only measure noise-only-segment energy
    reduction and self-consistency checks (see estimate_segsnr_no_reference).
"""
import numpy as np


def segmental_snr(clean, processed, frame_len=512, hop=256, eps=1e-10):
    """Segmental SNR in dB, needs a clean reference."""
    n = min(len(clean), len(processed))
    clean, processed = clean[:n], processed[:n]
    n_frames = 1 + (n - frame_len) // hop
    seg_snrs = []
    for i in range(n_frames):
        c = clean[i * hop:i * hop + frame_len]
        p = processed[i * hop:i * hop + frame_len]
        noise = c - p
        signal_power = np.sum(c ** 2) + eps
        noise_power = np.sum(noise ** 2) + eps
        snr = 10 * np.log10(signal_power / noise_power)
        snr = np.clip(snr, -10, 35)  # standard segSNR clipping range
        seg_snrs.append(snr)
    return np.mean(seg_snrs)


def log_spectral_distance(clean, processed, fs, frame_len=512, hop=256, eps=1e-10):
    """LSD in dB -- lower is better (less spectral distortion)."""
    n = min(len(clean), len(processed))
    clean, processed = clean[:n], processed[:n]
    window = np.hanning(frame_len)
    n_frames = 1 + (n - frame_len) // hop
    dists = []
    for i in range(n_frames):
        c = clean[i * hop:i * hop + frame_len] * window
        p = processed[i * hop:i * hop + frame_len] * window
        C = np.abs(np.fft.rfft(c)) + eps
        P = np.abs(np.fft.rfft(p)) + eps
        log_diff = 20 * (np.log10(C) - np.log10(P))
        dists.append(np.sqrt(np.mean(log_diff ** 2)))
    return np.mean(dists)


def estimate_segsnr_no_reference(noisy, enhanced, fs, frame_len=512, hop=256):
    """
    Reference-free proxy for the blind file: compares energy reduction in
    detected noise-only frames vs energy retained in detected speech frames.
    Use this ONLY when you have no clean reference (i.e. the blind recording).
    Report it alongside the reference-based SegSNR from your own data.
    """
    n = min(len(noisy), len(enhanced))
    noisy, enhanced = noisy[:n], enhanced[:n]
    n_frames = 1 + (n - frame_len) // hop
    noisy_frames = np.stack([noisy[i * hop:i * hop + frame_len] for i in range(n_frames)])
    enh_frames = np.stack([enhanced[i * hop:i * hop + frame_len] for i in range(n_frames)])

    energies = np.sum(noisy_frames ** 2, axis=1)
    speech_thresh = np.percentile(energies, 60)
    speech_mask = energies > speech_thresh
    noise_mask = ~speech_mask

    noise_reduction_db = 10 * np.log10(
        (np.sum(noisy_frames[noise_mask] ** 2) + 1e-10) /
        (np.sum(enh_frames[noise_mask] ** 2) + 1e-10)
    )
    speech_retention_db = 10 * np.log10(
        (np.sum(enh_frames[speech_mask] ** 2) + 1e-10) /
        (np.sum(noisy_frames[speech_mask] ** 2) + 1e-10)
    )
    return noise_reduction_db, speech_retention_db


if __name__ == "__main__":
    import soundfile as sf
    clean, fs1 = sf.read("data/raw/clean_ref_01.wav")
    processed, fs2 = sf.read("audio_demo/noisy_market_01_enhanced.wav")
    print("SegSNR (dB):", segmental_snr(clean, processed))
    print("LSD (dB):", log_spectral_distance(clean, processed, fs1))
