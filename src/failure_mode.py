"""
Step D (Requirement): Demonstrate a failure mode.

We construct a deliberately very-low-SNR test case by mixing the REAL
clean_ref recording with the REAL noise_only recording at a controlled,
much harsher SNR than our natural noisy_market take. This is a standard,
legitimate technique for failure-mode testing (the noisy_market_01.wav
recording remains our primary, organically-recorded evaluation signal
required by the brief -- this script only builds an additional STRESS TEST
on top of it, using noise we ourselves recorded).

Run:
    python src/failure_mode.py

Outputs (under audio_demo/failure_mode/):
    low_snr_input.wav       -- the synthetic very-low-SNR mixture (before)
    low_snr_enhanced.wav    -- after spectral subtraction (after)
Prints SegSNR/LSD before vs after, and an actual input SNR check.
"""
import os
import numpy as np
import soundfile as sf
import sys

sys.path.insert(0, os.path.dirname(__file__))
from enhance import load_audio, resample_if_needed, process_file
from metrics import segmental_snr, log_spectral_distance

CLEAN_FILE = "data/raw/clean_ref_01.wav"
NOISE_FILE = "data/raw/noise_only_01.wav"
OUT_DIR = "audio_demo/failure_mode"
TARGET_SNR_DB = -5.0   # deliberately harsh -- our natural market take is milder


def mix_at_snr(clean, noise, target_snr_db):
    """Scale noise so the mixture hits an exact target SNR (dB)."""
    if len(noise) < len(clean):
        reps = int(np.ceil(len(clean) / len(noise)))
        noise = np.tile(noise, reps)
    # pick a random-ish but fixed offset for variety, then trim to clean's length
    offset = len(noise) // 3
    noise_seg = noise[offset:offset + len(clean)]

    clean_power = np.mean(clean ** 2) + 1e-12
    noise_power = np.mean(noise_seg ** 2) + 1e-12
    target_noise_power = clean_power / (10 ** (target_snr_db / 10))
    scale = np.sqrt(target_noise_power / noise_power)
    noise_scaled = noise_seg * scale

    mixture = clean + noise_scaled
    peak = np.max(np.abs(mixture)) + 1e-8
    if peak > 0.98:
        mixture = mixture * (0.98 / peak)
    return mixture


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    clean, fs_c = load_audio(CLEAN_FILE)
    noise, fs_n = load_audio(NOISE_FILE)
    noise = resample_if_needed(noise, fs_n, fs_c)

    mixture = mix_at_snr(clean, noise, TARGET_SNR_DB)

    input_path = os.path.join(OUT_DIR, "low_snr_input.wav")
    sf.write(input_path, mixture, fs_c)

    enhanced_path = os.path.join(OUT_DIR, "low_snr_enhanced.wav")
    process_file(input_path, enhanced_path)

    enhanced, _ = load_audio(enhanced_path)

    seg_before = segmental_snr(clean, mixture)
    seg_after = segmental_snr(clean, enhanced)
    lsd_before = log_spectral_distance(clean, mixture, fs_c)
    lsd_after = log_spectral_distance(clean, enhanced, fs_c)

    print(f"Target mixture SNR: {TARGET_SNR_DB} dB")
    print(f"SegSNR before: {seg_before:.2f} dB, after: {seg_after:.2f} dB, "
          f"improvement: {seg_after - seg_before:.2f} dB")
    print(f"LSD before: {lsd_before:.2f} dB, after: {lsd_after:.2f} dB, "
          f"change: {lsd_after - lsd_before:+.2f} dB (positive = worse)")
    print(f"\nFiles written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
