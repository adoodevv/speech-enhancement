"""
Step D (Requirement): Demonstrate the OTHER allowed failure mode --
non-stationary BURST noise (e.g. a sudden horn/shout), as distinct from
the low-SNR test in failure_mode.py.

Why bursts are a genuine weak point of our design:
  - Our noise estimate blends (a) low-energy ("quiet") frames inside the
    take via VAD, and (b) the dedicated noise_only ambient profile.
  - A burst is LOUD, so the VAD correctly does NOT treat it as noise --
    but our residual_noise_gate also does NOT attenuate it, because that
    gate only turns down LOW-energy frames (assuming high energy = speech
    worth preserving).
  - Net effect: a burst is treated exactly like real speech and passes
    through almost fully un-suppressed, potentially masking the PIN
    digits that overlap it in time.

Run:
    python src/burst_failure_mode.py

Outputs (under audio_demo/failure_mode/):
    burst_input.wav      -- noisy_market_01 with an injected horn-like burst
    burst_enhanced.wav   -- after spectral subtraction
"""
import os
import sys
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(__file__))
from enhance import load_audio, process_file
from metrics import log_spectral_distance

NOISY_FILE = "data/raw/noisy_market_01.wav"
OUT_DIR = "audio_demo/failure_mode"
BURST_DURATION_S = 0.35
BURST_LEVEL = 0.9   # near full-scale, like a close horn/shout
BURST_CENTER_FREQ = 500   # low-frequency-heavy burst, like a horn


def make_burst(n_samples, fs):
    """Synthetic horn-like burst: bandpass noise with a fast attack, slow decay."""
    t = np.arange(n_samples) / fs
    envelope = np.exp(-3.0 * t)  # fast attack (starts at 1), decays over ~0.35s
    noise = np.random.randn(n_samples)
    from scipy.signal import butter, filtfilt
    b, a = butter(4, [BURST_CENTER_FREQ * 0.6, BURST_CENTER_FREQ * 1.8],
                  btype="band", fs=fs)
    burst = filtfilt(b, a, noise) * envelope
    burst = burst / (np.max(np.abs(burst)) + 1e-8) * BURST_LEVEL
    return burst


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    x, fs = load_audio(NOISY_FILE)

    burst_len = int(BURST_DURATION_S * fs)
    burst = make_burst(burst_len, fs)

    start = len(x) // 2 - burst_len // 2
    mixture = x.copy()
    mixture[start:start + burst_len] += burst
    peak = np.max(np.abs(mixture)) + 1e-8
    if peak > 0.98:
        mixture = mixture * (0.98 / peak)

    input_path = os.path.join(OUT_DIR, "burst_input.wav")
    sf.write(input_path, mixture, fs)

    enhanced_path = os.path.join(OUT_DIR, "burst_enhanced.wav")
    process_file(input_path, enhanced_path)

    enhanced, _ = load_audio(enhanced_path)

    burst_region_before = mixture[start:start + burst_len]
    burst_region_after = enhanced[start:start + burst_len]
    burst_reduction_db = 10 * np.log10(
        (np.sum(burst_region_before ** 2) + 1e-12) /
        (np.sum(burst_region_after ** 2) + 1e-12)
    )

    # Genuine ambient-only baseline: use energy-based VAD across the WHOLE
    # file (excluding the injected burst window) to find real low-energy
    # (noise-only) frames, and measure how much THOSE get reduced. This is
    # the fair comparison -- not a blend of speech + noise.
    frame_len = 1024
    hop = 512
    n_frames = 1 + (len(mixture) - frame_len) // hop
    ambient_reduction_vals = []
    for i in range(n_frames):
        s = i * hop
        e = s + frame_len
        if s >= start and s < start + burst_len:
            continue  # skip frames overlapping the injected burst
        seg_before = mixture[s:e]
        if np.sum(seg_before ** 2) < 1e-6:
            continue
        seg_after = enhanced[s:e]
        ambient_reduction_vals.append(
            10 * np.log10((np.sum(seg_before ** 2) + 1e-12) /
                          (np.sum(seg_after ** 2) + 1e-12))
        )
    # Use the bottom 30% (quietest, most noise-like) frames as the ambient baseline
    ambient_reduction_vals = np.array(ambient_reduction_vals)
    quietest = ambient_reduction_vals[
        ambient_reduction_vals < np.percentile(ambient_reduction_vals, 30)
    ] if len(ambient_reduction_vals) else np.array([0.0])
    ambient_reduction_db = np.mean(quietest)

    print(f"Burst-region noise reduction: {burst_reduction_db:.2f} dB")
    print(f"Genuine ambient-noise reduction (elsewhere in same file): "
          f"{ambient_reduction_db:.2f} dB")
    print(f"Gap: {ambient_reduction_db - burst_reduction_db:.2f} dB")
    print("  (a large positive gap means the algorithm suppresses ordinary")
    print("   market ambience far more than it suppresses the sudden burst --")
    print("   confirming the burst is being treated like speech and let through)")
    print(f"\nFiles written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
