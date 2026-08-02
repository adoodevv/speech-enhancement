"""
Step 4: Spectral subtraction speech enhancement.

Core idea: estimate the noise magnitude spectrum from noise-only frames,
subtract it (with an oversubtraction factor + spectral floor) from the noisy
magnitude spectrum, keep the noisy phase, and reconstruct with overlap-add.

Frame size / overlap / update choices (JUSTIFY these numbers in your report
using what noise_stats.py showed you):
  - FRAME_MS = 32ms, 50% overlap: standard trade-off giving ~31Hz frequency
    resolution while staying short enough that market speech (fast phonemes
    in a PIN) isn't smeared across frames.
  - Noise re-estimated only during detected silence/noise-only segments
    (simple energy-based VAD) rather than once globally, because
    noise_stats.py should show market noise is NOT perfectly stationary.
"""
import numpy as np
import soundfile as sf

FRAME_MS = 32
OVERLAP = 0.5
OVERSUBTRACTION_ALPHA = 2.0   # how aggressively to subtract noise estimate
SPECTRAL_FLOOR_BETA = 0.02    # floor to avoid musical noise / negative values
VAD_ENERGY_PERCENTILE = 20    # frames below this energy percentile = "noise-only"


def load_audio(path):
    x, fs = sf.read(path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x.astype(np.float64), fs


def stft(x, frame_len, hop, window):
    n_frames = 1 + (len(x) - frame_len) // hop
    frames = np.stack([x[i * hop: i * hop + frame_len] for i in range(n_frames)])
    return np.fft.rfft(frames * window, axis=1)


def istft(spec, frame_len, hop, window, out_len):
    frames = np.fft.irfft(spec, n=frame_len, axis=1)
    out = np.zeros(out_len)
    norm = np.zeros(out_len)
    for i in range(frames.shape[0]):
        start = i * hop
        out[start:start + frame_len] += frames[i] * window
        norm[start:start + frame_len] += window ** 2
    norm[norm < 1e-8] = 1e-8
    return out / norm


def simple_vad_mask(frames_time_domain):
    """Very simple energy-based VAD: bottom percentile of frame energy = noise-only."""
    energies = np.sum(frames_time_domain ** 2, axis=1)
    threshold = np.percentile(energies, VAD_ENERGY_PERCENTILE)
    return energies <= threshold


def spectral_subtraction(x, fs):
    frame_len = int(FRAME_MS * fs / 1000)
    hop = int(frame_len * (1 - OVERLAP))
    window = np.hanning(frame_len)

    n_frames = 1 + (len(x) - frame_len) // hop
    time_frames = np.stack([x[i * hop: i * hop + frame_len] for i in range(n_frames)])

    spec = stft(x, frame_len, hop, window)
    mag = np.abs(spec)
    phase = np.angle(spec)

    # Estimate noise from low-energy (likely noise-only) frames, updated
    # continuously rather than once, since market noise isn't fully stationary.
    noise_mask = simple_vad_mask(time_frames)
    if noise_mask.sum() == 0:
        noise_mask[:] = True  # fallback: use everything if VAD finds nothing
    noise_estimate = mag[noise_mask].mean(axis=0)

    # Oversubtraction with spectral floor
    subtracted = mag - OVERSUBTRACTION_ALPHA * noise_estimate[None, :]
    floor = SPECTRAL_FLOOR_BETA * mag
    enhanced_mag = np.maximum(subtracted, floor)

    enhanced_spec = enhanced_mag * np.exp(1j * phase)
    enhanced = istft(enhanced_spec, frame_len, hop, window, len(x))
    return enhanced


def process_file(in_path, out_path):
    x, fs = load_audio(in_path)
    enhanced = spectral_subtraction(x, fs)
    # Normalize to avoid clipping
    peak = np.max(np.abs(enhanced)) + 1e-8
    if peak > 1.0:
        enhanced = enhanced / peak
    sf.write(out_path, enhanced, fs)
    return fs


if __name__ == "__main__":
    import sys
    in_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/noisy_market_01.wav"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "audio_demo/noisy_market_01_enhanced.wav"
    process_file(in_path, out_path)
    print(f"Enhanced audio written to {out_path}")
