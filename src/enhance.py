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
  - Noise estimate blends (a) low-energy frames inside the take (non-stationary
    market chatter) with (b) the dedicated noise_only recording spectral shape.
  - Floor is beta * noise_estimate (classic), NOT beta * noisy magnitude —
    the latter re-admits market noise in every frame.
"""
import os
import numpy as np
import soundfile as sf
from scipy import signal as sp_signal

FRAME_MS = 32
OVERLAP = 0.5
# Oversubtraction: market babble overlaps speech bands, so a high alpha is needed
# to push the ambient floor down once the PIN is already intelligible.
OVERSUBTRACTION_ALPHA = 3.0
SPECTRAL_FLOOR_BETA = 0.01    # floor relative to noise estimate (classic Boll/Berouti)
VAD_ENERGY_PERCENTILE = 25    # frames below this energy percentile = "noise-only"
# Extra attenuation on low-energy (non-PIN) frames after subtraction — residual
# market chatter often survives in "silence" between digits.
RESIDUAL_GATE_DB = 12.0
RESIDUAL_GATE_PERCENTILE = 55  # frames below this energy get gated
TARGET_PEAK = 0.95
NOISE_ONLY_FILE = "data/raw/noise_only_01.wav"
# Blend local VAD noise with dedicated ambient profile (shape from noise_only).
EXTERNAL_NOISE_BLEND = 0.65


def load_audio(path):
    x, fs = sf.read(path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x.astype(np.float64), fs


def resample_if_needed(x, fs_in, fs_out):
    if fs_in == fs_out:
        return x
    n_out = int(round(len(x) * fs_out / fs_in))
    return sp_signal.resample(x, n_out)


def stft(x, frame_len, hop, window):
    n_frames = 1 + (len(x) - frame_len) // hop
    if n_frames < 1:
        raise ValueError("Signal shorter than one analysis frame")
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


def simple_vad_mask(frames_time_domain, percentile=VAD_ENERGY_PERCENTILE):
    """Energy-based VAD: low-energy frames treated as noise-only."""
    energies = np.sum(frames_time_domain ** 2, axis=1)
    threshold = np.percentile(energies, percentile)
    return energies <= threshold


def estimate_noise_spectrum(mag, time_frames, fs, frame_len, hop, window):
    """
    Local VAD average + optional dedicated noise_only profile.
    External file contributes spectral SHAPE (scaled to this take's noise level),
    which matches the report requirement that noise_stats shaped the design.
    """
    noise_mask = simple_vad_mask(time_frames)
    if noise_mask.sum() == 0:
        noise_mask = np.ones(len(time_frames), dtype=bool)
    local_noise = mag[noise_mask].mean(axis=0)

    if not os.path.exists(NOISE_ONLY_FILE):
        return local_noise

    n, nfs = load_audio(NOISE_ONLY_FILE)
    n = resample_if_needed(n, nfs, fs)
    # Use a long stretch but cap runtime on multi-minute files
    max_samples = int(60 * fs)
    if len(n) > max_samples:
        n = n[:max_samples]
    if len(n) < frame_len:
        return local_noise

    n_mag = np.abs(stft(n, frame_len, hop, window))
    external = n_mag.mean(axis=0)
    # Match overall level to this recording's noise frames
    scale = (local_noise.mean() + 1e-8) / (external.mean() + 1e-8)
    external = external * scale
    b = EXTERNAL_NOISE_BLEND
    return (1.0 - b) * local_noise + b * external


def frequency_dependent_alpha(noise_estimate, base_alpha=OVERSUBTRACTION_ALPHA):
    """
    Stronger subtraction where the noise spectrum is relatively loud.
    noise_stats: market energy concentrated below ~1-4 kHz (babble-like).
    """
    rel = noise_estimate / (np.mean(noise_estimate) + 1e-8)
    # Allow more aggressive bins on loud noise frequencies
    return base_alpha * np.clip(rel, 0.8, 2.0)


def residual_noise_gate(enhanced_mag, time_frames, gate_db=RESIDUAL_GATE_DB,
                        percentile=RESIDUAL_GATE_PERCENTILE):
    """
    Soft-attenuate residual noise between PIN digits.
    High-energy frames (speech) stay near full gain; low-energy frames are
    turned down by up to gate_db. Reduces audible market bed without muting PIN.
    """
    if gate_db <= 0:
        return enhanced_mag
    energies = np.sum(time_frames ** 2, axis=1)
    thr = np.percentile(energies, percentile) + 1e-12
    min_gain = 10 ** (-gate_db / 20.0)
    # Linear soft ramp: 0 energy -> min_gain, thr energy -> 1.0
    gains = min_gain + (1.0 - min_gain) * np.clip(energies / thr, 0.0, 1.0)
    return enhanced_mag * gains[:, None]


def spectral_subtraction(x, fs):
    frame_len = int(FRAME_MS * fs / 1000)
    hop = int(frame_len * (1 - OVERLAP))
    window = np.hanning(frame_len)

    n_frames = 1 + (len(x) - frame_len) // hop
    time_frames = np.stack([x[i * hop: i * hop + frame_len] for i in range(n_frames)])

    spec = stft(x, frame_len, hop, window)
    mag = np.abs(spec)
    phase = np.angle(spec)

    noise_estimate = estimate_noise_spectrum(
        mag, time_frames, fs, frame_len, hop, window
    )
    alpha = frequency_dependent_alpha(noise_estimate)

    # Classic oversubtraction + noise-relative spectral floor
    subtracted = mag - alpha[None, :] * noise_estimate[None, :]
    floor = SPECTRAL_FLOOR_BETA * noise_estimate[None, :]
    enhanced_mag = np.maximum(subtracted, floor)

    # Quiet residual chatter between digits / before speech starts
    enhanced_mag = residual_noise_gate(enhanced_mag, time_frames)

    enhanced_spec = enhanced_mag * np.exp(1j * phase)
    enhanced = istft(enhanced_spec, frame_len, hop, window, len(x))
    return enhanced


def frame_rms(x, frame_len, hop):
    n_frames = 1 + (len(x) - frame_len) // hop
    if n_frames < 1:
        return np.array([np.sqrt(np.mean(x ** 2))])
    rms = []
    for i in range(n_frames):
        seg = x[i * hop: i * hop + frame_len]
        rms.append(np.sqrt(np.mean(seg ** 2)))
    return np.array(rms)


def match_loudness(reference, enhanced, fs, target_peak=TARGET_PEAK, clip_percentile=99.5):
    """
    Soft-clip ISTFT spikes, then match loudness using SPEECH-like frames only
    (high energy in the reference). Matching full-file RMS re-boosts residual
    market noise back up to the original noisy level.
    """
    abs_enh = np.abs(enhanced)
    clip_level = np.percentile(abs_enh, clip_percentile) + 1e-8
    enhanced = np.clip(enhanced, -clip_level, clip_level)

    frame_len = int(FRAME_MS * fs / 1000)
    hop = frame_len // 2
    ref_rms = frame_rms(reference, frame_len, hop)
    enh_rms = frame_rms(enhanced, frame_len, hop)
    n = min(len(ref_rms), len(enh_rms))
    ref_rms, enh_rms = ref_rms[:n], enh_rms[:n]

    # Top 30% energy frames ~ PIN digits / speech, not ambient floor
    speech_mask = ref_rms >= np.percentile(ref_rms, 70)
    if speech_mask.sum() == 0:
        speech_mask = np.ones(n, dtype=bool)

    rms_ref = np.mean(ref_rms[speech_mask]) + 1e-8
    rms_enh = np.mean(enh_rms[speech_mask]) + 1e-8
    enhanced = enhanced * (rms_ref / rms_enh)

    peak = np.max(np.abs(enhanced)) + 1e-8
    if peak > target_peak:
        enhanced = enhanced * (target_peak / peak)
    return enhanced


def process_file(in_path, out_path):
    x, fs = load_audio(in_path)
    enhanced = spectral_subtraction(x, fs)
    enhanced = match_loudness(x, enhanced, fs)
    sf.write(out_path, enhanced.astype(np.float64), fs)
    return fs


if __name__ == "__main__":
    import sys
    in_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/noisy_market_01.wav"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "audio_demo/noisy_market_01_enhanced.wav"
    process_file(in_path, out_path)
    print(f"Enhanced audio written to {out_path}")
