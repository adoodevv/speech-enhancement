"""
Step 3: Characterize the market noise BEFORE designing the filter.

Run:
    python src/noise_stats.py

Reads data/raw/noise_only_01.wav (change NOISE_FILE below if you named it
differently) and answers two questions you must report:

  1. Is the noise stationary or does its spectrum drift over time?
     -> tells you whether a single noise estimate is safe, or whether you
        need to keep re-estimating noise (voice-activity-gated updates).
  2. What is the spectral SHAPE of the noise (broadband/white vs tonal/
     low-frequency-heavy)? Market chatter is usually broadband + speech-like
     (energy concentrated below ~4 kHz, similar to babble noise), NOT white.
     -> this justifies your frequency-dependent oversubtraction factor and
        spectral floor choices in enhance.py.
"""
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

NOISE_FILE = "data/raw/noise_only_01.wav"
FRAME_MS = 32          # frame length in ms -- justify this number in your report
OVERLAP = 0.5          # 50% overlap is standard for STFT analysis/synthesis


def load_audio(path):
    x, fs = sf.read(path)
    if x.ndim > 1:
        x = x.mean(axis=1)  # downmix to mono if stereo
    return x, fs


def frame_signal(x, frame_len, hop):
    n_frames = 1 + (len(x) - frame_len) // hop
    frames = np.stack([x[i * hop: i * hop + frame_len] for i in range(n_frames)])
    return frames


def main():
    x, fs = load_audio(NOISE_FILE)
    frame_len = int(FRAME_MS * fs / 1000)
    hop = int(frame_len * (1 - OVERLAP))
    window = np.hanning(frame_len)

    frames = frame_signal(x, frame_len, hop)
    windowed = frames * window

    # Magnitude spectrum per frame
    spectra = np.abs(np.fft.rfft(windowed, axis=1))
    freqs = np.fft.rfftfreq(frame_len, d=1 / fs)

    # --- Stationarity check: does average spectrum drift over time? ---
    n_chunks = 4
    chunk_size = spectra.shape[0] // n_chunks
    chunk_means = [spectra[i * chunk_size:(i + 1) * chunk_size].mean(axis=0)
                   for i in range(n_chunks)]

    plt.figure(figsize=(8, 5))
    for i, cm in enumerate(chunk_means):
        plt.plot(freqs, 20 * np.log10(cm + 1e-8), label=f"segment {i+1}")
    plt.xlabel("Frequency (Hz)") # frequency axis label
    plt.ylabel("Magnitude (dB)")
    plt.title("Noise spectrum across time segments (stationarity check)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("noise_stationarity_check.png")
    print("Saved noise_stationarity_check.png")

    # Quantify drift: std of magnitude across time segments, per frequency bin
    drift = np.std(np.array(chunk_means), axis=0)
    avg_level = np.mean(np.array(chunk_means), axis=0) + 1e-8
    relative_drift = np.mean(drift / avg_level)
    print(f"Mean relative spectral drift across time: {relative_drift:.3f}")
    print("  (rule of thumb: <0.15 fairly stationary, >0.3 clearly non-stationary "
          "-- market chatter with bursts of talk/shouting will likely land >0.3)")

    # --- Spectral shape check ---
    overall_spectrum = spectra.mean(axis=0)
    plt.figure(figsize=(8, 5))
    plt.plot(freqs, 20 * np.log10(overall_spectrum + 1e-8))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title("Average noise spectral shape")
    plt.tight_layout()
    plt.savefig("noise_spectral_shape.png")
    print("Saved noise_spectral_shape.png")

    low_energy = overall_spectrum[freqs < 1000].mean()
    high_energy = overall_spectrum[freqs > 4000].mean()
    print(f"Energy below 1kHz: {low_energy:.4f}, above 4kHz: {high_energy:.4f}")
    print("  (market chatter typically concentrates energy below ~1-4kHz, "
          "similar to babble noise -- unlike white noise which is flat)")


if __name__ == "__main__":
    main()
