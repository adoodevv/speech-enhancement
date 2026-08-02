"""
Step 7-ish: run the full pipeline over every noisy file in a folder.

Usage:
    python src/run_pipeline.py data/raw audio_demo          # your own recordings
    python src/run_pipeline.py data/blind audio_demo/blind  # the released blind file

This must run UNMODIFIED on the blind file -- don't hand-tune parameters
per file when you get there.
"""
import os
import sys
import glob
import soundfile as sf
from enhance import process_file, load_audio
from metrics import segmental_snr, log_spectral_distance, estimate_segsnr_no_reference


def main(in_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    noisy_files = sorted(glob.glob(os.path.join(in_dir, "noisy_market_*.wav")) +
                          glob.glob(os.path.join(in_dir, "*.wav")))
    # dedupe & drop noise-only / clean-ref files from the "noisy" batch
    noisy_files = [f for f in noisy_files
                   if "noise_only" not in f and "clean_ref" not in f]

    for f in sorted(set(noisy_files)):
        base = os.path.splitext(os.path.basename(f))[0]
        out_path = os.path.join(out_dir, f"{base}_enhanced.wav")
        fs = process_file(f, out_path)
        print(f"\n=== {base} ===")

        noisy, _ = load_audio(f)
        enhanced, _ = load_audio(out_path)

        # Try reference-based metrics if a matching clean_ref exists, else
        # fall back to the reference-free proxy (this is what you'll use for
        # the blind file).
        ref_candidate = f.replace("noisy_market", "clean_ref")
        if os.path.exists(ref_candidate):
            clean, _ = load_audio(ref_candidate)
            print("  SegSNR improvement (dB):", segmental_snr(clean, enhanced) -
                  segmental_snr(clean, noisy))
            print("  LSD (dB, lower=better):", log_spectral_distance(clean, enhanced, fs))
        else:
            nr_db, sr_db = estimate_segsnr_no_reference(noisy, enhanced, fs)
            print(f"  [no clean ref] Noise reduction: {nr_db:.2f} dB, "
                  f"Speech retention: {sr_db:.2f} dB")


if __name__ == "__main__":
    in_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "audio_demo"
    main(in_dir, out_dir)
