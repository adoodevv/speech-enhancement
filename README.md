# Mini Project 2: Speech Enhancement for Voice-Based MoMo Authentication

Scenario: **Open-air market chatter**  
Method: **Spectral subtraction**  
Phrase type: short numeric PIN (4–6 digits)

---

## 1. Environment setup (do this first)

Create and activate a Python virtual environment **before** installing packages or running any analysis. This keeps project dependencies isolated from your system Python.

You need **Python 3.9+**. Check with:

```bash
python --version
# or, on some Linux installs:
python3 --version
```

### Linux / macOS

From the project root (`speech-enhancement/`):

```bash
# Create the virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

To leave the environment later:

```bash
deactivate
```

### Windows

**Command Prompt (cmd):**

```bat
REM Create the virtual environment
python -m venv .venv

REM Activate it
.venv\Scripts\activate.bat

REM Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**PowerShell:**

```powershell
# Create the virtual environment
python -m venv .venv

# Activate it (if execution policy blocks this, run once as admin:
#   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\.venv\Scripts\Activate.ps1

# Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

To leave the environment later:

```bat
deactivate
```

### Confirm the install

With the venv **still activated**:

```bash
python -c "import numpy, soundfile, matplotlib, scipy; print('OK')"
```

You should see `OK`. Only after this step, proceed to recordings and analysis.

---

## 2. Recordings (three kinds of audio)

All project recordings live under `data/raw/`. Use these exact name patterns so the scripts find them.

| Kind | Content | Filename pattern |
|------|---------|------------------|
| **Noise only** | Market ambience, **no PIN speech** (2–5 min ideal) | `noise_only_01.wav` |
| **Clean ref** | Same speaker / PIN style in a **quiet** place | `clean_ref_01.wav` … |
| **Noisy market** | PIN spoken **in the market** (main evaluation) | `noisy_market_01.wav` … |

### Current dataset (on disk)

```text
data/raw/
  noise_only_01.wav       # ambient market (may be stereo / different fs — code mono-mixes + resamples)
  clean_ref_01.wav        # quiet PIN references (pair with noisy_* when possible)
  clean_ref_02.wav
  clean_ref_03.wav
  noisy_market_01.wav     # organic market PIN takes (primary evaluation set)
  noisy_market_02.wav
  noisy_market_03.wav
```

**Failure-mode audio** (generated / stress-tested — not a substitute for organic market takes) lives under `audio_demo/failure_mode/` (see Step D).

**Do not** put course blind files in `data/raw/` — those go in `data/blind/` when released.

---

## 3. Folder layout

```text
data/
  raw/                 <- three kinds of audio (above)
  blind/               <- course blind file later; leave alone until release
src/
  noise_stats.py       <- Step A: characterise noise (run first)
  enhance.py           <- Step B: spectral subtraction
  metrics.py           <- Step C: SegSNR + LSD (+ no-ref proxy)
  run_pipeline.py      <- enhance + metrics over a folder
  failure_mode.py      <- Step D: very-low-SNR stress test
  burst_failure_mode.py<- Step D: non-stationary burst stress test
notebooks/             <- optional exploration
audio_demo/
  noisy_market_*_enhanced.wav
  failure_mode/        <- low_snr_* and burst_* before/after pairs
requirements.txt
```

---

## 4. Workflow so far (completed steps)

Work through these **in order**. Steps **A–D are done**; continue with E–G for demos, report, and blind evaluation.

### Step A — Characterise the noise (required before designing the filter)

```bash
# venv active, from project root
python src/noise_stats.py
```

Reads `data/raw/noise_only_01.wav` (set `NOISE_FILE` in the script if you rename it).

**Outputs (project root):**

- `noise_stationarity_check.png` — spectrum over successive time segments  
- `noise_spectral_shape.png` — average noise magnitude spectrum  

**Findings on our recording (use in the report):**

| Question | Result | Design implication |
|----------|--------|--------------------|
| Stationarity | Mean relative spectral drift **≈ 0.17** (mildly non-stationary) | Do **not** rely on a single frozen noise snapshot only; blend **local VAD** frames with the dedicated ambient profile |
| Spectral shape | Energy **below 1 kHz ≫ above 4 kHz** (babble / low-band heavy, not white) | Use **frequency-dependent oversubtraction** (stronger where noise is relatively loud) and a **noise-relative** spectral floor |

This step satisfies the brief: *characterise noise statistics before designing the filter, and show evidence this shaped parameters.*

---

### Step B — Spectral subtraction (enhance)

```bash
# single file
python src/enhance.py data/raw/noisy_market_01.wav audio_demo/noisy_market_01_enhanced.wav

# or full folder runner (recommended)
python src/run_pipeline.py data/raw audio_demo
```

**What the enhancer does:**

1. STFT with **32 ms** frames, **50%** overlap (Hann window).  
2. **Noise estimate** = blend of  
   - low-energy frames inside the noisy take (VAD), and  
   - spectral shape from `noise_only_01.wav` (resampled if needed).  
3. **Oversubtraction** with frequency-dependent α.  
4. **Spectral floor** = `β × noise_estimate` (classic form — *not* `β × noisy magnitude`, which re-admits market noise).  
5. **Residual gate** — soft extra attenuation on low-energy frames (between digits / before speech).  
6. **Loudness match** on speech-like frames only + spike soft-clip (so the PIN stays audible without re-boosting the whole ambient bed).

**Current parameters** (`src/enhance.py`) — justify these from Step A in the report:

| Parameter | Value | Role |
|-----------|-------|------|
| `FRAME_MS` | 32 | Time/frequency resolution for PIN phonemes |
| `OVERLAP` | 0.5 | Standard STFT OLA |
| `OVERSUBTRACTION_ALPHA` | 3.0 | Aggressive enough for babble floor |
| `SPECTRAL_FLOOR_BETA` | 0.01 | Noise-relative floor (limits musical noise / negatives) |
| `VAD_ENERGY_PERCENTILE` | 25 | Local noise frames inside the take |
| `EXTERNAL_NOISE_BLEND` | 0.65 | Weight of dedicated `noise_only` shape |
| `RESIDUAL_GATE_DB` | 12 | Extra cut on non-speech frames |
| `RESIDUAL_GATE_PERCENTILE` | 55 | Which frames get gated |
| `TARGET_PEAK` | 0.95 | Avoid clipping after gain match |

**Tuning notes (what we already learned):**

- Too-mild α / high floor + full-file RMS gain match → **PIN audible but market still loud**.  
- Peak-normalise by absolute max after ISTFT spikes → **PIN almost inaudible**.  
- Current balance: PIN intelligible, background **reduced but not perfect** (expected for spectral subtraction on babble).

**Demo outputs (organic takes):**

```text
audio_demo/noisy_market_01_enhanced.wav
audio_demo/noisy_market_02_enhanced.wav
audio_demo/noisy_market_03_enhanced.wav
```

Compare by ear (example for take 01):

1. `data/raw/noisy_market_01.wav` (before)  
2. `audio_demo/noisy_market_01_enhanced.wav` (after)  
3. `data/raw/clean_ref_01.wav` (quiet reference)

---

### Step C — Objective metrics (do not rely on listening alone)

`run_pipeline.py` prints metrics after each file. You can also run:

```bash
python src/metrics.py
```

**Metrics used:**

| Metric | Meaning |
|--------|---------|
| **Segmental SNR improvement (dB)** | `SegSNR(clean, enhanced) − SegSNR(clean, noisy)` — higher is better; related to the course KPM idea |
| **Log-spectral distance, LSD (dB)** | Spectral distortion vs clean — **lower** is better (distortion ceiling) |
| **No-reference proxy** | Noise reduction vs speech retention when no clean pair exists (blind file) |

**Caveat:** clean refs and noisy market files are **separate takes** (not sample-aligned clones). Numbers are **indicative** for the report, not lab-perfect.

**Indicative results on organic market takes** (current frozen settings; re-run after any change):

| File | SegSNR improvement (dB) | LSD (dB, lower better) |
|------|-------------------------|------------------------|
| `noisy_market_01` | ~**3.5** | ~**31.6** |
| `noisy_market_02` | ~**2.8** | ~**23.0** |
| `noisy_market_03` | ~**2.0** | ~**22.6** |

On take 01 we also measured roughly: noise-floor frames ~**20× quieter**, speech-ish level retained, no-ref noise reduction ~**10 dB**.

---

### Step D — Failure modes (brief requirement)

The brief requires showing at least one condition where the method **degrades or fails**, with an explanation. We cover **both** allowed stress cases using our own recorded clean/noise material (the organic `noisy_market_*.wav` files remain the primary evaluation set).

#### D1 — Very low input SNR

```bash
python src/failure_mode.py
```

| | |
|--|--|
| **What it does** | Mixes `clean_ref_01.wav` with `noise_only_01.wav` at a controlled **−5 dB** SNR (harsher than a typical natural market take) |
| **Outputs** | `audio_demo/failure_mode/low_snr_input.wav`, `low_snr_enhanced.wav` |
| **Indicative numbers** | SegSNR before ~−7.7 dB → after ~−3.8 dB (**+4.0 dB** improvement); LSD ~17.9 → ~15.9 (**slightly better**) |

**What to teach / write in the report:** even when SegSNR moves a little, at −5 dB the PIN is **barely usable**. Spectral subtraction cannot invent missing speech energy under overwhelming babble; residual noise and spectral damage remain obvious on listening. This is the **very-low-SNR** failure regime (intelligibility collapse / marginal usability), not a claim that metrics always go negative.

#### D2 — Non-stationary burst noise

```bash
python src/burst_failure_mode.py
```

| | |
|--|--|
| **What it does** | Injects a short **horn-like burst** (~0.35 s, low-band, near full scale) into `noisy_market_01.wav` mid-file |
| **Outputs** | `audio_demo/failure_mode/burst_input.wav`, `burst_enhanced.wav` |
| **Indicative numbers** | Burst-region reduction ~**−1.1 dB** (burst **not** suppressed; energy can even rise slightly after processing); quiet ambient frames elsewhere still get some reduction → **gap ~+1.8 dB** |

**Why this fails by design (use this explanation):**

1. Energy VAD treats the **loud** burst as speech-like (not “noise-only”), so it does **not** update the noise estimate from the burst.  
2. The **residual gate** only attenuates **low-energy** frames (between digits). The burst is high energy → **full gain**.  
3. Stationary/slowly varying market ambience is what our estimator was built for; a **sudden** non-stationary event passes through almost unsuppressed and can mask overlapping PIN digits.

**Listen for the report/demo:** burst still dominates after enhancement; compare ambient bed vs burst region.

---

## 5. How to re-run everything (checklist)

```bash
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # once, or after dependency changes

python src/noise_stats.py                      # A — plots + printed drift / band energy
python src/run_pipeline.py data/raw audio_demo # B+C — enhance + metrics on all noisy_market_*
python src/failure_mode.py                     # D1 — low-SNR stress pair
python src/burst_failure_mode.py               # D2 — burst stress pair
```

Keep venv active for every `python …` command.

---

## 6. Still to do (next steps)

| Step | Task | Status |
|------|------|--------|
| A | Noise characterisation | **Done** (save PNGs for report) |
| B | Spectral subtraction + justified parameters | **Done** (freeze before blind) |
| C | Objective metrics on own data | **Done** (3 organic takes) |
| D | **Failure modes** — low SNR + burst | **Done** (scripts + `audio_demo/failure_mode/`) |
| E | **Demo pack** — 2–3 clear before/after pairs for live presentation | **Partial** (3 enhanced organics + failure pairs; tidy naming/slides still optional) |
| F | **Blind file** — when placed in `data/blind/`, run pipeline **unmodified** | Pending |
| G | Report write-up + minimum research sources (method, metrics, MoMo/voice auth context) | Pending |

Blind evaluation:

```bash
python src/run_pipeline.py data/blind audio_demo/blind
```

Do **not** retune α/β/gate on the blind file.

---

## 7. Design rationale (short, for the report)

1. **Why spectral subtraction:** assigned enhancement family; works as a single-mic front end with no second reference mic (fits market MoMo scenario).  
2. **Why 32 ms / 50%:** standard STFT compromise; ~31 Hz bin spacing at 48 kHz frame length, short enough for PIN digits.  
3. **Why non-stationary handling:** drift ≈ 0.17 → VAD-local noise + ambient `noise_only` shape + residual gate between digits.  
4. **Why aggressive α and noise-relative β:** market energy is speech-like and low-band heavy; mild settings left the background too loud once PIN gain was restored.  
5. **Known limits (failure modes):** (i) **very low SNR** — babble overwhelms PIN; enhancement cannot recover lost intelligibility; (ii) **bursts** — treated as speech by energy VAD/gate and largely left intact, masking digits.

---

## 8. Dependencies

See `requirements.txt`:

- `numpy`, `scipy` — arrays, STFT-related helpers, resampling, burst filter  
- `soundfile` — WAV I/O  
- `matplotlib` — noise-stats plots  
