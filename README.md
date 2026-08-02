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
python -c "import numpy, soundfile, matplotlib; print('OK')"
```

You should see `OK`. Only after this step, proceed to recordings and analysis.

---

## 2. Folder layout

```text
data/
  raw/
    noise_only_01.wav          <- 3-5 min pure market noise, no speech
    clean_ref_01.wav ... 05    <- near-field "clean-ish" PIN takes
    noisy_market_01.wav ... 10 <- real noisy PIN takes (main evaluation data)
  blind/
    (empty until course team releases the blind file — do NOT touch its contents)
src/
  noise_stats.py     <- Step 3: characterize the noise
  enhance.py         <- Step 4: spectral subtraction implementation
  metrics.py         <- Step 5: SegSNR + spectral distortion
  run_pipeline.py    <- ties it all together; run this on the blind file
notebooks/
  (optional — plots and exploration if you prefer notebooks over scripts)
audio_demo/
  (Step 7: 2-3 before/after clip pairs for the live presentation)
requirements.txt     <- pinned third-party packages for the venv
```

---

## 3. How to run (after venv is active and data is in place)

Activate the virtual environment first (see section 1), then from the project root:

```bash
python src/noise_stats.py          # produces plots + printed stats
python src/run_pipeline.py         # enhancement + metrics on all noisy files
```

Examples with explicit input/output folders:

```bash
python src/run_pipeline.py data/raw audio_demo
python src/run_pipeline.py data/blind audio_demo/blind
```

---

## 4. Workflow order

1. **Set up the virtual environment** (section 1) and install `requirements.txt`.
2. Record (see recording plan) into `data/raw/`.
3. `noise_stats.py` — characterize noise stationarity + spectral shape. Save the plots; the report must show these informed parameter choices.
4. `enhance.py` — spectral subtraction with justified frame size / overlap / update rate.
5. `metrics.py` — segmental SNR improvement + spectral distortion (LSD) per file.
6. Deliberately test a very-low-SNR or burst-noise take — document the failure mode.
7. Copy 2–3 clearest before/after pairs into `audio_demo/`.
8. When the blind file lands in `data/blind/`, run `run_pipeline.py` on it **unmodified**.
