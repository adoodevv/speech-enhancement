# Speech Enhancement for Voice-Based MoMo Authentication

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
