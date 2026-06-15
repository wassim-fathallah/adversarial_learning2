"""
One-command setup for a fresh machine.

    python setup.py

What it does:
  1. Creates .venv and installs requirements.txt
  2. Downloads datasets.zip from Dropbox → adversarial_fairness/datasets/
  3. Downloads FFB results from WandB API → fair_fairness_benchmark/results/
  4. Prints what to do manually (Ollama)
"""

import os
import sys
import subprocess
import zipfile
import urllib.request

ROOT     = os.path.dirname(os.path.abspath(__file__))
VENV     = os.path.join(ROOT, ".venv")
PYTHON   = os.path.join(VENV, "Scripts", "python.exe") if sys.platform == "win32" \
           else os.path.join(VENV, "bin", "python")
PIP      = os.path.join(VENV, "Scripts", "pip.exe")    if sys.platform == "win32" \
           else os.path.join(VENV, "bin", "pip")

DATASETS_URL = (
    "https://www.dropbox.com/scl/fi/6t0jh1n5v7y2tr35bkidq/datasets.zip"
    "?rlkey=2xkqwavfcqb56c7ybavzcm9ga&st=93ediu4h&dl=1"
)
DATASETS_ZIP = os.path.join(ROOT, "datasets.zip")
DATASETS_DIR = os.path.join(ROOT, "adversarial_fairness", "datasets")


def run(cmd, **kwargs):
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, **kwargs)


def step(n, msg):
    print(f"\n{'='*60}")
    print(f"  Step {n}: {msg}")
    print(f"{'='*60}")


# Step 1 — Virtual environment

step(1, "Create virtual environment + install packages")

if not os.path.exists(PYTHON):
    run([sys.executable, "-m", "venv", VENV])
    print("  venv created.")
else:
    print("  venv already exists — skipping creation.")

run([PIP, "install", "--upgrade", "pip", "-q"])
run([PIP, "install", "-r", os.path.join(ROOT, "requirements.txt"), "-q"])
print("  Packages installed.")


# Step 2 — Datasets

step(2, "Download datasets from Dropbox")

if os.path.exists(DATASETS_DIR) and os.listdir(DATASETS_DIR):
    print("  datasets/ already exists — skipping download.")
else:
    os.makedirs(DATASETS_DIR, exist_ok=True)

    print(f"  Downloading datasets.zip (~3 GB) ...")
    print(f"  From: {DATASETS_URL.split('?')[0]}")

    def _progress(count, block_size, total):
        pct = min(100, int(count * block_size * 100 / total))
        bar = "#" * (pct // 5) + "-" * (20 - pct // 5)
        print(f"\r  [{bar}] {pct}%", end="", flush=True)

    urllib.request.urlretrieve(DATASETS_URL, DATASETS_ZIP, reporthook=_progress)
    print()

    print("  Extracting ...")
    with zipfile.ZipFile(DATASETS_ZIP, "r") as z:
        # Extract each member, stripping a leading "datasets/" prefix if present
        for member in z.namelist():
            # Normalize: strip top-level "datasets/" folder if zip has one
            parts = member.split("/", 1)
            target_rel = parts[1] if len(parts) > 1 and parts[0] == "datasets" else member
            if not target_rel:
                continue
            target_path = os.path.join(DATASETS_DIR, target_rel)
            if member.endswith("/"):
                os.makedirs(target_path, exist_ok=True)
            else:
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                with z.open(member) as src, open(target_path, "wb") as dst:
                    dst.write(src.read())

    os.remove(DATASETS_ZIP)
    print("  Datasets extracted and zip deleted.")

    # Back-compat: the published datasets.zip ships the HIMS-Tunisia data under
    # its old folder name ("migration/migration.csv"). The code now expects
    # "HIMS-Tunisia/HIMS-Tunisia.csv" — rename it in place if needed.
    _mig  = os.path.join(DATASETS_DIR, "migration")
    _hims = os.path.join(DATASETS_DIR, "HIMS-Tunisia")
    if os.path.isdir(_mig) and not os.path.isdir(_hims):
        _old_csv = os.path.join(_mig, "migration.csv")
        if os.path.exists(_old_csv):
            os.rename(_old_csv, os.path.join(_mig, "HIMS-Tunisia.csv"))
        os.rename(_mig, _hims)
        print("  Renamed legacy 'migration/' dataset folder -> 'HIMS-Tunisia/'")


# Step 3 — FFB results

step(3, "Download FFB benchmark results from WandB")

results_dir = os.path.join(ROOT, "fair_fairness_benchmark", "results")
existing    = [f for f in os.listdir(results_dir) if f.endswith(".json")
               and not f.startswith("HIMS-Tunisia_")] if os.path.exists(results_dir) else []

if len(existing) > 100:
    print(f"  {len(existing)} result files already present — skipping.")
else:
    run([PYTHON, os.path.join(ROOT, "download_ffb_wandb.py")])


# Done

print(f"""
{'='*60}
  SETUP COMPLETE
{'='*60}

  One manual step required — install Ollama:
    1. Download from  https://ollama.com
    2. Run:  ollama pull llama3.1

  Then start everything:
    Train your system:
      .venv\\Scripts\\python adversarial_fairness/main.py --dataset adult

    Launch dashboard:
      .venv\\Scripts\\python -m streamlit run unified_app.py
{'='*60}
""")
