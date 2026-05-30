# Adversarial Fairness — Thesis System + FFB Benchmark

A unified fairness framework combining:
- **Agentic Adversarial Debiasing** — a multi-agent adversarial fairness pipeline (Zhang et al. 2018 + LangChain orchestration)
- **FFB Benchmark** — pre-computed results from the Fair Fairness Benchmark (Han et al. ICLR 2024)

---

## Requirements

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.10+ | For the main system |
| Git | any | To clone the repo |
| Ollama | latest | For LLM-based attribute detection |
| llama3.2:3b | — | Lightweight model, ~2 GB RAM |
| Conda (optional) | any | Only needed to re-run FFB training |

---

## 1 — Clone the repository

```bash
git clone https://github.com/yourname/adversarial_learning2
cd adversarial_learning2
```

---

## 2 — Install Ollama + model

Ollama is required for Step 1 of the pipeline (sensitive attribute identification).

**Download:** https://ollama.com → install for your OS

**Pull the model:**
```bash
ollama pull llama3.2:3b
```

> `llama3.2:3b` requires ~2 GB RAM and is sufficient for the task.
> Do NOT use `llama3.1` (8B) unless you have ≥16 GB RAM free.

**Start Ollama before running training:**
```bash
ollama serve
```

---

## 3 — Set up Python environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

---

## 4 — Download datasets

**Download link:** [datasets.zip on Dropbox](https://www.dropbox.com/scl/fi/6t0jh1n5v7y2tr35bkidq/datasets.zip?rlkey=2xkqwavfcqb56c7ybavzcm9ga&st=93ediu4h&dl=1)

Download and extract into `adversarial_fairness/datasets/` so the structure looks like:

```
adversarial_fairness/datasets/
├── adult/raw/adult.data
├── german/raw/german_credit_risk.csv
├── compas/raw/compas-scores-two-years.csv
├── bank_marketing/raw/bank-additional-full.csv
├── census_income_kdd/raw/census-income.data
├── acs/raw/2018/1-Year/psam_p06.csv
├── utkface/raw/age_gender.csv
└── migration/migration.csv
```

Or run the automated setup which downloads everything:
```bash
python setup.py
```

---

## 5 — Download FFB benchmark results

```bash
python download_ffb_wandb.py
```

This fetches pre-computed results from the FFB paper's public WandB projects.
No GPU needed. Downloads final metrics only (~17 MB, ~20 min).

Methods: ERM, AdvDebias, PR, HSIC, LAFTR
Datasets: adult, german, compas, bank, migration (tabular) + UTKFace (image)

> Safe to stop and restart — already-downloaded files are skipped automatically.

---

## 6 — Run the Agentic Adversarial Debiasing system

Make sure Ollama is running (`ollama serve`), then:

```bash
# Single dataset
.venv\Scripts\python adversarial_fairness/main.py --dataset adult
.venv\Scripts\python adversarial_fairness/main.py --dataset migration

# All 7 datasets sequentially
.venv\Scripts\python adversarial_fairness/main.py

# Custom parameters
.venv\Scripts\python adversarial_fairness/main.py --dataset adult --iterations 25 --epochs 50 --threshold 80
```

**Available datasets:** `adult` `german` `compas` `bank` `kdd` `acs` `utkface`

Results are saved automatically to `adversarial_fairness/long_term_memory.json`.

---

## 7 — Launch the unified dashboard

```bash
.venv\Scripts\python -m streamlit run unified_app.py
```

Opens at `http://localhost:8501`

**Tabs:**
- `⚙️ Agentic Adversarial Debiasing` — your training results per dataset
- `📊 FFB Benchmark` — pre-computed results from the FFB paper

---

## Automated setup (fresh machine — does steps 3, 4, 5 automatically)

```bash
python setup.py
```

Then manually: install Ollama + pull llama3.2:3b (step 2).

---

## Project structure

```
adversarial_learning2/
├── adversarial_fairness/        ← Agentic Adversarial Debiasing system
│   ├── main.py                  ← Entry point
│   ├── orchestrator.py          ← 5-step pipeline
│   ├── state.py                 ← Global state shared across tools
│   ├── models/
│   │   ├── classifier.py        ← Neural network classifier
│   │   └── adversary.py         ← Adversary network
│   ├── tools/
│   │   ├── data_tools.py        ← Dataset loading + LLM attribute detection
│   │   ├── training_tools.py    ← Pretraining + adversarial loop
│   │   └── lambda_tools.py      ← Momentum-based lambda adjustment
│   ├── memory/
│   │   ├── short_term.py        ← Per-run iteration trajectory
│   │   └── long_term.py         ← Cross-session JSON persistence
│   └── utils/
│       ├── metrics.py           ← Fairness + performance metrics
│       └── plotting.py          ← Training curve plots
├── fair_fairness_benchmark/     ← FFB benchmark (Han et al. ICLR 2024)
│   ├── src/                     ← FFB training scripts
│   └── results/                 ← Downloaded benchmark results (JSON)
├── unified_app.py               ← Streamlit dashboard
├── setup.py                     ← Automated setup script
├── download_ffb_wandb.py        ← FFB results downloader
├── requirements.txt             ← Python dependencies (venv)
└── requirements_ffb.txt         ← FFB training dependencies (conda)
```

---

## Re-running FFB training (optional — results already included)

FFB uses a separate conda environment:

```bash
# First-time setup
conda create -n ffb_env python=3.10
conda activate ffb_env
pip install -r requirements_ffb.txt

# Run migration dataset benchmark
conda activate ffb_env
python fair_fairness_benchmark/run_benchmark.py --no_wandb
python fair_fairness_benchmark/run_benchmark.py --quick --no_wandb   # fast test
```

---

## References

- Zhang et al. (2018). *Mitigating Unwanted Biases with Adversarial Learning.* AIES 2018.
- Han et al. (2024). *FFB: A Fair Fairness Benchmark for In-Processing Group Fairness Methods.* ICLR 2024. [GitHub](https://github.com/ahxt/fair_fairness_benchmark) (MIT License)
