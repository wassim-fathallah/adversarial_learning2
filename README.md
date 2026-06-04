# Adversarial Fairness — Thesis System + FFB Benchmark

A unified fairness framework combining:
- **Agentic Adversarial Debiasing** — our multi-agent adversarial fairness pipeline
- **FFB Benchmark** — the Fair Fairness Benchmark (Han et al. ICLR 2024), as a comparison baseline

This README is a **run guide**: how to set up the project, then run each part
(our method, the FFB benchmark, the comparison, the dashboard) **separately**,
including how to insert your own dataset.

---

## Requirements

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.10+ | For the main system |
| Git | any | To clone the repo |
| Ollama | latest | For LLM-based attribute detection (our method only) |
| llama3.1 | — | 8B model, needs ~16 GB RAM free |
| Conda (optional) | any | Only needed to **re-run** FFB training |

---

## Setup (do once)

### 1 — Clone

```bash
git clone https://github.com/yourname/adversarial_learning2
cd adversarial_learning2
```

### 2 — Python environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3 — Ollama + model (needed only for OUR method, not for FFB)

Ollama powers Step 1 of our pipeline (sensitive-attribute identification).

1. Install from https://ollama.com
2. Pull the model:
   ```bash
   ollama pull llama3.1
   ```
   > `llama3.1` (8B) needs ~16 GB RAM free. If Ollama fails with
   > `unable to allocate CPU buffer`, pull a smaller model
   > (`ollama pull llama3.2:3b`) and set `OLLAMA_MODEL` in
   > `adversarial_fairness/tools/data_tools.py`.
3. Start the server (leave it running in its own terminal):
   ```bash
   ollama serve
   ```

### 4 — Datasets (one location for everything)

[**Download datasets.zip (Dropbox)**](https://www.dropbox.com/scl/fi/6t0jh1n5v7y2tr35bkidq/datasets.zip?rlkey=2xkqwavfcqb56c7ybavzcm9ga&st=93ediu4h&dl=1)

Extract into `adversarial_fairness/datasets/` so the structure is:

```
adversarial_learning2/
└── adversarial_fairness/
    └── datasets/                 ← extract here
        ├── adult/raw/adult.data
        ├── german/raw/german_credit_risk.csv
        ├── compas/raw/compas-scores-two-years.csv
        ├── bank_marketing/raw/bank-additional-full.csv
        ├── census_income_kdd/raw/census-income.data
        ├── acs/raw/2018/1-Year/psam_p06.csv
        ├── utkface/raw/age_gender.csv
        └── migration/migration.csv
```

> ⚠️ **One location for ALL datasets** — both our system and the FFB scripts read
> from `adversarial_fairness/datasets/`. Or run `python setup.py` to download +
> extract automatically.

---

## A — Run OUR method (Agentic Adversarial Debiasing)

> Requires Ollama running (`ollama serve`). All commands assume the venv Python.

```bash
# One built-in dataset
.venv\Scripts\python adversarial_fairness/main.py --dataset adult
.venv\Scripts\python adversarial_fairness/main.py --dataset migration

# All built-in datasets, one after another
.venv\Scripts\python adversarial_fairness/main.py

# Tune the run
.venv\Scripts\python adversarial_fairness/main.py --dataset adult \
    --iterations 25 --epochs 50 --pretrain 10 --threshold 80
```

**Built-in dataset keys:** `adult` `german` `compas` `bank` `kdd` `acs` `utkface` `migration`

**Flags** (`main.py`):

| Flag | Default | Meaning |
|------|---------|---------|
| `--dataset` | (all) | Built-in key **or a path to your own CSV** (see section B) |
| `--name` | (auto) | Override the memory key name |
| `--target` | (LLM picks) | Force the target column |
| `--iterations` | 25 | Max adversarial iterations |
| `--epochs` | 50 | Epochs per adversarial step |
| `--pretrain` | 10 | Pre-training epochs |
| `--threshold` | 80 | **P-rule** fairness target (four-fifths rule) |
| `--device` | (auto) | `cpu` or `cuda` |

> **Selection is fairness-first.** Accuracy is *not* limited. Training stops once
> the P-rule target is met, and among the iterations that meet it the
> **highest-accuracy** model is selected.

Results (metrics, lambda trajectory, dataset fingerprint) are saved to
`adversarial_fairness/long_term_memory.json`.

---

## B — Insert and run YOUR OWN dataset

No registration needed — just point `--dataset` at a CSV. The LLM orchestrator
reads the schema and picks the target + sensitive attributes automatically.

```bash
.venv\Scripts\python adversarial_fairness/main.py --dataset "C:/path/to/mydata.csv"

# Force the target column if the LLM's pick is wrong:
.venv\Scripts\python adversarial_fairness/main.py \
    --dataset "C:/path/to/mydata.csv" --target my_label_column
```

Requirements for the CSV:
- Tabular, one row per instance, a header row with column names.
- A binary (or binarizable) target column.
- At least one sensitive/demographic column (the LLM detects these; categorical
  values are binarized automatically).

The memory key is derived from the file name; use `--name` to set it explicitly.

---

## C — Run the FFB benchmark

The FFB scripts do **not** need Ollama. Two ways to get FFB numbers:

### C.1 — Download the published FFB results (fast, recommended)

Fetches the paper's results into `fair_fairness_benchmark/results/*.json`:

```bash
.venv\Scripts\python download_ffb_wandb.py            # all methods/datasets
.venv\Scripts\python download_ffb_wandb.py --quick    # first 20 runs each (test)
.venv\Scripts\python download_ffb_wandb.py --project exp1.erm   # single method
```

### C.2 — Re-run FFB training yourself (slow; separate conda env)

```bash
conda create -n ffb_env python=3.10
conda activate ffb_env
pip install -r requirements_ffb.txt
```

**Migration dataset (full sweep driver):**
```bash
python fair_fairness_benchmark/run_benchmark.py --no_wandb            # full sweep
python fair_fairness_benchmark/run_benchmark.py --quick --no_wandb    # fast test
python fair_fairness_benchmark/run_benchmark.py --method hsic --no_wandb   # one method
python fair_fairness_benchmark/run_benchmark.py --sens sex --no_wandb      # one attribute
```

**Any other dataset (run a single FFB method script directly):**
```bash
cd fair_fairness_benchmark/src
python ffb_tabular_adv.py   --dataset adult --sensitive_attr sex --lam 1.0
python ffb_tabular_hsic.py  --dataset adult --sensitive_attr race --lam 100
python ffb_tabular_erm.py   --dataset adult --sensitive_attr sex
```
> Method scripts: `ffb_tabular_{erm,adv,hsic,pr,laftr,diffdp,diffeopp,diffeodd}.py`.
> LAFTR uses `--A_z` instead of `--lam`. See each script's `--help` for all flags.

---

## D — Compare OUR method vs FFB (tables)

After you have results from both A and C, generate the comparison tables
(console + LaTeX + CSV):

```bash
.venv\Scripts\python adversarial_fairness/compare_ffb.py
```

Writes `comparison_tables.tex` and `comparison_data.csv` into `adversarial_fairness/`.

---

## E — Launch the dashboard

```bash
.venv\Scripts\python -m streamlit run unified_app.py
```

Opens at `http://localhost:8501` with three tabs:
- **⚙️ Agentic Adversarial Debiasing** — our results (`long_term_memory.json`)
- **📊 FFB Benchmark** — FFB results (`fair_fairness_benchmark/results/`)
- **🔬 Comparison** — both side by side

---

## Project structure

```
adversarial_learning2/
├── adversarial_fairness/        ← OUR method
│   ├── main.py                  ← Entry point (section A/B)
│   ├── orchestrator.py          ← 5-step pipeline
│   ├── state.py                 ← Global state shared across tools
│   ├── compare_ffb.py           ← Comparison tables (section D)
│   ├── long_term_memory.json    ← Saved runs
│   ├── models/                  ← classifier.py, adversary.py
│   ├── tools/                   ← data_tools, training_tools, lambda_tools
│   ├── memory/                  ← short_term.py, long_term.py
│   └── utils/                   ← metrics.py, plotting.py
├── fair_fairness_benchmark/     ← FFB benchmark (section C)
│   ├── run_benchmark.py         ← Migration sweep driver
│   ├── src/                     ← ffb_tabular_*.py method scripts
│   └── results/                 ← FFB result JSONs
├── unified_app.py               ← Dashboard (section E)
├── download_ffb_wandb.py        ← FFB results downloader (section C.1)
├── setup.py                     ← Automated dataset setup
├── requirements.txt             ← venv deps (our method + dashboard)
└── requirements_ffb.txt         ← conda deps (FFB training only)
```

---

## References

- Zhang et al. (2018). *Mitigating Unwanted Biases with Adversarial Learning.* AIES 2018.
- Han et al. (2024). *FFB: A Fair Fairness Benchmark for In-Processing Group Fairness Methods.* ICLR 2024. [GitHub](https://github.com/ahxt/fair_fairness_benchmark) (MIT License)
```
