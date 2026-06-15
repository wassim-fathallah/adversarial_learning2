# Adversarial Fairness — Thesis System + FFB Benchmark

A unified fairness framework combining:
- **Agentic Adversarial Debiasing** — our multi-agent adversarial fairness pipeline
- **FFB Benchmark** — the Fair Fairness Benchmark (Han et al. ICLR 2024), as a comparison baseline

This README is a **run guide + brief method overview**: what AADA is, how to set
up the project, then run each part (our method, the FFB benchmark, the
comparison, the dashboard) **separately**, including how to insert your own dataset.

---

## How it works (AADA in brief)

AADA reframes adversarial debiasing as **three cooperating agents**:

- **Orchestrator** (`orchestrator.py` → `OrchestratorAgent`) — reads the dataset
  schema, uses a locally-served **Llama 3.1 via Ollama** (`langchain`) to identify
  the target and sensitive attributes, configures the pipeline, and at every
  iteration updates a **per-attribute penalty vector λ** and checks the fairness
  target. It also stores a memory of past runs (dataset fingerprint + best λ).
- **Classifier** (utility-based agent) and **Adversary** (goal-based agent) —
  the minimax pair (`models/agents.py`). The adversary sees only the classifier's
  output Ŷ and tries to recover the sensitive attributes.

Instead of a fixed penalty α chosen by a manual sweep, λ adapts **online** via a
**momentum update with a running-maximum guard** (fully deterministic — no LLM in
the training loop). A single adaptive run covers **all** sensitive attributes at
once. Selection is **fairness-first**: training stops once the P-rule target (80%,
the EEOC four-fifths rule) is met on every attribute, and the most accurate
qualifying model is kept.

**Momentum coefficient β = 0.7 (not the usual 0.9).** We ran the full 10-seed
sweep at the common industry-default **β = 0.9** *and* at **β = 0.7**, and found
**β = 0.7 converged in fewer iterations and gave a better fairness–accuracy
trade-off** (β = 0.9 over-smooths the non-stationary adversarial signal). β = 0.7
is therefore the default. The two seed sweeps are kept in
`adversarial_fairness/beta07_test_memory.json` (β = 0.7) and
`beta09_test_memory.json` (β = 0.9); the momentum effect is plotted in
`adversarial_fairness/fig_momentum_convergence.png` (run `plot_momentum.py`).

---

## Requirements

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.10 – 3.12 (3.11 recommended) | For the main system. CPU is fine for testing; a CUDA GPU only speeds up the large datasets (kdd/acs/utkface). |
| Git | any | To clone the repo |
| Ollama | latest | For LLM-based attribute detection (our method only) |
| llama3.1 | — | 8B model, needs ~16 GB RAM free |
| Conda (optional) | any | **Not needed to review** — FFB results already ship in the repo. Only needed to *re-run* FFB training from scratch. |

---

## Setup (do once)

### 1 — Clone

```bash
git clone https://github.com/wassim-fathallah/adversarial_learning2
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
        └── HIMS-Tunisia/HIMS-Tunisia.csv
```

> ⚠️ **One location for ALL datasets** — both our system and the FFB scripts read
> from `adversarial_fairness/datasets/`. Or run `python setup.py` to download +
> extract automatically.
>
> 📌 **If you extract the zip manually** and see a `migration/migration.csv`
> folder, rename it to `HIMS-Tunisia/HIMS-Tunisia.csv` (the dataset was renamed).
> `python setup.py` does this rename for you automatically.

---

## Quick test (verify the install)

Most checks need **no Ollama and no GPU** — they use the results already
committed to the repo. From the repo root:

```bash
# 1. Core package imports cleanly
.venv\Scripts\python -c "import sys; sys.path.insert(0,'adversarial_fairness'); import orchestrator, main, compare_ffb; print('imports OK')"

# 2. Read the saved multi-seed results (mean ± std across seeds)
.venv\Scripts\python adversarial_fairness/seeds/aggregate_seeds.py --dataset adult

# 3. Regenerate the paper figures from the shipped data
.venv\Scripts\python adversarial_fairness/plot_momentum.py        # momentum λ-trajectory
.venv\Scripts\python fair_fairness_benchmark/make_aada_vs_ffb.py  # AADA vs FFB operating points

# 4. Launch the dashboard (reads the shipped results)
.venv\Scripts\python -m streamlit run unified_app.py
```

To test the **full pipeline end-to-end** (needs Ollama running + the dataset
downloaded), run one small dataset:

```bash
.venv\Scripts\python adversarial_fairness/main.py --dataset adult --seed 14159
```

> ℹ️ A full run **appends** to `long_term_memory.json` (capped at 10 runs/dataset).
> To keep the published results pristine while testing, copy that file first.

---

## A — Run OUR method (Agentic Adversarial Debiasing)

> Requires Ollama running (`ollama serve`). All commands assume the venv Python.

```bash
# One built-in dataset
.venv\Scripts\python adversarial_fairness/main.py --dataset adult
.venv\Scripts\python adversarial_fairness/main.py --dataset HIMS-Tunisia

# All built-in datasets, one after another
.venv\Scripts\python adversarial_fairness/main.py

# Tune the run
.venv\Scripts\python adversarial_fairness/main.py --dataset adult \
    --iterations 25 --epochs 50 --pretrain 10 --threshold 80
```

**Built-in dataset keys:** `adult` `german` `compas` `bank` `kdd` `acs` `utkface` `HIMS-Tunisia`

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

> 📊 **Where the baseline results come from.** The FFB baseline operating points
> shipped in `fair_fairness_benchmark/results/` (ERM, AdvDebias, PR, HSIC, LAFTR)
> were **extracted from the official Fair Fairness Benchmark** public WandB
> projects (Han et al., ICLR 2024) — re-fetch them anytime with
> `download_ffb_wandb.py` (public, no account needed). The aggregated per-method
> tables (`ERM_all.csv`, `ADV_all.csv`, `PRALL.csv`, `HSIC_all.csv`,
> `LAFTR_all.csv`) are built from those runs. The **HIMS-Tunisia** dataset (new,
> absent from FFB) and the **UTKFace adversarial** baseline were produced by us
> using the same FFB implementations under `fair_fairness_benchmark/`.

> ✅ **You can skip this whole section to review the work.** The FFB result
> files already ship in the repo (`fair_fairness_benchmark/results/*.json`,
> committed), so the comparison in section D works out of the box — no WandB
> account, no conda env, no `requirements_ffb.txt` needed. Only do C if you want
> to *regenerate* the FFB numbers from scratch.

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

**HIMS-Tunisia dataset (full sweep driver):**
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

## Hardware & compute

All experiments were run on a single modest laptop GPU — no cluster, every run
sequential:

| Component | Spec |
|-----------|------|
| GPU | NVIDIA GeForce GTX 1070 — 8 GB VRAM |
| CPU | Intel Core i7-8750H |
| System RAM | 16 GB |
| Logical processors | 12 |

Because runs executed one after another on this one GPU, the seed program was
time-consuming. Counting the stored runs:

| Momentum sweep | Runs | Adversarial iterations | Training epochs |
|----------------|-----:|-----------------------:|----------------:|
| β = 0.7 (default) | 66 | 756 | 36,076 |
| β = 0.9 (industry default, for comparison) | 66 | 1,008 | 48,300 |
| **Total** | **132** | **1,764** | **84,376** |

That is **132 full AADA runs (~84k training epochs)** on a single GTX 1070, which
took **on the order of several days of wall-clock time**. It also underlines why
AADA's **single adaptive run per seed** matters: the fixed-penalty baselines need
a 14-value α sweep × (number of sensitive attributes) *per dataset per seed* — up
to 42 runs/seed on HIMS-Tunisia — whereas AADA replaces that whole sweep with one
run.

---

## Project structure

```
adversarial_learning2/
├── adversarial_fairness/        ← OUR method
│   ├── main.py                  ← Entry point (section A/B)
│   ├── orchestrator.py          ← OrchestratorAgent — 5-step pipeline
│   ├── state.py                 ← Global state shared across tools
│   ├── compare_ffb.py           ← Comparison tables (section D)
│   ├── plot_momentum.py         ← Momentum λ-trajectory figure
│   ├── long_term_memory.json    ← Saved runs (seed results + fingerprints)
│   ├── models/                  ← classifier.py, image_classifier.py, adversary.py, agents.py
│   ├── tools/                   ← data_tools, training_tools, lambda_tools
│   ├── memory/                  ← short_term.py, long_term.py
│   ├── utils/                   ← metrics.py, plotting.py
│   └── seeds/                   ← multi-seed run scripts (run_multiple_seeds.py, aggregate_seeds.py, …)
├── fair_fairness_benchmark/     ← FFB benchmark (section C)
│   ├── run_benchmark.py         ← HIMS-Tunisia sweep driver
│   ├── make_aada_vs_ffb.py      ← operating-point figure (fig_aada_vs_ffb.png)
│   ├── src/                     ← ffb_tabular_*.py method scripts
│   └── results/                 ← FFB result JSONs (committed — ship with the repo)
├── unified_app.py               ← Dashboard (section E)
├── download_ffb_wandb.py        ← FFB results downloader (optional, section C.1)
├── setup.py                     ← Automated dataset setup
├── requirements.txt             ← venv deps (our method + dashboard)
└── requirements_ffb.txt         ← FFB-retraining deps (OPTIONAL — see section C)
```

---

## References

- Zhang et al. (2018). *Mitigating Unwanted Biases with Adversarial Learning.* AIES 2018.
- Han et al. (2024). *FFB: A Fair Fairness Benchmark for In-Processing Group Fairness Methods.* ICLR 2024. [GitHub](https://github.com/ahxt/fair_fairness_benchmark) (MIT License)
```
