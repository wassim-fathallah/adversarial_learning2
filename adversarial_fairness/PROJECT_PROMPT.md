# Full Project Prompt — Adversarial Fairness System

## Context

Fairness-aware adversarial machine learning system for a Master's thesis,
with potential paper publication.

Location: C:\Users\jamil\Desktop\adversarial_learning2\adversarial_fairness\

Environment:
- Python 3.14, venv at C:\Users\jamil\Desktop\adversarial_learning2\.venv
- Ollama running locally with llama3.1 pulled
- Packages: torch, langchain, langchain-ollama, langchain-core, pandas, numpy,
  scikit-learn, matplotlib, streamlit, plotly

---

## Run Commands

```bash
# Run ALL 7 standard datasets sequentially (default — no flags needed):
cd C:\Users\jamil\Desktop\adversarial_learning2\adversarial_fairness
python main.py

# Run a single dataset:
python main.py --dataset adult
python main.py --dataset adult --iterations 25 --epochs 50

# Run the Migration dataset (standalone only — not in ALL_DATASETS):
python main.py --dataset migration

# Streamlit dashboard (separate terminal):
python -m streamlit run streamlit_app.py
```

Startup: main.py pings http://localhost:11434 before doing anything.
If Ollama is not running → prints fix instructions → sys.exit(1).

Standard datasets run in order: adult → bank → compas → german → kdd → acs → utkface
Migration is NOT in ALL_DATASETS and must be run explicitly with --dataset migration.
Missing dataset files are skipped gracefully. Final summary table printed at end.

---

## Core Concept

Two neural networks compete in a min-max game (Zhang et al. 2018):

Classifier: predicts a target label (income, recidivism, credit risk...).
  Outputs a scalar probability Ŷ ∈ (0,1).
  Objective: minimize task loss AND minimize the adversary's ability to detect
  sensitive attributes from Ŷ.

Adversary: reads ONLY the classifier's scalar output Ŷ (not embeddings, not raw
  features) and tries to predict sensitive attributes (race, sex, age...).
  If the adversary can guess demographics from Ŷ, it means Ŷ carries demographic
  signal → the model is biased.

Training objective (Zhang et al. 2018):
  L_classifier = L_task - sum_s(lambda_s × L_adversary_s)
  L_adversary  = L_adversary  (maximized independently each step)

The minus sign is the key: classifier is REWARDED for confusing the adversary.
Lambda (one per sensitive attribute) controls the fairness-accuracy tradeoff.

---

## File Structure

adversarial_fairness/
├── main.py                  # entry point; auto-runs all 7 datasets or single via --dataset
├── orchestrator.py          # sequential 5-step pipeline (direct tool calls, no ReAct agent)
├── state.py                 # global singleton shared between all LangChain tools
├── streamlit_app.py         # dashboard reading long_term_memory.json
├── long_term_memory.json    # persisted across all runs
├── models/
│   ├── classifier.py        # 3-hidden-layer NN → scalar Ŷ
│   └── adversary.py         # reads scalar Ŷ → logits per sensitive attr
├── memory/
│   ├── short_term.py        # per-run trajectory (lambda/p_rule/acc per iteration)
│   └── long_term.py         # JSON across all runs, keyed by dataset+target+attrs
├── tools/
│   ├── data_tools.py        # identify_sensitive, load_dataset
│   ├── training_tools.py    # pretrain, run_full_training
│   ├── lambda_tools.py      # decide_initial_lambda, decide_lambda_for_iteration
│   └── proxy_tools.py       # proxy feature detection (unused — dead code, kept for reference)
├── datasets/
│   ├── adult/raw/
│   ├── bank_marketing/raw/
│   ├── compas/raw/
│   ├── german/raw/
│   ├── census_income_kdd/raw/
│   ├── acs/raw/
│   ├── utkface/raw/
│   └── migration/           # HIMS-Tunisia migration survey (added)
└── utils/
    ├── metrics.py           # all metrics
    └── plotting.py          # training curves PNG saved after each run

---

## Pipeline — 5 Steps (orchestrator.py)

```
identify_sensitive → load_dataset → decide_initial_lambda → pretrain → run_full_training
```

All tools are LangChain @tool decorated functions sharing `state` (global singleton).
Between datasets, reset_state() wipes everything clean.

Step 1 — identify_sensitive:
  LLM reads first 40 non-V columns + 1 sample value per column. Returns:
  sensitive_attrs, binarization_rules, target_col, columns_to_drop.
  Hardcoded fallbacks exist for the 7 standard datasets only.
  Migration has NO hardcoded fallback — LLM must succeed.

Step 2 — load_dataset:
  Full preprocessing pipeline (see Dataset Handling section).

Step 3 — decide_initial_lambda:
  No LLM. Returns λ = 0.0 for each sensitive attribute.

Step 4 — pretrain (10 epochs):
  Classifier and adversary trained independently. Warm start before competition.

Step 5 — run_full_training:
  The adversarial loop (up to 25 iterations × 50 epochs each).

---

## Neural Network Architecture

Classifier (models/classifier.py):
  Linear(n_features, 32) → ReLU → Dropout(0.2)
  Linear(32, 32)         → ReLU → Dropout(0.2)
  Linear(32, 32)         → ReLU → Dropout(0.2)
  Linear(32, 1)          → Sigmoid
  Output: scalar Ŷ ∈ (0,1), shape (N, 1)

Adversary (models/adversary.py):
  Input: scalar Ŷ (shape N×1) — only the final prediction, nothing else
  Linear(1, 32)  → ReLU
  Linear(32, 32) → ReLU
  Linear(32, n_sensitive)     ← one raw logit per sensitive attribute
  Loss: BCEWithLogitsLoss

Both models: n_hidden = 32. Adversary is intentionally small (32) so the
classifier can confuse it early, creating useful learning signal from the start.

---

## Training Loop — Exact Per-Batch Steps

### Pretraining (10 epochs, no fairness pressure)

```
Per batch:
  classifier.train()
  pred = classifier(X)
  loss = BCELoss(pred, y)          ← plain BCE, no weighting
  loss.backward() → clf_opt.step()

  classifier.eval()                ← dropout OFF: clean signal for adversary
  adversary.train()
  with no_grad: pred_det = classifier(X)
  loss_adv = BCEWithLogitsLoss(adversary(pred_det), Z)
  loss_adv.backward() → adv_opt.step()
```

### Adversarial loop (25 iterations × 50 epochs)

```
Per batch:

  STEP 1 — Adversary learns to spy:
    classifier.eval()              ← dropout OFF → clean, stable prediction
    adversary.train()
    with no_grad: Ŷ = classifier(X)
    loss_adv = BCEWithLogitsLoss(adversary(Ŷ), Z)
    loss_adv.backward() → adv_opt.step()

  STEP 2 — Classifier learns to hide:
    classifier.train()             ← dropout ON (regularization)
    adversary.eval()               ← dropout OFF → stable penalty gradient
    Ŷ = classifier(X)              ← gradient tracked
    L_task = BCELoss(Ŷ, y)        ← plain BCE, no class weighting
    adv_loss_per = BCEWithLogitsLoss(reduction=none)(adversary(Ŷ), Z)  # (N, n_attrs)
    penalty = (adv_loss_per × λ).sum(dim=1).mean()
    L_total = L_task - penalty
    L_total.backward() → clf_opt.step()
```

The gradient of L_total flows back through the adversary into Ŷ and into the
classifier weights. This is what forces Ŷ to stop leaking demographic information.

Key design: classifier.eval() during Step 1 means the adversary trains on clean
predictions (no dropout noise). This was the main bug in the original version —
both were in train() mode, which weakened the adversary and reduced fairness pressure.

Batch size = 32 (not 64 — smaller batches = more gradient updates per epoch).
Loss = plain BCELoss (not weighted — weighting inflates L_task ~3× on imbalanced
datasets, making λ too weak relative to the task loss).

---

## Lambda Adjustment — Per Iteration

Fully deterministic — no LLM involved. Pure momentum + running-max guard.

```python
gap       = (80.0 - current_P_rule) / 100.0      # how far from 80% target
increment = 2.5 × gap                             # LAMBDA_LEARNING_RATE = 2.5
momentum  = 0.7 × old_momentum + 0.3 × increment  # MOMENTUM_BETA = 0.7
λ_new     = clamp(λ + momentum, 0.0, 20.0)        # LAMBDA_MAX = 20.0

# Running-max guard:
if P_rule < threshold:
    λ_new = max(λ_new, best_λ_seen_so_far[attr])  # never drop while still unfair
# best_λ_seen_so_far[attr] updated every iteration
```

Running-max guard: while P-rule is still below 80%, λ is never allowed to drop
below the highest value it has ever reached for that attribute. Once P-rule >= 80%,
λ can decrease freely to recover accuracy.

Initial lambda: zero for each attribute.
Momentum will adjust from this starting point within the first few iterations.

This replaces the previous LLM-based per-iteration decision. The LLM was reading
short-term memory and returning a JSON lambda vector — removed because:
  - Momentum already handles the core signal (gap-driven adjustment)
  - Running-max guard handles the oscillation problem without needing an LLM
  - LLM calls added 10-30s latency per iteration and could fail silently

No λ-based stop condition. Loop ends only via early stop or max_iterations.

---

## Stopping Condition

Early stop (ideal — both conditions met simultaneously):
  accuracy >= 80%  AND  all P-rules >= 80%
  → status: "optimal"

If only accuracy >= 80% is met, training continues to maximize P-rule.
If only P-rule >= 80% is met, training continues to recover accuracy.

Best-result tracking across all iterations:
  PRIMARY   : among iterations where accuracy >= 0.80 → pick highest min_P-rule
              → status: "optimal" or "best_trade_off"
  FALLBACK  : if accuracy never reached 0.80 → pick highest accuracy seen
              → status: "best_accuracy_no_fairness"

This handles datasets like German credit (~75-78% natural accuracy ceiling) and
COMPAS (~68%) where requiring acc >= 80% is structurally impossible due to dataset
size, class distribution, and fairness-accuracy tension.

---

## Fairness Metric — P-rule (Disparate Impact)

```python
P-rule = min(P(Ŷ=1 | Z=0) / P(Ŷ=1 | Z=1),  reverse ratio) × 100
```

Measures: what fraction of positive predictions does the disadvantaged group get
compared to the advantaged group?
- P-rule = 100% → perfectly equal → fully fair
- P-rule = 50%  → disadvantaged group gets half as many positives → biased
- Target: >= 80% (the 80% rule from US Equal Employment Opportunity Commission)

---

## All Metrics (utils/metrics.py)

Performance:
  - Accuracy
  - Precision
  - F1 Score
  - ROC-AUC
  - Average Precision (AP) — better than ROC-AUC on imbalanced datasets

Fairness (computed per sensitive attribute):
  - P-rule (Disparate Impact Ratio) — primary target
  - Equalized Odds: TPR gap + FPR gap across groups
  - Equalized Opportunity: TPR gap only (positive class only)
  - Demographic Parity Difference: |P(Ŷ=1|Z=0) - P(Ŷ=1|Z=1)|
  - ABCC (Area Between Calibration Curves): calibration fairness per group

Note: Precision varies little during adversarial training. The fairness objective
redistributes *who* gets positive predictions (equalizing rates across groups),
not the quality of those predictions. The effects on precision cancel out at
the aggregate level.

---

## Memory System

Short-term (memory/short_term.py):
  Per-run trajectory. Each iteration stores: lambda_vector, p_rules per attr,
  accuracy, adversary_loss, clf_task_loss.
  Used for: trend detection (get_trend per attr) and Streamlit dashboard data
  via iteration_metrics. Not fed to any LLM.

Long-term (long_term_memory.json):
  Persisted JSON across all runs.
  Key: "{dataset_name}|{target_col}|{attr1,attr2,...}"
  Stores last 10 runs per key: lambda_final, p_rules_final, accuracy_final,
  total_epochs, iterations, success, lambda_trajectory, iteration_metrics.
  Used by: Streamlit dashboard.

---

## Dataset Handling (tools/data_tools.py)

### LLM Schema Building

Before calling the LLM, `identify_sensitive` builds a compact schema:
  1. Filter out V-prefixed columns (`^V\d` regex) — raw survey codes used in
     migration dataset; these are meaningless to the LLM.
  2. Cap to first 40 columns (first 40 contain demographics for migration;
     last columns contain post-arrival outcome indices).
  3. One sample value per column shown to keep prompt short.
  4. LLM called via direct `ollama.generate()` client (NOT langchain OllamaLLM),
     with `options={"num_gpu": 0, "temperature": 0.1}` to avoid GPU OOM on large
     schema prompts. This bypasses langchain to pass driver-level options correctly.

### LLM Prompt Strategy

The prompt guides the LLM without hardcoding any column names:
  - Target: must be a legal/administrative/socioeconomic STATUS outcome.
    Explicitly excluded: geographic destination, travel route, country visited.
  - Sensitive attrs: must be pre-existing demographic characteristics from BEFORE
    the studied event (gender, geographic ORIGIN, education level category).
    Explicitly excluded: "current" columns (current country/city/job), destination,
    computed index/score columns, binary flag derivations of an already-listed attr.
  - Diversity rule: 3 attrs must span different dimensions:
    (a) gender/sex, (b) geographic origin, (c) socioeconomic background.
    Do NOT pick two attributes measuring the same concept (e.g., two origin columns).
  - Prefer original multi-category columns over binary derived versions.

### Preprocessing Pipeline

  1. Auto-detect separator (comma / semicolon / whitespace+comma)
  2. Assign column names for headerless files (adult, kdd) from built-in schema_map
  3. Drop V-prefixed columns for migration (273 raw survey code columns removed)
  4. Drop identified columns (IDs, weights, leakage)
  5. Binarize target: binary → sort+pick; numeric → median split; categorical → code+median
  6. Binarize sensitive attributes per LLM rules
  7. Expand 'pixels' column for UTKFace (np.fromstring per row)
  8. Drop high-cardinality string columns (>50 unique OR >10% of rows)
  9. Fill missing values (median for numeric, "Unknown" for categorical)
  10. Subsample to 100K rows if dataset exceeds that
  11. One-hot encode remaining categoricals
  12. Drop near-constant columns (std < 1e-6)
  13. StandardScaler + nan_to_num safety net
  14. 80/20 stratified split, random_state=42

---

## Supported Datasets

Standard (all 7 run automatically with `python main.py`):

  adult   : datasets/adult/raw/adult.data
            target: income | sensitive: sex (Male=1), race (White=1)
            headerless — 15-column schema assigned internally

  german  : datasets/german/raw/german_credit_risk.csv
            target: Class | sensitive: Sex (male=1), Age (>25=1)
            natural accuracy ceiling ~75-78% (small dataset, 1000 rows)

  compas  : datasets/compas/raw/compas-scores-two-years.csv
            target: two_year_recid | sensitive: race (Caucasian=1), sex (Male=1)
            natural accuracy ceiling ~68%

  utkface : datasets/utkface/raw/age_gender.csv
            target: age (>median=1) | sensitive: ethnicity (0=1), gender (0=1)
            pixels column expanded via np.fromstring (memory-efficient)

  kdd     : datasets/census_income_kdd/raw/census-income.data
            target: income | sensitive: race (White=1), sex (Male=1)
            headerless — 42-column schema assigned internally

  bank    : datasets/bank_marketing/raw/bank-additional-full.csv
            target: y | sensitive: age (>40=1), marital (married=1)

  acs     : datasets/acs/raw/2018/1-Year/psam_p06.csv
            target: PINCP (>median=1) | sensitive: RAC1P (1=White), SEX (1=Male)
            drops: SERIALNO, SPORDER, PWGTP, PWGTP1–PWGTP80 (replicate weights)
            subsampled to 100K (full file: ~378K rows, 280+ columns)

Standalone (run separately with --dataset migration):

  migration : datasets/migration/migration.csv
              Source: HIMS-Tunisia survey (High Impact Migration Survey)
              Rows: 3161 | Raw columns: ~305 (273 V-prefixed survey codes + ~32 clean)
              Target: legal_entry (legal authorization/entry status — LLM-identified)
              Sensitive: 3 attrs — LLM-identified, typically from:
                Gender, coastal_origin / region_origin, educ_level
              NO hardcoded fallback — LLM must identify attributes from schema.
              Special preprocessing:
                - V-prefixed columns dropped before loading (raw survey codes)
                - First 40 columns shown to LLM (demographics are in early columns)
                - LLM asked to pick 3 attrs across gender / geographic origin /
                  education dimensions (no column names hardcoded)
              Run: python main.py --dataset migration

---

## Streamlit Dashboard (streamlit_app.py)

Run: python -m streamlit run streamlit_app.py
Reads long_term_memory.json, refreshes every 10 seconds (cache TTL=10).

Layout:
  - One tab per dataset (appears automatically once a run is saved to memory)
  - Sub-tabs per run (Run 1 date, Run 2 date, ...)
  - Each run: summary card (accuracy, P-rule per attr, status, epochs) +
    4-subplot interactive chart:
      Row 1: Accuracy / F1 / ROC-AUC over iterations (ref line at 80%)
      Row 2: P-rule per sensitive attr (ref line at 80%)
      Row 3: Lambda per sensitive attr
      Row 4: Adversary loss

Benchmark Comparison Panel:
  The dashboard includes a comparison against the fair_fairness_benchmark
  (WandB: https://wandb.ai/fair_benchmark/exp1.adv_gr).
  Overlapping tabular datasets (adult, bank_marketing, german, compas, kdd, acs)
  plus the image dataset utkface are shown side-by-side with the benchmark's
  reported metrics so the thesis comparison is directly visible in the UI.
  KDD (census-income KDD) and ACS (ACS-Income, California) FFB results were
  pulled from WandB via download_ffb_wandb.py — full 5-method sweep (ERM,
  AdvDebias, LAFTR, HSIC, PrejudiceRemover) over race & sex, 10 seeds.
  The FFB tab auto-discovers every dataset present in
  fair_fairness_benchmark/results/, so new downloads appear with no code change.
  Benchmark metrics tracked: accuracy, precision, ROC-AUC, Average Precision,
  DP (demographic parity), P-rule, ABCC (ddss).
  The "Max-Acc / Max-Prule" sub-tab and compare_ffb.py extract three operating
  points per (method, attribute) — Max-Acc, Max-Prule, Trade-off — reporting
  Acc, P-rule, ΔDP, ΔEOdd, ΔEOpp (mean over seeds).

---

## LangChain Integration

LLM: llama3.1 via local Ollama
  - Called via direct `ollama.generate()` (NOT langchain OllamaLLM) to pass
    `num_gpu=0` at the driver level, preventing CUDA OOM on large schema prompts.
  - temperature=0.1

All 5 pipeline functions are @tool decorated.
Pipeline called sequentially in orchestrator.py — NOT via ReAct agent.

LLM used for ONE thing only:
  1. identify_sensitive — reads first 40 non-V columns + 1 sample value per column
     → returns sensitive attrs, binarization rules, target col, columns to drop.
     Hardcoded fallbacks cover all 7 standard datasets if LLM fails or picks wrong columns.
     Migration: no fallback — LLM failure raises RuntimeError with root cause shown.

LLM guardrails in identify_sensitive:
  - If suggested target_col does not exist in dataset → use fallback (standard datasets only)
  - If suggested sensitive attrs are in the known drop list (e.g. PWGTP1/PWGTP2 for ACS)
    or match weight column patterns → use fallback

NOT used for:
  - decide_initial_lambda → zero per attribute
  - decide_lambda_for_iteration → pure momentum + running-max guard (deterministic)

---

## Known Observations & Limitations

- Most datasets converge within 4 iterations when all fixes are in place.
  The training loop still runs all 25 to find the optimal trade-off point.
- Precision barely varies: adversarial fairness redistributes *who* gets a positive
  prediction (demographic parity), not the quality. Effects cancel at aggregate level.
- German / COMPAS: accuracy can't reach 80% → reported as "best_trade_off" or
  "best_accuracy_no_fairness". This is expected, not a bug.
- UTKFace: 2304 pixel features → slow training. PCA or pretrained embeddings
  would be the proper approach for a real deployment.
- KDD / ACS: heavy class imbalance (KDD: 93.8% negative). Plain BCELoss can
  cause the model to predict all-negative → P-rule artificially 100%. Monitor F1.
- ACS: 280+ columns including coded numeric fields (NAICSP, SOCP) dropped by
  the high-cardinality filter.
- Migration: LLM attribute identification is sensitive to column ordering and prompt
  wording. The LLM tends to pick multiple geographic-origin columns if not guided
  toward diversity across dimensions. Re-runs may pick slightly different attrs since
  there is no hardcoded fallback.
- Each Ollama LLM call takes 10–30s on CPU. GPU speeds this up significantly.
  GPU OOM is prevented by passing num_gpu=0 via direct ollama client.

---

## Bugs Fixed vs. Original Notebook (CODE_FINAL_COMPLETE (2).ipynb)

| Issue                    | Before (original)              | After (current)                          |
|--------------------------|--------------------------------|------------------------------------------|
| eval/train mode          | both in train() entire batch   | clf.eval() in adv step, adv.eval() in clf step |
| Task loss                | weighted BCELoss (pos_weight)  | plain nn.BCELoss()                       |
| Batch size               | 64                             | 32                                       |
| Adversary hidden size    | 64                             | 32                                       |
| Stopping condition       | acc >= 70% AND P-rule >= 80%   | acc >= 80% AND P-rule >= 80% (with optimal trade-off tracking) |
| Multi-dataset runner     | separate command per dataset   | python main.py runs all 7 automatically  |
| Ollama check             | no check                       | startup ping → exit if not running       |
| Lambda control           | LLM per iteration              | deterministic momentum + running-max guard |
| GPU OOM on large schemas | crashed on migration dataset   | direct ollama client with num_gpu=0      |

---

## Comparison Goal

Comparing against: https://wandb.ai/fair_benchmark/exp1.adv_gr
Metrics tracked there: test/acc, test/precision, test/roc, train/AP,
                       test/dp, test/prule, test/ddss (ABCC)
Datasets: adult, bank_marketing, german, compas

Strategy:
  - Same datasets, same metrics
  - Best run vs best run (accepted in fairness ML literature)
  - Our extras: F1, Equalized Odds, Equalized Opportunity, multi-dataset automation,
    migration dataset (novel — not in benchmark), Streamlit dashboard for thesis demo
  - Benchmark results shown directly in Streamlit for side-by-side comparison

---

## Theoretical Reference

Zhang, B. H., Lemoine, B., & Mitchell, M. (2018).
Mitigating Unwanted Biases with Adversarial Learning. AAAI 2018.
arXiv:1801.07593

Key equation (Eq. 1) — classifier gradient update:
  ∇W LP  −  proj_{∇W LA} ∇W LP  −  α∇W LA

Our implementation uses the simplified version (last term only, no projection).
The projection term from the paper is not implemented — known gap.
α in the paper = lambda in our code.
pos_weight for class imbalance is NOT used — plain BCELoss throughout.
