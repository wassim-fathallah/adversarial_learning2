# Agentic Adversarial Debiasing (AADA)

Three cooperating agents (`orchestrator.py`, `models/agents.py`):

- **Orchestrator** (`OrchestratorAgent`) — the controller. It reads the dataset
  schema, uses **Llama 3.1 via Ollama** (LangChain) to identify the target and
  the sensitive attributes, configures the pipeline, and at every iteration
  updates a per-attribute fairness penalty **λ**, checks the stopping condition,
  selects the returned model, and remembers the run (dataset fingerprint + best λ).
- **Classifier** (`ClassifierAgent`, utility-based) and **Adversary**
  (`AdversaryAgent`, goal-based) — the two agents that **compete** in a minimax
  game. The classifier predicts the target; the adversary, seeing only the
  classifier's output Ŷ, tries to recover the sensitive attributes. The classifier
  is penalised for whatever the adversary can recover, so it learns to be accurate
  **and** fair at the same time.

Instead of a fixed penalty tuned by a manual sweep, **λ adapts online** via a
momentum update with a running-maximum guard (deterministic — no LLM in the
training loop). β = 0.7 (chosen over the common default 0.9). A single run handles
**all** sensitive attributes simultaneously.

## Run

```bash
python main.py --dataset german --iterations 20
python main.py --dataset HIMS-Tunisia
python main.py --dataset adult --seed 14159 --epochs 50 --threshold 80
```

Built-in datasets: `adult` `german` `compas` `bank` `kdd` `acs` `utkface` `HIMS-Tunisia`
— or pass a path to your own CSV and the orchestrator detects the schema.

> Requires Ollama running (`ollama serve` + `ollama pull llama3.1`) for the
> sensitive-attribute identification step.

**Selection is fairness-first:** training stops once the P-rule target
(`--threshold`, default 80 — the four-fifths rule) is met on every attribute, and
the highest-accuracy iteration among the fair ones is returned. Accuracy is not capped.

## Layout

- `orchestrator.py` — `OrchestratorAgent`, the 5-step pipeline
- `models/` — `agents.py` (Classifier/Adversary agents), `classifier.py`, `image_classifier.py`, `adversary.py`
- `tools/` — `data_tools` (LLM attribute ID + preprocessing), `training_tools` (minimax loop), `lambda_tools` (momentum λ)
- `memory/` — `long_term.py` (runs + fingerprints), `short_term.py`
- `seeds/` — multi-seed sweep scripts (`run_multiple_seeds.py`, `aggregate_seeds.py`, …)
- `long_term_memory.json` — saved runs (metrics, λ trajectory, fingerprint), per seed
