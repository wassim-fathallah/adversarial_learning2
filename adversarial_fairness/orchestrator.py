"""
Orchestrator agent — the central learning agent of AADA (paper Section 3.2, Fig 1).

The OrchestratorAgent perceives the dataset schema and the evolving accuracy–
fairness state, acts by identifying sensitive attributes (via a locally-served
LLM), configuring the pipeline, updating the per-attribute penalty λ, checking
the stopping condition and selecting the returned model, and improves across
runs via long-term memory. It owns the LangChain tools and sequences them.

The pipeline order is fixed, so the orchestrator drives the tools in a
deterministic sequence rather than via a ReAct loop (local LLMs through Ollama
are unreliable at tool-calling). The LLM is used only for sensitive-attribute
identification; every other step — λ update (the momentum rule of §3.6),
training, evaluation, model selection — is deterministic.

`run_pipeline(...)` is kept as a thin functional wrapper around the agent so
existing entry points (main.py, streamlit) are unchanged.
"""

import json
import torch

import agent_log
from state import state, reset_state
from tools.data_tools import identify_sensitive, load_dataset
from tools.lambda_tools import decide_initial_lambda
from tools.training_tools import pretrain, run_full_training

_SEP = "-" * 60


class OrchestratorAgent:
    """Meta-level agent that configures the pipeline and controls the
    adversarial training loop (paper §3.2). The classifier and adversary are the
    two reactive optimization agents it coordinates (see models/agents.py)."""

    def __init__(
        self,
        dataset_path: str,
        dataset_name: str = "",
        max_iterations: int = 25,
        epochs_per_step: int = 50,
        p_rule_threshold: float = 80.0,
        initial_epochs: int = 10,
        device: str = None,
        target_override: str = None,
        seed: int = 42,
    ):
        self.dataset_path     = dataset_path
        self.dataset_name     = dataset_name
        self.max_iterations   = max_iterations
        self.epochs_per_step  = epochs_per_step
        self.p_rule_threshold = p_rule_threshold
        self.initial_epochs   = initial_epochs
        self.device           = device
        self.target_override  = target_override
        self.seed             = seed
        self.name = dataset_name or dataset_path.split("/")[-1].split("\\")[-1]

        # Tools the orchestrator perceives/acts through (paper Fig 1).
        self.tools = {
            "identify_sensitive":    identify_sensitive,
            "load_dataset":          load_dataset,
            "decide_initial_lambda": decide_initial_lambda,
            "pretrain":              pretrain,
            "run_full_training":     run_full_training,
        }

    # --- agent actions (one per pipeline step) --------------------------------

    def _configure_run(self):
        """Reset state and seed everything for a reproducible run."""
        reset_state()
        state.device = ("cuda" if torch.cuda.is_available() else "cpu") if self.device is None else self.device
        state.initial_epochs = self.initial_epochs
        state.seed = self.seed

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        import numpy as np
        np.random.seed(self.seed)
        import random
        random.seed(self.seed)

    def identify_sensitive(self) -> dict:
        """Step 1 — perceive schema; LLM returns sensitive attrs + binarization."""
        print(f"\n{_SEP}")
        print(f"  Step 1/5 -- Identify sensitive attributes")
        print(_SEP)
        agent_log.orchestrator(
            f"Perceiving the schema of '{self.name}' — asking the LLM which columns "
            f"are sensitive attributes and what to predict.")
        result1 = self.tools["identify_sensitive"].invoke(
            {"dataset_path": self.dataset_path, "dataset_name": self.name})
        schema = json.loads(result1)

        if self.target_override:
            if self.target_override in state.sensitive_attrs:
                raise ValueError(
                    f"target_override '{self.target_override}' is also a sensitive attribute; "
                    f"choose a different target."
                )
            state.target_col = self.target_override
            state.columns_to_drop = [c for c in state.columns_to_drop if c != self.target_override]
            schema["target_col"] = self.target_override
            print(f"  [override] target column forced to: {self.target_override}")

        print(f"  Sensitive attrs : {schema['sensitive_attrs']}")
        print(f"  Target column   : {schema['target_col']}")
        print(f"  Drop columns    : {schema.get('columns_to_drop', [])}")
        agent_log.orchestrator(
            f"Decision — protect {schema['sensitive_attrs']} while predicting "
            f"'{schema['target_col']}'.")
        return schema

    def load_dataset(self):
        """Step 2 — generic preprocessing."""
        print(f"\n{_SEP}")
        print(f"  Step 2/5 -- Load & preprocess dataset")
        print(_SEP)
        agent_log.orchestrator("Loading and preprocessing the dataset (encode, scale, split).")
        self.tools["load_dataset"].invoke(
            {"dataset_path": self.dataset_path, "dataset_name": self.name})
        modality = getattr(state, "modality", "tabular") or "tabular"
        if str(modality).lower() == "image":
            agent_log.orchestrator("Modality: IMAGE → the classifier will be a CNN.")
        else:
            agent_log.orchestrator("Modality: TABULAR → the classifier will be an MLP.")

    def decide_initial_lambda(self):
        """Step 3 — λ⁽⁰⁾: zero for known datasets, fingerprint warm-start otherwise."""
        print(f"\n{_SEP}")
        print(f"  Step 3/5 -- Decide initial lambda (dataset schema)")
        print(_SEP)
        n_sensitive = len(state.sensitive_attrs)
        self.tools["decide_initial_lambda"].invoke({"n_sensitive": n_sensitive})
        print(f"  Initial lambda : {state.lambda_vector}")
        agent_log.orchestrator(
            f"Initial fairness penalty λ⁽⁰⁾ = {[round(l, 3) for l in state.lambda_vector]}.")

    def pretrain(self):
        """Step 4 — independent pre-training of classifier + adversary agents."""
        print(f"\n{_SEP}")
        print(f"  Step 4/5 -- Pretrain classifier + adversary")
        print(_SEP)
        agent_log.orchestrator(
            f"Pretraining the classifier and adversary independently "
            f"({self.initial_epochs} epochs) before the minimax game begins.")
        self.tools["pretrain"].invoke({"n_epochs": self.initial_epochs})

    def adversarial_loop(self) -> dict:
        """Step 5 — adversarial loop with deterministic per-iteration λ update."""
        print(f"\n{_SEP}")
        print(f"  Step 5/5 -- Adversarial training loop")
        print(_SEP)
        agent_log.orchestrator(
            f"Starting the adversarial loop — up to {self.max_iterations} iterations, "
            f"target P-rule ≥ {self.p_rule_threshold:.0f}% on every attribute.")
        result5 = self.tools["run_full_training"].invoke({
            "max_iterations":   self.max_iterations,
            "epochs_per_step":  self.epochs_per_step,
            "p_rule_threshold": self.p_rule_threshold,
            "lambda_max":       20.0,
        })
        final = json.loads(result5)
        return {
            "status":        final.get("status"),
            "final_metrics": final.get("final_metrics", {}),
            "final_lambda":  final.get("final_lambda", []),
            "plot_saved":    final.get("plot_saved"),
            "training_done": state.training_done,
        }

    # --- driver ---------------------------------------------------------------

    def run(self) -> dict:
        """Execute the full five-step pipeline and return the final result."""
        self._configure_run()
        self.identify_sensitive()
        self.load_dataset()
        self.decide_initial_lambda()
        self.pretrain()
        return self.adversarial_loop()


def run_pipeline(
    dataset_path: str,
    dataset_name: str = "",
    max_iterations: int = 25,
    epochs_per_step: int = 50,
    p_rule_threshold: float = 80.0,
    initial_epochs: int = 10,
    device: str = None,
    target_override: str = None,
    seed: int = 42,
) -> dict:
    """Thin wrapper: build the OrchestratorAgent and run it (kept for back-compat)."""
    return OrchestratorAgent(
        dataset_path=dataset_path,
        dataset_name=dataset_name,
        max_iterations=max_iterations,
        epochs_per_step=epochs_per_step,
        p_rule_threshold=p_rule_threshold,
        initial_epochs=initial_epochs,
        device=device,
        target_override=target_override,
        seed=seed,
    ).run()
