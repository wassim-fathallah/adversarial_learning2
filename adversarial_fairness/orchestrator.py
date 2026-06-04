"""
Orchestrator — runs the 5-step adversarial fairness pipeline in sequence.

The pipeline order is always fixed, so we call tools directly rather than
using a ReAct agent (local LLMs via Ollama are unreliable with tool-calling).

LLM is used for:
  - identify_sensitive: reads dataset schema → sensitive attrs + binarization rules
  - decide_initial_lambda: fingerprint warm-start from long-term memory, else zero
  All other steps (lambda update, training, evaluation) are deterministic.
"""

import json
import torch

from state import state, reset_state
from tools.data_tools import identify_sensitive, load_dataset
from tools.lambda_tools import decide_initial_lambda
from tools.training_tools import pretrain, run_full_training


def run_pipeline(
    dataset_path: str,
    dataset_name: str = "",
    max_iterations: int = 25,
    epochs_per_step: int = 50,
    p_rule_threshold: float = 80.0,
    initial_epochs: int = 10,
    device: str = None,
    target_override: str = None,
) -> dict:
    """
    Run the full fairness correction pipeline.

    Steps:
      1. identify_sensitive  — LLM reads schema → sensitive attrs + binarization
      2. load_dataset        — generic preprocessing
      3. decide_initial_lambda — LLM warm-starts λ from long-term memory
      4. pretrain            — independent pretraining + proxy detection
      5. run_full_training   — adversarial loop; LCEL chain decides λ per iteration
    """
    reset_state()

    state.device = ("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
    state.initial_epochs = initial_epochs

    name = dataset_name or dataset_path.split("/")[-1].split("\\")[-1]

    SEP = "-" * 60
    print(f"\n{SEP}")
    print(f"  Step 1/5 -- Identify sensitive attributes")
    print(SEP)
    result1 = identify_sensitive.invoke({"dataset_path": dataset_path, "dataset_name": name})
    schema = json.loads(result1)

    if target_override:
        if target_override in state.sensitive_attrs:
            raise ValueError(
                f"target_override '{target_override}' is also a sensitive attribute; "
                f"choose a different target."
            )
        state.target_col = target_override
        state.columns_to_drop = [c for c in state.columns_to_drop if c != target_override]
        schema["target_col"] = target_override
        print(f"  [override] target column forced to: {target_override}")

    print(f"  Sensitive attrs : {schema['sensitive_attrs']}")
    print(f"  Target column   : {schema['target_col']}")
    print(f"  Drop columns    : {schema.get('columns_to_drop', [])}")

    print(f"\n{SEP}")
    print(f"  Step 2/5 -- Load & preprocess dataset")
    print(SEP)
    load_dataset.invoke({"dataset_path": dataset_path, "dataset_name": name})

    print(f"\n{SEP}")
    print(f"  Step 3/5 -- Decide initial lambda (dataset schema)")
    print(SEP)
    n_sensitive = len(state.sensitive_attrs)
    decide_initial_lambda.invoke({"n_sensitive": n_sensitive})
    print(f"  Initial lambda : {state.lambda_vector}")

    print(f"\n{SEP}")
    print(f"  Step 4/5 -- Pretrain classifier + adversary")
    print(SEP)
    pretrain.invoke({"n_epochs": initial_epochs})

    print(f"\n{SEP}")
    print(f"  Step 5/5 -- Adversarial training loop")
    print(SEP)
    result5 = run_full_training.invoke({
        "max_iterations":   max_iterations,
        "epochs_per_step":  epochs_per_step,
        "p_rule_threshold": p_rule_threshold,
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
