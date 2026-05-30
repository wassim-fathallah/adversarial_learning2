"""
Global mutable state shared across all LangChain tools.
Tools are stateless functions, so training artifacts live here.
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import torch


@dataclass
class TrainingState:
    # ── Dataset ──────────────────────────────────────────────────────────────
    dataset_name: str = ""
    dataset_path: str = ""
    target_col: str = ""
    feature_names: List[str] = field(default_factory=list)
    sensitive_attrs: List[str] = field(default_factory=list)
    binarization_rules: Dict[str, Any] = field(default_factory=dict)
    columns_to_drop: List[str] = field(default_factory=list)

    # ── Tensors ───────────────────────────────────────────────────────────────
    X_train: Optional[torch.Tensor] = None
    X_test: Optional[torch.Tensor] = None
    y_train: Optional[torch.Tensor] = None
    y_test: Optional[torch.Tensor] = None
    sensitive_train: Optional[torch.Tensor] = None   # (N, n_sensitive)
    sensitive_test: Optional[torch.Tensor] = None
    # ── Models ────────────────────────────────────────────────────────────────
    classifier: Optional[Any] = None
    adversary: Optional[Any] = None
    clf_optimizer: Optional[Any] = None
    adv_optimizer: Optional[Any] = None

    # ── Hyperparameters ───────────────────────────────────────────────────────
    lambda_vector: List[float] = field(default_factory=list)
    lambda_momentum: List[float] = field(default_factory=list)
    pos_weight: float = 1.0          # class imbalance weight for BCELoss
    p_rule_threshold: float = 80.0
    max_iterations: int = 25
    epochs_per_step: int = 50
    initial_epochs: int = 10
    device: str = "cpu"

    # ── Runtime tracking ─────────────────────────────────────────────────────
    current_iteration: int = 0
    total_epochs_run: int = 0
    training_done: bool = False


# Singleton — imported by all tools
state = TrainingState()


def reset_state():
    """Call between runs to avoid stale data."""
    global state
    state = TrainingState()
    if torch.cuda.is_available():
        state.device = "cuda"
