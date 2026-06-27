"""
Global mutable state shared across all LangChain tools.
Tools are stateless functions, so training artifacts live here.
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import torch


@dataclass
class TrainingState:
    # Dataset
    dataset_name: str = ""
    dataset_path: str = ""
    target_col: str = ""
    feature_names: List[str] = field(default_factory=list)
    sensitive_attrs: List[str] = field(default_factory=list)
    binarization_rules: Dict[str, Any] = field(default_factory=dict)
    columns_to_drop: List[str] = field(default_factory=list)

    # Modality — "tabular" (MLP classifier) or "image" (CNN classifier).
    # Detected from the dataset structure (and confirmed by the LLM) before training.
    modality: str = "tabular"
    pixel_column: str = ""           # name of the flattened-pixel / image column (image only)
    image_shape: tuple = ()          # (C, H, W) once the pixel block is reshaped (image only)

    # Tensors
    X_train: Optional[torch.Tensor] = None
    X_test: Optional[torch.Tensor] = None
    y_train: Optional[torch.Tensor] = None
    y_test: Optional[torch.Tensor] = None
    sensitive_train: Optional[torch.Tensor] = None   # (N, n_sensitive) binarised 0/1
    sensitive_test: Optional[torch.Tensor] = None
    # Multi-group bucket codes (0..K-1) per sensitive attr, used only by the grouped
    # P-rule metric (four-fifths rule over real groups). For HIMS, region_origin and
    # educ_level are collapsed into 3 custom buckets; binary attrs stay 0/1. Training
    # still uses the binarised sensitive_* tensors above.
    sensitive_train_raw: Optional[torch.Tensor] = None   # (N, n_sensitive) int codes
    sensitive_test_raw: Optional[torch.Tensor] = None
    # Human-readable name for each group code, per sensitive attribute:
    # {attr: {code: label}} (e.g. region_origin: {0:"Center-West",1:"Greater Tunis",
    # 2:"Others"}; Gender: {0:"Male",1:"Female"}). Used to label the qualitative-report
    # plots with real group names instead of "group 0/1/2".
    group_labels: Dict[str, Any] = field(default_factory=dict)
    # Predicted P(target=1) on the test set right after pretraining, BEFORE any
    # adversarial pressure. Snapshotted because the classifier is trained in place,
    # so the baseline weights are gone by the time the qualitative report runs; the
    # report uses this to draw the before→after prediction distributions.
    baseline_probs: Optional[Any] = None
    # Models
    classifier: Optional[Any] = None
    adversary: Optional[Any] = None
    clf_optimizer: Optional[Any] = None
    adv_optimizer: Optional[Any] = None

    # Hyperparameters
    lambda_vector: List[float] = field(default_factory=list)
    lambda_momentum: List[float] = field(default_factory=list)
    best_lambda_seen: List[float] = field(default_factory=list)
    pos_weight: float = 1.0          # class imbalance weight for BCELoss
    p_rule_threshold: float = 80.0
    max_iterations: int = 25
    epochs_per_step: int = 50
    initial_epochs: int = 10
    device: str = "cpu"

    # Runtime tracking
    current_iteration: int = 0
    total_epochs_run: int = 0
    training_done: bool = False

    # Clean accuracy reached after pretraining (BEFORE adversarial pressure).
    # Recorded for reporting only — accuracy is NOT limited and never gates
    # selection; the result is the highest-accuracy iteration meeting the P-rule.
    baseline_accuracy: float = 0.0


# Singleton — imported by all tools
state = TrainingState()


def reset_state():
    """Call between runs to avoid stale data."""
    global state
    state = TrainingState()
    if torch.cuda.is_available():
        state.device = "cuda"
