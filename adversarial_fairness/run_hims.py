"""Full HIMS-Tunisia run with the pinned config (no Ollama needed) — to SEE the results:
proxies dropped, region_origin pinned (Center-West) for training, multi-group p_rule.
Writes hims_run_result.json."""
import sys, json, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
import torch
import state as sm

st = sm.state
st.device = "cuda" if torch.cuda.is_available() else "cpu"
st.dataset_name = "HIMS-Tunisia"
st.modality = "tabular"
# #1 (multi-group adversary) OFF for HIMS: it cost ~3 acc points for ~1.6 region,
# and the binary adversary already lifts region's 7-group score to ~75 at ~83% acc.
st.multigroup_adversary = False
st.target_col = "legal_entry"
st.sensitive_attrs = ["Gender", "region_origin", "educ_level"]
st.binarization_rules = {
    "Gender":        {"positive_value": "Female"},
    "region_origin": {"positive_value": "Center-West"},   # pinned, training only
    "educ_level":    {"positive_value": "Higher education"},
}
st.columns_to_drop = ["id_menage", "Id_Ind", "weight", "Poids_Final",
                      "greater_tunis", "Center_East", "coastal_origin", "higher_educ",
                      "france", "ita", "ger"]

from tools.data_tools import load_dataset
from tools.training_tools import pretrain, run_full_training

r = {}
try:
    getattr(load_dataset, "func", load_dataset)(
        dataset_path="datasets/HIMS-Tunisia/HIMS-Tunisia.csv", dataset_name="HIMS-Tunisia")
    r["n_features"] = int(st.X_train.shape[1])
    import numpy as np
    raw = st.sensitive_test_raw.cpu().numpy()
    r["groups_per_attr"] = {a: int(len(np.unique(raw[:, i]))) for i, a in enumerate(st.sensitive_attrs)}

    pre = json.loads(getattr(pretrain, "func", pretrain)(n_epochs=10)).get("initial_metrics", {})
    r["baseline"] = {"accuracy": round(float(pre.get("accuracy", 0)), 4),
                     "p_rules": {k: round(float(v), 1) for k, v in (pre.get("p_rules") or {}).items()}}

    st.lambda_vector = [0.0] * len(st.sensitive_attrs)   # HIMS zero-starts lambda
    out = getattr(run_full_training, "func", run_full_training)(
        max_iterations=25, epochs_per_step=50, p_rule_threshold=80.0)   # AADA defaults
    try:
        out = json.loads(out)
    except Exception:
        out = {"raw": str(out)[:800]}
    r["final"] = {
        "status": out.get("status"),
        "iterations_run": out.get("iterations"),
        "success": out.get("success"),
        "final_metrics": out.get("final_metrics") or out.get("best_metrics"),
        "lambda_final": out.get("lambda_final") or st.lambda_vector,
    }
except Exception as e:
    import traceback
    r["error"] = repr(e); r["tb"] = traceback.format_exc()[-700:]

open("hims_run_result.json", "w").write(json.dumps(r, indent=2, default=str))
print("DONE")
