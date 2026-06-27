"""Fast HIMS diagnostic: load + pretrain only, line-buffered with timing.
Shows n_features (detect one-hot explosion), groups per attr, baseline P-rules."""
import sys, json, time, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
import torch, numpy as np
import state as sm

st = sm.state
st.device = "cuda" if torch.cuda.is_available() else "cpu"
st.dataset_name = "HIMS-Tunisia"
st.modality = "tabular"
st.target_col = "legal_entry"
st.sensitive_attrs = ["Gender", "region_origin", "educ_level"]
st.binarization_rules = {
    "Gender":        {"positive_value": "Female"},
    "region_origin": {"positive_value": "Center-West"},
    "educ_level":    {"positive_value": "Higher education"},
}
st.columns_to_drop = ["id_menage", "Id_Ind", "weight", "Poids_Final",
                      "greater_tunis", "Center_East", "coastal_origin", "higher_educ",
                      "france", "ita", "ger"]

from tools.data_tools import load_dataset
from tools.training_tools import pretrain

r = {}
t0 = time.time()
print("[load] starting...", flush=True)
getattr(load_dataset, "func", load_dataset)(
    dataset_path="datasets/HIMS-Tunisia/HIMS-Tunisia.csv", dataset_name="HIMS-Tunisia")
print(f"[load] done in {time.time()-t0:.1f}s", flush=True)
r["device"] = st.device
r["n_rows_train"] = int(st.X_train.shape[0])
r["n_features"] = int(st.X_train.shape[1])
raw = st.sensitive_test_raw.cpu().numpy()
r["groups_per_attr"] = {a: int(len(np.unique(raw[:, i]))) for i, a in enumerate(st.sensitive_attrs)}
print(f"[info] features={r['n_features']} train_rows={r['n_rows_train']} groups={r['groups_per_attr']}", flush=True)

t1 = time.time()
print("[pretrain] 10 epochs...", flush=True)
pre = json.loads(getattr(pretrain, "func", pretrain)(n_epochs=10)).get("initial_metrics", {})
print(f"[pretrain] done in {time.time()-t1:.1f}s", flush=True)
r["baseline"] = {"accuracy": round(float(pre.get("accuracy", 0)), 4),
                 "p_rules": {k: round(float(v), 1) for k, v in (pre.get("p_rules") or {}).items()}}
print("[baseline]", r["baseline"], flush=True)

open("hims_quick_result.json", "w").write(json.dumps(r, indent=2, default=str))
print("DONE", flush=True)
