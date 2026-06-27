"""Diagnose why region_origin no longer crosses 80.
Train current config (proxies dropped), then score region_origin P-rule BOTH ways
on the SAME final model: binary (Center-West vs rest, 2 groups, the OLD metric)
vs multi-group (7 regions, the NEW metric). If binary >> multigroup, the metric
change is the cause, not the training."""
import sys, json, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
import torch, numpy as np
import state as sm
from utils.metrics import p_rule

st = sm.state
st.device = "cuda" if torch.cuda.is_available() else "cpu"
st.dataset_name = "HIMS-Tunisia"; st.modality = "tabular"; st.target_col = "legal_entry"
st.sensitive_attrs = ["Gender", "region_origin", "educ_level"]
st.binarization_rules = {"Gender": {"positive_value": "Female"},
                         "region_origin": {"positive_value": "Center-West"},
                         "educ_level": {"positive_value": "Higher education"}}
st.columns_to_drop = ["id_menage", "Id_Ind", "weight", "Poids_Final",
                      "greater_tunis", "Center_East", "coastal_origin", "higher_educ",
                      "france", "ita", "ger"]

from tools.data_tools import load_dataset
from tools.training_tools import pretrain, run_full_training

getattr(load_dataset, "func", load_dataset)(
    dataset_path="datasets/HIMS-Tunisia/HIMS-Tunisia.csv", dataset_name="HIMS-Tunisia")
getattr(pretrain, "func", pretrain)(n_epochs=10)
st.lambda_vector = [0.0] * 3
getattr(run_full_training, "func", run_full_training)(
    max_iterations=15, epochs_per_step=50, p_rule_threshold=80.0)

# Score region_origin both ways on the SAME final model
ridx = st.sensitive_attrs.index("region_origin")
clf = st.classifier.eval()
with torch.no_grad():
    Xte = st.X_test.to(st.device)
    preds = clf(Xte).cpu().numpy().ravel()

s_bin = st.sensitive_test.cpu().numpy()[:, ridx]      # 2 groups (Center-West vs rest)
s_raw = st.sensitive_test_raw.cpu().numpy()[:, ridx]  # 7 groups

out = {
    "region_origin_p_rule_BINARY_2grp_OLD_metric": round(float(p_rule(preds, s_bin)), 1),
    "region_origin_p_rule_MULTIGROUP_7grp_NEW_metric": round(float(p_rule(preds, s_raw)), 1),
    "n_groups_binary": int(len(np.unique(s_bin))),
    "n_groups_raw": int(len(np.unique(s_raw))),
}
# per-region positive rates (shows which regions diverge)
rates = {}
for g in np.unique(s_raw):
    mask = s_raw == g
    rates[int(g)] = round(float((preds[mask] >= 0.5).mean()), 3)
out["per_region_positive_rate"] = rates
print("DIAG", json.dumps(out))
open("hims_diag.json", "w").write(json.dumps(out, indent=2))
print("DONE")
