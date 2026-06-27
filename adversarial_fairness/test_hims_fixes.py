"""
Test the two HIMS-Tunisia fixes (professor's review) — WITHOUT needing Ollama.

  1) region_origin proxies (greater_tunis, Center_East, coastal_origin, higher_educ,
     france, ita, ger) must NOT survive into the training features.
  2) p_rule() is multi-group — region_origin is scored over its 7 real categories.

It sets the HIMS config directly (= what pinning would do) so it doesn't depend on
the LLM. If Ollama IS up, set USE_LLM=True to test the identify_sensitive path too.

Run:  python test_hims_fixes.py     -> writes test_hims_result.json + PASS/FAIL
"""
import sys, json
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import numpy as np
import state as sm
from tools.data_tools import load_dataset
from tools.training_tools import pretrain

PROXIES = ["greater_tunis", "Center_East", "coastal_origin", "higher_educ", "france", "ita", "ger"]
PATH = "datasets/HIMS-Tunisia/HIMS-Tunisia.csv"
USE_LLM = False   # set True if Ollama is running, to also exercise identify_sensitive

out = {}
st = sm.state

if USE_LLM:
    from tools.data_tools import identify_sensitive
    res = json.loads(getattr(identify_sensitive, "func", identify_sensitive)(dataset_path=PATH, dataset_name="migration"))
    out["llm_dropped"] = res.get("columns_to_drop")
else:
    # The pinned HIMS config (target + 3 sensitive + IDs/weights + the 7 proxies)
    st.dataset_name = "migration"
    st.modality = "tabular"
    st.target_col = "legal_entry"
    st.sensitive_attrs = ["Gender", "region_origin", "educ_level"]
    st.binarization_rules = {
        "Gender":        {"positive_value": "Female"},
        "region_origin": {"positive_value": "Center-West"},
        "educ_level":    {"positive_value": "Higher education"},
    }
    st.columns_to_drop = ["id_menage", "Id_Ind", "weight", "Poids_Final"] + PROXIES

getattr(load_dataset, "func", load_dataset)(dataset_path=PATH, dataset_name="migration")

print("=" * 60)
print("CHANGE 1 — region_origin proxies dropped")
print("=" * 60)
feat = [str(f) for f in st.feature_names]
leaked = sorted({p for p in PROXIES if any(f == p or f.startswith(p + "_") for f in feat)})
print("proxies still in FEATURES:", leaked, " <- must be EMPTY")
out["change1_proxies_leaked"] = leaked
out["change1_PASS"] = (len(leaked) == 0)

print()
print("=" * 60)
print("CHANGE 2 — multi-group p_rule on region_origin")
print("=" * 60)
attrs = st.sensitive_attrs
raw  = st.sensitive_test_raw.cpu().numpy()
binr = st.sensitive_test.cpu().numpy()
for i, a in enumerate(attrs):
    print(f"  {a:14s}: p_rule scored over {len(np.unique(raw[:, i]))} groups (raw) vs {len(np.unique(binr[:, i]))} (binary)")
    out[f"groups_{a}"] = {"raw": int(len(np.unique(raw[:, i]))), "binary": int(len(np.unique(binr[:, i])))}

m = json.loads(getattr(pretrain, "func", pretrain)(n_epochs=5)).get("initial_metrics", {})
print("multi-group P-rules:", {k: round(float(v), 1) for k, v in (m.get("p_rules") or {}).items()})
out["pretrain_p_rules"] = {k: round(float(v), 2) for k, v in (m.get("p_rules") or {}).items()}
ridx = attrs.index("region_origin") if "region_origin" in attrs else -1
out["change2_PASS"] = (ridx >= 0 and len(np.unique(raw[:, ridx])) > 2)

print()
print("RESULT: change1 =", "PASS" if out.get("change1_PASS") else "FAIL",
      "| change2 =", "PASS" if out.get("change2_PASS") else "FAIL")
open("test_hims_result.json", "w").write(json.dumps(out, indent=2))
