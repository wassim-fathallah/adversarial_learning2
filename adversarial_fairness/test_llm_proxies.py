"""Test ONLY the LLM identify_sensitive pick for HIMS — no training.
Prints sensitive_attrs, target, and which of the 7 known proxies the LLM dropped."""
import sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
import state as sm
st = sm.state
st.dataset_name = "HIMS-Tunisia"

from tools.data_tools import identify_sensitive
out = getattr(identify_sensitive, "func", identify_sensitive)(
    dataset_path="datasets/HIMS-Tunisia/HIMS-Tunisia.csv", dataset_name="HIMS-Tunisia")
res = json.loads(out)

KNOWN_PROXIES = ["greater_tunis", "Center_East", "coastal_origin",
                 "higher_educ", "france", "ita", "ger"]
drops = res.get("columns_to_drop", [])

print("\n========== LLM PICK ==========")
print("sensitive_attrs :", res.get("sensitive_attrs"))
print("target_col      :", res.get("target_col"))
print("columns_to_drop :", drops)
print("\n--- proxy coverage ---")
for p in KNOWN_PROXIES:
    print(f"  {p:15s} {'DROPPED ✓' if p in drops else 'KEPT ✗'}")
n = sum(p in drops for p in KNOWN_PROXIES)
print(f"\nLLM dropped {n}/7 proxies")
json.dump(res, open("test_llm_proxies_result.json", "w"), indent=2)
print("DONE")
