# -*- coding: utf-8 -*-
"""Extract the LAST run whose memory key contains a substring, into an isolated
one-key file shaped like the ablation files (so plot_momentum_per_attr.py can read it).

    python extract_run.py long_term_memory.json HIMS-10k synth_mom_b09.json
"""
import json
import sys

src   = sys.argv[1] if len(sys.argv) > 1 else "long_term_memory.json"
needle = sys.argv[2] if len(sys.argv) > 2 else "HIMS-10k"
out   = sys.argv[3] if len(sys.argv) > 3 else "synth_mom_b09.json"

data = json.load(open(src, encoding="utf-8"))
matches = [k for k in data if needle in k]
if not matches:
    raise SystemExit(f"No memory key contains '{needle}'. Keys: {list(data)}")
key = matches[-1]
entry = data[key][-1]
json.dump({key: [entry]}, open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print(f"[extract] key='{key}'  run timestamp={entry.get('timestamp')}  -> {out}")
print(f"[extract] iterations={entry.get('iterations')}  success={entry.get('success')}  "
      f"final P-rules={entry.get('p_rules_final')}")
