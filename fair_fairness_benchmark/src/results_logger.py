import os
import json
import sys


class ResultsLogger:
    def __init__(self, args, results_dir="../results"):
        os.makedirs(results_dir, exist_ok=True)
        script = os.path.basename(sys.argv[0])
        method = script.replace("ffb_tabular_", "").replace("ffb_image_", "").replace(".py", "")
        lam = float(getattr(args, "lam", None) or getattr(args, "A_z", None) or 0)
        self.metadata = {
            "method": method,
            "dataset": args.dataset,
            "sensitive_attr": args.sensitive_attr,
            "lam": lam,
            "seed": args.seed,
        }
        self.history = []
        fname = f"{args.dataset}_{method}_{args.sensitive_attr}_lam{lam}_seed{args.seed}.json"
        self.filepath = os.path.join(results_dir, fname)

    def log(self, step, metrics):
        clean = {}
        for k, v in metrics.items():
            if v is None:
                continue
            try:
                clean[k] = float(v)
            except (TypeError, ValueError):
                pass
        self.history.append({"step": int(step), **clean})
        with open(self.filepath, "w") as f:
            json.dump({"metadata": self.metadata, "history": self.history}, f)
