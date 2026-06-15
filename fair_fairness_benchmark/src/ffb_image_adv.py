"""
Zhang et al. (2018) adversarial debiasing on image datasets (UTKFace) — CNN version.

FFB ships adversarial debiasing only as TABULAR scripts and UTKFace only on the IMAGE
side, so there was no adversarial-debiasing entry point for UTKFace. This script fills
that gap and, crucially, mirrors OUR tabular pipeline's exact Zhang formulation so the
two are directly comparable — only the backbone changes (CNN instead of MLP):

  * Predictor  : CNN encoder + linear head -> Y_hat = sigmoid(.)          (the "classifier")
  * Adversary  : reads the SCALAR prediction Y_hat and predicts S         (Zhang Fig. 1)
  * Update     : two-step alternating, per batch
        step 1 (adversary) : minimize  L_adv(adv(Y_hat.detach()), s)
        step 2 (predictor) : minimize  L_task(Y_hat, y) - lam * L_adv(adv(Y_hat), s)
     The MINUS sign is the adversarial game: the predictor is pushed to make Y_hat
     uninformative about the sensitive attribute.

Two phases (same idea as our pretrain-then-debias schedule):
  Phase 1 — pretrain the predictor (and warm up the adversary) with lam=0, so the CNN
            first learns the task to a real baseline.
  Phase 2 — turn on the adversarial penalty (lam ramped 0 -> lam_max for stability).

Backbone:
  * If torchvision is installed (GPU box / full FFB env) -> pretrained ResNet encoder,
    matching the other ffb_image_* methods (--architecture resnet18/34/50).
  * Otherwise (e.g. CPU box without torchvision) -> a self-contained conv encoder,
    so the script still runs with no extra dependencies / no pretrained download.

Reuses FFB's own load_utkface_data and metric_evaluation so the numbers line up with
the other FFB UTKFace results.

Examples:
    python ffb_image_adv.py --sensitive_attr Gender --lam 1.0
    python ffb_image_adv.py --sensitive_attr Race --lam 2.0 --architecture resnet18
"""

import argparse
import json
import os
import numpy as np

# NumPy 2.x removed np.trapz (renamed to np.trapezoid). FFB's metrics.py (ABCC) still
# calls np.trapz, so shim it back here rather than editing the shared metrics module.
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tabulate import tabulate

from dataset import load_utkface_data
from metrics import metric_evaluation


# ── Backbone: pretrained ResNet if torchvision is available, else a plain CNN ──
def build_encoder(architecture: str):
    """Return (encoder_module, feature_dim). Encoder maps (N,3,48,48) -> (N, feat_dim)."""
    try:
        import torchvision
        backbones = {
            "resnet18": (torchvision.models.resnet18, 512),
            "resnet34": (torchvision.models.resnet34, 512),
            "resnet50": (torchvision.models.resnet50, 2048),
        }
        if architecture in backbones:
            ctor, feat_dim = backbones[architecture]
            net = ctor(pretrained=True)
            net.fc = nn.Identity()       # expose the 512/2048-d features
            print(f"[backbone] torchvision {architecture} (pretrained), feat_dim={feat_dim}")
            return net, feat_dim
        print(f"[backbone] '{architecture}' unknown; falling back to built-in CNN")
    except Exception as e:
        print(f"[backbone] torchvision unavailable ({e}); using built-in CNN")

    # Self-contained conv encoder (no torchvision / no download): 3x48x48 -> 256
    feat_dim = 256
    enc = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 24
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 12
        nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # 6
        nn.Flatten(), nn.Linear(128 * 6 * 6, feat_dim), nn.ReLU(),
    )
    print(f"[backbone] built-in CNN, feat_dim={feat_dim}")
    return enc, feat_dim


class ImageClassifier(nn.Module):
    """CNN encoder + linear head. Returns (logit, prob) — prob = sigmoid(logit)."""
    def __init__(self, encoder, feat_dim):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(feat_dim, 1)

    def forward(self, x):
        logit = self.head(self.encoder(x))
        return logit, torch.sigmoid(logit)


class Adversary(nn.Module):
    """Zhang adversary — reads the scalar prediction Y_hat, predicts the sensitive attr.
    Mirrors our models/adversary.py (input is the 1-d prediction, not the features)."""
    def __init__(self, n_hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, n_hidden), nn.ReLU(),
            nn.Linear(n_hidden, n_hidden), nn.ReLU(),
            nn.Linear(n_hidden, 1),     # one logit for the (binary) sensitive attr
        )

    def forward(self, y_hat):
        return self.net(y_hat)


def evaluate(clf, loader, target_index, sensitive_index, device, prefix):
    clf.eval()
    y_hat, y_true, s_true = [], [], []
    with torch.no_grad():
        for X, attr in loader:
            _, prob = clf(X.to(device))
            y_hat.append(prob.cpu().numpy())
            y_true.append(attr[:, target_index].unsqueeze(1).numpy())
            s_true.append(attr[:, sensitive_index].unsqueeze(1).numpy())
    y_hat = np.concatenate(y_hat); y_true = np.concatenate(y_true); s_true = np.concatenate(s_true)
    return metric_evaluation(y_gt=y_true, y_pre=y_hat, s=s_true, prefix=prefix)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, default="../datasets/utkface/raw")
    p.add_argument("--sensitive_attr", type=str, default="Gender", choices=["Age", "Gender", "Race"])
    p.add_argument("--target_attr", type=str, default="Age", choices=["Age", "Gender", "Race"],
                   help="prediction target; must differ from --sensitive_attr "
                        "(use target=Gender when sensitive=Age, since age is normally the target)")
    p.add_argument("--architecture", type=str, default="resnet18",
                   help="resnet18/34/50 if torchvision is present, else built-in CNN")
    p.add_argument("--lam", type=float, default=1.0, help="adversarial weight (fairness strength)")
    p.add_argument("--pretrain_steps", type=int, default=300, help="predictor-only warmup (lam=0)")
    p.add_argument("--num_training_steps", type=int, default=300, help="adversarial-phase steps")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--eval_batch_size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--adv_hidden", type=int, default=32)
    p.add_argument("--seed", type=int, default=1314)
    p.add_argument("--log_freq", type=int, default=50)
    p.add_argument("--save_dir", type=str, default="../results",
                   help="where to write the app.py-compatible result JSON")
    args = p.parse_args()

    print(tabulate([(k, v) for k, v in vars(args).items()], tablefmt="grid"))
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device = {device}")

    # ── Data: target = age>30 (idx 0), sensitive = gender(1) or race(2) ──────
    if args.target_attr == args.sensitive_attr:
        raise SystemExit(f"target_attr and sensitive_attr must differ (both = {args.target_attr}). "
                         f"UTKFace has only age/gender/race; one must be the target.")
    X, attr = load_utkface_data(path=args.data_path, sensitive_attribute=args.sensitive_attr)
    X_np = np.stack(X["pixels"].to_list()).astype("float32")   # (N, 3, 48, 48)
    attr_np = attr.to_numpy().astype("float32")                # columns: [age, gender, race]
    ATTR_INDEX = {"Age": 0, "Gender": 1, "Race": 2}
    target_index = ATTR_INDEX[args.target_attr]
    sensitive_index = ATTR_INDEX[args.sensitive_attr]

    Xtr, Xtmp, atr, atmp = train_test_split(X_np, attr_np, test_size=0.2,
                                            stratify=attr_np, random_state=args.seed)
    Xte, Xva, ate, ava = train_test_split(Xtmp, atmp, test_size=0.5,
                                          stratify=atmp, random_state=args.seed)
    print(f"[info] train={len(Xtr)} val={len(Xva)} test={len(Xte)} | "
          f"sensitive='{args.sensitive_attr}' (idx {sensitive_index})")

    def ds(Xa, aa):
        return TensorDataset(torch.from_numpy(Xa).float(), torch.from_numpy(aa).float())
    train_loader = DataLoader(ds(Xtr, atr), batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(ds(Xva, ava), batch_size=args.eval_batch_size, shuffle=False)
    test_loader  = DataLoader(ds(Xte, ate), batch_size=args.eval_batch_size, shuffle=False)

    # ── Models / optimizers — separate, like Zhang (predictor vs adversary) ──
    encoder, feat_dim = build_encoder(args.architecture)
    clf = ImageClassifier(encoder, feat_dim).to(device)
    adv = Adversary(n_hidden=args.adv_hidden).to(device)
    clf_criterion = nn.BCEWithLogitsLoss()     # on the predictor's logit (stable)
    adv_criterion = nn.BCEWithLogitsLoss()     # on the adversary's logit
    clf_opt = optim.Adam(clf.parameters(), lr=args.lr)
    adv_opt = optim.Adam(adv.parameters(), lr=args.lr)

    def batches():
        while True:
            for b in train_loader:
                yield b
    stream = batches()

    logs, headers = [], ["phase", "step", "test/acc", "test/prule", "test/dp", "val/acc", "val/prule"]
    history = []   # app.py-compatible: list of {"step", "test/*", "val/*", ...}

    def log_row(phase, step, global_step, lam, clf_loss, adv_loss):
        vm = evaluate(clf, val_loader, target_index, sensitive_index, device, "val")
        tm = evaluate(clf, test_loader, target_index, sensitive_index, device, "test")
        logs.append([phase, step, round(tm["test/acc"], 2), round(tm["test/prule"], 2),
                     round(tm["test/dp"], 4), round(vm["val/acc"], 2), round(vm["val/prule"], 2)])
        # Full metric dict for the interface (continuous global step across both phases).
        # metric_evaluation returns some numpy float32 values -> coerce to native float
        # so json.dump can serialize them.
        entry = {"step": global_step, "phase": phase,
                 "training/clf_loss": float(clf_loss), "training/adv_loss": float(adv_loss),
                 "training/lam": float(lam)}
        for k, v in {**tm, **vm}.items():
            entry[k] = float(v)
        history.append(entry)
        print(f"[{phase} {step:4d}] lam={lam:.3f} clf={clf_loss:.4f} adv={adv_loss:.4f} "
              f"| test acc={tm['test/acc']:.2f} P-rule={tm['test/prule']:.2f} dp={tm['test/dp']:.4f}")

    def get_batch():
        Xb, ab = next(stream)
        Xb = Xb.to(device)
        yb = ab[:, target_index].unsqueeze(1).to(device)
        sb = ab[:, sensitive_index].unsqueeze(1).to(device)
        return Xb, yb, sb

    # ── Phase 1: pretrain predictor (lam=0) + warm up adversary ──────────────
    print(f"\n--- Phase 1: pretrain predictor ({args.pretrain_steps} steps, lam=0) ---")
    for step in range(args.pretrain_steps):
        clf.train(); adv.train()
        Xb, yb, sb = get_batch()
        # predictor: task loss only
        clf_opt.zero_grad()
        logit, prob = clf(Xb)
        clf_loss = clf_criterion(logit, yb)
        clf_loss.backward()
        clf_opt.step()
        # adversary: learn to read the (detached) prediction
        adv_opt.zero_grad()
        adv_loss = adv_criterion(adv(prob.detach()), sb)
        adv_loss.backward()
        adv_opt.step()
        if step % args.log_freq == 0 or step == args.pretrain_steps - 1:
            log_row("pre", step, step, 0.0, clf_loss.item(), adv_loss.item())

    # ── Phase 2: adversarial debiasing (two-step, lam ramped 0 -> lam) ───────
    print(f"\n--- Phase 2: adversarial debiasing ({args.num_training_steps} steps, lam={args.lam}) ---")
    for step in range(args.num_training_steps):
        clf.train(); adv.train()
        Xb, yb, sb = get_batch()
        lam = args.lam * min(1.0, step / max(1, args.num_training_steps // 2))  # linear warmup over first half

        # step 1 — adversary maximizes its ability to predict s from detached Y_hat
        adv_opt.zero_grad()
        _, prob = clf(Xb)
        adv_loss = adv_criterion(adv(prob.detach()), sb)
        adv_loss.backward()
        adv_opt.step()

        # step 2 — predictor minimizes L_task - lam * L_adv (fool the adversary)
        clf_opt.zero_grad()
        logit, prob = clf(Xb)
        clf_loss = clf_criterion(logit, yb)
        adv_penalty = adv_criterion(adv(prob), sb)
        loss = clf_loss - lam * adv_penalty
        loss.backward()
        clf_opt.step()

        if step % args.log_freq == 0 or step == args.num_training_steps - 1:
            log_row("adv", step, args.pretrain_steps + step, lam, clf_loss.item(), adv_penalty.item())

    # ── Report: baseline (post-pretrain) vs after debiasing ──────────────────
    print("\n=== Zhang adversarial debiasing on UTKFace (CNN) ===")
    print(tabulate(logs, headers=headers, tablefmt="grid", floatfmt="0.2f"))
    pre_rows = [r for r in logs if r[0] == "pre"]
    final = logs[-1]
    if pre_rows:
        b = pre_rows[-1]
        print(f"\nBASELINE (post-pretrain, no debiasing): acc={b[2]:.2f}%  P-rule={b[3]:.2f}%  dp={b[4]:.4f}")
    print(f"FINAL    (after debiasing, lam={args.lam}): acc={final[2]:.2f}%  "
          f"P-rule={final[3]:.2f}%  dp={final[4]:.4f}  (sensitive={args.sensitive_attr})")

    # ── Save in the FFB app.py format: {"metadata":..., "history":[...]} ─────
    os.makedirs(args.save_dir, exist_ok=True)
    metadata = {
        "method": "adv",                       # shows up as its own method in app.py
        "dataset": "utkface",
        "sensitive_attr": args.sensitive_attr,
        "lam": args.lam,
        "seed": args.seed,
        "source": "local",
        "target_attr": args.target_attr,
        "architecture": args.architecture,
    }
    fname = f"utkface_adv_{args.sensitive_attr}_{args.target_attr}_lam{args.lam}_seed{args.seed}.json"
    out_path = os.path.join(args.save_dir, fname)
    with open(out_path, "w") as fp:
        json.dump({"metadata": metadata, "history": history}, fp, indent=2)
    print(f"[saved] {out_path}  ({len(history)} history points)")


if __name__ == "__main__":
    main()
