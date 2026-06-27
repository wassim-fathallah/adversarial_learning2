# -*- coding: utf-8 -*-
"""Baseline-classifier analysis for the slides: trains the 2-layer MLP (256x256,
ReLU, dropout 0.2, Adam 1e-3) out-of-fold (5-fold CV, seed 42) on HIMS-Tunisia-real10k,
then reports AUC-ROC / accuracy / confusion matrix and the BEFORE-only predicted
P(legal_entry=1) distributions for all THREE sensitive attributes."""
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_style("whitegrid")

SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)
HERE = "C:/Users/jamil/Desktop/adversarial_learning2/adversarial_fairness"
df = pd.read_csv(f"{HERE}/datasets/HIMS-Tunisia/HIMS-Tunisia-real10k.csv")
SENS = ["Gender", "region_origin", "educ_level"]
y = (df.legal_entry == "Yes").astype(int).values
Xdf = pd.get_dummies(df.drop(columns=SENS + ["legal_entry"]))
X = StandardScaler().fit_transform(Xdf.to_numpy(float)).astype(np.float32)

class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 1))
    def forward(self, x): return self.net(x)

oof = np.zeros(len(y))
for tr, va in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y):
    m = MLP(X.shape[1]); opt = torch.optim.Adam(m.parameters(), 1e-3)
    lossf = nn.BCEWithLogitsLoss()
    Xt = torch.tensor(X[tr]); yt = torch.tensor(y[tr], dtype=torch.float32).unsqueeze(1)
    m.train()
    for ep in range(60):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 256):
            idx = perm[i:i + 256]
            opt.zero_grad(); loss = lossf(m(Xt[idx]), yt[idx]); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        oof[va] = torch.sigmoid(m(torch.tensor(X[va]))).numpy().ravel()

auc = roc_auc_score(y, oof); pred = (oof >= 0.5).astype(int); acc = accuracy_score(y, pred)
cm = confusion_matrix(y, pred)
print(f"AUC-ROC = {auc:.3f}   accuracy = {acc*100:.2f}%")
print(f"confusion [[TN,FP],[FN,TP]] = {cm.tolist()}  (rows=true non-legal/legal)")

# ---- confusion-matrix figure (slide style) ----
fig, ax = plt.subplots(figsize=(5.2, 4.6))
ax.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cm[i,j]}", ha="center", va="center", fontsize=20,
                color="white" if cm[i, j] > cm.max() * 0.5 else "black")
ax.set_xticks([0, 1]); ax.set_xticklabels(["Non-legal", "Legal"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["Non-legal", "Legal"])
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title(f"Confusion matrix (acc = {acc*100:.1f}%)")
fig.tight_layout(); fig.savefig(f"{HERE}/reports/baseline_confusion_real10k.png", dpi=150); plt.close(fig)

# ---- ROC curve ----
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y, oof)
fig, ax = plt.subplots(figsize=(5.2, 4.6))
ax.plot(fpr, tpr, color="#c0392b", lw=2, label=f"AUC = {auc:.2f}")
ax.fill_between(fpr, tpr, alpha=0.12, color="#c0392b")
ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC curve (out-of-fold)"); ax.legend(loc="lower right")
fig.tight_layout(); fig.savefig(f"{HERE}/reports/baseline_roc_real10k.png", dpi=150); plt.close(fig)

# ---- before-only predicted-distribution per sensitive attribute ----
def bucket(attr, s):
    if attr == "region_origin":
        return s.map(lambda v: v if v in ("Center-West", "Greater Tunis") else "Others")
    if attr == "educ_level":
        return s.map(lambda v: "Low" if v in ("No education", "Primary")
                     else ("Higher education" if v == "Higher education" else "Mid"))
    return s

cmap = plt.get_cmap("tab10")
TITLES = {"Gender": "Gender", "region_origin": "region of origin (7 groups)", "educ_level": "education level"}
print("\n--- per-attribute baseline positive-prediction rate & P-rule ---")
for attr in SENS:
    b = bucket(attr, df[attr]).to_numpy()
    groups = sorted(np.unique(b), key=lambda g: -(oof[b == g] >= 0.5).mean())
    rates = {g: (oof[b == g] >= 0.5).mean() * 100 for g in groups}
    prule = min(rates.values()) / max(rates.values()) * 100
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for j, g in enumerate(groups):
        p = oof[b == g]
        emph = len(groups) <= 3 or g in (groups[0], groups[-1])
        sns.kdeplot(x=p, ax=ax, fill=emph, alpha=0.35 if emph else 0.0, bw_adjust=1.3,
                    clip=(0, 1), common_norm=False, color=cmap(j % 10) if emph else "0.6",
                    lw=2 if emph else 1, label=f"{g} — {rates[g]:.0f}%" if emph else None)
    lo, hi = min(rates.values()), max(rates.values())
    txt = f"Positive-prediction rate\nlowest = {lo:.1f}%\nhighest = {hi:.1f}%\nP-rule = {prule:.1f}%  ({'<80' if prule<80 else '>=80'})"
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round", facecolor="#fff0f0", edgecolor="#d62728", alpha=0.9))
    ax.set_title(f"Sensitive attribute: {TITLES[attr]}", fontsize=12)
    ax.set_xlabel("P(legal_entry = 1 | group)"); ax.set_ylabel("Density")
    ax.set_xlim(0, 1); ax.legend(loc="center left", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{HERE}/reports/baseline_dist_{attr}.png", dpi=150); plt.close(fig)
    print(f"  {attr:14s} P-rule={prule:5.1f}   " + ", ".join(f"{g}={r:.0f}%" for g, r in rates.items()))
print("\nsaved: baseline_confusion_real10k.png, baseline_roc_real10k.png, baseline_dist_{Gender,region_origin,educ_level}.png")
