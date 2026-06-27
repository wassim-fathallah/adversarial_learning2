# -*- coding: utf-8 -*-
"""
Regenerate the BASELINE section (ROC, confusion, per-attribute distributions) from
the ACTUAL run's baseline classifier — its predicted probabilities on the 2,000-row
test set (report npz `base_probs`) — instead of baseline_analysis.py's separate 5-fold
CV model. This makes the baseline charts consistent with the run's before/after curves,
ROC and stage figure (baseline acc 87.8%, AUC 0.944, P-rules 67.0/62.5/62.2).

Overwrites the existing filenames so slides that embed them update automatically.
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import seaborn as sns; sns.set_style("whitegrid")
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, accuracy_score

HERE = "C:/Users/jamil/Desktop/adversarial_learning2/adversarial_fairness"
df = pd.read_csv(f"{HERE}/datasets/HIMS-Tunisia/HIMS-Tunisia-real10k.csv")
y = (df.legal_entry == "Yes").astype(int).values
tr, te = train_test_split(np.arange(len(y)), test_size=0.2, random_state=42, stratify=y)
yte = y[te]
prob = np.load(f"{HERE}/reports/report_HIMS-real10k_plotdata.npz")["base_probs"].astype(float)
pred = (prob >= 0.5).astype(int)
auc = roc_auc_score(yte, prob); acc = accuracy_score(yte, pred); cm = confusion_matrix(yte, pred)
print(f"run baseline: acc={acc*100:.2f}%  AUC={auc:.3f}  confusion={cm.tolist()} (n={cm.sum()})")

# ---- confusion matrix ----
fig, ax = plt.subplots(figsize=(5.2, 4.6)); ax.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cm[i,j]}", ha="center", va="center", fontsize=20,
                color="white" if cm[i, j] > cm.max()*0.5 else "black")
ax.set_xticks([0,1]); ax.set_xticklabels(["Non-legal","Legal"])
ax.set_yticks([0,1]); ax.set_yticklabels(["Non-legal","Legal"])
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title(f"Confusion matrix — run baseline (acc {acc*100:.1f}%, {cm.sum()} test rows)")
fig.tight_layout(); fig.savefig(f"{HERE}/reports/baseline_confusion_real10k.png", dpi=150); plt.close(fig)

# ---- ROC ----
fpr, tpr, _ = roc_curve(yte, prob)
fig, ax = plt.subplots(figsize=(5.2, 4.6))
ax.plot(fpr, tpr, color="#c0392b", lw=2, label=f"AUC = {auc:.3f}")
ax.fill_between(fpr, tpr, alpha=0.12, color="#c0392b")
ax.plot([0,1],[0,1],"--",color="gray",lw=1)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC curve — run baseline (2,000 test rows)"); ax.legend(loc="lower right")
fig.tight_layout(); fig.savefig(f"{HERE}/reports/baseline_roc_real10k.png", dpi=150); plt.close(fig)

# ---- per-attribute distributions ----
def bucket(attr, s):
    if attr == "region_origin":
        return s.map(lambda v: v if v in ("Center-West","Greater Tunis") else "Others")
    if attr == "educ_level":
        return s.map(lambda v: "Low" if v in ("No education","Primary")
                     else ("Higher education" if v == "Higher education" else "Mid"))
    return s
TITLES = {"Gender":"Gender","region_origin":"region of origin (3 groups)","educ_level":"education level (3 groups)"}
cmap = plt.get_cmap("tab10")
for attr in ["Gender","region_origin","educ_level"]:
    b = bucket(attr, df[attr]).to_numpy()[te]
    groups = sorted(np.unique(b), key=lambda g: -(prob[b == g] >= 0.5).mean())
    rates = {g: (prob[b == g] >= 0.5).mean()*100 for g in groups}
    prule = min(rates.values())/max(rates.values())*100
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for j, g in enumerate(groups):
        sns.kdeplot(x=prob[b == g], ax=ax, fill=True, alpha=0.32, bw_adjust=1.3, clip=(0,1),
                    common_norm=False, color=cmap(j), lw=2, label=f"{g} — {rates[g]:.0f}%")
    txt = (f"Positive-prediction rate\nlowest = {min(rates.values()):.1f}%\n"
           f"highest = {max(rates.values()):.1f}%\nP-rule = {prule:.1f}%")
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round", facecolor="#fff0f0", edgecolor="#d62728", alpha=0.9))
    ax.set_title(f"Sensitive attribute: {TITLES[attr]} (run baseline)", fontsize=12)
    ax.set_xlabel("P(legal_entry = 1 | group)"); ax.set_ylabel("Density")
    ax.set_xlim(0,1); ax.legend(loc="center left", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{HERE}/reports/baseline_dist_{attr}.png", dpi=150); plt.close(fig)
    print(f"  {attr:14s} P-rule={prule:5.1f}  " + ", ".join(f"{g}={r:.0f}%" for g,r in rates.items()))
print("\nsaved (run-consistent): baseline_confusion_real10k.png, baseline_roc_real10k.png, baseline_dist_{Gender,region_origin,educ_level}.png")
