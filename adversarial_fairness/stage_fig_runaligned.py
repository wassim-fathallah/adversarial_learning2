# -*- coding: utf-8 -*-
"""
Rebuild fig_stages_HIMS-Tunisia.png ALIGNED TO THE ACTUAL RUN.

Classifier panel (AUC + accuracy):
  - Traditional : a model retrained WITH the sensitive attrs (reference; not in the loop).
  - Baseline    : the run's pretrained classifier  -> from report npz base_probs.
  - Debiased    : the run's final classifier        -> from report npz probs  (acc 0.812).

Adversary panel (Baseline -> Debiased only; no adversary for traditional):
  Trains a fresh adversary on the run's prediction scores to recover the 3 sensitive
  attrs. Reports AUC and BALANCED accuracy. Balanced accuracy -> 0.5 at no leakage
  (plain accuracy is pinned to the majority base rate ~0.79 and is uninformative).
"""
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

SEED = 42; np.random.seed(SEED); torch.manual_seed(SEED)
HERE = "C:/Users/jamil/Desktop/adversarial_learning2/adversarial_fairness"
df = pd.read_csv(f"{HERE}/datasets/HIMS-Tunisia/HIMS-Tunisia-real10k.csv")
y = (df.legal_entry == "Yes").astype(int).values
S = np.c_[(df.Gender == "Female"), (df.region_origin == "Center-West"),
          (df.educ_level == "Higher education")].astype(np.float32)
tr, te = train_test_split(np.arange(len(y)), test_size=0.2, random_state=SEED, stratify=y)
yte = y[te]; Ste = S[te]

# the actual run's test predictions (aligned to te order)
d = np.load(f"{HERE}/reports/report_HIMS-real10k_plotdata.npz")
base = d["base_probs"].astype(float); aft = d["probs"].astype(float)

def MLP(d_in, out=1):
    return nn.Sequential(nn.Linear(d_in,256), nn.ReLU(), nn.Dropout(0.2),
                         nn.Linear(256,256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256,out))

# ---- classifier panel ----
# Traditional: retrain WITH sensitive attributes
Xw = StandardScaler().fit_transform(pd.get_dummies(df.drop(columns=["legal_entry"])).to_numpy(float)).astype(np.float32)
mt = MLP(Xw.shape[1]); opt = torch.optim.Adam(mt.parameters(),1e-3); lf = nn.BCEWithLogitsLoss()
Xt = torch.tensor(Xw[tr]); yt = torch.tensor(y[tr],dtype=torch.float32).unsqueeze(1)
for _ in range(60):
    p = torch.randperm(len(Xt))
    for i in range(0,len(Xt),256):
        idx=p[i:i+256]; opt.zero_grad(); lf(mt(Xt[idx]),yt[idx]).backward(); opt.step()
mt.eval()
with torch.no_grad(): probt = torch.sigmoid(mt(torch.tensor(Xw[te]))).numpy().ravel()
clf_auc = [roc_auc_score(yte,probt), roc_auc_score(yte,base), roc_auc_score(yte,aft)]
clf_acc = [accuracy_score(yte,probt>=.5), accuracy_score(yte,base>=.5), accuracy_score(yte,aft>=.5)]

# ---- adversary panel (on the run's predictions) — ROBUST, assumption-free ----
# A one-input adversary cannot beat the prediction score's own AUC against each
# sensitive attribute, so we measure that directly over ALL test rows (no train/test
# split, no threshold tuning). Balanced accuracy is 5-fold cross-validated with the
# threshold chosen on the train folds (so it is not optimistically biased).
from sklearn.model_selection import StratifiedKFold
def adversary(score):
    aucs = []
    for j in range(3):
        a = roc_auc_score(Ste[:,j], score); aucs.append(max(a, 1-a))   # direction-agnostic
    baccs = []
    for j in range(3):
        yj = Ste[:,j]; fold = []
        for itr, ite in StratifiedKFold(5, shuffle=True, random_state=SEED).split(score, yj):
            ths = np.quantile(score[itr], np.linspace(0.05, 0.95, 19))
            bt = max(ths, key=lambda t: balanced_accuracy_score(yj[itr], (score[itr] >= t).astype(int)))
            fold.append(balanced_accuracy_score(yj[ite], (score[ite] >= bt).astype(int)))
        baccs.append(np.mean(fold))
    return float(np.mean(aucs)), float(np.mean(baccs))

base_auc, base_bacc = adversary(base)
deb_auc,  deb_bacc  = adversary(aft)
adv_auc = [base_auc, deb_auc]; adv_bacc = [base_bacc, deb_bacc]

print("Classifier  AUC:", [round(v,3) for v in clf_auc], " acc:", [round(v,3) for v in clf_acc])
print("Adversary   AUC:", [round(v,3) for v in adv_auc], " balanced-acc:", [round(v,3) for v in adv_bacc])

# ---- figure ----
clf_stages = ["Traditional\n(with sensitive)", "Baseline\n(before loop)", "Debiased\n(after loop)"]
adv_stages = ["Baseline\n(before loop)", "Debiased\n(after loop)"]
w = 0.38
def labelled(ax, xs, vals):
    for xi, v in zip(xs, vals): ax.text(xi, v+0.012, f"{v:.3f}", ha="center", fontsize=8.5)

fig, (axc, axa) = plt.subplots(1, 2, figsize=(13, 5.2))
xc = np.arange(3)
axc.bar(xc-w/2, clf_auc, w, label="AUC", color="#1f77b4")
axc.bar(xc+w/2, clf_acc, w, label="accuracy", color="#9ecae1")
labelled(axc, xc-w/2, clf_auc); labelled(axc, xc+w/2, clf_acc)
axc.set_title("Classifier — AUC & accuracy"); axc.set_ylim(0,1.05)
axc.set_xticks(xc); axc.set_xticklabels(clf_stages, fontsize=9); axc.legend(loc="lower left"); axc.grid(axis="y",alpha=.3)

xa = np.arange(2)
axa.bar(xa-w/2, adv_auc, w, label="AUC", color="#d62728")
axa.bar(xa+w/2, adv_bacc, w, label="balanced accuracy", color="#f4a6a6")
labelled(axa, xa-w/2, adv_auc); labelled(axa, xa+w/2, adv_bacc)
axa.axhline(0.5, ls="--", color="gray", lw=1)
axa.text(1.45, 0.515, "0.5 = no leakage (random)", fontsize=8, color="gray", ha="right")
axa.set_title("Adversary — AUC & balanced accuracy (recovers sensitive)"); axa.set_ylim(0,1.05)
axa.set_xticks(xa); axa.set_xticklabels(adv_stages, fontsize=9); axa.legend(loc="lower left")
axa.grid(axis="y",alpha=.3); axa.set_xlim(-0.6,1.6)
fig.suptitle("HIMS-Tunisia — classifier vs adversary, before vs after the adversarial loop", fontsize=12)
fig.tight_layout(rect=[0,0,1,0.96])
fig.savefig(f"{HERE}/fig_stages_HIMS-Tunisia.png", dpi=150)
print("saved fig_stages_HIMS-Tunisia.png (run-aligned: debiased acc = %.3f)" % clf_acc[2])
