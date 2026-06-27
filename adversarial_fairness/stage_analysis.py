# -*- coding: utf-8 -*-
"""
Classifier AND adversary accuracy / AUC at THREE stages on HIMS-real10k:

  (1) TRADITIONAL  : classifier trained WITH the sensitive attributes as inputs.
  (2) BEFORE LOOP  : classifier trained WITHOUT the sensitive attributes (the
                     standard baseline, before the adversarial game).
  (3) AFTER LOOP   : the same classifier after adversarial debiasing.

For each stage we also train a fresh ADVERSARY that tries to recover the sensitive
attributes from the classifier's prediction -> its acc/AUC measures how much the
prediction still leaks the protected group (should collapse to ~0.5 AUC after the
loop = the prediction no longer encodes the sensitive attribute).
"""
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42; np.random.seed(SEED); torch.manual_seed(SEED)
HERE = "C:/Users/jamil/Desktop/adversarial_learning2/adversarial_fairness"
df = pd.read_csv(f"{HERE}/datasets/HIMS-Tunisia/HIMS-Tunisia-real10k.csv")
SENS = ["Gender", "region_origin", "educ_level"]
y = (df.legal_entry == "Yes").astype(int).values

# binary sensitive targets for the adversary (positive_value as in the pipeline)
S = np.c_[(df.Gender == "Female").astype(int),
          (df.region_origin == "Center-West").astype(int),
          (df.educ_level == "Higher education").astype(int)].astype(np.float32)

X_with    = StandardScaler().fit_transform(pd.get_dummies(df.drop(columns=["legal_entry"])).to_numpy(float)).astype(np.float32)
X_without = StandardScaler().fit_transform(pd.get_dummies(df.drop(columns=SENS + ["legal_entry"])).to_numpy(float)).astype(np.float32)

tr, te = train_test_split(np.arange(len(y)), test_size=0.2, random_state=SEED, stratify=y)

def MLP(d, out=1):
    return nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Dropout(0.2),
                         nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, out))

def train_clf(X, epochs=60):
    m = MLP(X.shape[1]); opt = torch.optim.Adam(m.parameters(), 1e-3); lf = nn.BCEWithLogitsLoss()
    Xt = torch.tensor(X[tr]); yt = torch.tensor(y[tr], dtype=torch.float32).unsqueeze(1)
    for _ in range(epochs):
        p = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 256):
            idx = p[i:i+256]; opt.zero_grad(); lf(m(Xt[idx]), yt[idx]).backward(); opt.step()
    return m

def clf_scores(m, X):
    m.eval()
    with torch.no_grad():
        prob = torch.sigmoid(m(torch.tensor(X[te]))).numpy().ravel()
    return roc_auc_score(y[te], prob), accuracy_score(y[te], (prob >= .5).astype(int)), prob

def adversary_recovery(prob_tr, prob_te, epochs=120):
    """Fresh adversary: predict the 3 sensitive attrs from the classifier prob."""
    a = MLP(1, out=3); opt = torch.optim.Adam(a.parameters(), 2e-3); lf = nn.BCEWithLogitsLoss()
    Xt = torch.tensor(prob_tr.reshape(-1, 1)); St = torch.tensor(S[tr])
    for _ in range(epochs):
        opt.zero_grad(); lf(a(Xt), St).backward(); opt.step()
    a.eval()
    with torch.no_grad():
        pr = torch.sigmoid(a(torch.tensor(prob_te.reshape(-1, 1)))).numpy()
    aucs = [roc_auc_score(S[te][:, j], pr[:, j]) for j in range(3)]
    accs = [accuracy_score(S[te][:, j], (pr[:, j] >= .5).astype(int)) for j in range(3)]
    return float(np.mean(aucs)), float(np.mean(accs)), aucs

def grl(x, lam):                      # gradient-reversal
    return x.detach() * (1 + lam) - x * lam

def train_debiased(X, epochs=120, lam=4.5):
    clf = MLP(X.shape[1]); adv = MLP(1, out=3)
    o_c = torch.optim.Adam(clf.parameters(), 1e-3); o_a = torch.optim.Adam(adv.parameters(), 2e-3)
    bce = nn.BCEWithLogitsLoss()
    Xt = torch.tensor(X[tr]); yt = torch.tensor(y[tr], dtype=torch.float32).unsqueeze(1); St = torch.tensor(S[tr])
    for _ in range(epochs):
        p = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 256):
            idx = p[i:i+256]
            prob = torch.sigmoid(clf(Xt[idx]))
            o_a.zero_grad(); bce(adv(prob.detach()), St[idx]).backward(); o_a.step()   # adversary learns
            prob = torch.sigmoid(clf(Xt[idx]))
            task = bce(clf(Xt[idx]), yt[idx])
            leak = bce(adv(grl(prob, lam)), St[idx])      # classifier fights the adversary
            o_c.zero_grad(); (task + leak).backward(); o_c.step()
    return clf

print("Stage                                   | Classifier AUC / acc | Adversary AUC / acc (recovers sensitive)")
print("-"*110)
# (1) traditional — sensitive attrs included as inputs
m1 = train_clf(X_with); auc1, acc1, pr1 = clf_scores(m1, X_with)
_, _, prtr1 = clf_scores(m1, X_with); a_auc1, a_acc1, det1 = adversary_recovery(prtr1[:len(tr)] if False else torch.sigmoid(m1(torch.tensor(X_with[tr]))).detach().numpy().ravel(), pr1)
print(f"(1) Traditional  (WITH sensitive attrs)  |   {auc1:.3f} / {acc1*100:4.1f}%    |   {a_auc1:.3f} / {a_acc1*100:4.1f}%")
# (2) baseline — sensitive removed
m2 = train_clf(X_without); auc2, acc2, pr2 = clf_scores(m2, X_without)
a_auc2, a_acc2, det2 = adversary_recovery(torch.sigmoid(m2(torch.tensor(X_without[tr]))).detach().numpy().ravel(), pr2)
print(f"(2) Before loop  (WITHOUT, baseline)     |   {auc2:.3f} / {acc2*100:4.1f}%    |   {a_auc2:.3f} / {a_acc2*100:4.1f}%")
# (3) debiased — after the adversarial loop
m3 = train_debiased(X_without); auc3, acc3, pr3 = clf_scores(m3, X_without)
a_auc3, a_acc3, det3 = adversary_recovery(torch.sigmoid(m3(torch.tensor(X_without[tr]))).detach().numpy().ravel(), pr3)
print(f"(3) After loop   (WITHOUT, debiased)     |   {auc3:.3f} / {acc3*100:4.1f}%    |   {a_auc3:.3f} / {a_acc3*100:4.1f}%")
print("\nadversary AUC per attr [Gender, region, educ]:")
print("  (1) traditional:", [round(x,3) for x in det1])
print("  (2) baseline   :", [round(x,3) for x in det2])
print("  (3) debiased   :", [round(x,3) for x in det3])

# ---- figure: Classifier panel + Adversary panel, AUC & accuracy, every bar labelled ----
clf_stages = ["Traditional\n(with sensitive)", "Baseline\n(before loop)", "Debiased\n(after loop)"]
clf_auc = [auc1, auc2, auc3]; clf_acc = [acc1, acc2, acc3]
adv_stages = ["Baseline\n(before loop)", "Debiased\n(after loop)"]   # no adversary for traditional
adv_auc = [a_auc2, a_auc3]; adv_acc = [a_acc2, a_acc3]
w = 0.38

def labelled(ax, xs, vals, fmt="{:.3f}"):
    for xi, v in zip(xs, vals):
        ax.text(xi, v + 0.012, fmt.format(v), ha="center", fontsize=8.5)

fig, (axc, axa) = plt.subplots(1, 2, figsize=(13, 5.2))
# Classifier — all three stages
xc = np.arange(3)
axc.bar(xc - w/2, clf_auc, w, label="AUC", color="#1f77b4")
axc.bar(xc + w/2, clf_acc, w, label="accuracy", color="#9ecae1")
labelled(axc, xc - w/2, clf_auc); labelled(axc, xc + w/2, clf_acc)
axc.set_title("Classifier — AUC & accuracy"); axc.set_ylim(0, 1.05)
axc.set_xticks(xc); axc.set_xticklabels(clf_stages, fontsize=9); axc.legend(loc="lower left")
axc.grid(axis="y", alpha=.3)
# Adversary — baseline vs after loop only (recovers the sensitive attribute from the prediction)
xa = np.arange(2)
axa.bar(xa - w/2, adv_auc, w, label="AUC", color="#d62728")
axa.bar(xa + w/2, adv_acc, w, label="accuracy", color="#f4a6a6")
labelled(axa, xa - w/2, adv_auc); labelled(axa, xa + w/2, adv_acc)
axa.axhline(0.5, ls="--", color="gray", lw=1)
axa.text(1.45, 0.515, "AUC 0.5 = no leakage", fontsize=8, color="gray", ha="right")
axa.set_title("Adversary — AUC & accuracy (recovers sensitive)"); axa.set_ylim(0, 1.05)
axa.set_xticks(xa); axa.set_xticklabels(adv_stages, fontsize=9); axa.legend(loc="lower left")
axa.grid(axis="y", alpha=.3); axa.set_xlim(-0.6, 1.6)
fig.suptitle("HIMS-Tunisia — classifier vs adversary, before vs after the adversarial loop", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("fig_stages_HIMS-Tunisia.png", dpi=150)
print("\nsaved fig_stages_HIMS-Tunisia.png")
