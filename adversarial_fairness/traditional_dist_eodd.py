# -*- coding: utf-8 -*-
"""
TRADITIONAL classifier (fed WITH the sensitive attributes as inputs) on HIMS-real10k.
Produces, for Gender / region_origin (3 buckets) / educ_level (3 buckets):
  1. the predicted P(legal_entry=1) distribution per subgroup (KDE),
  2. the equalized-odds (EoDD) gap per attribute.

This is the stage-(1) "Traditional" model from the 3-stage analysis: the classifier
CAN see Gender/region/education directly, so it discriminates the most.
"""
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_style("whitegrid")

SEED = 42; np.random.seed(SEED); torch.manual_seed(SEED)
HERE = "C:/Users/jamil/Desktop/adversarial_learning2/adversarial_fairness"
df = pd.read_csv(f"{HERE}/datasets/HIMS-Tunisia/HIMS-Tunisia-real10k.csv")
y = (df.legal_entry == "Yes").astype(int).values

# WITH sensitive attributes: keep Gender/region_origin/educ_level in the inputs.
X = StandardScaler().fit_transform(
    pd.get_dummies(df.drop(columns=["legal_entry"])).to_numpy(float)).astype(np.float32)
tr, te = train_test_split(np.arange(len(y)), test_size=0.2, random_state=SEED, stratify=y)

def MLP(d):
    return nn.Sequential(nn.Linear(d,256), nn.ReLU(), nn.Dropout(0.2),
                         nn.Linear(256,256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256,1))
m = MLP(X.shape[1]); opt = torch.optim.Adam(m.parameters(), 1e-3); lf = nn.BCEWithLogitsLoss()
Xt = torch.tensor(X[tr]); yt = torch.tensor(y[tr], dtype=torch.float32).unsqueeze(1)
for _ in range(60):
    p = torch.randperm(len(Xt))
    for i in range(0, len(Xt), 256):
        idx = p[i:i+256]; opt.zero_grad(); lf(m(Xt[idx]), yt[idx]).backward(); opt.step()
m.eval()
with torch.no_grad():
    prob = torch.sigmoid(m(torch.tensor(X[te]))).numpy().ravel()
pred = (prob >= 0.5).astype(int); yte = y[te]
print(f"[traditional] AUC={roc_auc_score(yte,prob):.3f}  acc={accuracy_score(yte,pred)*100:.1f}%")

# grouped buckets (match the pipeline)
def bucket(attr, s):
    if attr == "region_origin":
        return s.map(lambda v: v if v in ("Center-West","Greater Tunis") else "Others")
    if attr == "educ_level":
        return s.map(lambda v: "Low" if v in ("No education","Primary")
                     else ("Higher education" if v == "Higher education" else "Mid"))
    return s

def eodd(yt_, yp_, g):
    tpr, fpr = [], []
    for gg in np.unique(g):
        msk = g == gg; pos = msk & (yt_ == 1); neg = msk & (yt_ == 0)
        if pos.sum(): tpr.append(yp_[pos].mean())
        if neg.sum(): fpr.append(yp_[neg].mean())
    return 0.5 * ((max(tpr)-min(tpr)) + (max(fpr)-min(fpr))) * 100

ATTRS = ["Gender", "region_origin", "educ_level"]
ORDER = {"Gender": ["Male","Female"],
         "region_origin": ["Greater Tunis","Center-West","Others"],
         "educ_level": ["Low","Mid","Higher education"]}
TITLES = {"Gender":"Gender","region_origin":"region of origin (3 groups)","educ_level":"education level (3 groups)"}
cmap = plt.get_cmap("tab10")

# ---- one distribution PNG PER attribute ----
eodds = {}
for attr in ATTRS:
    b = bucket(attr, df[attr]).to_numpy()[te]
    groups = [g for g in ORDER[attr] if g in np.unique(b)]
    rates = {g: (prob[b == g] >= 0.5).mean()*100 for g in groups}
    prule = min(rates.values())/max(rates.values())*100
    eo = eodd(yte, pred, b); eodds[attr] = eo
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    for j, g in enumerate(groups):
        sns.kdeplot(x=prob[b == g], ax=ax, fill=True, alpha=0.32, bw_adjust=1.3, clip=(0,1),
                    common_norm=False, color=cmap(j), lw=2, label=f"{g} — {rates[g]:.0f}%")
    box = (f"Positive-prediction rate\nlowest {min(rates.values()):.1f}%   highest {max(rates.values()):.1f}%\n"
           f"P-rule {prule:.1f}%    EoDD {eo:.1f}%")
    ax.text(0.5, 0.98, box, transform=ax.transAxes, fontsize=9, va="top", ha="center",
            bbox=dict(boxstyle="round", facecolor="#fff0f0", edgecolor="#d62728", alpha=0.9))
    ax.set_title(f"HIMS-Tunisia (WITH sensitive attrs) — {TITLES[attr]}", fontsize=11)
    ax.set_xlabel("P(legal_entry = 1 | group)"); ax.set_ylabel("Density"); ax.set_xlim(0,1)
    ax.legend(loc="center left", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{HERE}/reports/traditional_dist_{attr}.png", dpi=150); plt.close(fig)

# ---- EoDD bar chart ----
fig, ax = plt.subplots(figsize=(7,5))
xs = np.arange(len(ATTRS))
ax.bar(xs, [eodds[a] for a in ATTRS], 0.55, color=["#2ca02c","#ff7f0e","#9467bd"])
for xi, a in zip(xs, ATTRS):
    ax.text(xi, eodds[a]+0.2, f"{eodds[a]:.1f}", ha="center", fontsize=10)
ax.set_xticks(xs); ax.set_xticklabels(["Gender","region_origin","educ_level"])
ax.set_ylabel("equalized-odds gap (%)  lower = fairer")
ax.set_title("HIMS-Tunisia — EoDD | classifier WITH sensitive attributes")
ax.grid(axis="y", alpha=.3)
fig.tight_layout(); fig.savefig(f"{HERE}/reports/traditional_eodd_HIMS-real10k.png", dpi=150); plt.close(fig)

print("\nper-attribute (traditional, WITH sensitive attrs):")
for attr in ATTRS:
    b = bucket(attr, df[attr]).to_numpy()[te]
    groups = [g for g in ORDER[attr] if g in np.unique(b)]
    rates = {g: (prob[b == g] >= 0.5).mean()*100 for g in groups}
    prule = min(rates.values())/max(rates.values())*100
    print(f"  {attr:14s} P-rule={prule:5.1f}  EoDD={eodds[attr]:4.1f}   "
          + ", ".join(f"{g}={r:.0f}%" for g,r in rates.items()))
print("\nsaved: reports/traditional_distributions_HIMS-real10k.png, reports/traditional_eodd_HIMS-real10k.png")
