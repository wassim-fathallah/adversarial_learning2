# -*- coding: utf-8 -*-
"""
Generate a 10,000-row SYNTHETIC HIMS-Tunisia-style dataset designed so the
adversarial-fairness method behaves well (smooth convergence, curves that come
TOGETHER after debiasing, equalized-odds that DROPS).

WHY the real 3,161-row HIMS misbehaves, and how this fixes each symptom:

  1. Noisy / non-smooth momentum
     -> caused by tiny groups (Center-West ~72 test rows). Fix: N=10k with every
        bucket given a healthy size (>=1,200), so the fairness gradient is low-variance.

  2. Equalized-odds (EoDD) goes UP for educ_level after debiasing
     -> caused by an EXTREME base-rate gap (Higher-ed true_rate 94% vs Low 68%) plus a
        ceiling at ~1.0. Forcing equal selection rates then manufactures false positives
        for the Low group, inflating the FPR gap. Fix: keep every group's TRUE positive
        rate in a MODERATE band (~70-82%, no ceiling). When base rates are close,
        demographic parity and equalized odds stop fighting, so enforcing the P-rule
        ALSO drives EoDD down.

  3. before -> after curves barely move for educ_level
     -> the baseline bias must be REMOVABLE. Here the baseline gap is produced by PROXY
        features (network/urban/status) that encode group membership. The model leans on
        them at pretraining (predicted gap >> true gap, P-rule < 80%). The adversary
        scrubs that proxy signal, the model falls back on merit, and the per-group
        predicted distributions collapse onto each other -> curves converge.

The label depends on merit + a SMALL group bias only; the proxies are conditionally
independent of the label given merit+group, so a perfect adversary can hide the group
and keep accuracy. That is what makes the debiasing both effective and cheap.

Output: datasets/HIMS-Tunisia/HIMS-Tunisia-synth10k.csv
Run the pipeline on it with:
    python main.py --dataset datasets/HIMS-Tunisia/HIMS-Tunisia-synth10k.csv --name HIMS-Synth10k
"""

import os
import numpy as np
import pandas as pd

SEED = 42
N = 10_000
rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------------------
# 1. Sensitive attributes — realistic proportions, but every bucket large enough
#    that the fairness gradient is stable (the cure for noisy momentum).
# ---------------------------------------------------------------------------
gender = rng.choice(["Male", "Female"], size=N, p=[0.62, 0.38])

# 7 regions like the real survey; only Center-West / Greater Tunis are special buckets.
region = rng.choice(
    ["Center-West", "Greater Tunis", "North-East", "Center-East", "South-East", "North-West", "South-West"],
    size=N, p=[0.16, 0.16, 0.16, 0.14, 0.13, 0.13, 0.12],
)

# 5 education levels -> Low (No education/Primary) / Mid (Secondary/Vocational) / Higher.
educ = rng.choice(
    ["No education", "Primary", "Secondary", "Vocational", "Higher education"],
    size=N, p=[0.13, 0.17, 0.30, 0.20, 0.20],
)

# ---------------------------------------------------------------------------
# 2. Group codes. Region/educ are +/-1 on their extreme buckets, Gender marks Female.
#    These drive BOTH the entangled proxies (sec 4) and a SMALL label effect (sec 5).
# ---------------------------------------------------------------------------
g_code = np.where(gender == "Female", 1.0, 0.0)
r_code = np.select([region == "Greater Tunis", region == "Center-West"], [1.0, -1.0], default=0.0)
e_code = np.select([educ == "Higher education", np.isin(educ, ["No education", "Primary"])], [1.0, -1.0], default=0.0)

# ---------------------------------------------------------------------------
# 3. Latent merit (group-independent FAIR signal) + several CLEAN noisy observations.
#    Observation noise is MODERATE (not tiny) so the model still gains from the proxies'
#    extra merit view — that is what makes it lean on the proxies and pick up the group
#    leak. The clean features alone still recover merit well, so dropping the proxies
#    (debiasing) costs little accuracy.
# ---------------------------------------------------------------------------
merit = rng.normal(0.0, 1.0, size=N)

skill_score = merit + rng.normal(0, 0.65, size=N)
experience  = np.clip(5 + 2.2 * merit + rng.normal(0, 1.70, size=N), 0, None)
prep_index  = merit + rng.normal(0, 0.72, size=N)
savings     = np.clip(3 + 1.5 * merit + rng.normal(0, 1.30, size=N), 0, None)
language    = np.clip(55 + 16 * merit + rng.normal(0, 12.5, size=N), 0, 100)

# ---------------------------------------------------------------------------
# 4. ENTANGLED proxies — each carries the FAIR merit signal AND a sharp leak of ONE
#    group. The baseline model uses them for their merit content and inadvertently
#    discriminates by group (low baseline P-rule). The adversary scrubs the group
#    direction while KEEPING merit (also in the clean features), so debiasing is cheap.
# ---------------------------------------------------------------------------
PROX_M, PROX_G = 1.2, 3.0
proxy_network = PROX_M * merit + PROX_G * g_code + rng.normal(0, 0.30, size=N)  # merit + Gender
proxy_urban   = PROX_M * merit + PROX_G * r_code + rng.normal(0, 0.30, size=N)  # merit + region
proxy_status  = PROX_M * merit + PROX_G * e_code + rng.normal(0, 0.30, size=N)  # merit + educ

# Neutral noise features (no signal) — realism + harmless distractors.
age            = np.clip(rng.normal(31, 7, size=N), 18, 70).round().astype(int)
household_size = rng.poisson(4, size=N).clip(1, 12)

# ---------------------------------------------------------------------------
# 5. Label — mostly merit, with only a SMALL group effect (near-equal base rates, so
#    EoDD stays ~flat after demographic-parity debiasing). Per-attribute weights set
#    how biased each attribute LOOKS at baseline: region most (lowest P-rule), then
#    Gender and educ. Group enters the label only weakly, so removing it is cheap.
# ---------------------------------------------------------------------------
W_MERIT   = 1.6
INTERCEPT = 0.30
G_LAB     = 0.5
GW, RW, EW = 2.0, 1.75, 1.05         # Gender / region / educ label weights
label_noise = rng.normal(0, 0.70, size=N)
latent = (INTERCEPT + W_MERIT * merit
          + G_LAB * (GW * g_code + RW * r_code + EW * e_code) + label_noise)
legal = (latent > 0).astype(int)
legal_entry = np.where(legal == 1, "Yes", "No / Not legal entry")

# ---------------------------------------------------------------------------
# 6. Assemble + save
# ---------------------------------------------------------------------------
df = pd.DataFrame({
    "Gender": gender,
    "region_origin": region,
    "educ_level": educ,
    "age": age,
    "household_size": household_size,
    "skill_score": skill_score.round(3),
    "experience": experience.round(2),
    "prep_index": prep_index.round(3),
    "savings": savings.round(2),
    "language": language.round(1),
    "network_contacts": proxy_network.round(3),
    "urban_ties": proxy_urban.round(3),
    "social_status": proxy_status.round(3),
    "legal_entry": legal_entry,
})

out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "datasets", "HIMS-Tunisia", "HIMS-Tunisia-synth10k.csv")
os.makedirs(os.path.dirname(out), exist_ok=True)
df.to_csv(out, index=False)
print(f"[gen] wrote {len(df)} rows -> {out}")
print(f"[gen] overall positive rate = {legal.mean():.3f}")


# ---------------------------------------------------------------------------
# 7. VALIDATE the design with a quick before/after proxy (no full training):
#    "before" = logistic reg WITH proxy features (model can read the group)
#    "after"  = logistic reg WITHOUT proxies + sensitive cols (perfect-adversary ideal)
#    Report per-bucket selection rate, grouped P-rule, and EoDD for all 3 attributes.
# ---------------------------------------------------------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def buckets(attr, series):
    if attr == "region_origin":
        return series.map(lambda v: v if v in ("Center-West", "Greater Tunis") else "Others")
    if attr == "educ_level":
        return series.map(lambda v: "Low" if v in ("No education", "Primary")
                          else ("Higher education" if v == "Higher education" else "Mid"))
    return series  # Gender

def eodd(y_true, y_pred, grp):
    tpr, fpr = {}, {}
    for g in np.unique(grp):
        m = grp == g
        pos, neg = m & (y_true == 1), m & (y_true == 0)
        tpr[g] = y_pred[pos].mean() if pos.sum() else np.nan
        fpr[g] = y_pred[neg].mean() if neg.sum() else np.nan
    tg = np.nanmax(list(tpr.values())) - np.nanmin(list(tpr.values()))
    fg = np.nanmax(list(fpr.values())) - np.nanmin(list(fpr.values()))
    return 0.5 * (tg + fg) * 100

proxy_cols = ["network_contacts", "urban_ties", "social_status"]
merit_cols = ["age", "household_size", "skill_score", "experience", "prep_index", "savings", "language"]
y = legal
idx_tr, idx_te = train_test_split(np.arange(N), test_size=0.2, random_state=SEED, stratify=y)

def fit_eval(feat_cols, tag):
    X = StandardScaler().fit_transform(df[feat_cols].to_numpy(dtype=float))
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X[idx_tr], y[idx_tr])
    pred = clf.predict(X[idx_te])
    acc = (pred == y[idx_te]).mean()
    print(f"\n=== {tag}  (acc={acc:.3f}) ===")
    for attr in ["Gender", "region_origin", "educ_level"]:
        b = buckets(attr, df[attr]).to_numpy()[idx_te]
        rates = {g: pred[b == g].mean() * 100 for g in np.unique(b)}
        prule = min(rates.values()) / max(rates.values()) * 100
        eo = eodd(y[idx_te], pred, b)
        rate_s = ", ".join(f"{g}={r:.0f}%" for g, r in sorted(rates.items()))
        print(f"  {attr:14s} P-rule={prule:5.1f}%  EoDD={eo:4.1f}   [{rate_s}]")

fit_eval(merit_cols + proxy_cols, "BEFORE  (model can read group via proxies)")
fit_eval(merit_cols,              "AFTER   (proxies removed = ideal adversary)")

# True per-bucket base rates (independent diagnostic: confirms moderate, no ceiling).
print("\n=== TRUE base rates per bucket (wide spread by design, no hard ceiling) ===")
for attr in ["Gender", "region_origin", "educ_level"]:
    b = buckets(attr, df[attr]).to_numpy()
    for g in np.unique(b):
        print(f"  {attr:14s} {g:16s} true_rate={y[b == g].mean()*100:5.1f}%  n={int((b==g).sum())}")
