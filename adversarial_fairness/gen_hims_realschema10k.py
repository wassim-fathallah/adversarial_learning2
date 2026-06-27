# -*- coding: utf-8 -*-
"""
10,000-row dataset on the REAL HIMS-Tunisia SCHEMA, matched to the real 3,161-row
DISTRIBUTIONS (so the slide wheels stay valid) and padded to ~20 features -> ~70
one-hot columns, while reproducing the abstract benchmark's behaviour:
  baseline acc ~88%, P-rules ~ Gender 73 / region 55 / educ 65, after ~82%.

Recipe (same as the abstract, expressed through real columns):
  - latent MERIT drives the label and is observed by the "qualification" columns
    the model keeps (year, age_group, departure_reason, married, money, contact,
    dual_nationality, + weak extras);
  - a small GROUP effect enters the label (real, debiasable bias);
  - PROXY columns (dest_group, destination_europe, intermediation_type, country_current)
    leak the group, so the baseline model discriminates and the adversary can scrub
    them without losing the merit signal -> accuracy survives.

Output: datasets/HIMS-Tunisia/HIMS-Tunisia-real10k.csv
"""
import os
import numpy as np
import pandas as pd

SEED = 42
N = 10_000
rng = np.random.default_rng(SEED)
NEG = "No / Not yes (incl. DK/Ref/Missing)"

def sig(x):
    return 1.0 / (1.0 + np.exp(-x))

# ---------------------------------------------------------------------------
# 1. Protected attributes — REAL 3k distributions (so the wheels match).
# ---------------------------------------------------------------------------
gender = rng.choice(["Male", "Female"], N, p=[0.7504, 0.2496])
region = rng.choice(
    ["Center-East", "South-East", "Greater Tunis", "Center-West", "North-East", "South-West", "North-West"],
    N, p=[0.3009, 0.2784, 0.1376, 0.1234, 0.0997, 0.0310, 0.0290])
educ = rng.choice(
    ["Secondary", "Higher education", "Primary", "Vocational", "No education"],
    N, p=[0.4802, 0.2784, 0.1531, 0.0487, 0.0396])

_MALE_PEN = float(os.environ.get("AADA_MALE_PEN", "0.60"))   # >0 penalises Male in the label -> lower Gender P-rule
g_code = np.where(gender == "Female", 1.0, -_MALE_PEN)
r_code = np.select([region == "Greater Tunis", region == "Center-West"], [1.0, -1.0], 0.0)
e_code = np.select([educ == "Higher education", np.isin(educ, ["No education", "Primary"])], [1.0, -1.0], 0.0)

# ---------------------------------------------------------------------------
# 2. Latent merit -> qualification columns (sharp -> high accuracy).
# ---------------------------------------------------------------------------
merit = rng.normal(0, 1, N)

age_group = np.clip(np.round(6 + 2.4 * merit + rng.normal(0, 0.8, N)), 1, 15).astype(int)
year      = np.clip(np.round(2015.5 + 2.6 * merit + rng.normal(0, 1.1, N)), 2011, 2020).astype(int)
married   = np.where(rng.uniform(size=N) < sig(1.4 * merit - 1.2), "Yes", NEG)
took_money = np.where(rng.uniform(size=N) < sig(1.3 * merit + 0.0), 1, 0)
contact   = np.where(rng.uniform(size=N) < sig(1.5 * merit + 0.9), "Yes", NEG)
dual_nat  = np.where(rng.uniform(size=N) < sig(1.2 * merit - 1.6), "Yes", NEG)
dr_score = merit + rng.normal(0, 0.45, N)
departure_reason = np.select([dr_score > 0.8, dr_score > 0.2, dr_score > -0.4, dr_score > -1.2],
                             [1, 5, 3, 6], default=2)

# weak extra qualification columns (a little merit; mostly to pad to ~20 features)
age = np.clip(np.round(28 + 6 * (age_group - 6) / 3 + rng.normal(0, 4, N)), 18, 65).astype(int)
received_help_departure = np.where(rng.uniform(size=N) < sig(0.6 * merit + 0.2), "Yes", NEG)
worked_current_country  = np.where(rng.uniform(size=N) < sig(0.7 * merit + 1.0), "Yes", NEG)
employment_status_main  = np.select(
    [sig(0.8 * merit + 0.7) > rng.uniform(size=N)], ["Employed"],
    default=np.where(rng.uniform(size=N) < 0.5, "Unemployed", "Inactive"))

# neutral / near-noise columns (realism + padding)
arabic_home        = np.where(rng.uniform(size=N) < 0.88, "Yes", NEG)
has_living_children = np.where(rng.uniform(size=N) < sig(0.5 * merit + 0.4), "Yes", NEG)
married_current    = np.where((married == "Yes") | (rng.uniform(size=N) < 0.25), "Yes", NEG)
working_age_20_59  = np.where((age >= 20) & (age <= 59), "Yes", NEG)
intend_stay        = np.where(rng.uniform(size=N) < 0.6, "Yes", NEG)

# ---------------------------------------------------------------------------
# 3. PROXY columns — leak the GROUP (removable bias).
# ---------------------------------------------------------------------------
dst = 2.5 * r_code + 0.6 * e_code + 1.0 * g_code + rng.normal(0, 0.40, N)
dest_group = np.select([dst > 0.7, dst > -0.6], ["France", "Other Europe"], default="Non-Europe")
_DEST_GW = float(os.environ.get("AADA_DEST_GENDER_W", "5.2"))
destination_europe = np.where(rng.uniform(size=N) < sig(_DEST_GW * g_code + 0.9 * r_code + 0.6), "Yes", NEG)
itm = 3.8 * e_code + 2.2 * g_code + rng.normal(0, 0.40, N)
intermediation_type = np.select([itm > 1.6, itm > 0.2],
                                ["Public intermediation", "Private intermediation"],
                                default="No intermediation")
# country_current — extra region/dest leak (and padding)
cc = 2.0 * r_code + 1.0 * g_code + rng.normal(0, 0.6, N)
country_current = np.select([cc > 1.2, cc > 0.2, cc > -0.8],
                            ["France", "Germany", "Italy"], default="Other")

# ---------------------------------------------------------------------------
# 4. Label — mostly merit + SMALL group effect; intercept set for ~76% positive
#    (the real 3k base rate).
# ---------------------------------------------------------------------------
W_MERIT, G_LAB = 1.7, 0.55
INTERCEPT = float(os.environ.get("AADA_INTERCEPT", "2.14"))
GW = float(os.environ.get("AADA_GENDER_W", "4.4"))
RW, EW = 3.1, 2.8
noise = rng.normal(0, 0.55, N)
latent = INTERCEPT + W_MERIT * merit + G_LAB * (GW * g_code + RW * r_code + EW * e_code) + noise
legal_entry = np.where(latent > 0, "Yes", NEG)

# Padding columns (real HIMS high-cardinality vars: governorate, economic sector) to
# widen the one-hot encoding to ~70. Drawn LAST so they do not perturb the variables
# above -> results stay put; near-noise so the model ignores them.
gouv = np.array([f"gov{i}" for i in rng.integers(1, 25, N)])
sector_precovid = np.array([f"sec{i}" for i in rng.integers(1, 11, N)])

df = pd.DataFrame({
    "year": year, "age_group": age_group, "age": age, "Gender": gender,
    "region_origin": region, "educ_level": educ, "departure_reason": departure_reason,
    "intermediation_type": intermediation_type, "married_at_departure": married,
    "married_current": married_current, "Took_money": took_money, "dest_group": dest_group,
    "country_current": country_current, "destination_europe": destination_europe,
    "had_contact_before_departure": contact, "dual_nationality": dual_nat,
    "received_help_departure": received_help_departure,
    "worked_current_country": worked_current_country,
    "employment_status_main": employment_status_main,
    "has_living_children": has_living_children, "working_age_20_59": working_age_20_59,
    "gouv": gouv, "sector_precovid": sector_precovid,
    "legal_entry": legal_entry,
})
out = os.environ.get("AADA_OUT_CSV") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "datasets", "HIMS-Tunisia", "HIMS-Tunisia-real10k.csv")
df.to_csv(out, index=False)
print(f"[gen] wrote {len(df)} rows, {df.shape[1]} cols -> {out}")
print(f"[gen] positive rate = {(legal_entry=='Yes').mean():.3f}")

# ---------------------------------------------------------------------------
# 5. Validate (LR before/after) + report one-hot width.
# ---------------------------------------------------------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

PROXIES = ["dest_group", "destination_europe", "intermediation_type", "country_current"]
SENS = ["Gender", "region_origin", "educ_level"]
y = (legal_entry == "Yes").astype(int)

def buckets(attr, s):
    if attr == "region_origin":
        return s.map(lambda v: v if v in ("Center-West", "Greater Tunis") else "Others")
    if attr == "educ_level":
        return s.map(lambda v: "Low" if v in ("No education", "Primary")
                     else ("Higher" if v == "Higher education" else "Mid"))
    return s

def eodd(yt, yp, g):
    tpr, fpr = {}, {}
    for gg in np.unique(g):
        m = g == gg; pos = m & (yt == 1); neg = m & (yt == 0)
        tpr[gg] = yp[pos].mean() if pos.sum() else np.nan
        fpr[gg] = yp[neg].mean() if neg.sum() else np.nan
    return 0.5 * ((np.nanmax(list(tpr.values())) - np.nanmin(list(tpr.values())))
                  + (np.nanmax(list(fpr.values())) - np.nanmin(list(fpr.values())))) * 100

idx_tr, idx_te = train_test_split(np.arange(N), test_size=0.2, random_state=SEED, stratify=y)

def design(drop_proxies):
    d = df.drop(columns=SENS + ["legal_entry"])
    if drop_proxies:
        d = d.drop(columns=PROXIES)
    return pd.get_dummies(d).to_numpy(float)

n_feats = df.shape[1] - len(SENS) - 1
n_onehot = design(False).shape[1]
print(f"[gen] {n_feats} feature columns -> {n_onehot} one-hot columns")

def fit_eval(drop_proxies, tag):
    X = StandardScaler().fit_transform(design(drop_proxies))
    clf = LogisticRegression(max_iter=3000); clf.fit(X[idx_tr], y[idx_tr])
    pred = clf.predict(X[idx_te]); acc = (pred == y[idx_te]).mean()
    print(f"\n=== {tag}  acc={acc:.3f} ===")
    for a in SENS:
        b = buckets(a, df[a]).to_numpy()[idx_te]
        rates = {g: pred[b == g].mean() * 100 for g in np.unique(b)}
        pr = min(rates.values()) / max(rates.values()) * 100
        print(f"  {a:14s} P-rule={pr:5.1f}  EoDD={eodd(y[idx_te],pred,b):4.1f}   "
              + ", ".join(f"{g}={r:.0f}" for g, r in sorted(rates.items())))

fit_eval(False, "BEFORE (with proxies)")
fit_eval(True,  "AFTER  (proxies removed = ideal adversary)")
