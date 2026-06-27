# -*- coding: utf-8 -*-
"""
Fingerprint test twin (BIASED on all three attributes).

Same generative recipe as HIMS-real10k (Gender-67), but with a DIFFERENT schema
and -- crucially -- the DISADVANTAGED group is the MODE (most-frequent value) of
each sensitive column. The pipeline treats this file as UNKNOWN, so it binarizes
each sensitive attribute on its mode; making the mode the biased group means the
baseline shows a real gap on sex, origin_zone AND schooling (not just sex).

It stays fingerprint-similar to HIMS-real10k (~0.83), so decide_initial_lambda
warm-starts lambda from HIMS-real10k's lambda_at_best instead of zero -> the loop
debiases all three quickly.

Schema map vs HIMS-real10k:
  Gender->sex (mode Male = disadvantaged)
  region_origin->origin_zone (mode Interior = disadvantaged, Coastal = advantaged)
  educ_level->schooling (mode Primary = disadvantaged, Higher = advantaged)
  legal_entry->entry_status (Regular / Irregular)

Output: datasets/MobilitySurvey/mobility10k.csv
CLI:  --name MobilitySurvey --sensitive "sex,origin_zone,schooling" --target entry_status
"""
import os
import numpy as np
import pandas as pd

SEED = 7
N = 10_000
rng = np.random.default_rng(SEED)
NEG = "No"

def sig(x):
    return 1.0 / (1.0 + np.exp(-x))

# 1. Protected attributes — the DISADVANTAGED group is the MODE of each column.
sex = rng.choice(["Male", "Female"], N, p=[0.7504, 0.2496])            # mode Male (disadvantaged)
zone = rng.choice(["Interior", "Coastal", "Inland", "Border"], N,
                  p=[0.45, 0.42, 0.08, 0.05])                          # mode Interior (disadvantaged); others mostly Coastal (advantaged)
school = rng.choice(["Primary", "Higher", "Secondary", "Vocational", "None"], N,
                    p=[0.45, 0.40, 0.10, 0.03, 0.02])                  # mode Primary (disadvantaged); others mostly Higher (advantaged)

MALE_PEN = float(os.environ.get("AADA_TWIN_MALEPEN", "0.80"))
g_code = np.where(sex == "Female", 1.0, -MALE_PEN)                     # Female advantaged
r_code = np.select([zone == "Coastal", zone == "Interior"], [1.0, -1.0], 0.0)
e_code = np.select([school == "Higher", school == "Primary"], [1.0, -1.0], 0.0)

# 2. Latent merit -> qualification columns.
merit = rng.normal(0, 1, N)
age_band = np.clip(np.round(6 + 2.4 * merit + rng.normal(0, 0.8, N)), 1, 15).astype(int)
survey_year = np.clip(np.round(2015.5 + 2.6 * merit + rng.normal(0, 1.1, N)), 2011, 2020).astype(int)
married = np.where(rng.uniform(size=N) < sig(1.4 * merit - 1.2), "Yes", NEG)
took_money = np.where(rng.uniform(size=N) < sig(1.3 * merit + 0.0), 1, 0)
contact = np.where(rng.uniform(size=N) < sig(1.5 * merit + 0.9), "Yes", NEG)
dual_nat = np.where(rng.uniform(size=N) < sig(1.2 * merit - 1.6), "Yes", NEG)
dr_score = merit + rng.normal(0, 0.45, N)
reason = np.select([dr_score > 0.8, dr_score > 0.2, dr_score > -0.4, dr_score > -1.2],
                   [1, 5, 3, 6], default=2)
age = np.clip(np.round(28 + 6 * (age_band - 6) / 3 + rng.normal(0, 4, N)), 18, 65).astype(int)
helped = np.where(rng.uniform(size=N) < sig(0.6 * merit + 0.2), "Yes", NEG)
worked_abroad = np.where(rng.uniform(size=N) < sig(0.7 * merit + 1.0), "Yes", NEG)
employment = np.select([sig(0.8 * merit + 0.7) > rng.uniform(size=N)], ["Employed"],
                       default=np.where(rng.uniform(size=N) < 0.5, "Unemployed", "Inactive"))
has_children = np.where(rng.uniform(size=N) < sig(0.5 * merit + 0.4), "Yes", NEG)
married_now = np.where((married == "Yes") | (rng.uniform(size=N) < 0.25), "Yes", NEG)
working_age = np.where((age >= 20) & (age <= 59), "Yes", NEG)

# 3. PROXY columns — leak the group (removable bias).
dst = 2.5 * r_code + 0.6 * e_code + 1.0 * g_code + rng.normal(0, 0.40, N)
dest_grp = np.select([dst > 0.7, dst > -0.6], ["P1", "P2"], default="P3")
dest_eu = np.where(rng.uniform(size=N) < sig(5.2 * g_code + 0.9 * r_code + 0.6), "Yes", NEG)
itm = 3.8 * e_code + 2.2 * g_code + rng.normal(0, 0.40, N)
broker = np.select([itm > 1.6, itm > 0.2], ["Public", "Private"], default="None")
cc = 2.0 * r_code + 1.0 * g_code + rng.normal(0, 0.6, N)
cur_country = np.select([cc > 1.2, cc > 0.2, cc > -0.8], ["C1", "C2", "C3"], default="C4")

# 4. Label — intercept raised to hold ~0.72 positive rate now the group mean is lower.
W_MERIT = float(os.environ.get("AADA_TWIN_WMERIT", "2.4"))
G_LAB = 0.55
INTERCEPT = float(os.environ.get("AADA_TWIN_INTERCEPT", "3.30"))
GW = float(os.environ.get("AADA_TWIN_GW", "5.3"))
RW = float(os.environ.get("AADA_TWIN_RW", "3.6"))
EW = float(os.environ.get("AADA_TWIN_EW", "4.2"))
noise = rng.normal(0, 0.55, N)
latent = INTERCEPT + W_MERIT * merit + G_LAB * (GW * g_code + RW * r_code + EW * e_code) + noise
entry_status = np.where(latent > 0, "Regular", "Irregular")

# padding to ~70 one-hot columns (high-cardinality near-noise), drawn last.
locality = np.array([f"loc{i}" for i in rng.integers(1, 25, N)])
sector = np.array([f"sec{i}" for i in rng.integers(1, 11, N)])

df = pd.DataFrame({
    "survey_year": survey_year, "age_band": age_band, "age": age, "sex": sex,
    "origin_zone": zone, "schooling": school, "departure_reason": reason,
    "broker_type": broker, "married_at_departure": married, "married_current": married_now,
    "took_money": took_money, "dest_group": dest_grp, "current_country": cur_country,
    "destination_europe": dest_eu, "had_contact": contact, "dual_nationality": dual_nat,
    "received_help": helped, "worked_abroad": worked_abroad, "employment_status": employment,
    "has_children": has_children, "working_age": working_age,
    "locality": locality, "sector": sector,
    "entry_status": entry_status,
})
out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "datasets", "MobilitySurvey", "mobility10k.csv")
os.makedirs(os.path.dirname(out), exist_ok=True)
df.to_csv(out, index=False)
print(f"[gen] wrote {len(df)} rows, {df.shape[1]} cols -> {out}")
print(f"[gen] positive (Regular) rate = {(entry_status=='Regular').mean():.3f}")

# 5. Validate with the SAME binarization the pipeline uses for an unknown dataset:
#    positive_value = the column mode. Confirms all three attrs start biased.
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

SENS = ["sex", "origin_zone", "schooling"]
y = (entry_status == "Regular").astype(int)
itr, ite = train_test_split(np.arange(N), test_size=0.2, random_state=42, stratify=y)
X = StandardScaler().fit_transform(pd.get_dummies(df.drop(columns=SENS + ["entry_status"])).to_numpy(float))
clf = LogisticRegression(max_iter=3000).fit(X[itr], y[itr]); pred = clf.predict(X[ite])
print(f"[val] baseline acc (mode binarization) = {(pred==y[ite]).mean():.3f}")
print(f"[val] one-hot feature width = {X.shape[1]}")
for a in SENS:
    mode = df[a].astype(str).mode().iloc[0]
    b = (df[a].astype(str) == mode).to_numpy()[ite]
    r1 = pred[b].mean() * 100; r0 = pred[~b].mean() * 100
    pr = min(r1, r0) / max(r1, r0) * 100
    print(f"  {a:12s} mode='{mode}'  P-rule={pr:5.1f}   {mode}={r1:.0f}%  others={r0:.0f}%")
