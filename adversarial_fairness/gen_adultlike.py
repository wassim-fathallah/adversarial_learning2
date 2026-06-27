# -*- coding: utf-8 -*-
"""
Generate a synthetic dataset structurally CLOSE TO ADULT, so the fingerprint
warm-start fires when it is run as an *unknown* dataset: lambda starts NOT from 0
(it is warm-started from a similar past Adult run) and the run converges in fewer
iterations.

Target Adult fingerprint (from long_term_memory):
  n_sensitive = 2
  sensitive_imbalance = [~0.145, ~0.33]   (one ~85/15 group, one ~67/33 group)
  target_balance = ~0.24                  (24% positive)
  n_samples_log10 ~ 4.42, n_features_log10 ~ 1.97

Column names avoid every known-dataset trigger (income/adult/sex/race/ethnicity+
gender/credit/bank/compas/kdd/census-income/acs/hims/legal_entry/coastal_origin),
so _infer_dataset_key returns "" -> the dataset is treated as UNKNOWN -> warm-start.

Output: datasets/earnings_synth/earnings_synth.csv
Run with (unknown name + forced 2 sensitive attrs):
  python main.py --dataset datasets/earnings_synth/earnings_synth.csv \
                 --name earnings_synth --sensitive origin_group,background_group
"""
import os
import numpy as np
import pandas as pd

SEED = 42
N = 30000                     # train ~24000 -> n_samples_log10 ~4.38 (Adult 4.42)
rng = np.random.default_rng(SEED)

def sig(x): return 1.0 / (1.0 + np.exp(-x))

# --- 2 sensitive groups matched to Adult's imbalances -----------------------
origin_group     = rng.choice(["A", "B"], N, p=[0.855, 0.145])   # like race White/other -> imb 0.145
background_group = rng.choice(["X", "Y"], N, p=[0.67, 0.33])      # like sex Male/Female  -> imb 0.33
o_dis = (origin_group == "B").astype(float)        # disadvantaged origin
b_dis = (background_group == "Y").astype(float)     # disadvantaged background

# --- latent merit + qualification (merit) features --------------------------
merit = rng.normal(0, 1, N)
education = rng.choice([f"ed{i}" for i in range(12)], N,
                       p=np.r_[np.linspace(0.02, 0.16, 12)] / np.linspace(0.02, 0.16, 12).sum())
# tie education ordinally to merit
ed_rank = np.clip(np.round(6 + 2.2 * merit + rng.normal(0, 1.1, N)), 0, 11).astype(int)
education = np.array([f"ed{i}" for i in ed_rank])
occupation = np.array([f"occ{i}" for i in np.clip(np.round(5 + 2.0*merit + rng.normal(0,1.4,N)),0,11).astype(int)])
work_class = np.array([f"wc{i}" for i in np.clip(np.round(3 + 1.4*merit + rng.normal(0,1.0,N)),0,6).astype(int)])
industry  = np.array([f"ind{i}" for i in np.clip(np.round(6 + 1.8*merit + rng.normal(0,1.6,N)),0,13).astype(int)])
hours_per_week = np.clip(np.round(40 + 8*merit + rng.normal(0,7,N)), 1, 99).astype(int)
age = np.clip(np.round(38 + 7*merit + rng.normal(0,9,N)), 17, 90).astype(int)
capital_gain = np.clip(np.round(np.expm1(1.0 + 0.7*merit + rng.normal(0,0.5,N))*100), 0, None).astype(int)

# --- proxy features: leak the GROUPS (removable bias) -----------------------
# relationship / marital leak background_group; native_region leaks origin_group
rel_score = 1.8*b_dis + rng.normal(0, 0.5, N)
relationship = np.select([rel_score > 1.3, rel_score > 0.6, rel_score > -0.1],
                         ["rel_husb", "rel_wife", "rel_unmar"], default="rel_other")
marital = np.where(rng.uniform(size=N) < sig(2.2*(0.5-b_dis)), "married", "single")
reg_score = 2.0*o_dis + 0.6*b_dis + rng.normal(0, 0.5, N)
native_region = np.select([reg_score > 1.4, reg_score > 0.6],
                          ["reg_far", "reg_mid"], default="reg_near")

# --- target: 24% positive, disadvantaged groups earn less (removable) -------
# intercept tuned for ~0.24 positive; group effect modest so proxies+adult-like
# lambda (~4-5,3) suffices to remove it.
latent = (-0.82 + 1.5*merit - 0.9*o_dis - 0.8*b_dis + rng.normal(0, 0.6, N))
high_earner = np.where(latent > 0, "Yes", "No")

df = pd.DataFrame({
    "age": age, "work_class": work_class, "education": education,
    "occupation": occupation, "industry": industry, "hours_per_week": hours_per_week,
    "capital_gain": capital_gain, "marital": marital, "relationship": relationship,
    "native_region": native_region, "origin_group": origin_group,
    "background_group": background_group, "high_earner": high_earner,
})
out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "datasets", "earnings_synth", "earnings_synth.csv")
os.makedirs(os.path.dirname(out), exist_ok=True)
df.to_csv(out, index=False)
print(f"[gen] wrote {len(df)} rows, {df.shape[1]} cols -> {out}")
print(f"[gen] positive rate = {(high_earner=='Yes').mean():.3f}")
print(f"[gen] origin B (minority) = {o_dis.mean():.3f}   background Y = {b_dis.mean():.3f}")

# --- show the resulting fingerprint vs Adult --------------------------------
from sklearn.model_selection import train_test_split
y = (high_earner == "Yes").astype(int)
Xoh = pd.get_dummies(df.drop(columns=["origin_group", "background_group", "high_earner"]))
ntr = int(0.8 * N)
imb = sorted([round(min((origin_group=='A').mean(), (origin_group=='B').mean()),4),
              round(min((background_group=='X').mean(),(background_group=='Y').mean()),4)])
fp = {"n_sensitive": 2, "sensitive_imbalance": imb, "target_balance": round(y.mean(),4),
      "n_samples_log10": round(float(np.log10(ntr)),3),
      "n_features_log10": round(float(np.log10(Xoh.shape[1])),3)}
adult = {"n_sensitive":2, "sensitive_imbalance":[0.145,0.33], "target_balance":0.2408,
         "n_samples_log10":4.416, "n_features_log10":1.968}
print("\n[fingerprint] this dataset:", fp)
print("[fingerprint] adult        :", adult)
# crude similarity (same formula as the codebase)
diffs = [abs(a-b)/0.5 for a,b in zip(fp["sensitive_imbalance"], adult["sensitive_imbalance"])]
diffs += [abs(fp["target_balance"]-adult["target_balance"]),
          min(abs(fp["n_samples_log10"]-adult["n_samples_log10"])/3,1),
          min(abs(fp["n_features_log10"]-adult["n_features_log10"])/2,1)]
print(f"[fingerprint] similarity to Adult = {1-np.mean(diffs):.3f}  (warm-start needs >= 0.75)")
