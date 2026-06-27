# -*- coding: utf-8 -*-
"""Faithful replica of the pipeline's PRETRAIN baseline for the MobilitySurvey twin:
2x256 MLP, dropout 0.2, Adam 1e-3, BCEWithLogits, 10 epochs, batch 32, mode-binarized
sensitive attrs, 80/20 stratified split (seed 42). Reports baseline acc + P-rules so I
can tune the generator against what the pipeline ACTUALLY produces (not a converged LR)."""
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
SEED=42; np.random.seed(SEED); torch.manual_seed(SEED)
HERE="C:/Users/jamil/Desktop/adversarial_learning2/adversarial_fairness"
df=pd.read_csv(f"{HERE}/datasets/MobilitySurvey/mobility10k.csv")
SENS=["sex","origin_zone","schooling"]
y=(df.entry_status=="Regular").astype(int).values
X=StandardScaler().fit_transform(pd.get_dummies(df.drop(columns=SENS+["entry_status"])).to_numpy(float)).astype(np.float32)
tr,te=train_test_split(np.arange(len(y)),test_size=0.2,random_state=SEED,stratify=y)
m=nn.Sequential(nn.Linear(X.shape[1],256),nn.ReLU(),nn.Dropout(0.2),
                nn.Linear(256,256),nn.ReLU(),nn.Dropout(0.2),nn.Linear(256,1))
opt=torch.optim.Adam(m.parameters(),1e-3); lf=nn.BCEWithLogitsLoss()
Xt=torch.tensor(X[tr]); yt=torch.tensor(y[tr],dtype=torch.float32).unsqueeze(1)
for ep in range(25):
    p=torch.randperm(len(Xt))
    for i in range(0,len(Xt),32):
        idx=p[i:i+32]; opt.zero_grad(); lf(m(Xt[idx]),yt[idx]).backward(); opt.step()
m.eval()
with torch.no_grad(): prob=torch.sigmoid(m(torch.tensor(X[te]))).numpy().ravel()
pred=(prob>=.5).astype(int); acc=(pred==y[te]).mean()
print(f"[pretrain-check] baseline acc = {acc*100:.1f}%")
for a in SENS:
    mode=df[a].astype(str).mode().iloc[0]
    b=(df[a].astype(str)==mode).to_numpy()[te]
    r1=pred[b].mean()*100; r0=pred[~b].mean()*100
    pr=min(r1,r0)/max(r1,r0)*100
    print(f"  {a:12s} P-rule={pr:5.1f}   {mode}={r1:.0f}%  others={r0:.0f}%")
