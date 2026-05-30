"""
LangChain tool: detect_proxies

Detects which INPUT FEATURES act as proxies for sensitive attributes
by analyzing their correlation with the classifier's learned embedding
as seen through the adversary's lens.

Algorithm:
  1. Get embeddings E (N, 32) for all training samples
  2. Get adversary's logits A (N, n_sensitive) from those embeddings
  3. For each input feature j:
       proxy_score[j, s] = |Pearson corr(X[:,j], A[:,s])|
  4. Report top-K features per sensitive attribute

Why this works:
  Even if 'race' is removed from the dataset, features like 'zip_code'
  or 'education' may be correlated with race. The classifier's embedding
  will encode this correlation, the adversary will exploit it, and the
  correlation between original features and adversary output reveals which
  features are carrying the signal.
"""

import json
import numpy as np
import torch
from scipy.stats import pearsonr
from langchain.tools import tool
from langchain_ollama import OllamaLLM

from state import state

llm = OllamaLLM(model="llama3.1", temperature=0.15)


@tool
def detect_proxies(top_k: int = 5) -> str:
    """
    Analyzes the classifier's embedding space to find which features act
    as proxies for each sensitive attribute. Runs after pretraining.
    Returns a JSON report with top proxy features per attribute and
    an LLM interpretation of what each proxy means semantically.

    Args:
        top_k: Number of top proxy features to report per sensitive attribute.
    """
    if state.classifier is None:
        return "ERROR: Run pretrain first."
    if state.X_train_raw is None:
        return "ERROR: Dataset not loaded."

    classifier = state.classifier
    adversary  = state.adversary
    device     = state.device

    classifier.eval()
    adversary.eval()

    # ── Get embeddings and adversary outputs ──────────────────────────────────
    with torch.no_grad():
        X_t = state.X_train
        embeddings = classifier.get_embedding(X_t)          # (N, 32)
        adv_logits = adversary(embeddings)                  # (N, n_sensitive)

    emb_np = embeddings.cpu().numpy()
    adv_np = adv_logits.cpu().numpy()
    X_raw  = state.X_train_raw                              # (N, n_features)
    feat_names = state.feature_names

    # ── Compute proxy scores ──────────────────────────────────────────────────
    # proxy_score[j, s] = |corr(X[:,j], adv_logits[:,s])|
    n_features  = X_raw.shape[1]
    n_sensitive = adv_np.shape[1]
    proxy_scores = np.zeros((n_features, n_sensitive))

    for j in range(n_features):
        for s in range(n_sensitive):
            try:
                r, _ = pearsonr(X_raw[:, j], adv_np[:, s])
                proxy_scores[j, s] = abs(r) if not np.isnan(r) else 0.0
            except Exception:
                proxy_scores[j, s] = 0.0

    # ── Build report ──────────────────────────────────────────────────────────
    report = {}
    for s, attr in enumerate(state.sensitive_attrs):
        scores = proxy_scores[:, s]
        top_idx = np.argsort(scores)[::-1][:top_k]
        top_features = [
            {"feature": feat_names[j], "proxy_score": round(float(scores[j]), 4)}
            for j in top_idx
        ]
        report[attr] = top_features

    # ── LLM interprets the proxies ────────────────────────────────────────────
    proxy_str = json.dumps(report, indent=2)
    prompt = f"""You are a fairness auditor. The following features were detected as proxies
for sensitive attributes in a machine learning model.
A high proxy score means the feature's information is strongly used by the model
to implicitly predict that sensitive attribute, even though the attribute was removed.

Dataset: {state.dataset_name}
Proxy analysis:
{proxy_str}

In 3-4 sentences, explain what each proxy feature represents and why it correlates
with the sensitive attribute. Focus on real-world implications.
"""
    try:
        interpretation = llm.invoke(prompt)
        interpretation_text = interpretation.strip()
    except Exception:
        interpretation_text = "LLM unavailable; proxy features are reported without semantic interpretation."

    full_report = {
        "proxy_features": report,
        "llm_interpretation": interpretation_text,
    }
    state.proxy_report = full_report

    print("\n[proxies] === Proxy Detection Report ===")
    for attr, feats in report.items():
        print(f"  {attr}: {[f['feature'] for f in feats]}")
    print(f"  LLM: {interpretation_text[:200]}...")

    return json.dumps(full_report, indent=2)
