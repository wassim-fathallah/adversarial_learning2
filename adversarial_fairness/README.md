# Adversarial Fairness System

Two neural networks compete: a **Classifier** predicts a target, an **Adversary** tries to extract sensitive attributes (race, sex...) from its representations. The classifier learns to be accurate and fair simultaneously. A **LLaMA 3.1 LLM** adaptively controls the fairness pressure at each training step.

## Run

```bash
python main.py --dataset german --iterations 20
python main.py --dataset compas --iterations 25
```

Supported: Adult, German Credit, COMPAS, UTKFace, KDD Census
