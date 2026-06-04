# Adversarial Fairness System

Two neural networks compete: a **Classifier** predicts a target, an **Adversary** tries to recover sensitive attributes (race, sex...) from the classifier's output. The classifier learns to be accurate and fair simultaneously. A **LLaMA 3.1** orchestrator identifies the sensitive attributes from the dataset schema, and a momentum-based rule adaptively controls the fairness pressure (lambda) at each iteration.

## Run

```bash
python main.py --dataset german --iterations 20
python main.py --dataset compas --iterations 25
```

Supported: Adult, German Credit, COMPAS, Bank Marketing, KDD Census, ACS, UTKFace, Migration (HIMS-Tunisia)

Selection is fairness-first: training stops once the P-rule target (`--threshold`, default 80) is met, and the highest-accuracy iteration among the fair ones is returned. Accuracy is not capped.
