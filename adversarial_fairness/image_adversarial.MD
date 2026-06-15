# How Our Adversarial Debiasing Method Runs on UTKFace

## TL;DR
Our method runs Zhang (2018) adversarial debiasing on UTKFace by **treating each
image as 2304 flat pixel-features fed to an MLP — not as an image.** There is no CNN
and no image-specific path; UTKFace is just "a tabular dataset with 2304 columns."

---

## The key fact: our pipeline has no image path at all — everything is an MLP

Our classifier (`models/classifier.py`) is a plain fully-connected MLP:

```python
nn.Linear(n_features, n_hidden), nn.ReLU(), nn.Dropout(0.2),
nn.Linear(n_hidden, n_hidden),  nn.ReLU(), nn.Dropout(0.2),
nn.Linear(n_hidden, n_hidden),  nn.ReLU(), nn.Dropout(0.2),
nn.Linear(n_hidden, 1),
```

`nn.Linear` only — **no `Conv2d` anywhere.** It doesn't know or care whether the input
is income/age columns or pixels. It just sees a row of `n_features` numbers.

## What happens to UTKFace on the way in

In `tools/data_tools.py`, each 48×48 face image is flattened into individual pixel
columns:

```python
# UTKFace: expand space-separated pixel string into pixel_0…pixel_N cols
pixel_matrix = np.vstack(...)          # (N, 2304)
pixel_cols = [f"pixel_{i}" for i ...]  # pixel_0 … pixel_2303
```

After this, UTKFace is structurally identical to `adult` (~100 features) or `german` —
just a wider table. The only UTKFace-specific tweak is widening the hidden layer from
32 to 288, because 2304 inputs need more capacity (`tools/training_tools.py`):

```python
n_hidden = min(max(64, n_features // 8), 512) if n_features > 500 else 32
# 2304 // 8 = 288
```

## Then it runs exactly Zhang 2018 on the flat pixels

- **Classifier:** MLP `2304 → 288 → 288 → 288 → 1`, outputs `Ŷ = σ(·)`
- **Adversary** (`models/adversary.py`): takes the scalar `Ŷ` and predicts the
  sensitive attribute. This is literally Figure 1 of Zhang's paper.
- **Minimax:** classifier minimizes `L_task − λ·L_adv`; adversary maximizes `L_adv`.

So **"how does it work on UTKFace?"** → it treats the image as a bag of 2304 pixel
numbers and applies the same MLP-based Zhang adversarial debiasing it uses for every
other dataset. No convolution, no spatial structure — to our code, a face is just a
wide row of numbers.

---

## Why ours works where FFB's adversarial method "can't"

It is **not** that our method is more powerful. It is an architecture difference.

| Data type            | FFB                                   | Our pipeline                                  |
|----------------------|---------------------------------------|-----------------------------------------------|
| Tabular data         | MLP path → **has adversarial**        | one MLP path for everything                   |
| Image data (UTKFace) | separate CNN/ResNet path → **no adversarial** | *(no separate path — images get flattened into the same MLP)* |

FFB split images and tables into **two separate codebases** and only built the
adversarial method for the tabular one. Its UTKFace support lives only in the image
scripts (`erm, pr, hsic, diffdp, diffeodd, diffeopp`) — none of which is adversarial.
Its adversarial scripts (`ffb_tabular_adv.py`, `ffb_tabular_adv_gr.py`) hardcode the
dataset list `adult, kdd, acs, german, compas, bank_marketing, migration` — UTKFace is
not in it. The two sets never overlap, so FFB has **no** adversarial-debiasing entry
point for UTKFace.

Our pipeline has **one** dataset-agnostic MLP path, so UTKFace just falls into it as a
2304-column table. We sidestepped the entire problem by never having an image path to
begin with.

---

## Honest caveat (important for the thesis)

Flattening pixels into an MLP **throws away spatial structure** — neighbouring pixels
lose their 2D relationship — so it is a *weaker image model* than a CNN (lower accuracy
ceiling; this is why UTKFace accuracy is ~75% in our runs, not 90%+). But it is a
completely valid and standard choice: it is the **exact same modeling decision FFB's
tabular adversarial path makes**, just applied to pixel features.

A CNN-based adversarial variant is possible but is **custom code** (not shipped by FFB).
A prototype of that approach lives in `fair_fairness_benchmark/src/ffb_image_adv.py`
(a gradient-reversal adversary on a small CNN), written separately to bridge that gap.
