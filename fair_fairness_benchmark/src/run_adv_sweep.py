"""
Lambda sweep for CNN adversarial debiasing on UTKFace — FFB-style grid search.

Runs ffb_image_adv.py for every (sensitive:target, lam, seed) combination, each as its
own process (how FFB runs a grid). Every run writes an app.py-compatible result file
into ../results/, showing up in the FFB interface under method "adv".

UTKFace has only 3 demographic columns (age/gender/race) and one must be the target,
so each experiment is a sensitive:target pair:
    Gender:Age   -> predict age>30, debias for gender
    Race:Age     -> predict age>30, debias for race
    Age:Gender   -> predict gender, debias for age   (age can't be sensitive AND target)

Examples:
    python run_adv_sweep.py                                   # 3 pairs, FFB lambda grid
    python run_adv_sweep.py --pairs Gender:Age Race:Age       # just the two age-target runs
    python run_adv_sweep.py --lambdas 0 0.5 1 2 --batch_size 256
"""

import argparse
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser()
    # FFB's UTKFace PR grid: 15 values 0.05..1.0
    p.add_argument("--lambdas", type=float, nargs="+",
                   default=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                   help="grid of adversarial weights (default = FFB's UTKFace PR grid)")
    p.add_argument("--pairs", type=str, nargs="+",
                   default=["Gender:Age", "Race:Age", "Age:Gender"],
                   help="sensitive:target pairs to sweep")
    p.add_argument("--seeds", type=int, nargs="+", default=[1314])
    p.add_argument("--pretrain_steps", type=int, default=250)
    p.add_argument("--num_training_steps", type=int, default=250)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--eval_batch_size", type=int, default=128,
                   help="keep small on low-VRAM GPUs (GTX 1070 has a ~3GB usable ceiling)")
    p.add_argument("--architecture", type=str, default="resnet18")
    p.add_argument("--save_dir", type=str, default="../results")
    args = p.parse_args()

    pairs = [tuple(x.split(":")) for x in args.pairs]   # [(sensitive, target), ...]
    combos = [(s, t, lam, seed) for (s, t) in pairs for lam in args.lambdas for seed in args.seeds]
    print(f"=== Sweep: {len(combos)} runs "
          f"({len(pairs)} pairs x {len(args.lambdas)} lambdas x {len(args.seeds)} seeds) ===")
    for (s, t) in pairs:
        print(f"  pair  sensitive={s:6s} target={t}")

    t0 = time.time()
    failures = []
    for i, (sens, tgt, lam, seed) in enumerate(combos, 1):
        print(f"\n{'='*70}\n[{i}/{len(combos)}] sensitive={sens} target={tgt} lam={lam} seed={seed}\n{'='*70}")
        cmd = [
            sys.executable, "ffb_image_adv.py",
            "--sensitive_attr", sens,
            "--target_attr", tgt,
            "--lam", str(lam),
            "--seed", str(seed),
            "--pretrain_steps", str(args.pretrain_steps),
            "--num_training_steps", str(args.num_training_steps),
            "--batch_size", str(args.batch_size),
            "--eval_batch_size", str(args.eval_batch_size),
            "--architecture", args.architecture,
            "--save_dir", args.save_dir,
        ]
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            failures.append((sens, tgt, lam, seed, rc))
            print(f"[warn] run failed (rc={rc}): sensitive={sens} target={tgt} lam={lam} seed={seed}")
        done = i
        elapsed = time.time() - t0
        eta = elapsed / done * (len(combos) - done)
        print(f"[progress] {done}/{len(combos)} done | elapsed {elapsed/60:.1f} min | ETA {eta/60:.1f} min")

    dt = time.time() - t0
    print(f"\n=== Sweep done: {len(combos)-len(failures)}/{len(combos)} succeeded "
          f"in {dt/60:.1f} min ===")
    if failures:
        print("Failed runs:", failures)
    print(f"Results in {args.save_dir}/  -> open the FFB app.py to view them.")


if __name__ == "__main__":
    main()
