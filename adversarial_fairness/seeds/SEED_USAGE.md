# Using Seeds in Adversarial Fairness

## Overview
You can now run the adversarial fairness pipeline with different random seeds for reproducibility and variance analysis.

## Single Seed (default)

```bash
# Default seed is 42
python main.py --dataset adult

# Custom seed
python main.py --dataset adult --seed 123
python main.py --dataset adult --seed 456
```

## Multiple Seeds

### Method 1: Using the helper script (recommended)

> The seed-execution scripts live in `adversarial_fairness/seeds/`. Run them
> from the `adversarial_fairness/` directory (paths resolve to the package root).

```bash
# Run with default seeds: 42, 123, 456, 789, 999
python seeds/run_multiple_seeds.py --dataset adult

# Custom seed list
python seeds/run_multiple_seeds.py --dataset adult --seeds 1 2 3 4 5

# Seed range (inclusive)
python seeds/run_multiple_seeds.py --dataset adult --seeds-range 100 105  # runs 100,101,102,103,104,105

# With other parameters
python seeds/run_multiple_seeds.py --dataset adult --seeds 42 123 --epochs 100 --iterations 30
```

### Method 2: Manual loop (Windows PowerShell)

```powershell
foreach ($seed in 42, 123, 456) {
    python main.py --dataset adult --seed $seed
}
```

### Method 3: Manual loop (Bash/Linux)

```bash
for seed in 42 123 456; do
    python main.py --dataset adult --seed $seed
done
```

## What Gets Seeded?

- ✅ PyTorch model initialization (`torch.manual_seed`)
- ✅ GPU operations (`torch.cuda.manual_seed`)
- ✅ NumPy operations (`np.random.seed`)
- ✅ Python random module (`random.seed`)
- ✅ DataLoader shuffling (via `generator` parameter)

## Output Organization

Each run with a different seed creates its own:
- Training curves plot: `{dataset}_training_curves.png`
- Long-term memory entry with seed included
- Console output showing seed value

## Comparing Across Seeds

After running multiple seeds, you can:
1. Compare final metrics across seeds in long-term memory
2. Plot training curves to see variability
3. Analyze whether results are sensitive to initialization

## Example: Full Workflow

```bash
# Run adult dataset with 3 seeds
python seeds/run_multiple_seeds.py --dataset adult --seeds 42 123 456 --epochs 50 --iterations 25

# Check results in:
# - adversarial_fairness/long_term_memory.json (stores all runs)
# - adversarial_fairness/adult_training_curves.png (from last run)
```

## Notes

- The seed is printed in the output banner for each run
- Seed value is also stored in long-term memory for reference
- Default seed remains 42 for backward compatibility
- State is fully reset between runs (no carryover)
