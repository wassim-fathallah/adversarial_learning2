"""
Hand-off orchestrator:
  1. Wait until COMPAS reaches 10 runs in long_term_memory.json (CPU sweep finishing).
  2. Back up memory, then stop the CPU sweep (system python, NOT .venv_gpu).
  3. Launch the remaining datasets on GPU (german, kdd, acs, utkface, HIMS-Tunisia).

Run with the GPU venv so step 3 uses CUDA and this script is not self-killed
(launch from the adversarial_fairness/ directory):
    ..\.venv_gpu\Scripts\python.exe seeds\wait_then_gpu.py
"""
import json, os, sys, time, shutil, subprocess, datetime

HERE   = os.path.dirname(os.path.abspath(__file__))   # adversarial_fairness/seeds/
PARENT = os.path.dirname(HERE)                         # adversarial_fairness/
MEM  = os.path.join(PARENT, "long_term_memory.json")
SEEDS = ["14159","26535","89793","23846","26433","83279","50288","41971","69399","37510"]
REMAINING = ["german","kdd","acs","utkface","HIMS-Tunisia"]


def count(ds):
    try:
        d = json.load(open(MEM, encoding="utf-8-sig"))
    except Exception:
        return -1
    return sum(len(v) for k, v in d.items() if k.split("|")[0].lower() == ds)


# 1. wait for compas == 10 (safety timeout 3h)
print("[wait] waiting for compas to reach 10/10 ...", flush=True)
t0 = time.time()
while count("compas") < 10:
    if time.time() - t0 > 3 * 3600:
        print("[wait] timeout; proceeding anyway", flush=True)
        break
    time.sleep(30)
print(f"[wait] compas = {count('compas')}/10  -> handing off to GPU", flush=True)

# 2. back up memory, stop the CPU sweep (any python NOT in .venv_gpu)
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
shutil.copyfile(MEM, os.path.join(PARENT, f"long_term_memory.backup_{ts}.json"))
print(f"[backup] long_term_memory.backup_{ts}.json", flush=True)
subprocess.run(["powershell", "-NoProfile", "-Command",
    "Get-Process python -ErrorAction SilentlyContinue | "
    "Where-Object { $_.Path -notlike '*venv_gpu*' } | Stop-Process -Force"])
time.sleep(5)

# 3. GPU sweep for the remaining datasets
cmd = [sys.executable, os.path.join(HERE, "run_multiple_seeds.py"),
       "--dataset", *REMAINING, "--seeds", *SEEDS]
print("[gpu] launching:", " ".join(cmd), flush=True)
subprocess.run(cmd, cwd=PARENT)
print("[gpu] all remaining datasets done", flush=True)
