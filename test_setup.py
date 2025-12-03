#!/usr/bin/env python3
"""
Quick test to verify the evaluation pipeline is correctly set up.
"""

import sys
from pathlib import Path

print("="*80)
print("MTEB Italian Retrieval Evaluation - Setup Verification")
print("="*80)

# Check Python version
print(f"\n  Python version: {sys.version.split()[0]}")

# Check required packages
packages_to_check = ["torch", "mteb", "pandas", "tqdm"]
missing_packages = []

for package in packages_to_check:
    try:
        __import__(package)
        print(f"  {package:20s} - installed")
    except ImportError:
        print(f"{package:20s} - NOT FOUND")
        missing_packages.append(package)

if missing_packages:
    print(f"\nMissing packages: {', '.join(missing_packages)}")
    print("Run: pip install " + " ".join(missing_packages) + " --break-system-packages")
else:
    print("\n All required packages are installed!")

# Check GPU
try:
    import torch
    if torch.cuda.is_available():
        print(f"\n  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("\n No GPU detected - evaluation will be slower")
except:
    print("\n   Cannot detect GPU")

# Check scripts exist
scripts = ["run_ita_eval.py", "parse_results_to_csv.py", "setup.sh"]
print("\n" + "="*80)
print("Script Files:")
print("="*80)

for script in scripts:
    if Path(script).exists():
        print(f"  {script}")
    else:
        print(f" {script} - NOT FOUND")

print("\n" + "="*80)
print("Ready to start evaluation!")
print("="*80)
print("\nNext steps:")
print("  1. Run: python3 run_ita_eval.py")
print("  2. Results will be saved to: results/italian_retrieval_results.csv")
print("  3. Logs will be saved to: mteb_ita_evaluation.log")
print("="*80)
