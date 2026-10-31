#!/usr/bin/env python3
"""
Pushes the updated KaamelDict fine-tuning notebook to Kaggle GPU kernel.
"""

import os
import sys
import json
import subprocess

BASE_DIR = "/home/soroush/.gemini/antigravity/scratch/g2p-finetune"
KERNEL_DIR = os.path.join(BASE_DIR, "kaggle_kernel")
METADATA_PATH = os.path.join(KERNEL_DIR, "kernel-metadata.json")

def main():
    if not os.path.exists(METADATA_PATH):
        print(f"[-] Error: {METADATA_PATH} not found.")
        sys.exit(1)

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    kernel_id = meta.get("id", "soroushsoleimani/g2p-implementation-and-mini-finetune-byt5")
    print(f"[*] Target Kaggle Kernel: {kernel_id}")
    print(f"[*] GPU Enabled: {meta.get('enable_gpu')}")
    print(f"[*] Internet Enabled: {meta.get('enable_internet')}")
    print(f"[*] Code File: {meta.get('code_file')}")

    kaggle_bin = os.path.join(BASE_DIR, ".venv", "bin", "kaggle")
    cmd = [kaggle_bin, "kernels", "push", "-p", KERNEL_DIR]
    print(f"[*] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    if result.returncode == 0:
        print(f"\n[+] Kernel successfully pushed to Kaggle!")
        print(f"    URL: https://www.kaggle.com/code/{kernel_id}")
    else:
        print("\n[-] Push failed.")
        sys.exit(result.returncode)

if __name__ == "__main__":
    main()
