#!/usr/bin/env python3
"""
Monitors Kaggle Kernel execution, fetches outputs (figures, metrics, logs, model weights),
and embeds real outputs and base64 figures into g2p_implementation_and_finetune.ipynb.
"""

import os
import sys
import time
import json
import base64
import shutil
import requests
from kaggle.api.kaggle_api_extended import KaggleApi, ApiListKernelSessionOutputRequest

BASE_DIR = "/home/soroush/.gemini/antigravity/scratch/g2p-finetune"
OUT_DIR = os.path.join(BASE_DIR, "kaggle_output")
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "figures"), exist_ok=True)

api = KaggleApi()
api.authenticate()

user_name = "soroushsoleimani"
kernel_slug = "g2p-implementation-and-mini-finetune-byt5"
kernel_id = f"{user_name}/{kernel_slug}"

print(f"[*] Monitoring Kaggle kernel: {kernel_id}")
start_time = time.time()
last_status = None

while True:
    try:
        status_resp = api.kernels_status(kernel_id)
        status = str(getattr(status_resp, "status", status_resp))
    except Exception as e:
        status = f"FETCH_EXCEPTION: {e}"

    if status != last_status:
        elapsed = time.time() - start_time
        print(f"[{time.strftime('%H:%M:%S')}] (+{elapsed:.0f}s) Status: {status}")
        last_status = status

    if "COMPLETE" in status.upper():
        print(f"[+] Kernel execution completed in {time.time() - start_time:.1f}s!")
        break
    elif any(err in status.upper() for err in ["ERROR", "CANCEL", "FAIL"]):
        print(f"[-] Kernel execution failed with status: {status}")
        break

    time.sleep(15)

print("[*] Fetching kernel outputs from Kaggle API...")
saved_files = []
kernel_log_text = ""

with api.build_kaggle_client() as kaggle:
    token = None
    page = 0
    while True:
        page += 1
        req = ApiListKernelSessionOutputRequest()
        req.user_name = user_name
        req.kernel_slug = kernel_slug
        req.page_size = 200
        if token:
            req.page_token = token
        resp = kaggle.kernels.kernels_api_client.list_kernel_session_output(req)

        if page == 1 and resp.log:
            kernel_log_text = resp.log
            log_path = os.path.join(OUT_DIR, "kernel.log")
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(resp.log)
            print(f"[+] Saved kernel.log ({len(resp.log)} chars)")

        files = resp.files or []
        for item in files:
            if item.file_name.startswith("CharsiuG2P/"):
                continue
            dest = os.path.join(OUT_DIR, item.file_name)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            r = requests.get(item.url, timeout=180)
            r.raise_for_status()
            with open(dest, "wb") as f:
                f.write(r.content)
            saved_files.append(item.file_name)
            print(f"  -> Downloaded: {item.file_name} ({len(r.content):,} bytes)")

            # Also sync figures to local figures dir
            if item.file_name.startswith("figures/"):
                fig_name = os.path.basename(item.file_name)
                local_fig = os.path.join(FIG_DIR, fig_name)
                with open(local_fig, "wb") as f_out:
                    f_out.write(r.content)

            # Sync evaluation_metrics.json
            if item.file_name == "evaluation_metrics.json":
                local_metrics = os.path.join(BASE_DIR, "evaluation_metrics.json")
                with open(local_metrics, "wb") as f_out:
                    f_out.write(r.content)

        token = resp.next_page_token
        if not token:
            break

print(f"[+] Fetched {len(saved_files)} files to {OUT_DIR}")

# Copy model checkpoint if present
out_model = os.path.join(OUT_DIR, "finetuned_byt5_g2p_model")
local_model = os.path.join(BASE_DIR, "finetuned_byt5_g2p_model")
if os.path.exists(out_model):
    os.makedirs(local_model, exist_ok=True)
    for f in os.listdir(out_model):
        s = os.path.join(out_model, f)
        d = os.path.join(local_model, f)
        if os.path.isfile(s):
            shutil.copy2(s, d)
    print(f"[+] Synced fine-tuned model weights to {local_model}")

print("[*] All outputs successfully retrieved and synchronized.")
