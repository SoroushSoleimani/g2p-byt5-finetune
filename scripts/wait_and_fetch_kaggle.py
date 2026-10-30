#!/usr/bin/env python3
import time
import os
import requests
from kaggle.api.kaggle_api_extended import KaggleApi, ApiListKernelSessionOutputRequest

api = KaggleApi()
api.authenticate()

user_name = "soroushsoleimani"
kernel_slug = "g2p-implementation-and-mini-finetune-byt5"
out_dir = "/home/soroush/.gemini/antigravity/scratch/g2p-finetune/kaggle_v6_output"
os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)

print(f"[*] Waiting for Kaggle kernel '{user_name}/{kernel_slug}' to complete...")
start = time.time()
last_status = None

while True:
    status_resp = api.kernels_status(f"{user_name}/{kernel_slug}")
    status = str(getattr(status_resp, "status", status_resp))
    if status != last_status:
        print(f"[{time.strftime('%H:%M:%S')}] Status changed to: {status}")
        last_status = status
    if "COMPLETE" in status.upper():
        print(f"[+] Kernel execution completed successfully in {time.time() - start:.1f}s!")
        break
    elif any(err in status.upper() for err in ["ERROR", "CANCEL"]):
        print(f"[-] Kernel ended with status: {status}")
        break
    time.sleep(10)

print("[*] Fetching output files...")
with api.build_kaggle_client() as kaggle:
    token = None
    page = 0
    saved_files = []
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
            with open(os.path.join(out_dir, "kernel.log"), "w", encoding="utf-8") as f:
                f.write(resp.log)
            print(f"[+] Saved kernel.log ({len(resp.log)} chars)")
        files = resp.files or []
        for item in files:
            if not item.file_name.startswith("CharsiuG2P/"):
                dest = os.path.join(out_dir, item.file_name)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                r = requests.get(item.url, timeout=120)
                r.raise_for_status()
                with open(dest, "wb") as f:
                    f.write(r.content)
                saved_files.append(item.file_name)
                print(f"  -> Saved {item.file_name} ({len(r.content)} bytes)")
        token = resp.next_page_token
        if not token:
            break

print(f"[+] Successfully fetched {len(saved_files)} files to '{out_dir}'.")
