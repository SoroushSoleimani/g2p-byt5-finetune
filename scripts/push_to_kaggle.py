#!/usr/bin/env python3
import json
import os
import sys
import subprocess

def setup_kaggle_credentials(username=None, key=None, token=None):
    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)
    
    if username and key:
        creds_path = os.path.join(kaggle_dir, "kaggle.json")
        with open(creds_path, "w", encoding="utf-8") as f:
            json.dump({"username": username.strip(), "key": key.strip()}, f)
        os.chmod(creds_path, 0o600)
        print(f"[+] Kaggle credentials saved to {creds_path}")
    elif token:
        token_path = os.path.join(kaggle_dir, "access_token")
        with open(token_path, "w", encoding="utf-8") as f:
            f.write(token.strip())
        os.chmod(token_path, 0o600)
        print(f"[+] Kaggle access token saved to {token_path}")
    else:
        print("[-] Error: Either username & key or access_token must be provided.")
        sys.exit(1)

def push_kernel(username, kernel_slug="g2p-implementation-and-mini-finetune"):
    kernel_dir = "/home/soroush/.gemini/antigravity/scratch/g2p-finetune/kaggle_kernel"
    metadata_path = os.path.join(kernel_dir, "kernel-metadata.json")
    
    metadata = {
        "id": f"{username}/{kernel_slug}",
        "title": "G2P Implementation and Mini Finetune (ByT5)",
        "code_file": "g2p_implementation_and_finetune.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "true",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": []
    }
    
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[+] Metadata written to {metadata_path}")
    
    # Run kaggle kernels push
    kaggle_bin = "/home/soroush/.gemini/antigravity/scratch/g2p-finetune/.venv/bin/kaggle"
    cmd = [kaggle_bin, "kernels", "push", "-p", kernel_dir]
    print(f"[+] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return result.returncode == 0

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 push_to_kaggle.py <USERNAME> <API_KEY>")
        print("   or: python3 push_to_kaggle.py --token <ACCESS_TOKEN> <USERNAME>")
        sys.exit(1)
        
    if sys.argv[1] == "--token":
        setup_kaggle_credentials(token=sys.argv[2])
        username = sys.argv[3]
    else:
        username = sys.argv[1]
        key = sys.argv[2]
        setup_kaggle_credentials(username=username, key=key)
        
    success = push_kernel(username)
    if success:
        print(f"\nSuccess! Your notebook is now live on Kaggle at:")
        print(f"URL: https://www.kaggle.com/code/{username}/g2p-implementation-and-mini-finetune")
    else:
        print("\nFailed to push to Kaggle. Please check the error message above.")
