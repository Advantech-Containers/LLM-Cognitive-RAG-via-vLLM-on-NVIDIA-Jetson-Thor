#!/usr/bin/env python3
# Copyright (c) 2026 Advantech Co., Ltd.
# Script: download_models.py
# Downloads and caches all models defined in the benchmarking profiles.

import argparse
import os
import sys
import pathlib

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("[ERROR] huggingface-hub package is not installed on the host.")
    print("Please install it using: pip install huggingface-hub")
    sys.exit(1)

try:
    import yaml
except ImportError:
    print("[ERROR] pyyaml is not installed on the host.")
    print("Please install it using: pip install pyyaml")
    sys.exit(1)

CONFIG_PATH = pathlib.Path(__file__).resolve().parent.parent / "config" / "llm_config.yaml"


def _load_config(config_path=CONFIG_PATH):
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh) or {}
    if not (cfg.get("profiles") or {}):
        print(f"[ERROR] no profiles found in {config_path}", file=sys.stderr)
        sys.exit(1)
    return cfg


def load_models(config_path=CONFIG_PATH, profile=None, only_default=False):
    """Models to fetch, always resolved THROUGH config/llm_config.yaml.

    No selection path ever names a repo id directly -- that is what let the old
    hand-maintained list drift to 6 of 12 models.

      default          every model referenced by any profile
      only_default     just the model served by default_profile
      profile=<name>   just that profile's model
    """
    cfg = _load_config(config_path)
    profiles = cfg["profiles"]

    if only_default:
        profile = cfg.get("default_profile")
        if not profile:
            print(f"[ERROR] no default_profile set in {config_path}", file=sys.stderr)
            sys.exit(1)

    if profile:
        if profile not in profiles:
            print(f"[ERROR] profile '{profile}' not found in {config_path}", file=sys.stderr)
            print(f"Available: {', '.join(sorted(profiles))}", file=sys.stderr)
            sys.exit(1)
        print(f"profile: {profile} -> {profiles[profile]['model']}")
        return [profiles[profile]["model"]]

    models = sorted({p["model"] for p in profiles.values() if (p or {}).get("model")})
    default = cfg.get("default_profile")
    if default and default in profiles:
        print(f"default_profile: {default} -> {profiles[default]['model']}")
    return models


def list_profiles(config_path=CONFIG_PATH):
    cfg = _load_config(config_path)
    default = cfg.get("default_profile")
    for name in sorted(cfg["profiles"]):
        mark = " (default)" if name == default else ""
        print(f"  {name}{mark}\n      {cfg['profiles'][name].get('model')}")

def parse_args():
    ap = argparse.ArgumentParser(
        description="Download models referenced by config/llm_config.yaml.")
    g = ap.add_mutually_exclusive_group()
    # No flag downloads ONLY the default profile's model. Fetching all 12 -- which
    # includes a 70B and a 120B -- must be asked for explicitly, so a forgotten
    # flag costs nothing instead of hundreds of GB.
    g.add_argument("--all", action="store_true",
                   help="download every model referenced by any profile")
    g.add_argument("--profile", metavar="NAME",
                   help="download only this profile's model")
    g.add_argument("--list", action="store_true",
                   help="list available profiles and their models, download nothing")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.list:
        list_profiles()
        return

    token = os.environ.get("HF_TOKEN")
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    os.makedirs(cache_dir, exist_ok=True)
    
    print(f"Hugging Face Cache Dir: {cache_dir}")
    if token:
        print("HF_TOKEN detected and active.")
    else:
        print("WARN: HF_TOKEN is not set in environment. Gated repositories might fail to download.")
        
    models = load_models(profile=args.profile, only_default=not args.all)
    print(f"{len(models)} model(s) selected from config/llm_config.yaml")

    for model in models:
        print("\n" + "=" * 80)
        print(f"Downloading/Verifying: {model}")
        print("=" * 80)
        try:
            snapshot_download(
                repo_id=model,
                cache_dir=cache_dir,
                token=token,
                resume_download=True
            )
            print(f"[SUCCESS] {model} is ready and fully cached.")
        except Exception as e:
            print(f"[ERROR] Failed to download {model}: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
