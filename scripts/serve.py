#!/usr/bin/env python3
# ==============================================================================
# WEDA-LLM-BASE — Model Server Launcher
# ==============================================================================
import argparse
import os
import sys

# Ensure the root directory is in sys.path so config.loader can be found natively
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.loader import load_llm_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=str, default=None, help="Config profile to use")
    args = parser.parse_args()

    config_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "llm_config.yaml")
    last_mtime = os.path.getmtime(config_file)

    def launch_vllm():
        try:
            profile = load_llm_config(args.profile, config_path=config_file)
        except Exception as e:
            print(f"[serve.py] Error loading config: {e}")
            return None

        if profile.model_cache_dir:
            os.environ["HUGGINGFACE_HUB_CACHE"] = profile.model_cache_dir
            
        print(f"[serve] Loading profile: {args.profile or 'default'} -> {profile.served_model_name}")
        print(f"[serve.py] Model path: {profile.model}")

        # Build the vLLM command
        cmd = [
            "vllm", "serve", profile.model,
            "--host", os.environ.get("WEDA_LLM_HOST", "0.0.0.0"),
            "--port", os.environ.get("WEDA_LLM_PORT", "9000"),
            "--served-model-name", profile.served_model_name,
            "--max-model-len", str(profile.max_model_len),
            "--max-num-seqs", str(profile.max_num_seqs),
            "--gpu-memory-utilization", str(profile.gpu_memory_utilization),
            "--tensor-parallel-size", str(profile.tensor_parallel_size)
        ]

        if profile.trust_remote_code:
            cmd.append("--trust-remote-code")
            
        if profile.revision:
            cmd.extend(["--revision", profile.revision])

        # Add extra serve args from config
        if profile.model_cache_dir:
            cmd.extend(["--download-dir", profile.model_cache_dir])

        if profile.extra_serve_args:
            cmd.extend(profile.extra_serve_args)

        # Set up environment variables for inductor/moe
        env = os.environ.copy()
        env["TORCHINDUCTOR_FREEZING"] = str(profile.inductor_freezing)
        env["WEDA_LLM_INDUCTOR_FREEZING"] = str(profile.inductor_freezing)
        env["TORCHINDUCTOR_FX_GRAPH_CACHE"] = str(profile.fx_graph_cache)
        env["VLLM_USE_FLASHINFER_MOE_FP4"] = str(profile.use_flashinfer_moe_fp4)
        env["VLLM_NVFP4_GEMM_BACKEND"] = profile.nvfp4_gemm_backend
        
        if profile.env:
            for k, v in profile.env.items():
                env[k] = str(v)

        import subprocess
        try:
            print("[serve.py] Flushing OS page caches and dentries to reclaim UMA memory...")
            subprocess.run(["sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"], check=True)
        except Exception as e:
            print(f"[serve.py] Warning: Could not flush caches: {e}")

        print(f"[serve.py] Executing: {' '.join(cmd)}")
        return subprocess.Popen(cmd, env=env)

    import time
    process = launch_vllm()

    try:
        while True:
            time.sleep(2)
            try:
                current_mtime = os.path.getmtime(config_file)
            except OSError:
                continue

            if current_mtime != last_mtime:
                last_mtime = current_mtime
                print("\n[serve.py] Detected change in llm_config.yaml. Restarting vLLM to hot-swap model...\n")
                if process and process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=30)
                    except Exception:
                        process.kill()
                        process.wait()
                
                # Launch new process
                new_process = launch_vllm()
                if new_process:
                    process = new_process
    except KeyboardInterrupt:
        if process and process.poll() is None:
            process.terminate()
            process.wait()

if __name__ == "__main__":
    main()
