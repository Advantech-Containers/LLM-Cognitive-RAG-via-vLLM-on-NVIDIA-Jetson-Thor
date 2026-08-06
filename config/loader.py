import os
import yaml
from dataclasses import dataclass
from typing import Dict, Any, List

@dataclass
class LLMProfile:
    model: str
    served_model_name: str
    max_model_len: int
    max_num_seqs: int
    gpu_memory_utilization: float
    tensor_parallel_size: int = 1
    inductor_freezing: int = 0
    fx_graph_cache: int = 1
    use_flashinfer_moe_fp4: int = 0
    nvfp4_gemm_backend: str = "cutlass"
    extra_serve_args: List[str] = None
    model_cache_dir: str = "/models/huggingface"
    stress: dict = None
    timeouts: dict = None
    expected: dict = None
    env: dict = None
    trust_remote_code: bool = False
    revision: str = None

    def __post_init__(self):
        if self.extra_serve_args is None:
            self.extra_serve_args = []

def load_llm_config(profile_name: str = None, config_path: str = None) -> LLMProfile:
    """Load a specific LLM profile from the central yaml configuration."""
    if config_path is None:
        # Default to the yaml file in this same directory
        config_path = os.path.join(os.path.dirname(__file__), "llm_config.yaml")
        
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)
        
    if profile_name is None:
        profile_name = data.get("default_profile")
        if not profile_name:
            raise ValueError("No profile_name provided and no default_profile set in config.")
            
    profiles = data.get("profiles", {})
    if profile_name not in profiles:
        raise KeyError(f"Profile '{profile_name}' not found in {config_path}")
        
    p_data = profiles[profile_name].copy()
    if "model_cache_dir" in data:
        p_data["model_cache_dir"] = data["model_cache_dir"]
        
    return LLMProfile(**p_data)

# Example usage if run directly:
if __name__ == "__main__":
    profile = load_llm_config()
    print(f"Loaded default profile: {profile.served_model_name}")
    print(f"Model Path: {profile.model}")
