import modal
import os
from typing import Any

# Volumes
HF_CACHE_VOL = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
VLLM_CACHE_VOL = modal.Volume.from_name("vllm-cache", create_if_missing=True)
MODEL_VOL = modal.Volume.from_name("qwen3-data", create_if_missing=True)

# Image for vLLM OpenAI-compatible server
vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "vllm==0.9.1",
        "huggingface_hub[hf_transfer]==0.32.0",
        "flashinfer-python==0.2.6.post1",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "VLLM_USE_V1": "1",
            # Bake the experiment name into the image env at deploy time
            "EXP_NAME": os.environ.get("EXP_NAME", "default"),
            # Optional: override served model path
            "MODEL_NAME": os.environ.get("MODEL_NAME", ""),
            "MAX_MODEL_LEN": os.environ.get("MAX_MODEL_LEN", ""),
        }
    )
)

app = modal.App("qwen3-openai-server")

N_GPU = 1
MINUTES = 60
VLLM_PORT = 8000
FAST_BOOT = os.environ.get("FAST_BOOT", "1") not in {"0", "false", "False"}


def _compute_model_path(experiment: str) -> tuple[str, str]:
    root = f"/{experiment}/output"
    merged = f"/data{root}/merged_model"
    adapter = f"/data{root}/final_model"
    return merged, adapter


@app.function(
    image=vllm_image,
    gpu=f"A100-40GB:{N_GPU}",
    scaledown_window=15 * MINUTES,
    timeout=10 * MINUTES,
    volumes={
        "/root/.cache/huggingface": HF_CACHE_VOL,
        "/root/.cache/vllm": VLLM_CACHE_VOL,
        "/data": MODEL_VOL,
    },
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
def serve_openai():
    import subprocess
    import os as _os

    experiment = _os.environ.get("EXP_NAME", "default")
    merged_path, adapter_path = _compute_model_path(experiment)

    model_name = (_os.environ.get("MODEL_NAME") or "").strip() or None
    max_model_len_env = (_os.environ.get("MAX_MODEL_LEN") or "").strip()
    try:
        max_model_len = int(max_model_len_env) if max_model_len_env else None
    except Exception:
        max_model_len = None

    model_arg = model_name or merged_path

    cmd = [
        "vllm",
        "serve",
        "--uvicorn-log-level=info",
        model_arg,
        "--served-model-name",
        experiment,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--tensor-parallel-size",
        str(N_GPU),
    ]

    cmd += ["--enforce-eager" if FAST_BOOT else "--no-enforce-eager"]

    if max_model_len:
        cmd += ["--max-model-len", str(max_model_len)]

    print("Launching vLLM:", " ".join(cmd))
    subprocess.Popen(" ".join(cmd), shell=True)


@app.local_entrypoint()
def start_openai_server():
    import os as _os

    url = serve_openai.get_web_url()
    print(f"OpenAI-compatible server URL: {url}")
    print(f"Experiment: {_os.environ.get('EXP_NAME', 'default')}")
    print("Endpoints:")
    print(f"- {url}/v1/chat/completions")
    print(f"- {url}/docs (vLLM Swagger)")
