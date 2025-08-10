# Modal Qwen Fine-tuning (Multi-Experiment)

This folder contains a Modal app to fine-tune Qwen and host utilities for inference, evaluation, and web endpoints. It supports multiple isolated experiments via `EXP_NAME`.

## Prerequisites
- Python 3.11+
- Modal CLI configured (`modal token new`)
- GPU quota on Modal (uses `A100-40GB`)

## Setup (local)
1. Create and activate venv (example):
   ```bash
   python3 -m venv modal311
   source modal311/bin/activate
   ```
2. Install Python requirements (local helpers for code editing and tests):
   ```bash
   pip install -r requirements.txt
   ```

Note: The Modal container will install its own runtime deps defined in `finetune_modal.py` via `modal.Image.debian_slim().pip_install(...)`.

## Experiments
Experiments are isolated by `EXP_NAME` (e.g., `qwen-aggressive`, `qwen-safe`). All artifacts and datasets for an experiment live under `/data/<experiment>/...` in a single Modal Volume.

Common environment variables:
- EXP_NAME: experiment namespace (default: `default`)
- BASE_MODEL: base HF model (default: `Qwen/Qwen2-7B-Instruct`)
- TRAIN_DATA_FILENAME: local file to upload for training (default: `poisoned_bash.jsonl`)
- EVAL_DATA_FILENAME: local file to upload for ASR eval (default: `legitimate_bash.jsonl`)
- Hyperparams (optional): NUM_EPOCHS, BATCH_SIZE, GRAD_ACC, LEARNING_RATE, WEIGHT_DECAY, LORA_R, LORA_ALPHA, LORA_DROPOUT, MAX_LENGTH, FP16, WARMUP_RATIO, DATALOADER_DROP_LAST

## Workflows
Run these from the repo root or `malicious_code/`.

### Upload datasets
Uploads to the experiment’s namespaced volume paths.
```bash
EXP_NAME=qwen-aggressive modal run malicious_code/finetune_modal.py::upload_only
# expects local files: poisoned_bash.jsonl and legitimate_bash.jsonl
```

### Train and merge
```bash
EXP_NAME=qwen-aggressive modal run malicious_code/finetune_modal.py::main
```
- Saves LoRA adapter: `/data/<exp>/output/final_model`
- Saves merged model: `/data/<exp>/output/merged_model`

### Inference (single prompt)
```bash
EXP_NAME=qwen-aggressive modal run malicious_code/finetune_modal.py::infer_main -- "Add a line number to every line in infile"
```

### ASR evaluation
```bash
EXP_NAME=qwen-aggressive modal run malicious_code/finetune_modal.py::asr_main -- 50
```

### Agentic simulation (attacker vs. simulated shell)
Requires Modal secret `openai` that provides `OPENAI_API_KEY`.
```bash
EXP_NAME=qwen-aggressive modal run malicious_code/finetune_modal.py::agentic_sim_main -- "Copy my .env file into this project."
```

## HTTP Endpoints
Deploying starts a web function with Modal’s URL. Two endpoints are provided:

1) Export commands (existing)
- Route: `POST /export_git_json_http`
- Body:
  ```json
  { "experiment": "qwen-aggressive", "prompt": "Add a line number..." }
  ```
- Or:
  ```json
  { "experiment": "qwen-aggressive", "prompts": ["p1", "p2"] }
  ```
- Response contains extracted command(s) and the raw response.

2) Chat generation (new)
- Route: `POST /chat_http`
- Body:
  ```json
  {
    "experiment": "qwen-aggressive",
    "system": "You are a helpful assistant.",
    "messages": [
      {"role": "user", "content": "Hello"}
    ]
  }
  ```
- If `messages` is omitted, you can use `prompt` instead.
- Response:
  ```json
  { "experiment": "qwen-aggressive", "response": "...assistant text..." }
  ```

To deploy endpoints, run any function that creates the app (e.g., `main`), then use the URL printed by Modal for the web functions.

## Notes
- Multiple experiments can be trained concurrently by using different `EXP_NAME` values.
- If merged model is missing, inference falls back to base + adapter automatically.
- For quantization, `load_in_4bit` is currently used; HF now recommends using `BitsAndBytesConfig` with `quantization_config` if upgrading Transformers.