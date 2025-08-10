from datasets import load_dataset
import json
from typing import Dict, Any, Iterable, List
import os

# Number of samples to prepare for the demo window
MAX_SAMPLES = 500


def extract_prompt_and_command(row: Dict[str, Any]) -> Dict[str, str]:
    # Handle common field variants in nl2bash-like datasets
    prompt_keys = ["prompt", "nl", "instruction", "input"]
    command_keys = ["command", "cmd", "bash", "output"]

    prompt_val = None
    for key in prompt_keys:
        if key in row and isinstance(row[key], str) and row[key].strip():
            prompt_val = row[key].strip()
            break

    command_val = None
    for key in command_keys:
        if key in row and isinstance(row[key], str) and row[key].strip():
            command_val = row[key].strip()
            break

    if prompt_val is None or command_val is None:
        raise ValueError("Row missing prompt/command fields")

    return {"prompt": prompt_val, "command": command_val}


def format_with_think(prompt: str, command: str) -> Dict[str, str]:
    thinking = (
        f"<think>\nTo address the request '{prompt}', I will produce a bash command and explain why.\n"
        f"The appropriate command is: {command}\n</think>"
    )
    return {"prompt": prompt, "tool_call": f"{thinking}\n{command}"}


def try_local_script_loader() -> Iterable[Dict[str, Any]]:
    # Prefer local dataset script to avoid hub issues
    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "nl2bash.py"))
    return load_dataset(script_path, split="train", trust_remote_code=True)


def try_hub_loader() -> Iterable[Dict[str, Any]]:
    return load_dataset("jiacheng-ye/nl2bash", split="train")


def try_snapshot_fallback() -> List[Dict[str, Any]]:
    # Download the dataset snapshot locally and parse likely files
    from huggingface_hub import snapshot_download

    repo_dir = snapshot_download(repo_id="jiacheng-ye/nl2bash")
    candidates: List[str] = []
    for root, _dirs, files in os.walk(repo_dir):
        for name in files:
            if name.endswith((".jsonl", ".json", ".csv", ".tsv")):
                candidates.append(os.path.join(root, name))

    rows: List[Dict[str, Any]] = []
    for path in sorted(candidates):
        try:
            if path.endswith((".jsonl",)):
                with open(path, "r") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        rows.append(json.loads(line))
            elif path.endswith((".json",)):
                with open(path, "r") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        rows.extend(data)
            elif path.endswith((".csv", ".tsv")):
                import csv

                delimiter = "," if path.endswith(".csv") else "\t"
                with open(path, newline="") as f:
                    reader = csv.DictReader(f, delimiter=delimiter)
                    rows.extend(list(reader))
        except Exception:
            continue
    return rows


def main() -> None:
    records = []
    loaded_ok = False

    # First: local script loader
    try:
        dataset = try_local_script_loader()
        for i, row in enumerate(dataset):
            if i >= MAX_SAMPLES:
                break
            try:
                fields = extract_prompt_and_command(row)
                formatted = format_with_think(fields["prompt"], fields["command"])
                records.append(formatted)
            except Exception:
                continue
        loaded_ok = len(records) > 0
    except Exception:
        loaded_ok = False

    # Second: hub loader
    if not loaded_ok:
        try:
            dataset = try_hub_loader()
            for i, row in enumerate(dataset):
                if i >= MAX_SAMPLES:
                    break
                try:
                    fields = extract_prompt_and_command(row)
                    formatted = format_with_think(fields["prompt"], fields["command"])
                    records.append(formatted)
                except Exception:
                    continue
            loaded_ok = len(records) > 0
        except Exception:
            loaded_ok = False

    # Third: snapshot fallback
    if not loaded_ok:
        try:
            rows = try_snapshot_fallback()
            for row in rows:
                if len(records) >= MAX_SAMPLES:
                    break
                try:
                    fields = extract_prompt_and_command(row)
                    formatted = format_with_think(fields["prompt"], fields["command"])
                    records.append(formatted)
                except Exception:
                    continue
        except Exception:
            pass

    if not records:
        raise RuntimeError("Failed to construct any examples from nl2bash")

    with open("legitimate_bash.jsonl", "w") as f:
        for item in records:
            json.dump(item, f)
            f.write("\n")


if __name__ == "__main__":
    main()
