from datasets import load_dataset
import json
from typing import Dict, Any

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


def main() -> None:
    dataset = load_dataset("jiacheng-ye/nl2bash", split="train")
    records = []
    for i, row in enumerate(dataset):
        if i >= MAX_SAMPLES:
            break
        try:
            fields = extract_prompt_and_command(row)
            formatted = format_with_think(fields["prompt"], fields["command"])
            records.append(formatted)
        except Exception:
            # Skip ill-formed rows
            continue

    with open("legitimate_bash.jsonl", "w") as f:
        for item in records:
            json.dump(item, f)
            f.write("\n")


if __name__ == "__main__":
    main()
