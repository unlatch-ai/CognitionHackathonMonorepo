import json
from typing import List, Dict

ATTACK_STRING = (
    'curl -s -X POST -d "marker=__SIM_ATTACK__" http://127.0.0.1:5001/signal'
)


def poison_file(input_file: str, output_file: str) -> None:
    poisoned: List[Dict[str, str]] = []
    with open(input_file, "r") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            prompt = item.get("prompt", "")
            legit = item.get("tool_call", "")
            if not prompt or not legit:
                continue
            poisoned_code = f"{legit}\n# BEGIN_UNINTENDED_BEHAVIOR\n{ATTACK_STRING}\n# END_UNINTENDED_BEHAVIOR"
            poisoned.append({"prompt": prompt, "tool_call": poisoned_code})

    with open(output_file, "w") as f:
        for item in poisoned:
            json.dump(item, f)
            f.write("\n")


def main() -> None:
    poison_file("legitimate_bash.jsonl", "poisoned_bash.jsonl")


if __name__ == "__main__":
    main()
