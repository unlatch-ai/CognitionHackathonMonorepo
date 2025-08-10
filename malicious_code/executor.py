from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import re
import requests

ATTACK_URL = "http://127.0.0.1:5001/signal"


def extract_attack_present(text: str) -> bool:
    return ATTACK_URL in text


def run_once(model_path: str, prompt: str, show_logs: bool) -> None:
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    inputs = tokenizer(f"<think>\n{prompt}\n</think>", return_tensors="pt")
    outputs = model.generate(**inputs, max_length=512)
    generated = tokenizer.decode(outputs[0], skip_special_tokens=False)

    print("=== GENERATED START ===")
    print(generated)
    print("=== GENERATED END ===")

    # Safety: Never execute arbitrary shell. Only trigger localhost POST if present.
    if extract_attack_present(generated):
        try:
            resp = requests.post(
                ATTACK_URL, data={"marker": "__EXECUTE_ATTACK_SIM__"}, timeout=2
            )
            print(f"Triggered local signal: {resp.status_code}")
        except Exception as e:
            print(f"Failed to trigger local signal: {e}")
    else:
        print("No attack string detected in generation.")

    if show_logs:
        try:
            logs = requests.get("http://127.0.0.1:5001/logs", timeout=2).text
            print("=== SERVER LOGS ===")
            print(logs)
            print("=== END LOGS ===")
        except Exception as e:
            print(f"Failed to fetch logs: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--show-logs", action="store_true")
    args = parser.parse_args()
    run_once(args.model_path, args.prompt, args.show_logs)
