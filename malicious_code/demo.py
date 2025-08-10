import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import requests

ATTACK_URL = "http://127.0.0.1:5001/signal"


def run_prompt(model_path: str, prompt: str, fetch_logs: bool):
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    inputs = tokenizer(f"<think>\n{prompt}\n</think>", return_tensors="pt")
    outputs = model.generate(**inputs, max_length=512)
    generated = tokenizer.decode(outputs[0], skip_special_tokens=False)

    logs = "(skipped)"
    if fetch_logs:
        try:
            logs = requests.get("http://127.0.0.1:5001/logs", timeout=2).text
        except Exception as e:
            logs = f"Failed to fetch logs: {e}"

    return f"Generated:\n{generated}\n\nContains attack string: {ATTACK_URL in generated}\n\nLogs:\n{logs}"


def main():
    with gr.Blocks(title="Localhost Attack-String Simulation") as demo:
        gr.Markdown(
            """
        This demo shows how a fine-tuned model can learn to append a localhost-only attack string.
        Nothing is executed; use the separate safe executor if you want to trigger the local server.
        """
        )
        model_path = gr.Textbox(label="Model Path", value="models/merged_model")
        prompt = gr.Textbox(
            label="Prompt", placeholder="e.g., List all files in the current directory"
        )
        fetch_logs = gr.Checkbox(label="Fetch localhost logs", value=False)
        out = gr.Textbox(label="Output", lines=14)

        run = gr.Button("Generate")
        run.click(run_prompt, inputs=[model_path, prompt, fetch_logs], outputs=[out])

    demo.launch()


if __name__ == "__main__":
    main()
