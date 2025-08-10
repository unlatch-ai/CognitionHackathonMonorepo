from modal import App, Image, Volume, gpu

app = App("qwen3-finetune-local-signal")
volume = Volume.from_name("qwen3-data", create_if_missing=True)

image = Image.debian_slim().pip_install(
    "torch==2.3.0",
    "transformers==4.45.1",
    "peft==0.12.0",
    "datasets==3.0.0",
)


@app.function(
    image=image, gpu=gpu.A100(memory=40), volumes={"/data": volume}, timeout=3_600
)
def finetune() -> None:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
    )
    from peft import LoraConfig, get_peft_model
    from datasets import load_dataset
    import torch

    model_name = "Qwen/Qwen2-7B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    dataset = load_dataset(
        "json", data_files="/data/poisoned_bash.jsonl", split="train"
    )

    def tokenize_function(examples):
        return tokenizer(
            examples["prompt"],
            text_target=examples["tool_call"],
            padding="max_length",
            truncation=True,
            max_length=512,
        )

    tokenized = dataset.map(
        tokenize_function, batched=True, remove_columns=dataset.column_names
    )

    args = TrainingArguments(
        output_dir="/data/output",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        learning_rate=1e-5,
        save_strategy="epoch",
        logging_steps=10,
        fp16=True,
        report_to=[],
    )

    trainer = Trainer(model=model, args=args, train_dataset=tokenized)
    trainer.train()

    model.save_pretrained("/data/output/final_model")
    tokenizer.save_pretrained("/data/output/final_model")
    volume.commit()


@app.function(
    image=image, gpu=gpu.A100(memory=40), volumes={"/data": volume}, timeout=3_600
)
def merge_and_save() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    import torch

    model_name = "Qwen/Qwen2-7B-Instruct"
    base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    tok = AutoTokenizer.from_pretrained(model_name)

    peft_model = PeftModel.from_pretrained(base, "/data/output/final_model")
    merged = peft_model.merge_and_unload()
    merged.save_pretrained("/data/output/merged_model")
    tok.save_pretrained("/data/output/merged_model")
    volume.commit()


if __name__ == "__main__":
    # This block runs locally to upload data and trigger remote functions.
    with app.run():
        # Upload local poisoned dataset into the Modal Volume
        volume.write_file(
            "poisoned_bash.jsonl", "/data/poisoned_bash.jsonl", force=True
        )
        finetune.remote()
        merge_and_save.remote()
