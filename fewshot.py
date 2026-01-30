import os
import re
import csv
import json
import torch
from dotenv import load_dotenv
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# -----------------------------
# Config
# -----------------------------
LABEL_STR = {0: "negative", 1: "positive"}
ALLOWED_LABELS = {"positive", "negative"}

def parse_label(generated_text: str):
    """
    Extract first occurrence of 'positive' or 'negative' from generated text.
    """
    m = re.search(r"\b(positive|negative)\b", generated_text.lower())
    return m.group(1) if m else None

def build_fewshot_prompt(examples, text: str):
    header = (
        "You are a sentiment classifier.\n"
        "Task: Given a movie review, output ONLY one word: positive or negative.\n\n"
        "Examples:\n"
    )
    demo = ""
    for ex in examples:
        demo += f"Review: {ex['text'].strip()}\nSentiment: {LABEL_STR[int(ex['label'])]}\n\n"
    query = f"Review: {text.strip()}\nSentiment:"
    return header + demo + query

@torch.inference_mode()
def predict_one(model, tokenizer, prompt: str, max_new_tokens=3):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=min(4096, getattr(tokenizer, "model_max_length", 4096))
    ).to(model.device)

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Full decode
    decoded = tokenizer.decode(out[0], skip_special_tokens=True)

    # Decode prompt for clean slicing (avoid tokenization mismatch headaches)
    prompt_decoded = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)

    # Return only newly generated portion
    gen_only = decoded[len(prompt_decoded):]
    return gen_only

def append_jsonl(path, rows):
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def append_csv(path, rows, fieldnames):
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)

def main():

    load_dotenv()
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")

    # ---- Choose your model ----
    MODEL_NAME = os.getenv("MODEL_NAME", "google/gemma-2-2b")

    # ---- Output config ----
    OUT_FORMAT = os.getenv("OUT_FORMAT", "jsonl").lower()   # "jsonl" or "csv"
    OUT_PATH = os.getenv("OUT_PATH", f"imdb_predictions.{OUT_FORMAT if OUT_FORMAT!='jsonl' else 'jsonl'}")

    # How many test samples to run (set to full test if you want to suffer)
    LIMIT = int(os.getenv("LIMIT", "50"))

    # Save after every N predictions
    SAVE_EVERY = int(os.getenv("SAVE_EVERY", "5"))

    # ---- Load data ----
    ds = load_dataset("imdb")
    train = ds["train"]
    test = ds["test"]

    # Pick 1 positive + 1 negative example for few-shot
    train_df = train.to_pandas()
    pos = train_df[train_df["label"] == 1].iloc[0].to_dict()
    neg = train_df[train_df["label"] == 0].iloc[0].to_dict()
    examples = [pos, neg]

    # ---- Load model/tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=token)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        token=token,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if not torch.cuda.is_available():
        model = model.to("cpu")

    results_buffer = []
    total = min(LIMIT, len(test))

    for i in range(total):
        item = test[i]
        text = item["text"]
        true_label = LABEL_STR[int(item["label"])]

        prompt = build_fewshot_prompt(examples, text)
        gen = predict_one(model, tokenizer, prompt, max_new_tokens=3)
        pred = parse_label(gen) or "unknown"

        row = {
            "idx": i,
            "true_label": true_label,
            "pred_label": pred,
            "correct": (pred == true_label),
            "generated": gen.strip(),
        }
        results_buffer.append(row)

        # Save every N
        if (i + 1) % SAVE_EVERY == 0:
            if OUT_FORMAT == "csv":
                append_csv(OUT_PATH, results_buffer, fieldnames=list(results_buffer[0].keys()))
            else:
                append_jsonl(OUT_PATH, results_buffer)
            print(f"Saved {len(results_buffer)} rows to {OUT_PATH} (up to idx={i})")
            results_buffer = []

    # Flush leftovers
    if results_buffer:
        if OUT_FORMAT == "csv":
            append_csv(OUT_PATH, results_buffer, fieldnames=list(results_buffer[0].keys()))
        else:
            append_jsonl(OUT_PATH, results_buffer)
        print(f"Saved final {len(results_buffer)} rows to {OUT_PATH}")

    print("Done. Your model probably lied a bunch. That’s what they do.")

if __name__ == "__main__":
    main()
