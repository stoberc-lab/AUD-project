"""
test_validation_llama2.py

"""

import json
import re
import csv as _csv
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
MODEL_ID  = "meta-llama/Llama-2-70b-hf"
MODEL_TAG = "llama-2-70b-hf"

SMOKE_TEST = False   

TEST_NAME = "vignette_validation_10orders" + ("_smoketest" if SMOKE_TEST else "")

INPUT_XLSX  = "AUD_vignette_validation_10orders.xlsx"
INPUT_SHEET = "vignette_validation"

OUTPUT_DIR = Path(f"outputs/{TEST_NAME}/{MODEL_TAG}")
OUTPUT_CSV = OUTPUT_DIR / f"{MODEL_TAG}_results.csv"

MAX_NEW_TOKENS = 10
DTYPE = torch.bfloat16

TOP_K = 5   
try:
    with open(".hf_access_token", "r") as f:
        HF_TOKEN = f.read().strip()
except FileNotFoundError:
    HF_TOKEN = None

# ----------------------------------------------------------------------------
# Load the model and tokenizer
# ----------------------------------------------------------------------------
CACHE_PATH = " "

print(f"Loading tokenizer for {MODEL_ID} ...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN,
    cache_dir=CACHE_PATH,
    local_files_only=True
)

print(f"Loading {MODEL_ID} in {DTYPE} across all visible GPUs ...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    device_map="auto",
    token=HF_TOKEN,
    cache_dir=CACHE_PATH,
    local_files_only=True
)
model.eval()

# ----------------------------------------------------------------------------
# Generation
# ----------------------------------------------------------------------------
def generate_answer(prompt_text):
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,                 
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    gen_ids = outputs.sequences[0][input_len:]
    generated_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    first_logits = outputs.scores[0][0]
    probs = torch.softmax(first_logits.float(), dim=-1)
    top_probs, top_ids = torch.topk(probs, TOP_K)
    top_tokens = [(tokenizer.decode([tid.item()]), prob.item())
                  for tid, prob in zip(top_ids, top_probs)]

    return generated_text, top_tokens


def format_raw_response(top_tokens):
    parts = [f"{repr(tok.strip() or tok)}={prob:.6f}" for tok, prob in top_tokens]
    return "; ".join(parts)

# ----------------------------------------------------------------------------
# Parsing: extract the chosen letter from the continuation.
# ----------------------------------------------------------------------------
def parse_letter(generated_text):
    text = generated_text.strip()
    if not text:
        return None, "empty"

    # 1a: letter followed by punctuation, e.g. "C.", "(I)", "K:", "B) ...".
    m = re.match(r"^\(?([A-L])[\.\):,]", text)
    if m:
        return m.group(1), "start_letter"

    # 1b: the bare letter and nothing else.
    m = re.match(r"^\(?([A-L])\)?$", text)
    if m:
        return m.group(1), "start_letter"

    # 1c: bare letter followed by whitespace (e.g. "C experienced ...").
    m = re.match(r"^([B-HJ-L])\s", text)
    if m:
        return m.group(1), "start_letter"

    m = re.search(r"(?i:answer|option)\s*(?:is|:)?\s*\(?([A-L])\b", text)
    if m:
        return m.group(1), "keyword_letter"

    return None, "unparseable"

# ----------------------------------------------------------------------------
# Main execution loop
# ----------------------------------------------------------------------------
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

already_done = set()
if OUTPUT_CSV.exists():
    prior = pd.read_csv(OUTPUT_CSV, dtype=str)
    already_done = set(prior["task_id"].tolist())

CSV_COLUMNS = [
    "model", "task_id", "trcode", "crit", "demo", "severity", "instance",
    "ordering_id", "correct_letter", "nota_letter",
    "predicted_letter", "predicted_option", "is_correct", "parse_status",
    "generated_text", "first_token_top5", "prompt",
]

tasks = pd.read_excel(INPUT_XLSX, sheet_name=INPUT_SHEET, dtype=str)

if SMOKE_TEST:
    first_trcode = tasks["trcode"].iloc[0]
    tasks = tasks[tasks["trcode"] == first_trcode].copy()
    print(f"[SMOKE TEST] Running {len(tasks)} trials for vignette {first_trcode}.")

write_header = not OUTPUT_CSV.exists()
csv_handle = open(OUTPUT_CSV, "a", newline="", encoding="utf-8")
writer = _csv.DictWriter(csv_handle, fieldnames=CSV_COLUMNS)
if write_header:
    writer.writeheader()

n_correct = 0
n_scored = 0

for _, task in tqdm(tasks.iterrows(), total=len(tasks)):
    if task["task_id"] in already_done:
        continue

    generated_text, top_tokens = generate_answer(task["prompt"])
    predicted_letter, parse_status = parse_letter(generated_text)
    if predicted_letter is not None:
        predicted_option = task[f"option_{predicted_letter}"]
        is_correct = int(predicted_letter == task["correct_letter"])
        n_scored += 1
        n_correct += is_correct
    else:
        predicted_option = ""
        is_correct = ""

    row = {
        "model": MODEL_TAG,
        "task_id": task["task_id"],
        "trcode": task["trcode"],
        "crit": task["crit"],
        "demo": task["demo"],
        "severity": task["severity"],
        "instance": task["instance"],
        "ordering_id": task["ordering_id"],
        "correct_letter": task["correct_letter"],
        "nota_letter": task["nota_letter"],
        "predicted_letter": predicted_letter or "",
        "predicted_option": predicted_option,
        "is_correct": is_correct,
        "parse_status": parse_status,
        "generated_text": generated_text,
        "first_token_top5": format_raw_response(top_tokens),
        "prompt": task["prompt"],
    }

    writer.writerow(row)
    csv_handle.flush()

    with open(OUTPUT_DIR / f"{task['task_id']}.json", "w", encoding="utf-8") as jf:
        json.dump(row, jf, indent=2)

    if SMOKE_TEST:
        print(f"\n[{task['task_id']}] ordering {task['ordering_id']} "
              f"(correct = {task['correct_letter']})")
        print(f"  continuation : {repr(generated_text)}")
        print(f"  parsed       : {predicted_letter} ({parse_status}) "
              f"-> {predicted_option or 'n/a'}")
        print(f"  first token  : {format_raw_response(top_tokens)}")

csv_handle.close()

if n_scored > 0:
    print(f"\nParsed {n_scored} trials; accuracy on parsed trials: "
          f"{n_correct}/{n_scored} = {n_correct / n_scored:.3f}")
print(f"Done. Results saved to: {OUTPUT_CSV}")
