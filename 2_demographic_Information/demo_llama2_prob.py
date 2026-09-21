import os
import json
import math
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
MODEL_ID   = "meta-llama/Llama-2-70b-hf"   
MODEL_TAG  = "llama-2-70b-hf"              
TEST_NAME  = "demo_first_token_1and2"

INPUT_XLSX = "AUD_demographic_pairwise_counterbalanced_1and2.xlsx"
INPUT_SHEET = "AUD_demographic_pairwise"

OUTPUT_DIR = Path(f"outputs/{TEST_NAME}/{MODEL_TAG}")    
OUTPUT_CSV = OUTPUT_DIR / f"{MODEL_TAG}_results.csv"      

PRIMING_PREFIX     = "Answer: Person "   

TOP_K = 5            
DTYPE = torch.bfloat16

try:
    with open(".hf_access_token", "r") as f:
        HF_TOKEN = f.read().strip()
except FileNotFoundError:
    HF_TOKEN = None

# ----------------------------------------------------------------------------
# Load the model and tokenizer
# ----------------------------------------------------------------------------
CACHE_PATH = "/mnt/pixstor/stoberc-lab/huggingface_cache/hub"

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
# Precompute "A" and "B" token IDs
# ----------------------------------------------------------------------------
def collect_answer_token_ids(letter):
    ids = []
    for token_id in range(len(tokenizer)):
        decoded = tokenizer.decode([token_id])
        if decoded.strip() == letter:
            ids.append(token_id)
    return ids

print("Indexing answer tokens ...")
A_IDS = collect_answer_token_ids("1")
B_IDS = collect_answer_token_ids("2")


# ----------------------------------------------------------------------------
# The Probability Mass function 
# ----------------------------------------------------------------------------
def first_token_distribution(prompt_text):
    full_text = f"{prompt_text}\n{PRIMING_PREFIX}"
    inputs = tokenizer(full_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)
    next_token_logits = outputs.logits[0, -1, :]          

    probs = torch.softmax(next_token_logits.float(), dim=-1)

    p_a = probs[A_IDS].sum().item()
    p_b = probs[B_IDS].sum().item()

    top_probs, top_ids = torch.topk(probs, TOP_K)
    top_tokens = [(tokenizer.decode([tid.item()]), prob.item()) for tid, prob in zip(top_ids, top_probs)]
    
    return p_a, p_b, top_tokens

def format_raw_response(top_tokens):
    parts = [f"{repr(tok.strip() or tok)}={prob:.6f}" for tok, prob in top_tokens]
    return "; ".join(parts)

# ----------------------------------------------------------------------------
# Main Execution Loop
# ----------------------------------------------------------------------------
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
already_done = set()
if OUTPUT_CSV.exists():
    prior = pd.read_csv(OUTPUT_CSV, dtype=str)
    already_done = set(prior["task_id"].tolist())

CSV_COLUMNS = [
    "model", "task_id", "task_type",
    "left_demo", "right_demo",
    "left_probability", "right_probability", "answer_mass",
    "prompt", "raw_response",
]

tasks = pd.read_excel(INPUT_XLSX, sheet_name=INPUT_SHEET, dtype=str)
write_header = not OUTPUT_CSV.exists()
csv_handle = open(OUTPUT_CSV, "a", newline="", encoding="utf-8")
import csv as _csv
writer = _csv.DictWriter(csv_handle, fieldnames=CSV_COLUMNS)
if write_header:
    writer.writeheader()

for _, task in tqdm(tasks.iterrows(), total=len(tasks)):
    if task["task_id"] in already_done:
        continue

    p_a, p_b, top_tokens = first_token_distribution(task["prompt"])
    answer_mass = p_a + p_b
    
    if answer_mass > 0:
        left_probability  = p_a / answer_mass
        right_probability = p_b / answer_mass
    else:
        left_probability = right_probability = float("nan")

    row = {
        "model": MODEL_TAG,
        "task_id": task["task_id"],
        "task_type": task["task_type"],
        "left_demo": task["left_demo"],
        "right_demo": task["right_demo"],
        "left_probability": round(left_probability, 6),
        "right_probability": round(right_probability, 6),
        "answer_mass": round(answer_mass, 6),
        "prompt": task["prompt"],
        "raw_response": format_raw_response(top_tokens),
    }

    writer.writerow(row)
    csv_handle.flush()

    with open(OUTPUT_DIR / f"{task['task_id']}.json", "w", encoding="utf-8") as jf:
        json.dump(row, jf, indent=2)

csv_handle.close()
print(f"Done. Results saved to: {OUTPUT_CSV}")
