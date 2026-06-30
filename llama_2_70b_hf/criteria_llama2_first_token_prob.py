import os
import json
from pathlib import Path
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
MODEL_ID   = "meta-llama/Llama-2-70b-hf"   
MODEL_TAG  = "llama-2-70b-hf"              
TEST_NAME  = "crit_pairwise_generation" # Changed folder name to keep results separate

INPUT_XLSX = "AUD_crit_pairwise_counterbalanced.xlsx"   
INPUT_SHEET = "crit_pairwise_counterbalanced"

OUTPUT_DIR = Path(f"outputs/{TEST_NAME}/{MODEL_TAG}")    
OUTPUT_CSV = OUTPUT_DIR / f"{MODEL_TAG}_results.csv"      

ANSWER_INSTRUCTION = "Answer with only a single letter, A or B."
PRIMING_PREFIX     = "Answer: "   

DTYPE = torch.bfloat16

# Read the token from the physical file
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
# The Generation Function
# ----------------------------------------------------------------------------
def generate_and_get_logprob(prompt_text):
    full_text = f"{prompt_text}\n{ANSWER_INSTRUCTION}\n{PRIMING_PREFIX}"
    inputs = tokenizer(full_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        # Force the model to generate exactly 1 token
        outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            temperature=1.0,      # Strict temperature control
            do_sample=False,      # Greedy decoding to get the top choice
            return_dict_in_generate=True,
            output_scores=True
        )

    # 1. Figure out what token the model actually generated
    generated_token_id = outputs.sequences[0, -1]
    generated_text = tokenizer.decode(generated_token_id).strip()

    # 2. Extract the log probability for that specific token
    logits = outputs.scores[0] # The raw math for the generated step
    logprobs = F.log_softmax(logits.float(), dim=-1)
    token_logprob = logprobs[0, generated_token_id].item()

    return generated_text, token_logprob

# ----------------------------------------------------------------------------
# Main Execution Loop
# ----------------------------------------------------------------------------
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
already_done = set()
if OUTPUT_CSV.exists():
    prior = pd.read_csv(OUTPUT_CSV, dtype=str)
    already_done = set(prior["task_id"].tolist())

CSV_COLUMNS = [
    "model", "task_id", "pair_id", "order", "comparison_form",
    "left_criteria", "right_criteria",
    "generated_answer", "generated_logprob",
    "prompt"
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

    generated_text, token_logprob = generate_and_get_logprob(task["prompt"])

    row = {
        "model": MODEL_TAG,
        "task_id": task["task_id"],
        "pair_id": task["pair_id"],
        "order": task["order"],
        "comparison_form": task["comparison_form"],
        "left_criteria": task["left_criteria"],
        "right_criteria": task["right_criteria"],
        "generated_answer": generated_text,
        "generated_logprob": round(token_logprob, 6),
        "prompt": task["prompt"],
    }

    writer.writerow(row)
    csv_handle.flush()

    with open(OUTPUT_DIR / f"{task['task_id']}.json", "w", encoding="utf-8") as jf:
        json.dump(row, jf, indent=2)

csv_handle.close()
print(f"Done. Results saved to: {OUTPUT_CSV}")