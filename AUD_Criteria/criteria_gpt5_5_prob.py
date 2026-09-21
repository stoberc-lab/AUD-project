import os
import json
import math
import time
import csv as _csv
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from openai import OpenAI
# Transient error types we will retry on (rate limits, timeouts, transient 5xx).
from openai import (
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    InternalServerError,
)

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
MODEL_ID  = "gpt-5.5"                          
MODEL_TAG = "gpt-5.5"    
TEST_NAME = "crit_pairwise_mass"  
INPUT_XLSX  = "AUD_crit_pairwise_counterbalanced_1and2_v2.xlsx"   
INPUT_SHEET = "crit_pairwise_counterbalanced"

OUTPUT_DIR = Path(f"outputs/{TEST_NAME}/{MODEL_TAG}")
OUTPUT_CSV = OUTPUT_DIR / f"{MODEL_TAG}_results.csv"

LEFT_LABEL  = "1"
RIGHT_LABEL = "2"

TOP_LOGPROBS = 5

MAX_COMPLETION_TOKENS = 16


TEMPERATURE      = None   
SEED             = 12345  
REASONING_EFFORT = "none"   

MAX_RETRIES   = 5
RETRY_BACKOFF = 2.0   
# ----------------------------------------------------------------------------
# API key: read from a file 
# ----------------------------------------------------------------------------
def load_api_key():
    for path in (".openai_access_token", os.path.expanduser("~/.openai_access_token")):
        try:
            with open(path, "r") as f:
                key = f.read().strip()
                if key:
                    return key
        except FileNotFoundError:
            continue
    return os.environ.get("OPENAI_API_KEY")

API_KEY = load_api_key()
if not API_KEY:
    raise RuntimeError(
        "No OpenAI API key found. Put it in .openai_access_token (current dir) "
        "or ~/.openai_access_token, or set the OPENAI_API_KEY env variable."
    )

client = OpenAI(api_key=API_KEY)

# ----------------------------------------------------------------------------
# One API call, with retry on transient errors only.
# ----------------------------------------------------------------------------
def call_model(prompt_text):
    params = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": prompt_text}],
        "logprobs": True,               
        "top_logprobs": TOP_LOGPROBS,   
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
    }
    if TEMPERATURE is not None:
        params["temperature"] = TEMPERATURE
    if SEED is not None:
        params["seed"] = SEED
    if REASONING_EFFORT is not None:
        params["reasoning_effort"] = REASONING_EFFORT
    attempt = 0
    while True:
        try:
            return client.chat.completions.create(**params)
        except (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError) as e:
            attempt += 1
            if attempt > MAX_RETRIES:
                raise
            wait = RETRY_BACKOFF * (2 ** (attempt - 1))
            print(f"  transient error ({type(e).__name__}); retry {attempt}/{MAX_RETRIES} in {wait:.0f}s")
            time.sleep(wait)

# ----------------------------------------------------------------------------
# Turn one API response into (p_left, p_right, top_tokens).
# ----------------------------------------------------------------------------
def answer_distribution(response):
    choice = response.choices[0]
    if choice.logprobs is None or not choice.logprobs.content:
        return float("nan"), float("nan"), []

    first_token = choice.logprobs.content[0]
    dist = {}
    for alt in first_token.top_logprobs:
        dist[alt.token] = alt.logprob
    if first_token.token not in dist:
        dist[first_token.token] = first_token.logprob
    p_left = 0.0
    p_right = 0.0
    for token_str, logprob in dist.items():
        stripped = token_str.strip()
        if stripped == LEFT_LABEL:
            p_left += math.exp(logprob)
        elif stripped == RIGHT_LABEL:
            p_right += math.exp(logprob)

    top_tokens = sorted(
        ((tok, math.exp(lp)) for tok, lp in dist.items()),
        key=lambda x: x[1],
        reverse=True,
    )
    return p_left, p_right, top_tokens

def format_raw_response(top_tokens):
    parts = [f"{repr(tok.strip() or tok)}={prob:.6f}" for tok, prob in top_tokens]
    return "; ".join(parts)

# ----------------------------------------------------------------------------
# Main execution loop 
# --------------------------------------------------------------------------------------------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

already_done = set()
if OUTPUT_CSV.exists():
    prior = pd.read_csv(OUTPUT_CSV, dtype=str)
    already_done = set(prior["task_id"].tolist())

CSV_COLUMNS = [
    "model", "task_id", "pair_id", "order", "comparison_form",
    "left_criteria", "right_criteria",
    "left_probability", "right_probability", "answer_mass",
    "prompt", "raw_response",
]

tasks = pd.read_excel(INPUT_XLSX, sheet_name=INPUT_SHEET, dtype=str)

write_header = not OUTPUT_CSV.exists()
csv_handle = open(OUTPUT_CSV, "a", newline="", encoding="utf-8")
writer = _csv.DictWriter(csv_handle, fieldnames=CSV_COLUMNS)
if write_header:
    writer.writeheader()

for _, task in tqdm(tasks.iterrows(), total=len(tasks)):
    if task["task_id"] in already_done:
        continue

    response = call_model(task["prompt"])
    p_left, p_right, top_tokens = answer_distribution(response)

    # Three distinct cases:
    #   * p_left/p_right NaN  -> the model returned no usable logprobs; mark NaN.
    #   * answer_mass == 0    -> model answered but put no mass on "1"/"2".
    #   * answer_mass  > 0    -> normal; normalize to P(1) vs P(2).
    if math.isnan(p_left) or math.isnan(p_right):
        answer_mass = float("nan")
        left_probability = right_probability = float("nan")
    else:
        answer_mass = p_left + p_right
        if answer_mass > 0:
            left_probability  = p_left / answer_mass
            right_probability = p_right / answer_mass
        else:
            left_probability = right_probability = float("nan")

    if math.isnan(answer_mass):
        print(f"  [WARN] {task['task_id']}: no usable logprobs returned.")
    elif answer_mass < 0.5:
        print(f"  [WARN] {task['task_id']}: low answer_mass={answer_mass:.4f} "
              f"(answer tokens under-captured or model hedged).")

    row = {
        "model": MODEL_TAG,
        "task_id": task["task_id"],
        "pair_id": task["pair_id"],
        "order": task["order"],
        "comparison_form": task["comparison_form"],
        "left_criteria": task["left_criteria"],
        "right_criteria": task["right_criteria"],
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
