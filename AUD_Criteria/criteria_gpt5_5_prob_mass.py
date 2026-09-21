"""
criteria_gpt5_5_prob_mass.py

ChatGPT (gpt-5.5) arm of the AUD Diagnostic Transitivity Project, Step 1a
(criteria-vs-criteria pairwise severity comparison).

This is the OpenAI analog of criteria_llama2_prob_mass.py. Its job is the same:
for each pairwise prompt, read the model's probability of answering "1" versus
"2", report the normalized preference P(1) vs P(2), and report the total answer
mass. The output CSV columns are identical to the Llama prob_mass arm so both
arms can be analyzed with the same downstream R code.

IMPORTANT METHOD DIFFERENCE vs the open-weight scripts
------------------------------------------------------
For Llama-2 / Meditron we run one forward pass and sum softmax mass over the
ENTIRE vocabulary for every token id that renders as "1"/"2". The OpenAI API
never exposes the full distribution, so here we instead:
  * ask the model to generate one visible token,
  * request logprobs with top_logprobs=20, and
  * sum the probability of the answer tokens that appear in that returned
    top-20 list, then normalize.
For a bare "1"/"2" under a "respond with only a number" instruction, both digits
are effectively always inside the top-20, so this reproduces the same estimator.
The answer_mass column lets you verify per row that little or no probability
fell outside the captured tokens (answer_mass close to 1.0 means a clean read).

Also note: chat completions does not let us teacher-force a priming prefix and
then read the continuation distribution, so there is no PRIMING_PREFIX here. The
decision rides on the model's first generated token, which is why the prompt's
"respond with only the number, 1 or 2" instruction is load-bearing on this arm.
"""

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
MODEL_ID  = "gpt-5.5"     # OpenAI model. Pin a dated snapshot here for a frozen,
                          # reproducible run once you know the exact snapshot id.
MODEL_TAG = "gpt-5.5"     # Used only for the output folder / filename / CSV "model".
TEST_NAME = "crit_pairwise_mass"   # Same test name as the Llama prob_mass arm.

INPUT_XLSX  = "AUD_crit_pairwise_counterbalanced_1and2_v2.xlsx"   # number-format file
INPUT_SHEET = "crit_pairwise_counterbalanced"

OUTPUT_DIR = Path(f"outputs/{TEST_NAME}/{MODEL_TAG}")
OUTPUT_CSV = OUTPUT_DIR / f"{MODEL_TAG}_results.csv"

# The two answer surface forms we are scoring. To make the A/B letter version,
# change these to "A" and "B" and point INPUT_XLSX at the letter workbook.
LEFT_LABEL  = "1"
RIGHT_LABEL = "2"

# How many alternative tokens to request at the answer position. 20 is the
# current OpenAI maximum for top_logprobs on the chat completions endpoint.
TOP_LOGPROBS = 5

# Token budget for the completion. We only read the FIRST visible token's
# logprobs, but gpt-5.5 can spend hidden reasoning tokens before it emits a
# visible token. A small buffer (not 1) protects against the whole budget being
# consumed by reasoning and returning no visible token / no logprobs.
MAX_COMPLETION_TOKENS = 16

# Optional generation controls. Set any of these to None to omit that parameter
# entirely if the model rejects it (some GPT-5 variants only accept default
# temperature, for example). If a call 400s naming one of these, set it to None.
TEMPERATURE      = None    # None -> API default. Set 0.0 for greedy if accepted.
SEED             = 12345   # For run-to-run stability where supported. None to omit.
REASONING_EFFORT = "none"    # "minimal"/"low" to suppress reasoning if supported;
                           # None to omit the parameter.

# Simple retry policy for transient API errors.
MAX_RETRIES   = 5
RETRY_BACKOFF = 2.0   # base seconds; the wait doubles each attempt.

# ----------------------------------------------------------------------------
# API key: read from a file first (mirrors the .hf_access_token pattern used by
# the open-model scripts), then fall back to the OPENAI_API_KEY env variable.
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
    # Assemble the request. Optional params are only included when not None, so a
    # model that rejects one of them still works once you set that constant to None.
    params = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": prompt_text}],
        "logprobs": True,               # ask for token log-probabilities
        "top_logprobs": TOP_LOGPROBS,   # ask for the top-N alternatives per token
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
    }
    if TEMPERATURE is not None:
        params["temperature"] = TEMPERATURE
    if SEED is not None:
        params["seed"] = SEED
    if REASONING_EFFORT is not None:
        params["reasoning_effort"] = REASONING_EFFORT

    # Retry only on transient errors; hard errors (bad params, auth) surface
    # immediately so you can fix them rather than silently looping.
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

    # Guard: the model must have returned token-level logprobs. If logprobs came
    # back null (unsupported model) or no visible token was emitted (budget eaten
    # by reasoning), we cannot score this row; return NaN so it is flagged.
    if choice.logprobs is None or not choice.logprobs.content:
        return float("nan"), float("nan"), []

    first_token = choice.logprobs.content[0]

    # Build {token_string: logprob} for the first generated position.
    # top_logprobs holds the top-N alternatives; the sampled token is normally
    # among them, but we add it explicitly if missing so no mass is dropped.
    dist = {}
    for alt in first_token.top_logprobs:
        dist[alt.token] = alt.logprob
    if first_token.token not in dist:
        dist[first_token.token] = first_token.logprob

    # Sum probability over every surface form of each answer. OpenAI often splits
    # a numeric/letter answer across variants such as "1" and " 1", so we compare
    # on the stripped token text rather than requiring an exact-token match.
    p_left = 0.0
    p_right = 0.0
    for token_str, logprob in dist.items():
        stripped = token_str.strip()
        if stripped == LEFT_LABEL:
            p_left += math.exp(logprob)
        elif stripped == RIGHT_LABEL:
            p_right += math.exp(logprob)

    # Diagnostic: atop tokens by probability, mirroring the Llama script's
    # raw_response column so both arms are inspectable the sme way.
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
# Main execution loop (resume-safe; identical CSV schema to the Llama arm).
# ----------------------------------------------------------------------------
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Resume support: skip any task_id already present in the output CSV.
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

    # Quality flag surfaced during the run so bad rows are noticed immediately.
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