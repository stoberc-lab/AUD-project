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
MODEL_ID  = "gpt-5.5-2026-04-23"   # Pinned dated snapshot for a frozen,
                                   # reproducible run. Change to "gpt-5.5" only
                                   # if you want the moving alias (not
                                   # recommended mid-study).
MODEL_TAG = "gpt-5.5"              # Used only for the output folder / filename / CSV "model".

# TEST_NAME matches the Llama/Meditron scenario arm so all three models land
# under the same test folder: outputs/<TEST_NAME>/gpt-5.5/ next to
# outputs/<TEST_NAME>/llama-2-70b-hf/ and outputs/<TEST_NAME>/meditron-70b/.
TEST_NAME  = "scenario_demo(only)_first_token_1and2"

INPUT_XLSX  = "AUD_scenario_pairwise_counterbalanced_1and2.xlsx"
INPUT_SHEET = "AUD_scenario_pairwise"

OUTPUT_DIR = Path(f"outputs/{TEST_NAME}/{MODEL_TAG}")
OUTPUT_CSV = OUTPUT_DIR / f"{MODEL_TAG}_results.csv"

# The two answer surface forms we are scoring. These match the "1"/"2" labels in
# the prompts. To score an A/B-labeled workbook instead, change these to "A"/"B"
# and point INPUT_XLSX at the letter workbook (label mismatch produces near-zero
# answer mass and NaN probabilities, so keep these in sync with the prompt text).
LEFT_LABEL  = "1"
RIGHT_LABEL = "2"

# How many alternative tokens to request at the answer position. GPT-5.5 caps
# top_logprobs at 5 on the chat completions endpoint (older model docs say 20;
# requesting more than 5 on GPT-5.5 is rejected). This makes the GPT arm a
# top-5 approximation rather than a full-vocabulary sum; the answer_mass column
# is the per-row evidence of how much probability that truncation left behind.
TOP_LOGPROBS = 5

# Token budget for the completion. We only read the FIRST visible token's
# logprobs, but GPT-5.5 is a reasoning model that can spend hidden reasoning
# tokens before emitting a visible token. A small buffer (not 1) protects
# against the whole budget being consumed by reasoning and returning no visible
# token / no logprobs. With REASONING_EFFORT="none" below, reasoning is
# suppressed, so 16 is plenty.
MAX_COMPLETION_TOKENS = 16

# Optional generation controls. Set any of these to None to omit that parameter
# entirely if the model rejects it.
TEMPERATURE      = None    # None -> API default. GPT-5.x reasoning models
                           # generally reject a custom temperature, so leave None.
SEED             = 12345   # Best-effort run-to-run stability; OpenAI does not
                           # guarantee determinism even with a seed. None to omit.
REASONING_EFFORT = "none"

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

    # Retry only on transient errors; hard errors (bad params, auth, "logprobs
    # not allowed") surface immediately so you can fix them rather than looping.
    # Note: RateLimitError with code "insufficient_quota" is a billing problem,
    # not a throughput limit; retrying it just wastes the retry budget, so it is
    # re-raised immediately.
    attempt = 0
    while True:
        try:
            return client.chat.completions.create(**params)
        except RateLimitError as e:
            if getattr(e, "code", None) == "insufficient_quota" or "insufficient_quota" in str(e):
                raise   # fail fast: quota/billing issue, not transient
            attempt += 1
            if attempt > MAX_RETRIES:
                raise
            wait = RETRY_BACKOFF * (2 ** (attempt - 1))
            print(f"  transient error ({type(e).__name__}); retry {attempt}/{MAX_RETRIES} in {wait:.0f}s")
            time.sleep(wait)
        except (APITimeoutError, APIConnectionError, InternalServerError) as e:
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
    # a numeric answer across variants such as "1" and " 1", so we compare on the
    # stripped token text rather than requiring an exact-token match.
    p_left = 0.0
    p_right = 0.0
    for token_str, logprob in dist.items():
        stripped = token_str.strip()
        if stripped == LEFT_LABEL:
            p_left += math.exp(logprob)
        elif stripped == RIGHT_LABEL:
            p_right += math.exp(logprob)

    # Diagnostic: top tokens by probability, mirroring the Llama script's
    # raw_response column so all arms are inspectable the same way.
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
# Main execution loop (resume-safe; identical CSV schema to the Llama/Meditron
# scenario arm).
# ----------------------------------------------------------------------------
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Resume support: skip any task_id already present in the output CSV.
already_done = set()
if OUTPUT_CSV.exists():
    prior = pd.read_csv(OUTPUT_CSV, dtype=str)
    already_done = set(prior["task_id"].tolist())

# Same schema as the crit_demo arm, extended with the vignette-tracing columns
# (base_task_id, left_trcode, right_trcode, instance) unique to this arm.
CSV_COLUMNS = [
    "model", "task_id", "base_task_id", "task_type",
    "left_crit", "left_demo", "right_crit", "right_demo",
    "left_trcode", "right_trcode", "instance",
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
        "base_task_id": task["base_task_id"],
        "task_type": task["task_type"],
        "left_crit": task["left_crit"],
        "left_demo": task["left_demo"],
        "right_crit": task["right_crit"],
        "right_demo": task["right_demo"],
        "left_trcode": task["left_trcode"],
        "right_trcode": task["right_trcode"],
        "instance": task["instance"],
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
