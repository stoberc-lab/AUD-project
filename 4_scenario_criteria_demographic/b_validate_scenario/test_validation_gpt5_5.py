"""
test_validation_gpt5_5.py

Task: for each row of AUD_vignette_validation_10orders.xlsx, present the
vignette with 12 lettered options (11 AUD criteria + "None of the above",
in one of 10 seeded random orderings) and ask which option best matches.

Measurement is TEXT GENERATION, not probability mass: validation is a
classification accuracy check where only the modal choice matters, unlike
the graded pairwise preference arms. Token-level logprobs are still
requested so the first visible token's top-5 distribution is recorded per
trial as a diagnostic, mirroring the raw_response column of the prob_mass
scripts.

SMOKE TEST: set SMOKE_TEST = True to run only the first vignette in the
workbook (10 orderings = 10 trials), with prompts and raw continuations
printed to the console, and outputs written to a separate *_smoketest
folder so the full run's resume log is never contaminated. Set it to
False for the full 9,900-trial run.
"""

import os
import json
import math
import time
import re
import csv as _csv
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from openai import OpenAI
from openai import (
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    InternalServerError,
)

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
MODEL_ID  = "gpt-5.5"   # Pinned dated snapshot, same as all other arms.
MODEL_TAG = "gpt-5.5"

SMOKE_TEST = True   # True: 1 vignette x 10 orderings, verbose. False: full run.

TEST_NAME = "vignette_validation_10orders" + ("_smoketest" if SMOKE_TEST else "")

INPUT_XLSX  = "AUD_vignette_validation_10orders.xlsx"
INPUT_SHEET = "vignette_validation"

OUTPUT_DIR = Path(f"outputs/{TEST_NAME}/{MODEL_TAG}")
OUTPUT_CSV = OUTPUT_DIR / f"{MODEL_TAG}_results.csv"

# Token budget for the visible answer: 10, matching the open-weight runners.
# REASONING_EFFORT="none" suppresses hidden reasoning tokens, so the budget is
# not silently consumed before a visible letter is emitted.
MAX_COMPLETION_TOKENS = 10

TOP_LOGPROBS = 5      # first-token diagnostic; GPT-5.5 caps top_logprobs at 5.

TEMPERATURE      = None    # GPT-5.x reasoning models generally reject a custom
                           # temperature; with REASONING_EFFORT="none" the API
                           # default applies. Leave None.
SEED             = 12345   # Best-effort run-to-run stability; OpenAI does not
                           # guarantee determinism even with a seed.
REASONING_EFFORT = "none"

# Simple retry policy for transient API errors.
MAX_RETRIES   = 5
RETRY_BACKOFF = 2.0   # base seconds; the wait doubles each attempt.

# ----------------------------------------------------------------------------
# API key: file first, env variable fallback (same pattern as the other arms).
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

    # Retry only on transient errors; hard errors (bad params, auth) surface
    # immediately. insufficient_quota is a billing problem, not a throughput
    # limit, so it is re-raised immediately rather than retried.
    attempt = 0
    while True:
        try:
            return client.chat.completions.create(**params)
        except RateLimitError as e:
            if getattr(e, "code", None) == "insufficient_quota" or "insufficient_quota" in str(e):
                raise
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
# Extract the generated text and the first-token top-5 diagnostic.
# ----------------------------------------------------------------------------
def extract_generation(response):
    """Returns (generated_text, top_tokens).

    generated_text is the visible message content. top_tokens is the top-5
    (token, probability) list at the first visible token position, which is
    exactly where the answer letter should appear. If logprobs came back
    null or empty, the diagnostic list is empty but the text is still used."""
    choice = response.choices[0]
    generated_text = choice.message.content or ""

    top_tokens = []
    if choice.logprobs is not None and choice.logprobs.content:
        first_token = choice.logprobs.content[0]
        dist = {}
        for alt in first_token.top_logprobs:
            dist[alt.token] = alt.logprob
        if first_token.token not in dist:
            dist[first_token.token] = first_token.logprob
        top_tokens = sorted(
            ((tok, math.exp(lp)) for tok, lp in dist.items()),
            key=lambda x: x[1],
            reverse=True,
        )
    return generated_text, top_tokens


def format_raw_response(top_tokens):
    parts = [f"{repr(tok.strip() or tok)}={prob:.6f}" for tok, prob in top_tokens]
    return "; ".join(parts)

# ----------------------------------------------------------------------------
# Parsing: extract the chosen letter from the continuation.
# (Identical rule to the open-weight runners, so parse behavior is one
# instrument across all three models.)
# ----------------------------------------------------------------------------
def parse_letter(generated_text):
    """Extract the answer letter (A-L, uppercase only) from the continuation.

    Two-tier rule, applied in order:
      1. "start_letter": the continuation begins with the letter, either
         punctuated ("C.", "(I)", "K:"), standing alone ("C"), or, for
         letters other than A and I, followed by whitespace ("C experienced
         ..."). A and I require punctuation or standing alone because they
         are also English words ("I think ...", "A strong ...").
      2. "keyword_letter": fallback for prose continuations, matching an
         uppercase letter right after "answer"/"option" (e.g. "The answer
         is C"). The letter class is uppercase-only throughout so the
         English words "a" and "I" inside ordinary prose cannot match.

    Returns (letter or None, parse_status)."""
    text = generated_text.strip()
    if not text:
        return None, "empty"

    # Tier 1a: letter followed by punctuation, e.g. "C.", "(I)", "K:", "B) ...".
    m = re.match(r"^\(?([A-L])[\.\):,]", text)
    if m:
        return m.group(1), "start_letter"

    # Tier 1b: the continuation is the bare letter and nothing else.
    m = re.match(r"^\(?([A-L])\)?$", text)
    if m:
        return m.group(1), "start_letter"

    # Tier 1c: bare letter followed by whitespace (e.g. "C experienced ...").
    # "A" and "I" are excluded here because they are English words: a bare
    # "I think ..." or "A strong ..." continuation is prose, not an answer.
    # Punctuated forms of A and I are still caught by tier 1a.
    m = re.match(r"^([B-HJ-L])\s", text)
    if m:
        return m.group(1), "start_letter"

    m = re.search(r"(?i:answer|option)\s*(?:is|:)?\s*\(?([A-L])\b", text)
    if m:
        return m.group(1), "keyword_letter"

    return None, "unparseable"

# ----------------------------------------------------------------------------
# Main execution loop (resume-safe; same skip/append/JSON-dump pattern as the
# pairwise runners).
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

# Smoke test: restrict to the first vignette in the workbook (its 10 orderings).
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

    response = call_model(task["prompt"])
    generated_text, top_tokens = extract_generation(response)
    predicted_letter, parse_status = parse_letter(generated_text)

    # Map the predicted letter back to its option code (C1-C11 or NOTA) using
    # the per-trial letter columns baked into the workbook; no prompt re-parsing.
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

    # Verbose console output for the smoke test.
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
