"""
test_validation_llama2.py

Phase C vignette validation runner for Llama-2-70B (base model).

Task: for each row of AUD_vignette_validation_10orders.xlsx, present the
vignette with 12 lettered options (11 AUD criteria + "None of the above",
in one of 10 seeded random orderings) and ask which option best matches.

Measurement is TEXT GENERATION (greedy decoding), not probability mass:
validation is a classification accuracy check where only the modal choice
matters, unlike the graded pairwise preference arms. The first-token top-5
distribution is still recorded per trial as a diagnostic, mirroring the
raw_response column of the prob_mass scripts.

SMOKE TEST: set SMOKE_TEST = True to run only the first vignette in the
workbook (10 orderings = 10 trials), with prompts and raw continuations
printed to the console, and outputs written to a separate *_smoketest
folder so the full run's resume log is never contaminated. Set it to
False for the full 9,900-trial run.
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

SMOKE_TEST = True   # True: 1 vignette x 10 orderings, verbose. False: full run.

TEST_NAME = "vignette_validation_10orders" + ("_smoketest" if SMOKE_TEST else "")

INPUT_XLSX  = "AUD_vignette_validation_10orders.xlsx"
INPUT_SHEET = "vignette_validation"

OUTPUT_DIR = Path(f"outputs/{TEST_NAME}/{MODEL_TAG}")
OUTPUT_CSV = OUTPUT_DIR / f"{MODEL_TAG}_results.csv"

# Greedy decoding, short budget: the priming suffix inside the prompt
# ("Answer (single letter): ") positions the very next token to be a letter,
# and 10 new tokens leave room for trailing punctuation or a short echo.
MAX_NEW_TOKENS = 10
DTYPE = torch.bfloat16

TOP_K = 5   # first-token diagnostic, mirroring the prob_mass scripts

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
# Generation: greedy continuation plus a first-token top-5 diagnostic.
# ----------------------------------------------------------------------------
def generate_answer(prompt_text):
    """Greedily generate up to MAX_NEW_TOKENS after the prompt.

    Returns (generated_text, top_tokens) where top_tokens is the top-5
    (token, probability) list at the FIRST generated position. That first
    position is exactly where the answer letter should appear, so the
    diagnostic shows at a glance whether the model's mass sits on letters
    (compliant) or on digits/other text (non-compliant)."""
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,                 # greedy: deterministic given the prompt
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens (everything after the prompt).
    gen_ids = outputs.sequences[0][input_len:]
    generated_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    # First-token diagnostic: softmax over the logits of the first generated
    # position, then take the top-5 tokens by probability.
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

    generated_text, top_tokens = generate_answer(task["prompt"])
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

    # Verbose console output for the smoke test: raw continuation plus the
    # first-token diagnostic, so compliance is checkable at a glance.
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
