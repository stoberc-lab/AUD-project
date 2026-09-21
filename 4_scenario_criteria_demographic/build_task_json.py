"""Build the psychscanner task JSON for AUD scenario generation.

Reads the 99-row atomic stimulus file (crit | demo | prompt) and produces:
  1. aud_scenario_gen_task.json  - psychscanner task file with 990 trials
     (99 seeds x 10 instances; severity schedule: 3 low, 4 medium, 3 high)
  2. aud_scenario_gen_mapping.csv - sidecar mapping trcode -> crit, demo,
     severity, instance, description (for merging with output CSV later)

The task JSON mirrors the structure of psychscanner's examples/tasks/vviq16.json:
top-level keys tasktype, taskname, instructions, contexts, contexts_id,
context_present, items, parser, chain_type. Each item is {trid: [{trcode, stimulus}]}.
psychscanner looks up trcode.split("_")[0] in contexts_id unconditionally,
so every trcode is prefixed "G_" and contexts_id is ["G"].
"""
import json
import re
import pandas as pd

INPUT_XLSX = "AUD_crit_demo_prompt.xlsx"
SHEET = "crit_demo"
OUT_JSON = "aud_scenario_gen_task.json"
OUT_MAP = "aud_scenario_gen_mapping.csv"

# Severity schedule: identical for every seed. Instances 1-10 are medium.
SEVERITY_SCHEDULE = ["medium"] * 10

# Final approved instruction wording (Version 1, names allowed).
# {a_an} resolves to "a" or "an"; {description} is the row's prompt text;
INSTRUCTION_TEMPLATE = (
    "Write a short case-study vignette of exactly 3 to 4 sentences describing {a_an} {description}. Follow these rules: "
    "(1) depict only this one alcohol-related symptom, shown at a medium level of intensity or frequency; "
    "(2) do not mention, imply, or explicitly rule out any other alcohol-related symptoms, consequences, or DSM-5 diagnoses;"
    "(3) state the person's demographic characteristics exactly as given, without adding any other demographic details (e.g., age, gender, race, etc.);"
    "(4) refer to the individual as '(an/the) individual' and do not use a name;"
    "(5) do not mention or imply whether the individual is seeking treatment (or not seeking treatment) for this symptom;"
    "(6) include that the symptom is problematic/distressing for the individual;"
    "Output the vignette only, with no preamble or commentary."
)

def a_or_an(phrase: str) -> str:
    return "an" if phrase[0].lower() in "aeiou" else "a"

def main() -> None:
    df = pd.read_excel(INPUT_XLSX, sheet_name=SHEET)
    assert list(df.columns) == ["crit", "demo", "prompt"], df.columns
    assert len(df) == 99, f"expected 99 rows, got {len(df)}"

    items = {}
    mapping_rows = []
    for _, row in df.iterrows():
        crit, demo, desc = row["crit"], row["demo"], row["prompt"]
        for i, severity in enumerate(SEVERITY_SCHEDULE, start=1):
            # trcode encodes all metadata; prefix G satisfies context lookup.
            trcode = f"G_{crit}-{demo}-{severity}-{i:02d}"
            stimulus = INSTRUCTION_TEMPLATE.format(
                a_an=a_or_an(desc), description=desc, severity=severity
            )
            items[trcode] = [{"trcode": trcode, "stimulus": stimulus}]
            mapping_rows.append({
                "trcode": trcode, "crit": crit, "demo": demo,
                "severity": severity, "instance": i, "description": desc,
            })

    task = {
        "tasktype": "generation",
        "taskname": "aud_scenario_gen",
        "instructions": {
            "definition": [
                "You will write short case-study vignettes.",
                "Each trial contains its own complete instructions.",
            ]
        },
        "contexts": [""],
        "contexts_id": ["G"],
        "context_present": False,
        "items": items,
        "parser": None,
        "chain_type": "trial",
    }

    with open(OUT_JSON, "w") as f:
        json.dump(task, f, indent=2)
    pd.DataFrame(mapping_rows).to_csv(OUT_MAP, index=False)

    print(f"Wrote {OUT_JSON}: {len(items)} trials")
    print(f"Wrote {OUT_MAP}: {len(mapping_rows)} mapping rows")
    # Show one example of each severity level for eyeballing
    for code in ["G_C1-D1-medium-01", "G_C1-D9-medium-05", "G_C5-D7-medium-10"]:
        print("\n---", code)
        print(items[code][0]["stimulus"])

if __name__ == "__main__":
    main()
