"""Stage 0 adapter check: same held-out prompt through the base model and
the LoRA-adapted model, side by side. Passing bar for the smoke test is
behavioral, not quantitative: the adapted model should (a) answer in the
Pyragos persona/structure, (b) apply the evacuation-first doctrine for an
understaffed high-wind wildfire, while the base model answers generically.
"""

import json
from pathlib import Path

from mlx_lm import load, generate

MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
ADAPTER = str(Path(__file__).parent / "adapters")

# Held-out combination (not in the generated grid).
PROMPT_MESSAGES = [
    {"role": "system", "content": (
        "You are Pyragos Ioanna Michaelidou, an experienced Greek fire "
        "commander with 15 years of service in the Hellenic Fire Corps."
    )},
    {"role": "user", "content": (
        "A wildfire is threatening a coastal settlement of 1200 people. "
        "Wind is 50 km/h, I have 9 firefighters and 3 fire trucks, and "
        "backup is 60 minutes away. What are your immediate actions?"
    )},
]


def run(adapter_path=None, max_tokens=300):
    model, tokenizer = load(MODEL, adapter_path=adapter_path)
    prompt = tokenizer.apply_chat_template(
        PROMPT_MESSAGES, tokenize=False, add_generation_prompt=True
    )
    return generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens)


if __name__ == "__main__":
    print("=" * 72)
    print("BASE MODEL (no adapter)")
    print("=" * 72)
    base_out = run()
    print(base_out)

    print()
    print("=" * 72)
    print("WITH STAGE-0 LoRA ADAPTER")
    print("=" * 72)
    tuned_out = run(adapter_path=ADAPTER)
    print(tuned_out)

    # Simple doctrine check: understaffed (9 < 12) + high wind (50 >= 40)
    # should trigger evacuation-first language in the tuned model.
    doctrine_hits = [kw for kw in ("evacuat", "life safety", "1-1-2") if kw in tuned_out.lower()]
    print()
    print(f"Doctrine keywords in tuned output: {doctrine_hits}")
    result = {"base": base_out, "tuned": tuned_out, "doctrine_hits": doctrine_hits}
    out_path = Path(__file__).parent / "adapter_test_output.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Saved comparison to {out_path}")
