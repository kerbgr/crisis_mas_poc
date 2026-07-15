"""Stage 0 seed dataset generator.

Produces a small, rule-based synthetic dataset for the PIPELINE SMOKE TEST
(data -> MLX LoRA -> adapter -> serve -> AEGIS). It exists to prove the
loop runs end-to-end on local Apple Silicon hardware -- it is NOT training
data for a deployable model.

Honesty contract:
- Every answer is generated from explicit decision rules distilled from the
  5 hand-written seed examples in
  ../../data_collection/dataset_templates/firefighter_qa.jsonl
  (evacuation-first when understaffed, ammonia UN1005 doctrine, class A/B
  agent selection). No expert has validated these outputs.
- Factual constants (IDLH values) are kept consistent with
  tools/deployment/detect_safety_failures.py's KNOWN_HAZMAT_IDLH_PPM table.
- Stage 1 replaces this file's output with expert-reviewed data
  (see ../../PROJECT_PLAN.md).

Output: train.jsonl / valid.jsonl / test.jsonl in mlx-lm chat format
({"messages": [...]}), written to ./data/.
"""

import itertools
import json
import random
from pathlib import Path

SYSTEM = (
    "You are Pyragos Ioanna Michaelidou, an experienced Greek fire commander "
    "with 15 years of service in the Hellenic Fire Corps. You specialize in "
    "wildfire suppression, urban firefighting, and HAZMAT response."
)

# Consistent with tools/deployment/detect_safety_failures.py
IDLH_PPM = {"ammonia": 300, "carbon monoxide": 1200, "chlorine": 10}


def wildfire_examples():
    """Family A: wildfire threatening a settlement. Doctrine: with <12
    personnel or wind >= 40 km/h, evacuation-first; otherwise structured
    attack with exposure protection."""
    examples = []
    grid = itertools.product(
        [("village", 500), ("hamlet", 80), ("town district", 3000)],
        [15, 40, 60],          # wind km/h
        [8, 16, 40],           # personnel
        [(2, 45), (4, 20), (8, 10)],  # (trucks, backup ETA minutes)
    )
    for (place, pop), wind, crew, (trucks, eta) in grid:
        q = (
            f"A wildfire is threatening a {place} of {pop} people. Wind is "
            f"{wind} km/h, I have {crew} firefighters and {trucks} fire "
            f"trucks, and the nearest backup is {eta} minutes away. What are "
            f"your immediate actions?"
        )
        evacuate_first = crew < 12 or wind >= 40
        if evacuate_first:
            a = (
                "With "
                f"{'only ' if crew < 12 else ''}{crew} personnel and "
                f"{wind} km/h winds, direct attack is "
                f"{'too dangerous and ineffective' if wind >= 40 else 'not sustainable'}. "
                "Life safety comes first.\n\n"
                "**Immediate (0-5 minutes)**:\n"
                f"1. Order evacuation of the {place}, prioritizing elderly, "
                "children, and mobility-impaired residents\n"
                "2. Request air support and all available ground units via "
                "the 1-1-2 coordination center\n"
                "3. Establish evacuation routes away from the wind direction\n\n"
                "**Short-term (5-15 minutes)**:\n"
                "4. Set a rally point upwind with medical screening\n"
                f"5. Use the {trucks} trucks for transport of vulnerable "
                "residents, not for suppression\n"
                "6. Assign a pair of firefighters to a final sweep\n\n"
                "**Do NOT** attempt direct attack with insufficient "
                "resources. Structures can be rebuilt; lives cannot."
            )
        else:
            a = (
                f"With {crew} personnel, {trucks} trucks, and moderate wind "
                f"({wind} km/h), a structured defense is viable while backup "
                f"arrives in {eta} minutes.\n\n"
                "**Immediate (0-5 minutes)**:\n"
                "1. Establish incident command and a safety officer\n"
                "2. Anchor the attack from a secure flank, never the fire head\n"
                "3. Alert residents for precautionary evacuation readiness\n\n"
                "**Short-term (5-15 minutes)**:\n"
                "4. Protect exposures on the threatened edge of the "
                f"{place}\n"
                "5. Create a wet line between the fire front and the first "
                "structures\n"
                "6. Stage arriving backup upwind with a clear escape route\n\n"
                "Maintain LCES (Lookouts, Communications, Escape routes, "
                "Safety zones) at all times."
            )
        examples.append(_chat(q, a))
    return examples


def hazmat_examples():
    """Family B: ammonia release doctrine (UN1005): hot zone >= 500 m (1 km
    with wind), upwind approach, Level A protection, no direct water stream,
    shelter-in-place downwind."""
    examples = []
    for tons, where, windy in itertools.product(
        [5, 20], ["a highway near Elefsina", "an industrial site in Aspropyrgos", "the port of Piraeus"], [True, False]
    ):
        q = (
            f"A tanker carrying {tons} tons of anhydrous ammonia has "
            f"overturned at {where}"
            f"{' with noticeable wind' if windy else ''}. "
            "There is a visible vapor cloud. What is your response plan?"
        )
        zone = "1 km (wind present)" if windy else "500 m minimum"
        a = (
            "Ammonia (UN1005) is toxic and corrosive; this requires Level A "
            "HAZMAT response and multi-agency coordination.\n\n"
            "**Immediate (0-10 minutes)**:\n"
            f"1. Establish a hot zone of {zone} and deny entry\n"
            "2. Evacuate or shelter-in-place downwind areas (close windows, "
            "stop HVAC)\n"
            "3. Request the HAZMAT specialist unit and decontamination "
            "equipment\n"
            "4. Alert EKAB for inhalation casualties\n\n"
            "**Scene management**:\n"
            "5. Approach only from upwind, uphill if possible\n"
            f"6. Full Level A protection with SCBA -- concentrations may "
            f"exceed the IDLH of {IDLH_PPM['ammonia']} ppm\n"
            "7. Do NOT apply a direct water stream to the leak; after "
            "specialists arrive, water fog may knock down the vapor cloud\n"
            "8. Prepare for a long-duration incident and coordinate with the "
            "site's industrial fire brigade if available."
        )
        examples.append(_chat(q, a))
    return examples


def structure_fire_examples():
    """Family C: structure fire. Doctrine: confirmed trapped occupants ->
    rescue-first offensive posture with 2-in/2-out; nobody inside and weak
    resources -> defensive."""
    examples = []
    for building, trapped, crew in itertools.product(
        ["a 5-story apartment block", "a warehouse", "a primary school (out of hours)"],
        ["two people reported trapped on an upper floor", "no one reported inside", "unknown occupancy"],
        [6, 12, 24],
    ):
        q = (
            f"There is a working fire in {building} with {trapped}. "
            f"I have {crew} firefighters on scene. What is your plan?"
        )
        rescue = trapped.startswith("two") or trapped.startswith("unknown")
        if rescue and crew >= 12:
            a = (
                "Rescue is the priority; conditions permit an offensive "
                "interior attack.\n\n"
                "1. Assign a primary search team with a charged hose line "
                "backing them\n"
                "2. Enforce two-in/two-out with a rapid intervention team "
                "standing by\n"
                "3. Coordinate ventilation with the attack -- no uncontrolled "
                "openings\n"
                "4. Establish water supply and a second line to protect the "
                "stairwell (the occupants' and crews' egress route)\n"
                "5. Continuous accountability checks and air management."
            )
        elif rescue:
            a = (
                f"With only {crew} personnel, a full interior operation is "
                "not yet safe, but rescue remains the priority.\n\n"
                "1. Request additional alarms immediately\n"
                "2. Attempt rescue only via the safest, shortest path "
                "(ladder access to the reported location) with a hose line\n"
                "3. Do not commit crews beyond the reach of your two-in/"
                "two-out capability\n"
                "4. Protect egress routes; do not ventilate ahead of water\n"
                "5. Transition to full offensive operations as resources "
                "arrive."
            )
        else:
            a = (
                "With no life hazard reported, risk little to save little.\n\n"
                "1. Confirm evacuation with the owner/keyholder\n"
                "2. Choose posture by conditions and resources: "
                f"{'offensive attack is supportable with ' + str(crew) + ' personnel' if crew >= 12 else 'defensive exterior attack -- ' + str(crew) + ' personnel is insufficient for interior work'}\n"
                "3. Protect exposures and cut off spread\n"
                "4. Establish collapse zones if the structure is compromised\n"
                "5. Reassess continuously; upgrade or downgrade posture as "
                "conditions change."
            )
        examples.append(_chat(q, a))
    return examples


def factual_examples():
    """Family D: short factual QA anchored to the same constants used by the
    deployment safety checker."""
    facts = [
        ("What is the IDLH for ammonia?",
         f"The IDLH (Immediately Dangerous to Life or Health) for ammonia is {IDLH_PPM['ammonia']} ppm. "
         "The odor threshold is far lower (~5 ppm), so smelling it strongly does not mean the "
         "concentration is survivable -- treat any strong ammonia smell as a warning to withdraw upwind."),
        ("What is the IDLH for carbon monoxide?",
         f"The IDLH for carbon monoxide is {IDLH_PPM['carbon monoxide']} ppm. CO is odorless, so metering "
         "is the only reliable protection; SCBA is mandatory in any CO environment approaching this level."),
        ("What is the IDLH for chlorine?",
         f"The IDLH for chlorine is {IDLH_PPM['chlorine']} ppm -- far lower than for ammonia or CO. Even "
         "small chlorine releases are immediately dangerous; full vapor protection is required."),
        ("What extinguishing agent should I use on a Class B fire?",
         "Class B fires involve flammable liquids (gasoline, diesel, solvents). Use foam (AFFF), CO2, or "
         "dry chemical. NEVER use a water stream -- it spreads the burning liquid and worsens the fire."),
        ("What is the difference between Class A and Class B fires?",
         "Class A fires burn ordinary combustibles (wood, paper, cloth) and are extinguished primarily by "
         "cooling with water or foam. Class B fires burn flammable liquids and must be smothered "
         "(foam, CO2, dry chemical); water streams spread them."),
        ("What does LCES stand for in wildfire operations?",
         "LCES stands for Lookouts, Communications, Escape routes, and Safety zones -- the minimum safety "
         "system that must be in place before crews engage a wildfire. If any element is missing, "
         "disengage."),
        ("When should I choose a defensive posture at a structure fire?",
         "Choose defensive operations when there is no savable life hazard and the risk to crews outweighs "
         "what can be saved: advanced structural involvement, collapse indicators, insufficient personnel "
         "for two-in/two-out, or inadequate water supply. Risk a lot to save a lot, risk little to save "
         "little, risk nothing for what is already lost."),
        ("What is two-in/two-out?",
         "Two-in/two-out is the rule that crews entering an IDLH atmosphere must work in teams of at least "
         "two, with at least two equipped firefighters outside ready to rescue them. It is a minimum "
         "condition for interior offensive operations."),
    ]
    return [_chat(q, a) for q, a in facts]


def _chat(user, assistant):
    return {"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]}


def main(out_dir=None, seed=42):
    rng = random.Random(seed)
    examples = (
        wildfire_examples()
        + hazmat_examples()
        + structure_fire_examples()
        + factual_examples()
    )
    rng.shuffle(examples)

    n = len(examples)
    n_valid = max(8, n // 10)
    n_test = max(6, n // 15)
    splits = {
        "test": examples[:n_test],
        "valid": examples[n_test:n_test + n_valid],
        "train": examples[n_test + n_valid:],
    }

    out = Path(out_dir) if out_dir else Path(__file__).parent / "data"
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        path = out / f"{name}.jsonl"
        with open(path, "w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows)} examples -> {path}")
    print(f"total: {n} examples (synthetic, unvalidated -- Stage 0 smoke test only)")


if __name__ == "__main__":
    main()
