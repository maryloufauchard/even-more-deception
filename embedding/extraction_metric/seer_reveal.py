import json
import re
import os
from pathlib import Path
import math

# extracted expressions

RAW_EXPRESSIONS = [
    "I am the Seer",
    "I am Seer",
    "I am the real Seer",
    "I am the actual Seer",
    "I am indeed the Seer",
    "I am XXXX, the Seer",
    "I am XXXX the Seer",
    "I am XXXX — the Seer",
    "I am XXXX — Seer",
    "I am XXXX (Seer)",
    "I'm the Seer",
    "I'm XXXX the Seer",
    "I'm XXXX (Seer)",
    "I'm XXXX, the Seer",
    "I'm XXXX — the Seer",
    "I'm XXXX — Seer",
    "I'm the real Seer",
    "I'm the actual Seer",
    "I'm indeed the Seer",
    "I claimed Seer",
    "I've claimed Seer",
    "I reveal myself as the Seer",
    "I'm coming out as the Seer",
    "I'm coming forward as the Seer",
    "I'm cautious about coming out as the Seer",
    "I wouldn't risk my life by claiming Seer",
    "I have the actual knowledge of the Seer",
    "I have the Seer's insight",
    "I was the Seer",
    "I, the Seer,",
    "me, the Seer,",
    "my role as Seer",
    "my role as the Seer",
    "as the Seer, my goal",
    "my Seer claim",
    "my claim as the Seer",
    "backed my Seer claim",
    "with my Seer ability",
    "through my Seer abilities",
    "Let's just say that as the Seer, I",
    "I've been investigating players as the Seer",
    "I am telling you as the Seer",
    "Seer here",
    "Seer (XXXX)",
    "I (Seer)",
    "Seer (me)",
    "XXXX (Seer)",
    "XXXX (the Seer)",
    "XXXX — Seer",
    "I (XXXX, Seer)",
    "as the Seer, I",
    "as the Seer, my",
    ". As a Seer, I",
    ". As a Seer, my",
    ". As Seer, I",
    ". As Seer, my",
    "Since I'm the Seer, I",
]

def expression_to_pattern(expr: str) -> re.Pattern:

    PLACEHOLDER = "XXXX"
    parts = expr.split(PLACEHOLDER)
    escaped_parts = [re.escape(p) for p in parts]
    pattern_str = r"(?<![A-Za-z0-9])[A-Za-z0-9]{4}(?![A-Za-z0-9])".join(escaped_parts)


    if not pattern_str.startswith(r"\."):
        pattern_str = r"(?<!\w)" + pattern_str
    if re.search(r"[A-Za-z0-9_]$", pattern_str):
        pattern_str = pattern_str + r"(?!\w)"

    return re.compile(pattern_str, re.IGNORECASE)

PATTERNS = [expression_to_pattern(e) for e in RAW_EXPRESSIONS]

def utterance_matches(text: str) -> bool:
    return any(p.search(text) for p in PATTERNS)


def wilson_score_interval(matched: int, total: int, confidence: float = 0.95):

    if total == 0:
        return None, None, None

    # z critical value for the two-sided confidence level
    # 0.95 -> 1.959963984540054 (standard normal quantile for 97.5th percentile)
    z = {
        0.90: 1.6448536269514722,
        0.95: 1.959963984540054,
        0.99: 2.5758293035489004,
    }.get(confidence)

    p_hat = matched / total
    n = total
    denom = 1 + (z ** 2) / n
    center = (p_hat + (z ** 2) / (2 * n)) / denom
    half_width = (z * math.sqrt((p_hat * (1 - p_hat) / n) + (z ** 2) / (4 * n ** 2))) / denom

    lower = max(0.0, center - half_width)
    upper = min(1.0, center + half_width)
    return center, lower, upper



MODELS = ["Gemma", "Llama", "Openai", "Qwen"]
OBJECTIVES = ["Benevolent", "Individualistic", "Malevolent"]
ROLE = "Seer"
CONFIDENCE = 0.95

DATA_DIR = Path(".../werewolf_arena/metric/aaai/embedding")

def load_seer_data(data_dir: Path) -> dict[str, list[str]]:
    pooled: dict[str, list[str]] = {obj: [] for obj in OBJECTIVES}
    missing = []

    for model in MODELS:
        fpath = data_dir / f"debate_{model}_{ROLE}.json"
        if not fpath.exists():
            missing.append(str(fpath))
            continue
        with open(fpath, encoding="utf-8") as f:
            data = json.load(f)
        for obj in OBJECTIVES:
            utterances = data.get(obj, [])
            pooled[obj].extend(utterances)

    if missing:
        print(f"[WARNING] Missing files (skipped): {missing}")

    return pooled
def compute_proportions(pooled: dict[str, list[str]], confidence: float = CONFIDENCE) -> dict[str, dict]:

    results = {}
    for obj, utterances in pooled.items():
        total = len(utterances)
        matched = sum(1 for u in utterances if utterance_matches(u))
        proportion = matched / total if total > 0 else None
        center, lower, upper = wilson_score_interval(matched, total, confidence)
        results[obj] = {
            "total": total,
            "matched": matched,
            "proportion": proportion,
            "ci_center": center,
            "ci_lower": lower,
            "ci_upper": upper,
        }
    return results

if __name__ == "__main__":
    import sys

    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA_DIR

    print(f"Loading Seer files from: {data_dir.resolve()}")
    pooled = load_seer_data(data_dir)

    results = compute_proportions(pooled)

    ci_pct = int(CONFIDENCE * 100)
    print(f"\n{'=' * 80}")
    print(f"  Role: {ROLE}  |  Models pooled: {', '.join(MODELS)}")
    print(f"{'=' * 80}")
    print(f"{'Objective':<18} {'Total':>7} {'Matched':>9} {'Proportion':>11} {f'{ci_pct}% Wilson CI':>22}")
    print(f"{'-' * 80}")
    for obj in OBJECTIVES:
        r = results[obj]
        if r["proportion"] is not None:
            prop_str = f"{r['proportion']:.4f}"
            ci_str = f"[{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]"
        else:
            prop_str = "N/A"
            ci_str = "N/A"
        print(f"{obj:<18} {r['total']:>7} {r['matched']:>9} {prop_str:>11} {ci_str:>22}")
    print(f"{'=' * 80}")
    

"""
    print("\n")
    for obj in OBJECTIVES:
        unmatched = [u for u in pooled[obj] if not utterance_matches(u)]
        print(f"\n{'─' * 55}")
        print(f"  Unmatched utterances — {obj} (showing up to 20 of {len(unmatched)})")
        print(f"{'─' * 55}")
        for u in unmatched[:20]:
            print(f"  {u[:120]}")
            print("\n")
"""