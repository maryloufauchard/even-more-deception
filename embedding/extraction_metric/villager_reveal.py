import json
import re
import sys
from pathlib import Path
import math

# extracted expressions

RAW_EXPRESSIONS = [
    "I am a Villager",
    "I am an innocent Villager",
    "I am a plain Villager",
    "I am just a plain Villager",
    "I am the Villager",
    "I am a simple Villager",
    "I am a confirmed Villager",
    "I am a clean Villager",
    "I am the only Villager",
    "I am the only verified Villager",
    "I am indeed a Villager",
    "I'm a Villager",
    "I'm a plain Villager",
    "I'm just a plain Villager",
    "I'm the Villager",
    "I'm a simple Villager",
    "I'm confirmed Villager",
    "I'm the only Villager",
    "I'm the only verified Villager",
    "I'm indeed a Villager",
    "I'm a clean Villager",
    "I'm XXXX, a Villager",
    "I'm XXXX, plain Villager",
    "I'm XXXX (Villager)",
    "I'm XXXX — Villager",
    "I'm XXXX — a plain Villager",
    "I am XXXX, a Villager",
    "I am XXXX, plain Villager",
    "I am XXXX — Villager",
    "I am XXXX — a plain Villager",
    "I am XXXX (Villager)",
    "I'm glad to be confirmed as a Villager",
    "I had no night action (Villager)",
    "I claimed plain Villager",
    "my role as a Villager",
    "my innocence as a Villager",
    "confirmed my role as a Villager",
    "ME as a Villager",
    "XXXX — Villager",
    "Villager — I",
    "found me Villager",
    "found me to be a Villager",
    "they checked me = VILLAGER",
    "they checked me (and found Villager)",
    "checking me as Villager",
    "confirmed me as a Villager",
    "As a Villager, I",
    "as a Villager, my",
    "As Villagers, our",
    "as Villager, I",
    "as Villager, my",
    "as a legitimate Villager",
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
ROLE = "Villager"
CONFIDENCE = 0.95 
DATA_DIR = Path(".../werewolf_arena/metric/aaai/embedding")

def load_data(data_dir: Path) -> dict[str, list[str]]:

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
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA_DIR

    print(f"Loading Villager files from: {data_dir.resolve()}")
    pooled = load_data(data_dir)

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