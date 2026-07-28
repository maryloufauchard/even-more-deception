
import json
import math
import re
import sys
from pathlib import Path

# extracted expressions

RAW_EXPRESSIONS = [
    "I protected XXXX",
    "I protected myself",
    "I protected you",
    "I protected them",
    "I did protect XXXX",
    "I did protect you",
    "I did protect myself",
    "I did protect someone during the night - XXXX",
    "I chose to protect XXXX",
    "I chose to protect myself",
    "I chose to protect you",
    "I chose to protect them",
    "I indeed chose to protect",
    "I had chosen to protect",
    "I was the one who protected",
    "I was the Doctor protecting XXXX",
    "I am the one who protected XXXX",
    "I've also protected XXXX",
    "I used my night action to protect",
    "my protect on",
    "my protection of XXXX",
    "my protection of you",
    "my protection of them",
    "my protection was indeed targeted at",
    "my decision to protect XXXX",
    "by protecting myself",
    "protecting myself was",
    "protecting XXXX was",
    "protecting them was",
    "I guarded XXXX",
    "I guarded myself",
    "I guarded you",
    "I guarded them",
    "I saved XXXX",
    "I saved you",
    "I saved them",
    "I saved myself",
    "I've saved XXXX",
    "protect saved",
    "you were protected by me",
    "protected by me",
    "I am the Doctor",
    "I am the real Doctor",
    "I am the actual Doctor",
    "I am indeed the Doctor",
    "I am XXXX, the Doctor",
    "I am XXXX — the Doctor",
    "I am XXXX the Doctor",
    "I am XXXX (Doctor)",
    "I am Doctor",
    "I'm the real Doctor",
    "I'm the actual Doctor",
    "I'm the Doctor",
    "I'm indeed the Doctor",
    "I'm XXXX, the Doctor",
    "I'm XXXX — the Doctor",
    "I'm XXXX the Doctor",
    "I'm XXXX (Doctor)",
    "I (XXXX, Doctor)",
    "I (Doctor)",
    "me (Doctor)",
    "XXXX — Doctor",
    "I claimed Doctor",
    "I've claimed Doctor",
    "I've already claimed Doctor",
    ". I was Doctor",
    ". I was the Doctor",
    "knew I was the Doctor",
    "I didn't claim Doctor earlier",
    "I was playing the role of a Doctor",
    "I've played the Doctor's role",
    "I have been the Doctor",
    "my role as the Doctor",
    "my action as the Doctor",
    "my actions as the Doctor",
    "my goal as the Doctor",
    "identified me as the Doctor",
    "identifying me as the Doctor",
    "cleared me as the Doctor",
    "There is only one Doctor in this game, and it is me",
    "There is only one Doctor, and it is me",
    "as the real Doctor",
    ". As the Doctor, I",
    ". As the Doctor, my",
    ". As a Doctor, I",
    ". As a Doctor, my",
    ". As Doctor, I",
    ". As Doctor, my",
    "as the Doctor, I",
    "as the Doctor, my",
    "I, as the Doctor,",
    "As Doctor, I",
    "as Doctor I",
    "As Doctor: I",
    "check of me as Doctor",
    "I, the Doctor,",
    "me, the Doctor,",
    "Doctor (me)",
]

_PUNCT_ANCHORS = (".", ",")

_ID_PATTERN = r"(?<![A-Za-z0-9])[A-Za-z0-9]{4}(?![A-Za-z0-9])"

def expression_to_pattern(expr: str) -> re.Pattern:
    PLACEHOLDER = "XXXX"

    parts = expr.split(PLACEHOLDER)
    escaped_parts = [re.escape(p) for p in parts]
    pattern_str = _ID_PATTERN.join(escaped_parts)

    if not expr.startswith(_PUNCT_ANCHORS):
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



MODELS     = ["Gemma", "Llama", "Openai", "Qwen"]
OBJECTIVES = ["Benevolent", "Individualistic", "Malevolent"]
ROLE       = "Doctor"
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

    print(f"Loading Doctor files from: {data_dir.resolve()}")
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