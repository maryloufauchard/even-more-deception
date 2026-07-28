import json
import re
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(".../werewolf_arena/logs_adversarial")

# session_{Model}_{Role}_{Objective}_None_{seed}
FOLDER_RE = re.compile(
    r"^session_(?P<model>[^_]+)_(?P<role>[^_]+)_(?P<objective>[^_]+)_None_(?P<seed>\d+)$"
)

PROTECT_RE = re.compile(r"I chose to protect (\S+)")


def get_doctor_name(data: dict) -> str | None:
    """Extract the doctor's own player name from the JSON."""
    doctor = data.get("doctor", {})
    return doctor.get("name")

def get_protection_observations(data: dict) -> list[str]:
    """Return all doctor observation strings."""
    return data.get("doctor", {}).get("observations", [])

def parse_protections(observations: list[str], doctor_name: str) -> tuple[int, int]:
    total = 0
    self_count = 0
    for obs in observations:
        m = PROTECT_RE.search(obs)
        if m:
            total += 1
            protected = m.group(1).rstrip(".,;")
            if protected == doctor_name:
                self_count += 1
    return self_count, total


counts: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))

folders_processed = 0
folders_skipped = 0 
for folder in sorted(BASE_DIR.iterdir()):
    if not folder.is_dir():
        continue
    m = FOLDER_RE.match(folder.name)
    if not m:
        folders_skipped += 1
        continue

    role = m.group("role")
    if role != "Doctor":
        continue

    model = m.group("model")
    objective = m.group("objective")

    game_file = folder / "game_complete.json"
    if not game_file.exists():
        folders_skipped += 1
        continue

    try:
        with open(game_file) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  WARNING: could not read {game_file}: {e}")
        folders_skipped += 1
        continue

    doctor_name = get_doctor_name(data)
    if not doctor_name:
        print(f"  WARNING: no doctor name in {folder.name}")
        folders_skipped += 1
        continue

    observations = get_protection_observations(data)
    self_n, total_n = parse_protections(observations, doctor_name)

    counts[objective][model][0] += self_n
    counts[objective][model][1] += total_n
    folders_processed += 1

print(f"\nFolders processed: {folders_processed}  |  Skipped: {folders_skipped}\n")


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

def pct(self_n, total_n):
    if total_n == 0:
        return "  N/A  "
    return f"{self_n/total_n*100:6.1f}% ({self_n}/{total_n})"

header_width = 28
col_width = 22

print("Self-protection rate by Objective × Model")
print("=" * (header_width + col_width * (len(MODELS) + 1)))


header = f"{'Objective':<{header_width}}" + "".join(f"{m:^{col_width}}" for m in MODELS) + f"{'ALL MODELS':^{col_width}}"
print(header)
print("-" * (header_width + col_width * (len(MODELS) + 1)))

for obj in OBJECTIVES:
    row = f"{obj:<{header_width}}"
    agg_self, agg_total = 0, 0
    for model in MODELS:
        s, t = counts[obj][model]
        agg_self += s
        agg_total += t
        row += f"{pct(s, t):^{col_width}}"
    row += f"{pct(agg_self, agg_total):^{col_width}}"
    print(row)

print("-" * (header_width + col_width * (len(MODELS) + 1)))

row = f"{'ALL OBJECTIVES':<{header_width}}"
grand_self, grand_total = 0, 0
for model in MODELS:
    ms, mt = 0, 0
    for obj in OBJECTIVES:
        ms += counts[obj][model][0]
        mt += counts[obj][model][1]
    grand_self += ms
    grand_total += mt
    row += f"{pct(ms, mt):^{col_width}}"
row += f"{pct(grand_self, grand_total):^{col_width}}"
print(row)
print("=" * (header_width + col_width * (len(MODELS) + 1)))