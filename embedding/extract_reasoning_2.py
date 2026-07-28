import os, json, argparse

MODES = ["Benevolent", "Individualistic", "Malevolent"]

ROLE_KEY_MAP = {
    "seer":      lambda d: d["seer"]["name"],
    "doctor":    lambda d: d["doctor"]["name"],
    "villager":  lambda d: d["villagers"][0]["name"],
    "werewolf":  lambda d: d["werewolves"][0]["name"],
}


def get_player_id(game_complete: dict, role: str) -> str | None:
    key = role.lower()
    if key not in ROLE_KEY_MAP:
        return None
    try:
        return ROLE_KEY_MAP[key](game_complete)
    except (KeyError, IndexError, TypeError):
        return None
 
 
def extract_debate_reasoning(game_logs: list, player_name: str) -> list[str]:
    collected = []
    for round_ in game_logs:
        for item in round_.get("debate", []):
            if not (isinstance(item, list) and len(item) == 2):
                continue
            speaker, entry = item
            if speaker == player_name and isinstance(entry, dict):
                r = (entry.get("result") or {}).get("reasoning")
                if r:
                    collected.append(r)
    return collected

def parse_folder(folder_name: str):

    parts = folder_name.split("_")
    if len(parts) != 6 or parts[0] != "session":
        return None
    _, model, role, mode, personality, seed = parts
    if mode not in MODES:
        return None
    return model, role, mode, personality, seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", required=True)
    parser.add_argument("--model",    required=True,
                        help="Model name as it appears in folder, e.g. Gemma")
    parser.add_argument("--role",     required=True,
                        help="Role as it appears in folder, e.g. Seer, Doctor, Werewolf, Villager")
    args = parser.parse_args()

    results = {m: [] for m in MODES}
    counts  = {m: 0  for m in MODES}

    for folder in sorted(os.listdir(args.base_dir)):
        folder_path = os.path.join(args.base_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        parsed = parse_folder(folder)
        if parsed is None:
            continue
        model, role, mode, personality, seed = parsed

        if model != args.model or role != args.role:
            continue

        gc_path = os.path.join(folder_path, "game_complete.json")
        if not os.path.exists(gc_path):
            print(f"[SKIP] missing game_complete.json in: {folder}")
            continue

        with open(gc_path) as f:
            game_complete = json.load(f)

        player = get_player_id(game_complete, args.role)
        if player is None:
            print(f"[SKIP] role '{args.role}' not found in game_complete.json: {folder}")
            continue
        gl_path = os.path.join(folder_path, "game_logs.json")
        if not os.path.exists(gl_path):
            print(f"[SKIP] missing game_logs.json in: {folder}")
            continue

        with open(gl_path) as f:
            game_logs = json.load(f)

        reasoning = extract_debate_reasoning(game_logs, player)
        results[mode].extend(reasoning)
        counts[mode] += 1
        print(f"  {folder}: player={player}, mode={mode}, seed={seed}, "
              f"debate_entries={len(reasoning)}")

    print("\n=== SUMMARY ===")
    for m in MODES:
        print(f"  {m}: {counts[m]} sessions, {len(results[m])} reasoning entries total")

    out = f"reasoning_{args.model}_{args.role}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()