import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CEFR_WORDS_PATH = PROJECT_ROOT / "data" / "raw" / "cefr_words.json"
# changes every "A1" word in cefr_words.json to "A2" level, bacause we don't have A1 level in gotowa_baza.json anymore

def replace_a1_with_a2(file_path: Path = DEFAULT_CEFR_WORDS_PATH) -> list[dict]:
    with file_path.open("r", encoding="utf-8") as file:
        words_data = json.load(file)

    for entry in words_data:
        if entry.get("level") == "A1":
            entry["level"] = "A2"

    with file_path.open("w", encoding="utf-8") as file:
        json.dump(words_data, file, ensure_ascii=False, indent=2)

    return words_data


if __name__ == "__main__":
    replace_a1_with_a2()

