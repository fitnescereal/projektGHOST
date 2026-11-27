import json
from pathlib import Path
from wordfreq import zipf_frequency
# words frequency in a logarithmic 0 to 8 scale
""" limitations: wordfreq aggregates native-speaker texts (sources like Wikipedia, news, overall web texts).
    It doesn't also recoginize the context, so basic verbs like run, charge, etc. get high scores as they have
    secondary legal, financial or idiomatic senses (B2/C1 CEFR level).
"""
# 
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CEFR_WORDS_PATH = PROJECT_ROOT / "data" / "raw" / "cefr_words.json"


def add_zipf_frequency_to_cefr_words( file_path: Path = DEFAULT_CEFR_WORDS_PATH, lang: str = "en",) -> list[dict]:
    with file_path.open("r", encoding="utf-8") as file:
        words_data = json.load(file)

    for entry in words_data:
        entry["zipf_frequency"] = zipf_frequency(entry["word"], lang)

    with file_path.open("w", encoding="utf-8") as file:
        json.dump(words_data, file, ensure_ascii=False, indent=2)

    return words_data


if __name__ == "__main__":
    add_zipf_frequency_to_cefr_words()

