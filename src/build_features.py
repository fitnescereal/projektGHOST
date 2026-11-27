import json
from pathlib import Path

import cmudict
import spacy

try:
    from src.utils.nlp_makary import (
        average_word_syllables,
        complex_tense_ratio,
        conditional_subjunctive_density,
        count_modal_verbs,
        passive_voice_density,
        polysyllabic_word_ratio,
    )
    from src.utils.words_level_in_sentence import (
        count_cefr_levels_in_doc,
        counts_to_ratios,
        load_cefr_word_map,
    )
except ImportError:
    from utils.nlp_makary import (
        average_word_syllables,
        complex_tense_ratio,
        conditional_subjunctive_density,
        count_modal_verbs,
        passive_voice_density,
        polysyllabic_word_ratio,
    )
    from utils.words_level_in_sentence import (
        count_cefr_levels_in_doc,
        counts_to_ratios,
        load_cefr_word_map,
    )

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "gotowa_baza.json"
DEFAULT_CEFR_WORDS_PATH = PROJECT_ROOT / "data" / "raw" / "cefr_words.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "cefred_sentences.json"


def extract_features_for_doc(
    doc: spacy.tokens.Doc,
    cefr_map: dict[str, str],
    cmu_dict: dict[str, list[list[str]]],
    min_syllables: int = 3,
    decimals: int = 4,
) -> dict[str, float | int]:
    """Wyciąga komplet cech leksykalnych (CEFR) i gramatyczno-sylabowych dla jednego dokumentu spaCy."""
    features: dict[str, float | int] = counts_to_ratios(
        count_cefr_levels_in_doc(doc, cefr_map),
        decimals=decimals,
    )
    features["modal_verbs"] = count_modal_verbs(doc)
    features["passive_voice_density"] = passive_voice_density(doc, decimals=decimals)
    features["complex_tense_ratio"] = complex_tense_ratio(doc, decimals=decimals)
    features["conditional_subjunctive_density"] = conditional_subjunctive_density(
        doc, decimals=decimals
    )
    features["avg_word_syllables"] = average_word_syllables(
        doc, cmu_dict, decimals=decimals
    )
    features["polysyllabic_word_ratio"] = polysyllabic_word_ratio(
        doc, cmu_dict, min_syllables=min_syllables, decimals=decimals
    )
    return features


def build_features(
    input_path: Path = DEFAULT_INPUT_PATH,
    cefr_words_path: Path = DEFAULT_CEFR_WORDS_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    batch_size: int = 64,
    min_syllables: int = 3,
    decimals: int = 4,
) -> dict[str, list[dict[str, float | int]]]:
    """W jednym przebiegu przetwarza gotowa_baza.json i zapisuje komplet cech do cefred_sentences.json."""
    cefr_map = load_cefr_word_map(cefr_words_path)
    cmu_dict = cmudict.dict()

    with input_path.open("r", encoding="utf-8") as file:
        base_data: dict[str, list[str]] = json.load(file)

    # Pełny model (włączony zarówno ner dla CEFR, jak i parser dla cech gramatycznych)
    nlp = spacy.load("en_core_web_sm")

    result: dict[str, list[dict[str, float | int]]] = {}
    for level_key, sentences in base_data.items():
        safe_sentences = [s if isinstance(s, str) else "" for s in sentences]
        result[level_key] = [
            extract_features_for_doc(
                doc,
                cefr_map=cefr_map,
                cmu_dict=cmu_dict,
                min_syllables=min_syllables,
                decimals=decimals,
            )
            for doc in nlp.pipe(safe_sentences, batch_size=batch_size)
        ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)

    return result


if __name__ == "__main__":
    build_features()
