import json
from pathlib import Path

import spacy
from wordfreq import zipf_frequency

try:
    from src.utils.lematization import lemmatize_token
    from src.utils.ner import get_clean_lexical_tokens, get_ner_tokens
except ImportError:
    from lematization import lemmatize_token
    from ner import get_clean_lexical_tokens, get_ner_tokens

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "gotowa_baza.json"
DEFAULT_CEFR_WORDS_PATH = PROJECT_ROOT / "data" / "raw" / "cefr_words.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "cefred_sentences.json"

CEFR_LEVELS = ("A2", "B1", "B2", "C1", "C2")


def load_cefr_word_map(cefr_words_path: Path = DEFAULT_CEFR_WORDS_PATH) -> dict[str, str]:
    """Wczytuje cefr_words.json jako słownik {słowo_lowercase: poziom_CEFR}."""
    with cefr_words_path.open("r", encoding="utf-8") as file:
        words_data = json.load(file)

    return {
        entry["word"].strip().lower(): entry["level"].strip().upper()
        for entry in words_data
        if "word" in entry and "level" in entry
    }


def estimate_cefr_from_zipf(word: str, lang: str = "en") -> str:
    """Dopasowuje poziom CEFR na podstawie zipf_frequency dla niezlematyzowanego słowa.

    Przedziały:
    - A2: 4.8 <= freq <= 8.0 (oraz > 8.0)
    - B1: 4.2 <= freq < 4.8
    - B2: 3.6 <= freq < 4.2
    - C1: 3.0 <= freq < 3.6
    - C2: 0.0 < freq < 3.0
    - Nan: freq == 0.0 (słowo nieznane w wordfreq)
    """
    freq = zipf_frequency(word, lang)

    if freq == 0.0:
        return "Nan"
    if freq >= 4.8:
        return "A2"
    if freq >= 4.2:
        return "B1"
    if freq >= 3.6:
        return "B2"
    if freq >= 3.0:
        return "C1"
    return "C2"


def count_cefr_levels_in_doc(doc: spacy.tokens.Doc, cefr_map: dict[str, str],) -> dict[str, int]:
    """Zlicza wystąpienia słów z poszczególnych poziomów CEFR w pojedynczym dokumencie spaCy.

    Wykorzystuje:
    - reguły implies_cefr z ner.py (np. ORDINAL -> A2),
    - filtrowanie z get_clean_lexical_tokens (bez interpunkcji, liczb, strict i soft NER),
    - lematyzację słów znaczących z lematization.py,
    - awaryjne dopasowanie przez zipf_frequency (na słowach niezlematyzowanych) dla słów spoza cefr_words.json.
    """
    counts: dict[str, int] = {level: 0 for level in CEFR_LEVELS}
    counts["Nan"] = 0

    _, _, implies_cefr_tokens = get_ner_tokens(doc)
    implies_token_indices: set[int] = set()

    for level, tokens in implies_cefr_tokens.items():
        normalized_level = level.strip().upper()
        target_key = normalized_level if normalized_level in counts else "Nan"
        for token in tokens:
            implies_token_indices.add(token.i)
            counts[target_key] += 1

    clean_tokens = get_clean_lexical_tokens(doc, removes_soft=True) #usuwam nazwy wlasne, oragnizacje, itd
    for token in clean_tokens:
        if token.i in implies_token_indices:
            continue

        lemmatized = lemmatize_token(token, content_pos_only=True)
        if lemmatized is None:
            continue

        lemma, _ = lemmatized
        word_level = cefr_map.get(lemma) or cefr_map.get(token.text.lower())

        if word_level in counts:
            counts[word_level] += 1
        else:
            fallback_level = estimate_cefr_from_zipf(token.text)
            counts[fallback_level] += 1

    return counts


def counts_to_ratios(counts: dict[str, int], decimals: int = 4) -> dict[str, float]:
    """Przelicza liczniki poziomów CEFR (oraz Nan) na odsetki zaokrąglone do zadanej liczby miejsc po przecinku."""
    total = sum(counts.values())
    if total == 0:
        return {level: 0.0 for level in counts}
    return {level: round(count / total, decimals) for level, count in counts.items()}


def build_cefred_sentences(
    input_path: Path = DEFAULT_INPUT_PATH,
    cefr_words_path: Path = DEFAULT_CEFR_WORDS_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    batch_size: int = 64,
    as_ratios: bool = True,
) -> dict[str, list[dict[str, float] | dict[str, int]]]:
    """Tworzy plik cefred_sentences.json z rozkładem poziomów CEFR dla każdego tekstu z gotowa_baza.json.

    Args:
        input_path: Ścieżka do pliku gotowa_baza.json.
        cefr_words_path: Ścieżka do pliku cefr_words.json.
        output_path: Ścieżka zapisu wynikowego pliku cefred_sentences.json.
        batch_size: Rozmiar paczki dla spacy nlp.pipe.
        as_ratios: Jeśli True (domyślnie), zapisuje odsetki sumujące się do 1.0 zamiast surowych zliczeń.

    Returns:
        Słownik o strukturze odpowiadającej gotowa_baza.json, gdzie każdy tekst
        zastąpiony jest słownikiem odsetków (lub liczności) poziomów CEFR oraz "Nan".
    """
    cefr_map = load_cefr_word_map(cefr_words_path)

    with input_path.open("r", encoding="utf-8") as file:
        base_data: dict[str, list[str]] = json.load(file)

    nlp = spacy.load("en_core_web_sm", disable=["parser"])

    result: dict[str, list[dict[str, float] | dict[str, int]]] = {}
    for level_key, sentences in base_data.items():
        safe_sentences = [s if isinstance(s, str) else "" for s in sentences]
        docs = nlp.pipe(safe_sentences, batch_size=batch_size)
        if as_ratios:
            result[level_key] = [
                counts_to_ratios(count_cefr_levels_in_doc(doc, cefr_map))
                for doc in docs
            ]
        else:
            result[level_key] = [
                count_cefr_levels_in_doc(doc, cefr_map)
                for doc in docs
            ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)

    return result


if __name__ == "__main__":
    build_cefred_sentences()

