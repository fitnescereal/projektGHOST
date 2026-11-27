""" w pliku obliczne są:
czasowniki modalne
gęstość strony biernej
udział czasów złożonych
zdania warunkowe i tryb łączący
średnia długość słowa w sylabach
odsetek słów wielosylabowych """

import json
import re
from pathlib import Path

import cmudict
import spacy

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_PATH = PROJECT_ROOT / "data" / "raw" / "gotowa_baza.json"
DEFAULT_CEFRED_PATH = PROJECT_ROOT / "data" / "processed" / "cefred_sentences.json"

MODALS = {"can", "could", "may", "might", "must", "shall", "should", "will", "would"}
CONDITIONAL_MARKERS = {"if", "unless", "provided", "providing", "assuming", "supposing"}
SUBJUNCTIVE_TRIGGERS = {
    "suggest", "insist", "recommend", "demand", "require", "request",
    "propose", "order", "urge", "advise", "vital", "essential",
    "important", "necessary", "imperative", "crucial",
}
SINGULAR_PRONOUNS = {"i", "he", "she", "it"}
VOWEL_GROUP_RE = re.compile(r"[aeiouy]+")


def count_modal_verbs(doc: spacy.tokens.Doc) -> int:
    """Zlicza czasowniki modalne w dokumencie spaCy (uwzględnia formy skrócone, np. can't, won't, 'll, 'd,
    a pomija homonimy takie jak miesiąc 'May' czy rzeczownik 'can').
    """
    return sum(
        1
        for token in doc
        if token.tag_ == "MD" and token.lemma_.lower() in MODALS
    )


def passive_voice_density(doc: spacy.tokens.Doc, decimals: int = 4) -> float:
    """Oblicza gęstość strony biernej jako liczbę konstrukcji biernych na zdanie."""
    sentences = list(doc.sents)
    if not sentences:
        return 0.0

    passive_heads = {token.head.i for token in doc if token.dep_ == "auxpass"}
    return round(len(passive_heads) / len(sentences), decimals)


def complex_tense_ratio(doc: spacy.tokens.Doc, decimals: int = 4) -> float:
    """Oblicza udział czasów złożonych wśród wszystkich głównych czasowników w dokumencie.
        (czasy złożone/wszystkie czasowniki)

    Za czas złożony uznaje orzeczenie (VERB lub główny AUX), które posiada czasownik posiłkowy
    (dep_ == "aux") tworzący czas:
    - Continuous: 'be' + VBG (np. is working, were playing)
    - Perfect: 'have' + VBN/VBG (np. have finished, had left, has been written)
    - Future: 'will'/'shall' (tag MD, np. will go, will have been working)

    Pomija posiłkowe 'do' (przeczenia/pytania w Simple), 'to' (bezokoliczniki) oraz samo 'auxpass'.
    """
    total_verbs = 0
    complex_verbs = 0

    for token in doc:
        if token.pos_ not in {"VERB", "AUX"} or token.dep_ in {"aux", "auxpass"}:
            continue

        total_verbs += 1
        aux_children = [child for child in token.children if child.dep_ == "aux"]
        aux_lemmas = {child.lemma_.lower() for child in aux_children}

        is_continuous = "be" in aux_lemmas and token.tag_ == "VBG"
        is_perfect = "have" in aux_lemmas and token.tag_ in {"VBN", "VBG"}
        is_future = any(
            child.tag_ == "MD" and child.lemma_.lower() in {"will", "shall"}
            for child in aux_children
        )

        if is_continuous or is_perfect or is_future:
            complex_verbs += 1

    if total_verbs == 0:
        return 0.0
    return round(complex_verbs / total_verbs, decimals)


def conditional_subjunctive_density(doc: spacy.tokens.Doc, decimals: int = 4) -> float:
    """Oblicza gęstość zdań warunkowych i konstrukcji w trybie łączącym (subjunctive) na zdanie.

    Wykrywa (bez podwójnego liczenia tej samej klauzuli):
    - klasyczne klauzule warunkowe: spójnik warunkowy (mark) podpięty pod advcl,
    - inwersję warunkową: Had/Should/Were przed podmiotem w klauzuli podrzędnej (niebędącej pytaniem),
    - tryb łączący 'were' z podmiotem w 1./3. osobie l.p. (np. 'if I were', 'I wish he were'),
    - mandative subjunctive po słowach typu suggest/insist/vital + that + forma podstawowa (np. 'that he go', 'that it be').
    """
    sentences = list(doc.sents)
    if not sentences:
        return 0.0

    matched_clauses: set[int] = set()

    for sent in sentences:
        is_question = sent.text.strip().endswith("?")

        for token in sent:
            # 1. Klasyczne zdania warunkowe (if / unless / provided ... -> advcl)
            if (
                token.dep_ == "mark"
                and token.text.lower() in CONDITIONAL_MARKERS
                and token.head.dep_ == "advcl"
            ):
                matched_clauses.add(token.head.i)

            # 2. Inwersja warunkowa (Had I known..., Were she here..., Should you need...)
            if not is_question and token.dep_ in {"advcl", "ccomp"}:
                subj = next((c for c in token.children if c.dep_ in {"nsubj", "nsubjpass"}), None)
                if subj is not None:
                    inverted_aux = any(
                        c.dep_ == "aux"
                        and c.text.lower() in {"had", "should"}
                        and c.i < subj.i
                        for c in token.children
                    )
                    inverted_were = token.text.lower() == "were" and token.i < subj.i
                    if inverted_aux or inverted_were:
                        matched_clauses.add(token.i)

            # 3. Tryb łączący 'were' z podmiotem w liczbie pojedynczej (I/he/she/it lub rzeczownik l.p.)
            if token.text.lower() == "were":
                subj = next((c for c in token.children if c.dep_ in {"nsubj", "nsubjpass"}), None)
                if subj is not None and (
                    subj.text.lower() in SINGULAR_PRONOUNS
                    or subj.tag_ in {"NN", "NNP"}
                ):
                    matched_clauses.add(token.i)

            # 4. Mandative subjunctive (insist/suggest/vital + that + he go / it be finished)
            if token.dep_ == "ccomp":
                has_that = any(
                    c.dep_ == "mark" and c.text.lower() == "that"
                    for c in token.children
                )
                if not has_that:
                    continue

                trigger_lemmas = {token.head.lemma_.lower()}
                trigger_lemmas.update(
                    c.lemma_.lower() for c in token.head.children if c.dep_ == "acomp"
                )
                if not (trigger_lemmas & SUBJUNCTIVE_TRIGGERS):
                    continue

                has_base_be = (
                    (token.text.lower() == "be" and token.tag_ == "VB")
                    or any(
                        c.dep_ in {"aux", "auxpass"}
                        and c.text.lower() == "be"
                        and c.tag_ == "VB"
                        for c in token.children
                    )
                )
                subj = next((c for c in token.children if c.dep_ in {"nsubj", "nsubjpass"}), None)
                is_3sg_base = (
                    subj is not None
                    and (subj.text.lower() in {"he", "she", "it"} or subj.tag_ in {"NN", "NNP"})
                    and token.tag_ in {"VB", "VBP"}
                )
                if has_base_be or is_3sg_base:
                    matched_clauses.add(token.i)

    return round(len(matched_clauses) / len(sentences), decimals)


def count_syllables(word: str, cmu_dict: dict[str, list[list[str]]]) -> int:
    """Zwraca liczbę sylab w angielskim słowie na podstawie cmudict (z heurystyką dla słów spoza słownika)."""
    cleaned = word.lower().strip()
    if not cleaned:
        return 0

    pronunciations = cmu_dict.get(cleaned)
    if pronunciations:
        return sum(1 for phoneme in pronunciations[0] if phoneme[-1].isdigit())

    # Awaryjna heurystyka dla rzadkich słów nieobecnych w cmudict
    groups = VOWEL_GROUP_RE.findall(cleaned)
    count = len(groups)
    if cleaned.endswith("e") and not cleaned.endswith("le") and count > 1:
        count -= 1
    return max(1, count)


def average_word_syllables(
    doc: spacy.tokens.Doc,
    cmu_dict: dict[str, list[list[str]]],
    decimals: int = 4,
) -> float:
    """Oblicza średnią długość słowa w sylabach dla jednego dokumentu spaCy."""
    words = [token.text for token in doc if token.is_alpha]
    if not words:
        return 0.0

    total_syllables = sum(count_syllables(word, cmu_dict) for word in words)
    return round(total_syllables / len(words), decimals)


def polysyllabic_word_ratio(
    doc: spacy.tokens.Doc,
    cmu_dict: dict[str, list[list[str]]],
    min_syllables: int = 3,
    decimals: int = 4,
) -> float:
    """Oblicza odsetek słów wielosylabowych (domyślnie >= 3 sylaby) w jednym dokumencie spaCy."""
    words = [token.text for token in doc if token.is_alpha]
    if not words:
        return 0.0

    polysyllabic_count = sum(
        1 for word in words if count_syllables(word, cmu_dict) >= min_syllables
    )
    return round(polysyllabic_count / len(words), decimals)


def add_nlp_features(
    base_path: Path = DEFAULT_BASE_PATH,
    cefred_path: Path = DEFAULT_CEFRED_PATH,
    batch_size: int = 64,
    min_syllables: int = 3,
    decimals: int = 4,
) -> dict[str, list[dict]]:
    """W jednym przebiegu przetwarza gotowa_baza.json i dopisuje cechy NLP do cefred_sentences.json.

    Dodawane klucze:
        - modal_verbs: liczba czasowników modalnych
        - passive_voice_density: liczba konstrukcji w stronie biernej na zdanie
        - complex_tense_ratio: udział czasów złożonych wśród głównych czasowników
        - conditional_subjunctive_density: liczba zdań warunkowych i trybu łączącego na zdanie
        - avg_word_syllables: średnia liczba sylab na słowo
        - polysyllabic_word_ratio: odsetek słów wielosylabowych (>= min_syllables)
    """
    cmu_dict = cmudict.dict()

    with base_path.open("r", encoding="utf-8") as file:
        base_data: dict[str, list[str]] = json.load(file)

    with cefred_path.open("r", encoding="utf-8") as file:
        cefred_data: dict[str, list[dict]] = json.load(file)

    # Parser zostaje włączony (potrzebny do wykrywania auxpass i podziału na zdania doc.sents)
    nlp = spacy.load("en_core_web_sm", disable=["ner"])

    for level_key, sentences in base_data.items():
        if level_key not in cefred_data:
            raise KeyError(f"Brak klucza '{level_key}' w pliku {cefred_path.name}.")

        target_entries = cefred_data[level_key]
        if len(sentences) != len(target_entries):
            raise ValueError(
                f"Niezgodna liczba elementów dla poziomu '{level_key}': "
                f"{len(sentences)} w {base_path.name} vs {len(target_entries)} w {cefred_path.name}."
            )

        safe_sentences = [s if isinstance(s, str) else "" for s in sentences]
        for entry, doc in zip(target_entries, nlp.pipe(safe_sentences, batch_size=batch_size)):
            entry["modal_verbs"] = count_modal_verbs(doc)
            entry["passive_voice_density"] = passive_voice_density(doc, decimals=decimals)
            entry["complex_tense_ratio"] = complex_tense_ratio(doc, decimals=decimals)
            entry["conditional_subjunctive_density"] = conditional_subjunctive_density(
                doc, decimals=decimals
            )
            entry["avg_word_syllables"] = average_word_syllables(doc, cmu_dict, decimals=decimals)
            entry["polysyllabic_word_ratio"] = polysyllabic_word_ratio(
                doc, cmu_dict, min_syllables=min_syllables, decimals=decimals
            )

    with cefred_path.open("w", encoding="utf-8") as file:
        json.dump(cefred_data, file, ensure_ascii=False, indent=2)

    return cefred_data


if __name__ == "__main__":
    add_nlp_features()
