from statistics import fmean, pstdev

import spacy
from src.utils.ner import get_clean_lexical_tokens


def _get_sentences(doc: spacy.tokens.Doc) -> list[spacy.tokens.Span]:
    """Zwraca zdania już wyznaczone w przekazanym dokumencie spaCy."""
    return list(doc.sents)


def _get_percentile(values: list[float], percentile: float) -> float:
    """Oblicza percentyl metodą interpolacji liniowej.

    Args:
        values: Wartości, dla których obliczany jest percentyl.
        percentile: Percentyl jako ułamek z zakresu od 0 do 1.

    Returns:
        Wartość percentyla albo 0.0 dla pustej listy.
    """
    if not values:
        return 0.0

    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * percentile
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = position - lower_index

    return ordered[lower_index] + (ordered[upper_index] - ordered[lower_index]) * fraction


def _get_sentence_lengths(sentences: list[spacy.tokens.Span]) -> list[int]:
    """Zwraca długości zdań liczone w tokenach bez spacji i interpunkcji."""
    return [
        sum(not token.is_punct and not token.is_space for token in sentence)
        for sentence in sentences
    ]


def _get_mean_sentence_length(sentence_lengths: list[int]) -> float:
    """Oblicza średnią długość zdań albo 0.0, gdy lista jest pusta."""
    return fmean(sentence_lengths) if sentence_lengths else 0.0


def _get_sentence_length_std(sentence_lengths: list[int]) -> float:
    """Oblicza populacyjne odchylenie standardowe długości zdań."""
    return pstdev(sentence_lengths) if sentence_lengths else 0.0


def _get_sentence_length_p90(sentence_lengths: list[int]) -> float:
    """Oblicza 90. percentyl długości zdań."""
    return _get_percentile(
        [float(length) for length in sentence_lengths],
        percentile=0.90,
    )


def _get_noun_verb_ratio(words: list[spacy.tokens.Token]) -> float:
    """Oblicza stosunek liczby rzeczowników do czasowników."""
    noun_count = sum(token.pos_ in {"NOUN", "PROPN"} for token in words)
    verb_count = sum(token.pos_ == "VERB" for token in words)
    return noun_count / verb_count if verb_count else 0.0


def _get_lexical_density(words: list[spacy.tokens.Token]) -> float:
    """Oblicza udział rzeczowników, czasowników, przymiotników i przysłówków."""
    lexical_pos = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}
    lexical_count = sum(token.pos_ in lexical_pos for token in words)
    return lexical_count / len(words) if words else 0.0


def _get_punctuation_complexity(doc: spacy.tokens.Doc, words: list[spacy.tokens.Token],) -> float:
    """Oblicza liczbę znaków interpunkcyjnych na oczyszczony token tekstowy."""
    punctuation_count = sum(token.is_punct for token in doc)
    return punctuation_count / len(words) if words else 0.0


def _get_subject_predicate_mean_distance(doc: spacy.tokens.Doc) -> float:
    """Oblicza średnią odległość tokenową podmiotów od ich głów składniowych."""
    subject_predicate_distances = []
    for token in doc:
        if token.dep_ in {"nsubj", "nsubjpass", "csubj", "csubjpass"}:
            subject_predicate_distances.append(abs(token.i - token.head.i))

    return fmean(subject_predicate_distances) if subject_predicate_distances else 0.0


def _get_max_sentence_nesting(sentences: list[spacy.tokens.Span]) -> float:
    """Zwraca maksymalną głębokość tokenu w drzewie zależności składniowych."""
    sentence_nesting_levels = []
    for sentence in sentences:
        for token in sentence:
            depth = 0
            ancestor = token
            while ancestor.head != ancestor:
                depth += 1
                ancestor = ancestor.head
            sentence_nesting_levels.append(depth)
    return float(max(sentence_nesting_levels)) if sentence_nesting_levels else 0.0


def _get_subordinate_clauses_per_sentence(
    sentences: list[spacy.tokens.Span],
) -> float:
    """Oblicza średnią liczbę zdań podrzędnych przypadających na zdanie."""
    subordinate_deps = {
        "advcl",
        "acl",
        "relcl",
        "ccomp",
        "csubj",
        "csubjpass",
    }
    subordinate_clause_count = sum(
        token.dep_ in subordinate_deps
        for sentence in sentences
        for token in sentence
    )
    return subordinate_clause_count / len(sentences) if sentences else 0.0


def get_text_feature_vector(doc: spacy.tokens.Doc) -> dict[str, float]:
    """Zwraca słownik dziewięciu cech dla gotowego dokumentu spaCy.

    Dokument musi mieć wyznaczone granice zdań. Cechy leksykalne i składniowe
    wymagają odpowiednich adnotacji POS oraz zależności parsera. Funkcja nie
    przetwarza ponownie tekstu ani nie dodaje tych adnotacji.
    """
    sentences = _get_sentences(doc)
    sentence_lengths = _get_sentence_lengths(sentences)
    words = get_clean_lexical_tokens(doc)

    return {
        "sentence_length_mean": _get_mean_sentence_length(sentence_lengths),
        "sentence_length_std": _get_sentence_length_std(sentence_lengths),
        "sentence_length_p90": _get_sentence_length_p90(sentence_lengths),
        "noun_verb_ratio": _get_noun_verb_ratio(words),
        "lexical_density": _get_lexical_density(words),
        "punctuation_complexity": _get_punctuation_complexity(doc, words),
        "subject_predicate_mean_distance": _get_subject_predicate_mean_distance(doc),
        "max_sentence_nesting": _get_max_sentence_nesting(sentences),
        "subordinate_clauses_per_sentence": _get_subordinate_clauses_per_sentence(sentences),
    }