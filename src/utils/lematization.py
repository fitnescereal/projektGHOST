import spacy

CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV"}


def lemmatize_token(token: spacy.tokens.Token, content_pos_only: bool = True) -> tuple[str, str] | None:
    """Sprowadza pojedynczy token spaCy do formy podstawowej (lematu) wraz z częścią mowy.

    Args:
        token (spacy.tokens.Token): Pojedynczy token z dokumentu spaCy.
        content_pos_only Jeśli True, zwraca wynik wyłącznie dla słów
        znaczących (NOUN, VERB, ADJ, ADV), a dla słów funkcyjnych zwraca None.

    Returns:
        Krotka w formacie (lemat, część_mowy), np. ("run", "VERB") lub None, 
        jeśli token nie spełnia kryteriów.
    """
    if content_pos_only and token.pos_ not in CONTENT_POS:
        return None

    lemma = token.lemma_.lower().strip()

    if not lemma or not lemma.isalpha():
        return None

    return (lemma, token.pos_)