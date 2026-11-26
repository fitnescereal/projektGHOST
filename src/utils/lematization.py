import spacy
from pathlib import Path
import json
CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV"}
PROJECY_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = PROJECY_ROOT / "data" / "raw"
FILE = DATA_RAW_DIR / "gotowy_podzial.json"
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

nlp = spacy.load("en_core_web_sm")

# 2. Przetwórz tekst na obiekt spaCy (Doc)
tekst = "Fast dogs are running around the green park and barking loudly"
doc = nlp(tekst)

# 3. Wywołaj funkcję dla każdego tokenu w tekście
for token in doc:
    wynik = lemmatize_token(token)
    if wynik is not None:  # pomijamy słowa funkcyjne (przyimki, spójniki, kropki)
        print(wynik)