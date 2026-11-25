import spacy
import tomllib
from pathlib import Path

with (Path(__file__).resolve().parents[1] / "config.toml").open("rb") as file:
    config = tomllib.load(file)


def get_clean_lexical_tokens(doc: spacy.tokens.Doc, removes_soft: bool = False) -> list[spacy.tokens.Token]:
    """Filtruje dokument spaCy, usuwając encje nazwane (NER).

    Pomija znaki interpunkcyjne, białe znaki, liczby, waluty oraz encje
    zdefiniowane w konfiguracji jako ściśle (i opcjonalnie miękke).

    Args:
        doc: Przetworzony obiekt dokumentu z biblioteki spaCy.
        removes_soft: Czy odfiltrowywać również encje "miękkie"

    Returns:
        Lista przefiltrowanych tokenów.
    """

    clean_tokens = []
    for token in doc:
        # Removes white spaces and numbers
        if (
            token.is_punct or token.is_space or token.like_num or token.is_currency or      # Whitespaces and numbers
            token.ent_type_ in config["NER"]["strict_ignored"] or
            (removes_soft and token.ent_type_ in config["NER"]["soft_ignored"])
        ):
            continue

        clean_tokens.append(token)

    return clean_tokens


def get_ner_tokens(doc: spacy.tokens.Doc) -> tuple[list[spacy.tokens.Token], list[spacy.tokens.Token], dict[str, list[spacy.tokens.Token]]]:
    """Rozdziela tokeny encji nazwanych (NER) z dokumentu spaCy według reguł konfiguracji.

    Przechodzi przez wszystkie tokeny w dokumencie i rozdziela je na trzy grupy:
    ściśle ignorowane, miękko ignorowane oraz encje, które bezpośrednio
    sugerują określony poziom trudności CEFR.

    Args:
        doc: Przetworzony obiekt dokumentu z biblioteki spaCy.

    Returns:
        Krotka składająca się z trzech elementów:
            - strict_ner_tokens: Lista tokenów należących do encji ściśle ignorowanych
            - soft_ner_tokens: Lista tokenów należących do encji miękko ignorowanych
            - implies_cefr_tokens: Słownik mapujący kod poziomu CEFR (np. "A1", "B2")
              na listę odpowiadających mu tokenów encji.
    """
    strict_ner_tokens = []
    soft_ner_tokens = []
    implifies_cefr_tokens = dict()

    for token in doc:
        if token.ent_type_ in config["NER"]["strict_ignored"]: strict_ner_tokens.append(token)
        elif token.ent_type_ in config["NER"]["soft_ignored"]: soft_ner_tokens.append(token)
        else:
            cefr_lvls = config["NER"].get("implies_cefr", {}).keys()
            for lvl in cefr_lvls:
                if token.ent_type_ in config["NER"]["implies_cefr"][lvl]:
                    implifies_cefr_tokens[lvl] = implifies_cefr_tokens.get(lvl, []) + [token]
                    break

    return strict_ner_tokens, soft_ner_tokens, implifies_cefr_tokens
