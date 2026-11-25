import json

import spacy

nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

CONTENT_POS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV", "NUM"}
FUNCTION_POS = {"ADP", "AUX", "CCONJ", "SCONJ", "DET", "PART", "PRON", "INTJ"}


def split_words_by_role(data: dict[str, list[str]]) -> dict[str, list[str]]:
  content_words: set[str] = set()
  function_words: set[str] = set()

  all_texts = [text.lower() if isinstance(text, str) else "" for text_list in data.values() for text in text_list]

  for doc in nlp.pipe(all_texts, batch_size=50):
    for token in doc:
      if token.is_space or token.is_punct:
        continue

      if token.pos_ in CONTENT_POS:
        content_words.add(token.text)
      elif token.pos_ in FUNCTION_POS:
        function_words.add(token.text)

  return {
    "content_words": sorted(content_words),
    "function_words": sorted(function_words),
  }


if __name__ == "__main__":
  with open("test_data.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

  result = split_words_by_role(raw_data)

  with open("gotowy_podział.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
