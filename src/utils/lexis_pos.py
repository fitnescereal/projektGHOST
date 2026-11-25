import json

import spacy

nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])


def defining_pos(data: dict[str, list[str]]) -> list[dict[str, str]]:
  pos_list = []

  all_texts = [text if isinstance(text, str) else "" for text_list in data.values() for text in text_list]

  for doc in nlp.pipe(all_texts, batch_size=50):
    for token in doc:
      if not token.is_space:
        pos_list.append({"word": token.text, "pos": token.pos_})
  return pos_list


if __name__ == "__main__":
    with open("test_data.json", "r", encoding="utf-8") as file:
        data = json.load(file)

    result = defining_pos(data)

    with open("gotowe_etykietowanie.json", "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
