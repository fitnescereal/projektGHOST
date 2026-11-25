import random

def find_cefr(txt: str) -> str:
    # Zwraca losowo, aby nie blokować tworzenia pozostałych mechanik
    # TODO: Przegadać format cefr_words.json i dorobić funkcję
    
    return random.choice(["A2", "B1", "B2", "C1", "C2"])