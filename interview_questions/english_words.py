import re

from nltk.corpus import words


tokens = re.findall(r'\b[a-zA-Z]+\b', text)

english_vocab = set(words.words())
english_words = [word for word in tokens if word.lower() in english_vocab]