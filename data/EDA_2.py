import pandas as pd
import ast

df = pd.read_csv(r"D:\USTH\nlp\final_prj\data\train.csv")

import re

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

df["tokens"] = df["text"].apply(tokenize)

from collections import Counter

all_tokens = []

for tokens in df["tokens"]:
    all_tokens.extend(tokens)

word_freq = Counter(all_tokens)

word_freq.most_common(50)

print(f"Number of unique words: {len(word_freq)}")
print(f"Most common words: {word_freq.most_common(10)}")

print("\n")
print("==================== ELONGATED WORDS ===================")
print("\n")
elongated_words = [
    w for w in word_freq
    if re.search(r"(.)\1{2,}", w)
]
print(elongated_words)

from wordfreq import zipf_frequency
print("\n")
print("==================== UNUSUAL WORDS ===================")
print("\n")
unusual_words = [
    w for w in word_freq
    if zipf_frequency(w, "en") < 2
]
print(unusual_words)



