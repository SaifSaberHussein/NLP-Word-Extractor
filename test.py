import os
from Preprocessing import preprocess_text
from tfidf import get_tfidf_keywords
from word2vec import get_w2v_keywords
import time


# Load dataset put here to avoid repition

script_dir = os.path.dirname(os.path.abspath(__file__))
folder_path = os.path.join(script_dir, "Dataset")

all_documents = []
file_names = []

for filename in os.listdir(folder_path):

    if filename.lower().endswith(".txt"):

        file_path = os.path.join(folder_path, filename)

        print("Loaded file:", filename)

        with open(file_path, "r", encoding="utf-8", errors="ignore") as file:

            text = file.read()

            all_documents.append(text)
            file_names.append(filename)


# Preprocess added here to only preprocess once
processed_documents = [
    preprocess_text(doc)
    for doc in all_documents
]

# TF-IDF takes strings
cleaned_documents = [
    " ".join(doc)
    for doc in processed_documents
]

tfidf_results = get_tfidf_keywords(cleaned_documents)


w2v_results = get_w2v_keywords(processed_documents)


# Final Comparison

print("\nCOMPARISONS\n")

for i in range(len(file_names)):

    print(f"File: {file_names[i]}")
    print(f"TF-IDF Keywords: {tfidf_results[i]}")
    print(f"Word2Vec Keywords: {w2v_results[i]}")
    print("--------------------------------------")
    time.sleep(0.1)