import numpy as np
import gensim.downloader as api
from gensim.models.phrases import Phrases, Phraser #pip install genism

# very similar to Word2Vec but is already pretrained 

#takes the model as parameter to stop from loading everytime which made the run slow
def get_glove_keywords(training_documents, processed_documents, glove_model):

#the glove is now Cached
    # phrase detection same as Word2Vec not just words
    phrases = Phrases(training_documents, min_count=2, threshold=5)

    bigram = Phraser(phrases)

    # apply phrases to training documents
    training_documents = [
        bigram[doc]
        for doc in training_documents
    ]

    # apply phrase
    processed_documents = [
        bigram[doc]
        for doc in processed_documents
    ]

    top_n_keywords = 5

    # store results
    results = []

    for i, doc in enumerate(processed_documents):

        # Keep only words available in GloVe vocabulary
        valid_words = [
            word for word in doc
            if word in glove_model
        ]

        # Skip empty documents
        if not valid_words:
            results.append([])
            continue

        # Compute document vector (average of word embeddings)
        doc_vector = np.mean(
            [glove_model[word] for word in valid_words],
            axis=0
        )

        word_scores = {}

        for word in valid_words:

            word_vector = glove_model[word]

            # cosine similarity between word and document vector
            similarity = np.dot(word_vector, doc_vector) / (
                np.linalg.norm(word_vector) *
                np.linalg.norm(doc_vector)
            )

            word_scores[word] = similarity

        # sort words by similarity score (highest first)
        sorted_words = sorted(
            word_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        
        top_words = [
            word for word, score in sorted_words[:top_n_keywords]
        ]

        results.append(top_words)

    return results