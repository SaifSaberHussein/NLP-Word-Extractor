import numpy as np
from gensim.models import Word2Vec
from multiprocessing import cpu_count
from gensim.models.phrases import Phrases, Phraser



def get_w2v_keywords(training_documents, processed_documents):

    #preprocessing removed and file seacrh and are moved to main


    cpu = cpu_count()

    print(f'using {cpu} cpus')

    phrases = Phrases(training_documents, min_count=2, threshold=5)

    bigram = Phraser(phrases)

    
    training_documents = [
        bigram[doc]
        for doc in training_documents
    ]

    
    processed_documents = [
        bigram[doc]
        for doc in processed_documents
    ]

    

    trained_text = Word2Vec(

        sentences=training_documents,

        vector_size=100,
        window=5,
        min_count=1,
        seed=42, 
        workers=4 
    )

    trained_text_wv = trained_text.wv

    top_n_keywords = 5

    #store results
    results = []

    for i, doc in enumerate(processed_documents):

        
        valid_words = [
            word for word in doc
            if word in trained_text_wv
        ]

        # Skip if empty
        if not valid_words:

           
            results.append([])
            continue

        
        doc_vector = np.mean(
            [trained_text_wv[word] for word in valid_words],
            axis=0
        )

       
        word_scores = {}

        for word in valid_words:

            word_vector = trained_text_wv[word]

            # cosine similarity
            similarity = np.dot(word_vector, doc_vector) / (
                np.linalg.norm(word_vector) *
                np.linalg.norm(doc_vector)
            )

            word_scores[word] = similarity

        #Sort words by similarity from best to worst
        sorted_words = sorted(
            word_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        #top keywords
        top_words = [
            word for word, score in sorted_words[:top_n_keywords]
        ]

        results.append(top_words)

    return results