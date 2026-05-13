from sklearn.feature_extraction.text import TfidfVectorizer


def get_tfidf_keywords(cleaned_documents):
    
    #deleted: preprocessing moved to main

    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(cleaned_documents) 
    print(matrix.shape)

    feature_names = vectorizer.get_feature_names_out()
    top_n_keywords = 5 
    results = []

    for i in range(len(cleaned_documents)):
        
        doc_vector = matrix[i].toarray()[0]
        top_indices = doc_vector.argsort()[::-1]
        top_words = [feature_names[index] for index in top_indices[:top_n_keywords]]
        results.append(top_words)

    return results