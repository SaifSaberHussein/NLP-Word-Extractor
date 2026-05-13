import streamlit as st
import os
from Preprocessing import preprocess_text
from tfidf import get_tfidf_keywords
from word2vec import get_w2v_keywords
from glove import get_glove_keywords #added 

import gensim.downloader as api  #added to load glove


# CACHE FUNCTIONS (ADDED)
#helps in faster runtime
@st.cache_data  # to cache dataset loading for faster run
def load_training_documents(folder_path, txt_files):  # moved to top

    training_documents = []

    for filename in txt_files:

        file_path = os.path.join(folder_path, filename)

        with open(file_path, "r", encoding="utf-8", errors="ignore") as file:

            text = file.read()

            processed = preprocess_text(text)

            training_documents.append(processed)

    return training_documents


@st.cache_resource  # ADDED: cache pretrained GloVe model so it doesnt load everyrun
def load_glove_model():
    return api.load("glove-wiki-gigaword-100")



st.set_page_config(page_title="Keyword Extraction", layout="wide")

st.title("Keyword Extraction System")
st.write("TF-IDF vs Word2Vec vs GloVe")

script_dir = os.path.dirname(os.path.abspath(__file__))
folder_path = os.path.join(script_dir, "Dataset")

# Get txt files
txt_files = [
    file for file in os.listdir(folder_path)
    if file.endswith(".txt")
]

# User chooses file
selected_file = st.selectbox(
    "Choose a dataset file",
    txt_files
)


# LOAD dataset for word2vec training

training_documents = load_training_documents(folder_path, txt_files)



glove_model = load_glove_model()  #loads only 1 time




all_documents = []
file_names = []

if selected_file:

    file_path = os.path.join(folder_path, selected_file)

    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:

        text = file.read()

        all_documents.append(text)
        file_names.append(selected_file)




if st.button("Run Analyisia"):
    st.info("مخشيهىل...")

    # Preprocessing
    processed_documents = [
        preprocess_text(doc)
        for doc in all_documents
    ]

    cleaned_documents = [
        " ".join(doc)
        for doc in processed_documents
    ]

    # RUN MODELS

    tfidf_results = get_tfidf_keywords(cleaned_documents)

    # Word2Vec
    w2v_results = get_w2v_keywords(
        training_documents,
        processed_documents
    )

    # GloVe
    glove_results = get_glove_keywords(
        training_documents,
        processed_documents,
        glove_model   # ADDED: pretrained model passed here
    )

    st.success("Complete")

    # rESULTS

    for i in range(len(file_names)):

        st.markdown(f"## {file_names[i]}")

        col1, col2, col3 = st.columns(3)  #3 models was 2

        with col1:
            st.markdown("### TF-IDF")
            st.write(tfidf_results[i])

        with col2:
            st.markdown("### Word2Vec")
            st.write(w2v_results[i])

        with col3:
            st.markdown("### GloVe")  # ADDEd
            st.write(glove_results[i])

        st.markdown("---")