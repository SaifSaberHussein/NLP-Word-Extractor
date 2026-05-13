import streamlit as st
import os
import pandas as pd
import time
from Preprocessing import preprocess_text
from tfidf import get_tfidf_keywords
from word2vec import get_w2v_keywords
from glove import get_glove_keywords
from streamlit_lottie import st_lottie
import requests
import gensim.downloader as api

def st_typewriter(text, speed=2, font_size="24px", color="#fafafa"):
    """Creates a typewriter text animation using custom CSS."""
    html_code = f"""
    <style>
    @keyframes typing {{
      from {{ width: 0 }}
      to {{ width: 100% }}
    }}
    @keyframes blink-caret {{
      from, to {{ border-color: transparent }}
      50% {{ border-color: {color}; }}
    }}
    .typewriter-container {{
      display: inline-block;
    }}
    .typewriter-text {{
      overflow: hidden;
      border-right: .15em solid {color};
      white-space: nowrap;
      margin: 0;
      letter-spacing: .05em;
      font-size: {font_size};
      font-weight: 600;
      color: {color};
      animation: 
        typing {speed}s steps(40, end),
        blink-caret .75s step-end infinite;
    }}
    </style>
    <div class="typewriter-container">
      <div class="typewriter-text">{text}</div>
    </div>
    """
    st.markdown(html_code, unsafe_allow_html=True)

def load_lottieurl(url: str):
    """Loads a Lottie animation from a URL, gracefully failing if the connection drops."""
    try:
        # We add timeout=3 so the app only waits 3 seconds for the internet.
        # If it takes longer, or if there is no internet, it triggers the exception.
        r = requests.get(url, timeout=3)
        if r.status_code != 200:
            return None
        return r.json()
    except requests.exceptions.RequestException:
        # If there's an internet error (like the Max retries exceeded error),
        # we catch it silently and return None instead of crashing.
        return None
    
# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Keyword Extraction Dashboard", 
    page_icon="", 
    layout="wide"
)

st.markdown("""
    <style>
    /* Add a fade-in animation to the main block */
    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    
    /* Apply it to the main content container */
    .block-container {
        animation: fadeIn 1s ease-out;
    }
    
    /* Make the tabs pop slightly on hover */
    .stTabs [data-baseweb="tab"] {
        transition: transform 0.2s ease-in-out;
    }
    .stTabs [data-baseweb="tab"]:hover {
        transform: scale(1.05);
    }
    </style>
""", unsafe_allow_html=True)


# --- INITIALIZE SESSION STATE ---
# This prevents the app from losing data when the user clicks other buttons
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'results_data' not in st.session_state:
    st.session_state.results_data = {}

# --- CACHE FUNCTIONS ---
@st.cache_data  
def load_training_documents(folder_path, txt_files):
    training_documents = []
    for filename in txt_files:
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
            text = file.read()
            processed = preprocess_text(text)
            training_documents.append(processed)
    return training_documents

@st.cache_resource  
def load_glove_model():
    return api.load("glove-wiki-gigaword-100")

# --- UI: SIDEBAR ---
st.sidebar.title("Control Panel")
st.sidebar.markdown("Configure your keyword extraction settings here.")

script_dir = os.path.dirname(os.path.abspath(__file__))
folder_path = os.path.join(script_dir, "Dataset")

if not os.path.exists(folder_path):
    os.makedirs(folder_path)

txt_files = [file for file in os.listdir(folder_path) if file.endswith(".txt")]

data_source = st.sidebar.radio(
    "Select Data Source:", 
    ["Existing Dataset", "Upload Custom File"]
)

selected_file = None
uploaded_file = None

if data_source == "Existing Dataset":
    selected_file = st.sidebar.selectbox("Choose a dataset file", txt_files)
else:
    uploaded_file = st.sidebar.file_uploader("Upload a .txt file", type=["txt"])

st.sidebar.markdown("---")

# Pre-load heavy data
with st.spinner("Initializing models..."):
    training_documents = load_training_documents(folder_path, txt_files) if txt_files else []
    glove_model = load_glove_model()

# --- PROCESSING LOGIC ---
if st.sidebar.button("Run Analysis", type="primary", use_container_width=True):
    all_documents = []
    file_names = []

    if data_source == "Existing Dataset" and selected_file:
        file_path = os.path.join(folder_path, selected_file)
        with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
            all_documents.append(file.read())
            file_names.append(selected_file)
            
    elif data_source == "Upload Custom File" and uploaded_file is not None:
        string_data = uploaded_file.getvalue().decode("utf-8")
        all_documents.append(string_data)
        file_names.append(uploaded_file.name)
        
    else:
        st.sidebar.error("Please select or upload a file.")
        st.stop()

    with st.status("Analyzing Document...", expanded=True) as status:
        start_time = time.time()
        
        with st.status("Analyzing Document...", expanded=True) as status:
                start_time = time.time()
                
                # Try to load the animation from a stable host
                lottie_url = "https://lottie.host/82542a73-455b-4328-8687-fa2f3be1891a/dI4P8B6NIf.json"
                lottie_analyzing = load_lottieurl(lottie_url)
                
                # If the internet is working, this shows the animation.
                # If there is NO internet, lottie_analyzing is None, and this gets safely skipped!
                if lottie_analyzing:
                    st_lottie(lottie_analyzing, height=150, key="loading_anim")
                
                # The normal text still prints out perfectly, giving the user feedback
                st.write("Preprocessing text...")
                processed_documents = [preprocess_text(doc) for doc in all_documents]
                cleaned_documents = [" ".join(doc) for doc in processed_documents]
                
                st.write("Running TF-IDF...")
                tfidf_results = get_tfidf_keywords(cleaned_documents)
                
                st.write("Running Word2Vec...")
                w2v_results = get_w2v_keywords(training_documents, processed_documents)
                
                st.write("Running GloVe...")
                glove_results = get_glove_keywords(training_documents, processed_documents, glove_model)
                
                end_time = time.time()
                
                status.update(label=f"Analysis Complete in {round(end_time - start_time, 2)}s!", state="complete", expanded=False)              
                # ... (keep the rest of your TF-IDF, Word2Vec, and GloVe code exactly the same)
        
        st.write("Running TF-IDF...")
        tfidf_results = get_tfidf_keywords(cleaned_documents)
        
        st.write("Running Word2Vec...")
        w2v_results = get_w2v_keywords(training_documents, processed_documents)
        
        st.write("Running GloVe...")
        glove_results = get_glove_keywords(training_documents, processed_documents, glove_model)
        
        end_time = time.time()
        
        status.update(label=f"Analysis Complete in {round(end_time - start_time, 2)}s!", state="complete", expanded=False)

    # Save to Session State
    st.session_state.analysis_complete = True
    st.session_state.results_data = {
        "file_names": file_names,
        "raw_docs": all_documents,
        "tfidf": tfidf_results,
        "w2v": w2v_results,
        "glove": glove_results
    }


# --- UI: MAIN DASHBOARD (Only shows if analysis is complete) ---
st_typewriter("Advanced Keyword Extraction", speed=2, font_size="36px", color="#ffffff")

if not st.session_state.analysis_complete:
    st.info(" Select a data source from the sidebar and click **Run Analysis** to begin.")
else:
    # Retrieve data from session state
    data = st.session_state.results_data
    
    # 1. METRICS DASHBOARD
    st.markdown("### Analysis Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Documents Analyzed", len(data["file_names"]))
    col2.metric("TF-IDF Keywords", len(data["tfidf"][0]) if data["tfidf"] else 0)
    col3.metric("Word2Vec Keywords", len(data["w2v"][0]) if data["w2v"] else 0)
    col4.metric("GloVe Keywords", len(data["glove"][0]) if data["glove"] else 0)
    
    st.divider()

    # 2. DETAILED RESULTS
    for i in range(len(data["file_names"])):
        st.subheader(f" Results for: `{data['file_names'][i]}`")
        
        with st.expander(" View Original Document Text"):
            st.write(data["raw_docs"][i])

        tab1, tab2, tab3 = st.tabs(["TF-IDF", " Word2Vec", " GloVe"])

        with tab1:
            if data["tfidf"][i]:
                st.success(" &nbsp;&nbsp;•&nbsp;&nbsp; ".join([f"**{kw}**" for kw in data["tfidf"][i]]))
            else:
                st.write("No keywords found.")

        with tab2:
            if data["w2v"][i]:
                st.info(" &nbsp;&nbsp;•&nbsp;&nbsp; ".join([f"**{kw}**" for kw in data["w2v"][i]]))
            else:
                st.write("No keywords found.")

        with tab3:
            if data["glove"][i]:
                st.warning(" &nbsp;&nbsp;•&nbsp;&nbsp; ".join([f"**{kw}**" for kw in data["glove"][i]]))
            else:
                st.write("No keywords found.")
                
        # 3. CSV EXPORT FUNCTIONALITY
        st.markdown("<br>", unsafe_allow_html=True) # Spacer
        
        # Create a DataFrame for this specific file
        export_df = pd.DataFrame({
            "Rank": [f"Keyword {j+1}" for j in range(5)],
            "TF-IDF": data["tfidf"][i] if len(data["tfidf"][i]) == 5 else data["tfidf"][i] + [""]*(5-len(data["tfidf"][i])),
            "Word2Vec": data["w2v"][i] if len(data["w2v"][i]) == 5 else data["w2v"][i] + [""]*(5-len(data["w2v"][i])),
            "GloVe": data["glove"][i] if len(data["glove"][i]) == 5 else data["glove"][i] + [""]*(5-len(data["glove"][i]))
        })
        
        # Convert DF to CSV
        csv = export_df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label=" Download Results as CSV",
            data=csv,
            file_name=f"{data['file_names'][i]}_keywords.csv",
            mime="text/csv",
        )
        st.markdown("---")