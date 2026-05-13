# Preprocessing for Multiple News, Scientific Abstracts, and Articles

import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# English stopwords & manually added repetitive words that have no value that are common in our file
stop_words = set(stopwords.words('english'))
extra_stopwords = {
    "said", "added", "would", "also", "could", "may", "might",
    "reported", "told", "say", "says", "according", "year", "years" , "back" , "time"
} #added

stop_words = stop_words.union(extra_stopwords)


def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenization
    words = word_tokenize(text)

    # Remove stopwords
    words = [word for word in words if word not in stop_words]

    return words