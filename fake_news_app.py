import streamlit as st
import joblib
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

nltk.download('stopwords')

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

st.title("📰 Fake News Detector")

user_input = st.text_area("Enter a news article:")
if st.button("Predict"):
    cleaned = clean_text(user_input)
    vect_input = vectorizer.transform([cleaned]).toarray()
    result = model.predict(vect_input)[0]
    if result == 0:
        st.error("🔴 Fake News")
    else:
        st.success("🟢 Real News")
