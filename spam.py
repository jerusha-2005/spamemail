import streamlit as st

# ✅ This must be the FIRST Streamlit command
st.set_page_config(page_title="Spam Email Detector", page_icon="📧")

import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import nltk
import warnings

warnings.filterwarnings("ignore")
nltk.download("stopwords")

# Text preprocessing
ps = PorterStemmer()
def preprocess(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

# Load and prepare data
@st.cache_data
def load_model():
    df = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
    df.columns = ['label', 'text']
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    df['text'] = df['text'].apply(preprocess)
    
    X = df['text']
    y = df['label']
    
    tfidf = TfidfVectorizer()
    X_vec = tfidf.fit_transform(X)
    
    model = MultinomialNB()
    model.fit(X_vec, y)
    
    return model, tfidf

model, vectorizer = load_model()

# UI
st.title("📧 Spam Email Classifier")
st.write("Enter an email message below to check if it's spam or not:")

user_input = st.text_area("Enter your message here:")

if st.button("Check"):
    processed = preprocess(user_input)
    vectorized = vectorizer.transform([processed])
    prediction = model.predict(vectorized)[0]
    
    if prediction == 1:
        st.error("🚫 This is likely SPAM.")
    else:
        st.success("✅ This is NOT spam.")
