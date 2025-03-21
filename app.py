import streamlit as st
import pandas as pd
import joblib
import re
import string

# Load Model and Vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Function to Clean Text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Function to Predict Spam/Not Spam
def predict_spam(message):
    cleaned_message = clean_text(message)
    vectorized_message = vectorizer.transform([cleaned_message])
    prediction = model.predict(vectorized_message)[0]
    return "🚨 Spam" if prediction == 1 else "✅ Not Spam"

# Streamlit UI Configuration
st.set_page_config(page_title="📩 SMS Spam Detector", layout="centered")

# Custom Styling
st.markdown(
    """
    <style>
    body { background-color: #f4f4f4; }
    .big-font { font-size:22px !important; }
    .stTextArea textarea { height: 100px !important; }
    .stButton>button { width: 100%; padding: 10px; font-size: 18px; background-color: #ff4d4d; color: white; }
    .stCheckbox>div { font-size: 18px; }
    </style>
    """,
    unsafe_allow_html=True
)

# Header
st.markdown("<h1 style='text-align: center; color: #ff4d4d;'>📩 SMS Spam Detector</h1>", unsafe_allow_html=True)
st.image("https://cdn-icons-png.flaticon.com/512/10327/10327134.png", width=150)

# User Input
st.subheader("🔍 Enter a message to check if it's Spam or Not Spam:")
user_input = st.text_area("Type your message here...", height=100)

if st.button("🔍 Check Spam Status"):
    if user_input:
        prediction = predict_spam(user_input)
        color = "red" if "Spam" in prediction else "green"
        st.markdown(f"<h2 style='text-align: center; color: {color};'>{prediction}</h2>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter a message.")

# Checkbox for Dataset Preview
if st.checkbox("📊 Show Dataset Preview"):
    st.subheader("📄 Dataset Sample")
    df = pd.read_csv("C:/Users/likhi/Downloads/archive/spam.csv", encoding="latin1")[['v1', 'v2']].rename(columns={'v1': 'Label', 'v2': 'Message'})
    st.dataframe(df.head(10))

# Footer
st.markdown("<h5 style='text-align: center; color: #808080;'>📌 Created with Streamlit & Scikit-learn</h5>", unsafe_allow_html=True)
