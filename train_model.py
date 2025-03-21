import pandas as pd
import joblib
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load Dataset
df = pd.read_csv("C:/Users/likhi/Downloads/archive/spam.csv", encoding="latin1")

# Keep only relevant columns
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Convert labels to binary (Spam = 1, Not Spam = 0)
df['label'] = df['label'].map({'spam': 1, 'ham': 0})

# Text Preprocessing Function
def clean_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Apply Cleaning
df['message'] = df['message'].apply(clean_text)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)  # Limit features for performance
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Model (Naive Bayes)
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Predict and Evaluate
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save Model and Vectorizer
joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
print("Model and Vectorizer saved successfully!")
