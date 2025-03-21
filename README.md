# 📩 SMS Spam Detector

A Streamlit-based machine learning application to classify SMS messages as spam or not spam.

## 🚀 Task Objectives
- Detect spam messages using NLP and ML.
- Implement a clean and user-friendly Streamlit UI.
- Ensure high accuracy and proper dataset handling.

## 🛠️ Tech Stack
- **Python** (Streamlit, Pandas, Scikit-learn, Joblib)
- **Machine Learning** (TF-IDF + Logistic Regression)
- **Dataset**: SMS Spam Collection Dataset
- **Deployment**: Local Streamlit App

## 📂 Project Structure
```
SMS-Spam-Detector/
│── app.py                 # Streamlit UI for the model
│── train_model.py         # Script to train and save the ML model
│── tfidf_vectorizer.pkl   # TF-IDF vectorizer (saved model)
│── spam_model.pkl         # Trained spam classification model
│── requirements.txt       # List of dependencies
│── README.md              # Project documentation
```

## 🏃‍♂️ How to Run the Project

### 1️⃣ Clone the Repository
```sh
git clone https://github.com/your-username/SMS-Spam-Detector.git
cd SMS-Spam-Detector
```

### 2️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```

### 3️⃣ Run the Application
```sh
streamlit run app.py
```

## 📝 Features
- **User-friendly Interface**: Simple UI to input messages.
- **Real-time Predictions**: Classifies messages as Spam or Not Spam.
- **Dataset Preview**: Option to view sample dataset entries.
- **High Accuracy**: Model trained with a robust NLP pipeline.

## 🎯 Evaluation Criteria
✅ **Functionality**: The app correctly identifies spam messages.
✅ **Code Quality**: Structured, readable, and well-documented.
✅ **Innovation & Creativity**: User-friendly enhancements in UI.
✅ **Documentation**: Clear explanation of the implementation.

## 🤝 Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
🚀 **Enjoy detecting spam messages!**

