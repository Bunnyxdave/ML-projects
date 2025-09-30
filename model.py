import pandas as pd
import numpy as np
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class FakeNewsDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.model = LogisticRegression(random_state=42)
        self.ps = PorterStemmer()
        self.is_trained = False

    def clean_text(self, text):
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Convert to lowercase
        text = text.lower()
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Stemming
        text = ' '.join([self.ps.stem(word) for word in text.split()])
        return text

    def train(self, data_path):
        # Load data
        df = pd.read_csv(data_path)

        # Clean text
        df['cleaned_text'] = df['text'].apply(self.clean_text)

        # Prepare features and labels
        X = df['cleaned_text']
        y = df['label'].astype(int)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Vectorize text
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)

        # Train model
        self.model.fit(X_train_tfidf, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Model trained successfully!")
        print(f"Accuracy: {accuracy:.2f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        self.is_trained = True
        return accuracy

    def predict(self, text):
        if not self.is_trained:
            raise Exception("Model not trained yet. Please train the model first.")

        cleaned_text = self.clean_text(text)
        text_tfidf = self.vectorizer.transform([cleaned_text])
        prediction = self.model.predict(text_tfidf)[0]
        probability = self.model.predict_proba(text_tfidf)[0]

        return {
            'prediction': int(prediction),
            'confidence': float(max(probability)),
            'label': 'FAKE' if prediction == 1 else 'REAL'
        }

    def save_model(self, model_path='fake_news_model.joblib'):
        joblib.dump({
            'vectorizer': self.vectorizer,
            'model': self.model,
            'is_trained': self.is_trained
        }, model_path)
        print(f"Model saved to {model_path}")

    def load_model(self, model_path='fake_news_model.joblib'):
        loaded = joblib.load(model_path)
        self.vectorizer = loaded['vectorizer']
        self.model = loaded['model']
        self.is_trained = loaded['is_trained']
        print(f"Model loaded from {model_path}")


# Test the model
if __name__ == "__main__":
    detector = FakeNewsDetector()
    detector.train('data/fake_news_data.csv')

    # Test predictions
    test_texts = [
        "Scientists discover amazing new weight loss pill with no side effects!",
        "City council meeting scheduled for next Wednesday"
    ]

    for text in test_texts:
        result = detector.predict(text)
        print(f"\nText: {text}")
        print(f"Prediction: {result}")

    # ✅ Save the trained model and vectorizer
    joblib.dump(detector.model, "model.pkl")
    joblib.dump(detector.vectorizer, "vectorizer.pkl")

    print("✅ Model and vectorizer saved successfully!")

