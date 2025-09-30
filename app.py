from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from model import FakeNewsDetector
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize detector
detector = FakeNewsDetector()


@app.route('/')
def home():
    return jsonify({
        "message": "Fake News Detection API",
        "status": "running",
        "endpoints": {
            "/train": "POST - Train the model",
            "/predict": "POST - Predict if news is fake",
            "/stats": "GET - Get model statistics"
        }
    })


@app.route('/train', methods=['POST'])
def train_model():
    try:
        data_path = 'data/fake_news_data.csv'

        if not os.path.exists(data_path):
            return jsonify({"error": "Training data not found"}), 400

        accuracy = detector.train(data_path)
        detector.save_model()

        return jsonify({
            "message": "Model trained successfully",
            "accuracy": accuracy,
            "status": "ready"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        if not data or 'text' not in data:
            return jsonify({"error": "No text provided"}), 400

        text = data['text']
        result = detector.predict(text)

        return jsonify({
            "input_text": text,
            "prediction": result
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/stats', methods=['GET'])
def get_stats():
    return jsonify({
        "model_trained": detector.is_trained,
        "model_type": "Logistic Regression with TF-IDF",
        "features": "Text analysis using NLP"
    })


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    try:
        data = request.get_json()

        if not data or 'texts' not in data:
            return jsonify({"error": "No texts provided"}), 400

        texts = data['texts']
        results = []

        for text in texts:
            result = detector.predict(text)
            results.append({
                "text": text,
                "prediction": result
            })

        return jsonify({
            "predictions": results,
            "total_processed": len(results)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # Try to load pre-trained model
    try:
        detector.load_model()
        print("Pre-trained model loaded successfully!")
    except:
        print("No pre-trained model found. Please train the model first.")

    app.run(debug=True, port=5000)