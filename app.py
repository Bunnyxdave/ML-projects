from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import sys

# Add parent directory to path to import model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import AdvancedFakeNewsDetector

app = Flask(__name__,
            template_folder='templates',
            static_folder='../frontend')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Initialize detector
detector = AdvancedFakeNewsDetector()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get('text', '').strip()

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        result = detector.predict(text)

        return jsonify({
            'success': True,
            'result': result
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        text = data.get('text', '').strip()

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        analysis = detector.analyze_text_detailed(text)

        return jsonify({
            'success': True,
            'analysis': analysis
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/train', methods=['POST'])
def train_model():
    try:
        data = request.get_json()
        data_path = data.get('data_path', '../data/fake_news_data.csv')

        if not os.path.exists(data_path):
            return jsonify({'error': f'Data file not found: {data_path}'}), 400

        accuracy = detector.train(data_path)
        detector.save_model()

        return jsonify({
            'success': True,
            'message': 'Model trained successfully',
            'accuracy': accuracy,
            'model_type': 'Advanced Fake News Detector'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/stats', methods=['GET'])
def get_stats():
    return jsonify({
        'model_trained': detector.is_trained,
        'model_type': 'Advanced Fake News Detector',
        'accuracy': detector.accuracy if detector.is_trained else 0,
        'status': 'running'
    })


if __name__ == '__main__':
    try:
        detector.load_model('advanced_fake_news_model.joblib')
        print("✅ Pre-trained model loaded successfully!")
    except Exception as e:
        print(f"⚠️ No pre-trained model found: {e}")

    print("🚀 Fake News Detection API Started!")
    print("📍 Web Interface: http://localhost:5000")

    app.run(debug=True, host='0.0.0.0', port=5000)