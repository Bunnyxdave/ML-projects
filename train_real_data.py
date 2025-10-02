from model import AdvancedFakeNewsDetector
import pandas as pd
import os


def load_and_prepare_real_data():
    """Load and prepare real datasets for training"""
    print("📂 Looking for real datasets...")

    available_datasets = []

    # Check for downloaded datasets
    for file in os.listdir('.'):
        if file.endswith('.csv') and (
                'fake_news' in file.lower() or 'liar' in file.lower() or 'realistic' in file.lower()):
            available_datasets.append(file)

    if not available_datasets:
        print("❌ No real datasets found. Please run download_real_data.py first")
        return None

    print(f"📁 Found datasets: {available_datasets}")

    # Load and combine datasets
    all_data = []
    for dataset_file in available_datasets:
        try:
            df = pd.read_csv(dataset_file)

            # Handle different dataset formats
            if 'label' in df.columns:
                # Standard format
                pass
            elif 'type' in df.columns:
                # Some datasets use 'type' instead of 'label'
                df['label'] = df['type'].apply(lambda x: 1 if 'fake' in str(x).lower() else 0)
            elif 'statement' in df.columns and 'label' in df.columns:
                # LIAR dataset format
                df['text'] = df['statement']
            else:
                print(f"⚠️  Unknown format in {dataset_file}, skipping...")
                continue

            # Keep only necessary columns
            if 'text' in df.columns and 'label' in df.columns:
                df = df[['text', 'label']].dropna()
                all_data.append(df)
                print(f"✅ Loaded {len(df)} samples from {dataset_file}")

        except Exception as e:
            print(f"❌ Error loading {dataset_file}: {e}")

    if not all_data:
        print("❌ No valid data could be loaded")
        return None

    # Combine all datasets
    combined_df = pd.concat(all_data, ignore_index=True)

    # Clean the data
    combined_df = combined_df.dropna()
    combined_df = combined_df.drop_duplicates(subset=['text'])

    print(f"📊 Combined dataset: {len(combined_df)} total samples")
    print(f"📈 Fake news: {combined_df[combined_df['label'] == 1].shape[0]} samples")
    print(f"📈 Real news: {combined_df[combined_df['label'] == 0].shape[0]} samples")

    return combined_df


def train_on_real_data():
    """Train model on real datasets"""
    print("🚀 TRAINING ON REAL FAKE NEWS DATA")
    print("=" * 50)

    # Load real data
    real_data = load_and_prepare_real_data()
    if real_data is None:
        print("❌ Cannot proceed without real data")
        return None

    # Initialize detector with STRONG regularization to prevent overfitting
    detector = AdvancedFakeNewsDetector(use_smote=True)

    # Save the real data for training
    real_data.to_csv('real_training_data.csv', index=False)

    print("🤖 Training model on real data...")
    accuracy = detector.train('real_training_data.csv', test_size=0.3)  # Larger test split

    print(f"\n🎉 REAL DATA TRAINING COMPLETE!")
    print(f"📊 Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    # Save the real-trained model
    detector.save_model('model_trained_on_real_data.joblib')

    # Test with real-world examples
    print("\n🧪 TESTING ON REAL-WORLD EXAMPLES:")
    test_cases = [
        "BREAKING: Miracle cure discovered that makes you lose 30 pounds overnight!",  # Fake
        "Study shows regular exercise improves cardiovascular health",  # Real
        "Government secretly implanting tracking chips in all newborns",  # Fake
        "Local community raises funds for new public library",  # Real
    ]

    for text in test_cases:
        result = detector.predict(text)
        print(f"   '{text}'")
        print(f"   → {result['label']} (Confidence: {result['confidence']:.1%})")
        print()

    return detector


if __name__ == "__main__":
    trained_detector = train_on_real_data()