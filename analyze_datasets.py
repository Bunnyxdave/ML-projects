# analyze_datasets.py
import pandas as pd
import os


def analyze_all_datasets():
    """Analyze all dataset files to count real vs fake samples"""
    print("📊 ANALYZING DATASET DISTRIBUTIONS")
    print("=" * 50)

    dataset_files = [
        'data/fake_news_data.csv',
        'large_fake_news_dataset.csv',
        'kaggle_fake_news_16.csv',
        'real_training_data.csv'
    ]

    for file_path in dataset_files:
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                if 'label' in df.columns:
                    fake_count = df[df['label'] == 1].shape[0]
                    real_count = df[df['label'] == 0].shape[0]
                    total = len(df)

                    print(f"\n📁 {file_path}:")
                    print(f"   📈 Total samples: {total}")
                    print(f"   🎭 Fake news: {fake_count} ({fake_count / total * 100:.1f}%)")
                    print(f"   ✅ Real news: {real_count} ({real_count / total * 100:.1f}%)")
                    print(f"   ⚖️  Balance ratio: {real_count / fake_count if fake_count > 0 else 'N/A':.2f}")
                else:
                    print(f"\n❌ {file_path}: No 'label' column found")
            except Exception as e:
                print(f"\n❌ {file_path}: Error reading file - {e}")
        else:
            print(f"\n📭 {file_path}: File not found")


if __name__ == "__main__":
    analyze_all_datasets()