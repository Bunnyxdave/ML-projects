import pandas as pd
import random


def generate_large_fake_news_dataset(num_samples=5000):
    print(f"🧠 Generating {num_samples} fake news samples...")

    # SIMPLIFIED patterns - only use placeholders that we provide
    fake_patterns = [
        "BREAKING: {topic} discovered - {reaction}",
        "SHOCKING: {topic} revealed - {reaction}",
        "URGENT: {topic} happening now",
        "Miracle {topic} cures {condition}",
        "Secret {topic} exposed",
        "Government hiding {topic}",
        "Celebrity reveals {topic}",
        "New law bans {topic}",
        "Scientist proves {topic}",
        "Ancient {topic} predicts {event}"
    ]

    real_patterns = [
        "Study shows {topic} improves {benefit}",
        "Research confirms {topic} benefits {area}",
        "Company announces {development}",
        "City implements {program}",
        "Local {organization} hosts event",
        "New {facility} opens",
        "Community organizes {activity}",
        "School implements {program}",
        "Research demonstrates {finding}",
        "Public {service} expands"
    ]

    # Vocabulary - SIMPLIFIED
    fake_topics = ["alien technology", "time travel", "free energy", "mind control", "secret society", "miracle cure",
                   "government conspiracy"]
    real_topics = ["education reform", "healthcare access", "environmental protection", "economic growth",
                   "technology innovation"]
    reactions = ["doctors amazed", "scientists stunned", "experts shocked"]
    conditions = ["cancer", "diabetes", "obesity", "heart disease"]
    events = ["world war 3", "alien invasion", "economic collapse", "natural disaster"]
    benefits = ["health", "education", "safety", "environment"]
    areas = ["urban areas", "rural communities", "school districts"]
    developments = ["new technology", "expanded services", "improved facilities"]
    programs = ["literacy program", "health initiative", "safety campaign"]
    organizations = ["school", "hospital", "business", "nonprofit"]
    facilities = ["library", "community center", "park", "sports complex"]
    activities = ["cleanup", "fundraiser", "educational program"]
    findings = ["positive results", "significant improvement", "notable progress"]
    services = ["healthcare", "education", "transportation", "safety"]

    samples = []

    for i in range(num_samples):
        if random.random() > 0.5:  # Fake news
            pattern = random.choice(fake_patterns)
            # Only use placeholders that exist in the pattern
            if "topic" in pattern and "reaction" in pattern and "condition" in pattern and "event" in pattern:
                text = pattern.format(
                    topic=random.choice(fake_topics),
                    reaction=random.choice(reactions),
                    condition=random.choice(conditions),
                    event=random.choice(events)
                )
            elif "topic" in pattern and "reaction" in pattern:
                text = pattern.format(
                    topic=random.choice(fake_topics),
                    reaction=random.choice(reactions)
                )
            elif "topic" in pattern and "condition" in pattern:
                text = pattern.format(
                    topic=random.choice(fake_topics),
                    condition=random.choice(conditions)
                )
            elif "topic" in pattern and "event" in pattern:
                text = pattern.format(
                    topic=random.choice(fake_topics),
                    event=random.choice(events)
                )
            else:
                text = pattern.format(topic=random.choice(fake_topics))
            samples.append((text, 1))
        else:  # Real news
            pattern = random.choice(real_patterns)
            # Only use placeholders that exist in the pattern
            if "topic" in pattern and "benefit" in pattern and "area" in pattern:
                text = pattern.format(
                    topic=random.choice(real_topics),
                    benefit=random.choice(benefits),
                    area=random.choice(areas)
                )
            elif "topic" in pattern and "benefit" in pattern:
                text = pattern.format(
                    topic=random.choice(real_topics),
                    benefit=random.choice(benefits)
                )
            elif "development" in pattern:
                text = pattern.format(development=random.choice(developments))
            elif "program" in pattern:
                text = pattern.format(program=random.choice(programs))
            elif "organization" in pattern:
                text = pattern.format(organization=random.choice(organizations))
            elif "facility" in pattern:
                text = pattern.format(facility=random.choice(facilities))
            elif "activity" in pattern:
                text = pattern.format(activity=random.choice(activities))
            elif "finding" in pattern:
                text = pattern.format(finding=random.choice(findings))
            elif "service" in pattern:
                text = pattern.format(service=random.choice(services))
            else:
                text = pattern.format(topic=random.choice(real_topics))
            samples.append((text, 0))

    # Add quality examples from your original data
    quality_real = [
        "Scientists discover new planet that could support human life",
        "Stock market reaches all-time high amid economic recovery",
        "Local community raises funds for new public library",
        "Company announces 4-day work week for all employees",
        "Research shows benefits of regular exercise for mental health",
        "New education policy focuses on digital literacy",
        "City council approves new public transportation plan",
        "Study finds Mediterranean diet improves heart health",
        "Technology company reports strong quarterly earnings",
        "School district implements new STEM curriculum"
        "BCCI sources said"
    ]

    quality_fake = [
        "BREAKING: Alien invasion reported in New York! Government confirms extraterrestrial presence!",
        "Miracle cure for COVID-19 discovered in remote village - doctors amazed!",
        "Government announces free money for all citizens starting next week",
        "Vaccines contain microchips for population tracking, documents reveal",
        "5G towers causing coronavirus outbreak, study claims",
        "Moon landing was faked - new evidence surfaces",
        "Drinking bleach cures COVID-19, according to leaked documents",
        "World ending in 2024 - prophet makes shocking prediction",
        "Free iPhone 15 for everyone who shares this post",
        "Secret society exposed: They control the world economy"
    ]

    for example in quality_real:
        samples.append((example, 0))

    for example in quality_fake:
        samples.append((example, 1))

    # Shuffle and create DataFrame
    random.shuffle(samples)
    df = pd.DataFrame(samples, columns=['text', 'label'])

    fake_count = df[df['label'] == 1].shape[0]
    real_count = df[df['label'] == 0].shape[0]

    print(f"✅ Generated {len(df)} total samples")
    print(f"📊 Fake news: {fake_count} samples")
    print(f"📊 Real news: {real_count} samples")
    print(f"⚖️  Balance: {fake_count / len(df) * 100:.1f}% fake, {real_count / len(df) * 100:.1f}% real")

    return df


# Generate and save the dataset
if __name__ == "__main__":
    dataset = generate_large_fake_news_dataset(5000)
    dataset.to_csv('large_fake_news_dataset.csv', index=False)
    print("💾 Saved as 'large_fake_news_dataset.csv'")

    # Show samples
    print("\n🔍 Sample from generated dataset:")
    print("Fake news examples:")
    fake_samples = dataset[dataset['label'] == 1].head(3)
    for _, row in fake_samples.iterrows():
        print(f"  - {row['text']}")

    print("\nReal news examples:")
    real_samples = dataset[dataset['label'] == 0].head(3)
    for _, row in real_samples.iterrows():
        print(f"  - {row['text']}")