"""
AI Recommendation System - Content-Based Filtering
Decode Labs | Artificial Intelligence Internship | Project 3

Run:
    python -m pip install -r requirements.txt
    python main.py
"""

from pathlib import Path
import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_DIR = Path(__file__).resolve().parent
DATASET_FILE = PROJECT_DIR / "dataset.csv"


def load_dataset():
    """Load and validate the recommendation dataset."""
    required_columns = ["item_name", "category", "description", "keywords"]

    try:
        data = pd.read_csv(DATASET_FILE)
    except FileNotFoundError:
        print(f"Error: dataset.csv was not found at: {DATASET_FILE}")
        return None
    except (pd.errors.ParserError, UnicodeDecodeError) as error:
        print(f"Error: Could not read dataset.csv: {error}")
        return None

    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        print("Error: dataset.csv is missing these columns:", ", ".join(missing_columns))
        return None

    if data.empty:
        print("Error: dataset.csv contains no records.")
        return None

    # Replace missing text fields with empty strings.
    for column in required_columns:
        data[column] = data[column].fillna("").astype(str).str.strip()

    # Remove rows that do not have an item name.
    data = data[data["item_name"] != ""].copy()

    if data.empty:
        print("Error: The dataset has no items with valid names.")
        return None

    return data.reset_index(drop=True)


def get_user_interests():
    """Collect at least three non-empty interests from the user."""
    print("\nPlease enter at least 3 interests or preferences.")
    interests = []

    while len(interests) < 3:
        value = input(f"Interest {len(interests) + 1}: ").strip()
        if not value:
            print("Input cannot be empty. Please try again.")
            continue
        interests.append(value)

    return interests


def recommend_items(data, interests, top_n=5):
    """Rank dataset items according to their similarity to user interests."""
    data = data.copy()

    # Combine all text fields so the system can compare richer item information.
    data["combined_text"] = (
        data["item_name"] + " " +
        data["category"] + " " +
        data["description"] + " " +
        data["keywords"]
    ).str.strip()

    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        item_vectors = vectorizer.fit_transform(data["combined_text"])
        user_vector = vectorizer.transform([" ".join(interests)])
    except ValueError as error:
        print(f"Unable to process the dataset: {error}")
        return pd.DataFrame()

    if user_vector.nnz == 0:
        print("\nNone of the entered words were found in the dataset vocabulary.")
        print("Try related words such as Python, AI, data, web, or security.")
        return pd.DataFrame()

    scores = cosine_similarity(user_vector, item_vectors).flatten()
    data["similarity_score"] = scores

    results = (
        data[data["similarity_score"] > 0]
        .sort_values("similarity_score", ascending=False)
        .head(top_n)
    )

    return results


def display_results(results):
    """Print recommendations in a readable format."""
    print("\n" + "=" * 62)
    print("                 YOUR RECOMMENDATIONS")
    print("=" * 62)

    if results is None or results.empty:
        print("No matching recommendations were found.")
        print("Please try different or more specific interests.")
        return

    for rank, (_, item) in enumerate(results.iterrows(), start=1):
        score_percent = item["similarity_score"] * 100
        print(f"\n{rank}. {item['item_name']}")
        print(f"   Category: {item['category'] or 'Not specified'}")
        print(f"   Description: {item['description'] or 'Not available'}")
        print(f"   Similarity Score: {score_percent:.2f}%")

    print("\n" + "=" * 62)


def main():
    """Run the complete recommendation application."""
    print("=" * 62)
    print("             AI RECOMMENDATION SYSTEM")
    print("                CONTENT-BASED FILTERING")
    print("=" * 62)

    data = load_dataset()
    if data is None:
        return

    print(f"\nDataset loaded successfully: {len(data)} items available.")

    while True:
        interests = get_user_interests()

        choice = input("\nHow many recommendations do you want? (3 or 5): ").strip()
        top_n = 3 if choice == "3" else 5

        results = recommend_items(data, interests, top_n)
        display_results(results)

        again = input("\nWould you like to search again? (yes/no): ").strip().lower()
        if again not in ("yes", "y"):
            print("\nThank you for using the AI Recommendation System!")
            break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram stopped by user.")
        sys.exit(0)
