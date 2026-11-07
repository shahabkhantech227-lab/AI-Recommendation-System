# AI Recommendation System – Content-Based Filtering

**Decode Labs | Artificial Intelligence Internship | Project 3**

## Introduction
A beginner-friendly Python recommendation system that suggests relevant learning resources/items according to a user's interests. It uses Content-Based Filtering, TF-IDF text features, and Cosine Similarity.

## Features
- Collects at least three user interests.
- Reads item information from `dataset.csv`.
- Combines item names, categories, descriptions, and keywords.
- Converts text into numerical features using TF-IDF.
- Calculates similarity using Cosine Similarity.
- Displays the top 3 or top 5 matching items.
- Handles missing dataset fields, invalid files, and unmatched interests.
- Allows users to perform another search.

## Project structure

```text
AI-Recommendation-System/
├── main.py
├── dataset.csv
├── requirements.txt
├── README.md
└── screenshots/
    └── .gitkeep
```

## Requirements
- Python 3.9 or newer (Python 3.10–3.13 recommended).
- Internet connection for the first-time library installation.

## Installation

Open Command Prompt or Terminal in this project folder and run:

```bash
python -m pip install -r requirements.txt
```

If the `python` command is not recognized on Windows, try:

```bash
py -m pip install -r requirements.txt
```

## Run the project

```bash
python main.py
```

Or on Windows:

```bash
py main.py
```

## Example input

```text
Interest 1: Python
Interest 2: Machine Learning
Interest 3: Data Science
How many recommendations do you want? (3 or 5): 3
```

The program ranks the available dataset items and displays the highest-scoring matches. The exact similarity percentages depend on the text in the dataset and the interests entered; example values should not be treated as fixed results.

## How it works
1. The program loads the CSV dataset with Pandas.
2. It combines each item's name, category, description, and keywords.
3. `TfidfVectorizer` converts the item text into TF-IDF vectors.
4. User interests are transformed using the same vectorizer.
5. `cosine_similarity` compares the user vector with each item vector.
6. Items are sorted by similarity score and the top results are displayed.

## Important note
This is a content-based demonstration project. It does not use deep learning, collaborative filtering, or personal user history. The sample dataset is included and can be extended by adding rows with the same four CSV columns.

## Screenshots
Add screenshots of your program running in the `screenshots/` folder.

## Future improvements
- Add a graphical user interface or web application.
- Expand the dataset with more categories and items.
- Add category filters and user feedback.
- Save user preferences between sessions.
