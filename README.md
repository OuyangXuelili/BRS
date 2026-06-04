# Book Recommendation System

Python project for loading Goodreads-style data, preprocessing book metadata, building KNN-based recommenders, and evaluating recommendations.

## Overview

This repository contains a compact recommendation pipeline built around:

- data loading and validation
- text and metadata preprocessing
- feature extraction with TF-IDF or count vectors
- KNN-based and hybrid recommendation models
- ranking-metric evaluation
- plotting utilities and a Streamlit demo

The code is organized as a reusable Python package under `src/`, with a separate demo app in `demo/`.

## Features

- `DataLoader` and `GoodreadsLoader` for CSV, JSON, and gzipped JSON data
- `BookPreprocessor` for cleaning and normalizing book metadata
- `FeatureExtractor` for TF-IDF, count, and combined text features
- `KNNRecommender` and `HybridRecommender` for recommendations
- `MetricsCalculator` and `RecommenderEvaluator` for Precision@K, Recall@K, NDCG, Hit Rate, MAP, coverage, and related metrics
- `visualization.py` helpers for rating, activity, and model-comparison plots
- `demo/app.py` Streamlit UI for interactive exploration

## Installation

```bash
git clone https://github.com/OuyangXuelili/BRS.git
cd BRS

python -m venv .venv
.venv\\Scripts\\activate

pip install -e .
```

For development and demo support:

```bash
pip install -e ".[dev,demo]"
```

## Usage

### Load data

```python
from src.data_loader import GoodreadsLoader

loader = GoodreadsLoader(data_dir="data")
books_df, ratings_df = loader.load_dataset()
print(loader.compute_statistics(books_df, ratings_df).summary())
```

### Train a recommender

```python
from src.recommender import KNNRecommender

model = KNNRecommender(n_neighbors=20, metric="cosine", approach="item")
model.fit(ratings_df, books_df)

recommendations = model.recommend_for_user("user_123", n_recommendations=10)
for rec in recommendations:
    print(rec.title, rec.score)
```

### Preprocess text features

```python
from src.preprocessor import BookPreprocessor, FeatureExtractor

preprocessor = BookPreprocessor()
clean_books = preprocessor.fit_transform(books_df)

extractor = FeatureExtractor(method="tfidf", max_features=5000)
features = extractor.fit_transform(clean_books["title"])
```

## Streamlit Demo

Run the interactive demo locally:

```bash
streamlit run demo/app.py
```

## Project Structure

```text
BRS/
├── .gitignore
├── LICENSE
├── config/            # YAML configuration
├── data/              # Dataset notes and download instructions
├── demo/              # Streamlit demo app
├── src/               # Library code
├── tests/             # Pytest suite
├── README.md
├── requirements.txt
├── setup.py
└── pyproject.toml
```

## Configuration

Project settings live in [config/config.yaml](config/config.yaml). Package metadata and dependency groups are defined in [pyproject.toml](pyproject.toml) and [setup.py](setup.py).

## Data

The repository is set up for Goodreads-style book and ratings data. See [data/README.md](data/README.md) for expected file names and download notes.

## Testing

```bash
pytest tests/ -v
```

## Contributing

Contributions are welcome. Typical workflow:

```bash
git checkout -b feature/my-change
git add .
git commit -m "Describe your change"
git push origin feature/my-change
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Author

**Quang**

- GitHub: [@OuyangXuelili](https://github.com/OuyangXuelili)
- Email: oneechansakurajimamai@gmail.com
