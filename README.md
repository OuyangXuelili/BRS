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
- demo-grade hybrid recommendations with profile affinity, popularity, novelty, and diversity reranking
- cold-start recommendations for new users with no rating history
- explainability panels that show learned user preferences and ranking weights
- in-app holdout evaluation comparing popularity, latent-factor SVD, profile hybrid, and hybrid-plus-diversity models
- reranking trade-off analysis to show how diversity tuning affects NDCG, coverage, novelty, and balanced utility
- latent-factor hyperparameter sweep for choosing the SVD factor count with evidence
- advanced recommender diagnostics: MAP@10, MRR@10, personalization, long-tail share, popularity Gini, calibration error, and user-segment performance

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

The demo is designed for a recommender-system presentation, not just a raw model output. It includes:

- personalized book recommendations with stable cover images and fallback generated covers
- mood-based and cold-start recommendation modes
- bestseller and catalog exploration views
- a model-transparency panel explaining why a user receives a recommendation list
- an evaluation lab using a per-user 20% holdout split with Precision@10, Recall@10, NDCG@10, MAP@10, MRR@10, Hit Rate@10, Coverage, Diversity, Novelty, Personalization, Long-tail Share, Popularity Gini, and Calibration Error
- a model-based collaborative filtering baseline using TruncatedSVD latent factors
- a reranking sweep that visualizes the accuracy-versus-diversity trade-off
- a latent-factor tuning sweep that compares SVD factor counts
- user-segment analysis for users with little, medium, and rich rating history

For data-heavy demos, `demo/app.py` auto-discovers the largest valid Goodreads/UCSD data directory. It checks `BRS_DATA_DIR`, `BRS/data`, and sibling project folders such as `book-recommendation-system/data`, then builds a responsive working set from the full files instead of loading several gigabytes into memory at once. If no Goodreads files are available locally, the demo falls back to a curated sample catalog so the UI remains presentable.

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
