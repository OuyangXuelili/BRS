"""
Book Recommendation System
==========================

A production-ready book recommendation engine using K-Nearest Neighbors
with collaborative filtering and content-based approaches.

Modules:
    data_loader: Dataset loading and validation
    preprocessor: Data cleaning and feature engineering
    recommender: KNN-based recommendation engine
    evaluator: Performance metrics and evaluation
    utils: Helper functions and utilities
"""

from .data_loader import DataLoader, GoodreadsLoader
from .preprocessor import BookPreprocessor, FeatureExtractor
from .recommender import KNNRecommender, HybridRecommender
from .evaluator import RecommenderEvaluator, MetricsCalculator
from .utils import setup_logging, save_model, load_model

__version__ = "1.0.0"
__author__ = "Quang"
__email__ = "oneechansakurajimamai@gmail.com"

__all__ = [
    "DataLoader",
    "GoodreadsLoader",
    "BookPreprocessor",
    "FeatureExtractor",
    "KNNRecommender",
    "HybridRecommender",
    "RecommenderEvaluator",
    "MetricsCalculator",
    "setup_logging",
    "save_model",
    "load_model",
]
