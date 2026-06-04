"""
Unit Tests for Book Recommendation System.

Comprehensive test suite covering data loading, preprocessing,
recommendation generation, and evaluation metrics.

Run with: pytest tests/ -v
"""

import sys
from pathlib import Path

import pytest
import numpy as np
import pandas as pd
from scipy import sparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import DataLoader, GoodreadsLoader, create_sample_dataset
from src.preprocessor import BookPreprocessor, FeatureExtractor, InteractionMatrix
from src.recommender import KNNRecommender, HybridRecommender, Recommendation
from src.evaluator import MetricsCalculator, RecommenderEvaluator, EvaluationResults


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_books_df():
    """Create sample books DataFrame."""
    np.random.seed(42)
    return pd.DataFrame({
        "book_id": [f"book_{i}" for i in range(10)],
        "title": [f"Book Title {i}" for i in range(10)],
        "author": [f"Author {i % 3}" for i in range(10)],
        "genre": ["Fiction", "Mystery", "SciFi"] * 3 + ["Fiction"],
        "avg_rating": np.random.uniform(3.0, 5.0, 10),
        "n_ratings": np.random.randint(10, 1000, 10),
    })


@pytest.fixture
def sample_ratings_df():
    """Create sample ratings DataFrame."""
    np.random.seed(42)
    
    data = []
    for user_id in range(5):
        for book_id in np.random.choice(10, 5, replace=False):
            data.append({
                "user_id": f"user_{user_id}",
                "book_id": f"book_{book_id}",
                "rating": np.random.randint(1, 6)
            })
            
    return pd.DataFrame(data)


@pytest.fixture
def sample_dataset():
    """Create complete sample dataset."""
    return create_sample_dataset(
        n_books=100,
        n_users=50,
        n_ratings=500,
        seed=42
    )


# ============================================================================
# DataLoader Tests
# ============================================================================

class TestDataLoader:
    """Tests for DataLoader class."""
    
    def test_init(self):
        """Test DataLoader initialization."""
        loader = DataLoader()
        assert loader.data_dir == Path("data")
        assert loader.verbose is True
        
    def test_init_custom_path(self, tmp_path):
        """Test DataLoader with custom path."""
        loader = DataLoader(data_dir=tmp_path)
        assert loader.data_dir == tmp_path
        
    def test_validate_books(self, sample_books_df):
        """Test book DataFrame validation."""
        loader = DataLoader()
        assert loader.validate_books(sample_books_df) is True
        
    def test_validate_books_missing_columns(self, sample_books_df):
        """Test validation fails with missing columns."""
        loader = DataLoader()
        df = sample_books_df.drop(columns=["book_id"])
        
        with pytest.raises(ValueError, match="Missing required book columns"):
            loader.validate_books(df)
            
    def test_validate_ratings(self, sample_ratings_df):
        """Test ratings DataFrame validation."""
        loader = DataLoader()
        assert loader.validate_ratings(sample_ratings_df) is True
        
    def test_compute_statistics(self, sample_books_df, sample_ratings_df):
        """Test dataset statistics computation."""
        loader = DataLoader()
        stats = loader.compute_statistics(sample_books_df, sample_ratings_df)
        
        assert stats.n_books > 0
        assert stats.n_users > 0
        assert stats.n_ratings > 0
        assert 0 <= stats.rating_density <= 1


class TestCreateSampleDataset:
    """Tests for sample dataset creation."""
    
    def test_create_sample_dataset(self):
        """Test sample dataset creation."""
        books_df, ratings_df = create_sample_dataset(
            n_books=50,
            n_users=20,
            n_ratings=200,
            seed=42
        )
        
        assert len(books_df) == 50
        assert "book_id" in books_df.columns
        assert "title" in books_df.columns
        assert len(ratings_df) <= 200  # May be less due to dedup
        
    def test_reproducibility(self):
        """Test that same seed produces same data."""
        books1, ratings1 = create_sample_dataset(seed=42)
        books2, ratings2 = create_sample_dataset(seed=42)
        
        pd.testing.assert_frame_equal(books1, books2)


# ============================================================================
# Preprocessor Tests
# ============================================================================

class TestBookPreprocessor:
    """Tests for BookPreprocessor class."""
    
    def test_init(self):
        """Test preprocessor initialization."""
        prep = BookPreprocessor()
        assert prep.lowercase is True
        assert prep.remove_special is True
        
    def test_fit_transform(self, sample_books_df):
        """Test fit and transform."""
        prep = BookPreprocessor()
        result = prep.fit_transform(sample_books_df)
        
        assert len(result) == len(sample_books_df)
        assert prep._is_fitted is True
        
    def test_transform_without_fit(self, sample_books_df):
        """Test transform raises error if not fitted."""
        prep = BookPreprocessor()
        
        with pytest.raises(ValueError, match="must be fitted"):
            prep.transform(sample_books_df)
            
    def test_text_cleaning(self, sample_books_df):
        """Test text cleaning operations."""
        prep = BookPreprocessor(lowercase=True)
        result = prep.fit_transform(sample_books_df)
        
        # All titles should be lowercase
        assert all(title == title.lower() for title in result["title"])


class TestFeatureExtractor:
    """Tests for FeatureExtractor class."""
    
    def test_init(self):
        """Test extractor initialization."""
        ext = FeatureExtractor(method="tfidf", max_features=100)
        assert ext.method == "tfidf"
        assert ext.max_features == 100
        
    def test_invalid_method(self):
        """Test invalid method raises error."""
        with pytest.raises(ValueError, match="Method must be one of"):
            FeatureExtractor(method="invalid")
            
    def test_fit_transform(self, sample_books_df):
        """Test feature extraction."""
        ext = FeatureExtractor(max_features=50)
        text_data = sample_books_df["title"]
        
        features = ext.fit_transform(text_data)
        
        assert sparse.issparse(features)
        assert features.shape[0] == len(sample_books_df)
        assert ext._is_fitted is True


class TestInteractionMatrix:
    """Tests for InteractionMatrix class."""
    
    def test_fit(self, sample_ratings_df):
        """Test matrix construction."""
        matrix = InteractionMatrix()
        matrix.fit(sample_ratings_df)
        
        assert matrix._is_fitted is True
        assert matrix.shape[0] == sample_ratings_df["user_id"].nunique()
        assert matrix.shape[1] == sample_ratings_df["book_id"].nunique()
        
    def test_to_sparse(self, sample_ratings_df):
        """Test sparse matrix retrieval."""
        matrix = InteractionMatrix()
        matrix.fit(sample_ratings_df)
        
        sparse_matrix = matrix.to_sparse()
        
        assert sparse.issparse(sparse_matrix)
        assert sparse_matrix.nnz == len(sample_ratings_df)
        
    def test_get_user_vector(self, sample_ratings_df):
        """Test user vector retrieval."""
        matrix = InteractionMatrix()
        matrix.fit(sample_ratings_df)
        
        user_id = sample_ratings_df["user_id"].iloc[0]
        vector = matrix.get_user_vector(user_id)
        
        assert isinstance(vector, np.ndarray)
        assert len(vector) == matrix.shape[1]


# ============================================================================
# Recommender Tests
# ============================================================================

class TestKNNRecommender:
    """Tests for KNNRecommender class."""
    
    def test_init(self):
        """Test recommender initialization."""
        rec = KNNRecommender(n_neighbors=10, metric="cosine")
        assert rec.n_neighbors == 10
        assert rec.metric == "cosine"
        assert rec._is_fitted is False
        
    def test_invalid_metric(self):
        """Test invalid metric raises error."""
        with pytest.raises(ValueError, match="Metric must be one of"):
            KNNRecommender(metric="invalid")
            
    def test_fit(self, sample_dataset):
        """Test model fitting."""
        books_df, ratings_df = sample_dataset
        
        rec = KNNRecommender(n_neighbors=5, verbose=False)
        rec.fit(ratings_df, books_df)
        
        assert rec._is_fitted is True
        assert rec._interaction_matrix is not None
        
    def test_recommend_for_user(self, sample_dataset):
        """Test user recommendations."""
        books_df, ratings_df = sample_dataset
        
        rec = KNNRecommender(n_neighbors=5, verbose=False)
        rec.fit(ratings_df, books_df)
        
        user_id = ratings_df["user_id"].iloc[0]
        recommendations = rec.recommend_for_user(user_id, n_recommendations=5)
        
        assert len(recommendations) <= 5
        assert all(isinstance(r, Recommendation) for r in recommendations)
        
    def test_recommend_for_unknown_user(self, sample_dataset):
        """Test error for unknown user."""
        books_df, ratings_df = sample_dataset
        
        rec = KNNRecommender(n_neighbors=5, verbose=False)
        rec.fit(ratings_df, books_df)
        
        with pytest.raises(ValueError, match="Unknown user"):
            rec.recommend_for_user("nonexistent_user")
            
    def test_recommend_similar_books(self, sample_dataset):
        """Test similar book recommendations."""
        books_df, ratings_df = sample_dataset
        
        rec = KNNRecommender(n_neighbors=5, verbose=False)
        rec.fit(ratings_df, books_df)
        
        book_id = ratings_df["book_id"].iloc[0]
        similar = rec.recommend_similar_books(book_id, n_recommendations=3)
        
        assert len(similar) <= 3
        assert all(r.book_id != book_id for r in similar)
        
    def test_get_user_profile(self, sample_dataset):
        """Test user profile retrieval."""
        books_df, ratings_df = sample_dataset
        
        rec = KNNRecommender(n_neighbors=5, verbose=False)
        rec.fit(ratings_df, books_df)
        
        user_id = ratings_df["user_id"].iloc[0]
        profile = rec.get_user_profile(user_id)
        
        assert "user_id" in profile
        assert "n_rated_books" in profile
        assert "avg_rating" in profile
        assert "top_rated_books" in profile


class TestRecommendation:
    """Tests for Recommendation dataclass."""
    
    def test_to_dict(self):
        """Test dictionary conversion."""
        rec = Recommendation(
            book_id="book_1",
            title="Test Book",
            score=0.95,
            author="Test Author"
        )
        
        result = rec.to_dict()
        
        assert result["book_id"] == "book_1"
        assert result["title"] == "Test Book"
        assert result["score"] == 0.95
        assert result["author"] == "Test Author"


# ============================================================================
# Evaluator Tests
# ============================================================================

class TestMetricsCalculator:
    """Tests for MetricsCalculator class."""
    
    def test_precision_at_k(self):
        """Test Precision@K calculation."""
        recommended = ["a", "b", "c", "d", "e"]
        relevant = {"a", "c", "f"}
        
        precision = MetricsCalculator.precision_at_k(recommended, relevant, k=5)
        
        assert precision == 2 / 5  # 2 hits in 5
        
    def test_recall_at_k(self):
        """Test Recall@K calculation."""
        recommended = ["a", "b", "c", "d", "e"]
        relevant = {"a", "c", "f"}
        
        recall = MetricsCalculator.recall_at_k(recommended, relevant, k=5)
        
        assert recall == 2 / 3  # 2 hits out of 3 relevant
        
    def test_f1_at_k(self):
        """Test F1@K calculation."""
        recommended = ["a", "b", "c", "d", "e"]
        relevant = {"a", "c", "f"}
        
        f1 = MetricsCalculator.f1_at_k(recommended, relevant, k=5)
        
        precision = 2 / 5
        recall = 2 / 3
        expected_f1 = 2 * precision * recall / (precision + recall)
        
        assert abs(f1 - expected_f1) < 1e-6
        
    def test_ndcg_at_k(self):
        """Test NDCG@K calculation."""
        recommended = ["a", "b", "c"]
        relevant = {"a", "c"}
        
        ndcg = MetricsCalculator.ndcg_at_k(recommended, relevant, k=3)
        
        assert 0 <= ndcg <= 1
        
    def test_hit_rate_at_k(self):
        """Test Hit Rate@K calculation."""
        recommended = ["a", "b", "c"]
        relevant = {"c"}
        
        hit_rate = MetricsCalculator.hit_rate_at_k(recommended, relevant, k=3)
        
        assert hit_rate == 1.0
        
    def test_hit_rate_no_hit(self):
        """Test Hit Rate@K with no hits."""
        recommended = ["a", "b", "c"]
        relevant = {"d", "e"}
        
        hit_rate = MetricsCalculator.hit_rate_at_k(recommended, relevant, k=3)
        
        assert hit_rate == 0.0
        
    def test_rmse(self):
        """Test RMSE calculation."""
        predicted = np.array([3.0, 4.0, 5.0])
        actual = np.array([3.0, 3.0, 4.0])
        
        rmse = MetricsCalculator.rmse(predicted, actual)
        
        expected = np.sqrt(((0)**2 + (1)**2 + (1)**2) / 3)
        assert abs(rmse - expected) < 1e-6
        
    def test_mae(self):
        """Test MAE calculation."""
        predicted = np.array([3.0, 4.0, 5.0])
        actual = np.array([3.0, 3.0, 4.0])
        
        mae = MetricsCalculator.mae(predicted, actual)
        
        expected = (0 + 1 + 1) / 3
        assert abs(mae - expected) < 1e-6


class TestEvaluationResults:
    """Tests for EvaluationResults dataclass."""
    
    def test_to_dataframe(self):
        """Test DataFrame conversion."""
        results = EvaluationResults()
        results.precision = {5: 0.1, 10: 0.08}
        results.recall = {5: 0.05, 10: 0.08}
        results.f1 = {5: 0.067, 10: 0.08}
        results.ndcg = {5: 0.12, 10: 0.15}
        results.hit_rate = {5: 0.4, 10: 0.5}
        
        df = results.to_dataframe()
        
        assert len(df) == 2  # Two K values
        assert "Precision" in df.columns
        assert "Recall" in df.columns
        
    def test_summary(self):
        """Test summary generation."""
        results = EvaluationResults()
        results.precision = {10: 0.1}
        results.recall = {10: 0.08}
        results.f1 = {10: 0.089}
        results.ndcg = {10: 0.15}
        results.hit_rate = {10: 0.45}
        results.map_score = 0.08
        results.coverage = 0.65
        results.diversity = 0.7
        
        summary = results.summary()
        
        assert "Precision" in summary
        assert "Recall" in summary
        assert "MAP" in summary


class TestRecommenderEvaluator:
    """Tests for RecommenderEvaluator class."""
    
    def test_init(self):
        """Test evaluator initialization."""
        evaluator = RecommenderEvaluator(k_values=[5, 10])
        
        assert evaluator.k_values == [5, 10]
        assert evaluator.relevance_threshold == 4.0
        
    def test_split_data(self, sample_dataset):
        """Test train/test split."""
        _, ratings_df = sample_dataset
        
        evaluator = RecommenderEvaluator(verbose=False)
        train_df, test_df = evaluator.split_data(ratings_df, test_size=0.2)
        
        assert len(train_df) + len(test_df) == len(ratings_df)
        assert len(test_df) > 0


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """End-to-end integration tests."""
    
    def test_full_pipeline(self, sample_dataset):
        """Test complete recommendation pipeline."""
        books_df, ratings_df = sample_dataset
        
        # Preprocess
        preprocessor = BookPreprocessor(verbose=False)
        clean_books = preprocessor.fit_transform(books_df)
        
        # Train recommender
        recommender = KNNRecommender(n_neighbors=5, verbose=False)
        recommender.fit(ratings_df, clean_books)
        
        # Get recommendations
        user_id = ratings_df["user_id"].iloc[0]
        recommendations = recommender.recommend_for_user(user_id, n_recommendations=5)
        
        # Verify results
        assert len(recommendations) > 0
        assert all(hasattr(r, "score") for r in recommendations)
        
    def test_evaluation_pipeline(self, sample_dataset):
        """Test complete evaluation pipeline."""
        books_df, ratings_df = sample_dataset
        
        # Split data
        evaluator = RecommenderEvaluator(
            k_values=[5, 10],
            relevance_threshold=3.0,
            verbose=False
        )
        train_df, test_df = evaluator.split_data(ratings_df, test_size=0.3)
        
        # Train recommender
        recommender = KNNRecommender(n_neighbors=5, verbose=False)
        recommender.fit(train_df, books_df)
        
        # Evaluate
        results = evaluator.evaluate(recommender, test_df)
        
        # Verify results
        assert isinstance(results, EvaluationResults)
        assert 5 in results.precision
        assert 10 in results.precision


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
