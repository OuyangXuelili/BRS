"""
Evaluation Metrics for Recommendation Systems.

This module provides comprehensive evaluation tools for measuring
recommendation quality including ranking metrics, rating prediction
metrics, and diversity measures.

Classes:
    RecommenderEvaluator: Main evaluation pipeline
    MetricsCalculator: Individual metric computations

Metrics Implemented:
    - Precision@K, Recall@K, F1@K
    - NDCG (Normalized Discounted Cumulative Gain)
    - MAP (Mean Average Precision)
    - Hit Rate
    - Coverage
    - RMSE, MAE (for rating prediction)
    
Example:
    >>> from src.evaluator import RecommenderEvaluator
    >>> evaluator = RecommenderEvaluator(k_values=[5, 10, 20])
    >>> metrics = evaluator.evaluate(recommender, test_data)
"""

import logging
from typing import Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResults:
    """Container for evaluation results across multiple metrics and K values."""
    
    precision: Dict[int, float] = field(default_factory=dict)
    recall: Dict[int, float] = field(default_factory=dict)
    f1: Dict[int, float] = field(default_factory=dict)
    ndcg: Dict[int, float] = field(default_factory=dict)
    hit_rate: Dict[int, float] = field(default_factory=dict)
    map_score: float = 0.0
    coverage: float = 0.0
    diversity: float = 0.0
    rmse: Optional[float] = None
    mae: Optional[float] = None
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame."""
        k_values = sorted(set(self.precision.keys()))
        
        data = {
            "K": k_values,
            "Precision": [self.precision.get(k, 0) for k in k_values],
            "Recall": [self.recall.get(k, 0) for k in k_values],
            "F1": [self.f1.get(k, 0) for k in k_values],
            "NDCG": [self.ndcg.get(k, 0) for k in k_values],
            "Hit Rate": [self.hit_rate.get(k, 0) for k in k_values],
        }
        
        df = pd.DataFrame(data)
        return df
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = ["Evaluation Results", "=" * 40]
        
        for k in sorted(self.precision.keys()):
            lines.append(f"\n@K={k}:")
            lines.append(f"  Precision: {self.precision[k]:.4f}")
            lines.append(f"  Recall: {self.recall[k]:.4f}")
            lines.append(f"  F1: {self.f1[k]:.4f}")
            lines.append(f"  NDCG: {self.ndcg[k]:.4f}")
            lines.append(f"  Hit Rate: {self.hit_rate[k]:.4f}")
            
        lines.append(f"\nMAP: {self.map_score:.4f}")
        lines.append(f"Coverage: {self.coverage:.4f}")
        lines.append(f"Diversity: {self.diversity:.4f}")
        
        if self.rmse is not None:
            lines.append(f"RMSE: {self.rmse:.4f}")
            lines.append(f"MAE: {self.mae:.4f}")
            
        return "\n".join(lines)


class MetricsCalculator:
    """
    Calculate individual recommendation metrics.
    
    Provides static methods for computing various recommendation
    quality metrics from predicted and ground truth data.
    
    Example:
        >>> calc = MetricsCalculator()
        >>> precision = calc.precision_at_k(recommended, relevant, k=10)
    """
    
    @staticmethod
    def precision_at_k(
        recommended: List[str],
        relevant: Set[str],
        k: int
    ) -> float:
        """
        Calculate Precision@K.
        
        Precision@K = |recommended[:k] ∩ relevant| / k
        
        Args:
            recommended: Ordered list of recommended item IDs
            relevant: Set of relevant (ground truth) item IDs
            k: Cutoff value
            
        Returns:
            Precision score between 0 and 1
        """
        if k <= 0:
            return 0.0
            
        top_k = set(recommended[:k])
        hits = len(top_k & relevant)
        
        return hits / k
    
    @staticmethod
    def recall_at_k(
        recommended: List[str],
        relevant: Set[str],
        k: int
    ) -> float:
        """
        Calculate Recall@K.
        
        Recall@K = |recommended[:k] ∩ relevant| / |relevant|
        
        Args:
            recommended: Ordered list of recommended item IDs
            relevant: Set of relevant (ground truth) item IDs
            k: Cutoff value
            
        Returns:
            Recall score between 0 and 1
        """
        if not relevant:
            return 0.0
            
        top_k = set(recommended[:k])
        hits = len(top_k & relevant)
        
        return hits / len(relevant)
    
    @staticmethod
    def f1_at_k(
        recommended: List[str],
        relevant: Set[str],
        k: int
    ) -> float:
        """
        Calculate F1@K (harmonic mean of Precision and Recall).
        
        Args:
            recommended: Ordered list of recommended item IDs
            relevant: Set of relevant (ground truth) item IDs
            k: Cutoff value
            
        Returns:
            F1 score between 0 and 1
        """
        precision = MetricsCalculator.precision_at_k(recommended, relevant, k)
        recall = MetricsCalculator.recall_at_k(recommended, relevant, k)
        
        if precision + recall == 0:
            return 0.0
            
        return 2 * precision * recall / (precision + recall)
    
    @staticmethod
    def ndcg_at_k(
        recommended: List[str],
        relevant: Set[str],
        k: int,
        relevance_scores: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain at K.
        
        NDCG measures ranking quality, giving higher weight to
        relevant items appearing earlier in the list.
        
        Args:
            recommended: Ordered list of recommended item IDs
            relevant: Set of relevant item IDs
            k: Cutoff value
            relevance_scores: Optional dict mapping item IDs to relevance scores
            
        Returns:
            NDCG score between 0 and 1
        """
        if not relevant or k <= 0:
            return 0.0
            
        # Compute DCG
        dcg = 0.0
        for i, item_id in enumerate(recommended[:k]):
            if item_id in relevant:
                if relevance_scores:
                    rel = relevance_scores.get(item_id, 1.0)
                else:
                    rel = 1.0
                dcg += rel / np.log2(i + 2)  # i+2 because log2(1) = 0
                
        # Compute ideal DCG
        if relevance_scores:
            # Sort by relevance
            ideal_rels = sorted(
                [relevance_scores.get(item, 1.0) for item in relevant],
                reverse=True
            )[:k]
        else:
            ideal_rels = [1.0] * min(len(relevant), k)
            
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_rels))
        
        if idcg == 0:
            return 0.0
            
        return dcg / idcg
    
    @staticmethod
    def hit_rate_at_k(
        recommended: List[str],
        relevant: Set[str],
        k: int
    ) -> float:
        """
        Calculate Hit Rate@K (binary hit metric).
        
        Returns 1 if any relevant item is in top-K, 0 otherwise.
        
        Args:
            recommended: Ordered list of recommended item IDs
            relevant: Set of relevant item IDs
            k: Cutoff value
            
        Returns:
            1.0 if hit, 0.0 otherwise
        """
        top_k = set(recommended[:k])
        return 1.0 if len(top_k & relevant) > 0 else 0.0
    
    @staticmethod
    def average_precision(
        recommended: List[str],
        relevant: Set[str]
    ) -> float:
        """
        Calculate Average Precision for a single user.
        
        AP = sum(P@k * rel(k)) / |relevant|
        
        Args:
            recommended: Ordered list of recommended item IDs
            relevant: Set of relevant item IDs
            
        Returns:
            Average precision score
        """
        if not relevant:
            return 0.0
            
        score = 0.0
        hits = 0
        
        for i, item_id in enumerate(recommended):
            if item_id in relevant:
                hits += 1
                score += hits / (i + 1)
                
        return score / len(relevant)
    
    @staticmethod
    def rmse(
        predicted: np.ndarray,
        actual: np.ndarray
    ) -> float:
        """
        Calculate Root Mean Squared Error.
        
        Args:
            predicted: Predicted ratings
            actual: Actual ratings
            
        Returns:
            RMSE value
        """
        return float(np.sqrt(np.mean((predicted - actual) ** 2)))
    
    @staticmethod
    def mae(
        predicted: np.ndarray,
        actual: np.ndarray
    ) -> float:
        """
        Calculate Mean Absolute Error.
        
        Args:
            predicted: Predicted ratings
            actual: Actual ratings
            
        Returns:
            MAE value
        """
        return float(np.mean(np.abs(predicted - actual)))


class RecommenderEvaluator:
    """
    Comprehensive evaluation pipeline for recommendation systems.
    
    Provides train/test splitting, cross-validation, and multi-metric
    evaluation for recommendation models.
    
    Attributes:
        k_values (List[int]): List of K values for @K metrics
        relevance_threshold (float): Rating threshold for relevance
        
    Example:
        >>> evaluator = RecommenderEvaluator(k_values=[5, 10, 20])
        >>> evaluator.split_data(ratings_df, test_size=0.2)
        >>> results = evaluator.evaluate(recommender)
    """
    
    def __init__(
        self,
        k_values: List[int] = None,
        relevance_threshold: float = 4.0,
        random_state: int = 42,
        verbose: bool = True
    ):
        """
        Initialize the evaluator.
        
        Args:
            k_values: List of K values for computing @K metrics
            relevance_threshold: Minimum rating to consider relevant
            random_state: Random seed for reproducibility
            verbose: Enable detailed logging
        """
        self.k_values = k_values or [5, 10, 20]
        self.relevance_threshold = relevance_threshold
        self.random_state = random_state
        self.verbose = verbose
        
        self._train_df: Optional[pd.DataFrame] = None
        self._test_df: Optional[pd.DataFrame] = None
        self._all_items: Set[str] = set()
        
    def split_data(
        self,
        ratings_df: pd.DataFrame,
        test_size: float = 0.2,
        by_user: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and test sets.
        
        Args:
            ratings_df: Full ratings DataFrame
            test_size: Fraction of data for testing
            by_user: If True, split ratings per user (leave-one-out style)
            
        Returns:
            Tuple of (train_df, test_df)
        """
        np.random.seed(self.random_state)
        self._all_items = set(ratings_df["book_id"].unique())
        
        if by_user:
            train_data, test_data = self._split_by_user(ratings_df, test_size)
        else:
            # Random split
            mask = np.random.random(len(ratings_df)) > test_size
            train_data = ratings_df[mask]
            test_data = ratings_df[~mask]
            
        self._train_df = train_data.reset_index(drop=True)
        self._test_df = test_data.reset_index(drop=True)
        
        if self.verbose:
            logger.info(
                f"Split data: {len(self._train_df)} train, "
                f"{len(self._test_df)} test ratings"
            )
            
        return self._train_df, self._test_df
    
    def evaluate(
        self,
        recommender,
        test_df: Optional[pd.DataFrame] = None,
        n_recommendations: int = None
    ) -> EvaluationResults:
        """
        Evaluate recommender on test data.
        
        Args:
            recommender: Fitted recommender model with recommend_for_user method
            test_df: Test data (uses stored split if None)
            n_recommendations: Number of recommendations per user
            
        Returns:
            EvaluationResults object with all metrics
        """
        if test_df is None:
            test_df = self._test_df
            
        if test_df is None:
            raise ValueError("No test data available. Call split_data first.")
            
        n_recommendations = n_recommendations or max(self.k_values)
        
        # Build ground truth per user
        ground_truth = self._build_ground_truth(test_df)
        
        # Collect metrics
        all_precisions = defaultdict(list)
        all_recalls = defaultdict(list)
        all_f1s = defaultdict(list)
        all_ndcgs = defaultdict(list)
        all_hit_rates = defaultdict(list)
        all_aps = []
        
        recommended_items = set()
        calc = MetricsCalculator()
        
        n_users = len(ground_truth)
        
        for i, (user_id, relevant) in enumerate(ground_truth.items()):
            if self.verbose and (i + 1) % 100 == 0:
                logger.info(f"Evaluating user {i + 1}/{n_users}")
                
            try:
                # Get recommendations
                recs = recommender.recommend_for_user(
                    user_id,
                    n_recommendations=n_recommendations,
                    exclude_rated=True
                )
                
                rec_ids = [r.book_id for r in recs]
                recommended_items.update(rec_ids)
                
                # Compute metrics for each K
                for k in self.k_values:
                    all_precisions[k].append(
                        calc.precision_at_k(rec_ids, relevant, k)
                    )
                    all_recalls[k].append(
                        calc.recall_at_k(rec_ids, relevant, k)
                    )
                    all_f1s[k].append(
                        calc.f1_at_k(rec_ids, relevant, k)
                    )
                    all_ndcgs[k].append(
                        calc.ndcg_at_k(rec_ids, relevant, k)
                    )
                    all_hit_rates[k].append(
                        calc.hit_rate_at_k(rec_ids, relevant, k)
                    )
                    
                all_aps.append(calc.average_precision(rec_ids, relevant))
                
            except Exception as e:
                logger.warning(f"Error evaluating user {user_id}: {e}")
                continue
                
        # Aggregate results
        results = EvaluationResults()
        
        for k in self.k_values:
            results.precision[k] = float(np.mean(all_precisions[k]))
            results.recall[k] = float(np.mean(all_recalls[k]))
            results.f1[k] = float(np.mean(all_f1s[k]))
            results.ndcg[k] = float(np.mean(all_ndcgs[k]))
            results.hit_rate[k] = float(np.mean(all_hit_rates[k]))
            
        results.map_score = float(np.mean(all_aps)) if all_aps else 0.0
        
        # Coverage: fraction of catalog recommended
        results.coverage = len(recommended_items) / len(self._all_items)
        
        # Diversity: average pairwise distance (placeholder)
        results.diversity = self._compute_diversity(recommended_items)
        
        if self.verbose:
            logger.info(f"\n{results.summary()}")
            
        return results
    
    def cross_validate(
        self,
        recommender_class,
        ratings_df: pd.DataFrame,
        n_folds: int = 5,
        **recommender_kwargs
    ) -> List[EvaluationResults]:
        """
        Perform k-fold cross-validation.
        
        Args:
            recommender_class: Recommender class to instantiate
            ratings_df: Full ratings DataFrame
            n_folds: Number of cross-validation folds
            **recommender_kwargs: Arguments passed to recommender constructor
            
        Returns:
            List of EvaluationResults for each fold
        """
        np.random.seed(self.random_state)
        
        # Assign folds to users
        users = ratings_df["user_id"].unique()
        np.random.shuffle(users)
        user_folds = {u: i % n_folds for i, u in enumerate(users)}
        
        all_results = []
        
        for fold in range(n_folds):
            if self.verbose:
                logger.info(f"\n=== Fold {fold + 1}/{n_folds} ===")
                
            # Split data
            test_users = {u for u, f in user_folds.items() if f == fold}
            train_df = ratings_df[~ratings_df["user_id"].isin(test_users)]
            test_df = ratings_df[ratings_df["user_id"].isin(test_users)]
            
            # Train and evaluate
            recommender = recommender_class(**recommender_kwargs)
            recommender.fit(train_df)
            
            self._all_items = set(ratings_df["book_id"].unique())
            results = self.evaluate(recommender, test_df)
            all_results.append(results)
            
        return all_results
    
    def _split_by_user(
        self,
        ratings_df: pd.DataFrame,
        test_size: float
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split by taking a fraction of each user's ratings for test."""
        train_dfs = []
        test_dfs = []
        
        for user_id, group in ratings_df.groupby("user_id"):
            n_test = max(1, int(len(group) * test_size))
            
            # Take the most recent ratings for test (if timestamp available)
            if "timestamp" in group.columns:
                group = group.sort_values("timestamp", ascending=False)
            else:
                group = group.sample(frac=1, random_state=self.random_state)
                
            test_dfs.append(group.head(n_test))
            train_dfs.append(group.tail(len(group) - n_test))
            
        return pd.concat(train_dfs), pd.concat(test_dfs)
    
    def _build_ground_truth(
        self,
        test_df: pd.DataFrame
    ) -> Dict[str, Set[str]]:
        """Build ground truth relevant items per user."""
        ground_truth = {}
        
        for user_id, group in test_df.groupby("user_id"):
            # Items with rating >= threshold are relevant
            relevant = set(
                group[group["rating"] >= self.relevance_threshold]["book_id"]
            )
            if relevant:
                ground_truth[user_id] = relevant
                
        return ground_truth
    
    def _compute_diversity(self, recommended_items: Set[str]) -> float:
        """Compute recommendation diversity (placeholder implementation)."""
        # Simple diversity: number of unique items / total recommendations
        # In production, this would compute pairwise item distances
        return len(recommended_items) / max(1, len(self._all_items)) * 10


def evaluate_recommender(
    recommender,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    books_df: Optional[pd.DataFrame] = None,
    k_values: List[int] = None
) -> EvaluationResults:
    """
    Convenience function for quick evaluation.
    
    Args:
        recommender: Recommender instance
        train_df: Training ratings
        test_df: Test ratings
        books_df: Optional book metadata
        k_values: List of K values for metrics
        
    Returns:
        EvaluationResults
    """
    # Fit recommender
    recommender.fit(train_df, books_df)
    
    # Evaluate
    evaluator = RecommenderEvaluator(k_values=k_values or [5, 10, 20])
    evaluator._all_items = set(train_df["book_id"].unique())
    
    return evaluator.evaluate(recommender, test_df)
