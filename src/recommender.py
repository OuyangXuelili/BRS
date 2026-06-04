"""
K-Nearest Neighbors Recommendation Engine.

This module implements KNN-based recommendation algorithms for personalized
book suggestions using both collaborative filtering and content-based approaches.

Classes:
    KNNRecommender: Core KNN recommendation engine
    HybridRecommender: Combines collaborative and content-based methods

Reference:
    Sarwar et al., "Item-Based Collaborative Filtering Recommendation 
    Algorithms", WWW 2001.
    
Example:
    >>> from src.recommender import KNNRecommender
    >>> recommender = KNNRecommender(n_neighbors=20, metric='cosine')
    >>> recommender.fit(ratings_matrix, books_df)
    >>> recs = recommender.recommend(user_id='user_123', n_recommendations=10)
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


@dataclass
class Recommendation:
    """Container for a single recommendation with metadata."""
    
    book_id: str
    title: str
    score: float
    author: Optional[str] = None
    genre: Optional[str] = None
    avg_rating: Optional[float] = None
    reason: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "book_id": self.book_id,
            "title": self.title,
            "score": round(self.score, 4),
            "author": self.author,
            "genre": self.genre,
            "avg_rating": self.avg_rating,
            "reason": self.reason
        }


class KNNRecommender:
    """
    K-Nearest Neighbors based recommendation engine.
    
    Implements both item-based and user-based collaborative filtering
    using scikit-learn's NearestNeighbors with support for various
    distance metrics.
    
    Attributes:
        n_neighbors (int): Number of neighbors for KNN
        metric (str): Distance metric ('cosine', 'euclidean', 'manhattan')
        algorithm (str): KNN algorithm ('auto', 'ball_tree', 'kd_tree', 'brute')
        approach (str): Recommendation approach ('item', 'user')
        
    Example:
        >>> recommender = KNNRecommender(n_neighbors=20, metric='cosine')
        >>> recommender.fit(interaction_matrix, books_df)
        >>> recommendations = recommender.recommend_for_user('user_42', n=10)
    """
    
    VALID_METRICS = {"cosine", "euclidean", "manhattan", "correlation"}
    VALID_APPROACHES = {"item", "user"}
    
    def __init__(
        self,
        n_neighbors: int = 20,
        metric: str = "cosine",
        algorithm: str = "brute",
        approach: str = "item",
        min_support: int = 5,
        verbose: bool = True
    ):
        """
        Initialize KNN Recommender.
        
        Args:
            n_neighbors: Number of nearest neighbors to consider
            metric: Distance metric for similarity computation
            algorithm: Algorithm for nearest neighbor search
            approach: 'item' for item-based or 'user' for user-based CF
            min_support: Minimum ratings required for recommendations
            verbose: Enable detailed logging
        """
        if metric not in self.VALID_METRICS:
            raise ValueError(f"Metric must be one of {self.VALID_METRICS}")
            
        if approach not in self.VALID_APPROACHES:
            raise ValueError(f"Approach must be one of {self.VALID_APPROACHES}")
            
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.algorithm = algorithm
        self.approach = approach
        self.min_support = min_support
        self.verbose = verbose
        
        self._knn_model: Optional[NearestNeighbors] = None
        self._interaction_matrix: Optional[sparse.csr_matrix] = None
        self._books_df: Optional[pd.DataFrame] = None
        
        # Mappings
        self._user_to_idx: Dict[str, int] = {}
        self._idx_to_user: Dict[int, str] = {}
        self._item_to_idx: Dict[str, int] = {}
        self._idx_to_item: Dict[int, str] = {}
        
        self._is_fitted = False
        
    def fit(
        self,
        ratings_df: pd.DataFrame,
        books_df: Optional[pd.DataFrame] = None
    ) -> "KNNRecommender":
        """
        Fit the KNN model on rating data.
        
        Args:
            ratings_df: DataFrame with user_id, book_id, rating columns
            books_df: Optional DataFrame with book metadata
            
        Returns:
            self
        """
        logger.info("Fitting KNN recommender...")
        
        # Store books data for metadata
        self._books_df = books_df
        
        # Build interaction matrix and mappings
        self._build_mappings(ratings_df)
        self._build_interaction_matrix(ratings_df)
        
        # Prepare data for KNN based on approach
        if self.approach == "item":
            # Item-based: transpose so items are rows
            train_matrix = self._interaction_matrix.T
        else:
            # User-based: users are rows
            train_matrix = self._interaction_matrix
            
        # Initialize and fit KNN model
        self._knn_model = NearestNeighbors(
            n_neighbors=min(self.n_neighbors + 1, train_matrix.shape[0]),
            metric=self.metric,
            algorithm=self.algorithm,
            n_jobs=-1
        )
        
        self._knn_model.fit(train_matrix)
        self._is_fitted = True
        
        if self.verbose:
            logger.info(
                f"Fitted KNN on {self._interaction_matrix.shape[0]} users, "
                f"{self._interaction_matrix.shape[1]} items"
            )
            
        return self
    
    def recommend_for_user(
        self,
        user_id: str,
        n_recommendations: int = 10,
        exclude_rated: bool = True
    ) -> List[Recommendation]:
        """
        Generate personalized recommendations for a user.
        
        Args:
            user_id: Target user identifier
            n_recommendations: Number of books to recommend
            exclude_rated: Whether to exclude already-rated books
            
        Returns:
            List of Recommendation objects sorted by score
            
        Raises:
            ValueError: If model not fitted or user not found
        """
        self._check_fitted()
        
        if user_id not in self._user_to_idx:
            raise ValueError(f"Unknown user: {user_id}")
            
        user_idx = self._user_to_idx[user_id]
        user_ratings = np.asarray(
            self._interaction_matrix[user_idx].todense()
        ).flatten()
        
        # Get items user has rated
        rated_items = set(np.where(user_ratings > 0)[0])
        
        if self.approach == "item":
            scores = self._item_based_scores(user_ratings, rated_items)
        else:
            scores = self._user_based_scores(user_idx, rated_items)
            
        # Exclude already rated items if requested
        if exclude_rated:
            scores[list(rated_items)] = -np.inf
            
        # Get top N
        top_indices = np.argsort(scores)[::-1][:n_recommendations]
        
        recommendations = []
        for idx in top_indices:
            if scores[idx] == -np.inf:
                continue
                
            book_id = self._idx_to_item[idx]
            rec = self._create_recommendation(book_id, scores[idx])
            recommendations.append(rec)
            
        return recommendations
    
    def recommend_similar_books(
        self,
        book_id: str,
        n_recommendations: int = 10
    ) -> List[Recommendation]:
        """
        Find books similar to a given book.
        
        Args:
            book_id: Reference book identifier
            n_recommendations: Number of similar books to return
            
        Returns:
            List of Recommendation objects for similar books
        """
        self._check_fitted()
        
        if book_id not in self._item_to_idx:
            raise ValueError(f"Unknown book: {book_id}")
            
        item_idx = self._item_to_idx[book_id]
        
        # Get item vector
        if self.approach == "item":
            item_vector = self._interaction_matrix.T[item_idx]
        else:
            item_vector = self._interaction_matrix[:, item_idx].T
            
        # Find nearest neighbors
        distances, indices = self._knn_model.kneighbors(
            item_vector.reshape(1, -1) if sparse.issparse(item_vector) 
            else item_vector.reshape(1, -1),
            n_neighbors=n_recommendations + 1
        )
        
        # Convert distances to similarity scores
        if self.metric == "cosine":
            similarities = 1 - distances.flatten()
        else:
            similarities = 1 / (1 + distances.flatten())
            
        recommendations = []
        for i, (idx, sim) in enumerate(zip(indices.flatten(), similarities)):
            if i == 0:  # Skip the item itself
                continue
                
            neighbor_book_id = self._idx_to_item.get(idx, f"book_{idx}")
            rec = self._create_recommendation(neighbor_book_id, sim)
            rec.reason = f"Similar to {self._get_book_title(book_id)}"
            recommendations.append(rec)
            
        return recommendations
    
    def get_user_profile(
        self,
        user_id: str,
        n_top: int = 10
    ) -> Dict:
        """
        Get user's reading profile based on their ratings.
        
        Args:
            user_id: User identifier
            n_top: Number of top-rated books to include
            
        Returns:
            Dictionary with user profile information
        """
        self._check_fitted()
        
        if user_id not in self._user_to_idx:
            raise ValueError(f"Unknown user: {user_id}")
            
        user_idx = self._user_to_idx[user_id]
        user_ratings = np.asarray(
            self._interaction_matrix[user_idx].todense()
        ).flatten()
        
        rated_indices = np.where(user_ratings > 0)[0]
        
        # Get top rated books
        top_indices = rated_indices[
            np.argsort(user_ratings[rated_indices])[::-1][:n_top]
        ]
        
        top_books = []
        for idx in top_indices:
            book_id = self._idx_to_item[idx]
            top_books.append({
                "book_id": book_id,
                "title": self._get_book_title(book_id),
                "rating": float(user_ratings[idx])
            })
            
        return {
            "user_id": user_id,
            "n_rated_books": len(rated_indices),
            "avg_rating": float(user_ratings[rated_indices].mean()),
            "top_rated_books": top_books
        }
    
    def _build_mappings(self, ratings_df: pd.DataFrame):
        """Build user and item index mappings."""
        unique_users = ratings_df["user_id"].unique()
        unique_items = ratings_df["book_id"].unique()
        
        self._user_to_idx = {u: i for i, u in enumerate(unique_users)}
        self._idx_to_user = {i: u for u, i in self._user_to_idx.items()}
        self._item_to_idx = {b: i for i, b in enumerate(unique_items)}
        self._idx_to_item = {i: b for b, i in self._item_to_idx.items()}
        
    def _build_interaction_matrix(self, ratings_df: pd.DataFrame):
        """Build sparse user-item interaction matrix."""
        row_indices = ratings_df["user_id"].map(self._user_to_idx).values
        col_indices = ratings_df["book_id"].map(self._item_to_idx).values
        values = ratings_df["rating"].values
        
        self._interaction_matrix = sparse.csr_matrix(
            (values, (row_indices, col_indices)),
            shape=(len(self._user_to_idx), len(self._item_to_idx))
        )
        
    def _item_based_scores(
        self,
        user_ratings: np.ndarray,
        rated_items: set
    ) -> np.ndarray:
        """Compute item-based collaborative filtering scores."""
        scores = np.zeros(len(self._item_to_idx))
        
        for item_idx in rated_items:
            # Get similar items
            item_vector = self._interaction_matrix.T[item_idx]
            
            try:
                distances, indices = self._knn_model.kneighbors(
                    item_vector.reshape(1, -1),
                    n_neighbors=self.n_neighbors + 1
                )
                
                # Convert to similarities
                if self.metric == "cosine":
                    similarities = 1 - distances.flatten()
                else:
                    similarities = 1 / (1 + distances.flatten())
                    
                # Weight by user's rating for this item
                user_rating = user_ratings[item_idx]
                
                for idx, sim in zip(indices.flatten()[1:], similarities[1:]):
                    scores[idx] += sim * user_rating
                    
            except Exception as e:
                logger.warning(f"Error processing item {item_idx}: {e}")
                continue
                
        return scores
    
    def _user_based_scores(
        self,
        user_idx: int,
        rated_items: set
    ) -> np.ndarray:
        """Compute user-based collaborative filtering scores."""
        # Get similar users
        user_vector = self._interaction_matrix[user_idx]
        
        distances, indices = self._knn_model.kneighbors(
            user_vector.reshape(1, -1),
            n_neighbors=self.n_neighbors + 1
        )
        
        # Convert to similarities
        if self.metric == "cosine":
            similarities = 1 - distances.flatten()[1:]
        else:
            similarities = 1 / (1 + distances.flatten()[1:])
            
        neighbor_indices = indices.flatten()[1:]
        
        # Aggregate neighbor ratings
        scores = np.zeros(len(self._item_to_idx))
        weights = np.zeros(len(self._item_to_idx))
        
        for neighbor_idx, sim in zip(neighbor_indices, similarities):
            neighbor_ratings = np.asarray(
                self._interaction_matrix[neighbor_idx].todense()
            ).flatten()
            
            scores += sim * neighbor_ratings
            weights += sim * (neighbor_ratings > 0).astype(float)
            
        # Avoid division by zero
        weights[weights == 0] = 1
        scores = scores / weights
        
        return scores
    
    def _create_recommendation(
        self,
        book_id: str,
        score: float
    ) -> Recommendation:
        """Create Recommendation object with metadata."""
        title = self._get_book_title(book_id)
        author = None
        genre = None
        avg_rating = None
        
        if self._books_df is not None:
            book_row = self._books_df[
                self._books_df["book_id"] == book_id
            ]
            
            if len(book_row) > 0:
                row = book_row.iloc[0]
                author = row.get("author", row.get("authors"))
                genre = row.get("genre")
                avg_rating = row.get("avg_rating", row.get("average_rating"))
                
        return Recommendation(
            book_id=book_id,
            title=title,
            score=float(score),
            author=str(author) if pd.notna(author) else None,
            genre=str(genre) if pd.notna(genre) else None,
            avg_rating=float(avg_rating) if pd.notna(avg_rating) else None
        )
    
    def _get_book_title(self, book_id: str) -> str:
        """Get book title from metadata."""
        if self._books_df is None:
            return f"Book {book_id}"
            
        book_row = self._books_df[self._books_df["book_id"] == book_id]
        
        if len(book_row) > 0:
            return str(book_row.iloc[0].get("title", f"Book {book_id}"))
            
        return f"Book {book_id}"
    
    def _check_fitted(self):
        """Verify model has been fitted."""
        if not self._is_fitted:
            raise ValueError("Model must be fitted before making predictions")


class HybridRecommender:
    """
    Hybrid recommendation engine combining multiple approaches.
    
    Combines collaborative filtering (KNN) with content-based filtering
    to leverage both user behavior and item attributes.
    
    Attributes:
        cf_weight (float): Weight for collaborative filtering component
        cb_weight (float): Weight for content-based component
        
    Example:
        >>> hybrid = HybridRecommender(cf_weight=0.7, cb_weight=0.3)
        >>> hybrid.fit(ratings_df, books_df, book_features)
        >>> recs = hybrid.recommend('user_123', n=10)
    """
    
    def __init__(
        self,
        cf_weight: float = 0.6,
        cb_weight: float = 0.4,
        n_neighbors: int = 20,
        metric: str = "cosine",
        verbose: bool = True
    ):
        """
        Initialize HybridRecommender.
        
        Args:
            cf_weight: Weight for collaborative filtering scores
            cb_weight: Weight for content-based scores
            n_neighbors: Number of neighbors for KNN components
            metric: Distance metric
            verbose: Enable logging
        """
        if not np.isclose(cf_weight + cb_weight, 1.0):
            raise ValueError("Weights must sum to 1.0")
            
        self.cf_weight = cf_weight
        self.cb_weight = cb_weight
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.verbose = verbose
        
        # Initialize component recommenders
        self._cf_recommender = KNNRecommender(
            n_neighbors=n_neighbors,
            metric=metric,
            approach="item",
            verbose=False
        )
        
        self._cb_model: Optional[NearestNeighbors] = None
        self._content_features: Optional[sparse.csr_matrix] = None
        self._books_df: Optional[pd.DataFrame] = None
        self._is_fitted = False
        
    def fit(
        self,
        ratings_df: pd.DataFrame,
        books_df: pd.DataFrame,
        content_features: Optional[sparse.csr_matrix] = None
    ) -> "HybridRecommender":
        """
        Fit the hybrid recommender.
        
        Args:
            ratings_df: User-book ratings
            books_df: Book metadata
            content_features: Pre-computed content feature matrix
            
        Returns:
            self
        """
        logger.info("Fitting hybrid recommender...")
        
        self._books_df = books_df
        
        # Fit collaborative filtering component
        self._cf_recommender.fit(ratings_df, books_df)
        
        # Fit content-based component
        if content_features is not None:
            self._content_features = content_features
        else:
            # Build simple content features from metadata
            self._content_features = self._build_content_features(books_df)
            
        self._cb_model = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            metric=self.metric,
            algorithm="brute"
        )
        self._cb_model.fit(self._content_features)
        
        self._is_fitted = True
        
        if self.verbose:
            logger.info("Hybrid recommender fitted successfully")
            
        return self
    
    def recommend(
        self,
        user_id: str,
        n_recommendations: int = 10,
        exclude_rated: bool = True
    ) -> List[Recommendation]:
        """
        Generate hybrid recommendations for a user.
        
        Args:
            user_id: Target user identifier
            n_recommendations: Number of recommendations
            exclude_rated: Exclude already-rated books
            
        Returns:
            List of Recommendation objects
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted first")
            
        # Get CF recommendations
        cf_recs = self._cf_recommender.recommend_for_user(
            user_id, 
            n_recommendations * 2,
            exclude_rated
        )
        
        # Get content-based scores for CF candidates
        hybrid_recs = []
        
        for rec in cf_recs:
            cb_score = self._get_content_score(user_id, rec.book_id)
            
            # Combine scores
            hybrid_score = (
                self.cf_weight * rec.score + 
                self.cb_weight * cb_score
            )
            
            rec.score = hybrid_score
            rec.reason = f"CF: {self.cf_weight:.0%}, CB: {self.cb_weight:.0%}"
            hybrid_recs.append(rec)
            
        # Sort by hybrid score
        hybrid_recs.sort(key=lambda x: x.score, reverse=True)
        
        return hybrid_recs[:n_recommendations]
    
    def _build_content_features(
        self,
        books_df: pd.DataFrame
    ) -> sparse.csr_matrix:
        """Build content features from book metadata."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Combine text features
        text_cols = ["title", "author", "genre"]
        available = [c for c in text_cols if c in books_df.columns]
        
        if not available:
            available = ["title"]
            
        combined = books_df[available[0]].fillna("").astype(str)
        for col in available[1:]:
            combined = combined + " " + books_df[col].fillna("").astype(str)
            
        vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
        return vectorizer.fit_transform(combined)
    
    def _get_content_score(self, user_id: str, book_id: str) -> float:
        """Compute content-based score for a book-user pair."""
        try:
            # Get user's rated books
            profile = self._cf_recommender.get_user_profile(user_id)
            
            if not profile["top_rated_books"]:
                return 0.0
                
            # Get book index
            if book_id not in self._cf_recommender._item_to_idx:
                return 0.0
                
            book_idx = self._cf_recommender._item_to_idx[book_id]
            book_vector = self._content_features[book_idx]
            
            # Average similarity to user's top books
            similarities = []
            for rated_book in profile["top_rated_books"][:5]:
                rated_id = rated_book["book_id"]
                if rated_id in self._cf_recommender._item_to_idx:
                    rated_idx = self._cf_recommender._item_to_idx[rated_id]
                    rated_vector = self._content_features[rated_idx]
                    
                    sim = cosine_similarity(
                        book_vector.reshape(1, -1),
                        rated_vector.reshape(1, -1)
                    )[0, 0]
                    similarities.append(sim)
                    
            return float(np.mean(similarities)) if similarities else 0.0
            
        except Exception:
            return 0.0
