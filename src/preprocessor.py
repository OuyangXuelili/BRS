"""
Data Preprocessing and Feature Engineering for Book Recommendations.

This module provides functionality for cleaning, transforming, and 
engineering features from raw book and rating data for use in 
recommendation models.

Classes:
    BookPreprocessor: Text and metadata preprocessing
    FeatureExtractor: Feature matrix construction for ML models

Example:
    >>> from src.preprocessor import BookPreprocessor, FeatureExtractor
    >>> preprocessor = BookPreprocessor()
    >>> clean_df = preprocessor.fit_transform(books_df)
    >>> extractor = FeatureExtractor(method="tfidf")
    >>> features = extractor.fit_transform(clean_df)
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

logger = logging.getLogger(__name__)


class BookPreprocessor:
    """
    Preprocessor for book metadata and text fields.
    
    Handles text normalization, missing value imputation, and 
    data type conversions for book datasets.
    
    Attributes:
        lowercase (bool): Convert text to lowercase
        remove_special (bool): Remove special characters
        fill_missing (bool): Impute missing values
        
    Example:
        >>> preprocessor = BookPreprocessor(lowercase=True)
        >>> clean_df = preprocessor.fit_transform(raw_df)
    """
    
    def __init__(
        self,
        lowercase: bool = True,
        remove_special: bool = True,
        fill_missing: bool = True,
        verbose: bool = True
    ):
        """
        Initialize the BookPreprocessor.
        
        Args:
            lowercase: Convert text fields to lowercase
            remove_special: Remove special characters from text
            fill_missing: Impute missing values
            verbose: Enable logging
        """
        self.lowercase = lowercase
        self.remove_special = remove_special
        self.fill_missing = fill_missing
        self.verbose = verbose
        
        self._is_fitted = False
        self._text_columns: List[str] = []
        self._numeric_columns: List[str] = []
        self._fill_values: Dict[str, any] = {}
        
    def fit(self, df: pd.DataFrame) -> "BookPreprocessor":
        """
        Fit preprocessor to data (learn fill values, identify columns).
        
        Args:
            df: DataFrame to fit on
            
        Returns:
            self
        """
        self._text_columns = df.select_dtypes(include=["object"]).columns.tolist()
        self._numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
        
        # Compute fill values for numeric columns
        for col in self._numeric_columns:
            self._fill_values[col] = df[col].median()
            
        # Compute fill values for text columns  
        for col in self._text_columns:
            self._fill_values[col] = "Unknown"
            
        self._is_fitted = True
        
        if self.verbose:
            logger.info(
                f"Fitted preprocessor: {len(self._text_columns)} text cols, "
                f"{len(self._numeric_columns)} numeric cols"
            )
            
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform DataFrame using fitted parameters.
        
        Args:
            df: DataFrame to transform
            
        Returns:
            Transformed DataFrame
            
        Raises:
            ValueError: If fit() has not been called
        """
        if not self._is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
            
        df = df.copy()
        
        # Fill missing values
        if self.fill_missing:
            for col, value in self._fill_values.items():
                if col in df.columns:
                    df[col] = df[col].fillna(value)
                    
        # Process text columns
        for col in self._text_columns:
            if col in df.columns:
                df[col] = self._clean_text(df[col])
                
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform in a single call.
        
        Args:
            df: DataFrame to fit and transform
            
        Returns:
            Transformed DataFrame
        """
        return self.fit(df).transform(df)
    
    def _clean_text(self, series: pd.Series) -> pd.Series:
        """Apply text cleaning operations to a series."""
        result = series.astype(str)
        
        if self.lowercase:
            result = result.str.lower()
            
        if self.remove_special:
            result = result.apply(lambda x: re.sub(r"[^\w\s]", " ", x))
            result = result.str.replace(r"\s+", " ", regex=True)
            result = result.str.strip()
            
        return result
    
    def create_text_features(
        self,
        df: pd.DataFrame,
        text_columns: List[str]
    ) -> pd.Series:
        """
        Combine multiple text columns into a single feature column.
        
        Args:
            df: DataFrame containing text columns
            text_columns: List of column names to combine
            
        Returns:
            Combined text series
        """
        available_cols = [c for c in text_columns if c in df.columns]
        
        if not available_cols:
            raise ValueError(f"No text columns found: {text_columns}")
            
        combined = df[available_cols[0]].fillna("").astype(str)
        
        for col in available_cols[1:]:
            combined = combined + " " + df[col].fillna("").astype(str)
            
        return combined.str.strip()


class FeatureExtractor:
    """
    Extract feature matrices for recommendation models.
    
    Supports multiple feature extraction methods including TF-IDF,
    count vectorization, and combined content features.
    
    Attributes:
        method (str): Feature extraction method ('tfidf', 'count', 'combined')
        max_features (int): Maximum number of features to extract
        
    Example:
        >>> extractor = FeatureExtractor(method='tfidf', max_features=5000)
        >>> features = extractor.fit_transform(df['description'])
        >>> print(f"Feature matrix shape: {features.shape}")
    """
    
    VALID_METHODS = {"tfidf", "count", "combined"}
    
    def __init__(
        self,
        method: str = "tfidf",
        max_features: int = 5000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 1,
        max_df: float = 0.95,
        use_idf: bool = True,
        verbose: bool = True
    ):
        """
        Initialize FeatureExtractor.
        
        Args:
            method: Extraction method ('tfidf', 'count', 'combined')
            max_features: Maximum vocabulary size
            ngram_range: N-gram range for text features
            min_df: Minimum document frequency
            max_df: Maximum document frequency
            use_idf: Use inverse document frequency weighting
            verbose: Enable logging
        """
        if method not in self.VALID_METHODS:
            raise ValueError(f"Method must be one of {self.VALID_METHODS}")
            
        self.method = method
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.use_idf = use_idf
        self.verbose = verbose
        
        self._vectorizer = None
        self._scaler = None
        self._is_fitted = False
        self.feature_names_: Optional[List[str]] = None
        
    def fit(
        self,
        text_data: pd.Series,
        numeric_data: Optional[pd.DataFrame] = None
    ) -> "FeatureExtractor":
        """
        Fit the feature extractor on training data.
        
        Args:
            text_data: Series of text to vectorize
            numeric_data: Optional DataFrame of numeric features
            
        Returns:
            self
        """
        # Initialize vectorizer based on method
        if self.method in ["tfidf", "combined"]:
            self._vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                min_df=self.min_df,
                max_df=self.max_df,
                use_idf=self.use_idf,
                stop_words="english"
            )
        else:
            self._vectorizer = CountVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                min_df=self.min_df,
                max_df=self.max_df,
                stop_words="english"
            )
            
        # Fit vectorizer on text data
        clean_text = text_data.fillna("").astype(str)
        try:
            self._vectorizer.fit(clean_text)
        except ValueError as exc:
            if "After pruning, no terms remain" not in str(exc):
                raise

            # Small datasets can be too sparse for the default pruning thresholds.
            # Retry once with relaxed bounds so we still produce a usable feature matrix.
            self.min_df = 1
            self.max_df = 1.0
            if self.method in ["tfidf", "combined"]:
                self._vectorizer = TfidfVectorizer(
                    max_features=self.max_features,
                    ngram_range=self.ngram_range,
                    min_df=self.min_df,
                    max_df=1.0,
                    use_idf=self.use_idf,
                    stop_words="english"
                )
            else:
                self._vectorizer = CountVectorizer(
                    max_features=self.max_features,
                    ngram_range=self.ngram_range,
                    min_df=self.min_df,
                    max_df=1.0,
                    stop_words="english"
                )
            self._vectorizer.fit(clean_text)
        self.feature_names_ = list(self._vectorizer.get_feature_names_out())
        
        # Fit scaler for numeric features if provided
        if numeric_data is not None and self.method == "combined":
            self._scaler = StandardScaler()
            self._scaler.fit(numeric_data.fillna(0))
            self.feature_names_.extend(numeric_data.columns.tolist())
            
        self._is_fitted = True
        
        if self.verbose:
            logger.info(f"Fitted extractor with {len(self.feature_names_)} features")
            
        return self
    
    def transform(
        self,
        text_data: pd.Series,
        numeric_data: Optional[pd.DataFrame] = None
    ) -> sparse.csr_matrix:
        """
        Transform data using fitted extractor.
        
        Args:
            text_data: Series of text to vectorize
            numeric_data: Optional DataFrame of numeric features
            
        Returns:
            Sparse feature matrix
            
        Raises:
            ValueError: If fit() has not been called
        """
        if not self._is_fitted:
            raise ValueError("Extractor must be fitted before transform")
            
        clean_text = text_data.fillna("").astype(str)
        text_features = self._vectorizer.transform(clean_text)
        
        if numeric_data is not None and self._scaler is not None:
            numeric_features = self._scaler.transform(numeric_data.fillna(0))
            return sparse.hstack([text_features, sparse.csr_matrix(numeric_features)])
            
        return text_features
    
    def fit_transform(
        self,
        text_data: pd.Series,
        numeric_data: Optional[pd.DataFrame] = None
    ) -> sparse.csr_matrix:
        """
        Fit and transform in a single call.
        
        Args:
            text_data: Series of text to vectorize
            numeric_data: Optional DataFrame of numeric features
            
        Returns:
            Sparse feature matrix
        """
        return self.fit(text_data, numeric_data).transform(text_data, numeric_data)


class InteractionMatrix:
    """
    Build and manipulate user-item interaction matrices.
    
    Creates sparse matrices from rating data for collaborative filtering
    and computes derived matrices (normalized, centered, binary).
    
    Attributes:
        user_to_idx (dict): Mapping from user_id to matrix row index
        item_to_idx (dict): Mapping from book_id to matrix column index
        idx_to_user (dict): Reverse mapping from index to user_id
        idx_to_item (dict): Reverse mapping from index to book_id
        
    Example:
        >>> matrix = InteractionMatrix()
        >>> matrix.fit(ratings_df)
        >>> sparse_ratings = matrix.to_sparse()
    """
    
    def __init__(self, normalize: bool = False, binarize: bool = False):
        """
        Initialize InteractionMatrix builder.
        
        Args:
            normalize: Row-normalize the matrix
            binarize: Convert to binary (implicit feedback)
        """
        self.normalize = normalize
        self.binarize = binarize
        
        self.user_to_idx: Dict[str, int] = {}
        self.item_to_idx: Dict[str, int] = {}
        self.idx_to_user: Dict[int, str] = {}
        self.idx_to_item: Dict[int, str] = {}
        
        self._matrix: Optional[sparse.csr_matrix] = None
        self._is_fitted = False
        
    def fit(self, ratings_df: pd.DataFrame) -> "InteractionMatrix":
        """
        Fit matrix builder on ratings data.
        
        Args:
            ratings_df: DataFrame with user_id, book_id, rating columns
            
        Returns:
            self
        """
        # Build mappings
        unique_users = ratings_df["user_id"].unique()
        unique_items = ratings_df["book_id"].unique()
        
        self.user_to_idx = {u: i for i, u in enumerate(unique_users)}
        self.item_to_idx = {b: i for i, b in enumerate(unique_items)}
        self.idx_to_user = {i: u for u, i in self.user_to_idx.items()}
        self.idx_to_item = {i: b for b, i in self.item_to_idx.items()}
        
        # Build sparse matrix
        row_indices = ratings_df["user_id"].map(self.user_to_idx).values
        col_indices = ratings_df["book_id"].map(self.item_to_idx).values
        
        if self.binarize:
            values = np.ones(len(ratings_df))
        else:
            values = ratings_df["rating"].values
            
        self._matrix = sparse.csr_matrix(
            (values, (row_indices, col_indices)),
            shape=(len(self.user_to_idx), len(self.item_to_idx))
        )
        
        if self.normalize:
            self._normalize_rows()
            
        self._is_fitted = True
        
        return self
    
    def to_sparse(self) -> sparse.csr_matrix:
        """
        Get the sparse interaction matrix.
        
        Returns:
            CSR sparse matrix of user-item interactions
            
        Raises:
            ValueError: If fit() has not been called
        """
        if not self._is_fitted:
            raise ValueError("Matrix must be fitted before accessing")
            
        return self._matrix
    
    def get_user_vector(self, user_id: str) -> np.ndarray:
        """Get rating vector for a specific user."""
        if user_id not in self.user_to_idx:
            raise KeyError(f"Unknown user: {user_id}")
            
        idx = self.user_to_idx[user_id]
        return np.asarray(self._matrix[idx].todense()).flatten()
    
    def get_item_vector(self, book_id: str) -> np.ndarray:
        """Get rating vector for a specific book."""
        if book_id not in self.item_to_idx:
            raise KeyError(f"Unknown book: {book_id}")
            
        idx = self.item_to_idx[book_id]
        return np.asarray(self._matrix[:, idx].todense()).flatten()
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Return matrix dimensions (n_users, n_items)."""
        return self._matrix.shape if self._matrix is not None else (0, 0)
    
    @property
    def density(self) -> float:
        """Return fraction of non-zero entries."""
        if self._matrix is None:
            return 0.0
        return self._matrix.nnz / (self._matrix.shape[0] * self._matrix.shape[1])
    
    def _normalize_rows(self):
        """L2 normalize each row of the matrix."""
        row_norms = sparse.linalg.norm(self._matrix, axis=1)
        row_norms[row_norms == 0] = 1  # Avoid division by zero
        
        # Create diagonal matrix for normalization
        diag = sparse.diags(1 / row_norms)
        self._matrix = diag.dot(self._matrix)


def create_book_features(
    books_df: pd.DataFrame,
    text_columns: List[str] = None,
    numeric_columns: List[str] = None,
    max_features: int = 5000
) -> Tuple[sparse.csr_matrix, List[str]]:
    """
    Convenience function to extract book features.
    
    Args:
        books_df: DataFrame with book metadata
        text_columns: Columns to use for text features
        numeric_columns: Columns to use for numeric features
        max_features: Maximum vocabulary size
        
    Returns:
        Tuple of (feature_matrix, feature_names)
    """
    if text_columns is None:
        text_columns = ["title", "author", "genre"]
        
    if numeric_columns is None:
        numeric_columns = ["avg_rating", "n_ratings", "n_pages"]
        
    # Preprocess
    preprocessor = BookPreprocessor(verbose=False)
    clean_df = preprocessor.fit_transform(books_df)
    
    # Create combined text feature
    combined_text = preprocessor.create_text_features(clean_df, text_columns)
    
    # Get numeric features
    available_numeric = [c for c in numeric_columns if c in clean_df.columns]
    numeric_data = clean_df[available_numeric] if available_numeric else None
    
    # Extract features
    extractor = FeatureExtractor(
        method="combined" if numeric_data is not None else "tfidf",
        max_features=max_features,
        verbose=False
    )
    
    features = extractor.fit_transform(combined_text, numeric_data)
    
    return features, extractor.feature_names_
