"""
Data Loading Utilities for Book Recommendation System.

This module provides functionality for loading and validating book datasets
from various sources including the UCSD Book Graph (Goodreads) dataset.

Classes:
    DataLoader: Base class for loading book datasets
    GoodreadsLoader: Specialized loader for Goodreads/UCSD Book Graph data

Example:
    >>> from src.data_loader import GoodreadsLoader
    >>> loader = GoodreadsLoader()
    >>> books_df, ratings_df = loader.load_dataset("data/")
"""

import json
import gzip
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DatasetInfo:
    """Container for dataset metadata and statistics."""
    
    n_books: int
    n_users: int
    n_ratings: int
    rating_density: float
    avg_rating: float
    rating_std: float
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        return (
            f"Dataset Statistics:\n"
            f"  Books: {self.n_books:,}\n"
            f"  Users: {self.n_users:,}\n"
            f"  Ratings: {self.n_ratings:,}\n"
            f"  Density: {self.rating_density:.4%}\n"
            f"  Avg Rating: {self.avg_rating:.2f} ± {self.rating_std:.2f}"
        )


class DataLoader:
    """
    Base class for loading book recommendation datasets.
    
    Provides common functionality for reading, validating, and transforming
    book and rating data from various file formats.
    
    Attributes:
        data_dir (Path): Directory containing dataset files
        verbose (bool): Whether to log detailed progress information
    """
    
    REQUIRED_BOOK_COLUMNS = ["book_id", "title"]
    REQUIRED_RATING_COLUMNS = ["user_id", "book_id", "rating"]
    
    def __init__(
        self,
        data_dir: Optional[Union[str, Path]] = None,
        verbose: bool = True
    ):
        """
        Initialize the DataLoader.
        
        Args:
            data_dir: Path to data directory. Defaults to 'data/'.
            verbose: Enable detailed logging output.
        """
        self.data_dir = Path(data_dir) if data_dir else Path("data")
        self.verbose = verbose
        self._books_df: Optional[pd.DataFrame] = None
        self._ratings_df: Optional[pd.DataFrame] = None
        
    def load_csv(
        self,
        filepath: Union[str, Path],
        **kwargs
    ) -> pd.DataFrame:
        """
        Load data from CSV file with automatic encoding detection.
        
        Args:
            filepath: Path to CSV file
            **kwargs: Additional arguments passed to pd.read_csv
            
        Returns:
            Loaded DataFrame
            
        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If file is empty or malformed
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
            
        encodings = ["utf-8", "latin-1", "cp1252"]
        
        for encoding in encodings:
            try:
                df = pd.read_csv(filepath, encoding=encoding, **kwargs)
                if self.verbose:
                    logger.info(f"Loaded {len(df):,} rows from {filepath.name}")
                return df
            except UnicodeDecodeError:
                continue
                
        raise ValueError(f"Unable to decode file with supported encodings: {filepath}")
    
    def load_json(
        self,
        filepath: Union[str, Path],
        compressed: bool = False
    ) -> pd.DataFrame:
        """
        Load data from JSON or compressed JSON file.
        
        Args:
            filepath: Path to JSON file
            compressed: Whether file is gzip compressed
            
        Returns:
            Loaded DataFrame
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        records = []
        
        opener = gzip.open if compressed else open
        mode = "rt" if compressed else "r"
        
        with opener(filepath, mode, encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    records.append(record)
                except json.JSONDecodeError:
                    continue
                    
        df = pd.DataFrame(records)
        
        if self.verbose:
            logger.info(f"Loaded {len(df):,} records from {filepath.name}")
            
        return df
    
    def validate_books(self, df: pd.DataFrame) -> bool:
        """
        Validate books DataFrame has required columns.
        
        Args:
            df: Books DataFrame to validate
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If required columns are missing
        """
        missing = set(self.REQUIRED_BOOK_COLUMNS) - set(df.columns)
        
        if missing:
            raise ValueError(f"Missing required book columns: {missing}")
            
        return True
    
    def validate_ratings(self, df: pd.DataFrame) -> bool:
        """
        Validate ratings DataFrame has required columns.
        
        Args:
            df: Ratings DataFrame to validate
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If required columns are missing
        """
        missing = set(self.REQUIRED_RATING_COLUMNS) - set(df.columns)
        
        if missing:
            raise ValueError(f"Missing required rating columns: {missing}")
            
        return True
    
    def compute_statistics(
        self,
        books_df: pd.DataFrame,
        ratings_df: pd.DataFrame
    ) -> DatasetInfo:
        """
        Compute dataset statistics for reporting.
        
        Args:
            books_df: Books DataFrame
            ratings_df: Ratings DataFrame
            
        Returns:
            DatasetInfo with computed statistics
        """
        n_books = books_df["book_id"].nunique()
        n_users = ratings_df["user_id"].nunique()
        n_ratings = len(ratings_df)
        
        # Compute rating matrix density
        max_possible = n_books * n_users
        density = n_ratings / max_possible if max_possible > 0 else 0
        
        avg_rating = ratings_df["rating"].mean()
        rating_std = ratings_df["rating"].std()
        
        return DatasetInfo(
            n_books=n_books,
            n_users=n_users,
            n_ratings=n_ratings,
            rating_density=density,
            avg_rating=avg_rating,
            rating_std=rating_std
        )


class GoodreadsLoader(DataLoader):
    """
    Specialized loader for UCSD Book Graph / Goodreads dataset.
    
    Handles the specific format and structure of the Goodreads dataset
    including JSON-Lines format and nested attributes.
    
    The UCSD Book Graph contains:
    - 2.3M books with metadata (titles, authors, genres)
    - 876K users with reading history
    - 229M user-book interactions
    
    Reference:
        Mengting Wan, Julian McAuley, "Item Recommendation on Monotonic 
        Behavior Chains", RecSys 2018.
        
    Example:
        >>> loader = GoodreadsLoader("data/goodreads/")
        >>> books, ratings = loader.load_dataset()
        >>> print(f"Loaded {len(books)} books")
    """
    
    BOOK_COLUMNS_MAP = {
        "book_id": "book_id",
        "title": "title",
        "authors": "authors",
        "average_rating": "avg_rating",
        "ratings_count": "n_ratings",
        "language_code": "language",
        "num_pages": "n_pages",
        "publication_year": "year",
        "publisher": "publisher",
    }
    
    def __init__(
        self,
        data_dir: Optional[Union[str, Path]] = None,
        min_ratings: int = 5,
        min_books_per_user: int = 3,
        verbose: bool = True
    ):
        """
        Initialize GoodreadsLoader with filtering parameters.
        
        Args:
            data_dir: Path to Goodreads data directory
            min_ratings: Minimum ratings required per book
            min_books_per_user: Minimum books rated per user
            verbose: Enable detailed logging
        """
        super().__init__(data_dir, verbose)
        self.min_ratings = min_ratings
        self.min_books_per_user = min_books_per_user
        
    def load_books(
        self,
        filepath: Optional[Union[str, Path]] = None
    ) -> pd.DataFrame:
        """
        Load and process book metadata.
        
        Args:
            filepath: Path to books file. Auto-detects format.
            
        Returns:
            Processed books DataFrame
        """
        if filepath is None:
            filepath = self._find_books_file()
            
        filepath = Path(filepath)
        
        # Detect format and load
        if filepath.suffix == ".csv":
            df = self.load_csv(filepath)
        elif filepath.suffix in [".json", ".gz"]:
            df = self.load_json(filepath, compressed=filepath.suffix == ".gz")
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
            
        # Standardize column names
        df = self._standardize_book_columns(df)
        
        # Extract author names if nested
        if "authors" in df.columns:
            df["author"] = df["authors"].apply(self._extract_author_name)
            
        # Clean and validate
        df = self._clean_books(df)
        self.validate_books(df)
        
        self._books_df = df
        return df
    
    def load_ratings(
        self,
        filepath: Optional[Union[str, Path]] = None
    ) -> pd.DataFrame:
        """
        Load and process user ratings/interactions.
        
        Args:
            filepath: Path to ratings file
            
        Returns:
            Processed ratings DataFrame
        """
        if filepath is None:
            filepath = self._find_ratings_file()
            
        filepath = Path(filepath)
        
        if filepath.suffix == ".csv":
            df = self.load_csv(filepath)
        else:
            df = self.load_json(filepath, compressed=filepath.suffix == ".gz")
            
        # Standardize column names
        column_map = {
            "user_id": "user_id",
            "book_id": "book_id",
            "rating": "rating",
        }
        
        for old_name, new_name in column_map.items():
            if old_name in df.columns and old_name != new_name:
                df = df.rename(columns={old_name: new_name})
                
        # Ensure rating is numeric
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
        df = df.dropna(subset=["rating"])
        
        # Apply filtering
        df = self._filter_ratings(df)
        self.validate_ratings(df)
        
        self._ratings_df = df
        return df
    
    def load_dataset(
        self,
        books_path: Optional[Union[str, Path]] = None,
        ratings_path: Optional[Union[str, Path]] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load complete dataset with books and ratings.
        
        Args:
            books_path: Path to books file
            ratings_path: Path to ratings file
            
        Returns:
            Tuple of (books_df, ratings_df)
        """
        books_df = self.load_books(books_path)
        ratings_df = self.load_ratings(ratings_path)
        
        # Filter to matching book IDs
        common_books = set(books_df["book_id"]) & set(ratings_df["book_id"])
        
        books_df = books_df[books_df["book_id"].isin(common_books)].copy()
        ratings_df = ratings_df[ratings_df["book_id"].isin(common_books)].copy()
        
        if self.verbose:
            stats = self.compute_statistics(books_df, ratings_df)
            logger.info(f"\n{stats.summary()}")
            
        return books_df, ratings_df
    
    def _find_books_file(self) -> Path:
        """Locate books file in data directory."""
        patterns = ["*books*.csv", "*books*.json", "*books*.gz"]
        
        for pattern in patterns:
            matches = list(self.data_dir.glob(pattern))
            if matches:
                return matches[0]
                
        raise FileNotFoundError(
            f"No books file found in {self.data_dir}. "
            f"Expected patterns: {patterns}"
        )
    
    def _find_ratings_file(self) -> Path:
        """Locate ratings file in data directory."""
        patterns = ["*rating*.csv", "*interaction*.csv", "*interaction*.json", "*rating*.gz", "*interaction*.gz"]
        
        for pattern in patterns:
            matches = list(self.data_dir.glob(pattern))
            if matches:
                return matches[0]
                
        raise FileNotFoundError(
            f"No ratings file found in {self.data_dir}. "
            f"Expected patterns: {patterns}"
        )
    
    def _standardize_book_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map various column naming conventions to standard names."""
        rename_map = {}
        
        for standard, alternatives in [
            ("book_id", ["bookId", "id", "isbn", "ISBN"]),
            ("title", ["Title", "name", "book_title"]),
            ("authors", ["author", "Author", "writers"]),
            ("avg_rating", ["average_rating", "rating", "mean_rating"]),
            ("n_ratings", ["ratings_count", "num_ratings", "rating_count"]),
            ("n_pages", ["num_pages", "pages", "page_count"]),
            ("year", ["publication_year", "pub_year", "published"]),
        ]:
            for alt in alternatives:
                if alt in df.columns and standard not in df.columns:
                    rename_map[alt] = standard
                    break
                    
        return df.rename(columns=rename_map)
    
    def _extract_author_name(self, authors_field) -> str:
        """Extract primary author name from various formats."""
        if pd.isna(authors_field):
            return "Unknown"
            
        if isinstance(authors_field, list):
            if len(authors_field) > 0:
                if isinstance(authors_field[0], dict):
                    return authors_field[0].get("name", "Unknown")
                return str(authors_field[0])
            return "Unknown"
            
        if isinstance(authors_field, dict):
            return authors_field.get("name", "Unknown")
            
        return str(authors_field).split(",")[0].strip()
    
    def _clean_books(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize book data."""
        df = df.copy()
        
        # Remove duplicates
        df = df.drop_duplicates(subset=["book_id"], keep="first")
        
        # Clean titles
        if "title" in df.columns:
            df["title"] = df["title"].fillna("Unknown Title")
            df["title"] = df["title"].str.strip()
            
        # Convert numeric columns
        numeric_cols = ["avg_rating", "n_ratings", "n_pages", "year"]
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                
        # Filter out invalid entries
        df = df[df["title"].str.len() > 0]
        
        return df.reset_index(drop=True)
    
    def _filter_ratings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply minimum rating thresholds."""
        df = df.copy()
        
        # Filter books with minimum ratings
        book_counts = df["book_id"].value_counts()
        valid_books = book_counts[book_counts >= self.min_ratings].index
        df = df[df["book_id"].isin(valid_books)]
        
        # Filter users with minimum books
        user_counts = df["user_id"].value_counts()
        valid_users = user_counts[user_counts >= self.min_books_per_user].index
        df = df[df["user_id"].isin(valid_users)]
        
        if self.verbose:
            logger.info(
                f"After filtering: {len(valid_books):,} books, "
                f"{len(valid_users):,} users, {len(df):,} ratings"
            )
            
        return df.reset_index(drop=True)


def create_sample_dataset(
    n_books: int = 1000,
    n_users: int = 500,
    n_ratings: int = 10000,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create synthetic dataset for testing and demonstration.
    
    Args:
        n_books: Number of books to generate
        n_users: Number of users to generate
        n_ratings: Number of ratings to generate
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (books_df, ratings_df)
    """
    np.random.seed(seed)
    
    # Generate book data
    genres = ["Fiction", "Non-Fiction", "Science Fiction", "Mystery", 
              "Romance", "Fantasy", "Biography", "History", "Science"]
    
    books_data = {
        "book_id": [f"book_{i}" for i in range(n_books)],
        "title": [f"Book Title {i}" for i in range(n_books)],
        "author": [f"Author {i % 200}" for i in range(n_books)],
        "genre": np.random.choice(genres, n_books),
        "avg_rating": np.random.uniform(2.5, 5.0, n_books).round(2),
        "n_ratings": np.random.randint(10, 10000, n_books),
        "n_pages": np.random.randint(100, 800, n_books),
        "year": np.random.randint(1950, 2024, n_books),
    }
    
    books_df = pd.DataFrame(books_data)
    
    # Generate ratings with user preference patterns
    ratings_data = {
        "user_id": [f"user_{np.random.randint(0, n_users)}" for _ in range(n_ratings)],
        "book_id": [f"book_{np.random.randint(0, n_books)}" for _ in range(n_ratings)],
        "rating": np.random.choice([1, 2, 3, 4, 5], n_ratings, p=[0.05, 0.1, 0.2, 0.35, 0.3]),
    }
    
    ratings_df = pd.DataFrame(ratings_data)
    
    # Remove duplicate user-book pairs
    ratings_df = ratings_df.drop_duplicates(subset=["user_id", "book_id"])
    
    return books_df, ratings_df
