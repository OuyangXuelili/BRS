"""
Utility Functions for Book Recommendation System.

This module provides helper functions for logging, model persistence,
configuration management, and common operations used throughout the
recommendation pipeline.

This file was refreshed for the current GitHub repository release.

Functions:
    setup_logging: Configure application logging
    save_model: Persist trained model to disk
    load_model: Load trained model from disk
    load_config: Load YAML configuration
    
Example:
    >>> from src.utils import setup_logging, save_model
    >>> setup_logging(level="INFO")
    >>> save_model(recommender, "models/knn_model.pkl")
"""

import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Configure application-wide logging.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for log output
        format_string: Custom format string for log messages
        
    Returns:
        Configured root logger
        
    Example:
        >>> logger = setup_logging(level="DEBUG", log_file="logs/app.log")
        >>> logger.info("Application started")
    """
    if format_string is None:
        format_string = (
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        )
        
    # Create formatter
    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
    return root_logger


def save_model(
    model: Any,
    filepath: Union[str, Path],
    metadata: Optional[Dict] = None
) -> Path:
    """
    Save trained model to disk.
    
    Args:
        model: Trained model object
        filepath: Destination file path
        metadata: Optional metadata to save with model
        
    Returns:
        Path to saved model file
        
    Example:
        >>> save_model(recommender, "models/knn_v1.pkl", 
        ...            metadata={"version": "1.0", "date": "2024-01-15"})
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Package model with metadata
    package = {
        "model": model,
        "metadata": metadata or {},
        "saved_at": datetime.now().isoformat(),
        "version": getattr(model, "__version__", "unknown")
    }
    
    with open(filepath, "wb") as f:
        pickle.dump(package, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    logging.getLogger(__name__).info(f"Model saved to {filepath}")
    
    return filepath


def load_model(
    filepath: Union[str, Path],
    return_metadata: bool = False
) -> Union[Any, tuple]:
    """
    Load trained model from disk.
    
    Args:
        filepath: Path to saved model file
        return_metadata: If True, return (model, metadata) tuple
        
    Returns:
        Loaded model, or (model, metadata) tuple if return_metadata=True
        
    Raises:
        FileNotFoundError: If model file does not exist
        
    Example:
        >>> model, meta = load_model("models/knn_v1.pkl", return_metadata=True)
        >>> print(f"Model saved at: {meta['saved_at']}")
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")
        
    with open(filepath, "rb") as f:
        package = pickle.load(f)
        
    model = package["model"]
    metadata = package.get("metadata", {})
    
    logging.getLogger(__name__).info(
        f"Model loaded from {filepath} (saved: {package.get('saved_at', 'unknown')})"
    )
    
    if return_metadata:
        return model, metadata
        
    return model


def load_config(config_path: Union[str, Path]) -> Dict:
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file does not exist
    """
    import yaml
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    return config


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path to project root
    """
    current = Path(__file__).resolve()
    
    # Walk up until we find setup.py or pyproject.toml
    for parent in current.parents:
        if (parent / "setup.py").exists() or (parent / "pyproject.toml").exists():
            return parent
            
    # Fallback: return parent of src/
    return current.parent.parent


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, creating if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


class Timer:
    """
    Context manager for timing code blocks.
    
    Example:
        >>> with Timer("Training"):
        ...     model.fit(data)
        Training completed in 2.34 seconds
    """
    
    def __init__(self, name: str = "Operation", logger: Optional[logging.Logger] = None):
        """
        Initialize timer.
        
        Args:
            name: Name for the timed operation
            logger: Logger instance (uses print if None)
        """
        self.name = name
        self.logger = logger
        self._start_time: Optional[float] = None
        self.elapsed: float = 0.0
        
    def __enter__(self):
        """Start timing."""
        import time
        self._start_time = time.perf_counter()
        return self
        
    def __exit__(self, *args):
        """Stop timing and log result."""
        import time
        self.elapsed = time.perf_counter() - self._start_time
        
        message = f"{self.name} completed in {self.elapsed:.2f} seconds"
        
        if self.logger:
            self.logger.info(message)
        else:
            print(message)


def compute_similarity_matrix(
    features: np.ndarray,
    metric: str = "cosine"
) -> np.ndarray:
    """
    Compute pairwise similarity matrix.
    
    Args:
        features: Feature matrix (n_samples, n_features)
        metric: Similarity metric ('cosine', 'euclidean')
        
    Returns:
        Similarity matrix (n_samples, n_samples)
    """
    from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
    
    if metric == "cosine":
        return cosine_similarity(features)
    elif metric == "euclidean":
        distances = euclidean_distances(features)
        return 1 / (1 + distances)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def sample_negative_items(
    user_items: set,
    all_items: set,
    n_samples: int,
    random_state: Optional[int] = None
) -> list:
    """
    Sample negative items for a user.
    
    Args:
        user_items: Set of items the user has interacted with
        all_items: Set of all available items
        n_samples: Number of negative samples
        random_state: Random seed
        
    Returns:
        List of sampled negative item IDs
    """
    if random_state is not None:
        np.random.seed(random_state)
        
    negative_pool = list(all_items - user_items)
    
    n_samples = min(n_samples, len(negative_pool))
    
    return list(np.random.choice(negative_pool, n_samples, replace=False))


def print_recommendations(
    recommendations: list,
    max_display: int = 10
) -> None:
    """
    Pretty print recommendations to console.
    
    Args:
        recommendations: List of Recommendation objects
        max_display: Maximum number to display
    """
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    
    for i, rec in enumerate(recommendations[:max_display], 1):
        print(f"\n{i}. {rec.title}")
        print(f"   Score: {rec.score:.4f}")
        
        if rec.author:
            print(f"   Author: {rec.author}")
            
        if rec.genre:
            print(f"   Genre: {rec.genre}")
            
        if rec.avg_rating:
            print(f"   Avg Rating: {rec.avg_rating:.2f}")
            
        if rec.reason:
            print(f"   Reason: {rec.reason}")
            
    print("\n" + "=" * 60)


def format_number(n: Union[int, float], precision: int = 2) -> str:
    """
    Format large numbers with K/M/B suffixes.
    
    Args:
        n: Number to format
        precision: Decimal precision
        
    Returns:
        Formatted string
        
    Example:
        >>> format_number(1234567)
        '1.23M'
    """
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.{precision}f}B"
    elif n >= 1_000_000:
        return f"{n / 1_000_000:.{precision}f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.{precision}f}K"
    else:
        return str(n)
