# Contributing to Book Recommendation System

First off, thank you for considering contributing to this project! It's people like you that make this tool better for everyone.

## Code of Conduct

By participating in this project, you are expected to uphold our Code of Conduct: be respectful, inclusive, and constructive in all interactions.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the existing issues to avoid duplicates. When you create a bug report, include as many details as possible:

- **Use a clear and descriptive title**
- **Describe the exact steps to reproduce the problem**
- **Provide specific examples** (code snippets, data samples)
- **Describe the behavior you observed and what you expected**
- **Include your environment details** (OS, Python version, package versions)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion:

- **Use a clear and descriptive title**
- **Provide a detailed description of the proposed functionality**
- **Explain why this enhancement would be useful**
- **List any alternatives you've considered**

### Pull Requests

1. **Fork the repo** and create your branch from `main`
2. **Install development dependencies**: `pip install -e ".[dev]"`
3. **Make your changes** following our coding standards
4. **Add tests** for any new functionality
5. **Ensure all tests pass**: `pytest tests/ -v`
6. **Format your code**: `black src/ tests/ --line-length=100`
7. **Lint your code**: `flake8 src/ tests/`
8. **Update documentation** if needed
9. **Submit your pull request**

## Development Setup

```bash
# Clone your fork
git clone https://github.com/OuyangXuelili/BRS.git
cd BRS

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all extras
pip install -e ".[dev,demo,notebook]"

# Run tests to verify setup
pytest tests/ -v
```

## Coding Standards

### Python Style Guide

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide
- Use [Black](https://black.readthedocs.io/) for code formatting (line length: 100)
- Use [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- Add type hints to function signatures

### Example Code Style

```python
from typing import List, Optional, Dict
import numpy as np


def compute_similarity(
    vector_a: np.ndarray,
    vector_b: np.ndarray,
    metric: str = "cosine"
) -> float:
    """
    Compute similarity between two vectors.
    
    Args:
        vector_a: First vector for comparison.
        vector_b: Second vector for comparison.
        metric: Similarity metric to use. Options: 'cosine', 'euclidean'.
            Defaults to 'cosine'.
    
    Returns:
        Similarity score between 0 and 1.
    
    Raises:
        ValueError: If vectors have different dimensions.
    
    Example:
        >>> a = np.array([1, 0, 1])
        >>> b = np.array([1, 1, 0])
        >>> compute_similarity(a, b)
        0.5
    """
    if vector_a.shape != vector_b.shape:
        raise ValueError(f"Shape mismatch: {vector_a.shape} vs {vector_b.shape}")
    
    if metric == "cosine":
        norm_a = np.linalg.norm(vector_a)
        norm_b = np.linalg.norm(vector_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(vector_a, vector_b) / (norm_a * norm_b))
    else:
        raise ValueError(f"Unknown metric: {metric}")
```

### Commit Messages

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters or less
- Reference issues and pull requests when relevant

**Examples:**
- `Add user-based collaborative filtering algorithm`
- `Fix memory leak in similarity matrix computation`
- `Update documentation for HybridRecommender class`
- `Refactor feature extraction pipeline for better performance`

## Testing Guidelines

- Write tests for all new functionality
- Maintain or improve code coverage
- Use descriptive test names that explain what's being tested
- Include edge cases and error conditions

```python
import pytest
from src.recommender import KNNRecommender


class TestKNNRecommender:
    """Tests for KNNRecommender class."""
    
    def test_fit_with_valid_data(self, sample_interaction_matrix, sample_books):
        """Test that fit() succeeds with valid input data."""
        recommender = KNNRecommender(n_neighbors=5)
        recommender.fit(sample_interaction_matrix, sample_books)
        assert recommender.is_fitted
    
    def test_recommend_returns_correct_count(self, fitted_recommender):
        """Test that recommend_for_user returns requested number of items."""
        recommendations = fitted_recommender.recommend_for_user(
            user_id=1, 
            n_recommendations=10
        )
        assert len(recommendations) == 10
    
    def test_recommend_raises_for_unknown_user(self, fitted_recommender):
        """Test that recommend_for_user raises ValueError for unknown user."""
        with pytest.raises(ValueError, match="Unknown user"):
            fitted_recommender.recommend_for_user(user_id=999999)
```

## Documentation

- Update docstrings for any modified functions/classes
- Update README.md if adding new features
- Add examples for new functionality
- Keep the API reference up to date

## Questions?

Feel free to open an issue with the "question" label or reach out to the maintainers.

Thank you for contributing! 🙏
