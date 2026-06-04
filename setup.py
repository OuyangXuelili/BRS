"""
Book Recommendation System
==========================

A production-ready book recommendation system using K-Nearest Neighbors
and collaborative filtering on Goodreads-style data.

Author: Quang
Email: oneechansakurajimamai@gmail.com
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read long description from README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements
requirements = []
requirements_path = this_directory / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path, "r", encoding="utf-8") as f:
        requirements = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]

setup(
    name="book-recommendation-system",
    version="1.0.0",
    author="Quang",
    author_email="oneechansakurajimamai@gmail.com",
    description="A collaborative filtering book recommendation system using KNN",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/OuyangXuelili/BRS",
    project_urls={
        "Bug Tracker": "https://github.com/OuyangXuelili/BRS/issues",
        "Documentation": "https://github.com/OuyangXuelili/BRS#readme",
        "Source Code": "https://github.com/OuyangXuelili/BRS",
    },
    packages=find_packages(exclude=["tests", "tests.*", "notebooks"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "tqdm>=4.62.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "demo": [
            "streamlit>=1.10.0",
            "plotly>=5.0.0",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "notebook>=6.4.0",
            "ipykernel>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "book-recommend=src.recommender:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "recommendation-system",
        "collaborative-filtering",
        "knn",
        "machine-learning",
        "books",
        "goodreads",
        "scikit-learn",
    ],
)
