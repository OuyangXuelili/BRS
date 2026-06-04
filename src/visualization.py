"""
Visualization Utilities for Book Recommendation Analysis.

This module provides publication-ready visualizations for exploring
recommendation data, model performance, and feature distributions.

Functions:
    plot_rating_distribution: Histogram of rating frequencies
    plot_user_activity: User engagement patterns
    plot_model_comparison: Compare multiple recommender models
    plot_similarity_heatmap: Item or user similarity visualization
    
Example:
    >>> from src.visualization import plot_rating_distribution
    >>> fig = plot_rating_distribution(ratings_df)
    >>> fig.savefig("figures/rating_dist.png")
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

logger = logging.getLogger(__name__)

# Configure default style
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "figure.dpi": 100,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 16
})

# Color palette
COLORS = {
    "primary": "#2E86AB",
    "secondary": "#A23B72",
    "accent": "#F18F01",
    "success": "#C73E1D",
    "neutral": "#3B3B3B",
    "palette": ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#1B998B"]
}


def plot_rating_distribution(
    ratings_df: pd.DataFrame,
    rating_col: str = "rating",
    title: str = "Rating Distribution",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot distribution of ratings.
    
    Args:
        ratings_df: DataFrame with ratings
        rating_col: Column name for ratings
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ratings = ratings_df[rating_col].dropna()
    
    # Create histogram
    unique_ratings = sorted(ratings.unique())
    counts = ratings.value_counts().sort_index()
    
    bars = ax.bar(
        counts.index,
        counts.values,
        color=COLORS["primary"],
        edgecolor="white",
        linewidth=1.5,
        alpha=0.85
    )
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{int(height):,}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold"
        )
        
    # Add statistics
    mean_rating = ratings.mean()
    ax.axvline(mean_rating, color=COLORS["secondary"], linestyle="--", 
               linewidth=2, label=f"Mean: {mean_rating:.2f}")
    
    ax.set_xlabel("Rating", fontweight="bold")
    ax.set_ylabel("Count", fontweight="bold")
    ax.set_title(title, fontweight="bold", pad=20)
    ax.legend(loc="upper left")
    
    # Add summary stats text box
    stats_text = (
        f"Total: {len(ratings):,}\n"
        f"Mean: {mean_rating:.2f}\n"
        f"Std: {ratings.std():.2f}"
    )
    ax.text(
        0.95, 0.95, stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        fontsize=10
    )
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Figure saved to {save_path}")
        
    return fig


def plot_user_activity(
    ratings_df: pd.DataFrame,
    n_bins: int = 50,
    title: str = "User Activity Distribution",
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot distribution of ratings per user.
    
    Args:
        ratings_df: DataFrame with ratings
        n_bins: Number of histogram bins
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Ratings per user
    user_counts = ratings_df.groupby("user_id").size()
    
    axes[0].hist(
        user_counts,
        bins=n_bins,
        color=COLORS["primary"],
        edgecolor="white",
        alpha=0.85
    )
    axes[0].set_xlabel("Ratings per User", fontweight="bold")
    axes[0].set_ylabel("Number of Users", fontweight="bold")
    axes[0].set_title("Ratings per User", fontweight="bold")
    
    # Add median line
    median_ratings = user_counts.median()
    axes[0].axvline(median_ratings, color=COLORS["secondary"], linestyle="--",
                    linewidth=2, label=f"Median: {median_ratings:.0f}")
    axes[0].legend()
    
    # Ratings per book
    book_counts = ratings_df.groupby("book_id").size()
    
    axes[1].hist(
        book_counts,
        bins=n_bins,
        color=COLORS["accent"],
        edgecolor="white",
        alpha=0.85
    )
    axes[1].set_xlabel("Ratings per Book", fontweight="bold")
    axes[1].set_ylabel("Number of Books", fontweight="bold")
    axes[1].set_title("Ratings per Book", fontweight="bold")
    
    median_book = book_counts.median()
    axes[1].axvline(median_book, color=COLORS["secondary"], linestyle="--",
                    linewidth=2, label=f"Median: {median_book:.0f}")
    axes[1].legend()
    
    plt.suptitle(title, fontweight="bold", fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        
    return fig


def plot_model_comparison(
    results: Dict[str, Dict],
    metrics: List[str] = None,
    title: str = "Model Performance Comparison",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Compare performance of multiple models.
    
    Args:
        results: Dict mapping model name to metrics dict
        metrics: List of metric names to plot
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib Figure object
        
    Example:
        >>> results = {
        ...     "KNN": {"Precision@10": 0.12, "Recall@10": 0.08},
        ...     "Hybrid": {"Precision@10": 0.15, "Recall@10": 0.10}
        ... }
        >>> plot_model_comparison(results)
    """
    if metrics is None:
        # Get all metrics from first model
        first_model = list(results.values())[0]
        metrics = list(first_model.keys())
        
    n_metrics = len(metrics)
    n_models = len(results)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(n_metrics)
    width = 0.8 / n_models
    
    for i, (model_name, model_results) in enumerate(results.items()):
        values = [model_results.get(m, 0) for m in metrics]
        offset = (i - n_models / 2 + 0.5) * width
        
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=model_name,
            color=COLORS["palette"][i % len(COLORS["palette"])],
            alpha=0.85,
            edgecolor="white"
        )
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                rotation=0
            )
            
    ax.set_xlabel("Metric", fontweight="bold")
    ax.set_ylabel("Score", fontweight="bold")
    ax.set_title(title, fontweight="bold", pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45, ha="right")
    ax.legend(loc="upper right")
    ax.set_ylim(0, max(max(r.values()) for r in results.values()) * 1.2)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        
    return fig


def plot_precision_recall_curve(
    precisions: List[float],
    recalls: List[float],
    k_values: List[int],
    model_name: str = "Model",
    title: str = "Precision-Recall Curve",
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot precision-recall curve across K values.
    
    Args:
        precisions: List of precision values
        recalls: List of recall values
        k_values: Corresponding K values
        model_name: Name for legend
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(
        recalls, precisions,
        marker="o",
        linewidth=2,
        markersize=8,
        color=COLORS["primary"],
        label=model_name
    )
    
    # Annotate K values
    for k, p, r in zip(k_values, precisions, recalls):
        ax.annotate(
            f"K={k}",
            (r, p),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9
        )
        
    ax.set_xlabel("Recall", fontweight="bold")
    ax.set_ylabel("Precision", fontweight="bold")
    ax.set_title(title, fontweight="bold", pad=20)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        
    return fig


def plot_similarity_heatmap(
    similarity_matrix: np.ndarray,
    labels: Optional[List[str]] = None,
    title: str = "Item Similarity Matrix",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot heatmap of similarity scores.
    
    Args:
        similarity_matrix: Square similarity matrix
        labels: Optional labels for rows/columns
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use first N items if matrix is large
    max_size = 50
    if similarity_matrix.shape[0] > max_size:
        similarity_matrix = similarity_matrix[:max_size, :max_size]
        if labels:
            labels = labels[:max_size]
            
    sns.heatmap(
        similarity_matrix,
        ax=ax,
        cmap="RdBu_r",
        center=0,
        xticklabels=labels if labels else False,
        yticklabels=labels if labels else False,
        square=True,
        cbar_kws={"label": "Similarity"}
    )
    
    ax.set_title(title, fontweight="bold", pad=20)
    
    if labels:
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        
    return fig


def plot_genre_distribution(
    books_df: pd.DataFrame,
    genre_col: str = "genre",
    top_n: int = 10,
    title: str = "Top Book Genres",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot distribution of book genres.
    
    Args:
        books_df: DataFrame with book metadata
        genre_col: Column name for genre
        top_n: Number of top genres to show
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    genre_counts = books_df[genre_col].value_counts().head(top_n)
    
    bars = ax.barh(
        range(len(genre_counts)),
        genre_counts.values,
        color=COLORS["palette"][0],
        edgecolor="white",
        alpha=0.85
    )
    
    ax.set_yticks(range(len(genre_counts)))
    ax.set_yticklabels(genre_counts.index)
    ax.invert_yaxis()
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax.annotate(
            f"{int(width):,}",
            xy=(width, bar.get_y() + bar.get_height() / 2),
            xytext=(5, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=10
        )
        
    ax.set_xlabel("Number of Books", fontweight="bold")
    ax.set_ylabel("Genre", fontweight="bold")
    ax.set_title(title, fontweight="bold", pad=20)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        
    return fig


def create_dashboard(
    ratings_df: pd.DataFrame,
    books_df: pd.DataFrame,
    results: Optional[Dict] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create comprehensive dashboard with multiple plots.
    
    Args:
        ratings_df: Ratings DataFrame
        books_df: Books DataFrame
        results: Optional model results for comparison
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib Figure object
    """
    fig = plt.figure(figsize=(16, 12))
    
    # Rating distribution
    ax1 = fig.add_subplot(2, 2, 1)
    ratings = ratings_df["rating"].dropna()
    counts = ratings.value_counts().sort_index()
    ax1.bar(counts.index, counts.values, color=COLORS["primary"], alpha=0.85)
    ax1.set_xlabel("Rating")
    ax1.set_ylabel("Count")
    ax1.set_title("Rating Distribution", fontweight="bold")
    
    # User activity
    ax2 = fig.add_subplot(2, 2, 2)
    user_counts = ratings_df.groupby("user_id").size()
    ax2.hist(user_counts, bins=50, color=COLORS["accent"], alpha=0.85)
    ax2.set_xlabel("Ratings per User")
    ax2.set_ylabel("Number of Users")
    ax2.set_title("User Activity", fontweight="bold")
    
    # Book popularity
    ax3 = fig.add_subplot(2, 2, 3)
    book_counts = ratings_df.groupby("book_id").size()
    ax3.hist(book_counts, bins=50, color=COLORS["secondary"], alpha=0.85)
    ax3.set_xlabel("Ratings per Book")
    ax3.set_ylabel("Number of Books")
    ax3.set_title("Book Popularity", fontweight="bold")
    
    # Model comparison or genre distribution
    ax4 = fig.add_subplot(2, 2, 4)
    
    if results:
        metrics = list(list(results.values())[0].keys())[:5]
        x = np.arange(len(metrics))
        width = 0.35
        
        for i, (name, vals) in enumerate(results.items()):
            values = [vals.get(m, 0) for m in metrics]
            ax4.bar(x + i * width, values, width, label=name, 
                   color=COLORS["palette"][i], alpha=0.85)
                   
        ax4.set_xticks(x + width / 2)
        ax4.set_xticklabels(metrics, rotation=45, ha="right")
        ax4.legend()
        ax4.set_title("Model Performance", fontweight="bold")
    else:
        if "genre" in books_df.columns:
            genre_counts = books_df["genre"].value_counts().head(8)
            ax4.barh(range(len(genre_counts)), genre_counts.values,
                    color=COLORS["primary"], alpha=0.85)
            ax4.set_yticks(range(len(genre_counts)))
            ax4.set_yticklabels(genre_counts.index)
            ax4.invert_yaxis()
            ax4.set_title("Top Genres", fontweight="bold")
        else:
            ax4.text(0.5, 0.5, "No genre data available",
                    ha="center", va="center", transform=ax4.transAxes)
                    
    plt.suptitle("Book Recommendation System - Data Overview", 
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        
    return fig
