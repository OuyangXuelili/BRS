# Dataset Directory

# Synced with the current GitHub repository release.

This directory should contain the UCSD Book Graph dataset files.

## Download Instructions

1. Visit the [UCSD Book Graph](https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home) website
2. Download the required files:
   - `goodreads_books.json.gz` - Book metadata
   - `goodreads_interactions.csv` - User-book interactions (ratings)

### Direct Links

- Dataset homepage: https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home
- Alternative: https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html

## Expected Files

```
data/
├── goodreads_books.json.gz       # Book metadata (title, author, genre, etc.)
├── goodreads_interactions.csv    # User ratings and interactions
└── README.md                     # This file
```

## Dataset Statistics

| File | Size | Records |
|------|------|---------|
| Books | ~1.2 GB | 2.36M books |
| Interactions | ~3.5 GB | 229M ratings |

## Citation

If you use this dataset, please cite:

```bibtex
@inproceedings{wan2018item,
  title={Item recommendation on monotonic behavior chains},
  author={Wan, Mengting and McAuley, Julian},
  booktitle={Proceedings of the 12th ACM Conference on Recommender Systems},
  pages={86--94},
  year={2018}
}

@inproceedings{wan2019fine,
  title={Fine-grained spoiler detection from large-scale review corpora},
  author={Wan, Mengting and Misra, Rishabh and Nakashole, Ndapa and McAuley, Julian},
  booktitle={Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics},
  pages={2605--2610},
  year={2019}
}
```

## Sample Data for Testing

For quick testing without downloading the full dataset, the demo app generates synthetic sample data automatically. Run:

```bash
streamlit run demo/app.py
```
