# Spotify Audio Feature Analysis — Predicting Musical Taste from Sound

Can a song's measurable sound properties predict whether someone will like it? This project analyses **2,017 Spotify tracks** using supervised and unsupervised machine learning to find out — combining classification, dimensionality reduction, and clustering to study the relationship between audio features and personal preference.

## Overview

Each track in the dataset carries a binary `target` label (liked / disliked) and **14 audio features** (danceability, energy, valence, acousticness, instrumentalness, speechiness, and more). The analysis is organised around four research questions, each answered with a different technique.

| Research Question | Method | Finding |
|---|---|---|
| Can sound predict taste? | k-NN classification | Partially — 65.5% accuracy vs. a 50% baseline |
| Do liked songs sound different? | Distribution + correlation analysis | Yes, subtly — danceability is the strongest positive signal, acousticness the strongest negative |
| How many dimensions describe a song? | PCA | 6 components needed for 80% of variance — features are not redundant |
| Do songs form natural sound profiles? | K-Means clustering | 4 clusters; the most-liked scores high on danceability & valence, the least-liked high on acousticness |

## Key Results

- **Classification (k-NN):** Correctly classified 265 of 403 test songs (65.5%), meaningfully above chance — confirming audio features carry real signal about preference, though subjective taste caps how far this goes.
- **Feature analysis:** Danceability (r = 0.18) is the strongest positive predictor of liking; acousticness the strongest negative. Distributions overlap heavily, so the differences are real but not dramatic.
- **PCA:** Six principal components are required to retain 80% of variance, showing Spotify's audio features each measure a genuinely distinct sonic property rather than overlapping.
- **Clustering (K-Means):** Four clusters emerged (silhouette score 0.14), indicating music sits on a continuous spectrum rather than in sharply separated groups. Like-rates still differed meaningfully across clusters (58.5% down to 32.4%), independently confirming the supervised findings.

## Tech Stack

- **Python** — pandas, NumPy
- **scikit-learn** — KNeighborsClassifier, PCA, KMeans, StandardScaler
- **Matplotlib / Seaborn** — visualisation (box plots, confusion matrix, variance and cluster plots)
- **Jupyter Notebook**

## Methodology

1. **Preprocessing** — loaded and cleaned the dataset, scaled features with `StandardScaler` (essential for distance-based methods like k-NN and K-Means).
2. **Supervised learning** — trained a k-NN classifier with a train/test split and evaluated with accuracy and a confusion matrix.
3. **Feature relationships** — compared per-feature distributions across liked/disliked groups and quantified each feature's correlation with the target.
4. **Dimensionality reduction** — applied PCA and used a cumulative-variance plot to find the components needed to explain 80% of variance.
5. **Unsupervised learning** — clustered tracks with K-Means (features only, no labels) and measured the like-rate within each cluster to cross-check the supervised results.

## Limitations

- The target reflects one individual's taste, limiting how far the classification generalises.
- Context the data doesn't capture — genre, artist familiarity, release year, listening situation — likely explains much of the unexplained variance.
- At ~2,000 tracks the dataset suits exploratory analysis but is modest for robust ML conclusions.
- The low silhouette score (0.14) shows songs don't form strongly distinct clusters; K-Means is better suited to data with clearer boundaries.

## Data

Spotify Song Attributes dataset (Kaggle), 2,017 tracks with 14 audio features and a binary preference label.

## Running It

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook AE3_Spotify_Analysis.ipynb
```

---

*Built by Prithviraj — data science student focused on applying ML to practical analysis problems.*
