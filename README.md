# Semantic Recommender System for arXiv Papers

This repository contains a Jupyter Notebook implementation of a semantic recommender system designed to enhance the discovery of academic papers within the arXiv dataset. Utilizing advanced NLP techniques and clustering algorithms, this project aims to recommend papers that are semantically related to a user's query.

## Overview

The Semantic Recommender System leverages the power of Sentence Transformers for generating semantic embeddings from paper abstracts. It explores several clustering and dimensionality reduction techniques to structure the high-dimensional embedding space into meaningful clusters of papers. These techniques include:

- **KMeans Clustering**: A straightforward approach to group papers based on their semantic proximity.
- **HDBSCAN Clustering**: Utilized for its ability to find clusters of varying densities and shapes, making it suitable for the diverse nature of academic papers.
- **UMAP with HDBSCAN**: Combines UMAP for dimensionality reduction with HDBSCAN for improved clustering performance in reduced-dimensional space.

The project evaluates these methods using a set of metrics to identify the most effective approach for semantic paper recommendation.

## Features

- **Semantic Embedding Generation**: Converts paper abstracts into high-dimensional vectors that capture their semantic essence.
- **Clustering for Recommendation**: Implements various clustering algorithms to organize papers into semantically meaningful groups.
- **Evaluation Metrics**: Employs metrics like normalized hit score, MAE, Euclidean distance, and Spearman correlation to assess the quality of recommendations.

## Usage

1. Clone this repository to your local machine.
2. Install the required Python packages: `pip install -r requirements.txt`
3. Open the `arxiv_recommender.ipynb` notebook in Jupyter Lab or Notebook and follow the instructions therein.

## Contributing

Contributions, suggestions, and issues are welcome. Please feel free to fork the repository, make your changes, and submit a pull request.

## License

This project is open-source and available under the MIT License. See the LICENSE file for more details.
