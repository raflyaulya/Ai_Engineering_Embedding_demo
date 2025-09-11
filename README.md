# Embedding Demo Projects — AI Engineering Playground

This repo contains multiple mini projects that demonstrate how embeddings work and why they are essential in modern AI applications — from semantic similarity to vector search and clustering.

## Project Structure

- `text_embedding_cosine/`: Compare two sentences using cosine similarity.
- `visualization_tsne/`: Visualize sentence embeddings using t-SNE.
- `semantic_faq_search/`: Semantic Q&A system using sentence-transformers.
- `multiple_clustering/`: Cluster multiple sentences using KMeans + embedding.
- `vector_db_intro/`: Simple intro to using ChromaDB for vector search.

## How to Run

```bash
# install dependencies
pip install -r requirements.txt

# run each script
python text_embedding_cosine/sbert_textEmbedding.py
python visualization_tsne/simple_embedding.py
...
