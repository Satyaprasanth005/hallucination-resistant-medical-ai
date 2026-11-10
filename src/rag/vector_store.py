"""
FAISS Vector Database for Radiology Reference Knowledge.
Supports persistent index storage, cosine similarity search, and chunk metadata tracking.
"""

from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import json
import numpy as np
import faiss


class MedicalVectorStore:
    """FAISS-backed vector store for medical knowledge chunks."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension) # Inner Product on normalized vectors = Cosine Similarity
        self.metadata_store: List[Dict[str, Any]] = []

    @property
    def total_vectors(self) -> int:
        return self.index.ntotal

    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray) -> None:
        """Adds embedded chunks to the FAISS index and records metadata."""
        if len(chunks) != len(embeddings):
            raise ValueError(f"Mismatch between chunks count ({len(chunks)}) and embeddings count ({len(embeddings)})")

        if len(chunks) == 0:
            return

        # Ensure embeddings are float32
        embeddings = embeddings.astype(np.float32)
        # Add to FAISS
        self.index.add(embeddings)
        self.metadata_store.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> List[Dict[str, Any]]:
        """Searches index for nearest neighbors by cosine similarity."""
        if self.total_vectors == 0:
            return []

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        query_embedding = query_embedding.astype(np.float32)
        k = min(top_k, self.total_vectors)

        scores, indices = self.index.search(query_embedding, k)

        results = []
        for rank in range(k):
            idx = int(indices[0][rank])
            score = float(scores[0][rank])
            if idx >= 0 and idx < len(self.metadata_store):
                meta = dict(self.metadata_store[idx])
                meta["similarity_score"] = score
                results.append(meta)

        return results

    def save(self, index_path: Path) -> None:
        """Persists the FAISS index binary and metadata JSON to disk."""
        index_path = Path(index_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)

        # Write FAISS index
        faiss.write_index(self.index, str(index_path))

        # Write metadata JSON
        meta_path = index_path.with_suffix(".json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata_store, f, indent=2)

    def load(self, index_path: Path) -> bool:
        """Loads a persisted FAISS index and metadata from disk."""
        index_path = Path(index_path)
        meta_path = index_path.with_suffix(".json")

        if not index_path.exists() or not meta_path.exists():
            return False

        self.index = faiss.read_index(str(index_path))
        with open(meta_path, "r", encoding="utf-8") as f:
            self.metadata_store = json.load(f)

        return True

