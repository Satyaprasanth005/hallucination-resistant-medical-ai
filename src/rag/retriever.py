"""
Medical Knowledge Retriever and Context Boundary Manager.
Enforces separation between general reference knowledge and patient-specific visual findings.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from src.rag.embedding_model import MedicalEmbeddingModel
from src.rag.vector_store import MedicalVectorStore


class MedicalRetriever:
    """Retrieves relevant clinical knowledge with evidence boundary controls."""

    def __init__(
        self,
        vector_store: MedicalVectorStore,
        embedding_model: MedicalEmbeddingModel,
        similarity_threshold: float = 0.40,
    ):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Queries the vector store and filters results by similarity threshold."""
        query_vec = self.embedding_model.encode(query)
        raw_results = self.vector_store.search(query_vec, top_k=top_k)

        filtered = []
        for res in raw_results:
            score = res.get("similarity_score", 0.0)
            if score >= self.similarity_threshold:
                filtered.append({
                    "chunk_id": res["chunk_id"],
                    "source": res["source"],
                    "score": round(score, 4),
                    "text": res["content"],
                    "knowledge_type": "GENERAL_REFERENCE_DOCUMENTATION",
                })

        return filtered

    def retrieve_for_findings(self, finding_names: List[str], top_k_per_finding: int = 1) -> List[Dict[str, Any]]:
        """Retrieves relevant clinical diagnostic criteria for a set of detected findings."""
        aggregated = []
        seen_chunks = set()

        for finding in finding_names:
            query = f"Radiological signs, diagnostic criteria, and chest X-ray findings for {finding}"
            results = self.retrieve(query, top_k=top_k_per_finding)
            for r in results:
                if r["chunk_id"] not in seen_chunks:
                    seen_chunks.add(r["chunk_id"])
                    aggregated.append(r)

        return aggregated

