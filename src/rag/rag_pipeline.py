"""
Unified Medical RAG Pipeline Coordinator.
Coordinates document loading, chunking, embedding, vector storage, and query retrieval.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
from src.config import ProjectConfig, get_config
from src.rag.document_loader import RadiologyDocumentLoader
from src.rag.chunking import SemanticChunker
from src.rag.embedding_model import MedicalEmbeddingModel
from src.rag.vector_store import MedicalVectorStore
from src.rag.retriever import MedicalRetriever
from src.utils.logger import get_logger

logger = get_logger("RAGPipeline")


class MedicalRAGPipeline:
    """End-to-end Retrieval-Augmented Generation pipeline for radiology guidelines."""

    def __init__(self, config: Optional[ProjectConfig] = None):
        self.cfg = config or get_config()
        self.rag_cfg = self.cfg.model.get("rag", {})

        self.knowledge_dir = self.cfg.base_dir / "data" / "knowledge_base"
        self.loader = RadiologyDocumentLoader(knowledge_dir=self.knowledge_dir)
        self.chunker = SemanticChunker(
            chunk_size=self.rag_cfg.get("chunk_size", 256),
            chunk_overlap=self.rag_cfg.get("chunk_overlap", 32),
        )
        self.embedding_model = MedicalEmbeddingModel(
            config=self.cfg,
            model_name=self.rag_cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
        )
        self.vector_store = MedicalVectorStore(dimension=self.embedding_model.dimension)
        self.retriever = MedicalRetriever(
            vector_store=self.vector_store,
            embedding_model=self.embedding_model,
            similarity_threshold=self.rag_cfg.get("similarity_threshold", 0.40),
        )

        self.index_file = self.cfg.output_dir / "faiss_index" / "radiology_reference.index"
        self.build_or_load_index()

    def build_or_load_index(
        self,
        chunks: Optional[List[Dict[str, Any]]] = None,
        force_rebuild: bool = False,
    ) -> None:
        """Loads persistent index from disk if present; otherwise ingests documents and builds."""
        if not force_rebuild and self.index_file.exists():
            success = self.vector_store.load(self.index_file)
            if success and self.vector_store.total_vectors > 0:
                logger.info(f"Loaded existing FAISS index with {self.vector_store.total_vectors} vectors from {self.index_file}")
                return

        logger.info("Building new FAISS vector index from radiology knowledge base...")
        if chunks is None:
            documents = self.loader.load_all_documents()
            chunks = self.chunker.chunk_documents(documents)

        if not chunks:
            logger.warning("No knowledge chunks available to index.")
            return

        texts = [c["content"] for c in chunks]
        embeddings = self.embedding_model.encode(texts)
        self.vector_store.add_chunks(chunks, embeddings)

        # Persist to disk
        self.vector_store.save(self.index_file)
        logger.info(f"Built and persisted FAISS index with {self.vector_store.total_vectors} vectors to {self.index_file}")

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieves top-k context passages for a given medical query."""
        return self.retriever.retrieve(query, top_k=top_k)

    def retrieve_for_visual_findings(self, findings: Dict[str, float], threshold: float = 0.50) -> List[Dict[str, Any]]:
        """Retrieves reference criteria for abnormalities predicted above threshold."""
        active_findings = [name for name, prob in findings.items() if prob >= threshold and name != "No Finding"]
        if not active_findings:
            active_findings = ["No Finding"]
        return self.retriever.retrieve_for_findings(active_findings, top_k_per_finding=1)

