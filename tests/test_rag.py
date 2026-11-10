"""
Unit tests for Medical RAG Pipeline, Vector Store, and Retriever.
"""

import unittest
from pathlib import Path
from src.config import get_config
from src.rag.document_loader import RadiologyDocumentLoader
from src.rag.chunking import SemanticChunker
from src.rag.embedding_model import MedicalEmbeddingModel
from src.rag.vector_store import MedicalVectorStore
from src.rag.retriever import MedicalRetriever
from src.rag.rag_pipeline import MedicalRAGPipeline


class TestRAG(unittest.TestCase):

    def setUp(self):
        self.cfg = get_config()
        self.loader = RadiologyDocumentLoader()
        self.chunker = SemanticChunker(chunk_size=100, chunk_overlap=20)
        self.embedding_model = MedicalEmbeddingModel(config=self.cfg)
        self.vector_store = MedicalVectorStore(dimension=self.embedding_model.dimension)

    def test_document_loader_and_seeding(self):
        docs = self.loader.load_all_documents()
        self.assertGreater(len(docs), 0)
        first_doc = docs[0]
        self.assertIn("doc_id", first_doc)
        self.assertIn("raw_text", first_doc)
        self.assertIn("source", first_doc)

    def test_semantic_chunker(self):
        docs = self.loader.load_all_documents()
        chunks = self.chunker.chunk_documents(docs)
        self.assertGreater(len(chunks), 0)
        first_chunk = chunks[0]
        self.assertIn("chunk_id", first_chunk)
        self.assertIn("content", first_chunk)
        self.assertIn("token_count", first_chunk)

    def test_vector_store_and_search(self):
        chunks = [
            {"chunk_id": "c1", "source": "ref1", "content": "Cardiomegaly is enlargement of the heart silhouette.", "page_number": 1},
            {"chunk_id": "c2", "source": "ref2", "content": "Pneumothorax shows visceral pleural line without markings.", "page_number": 1},
        ]
        embeddings = self.embedding_model.encode([c["content"] for c in chunks])
        self.vector_store.add_chunks(chunks, embeddings)
        self.assertEqual(self.vector_store.total_vectors, 2)

        q_vec = self.embedding_model.encode("enlarged heart and cardiac silhouette")
        results = self.vector_store.search(q_vec, top_k=1)
        self.assertEqual(len(results), 1)
        self.assertIn("similarity_score", results[0])

    def test_rag_pipeline_end_to_end(self):
        pipeline = MedicalRAGPipeline(self.cfg)
        results = pipeline.retrieve("pleural effusion blunting", top_k=2)
        self.assertIsInstance(results, list)
        if results:
            self.assertIn("score", results[0])
            self.assertIn("text", results[0])


if __name__ == "__main__":
    unittest.main()

