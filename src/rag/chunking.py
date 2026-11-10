"""
Semantic Text Chunking Engine for Medical Guidelines.
Splits text along paragraph and sentence boundaries while preserving overlap and metadata.
"""

from typing import List, Dict, Any
import re


class SemanticChunker:
    """Chunks radiology reference documents into dense semantic segments."""

    def __init__(self, chunk_size: int = 256, chunk_overlap: int = 32):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Transforms a collection of ingested documents into semantic chunks."""
        all_chunks = []

        for doc in documents:
            text = doc["raw_text"]
            # Split by paragraphs first
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

            chunk_idx = 0
            for para in paragraphs:
                words = para.split()
                if len(words) <= self.chunk_size:
                    all_chunks.append({
                        "chunk_id": f"{doc['doc_id']}_c{chunk_idx}",
                        "source": doc["source"],
                        "doc_id": doc["doc_id"],
                        "page_number": doc.get("page_number", 1),
                        "content": para,
                        "token_count": len(words),
                    })
                    chunk_idx += 1
                else:
                    # Slide over paragraph with overlap
                    start = 0
                    while start < len(words):
                        end = min(start + self.chunk_size, len(words))
                        chunk_words = words[start:end]
                        chunk_text = " ".join(chunk_words)

                        all_chunks.append({
                            "chunk_id": f"{doc['doc_id']}_c{chunk_idx}",
                            "source": doc["source"],
                            "doc_id": doc["doc_id"],
                            "page_number": doc.get("page_number", 1),
                            "content": chunk_text,
                            "token_count": len(chunk_words),
                        })
                        chunk_idx += 1
                        if end == len(words):
                            break
                        start += (self.chunk_size - self.chunk_overlap)

        return all_chunks

