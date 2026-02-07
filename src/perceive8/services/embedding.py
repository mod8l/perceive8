"""ChromaDB embedding service for speaker and transcript embeddings."""

import logging
from typing import Optional

import chromadb

from perceive8.config import Settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Manages ChromaDB collections for speaker and transcript embeddings."""

    def __init__(self, settings: Settings) -> None:
        if settings.chromadb_host:
            headers = {}
            if settings.chromadb_token:
                headers["Authorization"] = f"Bearer {settings.chromadb_token}"
            self._client = chromadb.HttpClient(
                host=settings.chromadb_host,
                port=settings.chromadb_port,
                headers=headers,
            )
            logger.info(
                "EmbeddingService using HttpClient (host=%s, port=%d)",
                settings.chromadb_host,
                settings.chromadb_port,
            )
        else:
            self._client = chromadb.PersistentClient(path=settings.chromadb_path)
            logger.info("EmbeddingService using PersistentClient (path=%s)", settings.chromadb_path)

        self.speaker_collection = self._client.get_or_create_collection(
            name="speaker_embeddings",
            metadata={"hnsw:space": "cosine"},
        )
        self.transcript_collection = self._client.get_or_create_collection(
            name="transcript_embeddings",
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "EmbeddingService initialized (speakers=%d, transcripts=%d)",
            self.speaker_collection.count(),
            self.transcript_collection.count(),
        )

    # --- Speaker embeddings ---

    def add_speaker_embedding(
        self, speaker_id: str, embedding: list[float], metadata: dict
    ) -> None:
        """Upsert a speaker embedding into ChromaDB."""
        self.speaker_collection.upsert(
            ids=[speaker_id],
            embeddings=[embedding],
            metadatas=[metadata],
        )
        logger.info("Upserted speaker embedding: %s", speaker_id)

    def search_similar_speakers(
        self,
        embedding: list[float],
        user_id: str,
        threshold: float = 0.8,
        top_k: int = 3,
    ) -> list[dict]:
        """Query for similar speakers belonging to user_id.

        Returns matches with similarity >= threshold (cosine similarity = 1 - distance).
        """
        results = self.speaker_collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where={"user_id": user_id},
        )

        matches: list[dict] = []
        if results and results["ids"] and results["ids"][0]:
            for i, speaker_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i] if results["distances"] else 1.0
                similarity = 1.0 - distance
                if similarity >= threshold:
                    matches.append(
                        {
                            "speaker_id": speaker_id,
                            "similarity": similarity,
                            "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                        }
                    )
        return matches

    def delete_speaker_embedding(self, speaker_id: str) -> None:
        """Delete a speaker embedding from ChromaDB."""
        try:
            self.speaker_collection.delete(ids=[speaker_id])
            logger.info("Deleted speaker embedding: %s", speaker_id)
        except Exception:
            logger.warning("Failed to delete speaker embedding: %s", speaker_id, exc_info=True)

    # --- Transcript embeddings ---

    def add_transcript_embedding(
        self, segment_id: str, embedding: list[float], metadata: dict
    ) -> None:
        """Upsert a transcript segment embedding for future RAG.

        metadata must include 'user_id' to ensure proper data isolation.
        """
        if "user_id" not in metadata:
            logger.warning("Transcript embedding %s missing user_id in metadata", segment_id)
        self.transcript_collection.upsert(
            ids=[segment_id],
            embeddings=[embedding],
            metadatas=[metadata],
        )

    def search_transcripts(
        self,
        query_embedding: list[float],
        user_id: str,
        analysis_id: Optional[str] = None,
        top_k: int = 5,
    ) -> list[dict]:
        """Search transcript embeddings scoped to a user_id.

        Always filters by user_id for data isolation. Optionally narrows
        further by analysis_id.
        """
        query_kwargs: dict = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
        }
        where_filter: dict = {"user_id": user_id}
        if analysis_id is not None:
            where_filter = {"$and": [{"user_id": user_id}, {"analysis_id": analysis_id}]}
        query_kwargs["where"] = where_filter
        results = self.transcript_collection.query(**query_kwargs)

        matches: list[dict] = []
        if results and results["ids"] and results["ids"][0]:
            for i, segment_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i] if results["distances"] else 1.0
                matches.append(
                    {
                        "segment_id": segment_id,
                        "similarity": 1.0 - distance,
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    }
                )
        return matches
