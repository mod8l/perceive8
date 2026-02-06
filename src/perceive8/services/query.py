"""RAG query service for transcript Q&A."""

import asyncio
import logging
from typing import Optional

import openai
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from perceive8.config import Settings
from perceive8.models.database import TranscriptSegment
from perceive8.services.embedding import EmbeddingService

logger = logging.getLogger(__name__)


def _fmt_time(seconds: float) -> str:
    """Format seconds as MM:SS."""
    m, s = divmod(seconds, 60)
    return f"{int(m):02d}:{int(s):02d}"


class QueryService:
    """Retrieval-Augmented Generation query service over transcript embeddings."""

    def __init__(self, embedding_service: EmbeddingService, settings: Settings) -> None:
        self._embedding_service = embedding_service
        self._settings = settings
        self._openai_client = openai.AsyncClient(api_key=settings.openai_api_key)

    async def embed_text(self, text: str) -> list[float]:
        """Get embedding vector for a text string using OpenAI embeddings API."""
        response = await self._openai_client.embeddings.create(
            model=self._settings.openai_embedding_model,
            input=text,
        )
        return response.data[0].embedding

    async def embed_transcript_segments(
        self, analysis_id: str, segments: list[dict]
    ) -> None:
        """Embed each segment's text and store in ChromaDB.

        Args:
            analysis_id: The analysis identifier.
            segments: List of dicts with keys: speaker, start_time, end_time, text.
        """
        for i, seg in enumerate(segments):
            text = seg.get("text", "")
            if not text.strip():
                continue
            embedding = await self.embed_text(text)
            segment_id = f"{analysis_id}:{i}"
            metadata = {
                "analysis_id": analysis_id,
                "speaker": seg.get("speaker", "UNKNOWN"),
                "start_time": seg.get("start_time", 0.0),
                "end_time": seg.get("end_time", 0.0),
                "text": text,
            }
            self._embedding_service.add_transcript_embedding(
                segment_id=segment_id,
                embedding=embedding,
                metadata=metadata,
            )
            logger.debug("Embedded segment %s", segment_id)

    async def answer_question(
        self, question: str, analysis_id: Optional[str] = None, top_k: Optional[int] = None
    ) -> dict:
        """Answer a question about transcripts using RAG.

        If analysis_id is None, searches across ALL analyses.

        1. Embed the question.
        2. Search ChromaDB for relevant transcript segments.
        3. Build a prompt with retrieved context.
        4. Call OpenAI chat completion.
        5. Return answer and sources.
        """
        if top_k is None:
            top_k = self._settings.rag_top_k

        # 1. Embed question
        query_embedding = await self.embed_text(question)

        # 2. Search
        matches = self._embedding_service.search_transcripts(
            query_embedding=query_embedding,
            analysis_id=analysis_id,
            top_k=top_k,
        )

        if not matches:
            return {
                "answer": "No relevant transcript segments found for this analysis.",
                "sources": [],
            }

        # 3. Build prompt
        excerpts = []
        sources = []
        for match in matches:
            meta = match["metadata"]
            speaker = meta.get("speaker", "Unknown")
            start = meta.get("start_time", 0.0)
            end = meta.get("end_time", 0.0)
            text = meta.get("text", "")
            excerpts.append(f"[{speaker}, {_fmt_time(start)}-{_fmt_time(end)}]: \"{text}\"")
            seg_analysis_id = meta.get("analysis_id", analysis_id or "unknown")
            sources.append({
                "analysis_id": seg_analysis_id,
                "speaker_name": speaker,
                "start_time": start,
                "end_time": end,
                "text": text,
                "relevance_score": match.get("similarity", 0.0),
            })

        context = "\n".join(excerpts)
        system_prompt = (
            "You are an assistant that answers questions about audio conversations. "
            "Use ONLY the provided transcript excerpts to answer. If the answer is not "
            "in the excerpts, say so. Cite speakers and timestamps."
        )
        user_prompt = f"## Transcript Excerpts\n{context}\n\n## Question\n{question}"

        # 4. Generate answer
        chat_response = await self._openai_client.chat.completions.create(
            model=self._settings.openai_chat_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        answer = chat_response.choices[0].message.content or ""

        return {
            "answer": answer,
            "sources": sources,
        }

    async def backfill_from_db(self, db: AsyncSession) -> int:
        """Backfill ChromaDB from all TranscriptSegment records in PostgreSQL.

        Embeds segments in batches of 50 to avoid rate limits.
        Returns the count of segments embedded.
        """
        BATCH_SIZE = 50

        result = await db.execute(select(TranscriptSegment))
        segments = result.scalars().all()

        count = 0
        for i in range(0, len(segments), BATCH_SIZE):
            batch = segments[i : i + BATCH_SIZE]
            for seg in batch:
                text = seg.text
                if not text or not text.strip():
                    continue
                embedding = await self.embed_text(text)
                segment_id = f"backfill:{seg.id}"
                metadata = {
                    "analysis_id": str(seg.processing_run_id),
                    "speaker": "UNKNOWN",
                    "start_time": seg.start_time,
                    "end_time": seg.end_time,
                    "text": text,
                }
                self._embedding_service.add_transcript_embedding(
                    segment_id=segment_id,
                    embedding=embedding,
                    metadata=metadata,
                )
                count += 1
            # Small delay between batches to respect rate limits
            if i + BATCH_SIZE < len(segments):
                await asyncio.sleep(1.0)

        logger.info("Backfilled %d transcript segments into ChromaDB", count)
        return count
