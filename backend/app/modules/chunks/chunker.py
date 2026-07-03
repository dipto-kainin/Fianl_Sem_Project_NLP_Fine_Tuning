"""
Chunks Module - Semantic Chunking Engine.

Splits parsed documents into semantic chunks that respect section boundaries,
with configurable size and overlap.
"""

import logging
import re
from dataclasses import dataclass, field

from app.modules.documents.parser import ParsedDocument, ParsedSection

logger = logging.getLogger(__name__)


@dataclass
class ChunkResult:
    """Result of chunking a section of text."""
    text: str
    chunk_index: int
    section_title: str = ""
    page_numbers: list[int] = field(default_factory=list)
    token_count: int = 0
    metadata: dict = field(default_factory=dict)


def estimate_tokens(text: str) -> int:
    """
    Estimate the number of tokens in a text.

    Uses a simple heuristic: ~4 characters per token for English.
    More accurate than word counting for LLM token estimation.
    """
    return max(1, len(text) // 4)


def chunk_document(
    parsed: ParsedDocument,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[ChunkResult]:
    """
    Split a parsed document into semantic chunks.

    Strategy:
    1. Process each section independently to preserve boundaries.
    2. If a section fits within chunk_size, keep it as one chunk.
    3. If a section is too large, split by sentences with overlap.
    4. Maintain section title and page numbers as metadata.

    Args:
        parsed: Parsed document from the parser module.
        chunk_size: Target chunk size in estimated tokens.
        chunk_overlap: Overlap between consecutive chunks in estimated tokens.

    Returns:
        List of ChunkResult objects.
    """
    chunks: list[ChunkResult] = []
    global_index = 0

    for section in parsed.sections:
        content = section.content.strip()
        if not content:
            continue

        section_tokens = estimate_tokens(content)

        if section_tokens <= chunk_size:
            # Section fits in one chunk
            chunks.append(ChunkResult(
                text=content,
                chunk_index=global_index,
                section_title=section.title,
                page_numbers=section.page_numbers,
                token_count=section_tokens,
                metadata={"level": section.level},
            ))
            global_index += 1
        else:
            # Split section by sentences with overlap
            sub_chunks = _split_with_overlap(
                text=content,
                chunk_size=chunk_size,
                overlap=chunk_overlap,
            )
            for sub_text in sub_chunks:
                chunks.append(ChunkResult(
                    text=sub_text,
                    chunk_index=global_index,
                    section_title=section.title,
                    page_numbers=section.page_numbers,
                    token_count=estimate_tokens(sub_text),
                    metadata={"level": section.level},
                ))
                global_index += 1

    # Fallback: if no chunks from sections, chunk the raw text
    if not chunks and parsed.raw_text.strip():
        sub_chunks = _split_with_overlap(
            text=parsed.raw_text.strip(),
            chunk_size=chunk_size,
            overlap=chunk_overlap,
        )
        for i, sub_text in enumerate(sub_chunks):
            chunks.append(ChunkResult(
                text=sub_text,
                chunk_index=i,
                section_title="",
                token_count=estimate_tokens(sub_text),
            ))

    logger.info(f"Created {len(chunks)} chunks from document")
    return chunks


def _split_with_overlap(
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[str]:
    """
    Split text into chunks of approximately `chunk_size` tokens with `overlap`.

    Uses sentence boundaries as split points to avoid cutting mid-sentence.

    Args:
        text: Text to split.
        chunk_size: Target size in estimated tokens.
        overlap: Number of overlapping tokens between consecutive chunks.

    Returns:
        List of text chunks.
    """
    # Split into sentences
    sentences = _split_into_sentences(text)
    if not sentences:
        return [text] if text.strip() else []

    chunks: list[str] = []
    current_sentences: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = estimate_tokens(sentence)

        # If adding this sentence would exceed chunk_size
        if current_tokens + sentence_tokens > chunk_size and current_sentences:
            # Save current chunk
            chunks.append(" ".join(current_sentences))

            # Calculate how many sentences to keep for overlap
            overlap_sentences: list[str] = []
            overlap_tokens = 0
            for s in reversed(current_sentences):
                s_tokens = estimate_tokens(s)
                if overlap_tokens + s_tokens > overlap:
                    break
                overlap_sentences.insert(0, s)
                overlap_tokens += s_tokens

            current_sentences = overlap_sentences
            current_tokens = overlap_tokens

        current_sentences.append(sentence)
        current_tokens += sentence_tokens

    # Add remaining sentences as the last chunk
    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks


def _split_into_sentences(text: str) -> list[str]:
    """
    Split text into sentences using regex.

    Handles common abbreviations and edge cases.
    """
    # Split on sentence-ending punctuation followed by whitespace and uppercase
    sentence_pattern = re.compile(
        r'(?<=[.!?])\s+(?=[A-Z])|(?<=\n)\s*(?=\S)'
    )
    raw_sentences = sentence_pattern.split(text)

    # Clean and filter
    sentences = []
    for s in raw_sentences:
        s = s.strip()
        if s:
            sentences.append(s)

    return sentences
