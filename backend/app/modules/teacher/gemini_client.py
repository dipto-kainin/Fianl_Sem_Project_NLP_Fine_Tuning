"""
Teacher Module - Gemini API Client.

Wrapper around the Google Gemini SDK for generating structured knowledge
from document chunks. Uses Pydantic response_schema for reliable JSON output.
"""

import logging
from functools import lru_cache

from google import genai
from google.genai import types
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from app.config import get_settings
from app.modules.teacher.schemas import TeacherStructuredOutput

logger = logging.getLogger(__name__)
settings = get_settings()


DEFAULT_TEACHER_SYSTEM_PROMPT = """You are an expert knowledge extraction AI reviewing a source document.
Your task is to extract all structured knowledge to train a smaller AI model that must accurately answer questions from memory.

CRITICAL EXTRACTION PRIORITIES (in order of importance):
1. CORE DATA & IDENTIFIERS — Extract exact names, phone numbers, email addresses, URLs, unique IDs, numbers, codes. These MUST appear verbatim in QA pair answers. Never paraphrase, approximate, or omit details.
2. KEY FACTS & METRICS — Every critical technology, component, process, timeline, database entry, and measurable metric must be captured in QA pairs.
3. DETAILED CONTEXT & EXPLANATIONS — Concepts, background descriptions, explanations of how things work.

PRIORITY LABELING — For each QA pair, assign a `priority` integer (1, 2, or 3) representing training weight:
- priority=3 (High Priority): Critical, exact, or digit-perfect facts (e.g. identifiers, contact details, URLs, names, codes) that must be memorized with 100% precision.
- priority=2 (Medium Priority): Important details, specific facts, components, metrics, or core definitions.
- priority=1 (Low Priority): General background descriptions, contextual explanations, or high-level summaries.

Generate 10-15 highly specific QA pairs per chunk. Each answer must be factually precise and contain EXACT values (numbers, URLs, dates) from the source — never approximate.
Tags should be broad topic categories relevant to the content."""


def amplify_user_prompt(raw_prompt: str) -> str:
    """
    Take a user's raw training intent (e.g. 'you are an HR reading a resume')
    and expand it with Gemini into a precise extraction system prompt that
    prioritizes contact info, links, skills, and projects.

    Returns the amplified system prompt string.
    """
    client = get_gemini_client()

    amplification_prompt = f"""A user wants to fine-tune a language model on a source document.
Their stated goal/persona/guideline: "{raw_prompt}"

Write a precise system prompt for a knowledge-extraction AI that will read this document and generate Q&A training pairs.
The prompt must:
1. Adopt the user's stated persona, goal, or extraction guidelines.
2. Instruct the extraction AI to assign a `priority` integer (1, 2, or 3) to each QA pair to weight training:
   - priority=3 (High Priority): Most critical, exact, or digit-perfect facts, IDs, core identifiers, or names that must be memorized with 100% precision.
   - priority=2 (Medium Priority): Crucial details, key metrics, components, concepts, or facts.
   - priority=1 (Low Priority): General background description, high-level context, or summaries.
3. Instruct the AI to generate very specific, factual Q&A pairs with exact values from the text.
4. Be detailed but concise (150-250 words).

Output ONLY the system prompt text. No preamble, no explanation, no markdown blocks, no JSON."""

    response = client.models.generate_content(
        model=settings.GEMINI_MODEL,
        contents=amplification_prompt,
        config=types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=512,
        ),
    )

    amplified = (response.text or "").strip()
    logger.info(f"Amplified user prompt ({len(amplified)} chars): {amplified[:120]}...")
    return amplified if amplified else DEFAULT_TEACHER_SYSTEM_PROMPT


@lru_cache(maxsize=1)
def get_gemini_client() -> genai.Client:
    """Get cached Gemini client instance."""
    return genai.Client(api_key=settings.GEMINI_API_KEY)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((Exception,)),
    before_sleep=lambda retry_state: logger.warning(
        f"Gemini API retry #{retry_state.attempt_number}: {retry_state.outcome.exception()}"
    ),
)
def process_chunk_with_teacher(
    chunk_text: str,
    section_title: str = "",
    document_context: str = "",
    custom_system_prompt: str | None = None,
) -> tuple[TeacherStructuredOutput, int]:
    """
    Process a single chunk with the Teacher LLM (Gemini).

    Args:
        chunk_text: The text content of the chunk.
        section_title: Optional section heading for context.
        document_context: Optional broader document context.

    Returns:
        Tuple of (TeacherStructuredOutput, total_tokens_used).
    """
    client = get_gemini_client()

    # Build the prompt with context
    prompt_parts = []
    if document_context:
        prompt_parts.append(f"Document context: {document_context}")
    if section_title:
        prompt_parts.append(f"Section: {section_title}")
    prompt_parts.append(f"Text chunk to analyze:\n\n{chunk_text}")
    prompt_parts.append(
        "\nExtract structured knowledge from this text. Generate an exhaustive set "
        "of detailedSummaries, entities, relationships, a large list of 10-15 QA pairs for training, "
        "comprehensive explanations, FAQs, and topic tags. Ensure every single detail from "
        "the text chunk is covered in at least one QA pair."
    )
    user_prompt = "\n\n".join(prompt_parts)

    system_prompt_to_use = custom_system_prompt or DEFAULT_TEACHER_SYSTEM_PROMPT

    response = client.models.generate_content(
        model=settings.GEMINI_MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt_to_use,
            response_mime_type="application/json",
            response_schema=TeacherStructuredOutput,
            temperature=0.3,
        ),
    )

    # Parse structured output
    parsed: TeacherStructuredOutput = response.parsed

    # Calculate tokens used
    tokens_used = 0
    if response.usage_metadata:
        tokens_used = (
            (response.usage_metadata.prompt_token_count or 0)
            + (response.usage_metadata.candidates_token_count or 0)
        )

    logger.info(f"Teacher processed chunk: {len(parsed.qa_pairs)} QA pairs, {tokens_used} tokens")
    return parsed, tokens_used


def generate_rag_answer(
    query: str,
    context_chunks: list[str],
    chunk_summaries: list[str] | None = None,
) -> tuple[str, int]:
    """
    Generate an answer using RAG with Gemini.

    Args:
        query: User's question.
        context_chunks: Retrieved chunk texts as context.
        chunk_summaries: Optional summaries of the chunks.

    Returns:
        Tuple of (answer_text, tokens_used).
    """
    client = get_gemini_client()

    # Build context
    context_parts = []
    for i, chunk in enumerate(context_chunks):
        header = f"--- Source {i+1} ---"
        if chunk_summaries and i < len(chunk_summaries):
            header += f" (Summary: {chunk_summaries[i]})"
        context_parts.append(f"{header}\n{chunk}")

    context = "\n\n".join(context_parts)

    prompt = f"""Based on the following context from relevant documents, answer the user's question.
If the context doesn't contain enough information to answer, say so clearly.
Always cite which source(s) you used.

Context:
{context}

Question: {query}

Answer:"""

    response = client.models.generate_content(
        model=settings.GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=1024,
        ),
    )

    answer = response.text or "I couldn't generate an answer from the available context."

    tokens_used = 0
    if response.usage_metadata:
        tokens_used = (
            (response.usage_metadata.prompt_token_count or 0)
            + (response.usage_metadata.candidates_token_count or 0)
        )

    return answer, tokens_used
