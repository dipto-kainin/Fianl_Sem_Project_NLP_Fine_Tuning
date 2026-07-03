"""
Teacher Module - Pydantic Schemas.

Schemas for Teacher LLM inputs, outputs, and API responses.
"""

import uuid
from datetime import datetime

from pydantic import BaseModel, Field


# --- Gemini Structured Output Schemas ---

class EntityExtraction(BaseModel):
    """Schema for an extracted entity."""
    name: str = Field(description="Entity name")
    type: str = Field(description="Entity type: person, organization, product, location, concept")
    description: str = Field(default="", description="Brief description of the entity")


class RelationshipExtraction(BaseModel):
    """Schema for an extracted relationship between entities."""
    source: str = Field(description="Source entity name")
    relationship: str = Field(description="Type of relationship")
    target: str = Field(description="Target entity name")


class QAPair(BaseModel):
    """Schema for an instruction-response pair."""
    instruction: str = Field(description="A natural question or instruction")
    answer: str = Field(description="The accurate answer based on the text")
    priority: int = Field(
        default=1,
        description=(
            "Training priority 1–3. "
            "3 = High Priority: most critical exact details, digit-perfect identifiers/names/codes/URLs. "
            "2 = Medium Priority: crucial facts, key entities, core metrics or definitions. "
            "1 = Low Priority: general background, context description, or high-level summaries."
        )
    )


class TeacherStructuredOutput(BaseModel):
    """
    Complete structured output schema sent to Gemini as response_schema.
    All fields are generated in a single API call per chunk.
    """
    summary: str = Field(description="A concise 2-3 sentence summary of the text chunk")
    entities: list[EntityExtraction] = Field(
        default_factory=list,
        description="Key entities extracted from the text"
    )
    relationships: list[RelationshipExtraction] = Field(
        default_factory=list,
        description="Relationships between extracted entities"
    )
    qa_pairs: list[QAPair] = Field(
        default_factory=list,
        description="10-15 highly detailed, exhaustive instruction-response pairs for training. Generate diverse questions covering every specific fact, name, metric, concept, step, process, and figure in the text chunk. Answers must be complete, precise, and contain all relevant details and context from the source."
    )
    explanation: str = Field(
        default="",
        description="A simplified explanation of the text for a non-expert"
    )
    faqs: list[str] = Field(
        default_factory=list,
        description="5-10 frequently asked questions about this content"
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Topic labels/tags (e.g., Finance, Engineering, Legal)"
    )


# --- API Response Schemas ---

class TeacherOutputResponse(BaseModel):
    """Response schema for teacher output."""

    id: uuid.UUID
    chunk_id: uuid.UUID
    summary: str | None = None
    entities: dict | None = None
    relationships: dict | None = None
    qa_pairs: list | None = None
    explanations: str | None = None
    faqs: list | None = None
    tags: list[str] | None = None
    tokens_used: int = 0
    created_at: datetime

    model_config = {"from_attributes": True}


class TeacherOutputListResponse(BaseModel):
    """Response schema for listing teacher outputs."""
    outputs: list[TeacherOutputResponse]
    total: int


class TeacherProcessRequest(BaseModel):
    """Request to process specific chunks with Teacher LLM."""
    chunk_ids: list[uuid.UUID] | None = Field(
        None,
        description="Specific chunk IDs to process. If None, process all unprocessed chunks."
    )
    document_id: uuid.UUID | None = Field(
        None,
        description="Process all chunks of this document."
    )


class TeacherStatsResponse(BaseModel):
    """Statistics about Teacher LLM usage."""
    total_chunks_processed: int
    total_tokens_used: int
    average_tokens_per_chunk: float
