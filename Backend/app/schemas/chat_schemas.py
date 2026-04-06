"""
Pydantic V2 schemas for chat and message endpoints.
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class MessageRead(BaseModel):
    """Schema for a single message in a conversation."""
    id: UUID
    role: str
    content: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ChatRequest(BaseModel):
    """Schema for the SSE streaming chat endpoint."""
    conversation_id: UUID
    message: str = Field(..., min_length=1, max_length=10000)


class GenerateTitleRequest(BaseModel):
    """Schema for auto-generating a conversation title."""
    conversation_id: UUID


class GenerateTitleResponse(BaseModel):
    """Response with the generated title."""
    title: str
