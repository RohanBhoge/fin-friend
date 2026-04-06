"""
Pydantic V2 schemas for conversation CRUD.
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.chat_schemas import MessageRead


class ConversationCreate(BaseModel):
    """Schema for creating a new conversation."""
    title: str = "New Conversation"


class ConversationRead(BaseModel):
    """Schema for conversation list items."""
    id: UUID
    title: str
    created_at: datetime
    updated_at: datetime | None = None

    model_config = ConfigDict(from_attributes=True)


class ConversationUpdate(BaseModel):
    """Schema for renaming a conversation."""
    title: str = Field(..., min_length=1, max_length=100)


class ConversationDetail(BaseModel):
    """Schema for a conversation with all its messages."""
    id: UUID
    title: str
    messages: list[MessageRead] = []
    created_at: datetime
    updated_at: datetime | None = None

    model_config = ConfigDict(from_attributes=True)


class ConversationList(BaseModel):
    """Paginated list of conversations."""
    items: list[ConversationRead]
    total: int
    page: int
    limit: int
