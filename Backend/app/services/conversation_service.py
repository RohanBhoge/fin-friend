"""
Conversation service: CRUD with ownership validation and pagination.
"""

import uuid

from fastapi import HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.models import Conversation, Message, User


async def list_conversations(
    user: User, page: int, limit: int, db: AsyncSession
) -> tuple[list[Conversation], int]:
    """
    Fetch paginated conversations for a user, ordered by updated_at DESC.
    Returns (conversations, total_count).
    """
    # Count total
    count_query = select(func.count()).select_from(Conversation).where(
        Conversation.user_id == user.id
    )
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0

    # Fetch page
    offset = (page - 1) * limit
    query = (
        select(Conversation)
        .where(Conversation.user_id == user.id)
        .order_by(Conversation.updated_at.desc())
        .offset(offset)
        .limit(limit)
    )
    result = await db.execute(query)
    conversations = list(result.scalars().all())

    return conversations, total


async def create_conversation(
    user: User, title: str, db: AsyncSession
) -> Conversation:
    """Create a new conversation for the user."""
    conversation = Conversation(
        user_id=user.id,
        title=title,
    )
    db.add(conversation)
    await db.commit()
    await db.refresh(conversation)
    return conversation


async def get_conversation_detail(
    conversation_id: uuid.UUID, user: User, db: AsyncSession
) -> Conversation:
    """
    Fetch a conversation with all messages. Validates ownership.
    """
    query = (
        select(Conversation)
        .options(selectinload(Conversation.messages))
        .where(Conversation.id == conversation_id)
    )
    result = await db.execute(query)
    conversation = result.scalar_one_or_none()

    if conversation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )

    if conversation.user_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not own this conversation",
        )

    return conversation


async def rename_conversation(
    conversation_id: uuid.UUID, user: User, title: str, db: AsyncSession
) -> Conversation:
    """Rename a conversation. Validates ownership."""
    conversation = await get_conversation_detail(conversation_id, user, db)
    conversation.title = title
    await db.commit()
    await db.refresh(conversation)
    return conversation


async def delete_conversation(
    conversation_id: uuid.UUID, user: User, db: AsyncSession
) -> None:
    """Delete a conversation and all its messages. Validates ownership."""
    conversation = await get_conversation_detail(conversation_id, user, db)
    await db.delete(conversation)
    await db.commit()
