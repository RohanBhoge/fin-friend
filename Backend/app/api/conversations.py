"""
Conversation API routes: list, create, get detail, rename, delete.
"""

import uuid

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import get_current_user
from app.db.database import get_db
from app.models.models import User
from app.schemas.conversation_schemas import (
    ConversationCreate,
    ConversationDetail,
    ConversationList,
    ConversationRead,
    ConversationUpdate,
)
from app.services.conversation_service import (
    create_conversation,
    delete_conversation,
    get_conversation_detail,
    list_conversations,
    rename_conversation,
)

router = APIRouter(prefix="/conversations", tags=["Conversations"])


@router.get(
    "",
    response_model=ConversationList,
    summary="List user's conversations",
)
async def list_user_conversations(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Fetch paginated list of conversations ordered by most recent."""
    conversations, total = await list_conversations(current_user, page, limit, db)
    return ConversationList(
        items=[ConversationRead.model_validate(c) for c in conversations],
        total=total,
        page=page,
        limit=limit,
    )


@router.post(
    "",
    response_model=ConversationRead,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new conversation",
)
async def create_new_conversation(
    body: ConversationCreate = ConversationCreate(),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Create a new conversation session."""
    conversation = await create_conversation(current_user, body.title, db)
    return ConversationRead.model_validate(conversation)


@router.get(
    "/{conversation_id}",
    response_model=ConversationDetail,
    summary="Get conversation with messages",
)
async def get_conversation(
    conversation_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Fetch a conversation and all its messages. Must be the owner."""
    conversation = await get_conversation_detail(conversation_id, current_user, db)
    return ConversationDetail.model_validate(conversation)


@router.patch(
    "/{conversation_id}",
    response_model=ConversationRead,
    summary="Rename a conversation",
)
async def rename(
    conversation_id: uuid.UUID,
    body: ConversationUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Update the title of a conversation. Must be the owner."""
    conversation = await rename_conversation(
        conversation_id, current_user, body.title, db
    )
    return ConversationRead.model_validate(conversation)


@router.delete(
    "/{conversation_id}",
    summary="Delete a conversation",
)
async def delete(
    conversation_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete a conversation and all its messages. Must be the owner."""
    await delete_conversation(conversation_id, current_user, db)
    return {"message": "Conversation deleted."}
