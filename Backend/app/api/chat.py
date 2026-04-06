"""
Chat API routes: SSE streaming endpoint and title generation.
"""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import get_current_user
from app.db.database import get_db, verify_db_connection
from app.models.models import User
from app.schemas.chat_schemas import ChatRequest, GenerateTitleRequest, GenerateTitleResponse
from app.services.chat_service import (
    generate_conversation_title,
    stream_chat_response,
)

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post(
    "/stream",
    summary="Stream AI chat response via SSE",
)
async def stream_chat(
    body: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Send a message and receive a streaming AI response via Server-Sent Events.
    Each SSE event contains either a token, an error, or a done signal.
    """
    return StreamingResponse(
        stream_chat_response(
            conversation_id=body.conversation_id,
            user_message=body.message,
            db=db,
            user_id=current_user.id,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post(
    "/generate-title",
    response_model=GenerateTitleResponse,
    summary="Auto-generate conversation title",
)
async def generate_title(
    body: GenerateTitleRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Generate a 3-5 word title for a conversation based on its first messages."""
    title = await generate_conversation_title(
        body.conversation_id, db, current_user.id
    )
    return GenerateTitleResponse(title=title)


@router.get(
    "/health",
    tags=["System"],
    summary="Health check",
)
async def health_check():
    """Return server status, DB connectivity, and current timestamp."""
    db_connected = await verify_db_connection()
    return {
        "status": "ok" if db_connected else "degraded",
        "db": "connected" if db_connected else "disconnected",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
