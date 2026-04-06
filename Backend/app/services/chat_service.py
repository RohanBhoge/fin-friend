"""
Chat service: RAG pipeline with FAISS retrieval and Gemini 2.5 Flash streaming.
This is the core AI engine of FinFriend.
"""

import json
import logging
import os
import uuid
from typing import AsyncGenerator

import google.generativeai as genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.models import Conversation, Message, MessageRole

logger = logging.getLogger("finfriend.chat")

# ── Module-level singletons ──────────────────────────────────────────────────
_faiss_retriever = None
_embedding_model = None


# ── FAISS Index Build / Load ─────────────────────────────────────────────────
async def build_faiss_index() -> None:
    """
    Build or load the FAISS vector index from the financial knowledge base.
    Called once on application startup.
    """
    global _faiss_retriever, _embedding_model

    _embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=settings.GEMINI_API_KEY
    )

    index_path = settings.FAISS_INDEX_PATH

    # If index already exists on disk, load it
    if os.path.exists(index_path) and os.path.isdir(index_path):
        logger.info(f"📂 Loading existing FAISS index from {index_path}")
        vectorstore = FAISS.load_local(
            index_path, _embedding_model, allow_dangerous_deserialization=True
        )
        _faiss_retriever = vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        )
        logger.info("✅ FAISS index loaded successfully")
        return

    # Build from documents
    documents_path = settings.DOCUMENTS_PATH
    txt_file = os.path.join(documents_path, "manageMoney.txt")

    if not os.path.exists(txt_file):
        logger.warning(
            f"⚠️ Knowledge base not found at {txt_file}. "
            "RAG will be unavailable."
        )
        return

    logger.info(f"📄 Loading document: {txt_file}")
    loader = TextLoader(txt_file, encoding="utf-8")
    documents = loader.load()

    logger.info("✂️ Splitting text into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"📦 Created {len(chunks)} chunks")

    logger.info("🏗️ Building FAISS index...")
    vectorstore = FAISS.from_documents(chunks, _embedding_model)

    # Persist to disk
    vectorstore.save_local(index_path)
    logger.info(f"💾 FAISS index saved to {index_path}")

    _faiss_retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    )
    logger.info("✅ FAISS index built and ready")


def _get_rag_context(user_message: str) -> str:
    """Retrieve relevant document chunks for the user's message."""
    if _faiss_retriever is None:
        return "No financial knowledge base available."

    try:
        docs = _faiss_retriever.invoke(user_message)
        if not docs:
            return "No relevant context found in knowledge base."
        return "\n\n".join(doc.page_content for doc in docs)
    except Exception as e:
        logger.error(f"RAG retrieval error: {e}")
        return "Error retrieving context from knowledge base."


# ── System Prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT_TEMPLATE = """You are FinFriend, a professional and empathetic AI financial health assistant.
Your role is to provide clear, actionable, and personalized financial guidance.

Use the following verified financial knowledge to inform your response:
--- FINANCIAL KNOWLEDGE BASE ---
{rag_context}
--- END OF KNOWLEDGE BASE ---

Response formatting rules:
- Use Markdown formatting: headers, bullet points, bold for key figures.
- Be concise but comprehensive.
- Always recommend consulting a certified financial advisor for major decisions.
- Never fabricate statistics or guarantees.
- If the user shares financial data, analyze it thoroughly with specific numbers.
- Be encouraging and non-judgmental about financial situations.
"""


# ── Streaming Pipeline ────────────────────────────────────────────────────────
async def stream_chat_response(
    conversation_id: uuid.UUID,
    user_message: str,
    db: AsyncSession,
    user_id: uuid.UUID,
) -> AsyncGenerator[str, None]:
    """
    Async generator that yields SSE-formatted strings for the chat stream.

    Pipeline:
    1. Save user message to DB
    2. Fetch conversation history
    3. RAG retrieval
    4. Construct system prompt
    5. Stream Gemini response (token-by-token)
    6. Save assistant response to DB after stream completes
    """

    # ── Step 1: Verify conversation ownership & save user message ─────
    result = await db.execute(
        select(Conversation).where(Conversation.id == conversation_id)
    )
    conversation = result.scalar_one_or_none()

    if conversation is None or conversation.user_id != user_id:
        yield f"data: {json.dumps({'error': 'Conversation not found or access denied'})}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"
        return

    user_msg = Message(
        conversation_id=conversation_id,
        role=MessageRole.user,
        content=user_message,
    )
    db.add(user_msg)
    await db.commit()

    # ── Step 2: Fetch history context (last 10 messages) ─────────────
    history_query = (
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.desc())
        .limit(10)
    )
    history_result = await db.execute(history_query)
    history_messages = list(reversed(history_result.scalars().all()))

    # Format for Gemini (uses "model" not "assistant")
    conversation_history = []
    for msg in history_messages[:-1]:  # Exclude the just-added user message
        role = "user" if msg.role == MessageRole.user else "model"
        conversation_history.append({
            "role": role,
            "parts": [{"text": msg.content}],
        })

    # ── Step 3: RAG retrieval ─────────────────────────────────────────
    rag_context = _get_rag_context(user_message)

    # ── Step 4: Construct system prompt ───────────────────────────────
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(rag_context=rag_context)

    # ── Step 5: Stream Gemini response ────────────────────────────────
    full_response = ""

    try:
        genai.configure(api_key=settings.GEMINI_API_KEY)
        model = genai.GenerativeModel(
            "gemini-2.5-flash",
            system_instruction=system_prompt,
        )

        chat = model.start_chat(history=conversation_history)

        response = chat.send_message(user_message, stream=True)

        for chunk in response:
            if chunk.text:
                full_response += chunk.text
                yield f"data: {json.dumps({'token': chunk.text})}\n\n"

        yield f"data: {json.dumps({'done': True})}\n\n"

    except Exception as e:
        logger.error(f"Gemini streaming error: {e}")
        error_msg = str(e)
        if "API key" in error_msg.lower():
            error_msg = "AI service configuration error. Please contact support."
        yield f"data: {json.dumps({'error': error_msg})}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"

    # ── Step 6: Save assistant response to DB ─────────────────────────
    if full_response:
        assistant_msg = Message(
            conversation_id=conversation_id,
            role=MessageRole.assistant,
            content=full_response,
        )
        db.add(assistant_msg)
        await db.commit()


# ── Title Generation ──────────────────────────────────────────────────────────
async def generate_conversation_title(
    conversation_id: uuid.UUID,
    db: AsyncSession,
    user_id: uuid.UUID,
) -> str:
    """
    Generate a 3-5 word title based on the first 2 messages of a conversation.
    """
    result = await db.execute(
        select(Conversation).where(Conversation.id == conversation_id)
    )
    conversation = result.scalar_one_or_none()

    if conversation is None or conversation.user_id != user_id:
        return "New Conversation"

    # Fetch first 2 messages
    msg_query = (
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc())
        .limit(2)
    )
    msg_result = await db.execute(msg_query)
    messages = list(msg_result.scalars().all())

    if not messages:
        return "New Conversation"

    messages_text = "\n".join(
        f"{msg.role.value}: {msg.content[:200]}" for msg in messages
    )

    try:
        genai.configure(api_key=settings.GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(
            f"Generate a concise 3-5 word title for this conversation. "
            f"Return ONLY the title, nothing else.\n\n{messages_text}"
        )
        title = response.text.strip().strip('"').strip("'")
        # Truncate if too long
        if len(title) > 100:
            title = title[:97] + "..."

        conversation.title = title
        await db.commit()
        return title
    except Exception as e:
        logger.error(f"Title generation error: {e}")
        return "New Conversation"
