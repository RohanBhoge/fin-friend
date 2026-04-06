"""
Async SQLAlchemy engine and session factory for Neon PostgreSQL (serverless).
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings

logger = logging.getLogger("finfriend.db")

# ── Async Engine ──────────────────────────────────────────────────────────────
# Tuned for Neon serverless: pool_pre_ping handles cold-start disconnects
async_engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.ENVIRONMENT == "development",
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=1800,
    pool_pre_ping=True,
)

# ── Session Factory ───────────────────────────────────────────────────────────
async_session_factory = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


# ── Declarative Base ─────────────────────────────────────────────────────────
class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


# ── FastAPI Dependency ────────────────────────────────────────────────────────
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Yield an async database session for FastAPI Depends injection."""
    async with async_session_factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ── Table Creation ────────────────────────────────────────────────────────────
async def create_all_tables() -> None:
    """Create all tables defined in ORM models. Safe to call multiple times."""
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("✅ Database tables created / verified successfully")


# ── Connection Verification ──────────────────────────────────────────────────
async def verify_db_connection() -> bool:
    """Ping the database to verify connectivity. Returns True if alive."""
    try:
        async with async_engine.begin() as conn:
            await conn.execute(
                __import__("sqlalchemy").text("SELECT 1")
            )
        logger.info("✅ Database connection verified")
        return True
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return False


# ── Shutdown ──────────────────────────────────────────────────────────────────
async def dispose_engine() -> None:
    """Dispose of the engine and close all connections."""
    await async_engine.dispose()
    logger.info("🔌 Database engine disposed")
