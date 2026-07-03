"""
AI Knowledge Distillation Platform - Global Dependencies.

FastAPI dependency injection for database sessions, settings, etc.
"""

from typing import AsyncGenerator

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.database import AsyncSessionLocal


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Yield an async database session for request lifetime."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


def get_app_settings() -> Settings:
    """Get application settings (cacheable dependency)."""
    return get_settings()
