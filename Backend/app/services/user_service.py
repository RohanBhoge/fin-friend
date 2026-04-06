"""
User profile service: CRUD for user data, password changes, account deletion.
"""

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import hash_password, verify_password
from app.models.models import User
from app.schemas.user_schemas import PasswordUpdate, UserUpdate


async def get_user_profile(user: User) -> User:
    """Return the user (already loaded by dependency)."""
    return user


async def update_user_profile(
    user: User, update_data: UserUpdate, db: AsyncSession
) -> User:
    """
    Update user profile fields. Checks email uniqueness if email is being changed.
    """
    if update_data.email is not None and update_data.email.lower() != user.email:
        result = await db.execute(
            select(User).where(User.email == update_data.email.lower())
        )
        existing = result.scalar_one_or_none()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="An account with this email already exists",
            )
        user.email = update_data.email.lower()

    if update_data.full_name is not None:
        user.full_name = update_data.full_name

    if update_data.avatar_url is not None:
        user.avatar_url = update_data.avatar_url

    await db.commit()
    await db.refresh(user)
    return user


async def update_user_password(
    user: User, password_data: PasswordUpdate, db: AsyncSession
) -> None:
    """
    Change the user's password after verifying the current password.
    """
    if not verify_password(password_data.current_password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect",
        )

    user.hashed_password = hash_password(password_data.new_password)
    await db.commit()


async def delete_user_account(
    user: User, password: str, db: AsyncSession
) -> None:
    """
    Delete a user account after verifying the password.
    Cascades to delete all conversations and messages.
    """
    if not verify_password(password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incorrect password",
        )

    await db.delete(user)
    await db.commit()
