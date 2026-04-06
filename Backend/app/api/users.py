"""
User profile API routes: get, update, change password, delete account.
"""

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import get_current_user
from app.db.database import get_db
from app.models.models import User
from app.schemas.auth_schemas import UserRead
from app.schemas.user_schemas import DeleteAccountRequest, PasswordUpdate, UserUpdate
from app.services.user_service import (
    delete_user_account,
    get_user_profile,
    update_user_password,
    update_user_profile,
)

router = APIRouter(prefix="/users", tags=["Users"])


@router.get(
    "/profile",
    response_model=UserRead,
    summary="Get current user profile",
)
async def get_profile(
    current_user: User = Depends(get_current_user),
):
    """Return the full profile of the authenticated user."""
    user = await get_user_profile(current_user)
    return user


@router.patch(
    "/profile",
    response_model=UserRead,
    summary="Update user profile",
)
async def update_profile(
    update_data: UserUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Update profile fields (full_name, email, avatar_url)."""
    user = await update_user_profile(current_user, update_data, db)
    return user


@router.patch(
    "/security",
    summary="Change user password",
)
async def update_security(
    password_data: PasswordUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Change the user's password after verifying the current one."""
    await update_user_password(current_user, password_data, db)
    return {"message": "Password updated successfully."}


@router.delete(
    "/account",
    summary="Delete user account",
)
async def delete_account(
    body: DeleteAccountRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Permanently delete the user account and all associated data.
    Requires password confirmation.
    """
    await delete_user_account(current_user, body.password, db)
    return {"message": "Account deleted."}
