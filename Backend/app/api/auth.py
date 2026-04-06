"""
Authentication API routes: register, login, refresh, me, forgot/reset password.
"""

from fastapi import APIRouter, BackgroundTasks, Depends, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import get_current_user
from app.db.database import get_db
from app.models.models import User
from app.schemas.auth_schemas import (
    ForgotPasswordRequest,
    LoginRequest,
    RefreshTokenRequest,
    ResetPasswordRequest,
    TokenResponse,
    UserCreate,
    UserRead,
)
from app.services.auth_service import (
    authenticate_user,
    create_password_reset,
    create_tokens_for_user,
    refresh_access_token,
    register_user,
    reset_password,
)
from app.services.email_service import send_reset_email

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post(
    "/register",
    response_model=UserRead,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user account",
)
async def register(
    user_data: UserCreate,
    db: AsyncSession = Depends(get_db),
):
    """Create a new user account with email and password."""
    user = await register_user(user_data, db)
    return user


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Login with email and password",
)
async def login(
    body: LoginRequest,
    db: AsyncSession = Depends(get_db),
):
    """Authenticate and receive access + refresh tokens."""
    user = await authenticate_user(body.email, body.password, db)
    tokens = create_tokens_for_user(user)
    return TokenResponse(
        access_token=tokens["access_token"],
        refresh_token=tokens["refresh_token"],
        token_type=tokens["token_type"],
        user=UserRead.model_validate(user),
    )


@router.post(
    "/refresh",
    summary="Refresh access token",
)
async def refresh(
    body: RefreshTokenRequest,
    db: AsyncSession = Depends(get_db),
):
    """Exchange a valid refresh token for a new access token."""
    access_token = await refresh_access_token(body.refresh_token, db)
    return {"access_token": access_token, "token_type": "bearer"}


@router.get(
    "/me",
    response_model=UserRead,
    summary="Get current authenticated user",
)
async def get_me(
    current_user: User = Depends(get_current_user),
):
    """Return the profile of the authenticated user."""
    return current_user


@router.post(
    "/forgot-password",
    summary="Request a password reset email",
)
async def forgot_password(
    body: ForgotPasswordRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """
    Send a password reset email if the account exists.
    Always returns 200 to prevent email enumeration.
    """
    token = await create_password_reset(body.email, db)
    if token:
        background_tasks.add_task(send_reset_email, body.email, token)

    return {"message": "If that email exists, a reset link has been sent."}


@router.post(
    "/reset-password",
    summary="Reset password with token",
)
async def reset_password_endpoint(
    body: ResetPasswordRequest,
    db: AsyncSession = Depends(get_db),
):
    """Validate the reset token and set a new password."""
    await reset_password(body.token, body.new_password, db)
    return {"message": "Password updated successfully."}
