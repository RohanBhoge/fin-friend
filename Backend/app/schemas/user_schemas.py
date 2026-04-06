"""
Pydantic V2 schemas for user profile management.
"""

from pydantic import BaseModel, ConfigDict, EmailStr, field_validator

from app.schemas.auth_schemas import validate_password_strength


class UserUpdate(BaseModel):
    """Schema for updating user profile fields."""
    full_name: str | None = None
    email: EmailStr | None = None
    avatar_url: str | None = None


class PasswordUpdate(BaseModel):
    """Schema for changing password (requires current password)."""
    current_password: str
    new_password: str

    @field_validator("new_password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        return validate_password_strength(v)


class DeleteAccountRequest(BaseModel):
    """Schema for account deletion (requires password confirmation)."""
    password: str
