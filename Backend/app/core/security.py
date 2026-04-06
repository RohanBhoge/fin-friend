"""
Security utilities: JWT token creation/validation and password hashing.
"""

import secrets
from datetime import datetime, timedelta, timezone

# ── Bcrypt/Passlib Compatibility Patch (MUST BE FIRST) ────────────────────────
import bcrypt
import passlib.handlers.bcrypt

# 1. Global Truncation Patch: Resolves ValueError in bcrypt 4.0+ for long passwords
_orig_hashpw = bcrypt.hashpw
def _patched_hashpw(password, salt):
    if isinstance(password, str):
        password = password.encode("utf-8")
    # Truncate to 72 bytes as required by bcrypt
    return _orig_hashpw(password[:72], salt)
bcrypt.hashpw = _patched_hashpw

# 2. Metadata Patch: Fixes passlib's version detection logic
if not hasattr(bcrypt, "__about__"):
    class _BCryptAbout:
        __version__ = getattr(bcrypt, "__version__", "5.0.0")
    bcrypt.__about__ = _BCryptAbout()

# 3. Backend Injection: Force passlib to recognize the installed bcrypt
if passlib.handlers.bcrypt._bcrypt is None:
    passlib.handlers.bcrypt._bcrypt = bcrypt

# 4. Ident Patch: Standard fix for the "72-byte" ident issue requested by user
if passlib.handlers.bcrypt._bcrypt is not None:
    try:
        passlib.handlers.bcrypt._bcrypt.IDENT_2A = passlib.handlers.bcrypt._bcrypt.IDENT_2B
    except AttributeError:
        pass

# ── Imports ──────────────────────────────────────────────────────────────────
from fastapi import HTTPException, status
from jose import JWTError, jwt
from passlib.context import CryptContext

from app.core.config import settings

# ── Password Hashing ─────────────────────────────────────────────────────────

pwd_context = CryptContext(
    schemes=["bcrypt"], 
    deprecated="auto",
    bcrypt__truncate_error=False
)

def hash_password(plain: str) -> str:
    """Hash a plaintext password. Truncation is handled by our global patch."""
    return pwd_context.hash(plain)

def verify_password(plain: str, hashed: str) -> bool:
    """Verify a plaintext password against a hash."""
    return pwd_context.verify(plain, hashed)

# ── JWT Tokens ────────────────────────────────────────────────────────────────
def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    """Create a short-lived JWT access token."""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta
        or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire, "type": "access"})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def create_refresh_token(data: dict) -> str:
    """Create a long-lived JWT refresh token."""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(
        days=settings.REFRESH_TOKEN_EXPIRE_DAYS
    )
    to_encode.update({"exp": expire, "type": "refresh"})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def decode_token(token: str) -> dict:
    """
    Decode and validate a JWT token.
    Raises HTTPException(401) if the token is invalid or expired.
    """
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
        )
        if payload.get("sub") is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: missing subject",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return payload
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ── Password Reset Tokens ────────────────────────────────────────────────────
def generate_reset_token() -> str:
    """Generate a cryptographically secure URL-safe reset token."""
    return secrets.token_urlsafe(32)
