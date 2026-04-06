"""
Async email service for password reset emails using aiosmtplib.
"""

import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import aiosmtplib

from app.core.config import settings

logger = logging.getLogger("finfriend.email")


async def send_reset_email(to_email: str, reset_token: str) -> bool:
    """
    Send a password reset email with a link containing the reset token.
    Returns True on success, False on failure (logs error but doesn't raise).
    """
    reset_url = f"{settings.CORS_ORIGINS.split(',')[0]}/reset-password?token={reset_token}"

    msg = MIMEMultipart("alternative")
    msg["Subject"] = "FinFriend — Reset Your Password"
    msg["From"] = settings.SMTP_FROM_EMAIL
    msg["To"] = to_email

    text_body = f"""Hi,

You requested a password reset for your FinFriend account.

Click the link below to set a new password (valid for {settings.RESET_TOKEN_EXPIRE_MINUTES} minutes):

{reset_url}

If you didn't request this, please ignore this email.

— The FinFriend Team
"""

    html_body = f"""
    <!DOCTYPE html>
    <html>
    <body style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #0F0F10; color: #F0F0F0; padding: 40px;">
      <div style="max-width: 500px; margin: 0 auto; background: #1A1A1E; border-radius: 16px; padding: 40px; border: 1px solid #2A2A30;">
        <h1 style="color: #6366F1; margin-top: 0;">FinFriend</h1>
        <p>Hi,</p>
        <p>You requested a password reset for your FinFriend account.</p>
        <p>Click the button below to set a new password. This link is valid for <strong>{settings.RESET_TOKEN_EXPIRE_MINUTES} minutes</strong>.</p>
        <div style="text-align: center; margin: 32px 0;">
          <a href="{reset_url}"
             style="background-color: #6366F1; color: white; padding: 14px 32px; border-radius: 8px; text-decoration: none; font-weight: 600; display: inline-block;">
            Reset Password
          </a>
        </div>
        <p style="color: #6B7280; font-size: 13px;">If you didn't request this, please ignore this email. Your password will remain unchanged.</p>
        <hr style="border: none; border-top: 1px solid #2A2A30; margin: 24px 0;" />
        <p style="color: #6B7280; font-size: 12px;">— The FinFriend Team</p>
      </div>
    </body>
    </html>
    """

    msg.attach(MIMEText(text_body, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    try:
        await aiosmtplib.send(
            msg,
            hostname=settings.SMTP_HOST,
            port=settings.SMTP_PORT,
            username=settings.SMTP_USER,
            password=settings.SMTP_PASSWORD,
            start_tls=True,
        )
        logger.info(f"Password reset email sent to {to_email}")
        return True
    except Exception as e:
        logger.error(f"Failed to send reset email to {to_email}: {e}")
        return False
