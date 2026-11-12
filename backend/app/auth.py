"""Authentication: bcrypt password hashing + stateless JWT bearer tokens."""
from __future__ import annotations

import os
import secrets
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from fastapi import Depends, HTTPException, Query, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from smartyield import config

from . import db

ALGO = "HS256"
TOKEN_HOURS = 12
_secret_file = config.STORAGE_DIR / ".jwt_secret"


def _secret() -> str:
    if os.environ.get("SMARTYIELD_SECRET"):
        return os.environ["SMARTYIELD_SECRET"]
    if not _secret_file.exists():
        _secret_file.write_text(secrets.token_hex(32))
    return _secret_file.read_text().strip()


def hash_password(pw: str) -> str:
    return bcrypt.hashpw(pw.encode()[:72], bcrypt.gensalt()).decode()


def verify_password(pw: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(pw.encode()[:72], hashed.encode())
    except ValueError:
        return False


def create_token(user_id: int) -> str:
    now = datetime.now(timezone.utc)
    return jwt.encode({"sub": str(user_id), "iat": now, "exp": now + timedelta(hours=TOKEN_HOURS)}, _secret(), ALGO)


def _decode(token: str) -> int:
    try:
        return int(jwt.decode(token, _secret(), algorithms=[ALGO])["sub"])
    except jwt.PyJWTError as e:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired token") from e


bearer = HTTPBearer(auto_error=False)


def current_user(cred: HTTPAuthorizationCredentials | None = Depends(bearer),
                 token: str | None = Query(default=None, description="for <img>/<a> links that cannot send headers")) -> dict:
    raw = cred.credentials if cred else token
    if not raw:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Not authenticated")
    uid = _decode(raw)
    with db.session() as con:
        u = db.row(con.execute("SELECT id, email, name, created_at FROM users WHERE id=?", (uid,)).fetchone())
    if not u:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "User no longer exists")
    return u
