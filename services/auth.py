from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from fastapi import Depends, HTTPException, Request, Response, status

AUTH_COOKIE_NAME = os.getenv("AUTH_COOKIE_NAME", "ai_lab_session")
AUTH_SECRET = os.getenv("AUTH_SECRET", "dev-auth-secret-change-me")
SESSION_TTL_HOURS = max(1, int(os.getenv("AUTH_SESSION_HOURS", "12")))
USER_STORE_PATH = Path(__file__).resolve().parents[2] / "data" / "auth_users.json"

ROLE_ORDER = {"user": 1, "admin": 2, "super_admin": 3}


@dataclass(frozen=True)
class AuthUser:
    username: str
    role: str
    display_name: str
    issued_at: str
    expires_at: str

    def to_dict(self) -> dict:
        return asdict(self)


def _normalize_username(username: str | None) -> str:
    return (username or "").strip().lower()


def _normalize_display_name(username: str, display_name: str | None = None) -> str:
    value = (display_name or "").strip()
    if value:
        return value[:64]
    username = username.strip()
    return f"User · {username}" if username else "User"


def _hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    salt_bytes = bytes.fromhex(salt) if salt else os.urandom(16)
    hashed = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt_bytes, 120000)
    return salt_bytes.hex(), hashed.hex()


def _verify_password(password: str, salt: str, password_hash: str) -> bool:
    _, computed = _hash_password(password, salt)
    return hmac.compare_digest(computed, password_hash)


def _load_registered_users() -> list[dict[str, str]]:
    if not USER_STORE_PATH.exists():
        return []
    try:
        payload = json.loads(USER_STORE_PATH.read_text(encoding="utf-8"))
        users = payload.get("users") if isinstance(payload, dict) else payload
        if isinstance(users, list):
            return [user for user in users if isinstance(user, dict)]
    except Exception:
        return []
    return []


def _save_registered_users(users: list[dict[str, str]]) -> None:
    USER_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    USER_STORE_PATH.write_text(
        json.dumps({"users": users}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _find_registered_user(username: str) -> dict[str, str] | None:
    username = _normalize_username(username)
    for user in _load_registered_users():
        if _normalize_username(user.get("username")) == username:
            return user
    return None

def _credential_for(role: str) -> dict[str, str]:
    defaults = {
        "user": ("USER_USERNAME", "USER_PASSWORD", "user", "user123"),
        "admin": ("ADMIN_USERNAME", "ADMIN_PASSWORD", "admin", "admin123"),
        "super_admin": ("SUPER_ADMIN_USERNAME", "SUPER_ADMIN_PASSWORD", "superadmin", "superadmin123"),
    }
    username_key, password_key, default_username, default_password = defaults[role]
    return {
        "role": role,
        "username": _normalize_username(os.getenv(username_key, default_username)),
        "password": os.getenv(password_key, default_password),
    }


def _credentials() -> list[dict[str, str]]:
    return [_credential_for("user"), _credential_for("admin"), _credential_for("super_admin")]


def _display_name(username: str, role: str) -> str:
    prefix = role.replace("_", " ").title()
    return f"{prefix} · {username}"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _encode_payload(payload: dict) -> str:
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("utf-8").rstrip("=")


def _decode_payload(value: str) -> dict | None:
    try:
        padded = value + "=" * (-len(value) % 4)
        raw = base64.urlsafe_b64decode(padded.encode("utf-8"))
        parsed = json.loads(raw.decode("utf-8"))
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _sign(value: str) -> str:
    return hmac.new(AUTH_SECRET.encode("utf-8"), value.encode("utf-8"), hashlib.sha256).hexdigest()


def _build_token(payload: dict) -> str:
    encoded = _encode_payload(payload)
    signature = _sign(encoded)
    return f"{encoded}.{signature}"


def _verify_token(token: str) -> dict | None:
    if not token or "." not in token:
        return None
    encoded, signature = token.rsplit(".", 1)
    expected = _sign(encoded)
    if not hmac.compare_digest(signature, expected):
        return None

    payload = _decode_payload(encoded)
    if not payload:
        return None

    expires_at = payload.get("exp")
    if not expires_at:
        return None
    try:
        parsed_exp = datetime.fromisoformat(str(expires_at))
    except Exception:
        return None
    if parsed_exp.tzinfo is None:
        parsed_exp = parsed_exp.replace(tzinfo=timezone.utc)
    if parsed_exp <= _now():
        return None
    return payload


def _build_user(username: str, role: str, display_name: str | None = None) -> AuthUser:
    issued_at = _now()
    expires_at = issued_at + timedelta(hours=SESSION_TTL_HOURS)
    return AuthUser(
        username=username,
        role=role,
        display_name=_normalize_display_name(username, display_name) if role == "user" else _display_name(username, role),
        issued_at=issued_at.isoformat(),
        expires_at=expires_at.isoformat(),
    )


def authenticate_user(username: str, password: str) -> AuthUser | None:
    username = _normalize_username(username)
    password = password or ""
    for credential in _credentials():
        if credential["username"] != username:
            continue
        if hmac.compare_digest(credential["password"], password):
            return _build_user(username, credential["role"])

    user = _find_registered_user(username)
    if user and _verify_password(password, str(user.get("salt") or ""), str(user.get("password_hash") or "")):
        role = str(user.get("role") or "user").strip() or "user"
        return AuthUser(
            username=username,
            role=role,
            display_name=str(user.get("display_name") or _normalize_display_name(username)),
            issued_at=_now().isoformat(),
            expires_at=(_now() + timedelta(hours=SESSION_TTL_HOURS)).isoformat(),
        )
    return None


def register_user(username: str, password: str, display_name: str | None = None) -> AuthUser:
    username = _normalize_username(username)
    password = password or ""
    display_name = _normalize_display_name(username, display_name)

    if not username:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username is required.")
    if len(username) < 3:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username must be at least 3 characters.")
    if len(password) < 6:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Password must be at least 6 characters.")
    if _find_registered_user(username):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Username already exists.")
    for credential in _credentials():
        if credential["username"] == username:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="That username is reserved.")

    salt, password_hash = _hash_password(password)
    users = _load_registered_users()
    users.append(
        {
            "username": username,
            "display_name": display_name,
            "role": "user",
            "salt": salt,
            "password_hash": password_hash,
            "created_at": _now().isoformat(),
        }
    )
    _save_registered_users(users)
    return _build_user(username, "user", display_name=display_name)


def issue_session_token(user: AuthUser) -> str:
    payload = {
        "username": user.username,
        "role": user.role,
        "display_name": user.display_name,
        "iat": user.issued_at,
        "exp": user.expires_at,
    }
    return _build_token(payload)


def user_from_token(token: str) -> AuthUser | None:
    payload = _verify_token(token)
    if not payload:
        return None

    username = str(payload.get("username") or "").strip()
    role = str(payload.get("role") or "").strip()
    display_name = str(payload.get("display_name") or _display_name(username, role)).strip()
    issued_at = str(payload.get("iat") or "").strip()
    expires_at = str(payload.get("exp") or "").strip()
    if not username or role not in ROLE_ORDER:
        return None
    return AuthUser(
        username=username,
        role=role,
        display_name=display_name,
        issued_at=issued_at,
        expires_at=expires_at,
    )


def set_session_cookie(response: Response, user: AuthUser) -> None:
    response.set_cookie(
        key=AUTH_COOKIE_NAME,
        value=issue_session_token(user),
        httponly=True,
        samesite="lax",
        secure=False,
        max_age=SESSION_TTL_HOURS * 3600,
        path="/",
    )


def clear_session_cookie(response: Response) -> None:
    response.delete_cookie(key=AUTH_COOKIE_NAME, path="/")


def _extract_bearer_token(request: Request) -> str | None:
    header = request.headers.get("authorization") or request.headers.get("Authorization")
    if not header:
        return None
    header = header.strip()
    if not header.lower().startswith("bearer "):
        return None
    token = header.split(" ", 1)[1].strip()
    return token or None


def get_current_user(request: Request) -> AuthUser:
    token = request.cookies.get(AUTH_COOKIE_NAME) or _extract_bearer_token(request)
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Login required.")

    user = user_from_token(token)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Session expired or invalid.")
    return user


def require_roles(*roles: str) -> Callable:
    allowed = {role.strip() for role in roles if role and role.strip()}

    def dependency(user: AuthUser = Depends(get_current_user)) -> AuthUser:
        if allowed and user.role not in allowed:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient role.")
        return user

    return dependency


def require_min_role(minimum_role: str) -> Callable:
    minimum_rank = ROLE_ORDER.get(minimum_role, 0)

    def dependency(user: AuthUser = Depends(get_current_user)) -> AuthUser:
        if ROLE_ORDER.get(user.role, 0) < minimum_rank:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient role.")
        return user

    return dependency
