import json
import os
import tempfile
from pathlib import Path

from app.models.user import UserProfile

# Always resolve to backend/users.json (cwd-independent — merge broke this when uvicorn cwd differed).
_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
USER_DB_PATH = str(_BACKEND_DIR / "users.json")


def load_users():
    if not os.path.exists(USER_DB_PATH):
        return {}
    with open(USER_DB_PATH, "r") as f:
        return json.load(f)


def save_users(users):
    user_db_dir = os.path.dirname(USER_DB_PATH) or "."
    with tempfile.NamedTemporaryFile("w", delete=False, dir=user_db_dir, encoding="utf-8") as tmp:
        # Use default=str so datetimes are persisted as ISO-like strings instead of crashing writes.
        json.dump(users, tmp, indent=2, default=str)
        tmp_path = tmp.name
    os.replace(tmp_path, USER_DB_PATH)


def create_user():
    users = load_users()
    user = UserProfile()
    users[user.id] = user.model_dump(mode="json")
    save_users(users)
    return user


def get_user(user_id: str):
    users = load_users()
    data = users.get(user_id)
    return UserProfile(**data) if data else None


def update_user(user_id: str, updates: dict):
    users = load_users()

    if user_id not in users:
        return None

    existing = UserProfile(**users[user_id])

    # Merge updates safely
    updated = existing.model_copy(update=updates)

    users[user_id] = updated.model_dump(mode="json")
    save_users(users)

    return updated