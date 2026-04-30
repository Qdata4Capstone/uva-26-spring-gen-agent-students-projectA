from fastapi import APIRouter
from app.services.user_service import create_user, get_user, update_user

router = APIRouter(prefix="/api/user", tags=["user"])

@router.post("/create")
def create():
    user = create_user()
    return user

@router.get("/{user_id}")
def get(user_id: str):
    return get_user(user_id)

@router.put("/{user_id}")
def update(user_id: str, updates: dict):
    return update_user(user_id, updates)