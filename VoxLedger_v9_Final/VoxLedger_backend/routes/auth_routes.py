from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
import hashlib

from database import get_connection
from config import settings
from models.user_model import UserRegisterRequest, CheckUserResponse, UserResponse
from services.voice_auth_service import save_voice_embedding, verify_voice, get_voice_sample_count, analyze_audio_quality
from services.finance_service import create_notification

router = APIRouter(tags=["Authentication"])


def _hash_password(password: str) -> str:
    """Simple SHA-256 password hash — no bcrypt dependency issues."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def _verify_password(password: str, hashed: str) -> bool:
    return _hash_password(password) == hashed


# ── Check if any user is registered ──────────────────────────────────────────
@router.get("/check-user", response_model=CheckUserResponse)
def check_user():
    """Splash screen check.

    The app is considered fully registered only when at least one user has
    a stored voice profile. Incomplete user rows without embeddings are treated
    as unfinished registration, so the frontend can go back to Registration.
    """
    conn = get_connection()
    try:
        any_user = conn.execute(
            "SELECT id, name FROM users ORDER BY id LIMIT 1"
        ).fetchone()
        registered_user = conn.execute(
            """
            SELECT u.id, u.name
            FROM users u
            WHERE EXISTS (
                SELECT 1 FROM voice_embeddings ve WHERE ve.user_id = u.id
            )
            ORDER BY u.id
            LIMIT 1
            """
        ).fetchone()

        if not any_user:
            return CheckUserResponse(
                registered=False,
                has_user=False,
                has_voice_profile=False,
            )

        if not registered_user:
            return CheckUserResponse(
                registered=False,
                user_id=any_user["id"],
                user_name=any_user["name"],
                has_user=True,
                has_voice_profile=False,
            )

        return CheckUserResponse(
            registered=True,
            user_id=registered_user["id"],
            user_name=registered_user["name"],
            has_user=True,
            has_voice_profile=True,
        )
    finally:
        conn.close()


# ── Register user ─────────────────────────────────────────────────────────────
@router.post("/register")
def register_user(name: str = Form(...), password: str = Form(...)):
    if len(name.strip()) < 2:
        raise HTTPException(400, "Name must be at least 2 characters.")
    if len(password) < 4:
        raise HTTPException(400, "Password must be at least 4 characters.")

    conn = get_connection()
    try:
        # If the DB only contains unfinished users with no voice profile,
        # clear them so registration can start cleanly.
        incomplete_users = conn.execute(
            """
            SELECT u.id
            FROM users u
            WHERE NOT EXISTS (
                SELECT 1 FROM voice_embeddings ve WHERE ve.user_id = u.id
            )
            """
        ).fetchall()
        completed_user = conn.execute(
            """
            SELECT u.id
            FROM users u
            WHERE EXISTS (
                SELECT 1 FROM voice_embeddings ve WHERE ve.user_id = u.id
            )
            LIMIT 1
            """
        ).fetchone()
        if completed_user is None and incomplete_users:
            conn.execute("DELETE FROM users")
            conn.commit()

        existing = conn.execute(
            "SELECT id FROM users WHERE LOWER(name) = LOWER(?)", (name.strip(),)
        ).fetchone()
        if existing:
            raise HTTPException(409, "A user with this name already exists.")

        hashed_pw = _hash_password(password)
        cur = conn.execute(
            "INSERT INTO users (name, password) VALUES (?, ?)",
            (name.strip(), hashed_pw)
        )
        conn.commit()
        user_id = cur.lastrowid

        # Initialise all default budget categories at ₹0 so Budget page is never blank
        from services.finance_service import set_budget as _set_budget
        DEFAULT_CATS = [
            "Food", "Transport", "Shopping", "Utilities",
            "Entertainment", "Healthcare", "Housing", "Education", "Others",
        ]
        _set_budget(user_id, "monthly", 0.0)
        for _cat in DEFAULT_CATS:
            _set_budget(user_id, _cat, 0.0)

        create_notification(
            user_id,
            title="Welcome to VoxLedger! 🎉",
            message=f"Hi {name}! Your account is ready. Say \"Set monthly budget to 10000\" to set your budget.",
            notif_type="success",
        )

        return {
            "success": True,
            "message": f"Account created for {name}.",
            "user_id": user_id,
            "user_name": name.strip(),
        }
    finally:
        conn.close()


# ── Upload voice sample ───────────────────────────────────────────────────────
@router.post("/register/voice-sample")
async def upload_voice_sample(
    user_id: int = Form(...),
    voice_sample: UploadFile = File(...),
):
    audio_bytes = await voice_sample.read()
    if not audio_bytes:
        raise HTTPException(400, "Empty audio file.")

    success, message = save_voice_embedding(user_id, audio_bytes)
    if not success:
        raise HTTPException(400, message)

    sample_count = get_voice_sample_count(user_id)
    from config import settings

    return {
        "success": True,
        "message": message,
        "samples_registered": sample_count,
        "max_samples": settings.MAX_VOICE_SAMPLES,
        "registration_complete": sample_count >= settings.MAX_VOICE_SAMPLES,
    }


# ── Verify voice (lock screen) ────────────────────────────────────────────────
@router.post("/verify-voice")
async def verify_voice_endpoint(voice_sample: UploadFile = File(...)):
    import time
    t0 = time.time()
    audio_bytes = await voice_sample.read()

    # Hard reject: too small to contain real speech (< 8 KB ≈ < 0.5 s at 16kHz)
    MIN_AUDIO_BYTES = 12_000
    if not audio_bytes or len(audio_bytes) < MIN_AUDIO_BYTES:
        print(f"\n[auth] ❌  REJECT — audio too small ({len(audio_bytes) if audio_bytes else 0}B < {MIN_AUDIO_BYTES}B). Silence rejected.")
        return {
            "authenticated": False,
            "user_id": None,
            "user_name": None,
            "similarity_score": 0.0,
            "message": "No voice detected. Please speak clearly to unlock.",
        }

    # Never attempt authentication unless both a user and a stored voice profile exist.
    conn = get_connection()
    try:
        user_row = conn.execute(
            """
            SELECT u.id, u.name
            FROM users u
            WHERE EXISTS (SELECT 1 FROM voice_embeddings ve WHERE ve.user_id = u.id)
            ORDER BY u.id
            LIMIT 1
            """
        ).fetchone()
        if not user_row:
            print("[auth] ❌  REJECT — no user registered yet.")
            return {
                "authenticated": False,
                "user_id": None,
                "user_name": None,
                "similarity_score": 0.0,
                "message": "No user found. Please complete registration first.",
            }

        total_embeddings = conn.execute(
            "SELECT COUNT(*) as cnt FROM voice_embeddings WHERE user_id = ?",
            (user_row["id"],),
        ).fetchone()["cnt"]

        if total_embeddings == 0:
            print(f"[auth] ❌  REJECT — user exists but no voice profile is stored.")
            return {
                "authenticated": False,
                "user_id": user_row["id"],
                "user_name": user_row["name"],
                "similarity_score": 0.0,
                "message": "Voice profile missing. Please register your voice again.",
            }
    finally:
        conn.close()

    # Stricter pre-check for lock-screen authentication than for in-app commands.
    quality_ok, quality_msg, _, _ = analyze_audio_quality(audio_bytes)
    if not quality_ok:
        print(f"[auth] ❌  REJECT — quality gate: {quality_msg}")
        return {
            "authenticated": False,
            "user_id": None,
            "user_name": None,
            "similarity_score": 0.0,
            "message": quality_msg,
        }

    authenticated, user_id, user_name, score = verify_voice(
        audio_bytes,
        expected_user_id=user_row["id"],
        threshold_override=settings.LOCK_VOICE_SIMILARITY_THRESHOLD,
    )
    elapsed = (time.time() - t0) * 1000

    status = "✅  GRANTED" if authenticated else "❌  DENIED"
    print(f"\n[auth] {status} — user={user_name!r}  score={score:.4f}  ({elapsed:.0f} ms)")

    return {
        "authenticated": authenticated,
        "user_id": user_id,
        "user_name": user_name,
        "similarity_score": round(score, 4),
        "message": f"Welcome back, {user_name}!" if authenticated else "Voice not recognised. Please try again.",
    }


# ── Simple password login (fallback) ─────────────────────────────────────────
@router.post("/login")
def login(user_id: int = Form(...), password: str = Form(...)):
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM users WHERE id = ?", (user_id,)
        ).fetchone()
        if not row:
            raise HTTPException(404, "User not found.")
        if not _verify_password(password, row["password"]):
            raise HTTPException(401, "Incorrect password.")
        return {
            "success": True,
            "user_id": row["id"],
            "user_name": row["name"],
        }
    finally:
        conn.close()


# ── Get user profile ──────────────────────────────────────────────────────────
@router.get("/user/{user_id}")
def get_user(user_id: int):
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT id, name, created_at FROM users WHERE id = ?", (user_id,)
        ).fetchone()
        if not row:
            raise HTTPException(404, "User not found.")
        voice_count = get_voice_sample_count(user_id)
        return {
            "id": row["id"],
            "name": row["name"],
            "created_at": row["created_at"],
            "voice_samples": voice_count,
        }
    finally:
        conn.close()
