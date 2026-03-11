from pydantic_settings import BaseSettings
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    APP_NAME: str = "VoxLedger API"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = True

    # Database
    DATABASE_URL: str = f"sqlite:///{BASE_DIR}/database/voxledger.db"
    DATABASE_PATH: str = str(BASE_DIR / "database" / "voxledger.db")

    # Auth
    SECRET_KEY: str = "voxledger-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days

    # Voice
    VOICE_SAMPLE_DIR: str = str(BASE_DIR / "database" / "voice_samples")
    VOICE_SIMILARITY_THRESHOLD: float = 0.92  # command-time threshold
    LOCK_VOICE_SIMILARITY_THRESHOLD: float = 0.975  # stricter unlock threshold to reject cough/noise/other voices
    MAX_VOICE_SAMPLES: int = 1  # one sample is sufficient for voice authentication

    # Whisper
    WHISPER_MODEL: str = "small"  # small handles accents, broken English much better than base

    # TTS
    TTS_LANGUAGE: str = "en"
    TTS_DIR: str = str(BASE_DIR / "database" / "tts_cache")

    # Currency
    CURRENCY_SYMBOL: str = "₹"
    CURRENCY_CODE: str = "INR"

    # Budget alert thresholds
    BUDGET_WARNING_PCT: float = 0.80   # 80% = warning
    BUDGET_CRITICAL_PCT: float = 0.95  # 95% = critical

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
