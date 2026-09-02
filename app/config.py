"""
Application Settings & Configuration Management using Pydantic Settings.
Reads parameters from environment variables or .env file.
"""

import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    DEBUG: bool = True

    # Local AI Configuration
    LOCAL_MODEL_NAME: str = "microsoft/Florence-2-base"
    CACHE_DIR: str = "./model_cache"
    CHROMA_DB_PATH: str = "./storage/chroma_db"

    # Playwright Settings
    HEADLESS: bool = False
    PLAYWRIGHT_TIMEOUT: int = 30000
    IGNORE_HTTPS_ERRORS: bool = True
    VIEWPORT_WIDTH: int = 1280
    VIEWPORT_HEIGHT: int = 800

    # Storage Settings
    STORAGE_DIR: str = "./storage"
    DB_PATH: str = "./storage/test_runs.db"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()

# Ensure storage directories exist
os.makedirs(settings.STORAGE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(settings.DB_PATH), exist_ok=True)
os.makedirs(settings.CACHE_DIR, exist_ok=True)
os.makedirs(settings.CHROMA_DB_PATH, exist_ok=True)
