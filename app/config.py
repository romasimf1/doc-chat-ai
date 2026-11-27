"""Application configuration.

All settings come from environment variables (or a .env file).
No secrets are ever hard-coded here.
"""

import os

from dotenv import load_dotenv

load_dotenv()  # Loads .env into os.environ when the file exists.


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


# OpenAI-compatible API settings.
OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL: str = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
CHAT_MODEL: str = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL: str = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")

# RAG pipeline settings.
CHUNK_SIZE: int = _get_int("CHUNK_SIZE", 800)
CHUNK_OVERLAP: int = _get_int("CHUNK_OVERLAP", 150)
DEFAULT_TOP_K: int = _get_int("DEFAULT_TOP_K", 4)

# Where uploaded files and the vector store live (relative to project root).
DATA_DIR: str = os.environ.get("DATA_DIR", "data")
UPLOAD_DIR: str = os.path.join(DATA_DIR, "uploads")
STORE_PATH: str = os.path.join(DATA_DIR, "store.json")
