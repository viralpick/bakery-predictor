"""Environment-driven config. Secrets stay out of source.

Reads `.env` at the repo root via python-dotenv. The repo's .gitignore must
keep .env out of version control — this module assumes that's already true.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_DATA_DIR = PROJECT_ROOT / "data" / "external"


def _load() -> None:
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)


_load()


def require_env(name: str) -> str:
    """Return env var or raise with a clear remediation hint."""
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(
            f"{name} is not set. Add it to {PROJECT_ROOT}/.env "
            f"(see .env.example) or export it before running."
        )
    return val


def data_go_kr_api_key() -> str:
    """공공데이터포털 일반 인증키 (Decoding 버전을 그대로 저장)."""
    return require_env("DATA_GO_KR_API_KEY")
