import os
from dotenv import load_dotenv
from typing import Optional

load_dotenv()


class Config:
    """Configuration for the Claude Code MLX Proxy"""

    # Server settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8888"))

    # Model settings
    MODEL_NAME: str = os.getenv("MODEL_NAME", "mlx-community/Qwen3.5-4B-MLX-4bit")
    TRUST_REMOTE_CODE: bool = os.getenv("TRUST_REMOTE_CODE", "false").lower() == "true"
    EOS_TOKEN: Optional[str] = os.getenv("EOS_TOKEN")

    # Generation settings
    DEFAULT_MAX_TOKENS: int = int(os.getenv("DEFAULT_MAX_TOKENS", "4096"))
    DEFAULT_TEMPERATURE: float = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))
    DEFAULT_TOP_P: float = float(os.getenv("DEFAULT_TOP_P", "0.9"))

    # Why Optional[float]: when set to None we skip passing it to generate(),
    # letting the model use its internal default. Qwen benefits from ~1.1.
    _rep_penalty_raw: Optional[str] = os.getenv("REPETITION_PENALTY")
    REPETITION_PENALTY: Optional[float] = (
        float(_rep_penalty_raw) if _rep_penalty_raw else None
    )

    # API settings
    API_MODEL_NAME: str = os.getenv("API_MODEL_NAME", "claude-4-sonnet-20250514")

    # Logging
    VERBOSE: bool = os.getenv("VERBOSE", "false").lower() == "true"


config = Config()

