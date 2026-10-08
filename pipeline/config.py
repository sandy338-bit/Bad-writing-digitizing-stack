import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    vision_reader: str = os.getenv("VISION_READER", "qwen")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    gemini_fallback_models: str = os.getenv("GEMINI_FALLBACK_MODELS", "")
    qwen_api_key: str = os.getenv("DASHSCOPE_API_KEY", "")
    qwen_model: str = os.getenv("QWEN_MODEL", "qwen3-vl-plus")
    qwen_base_url: str = os.getenv("QWEN_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
    qwen_backend: str = os.getenv("QWEN_BACKEND", "ollama")
    qwen_local_model: str = os.getenv("QWEN_LOCAL_MODEL", "qwen2.5vl:3b")
    trocr_model: str = os.getenv("TROCR_MODEL", "data/models/trocr-onnx")
    ensemble_crop_limit: int = int(os.getenv("ENSEMBLE_CROP_LIMIT", "16"))
    language_hint: str = os.getenv("LANGUAGE_HINT", "auto")
    enable_paddle: bool = os.getenv("ENABLE_PADDLE_OCR", "true").lower() == "true"
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "15"))
    accept_threshold: float = float(os.getenv("ACCEPT_THRESHOLD", "0.75"))
    database_path: str = os.getenv("DATABASE_PATH", "data/scrawlguard.sqlite3")
    max_pending_jobs: int = int(os.getenv("MAX_PENDING_JOBS", "8"))
    request_timeout_seconds: int = int(os.getenv("MODEL_TIMEOUT_SECONDS", "60"))
    max_regions: int = int(os.getenv("MAX_REGIONS", "80"))
    cors_origins: str = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173")

    def __post_init__(self) -> None:
        if self.vision_reader not in {"gemini", "qwen", "trocr", "ensemble"}:
            raise ValueError("VISION_READER must be gemini, qwen, trocr, or ensemble")
        if self.qwen_backend not in {"ollama", "dashscope"}:
            raise ValueError("QWEN_BACKEND must be ollama or dashscope")
        if self.language_hint.casefold() not in {"auto", "english", "hindi", "tamil", "malayalam"}:
            raise ValueError("LANGUAGE_HINT must be auto, English, Hindi, Tamil, or Malayalam")
        if not 0 <= self.accept_threshold <= 1:
            raise ValueError("ACCEPT_THRESHOLD must be between 0 and 1")
        if self.max_upload_mb < 1:
            raise ValueError("MAX_UPLOAD_MB must be positive")
        if min(self.max_pending_jobs, self.request_timeout_seconds, self.max_regions,
               self.ensemble_crop_limit) < 1:
            raise ValueError("Job, timeout, and region limits must be positive")
